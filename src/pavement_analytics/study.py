"""Private end-to-end comparison of observed deduct levels and their predictors."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .autorun import external_path, prepare
from .io import load_sources, read_json, sha256, write_json
from .pipeline import checksum_outputs, code_digest, environment_versions
from .security import Boundary, SafeError, disjoint, refuse_public_execution, reject_links
from .model_runtime import model_options, fit_budget, check_budget
from .model_tasks import task_key, verified_task


KEYS = {"columns", "features", "deduct_families", "date_format", "cutoff_year",
        "folds", "seed", "holdout_fraction", "blank_deducts", "climate", "models", "local_model_options"}
SETUP_KEYS = KEYS - {"climate", "features", "local_model_options"}


def resolve_options(data: Path, public: Path, explicit: Path | None = None) -> tuple[dict, Path | None]:
    """Discover one private profile; all relative paths refer to its directory."""
    data = external_path(data, public)
    folder = data if data.is_dir() else data.parent
    profile = explicit
    if profile is None:
        candidates = [folder / "pavement-study.json", folder.parent / "pavement-study.json",
                      Path.home() / ".config/pavement-analytics/study.json"]
        profile = next((p for p in candidates if p.is_file()), None)
    options = read_json(external_path(profile, public)) if profile else {}
    if set(options) - KEYS:
        raise SafeError("UNKNOWN_STUDY_OPTION")
    climate = options.setdefault("climate", {})
    if not isinstance(climate, dict):
        raise SafeError("INVALID_CLIMATE_CONFIGURATION")
    base = external_path(profile, public).parent if profile else folder
    options["local_model_options"] = model_options(options.get("local_model_options"))
    if options["local_model_options"]["cache_dir"]:
        value = Path(options["local_model_options"]["cache_dir"]).expanduser()
        options["local_model_options"]["cache_dir"] = str(external_path(value if value.is_absolute() else base/value, public))
    for key in ("mapping", "csv_dir", "cache_dir"):
        if key in climate and climate[key] is None:
            del climate[key]
        if climate.get(key) is not None:
            if not isinstance(climate[key], str) or not climate[key].strip():
                raise SafeError("INVALID_CLIMATE_PATH")
            path = Path(climate[key]).expanduser()
            climate[key] = str(external_path(path if path.is_absolute() else base / path, public))
    if climate.get("mode", "era5") == "era5" and not climate.get("mapping"):
        mapping = next((p for p in [folder / "data_file_mapping.json",
                                    folder.parent / "data_file_mapping.json"] if p.is_file()), None)
        if mapping:
            climate["mapping"] = str(external_path(mapping, public))
    if climate.get("mapping") and not climate.get("csv_dir"):
        climate["csv_dir"] = str(Path(climate["mapping"]).parent)
    for key, default in (("folds", 5), ("seed", 42), ("cutoff_year", 1980)):
        value = options.get(key, default)
        if type(value) is not int or value < (2 if key == "folds" else 0):
            raise SafeError("INVALID_STUDY_CONFIGURATION")
    fraction = options.get("holdout_fraction", .2)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < .5:
        raise SafeError("INVALID_HOLDOUT_FRACTION")
    if not isinstance(options.get("models", True), bool):
        raise SafeError("BOOLEAN_CONFIGURATION_REQUIRED")
    return options, external_path(profile, public) if profile else None


def snapshot_locations(boundary: Boundary, climate: dict) -> tuple[dict, set[Path], dict[str, str]]:
    """Freeze private location inputs and exclude them from inspection discovery."""
    climate = dict(climate)
    excluded, hashes = set(), {}
    if not climate.get("mapping"):
        return climate, excluded, hashes
    source = Path(climate["mapping"])
    hashes[str(source)] = sha256(source)
    mapping = read_json(source)
    folder = Path(climate.get("csv_dir", source.parent))
    destination = boundary.output("location_inputs")
    destination.mkdir()
    copied = {}
    for index, (name, location) in enumerate(mapping.items(), 1):
        if not isinstance(location, str) or not location or Path(name).name != name:
            raise SafeError("INVALID_LOCATION_MAPPING")
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise SafeError("INVALID_LOCATION_MAPPING")
        path = external_path(folder / relative, boundary.public)
        excluded.add(path)
        hashes[str(path)] = sha256(path)
        output = destination / f"location_{index:04d}.csv"
        shutil.copyfile(path, output)
        if sha256(output) != hashes[str(path)]:
            raise SafeError("INPUT_CHANGED_DURING_READ")
        copied[name] = output.name
    shutil.copyfile(source, destination / "original_mapping.json")
    write_json(destination / "mapping.json", copied)
    climate.update(mapping=str(destination / "mapping.json"), csv_dir=str(destination))
    return climate, excluded, hashes


def isolate_climate_sites(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join declared groups sharing an ERA5 location before assigning any split."""
    table = table.copy()
    parent = {str(g): str(g) for g in table.group_id.unique()}
    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    located = False
    if "climate_location_id" in table:
        known = table.loc[table.climate_location_id.notna() & table.climate_location_id.ne("")]
        located = not known.empty
        for _, block in known.groupby("climate_location_id"):
            groups = sorted(set(block.group_id.astype(str)))
            for group in groups[1:]:
                parent[find(group)] = find(groups[0])
    table["declared_group_id"] = table.group_id
    table["group_id"] = table.group_id.astype(str).map(lambda g: find(g))
    return table, {"declared_groups": len(parent), "validation_groups": int(table.group_id.nunique()),
                   "shared_climate_sites_joined": located,
                   "note": "Groups sharing climate locations are joined before all target splits when locations are available."}


def study(boundary: Boundary, data: Path, options_path: Path | None = None, *,
          by_code: bool = False, date_format: str | None = None,
          blank_deducts: str | None = None) -> str:
    from .climate import enrich
    from .observations import build_observations, select_covariates
    from .study_modeling import make_partitions, search_levels
    from .study_reporting import build_study_report

    refuse_public_execution()
    started = datetime.now(timezone.utc).isoformat()
    options, profile = resolve_options(data, boundary.public, options_path)
    profile_hash = sha256(profile) if profile else None
    climate, excluded, location_hashes = snapshot_locations(boundary, options["climate"])
    climate.setdefault("cache_dir", str(boundary.workspace.parent / "era5-cache"))
    cache = external_path(Path(climate["cache_dir"]), boundary.public)
    if any((p / ".git").exists() for p in (cache, *cache.parents)):
        raise SafeError("CLIMATE_CACHE_MUST_NOT_BE_VERSIONED")
    input_path = external_path(data, boundary.public)
    if not disjoint(cache, input_path):
        raise SafeError("CLIMATE_CACHE_MUST_BE_SEPARATE_FROM_INPUTS")
    if not disjoint(cache, boundary.workspace):
        raise SafeError("CLIMATE_CACHE_MUST_BE_SEPARATE_FROM_RUN")
    climate["cache_dir"] = str(cache)
    options["climate"] = climate
    if profile:
        shutil.copyfile(profile, boundary.output("study_options_original.json"))
    setup_options = {k: v for k, v in options.items() if k in SETUP_KEYS}
    setup_path = boundary.output("study_setup_options.json")
    write_json(setup_path, setup_options)
    config_path = prepare(boundary, data, setup_path, by_code=by_code, date_format=date_format,
                          blank_deducts=blank_deducts, exclude_files=excluded)
    config = read_json(config_path)
    config_path.rename(boundary.output("normalization_config.json"))
    setup = read_json(boundary.output("setup_report.json"))
    options.update(date_format=config["date_format"], blank_deducts=config["blank_deducts"],
                   deduct_families=config["deduct_families"])
    write_json(boundary.output("study.json"), options)
    raw, sources = load_sources(boundary.output("inputs"), config["input_files"], config["columns"])
    out = boundary.output("results")
    out.mkdir()
    write_json(out / "study_status.json", {"status": "running", "stage": "observations"})
    table, targets, rejected, observation_audit = build_observations(
        raw, config["columns"], config["deduct_families"], blank_deducts=config["blank_deducts"],
        date_format=config["date_format"], cutoff_year=options.get("cutoff_year", 1980))
    rejected.to_csv(out / "rejected_rows.csv", index=False)
    write_json(out / "observation_audit.json", observation_audit)
    if table.empty:
        raise SafeError("NO_VALID_INSPECTIONS")
    source_map = {Path(s["normalized"]).name: Path(s["original_path"]).name for s in setup["sources"]}
    features, feature_audit = select_covariates(table, options.get("features"))
    write_json(out / "study_status.json", {"status": "running", "stage": "climate"})
    table, climate_features, climate_audit = enrich(table, climate, out, source_map=source_map)
    if not climate_features:
        raise SafeError("NO_CLIMATE_FEATURES")
    if set(features) & set(climate_features):
        raise SafeError("DUPLICATE_STUDY_FEATURE")
    features += climate_features
    setup.update(target_mode="deduct_family_levels", features=features,
                 feature_policy="Reviewed inspection metadata and explicitly bounded historical climate",
                 feature_note="Observed deduct totals; contemporaneous PCI and distress outcomes excluded as predictors.")
    write_json(boundary.output("setup_report.json"), setup)
    table, group_audit = isolate_climate_sites(table)
    table.to_csv(out / "dataset.csv", index=False)
    eligible = table.climate_status.eq("ok")
    model_data = table.loc[eligible].copy()
    feature_audit.to_csv(out / "covariate_audit.csv", index=False)
    write_json(out / "features.json", {"features": features, "climate_features": climate_features})
    folds, seed = options.get("folds", 5), options.get("seed", 42)
    fraction = options.get("holdout_fraction", .2)
    partitions = make_partitions(model_data, folds=folds, seed=seed, holdout_fraction=fraction)
    partitions.to_csv(out / "partitions.csv", index=False)
    audit = {"schema_version": 1, "target_mode": "deduct_family_levels", "observations": observation_audit,
             "climate": climate_audit, "predictors": feature_audit.to_dict("records"),
             "features": features, "climate_features": climate_features,
             "partition": {"folds": folds, "seed": seed, "holdout_fraction": fraction,
                           "grouping": "climate_location" if group_audit["shared_climate_sites_joined"]
                           else setup["grouping"][0]["strategy"], **group_audit},
             "model_rows": len(model_data), "excluded_climate_rows": int((~eligible).sum()),
             "lookback_days": climate_audit.get("lookback_days"),
             "analysis_note": "Retrospective prediction of observed deduct levels from historical climate and reviewed covariates; not causal or a prospective forecast."}
    write_json(out / "study_audit.json", audit)
    summaries = []
    local_options = model_options(options.get("local_model_options"))
    budget = fit_budget(folds, local_options)
    task_count = model_data.material.nunique()*len(targets) if options.get("models", True) else 0
    sweep_budget = {"tasks": task_count, "per_task_upper_bound": budget,
                    "fit_calls_upper_bound": budget["fit_calls_upper_bound"]*task_count,
                    "native_fits_upper_bound": budget["native_fits_upper_bound"]*task_count}
    write_json(out/"model_fit_budget.json", sweep_budget)
    check_budget(sweep_budget, local_options)
    write_json(out / "study_status.json", {"status": "running", "stage": "target_sweep"})
    for material, subset in model_data.groupby("material", sort=True):
        numeric = subset[features].select_dtypes(include=[np.number])
        material_dir = out / "materials" / str(material)
        material_dir.mkdir(parents=True)
        numeric.corr(method="spearman").to_csv(material_dir / "feature_correlations.csv")
        for target in targets:
            if options.get("models", True):
                def build_task(destination):
                    return search_levels(subset, features, target, destination, partitions=partitions,
                        climate_features=climate_features, seed=seed, model_options=local_options)
                key = task_key(subset, partitions, features, climate_features, target, seed, None, local_options)
                cache = Path(local_options["cache_dir"]) if local_options["cache_dir"] else boundary.output("model_tasks")
                summary = verified_task(cache, key, build_task, material_dir/target)
            else:
                summary = {"target": target, "status": "modeling_disabled",
                           "valid_target_rows": int(np.isfinite(subset[target]).sum()),
                           "valid_target_groups": int(subset.loc[np.isfinite(subset[target]), "group_id"].nunique())}
            summaries.append({"material": str(material), **summary})
    write_json(out / "run_report.json", {"slots": summaries, "audit": audit})
    build_study_report(out, summaries, audit)
    write_json(out / "source_manifest.json", {"sources": sources, "location_inputs": location_hashes,
                                             "profile_sha256": profile_hash})
    for path, digest in location_hashes.items():
        if sha256(Path(path)) != digest:
            raise SafeError("INPUT_CHANGED_DURING_RUN")
    for source in setup["sources"]:
        if sha256(Path(source["original_path"])) != source["original_sha256"]:
            raise SafeError("INPUT_CHANGED_DURING_RUN")
        if sha256(boundary.path(source["snapshot"])) != source["original_sha256"]:
            raise SafeError("SOURCE_SNAPSHOT_CHANGED")
        if sha256(boundary.path(source["normalized"])) != source["normalized_sha256"]:
            raise SafeError("NORMALIZED_INPUT_CHANGED_DURING_RUN")
    if profile and sha256(profile) != profile_hash:
        raise SafeError("INPUT_CHANGED_DURING_RUN")
    write_json(out / "study_status.json", {"status": "complete", "stage": "report"})
    write_json(out / "manifest.json", {"schema_version": 1, "mode": "study", "config": options,
                                       "started": started, "finished": datetime.now(timezone.utc).isoformat(),
                                       "code_sha256": code_digest(), "environment": environment_versions(),
                                       "outputs": checksum_outputs(out)})
    return "PRIVATE_STUDY_COMPLETE"
