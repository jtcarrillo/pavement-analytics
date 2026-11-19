"""Compare pavement targets after every original curation stage has completed.

This adapter consumes the original feature audit and shared partition plan. It
does not re-derive climate, PCI, distress families, or inspection eligibility.
"""
from __future__ import annotations

import hashlib
import json
import os
from itertools import chain
from pathlib import Path
import re
import tempfile
import uuid
from urllib.parse import quote

import numpy as np
import pandas as pd

from .io import read_json, write_json
from .security import Boundary, SafeError, disjoint, reject_links, source_root
from .study_modeling import search_levels
from .study_reporting import build_study_report
from .model_runtime import model_options, fit_budget, check_budget
from .model_tasks import task_key, verified_task
from .rate_diagnostics import rate_interval_diagnostics


def _read(path: Path) -> pd.DataFrame:
    reject_links(path)
    if not path.is_file():
        raise SafeError("FULL_MODEL_STAGE_OUTPUT_MISSING")
    try:
        # Group identifiers such as 001 must survive each handoff unchanged.
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise SafeError("FULL_MODEL_STAGE_OUTPUT_EMPTY") from None


def target_alias(target: str) -> str:
    if target == "DetRatePerYear":
        return target
    match = re.fullmatch(r"fam_([A-Za-z0-9_]+) deduct_end", target)
    if match:
        return "level_" + match.group(1)
    return "target_" + hashlib.sha256(target.encode()).hexdigest()[:20]


def _feature_guard(name: str, targets: list[str]) -> None:
    """Check the original audit cannot accidentally admit an endpoint outcome."""
    if (name in targets or name in {"delta_days", "DetRatePerYear", "row_id", "group_id"}
            or name.startswith(("partition_", "_source_", "_pipeline_", "delta_", "export_gap_", "pci_diff_"))
            or name.endswith(("_end", "_delta"))
            or re.search(r" (emerged|regressed|resolved|persistent)$", name)
            or re.match(r"^(IntervalStart|IntervalEnd|IntervalYears|DataSource|SnapDistance)(_|$)", name)):
        raise SafeError("UNSAFE_ORIGINAL_MODEL_FEATURE")


def _header(path: Path) -> list[str]:
    reject_links(path)
    if not path.is_file():
        return []
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except pd.errors.EmptyDataError:
        return []


def _climate_columns(pipeline_out: Path, surface: str, features: list[str]) -> list[str]:
    before = set(_header(pipeline_out / "03_location_rate" / f"{surface}.csv"))
    after = set(_header(pipeline_out / "04_climate" / f"{surface}.csv"))
    return [name for name in features if name in after - before]


def _validate_partitions(plan: pd.DataFrame) -> pd.DataFrame:
    required = {"group_id", "partition", "fold", "planned_folds"}
    if not required <= set(plan) or plan.group_id.duplicated().any() or plan.group_id.str.strip().eq("").any():
        raise SafeError("FULL_MODEL_SHARED_PARTITIONS_INVALID")
    if not plan.partition.isin(["development", "holdout"]).all():
        raise SafeError("FULL_MODEL_SHARED_PARTITIONS_INVALID")
    for column in ("fold", "planned_folds"):
        values = pd.to_numeric(plan[column], errors="coerce")
        if not np.isfinite(values).all() or (values % 1 != 0).any():
            raise SafeError("FULL_MODEL_SHARED_PARTITIONS_INVALID")
        plan[column] = values.astype(int)
    if (not plan.loc[plan.partition.eq("holdout"), "fold"].eq(-1).all()
            or not plan.loc[plan.partition.eq("development"), "fold"].ge(0).all()
            or (len(plan) and (plan.planned_folds.nunique() != 1 or plan.planned_folds.iloc[0] < 2))):
        raise SafeError("FULL_MODEL_SHARED_PARTITIONS_INVALID")
    return plan


def _stage_stats(report: dict, name: str) -> dict:
    value = report.get("stages", {}).get(name, {}).get("stats", {})
    return value if isinstance(value, dict) else {}


def _target_metadata(original_target):
    if original_target == "DetRatePerYear":
        return {"target_kind": "signed_pci_rate", "target_units": "PCI points/year"}
    return {"target_kind": "deduct_total", "target_units": "deduct points"}


def _digest(records):
    digest = hashlib.sha256()
    for record in records:
        line = json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        digest.update(line.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _rate_benchmark(pipeline_out, surface, table, features, plan, cfg):
    """Fingerprint the evaluation inputs independently of CSV row order and model code."""
    curated = pipeline_out / "06_curated" / (surface + ".csv")
    keys = ["_source_file_start", "_source_file_end", "SectionUniqueID", "start_date", "end_date"]
    setup = pipeline_out.parent / "full_setup_report.json"
    reject_links(setup)
    source_names = {}
    if setup.is_file():
        source_names = {Path(item["normalized"]).name: Path(item["original_path"]).name
                        for item in read_json(setup)["sources"]}
    pair_ids = pd.Series("", index=table.index, dtype=str)
    if set(keys) <= set(_header(curated)) and "_pipeline_model_row" in table:
        identities = pd.read_csv(curated, usecols=keys, dtype=str, keep_default_na=False)
        positions = pd.to_numeric(table["_pipeline_model_row"], errors="coerce")
        for index, position in positions.items():
            if pd.isna(position) or not float(position).is_integer() or not 0 <= position < len(identities):
                continue
            start, end, section, first, last = identities.loc[int(position), keys]
            if all((start, end, section, first, last)):
                pair_ids.loc[index] = _digest([[surface, source_names.get(start, start),
                                                source_names.get(end, end), section, first, last]])
    pair_ids.loc[pair_ids.duplicated(keep=False)] = ""
    splits = plan.set_index("group_id")
    folds = table.group_id.map(splits.fold)
    roles = table.group_id.map(splits.partition)
    valid = np.isfinite(table.DetRatePerYear)
    records = []
    for role in ("development", "holdout"):
        selected = pair_ids.loc[valid & roles.eq(role)].sort_values()
        count = len(selected)
        status = "no_eligible_rows" if count == 0 else "ready" if selected.ne("").all() else "identity_unavailable"
        hashes = {key: None for key in ("pair_ids_sha256", "outcomes_weights_sha256", "group_folds_sha256", "features_sha256")}
        if status == "ready":
            idx = selected.index
            groups = table.loc[idx, "group_id"]
            if cfg.get("airfield_col", "_source_file_start") == "_source_file_start":
                groups = groups.map(lambda group: source_names.get(group, group))
            hashes["pair_ids_sha256"] = _digest(selected)
            hashes["outcomes_weights_sha256"] = _digest(zip(selected, table.loc[idx, "DetRatePerYear"].astype(float),
                                                              table.loc[idx, "delta_days"].astype(float)))
            hashes["group_folds_sha256"] = _digest(zip(selected, groups, folds.loc[idx].astype(int)))
            feature_rows = table.loc[idx, features].itertuples(index=False, name=None)
            feature_records = ([key, *[None if pd.isna(value) else str(value) for value in values]]
                               for key, values in zip(selected, feature_rows))
            hashes["features_sha256"] = _digest(chain([features], feature_records))
        records.append({"material": surface, "partition": role, "eligible_rows": count,
                        "identity_status": status, **hashes})
    return pair_ids, records


def _slot_without_features(table, target, original_target):
    metadata = _target_metadata(original_target)
    signed = metadata["target_kind"] == "signed_pci_rate"
    valid = np.isfinite(table[target])
    return {**metadata, "target": target, "original_target": original_target, "status": "no_features",
            "candidate": None, "valid_target_rows": int(valid.sum()),
            "valid_target_groups": int(table.loc[valid, "group_id"].nunique()),
            "excluded_missing_target": int((~valid).sum()), "importance_reliable": False,
            "nonzero_fraction": float(table.loc[valid, target].ne(0).mean()) if valid.any() else None,
            "selection_metric": "mean_interval_weighted_cv_RMSE" if signed else "mean_interval_weighted_cv_MAE", "weight_column": "delta_days"}


def run_full_models(pipeline_out: Path, pipeline_report: dict, cfg: dict,
                    out: Path | None = None, *, progress=None) -> tuple[list[dict], dict, list[Path]]:
    """Run the local target sweep; return summaries, report audit, and artifacts.

    The shared plan must already have been written by original stage 07. The
    original stage artifacts remain unchanged. Partial failures are retained
    separately, so a verified original-pipeline resume can retry this stage.
    """
    pipeline_out = Path(pipeline_out)
    out = Path(out) if out is not None else pipeline_out / "09_local_models"
    reject_links(pipeline_out)
    reject_links(out)
    if out.exists():
        raise SafeError("FULL_MODEL_OUTPUT_EXISTS_USE_VERIFIED_RESUME")
    Boundary.create(source_root(), out.parent)
    plan = _validate_partitions(_read(pipeline_out / "07_modeling" / "partitions.csv"))
    surfaces = cfg.get("surfaces", ["PCC", "APC", "AC_AAC_ACC"])
    if (not isinstance(surfaces, list) or len(set(surfaces)) != len(surfaces)
            or any(s not in {"PCC", "APC", "AC_AAC_ACC"} for s in surfaces)):
        raise SafeError("FULL_MODEL_SURFACES_INVALID")
    prepared, target_map, rows, next_row = [], [], [], 0
    benchmark = []
    options = model_options(cfg.get("local_model_options"))
    all_features, all_climate = set(), set()
    for surface in surfaces:
        folder = pipeline_out / "07_modeling" / "partitioned"
        if not (folder / f"{surface}.csv").is_file():
            raise SafeError("FULL_MODEL_PARTITIONED_TABLE_REQUIRED")
        table = _read(folder / f"{surface}.csv")
        audit = _read(folder / f"{surface}_feature_audit.csv")
        if not {"column", "disposition"} <= set(audit) or audit.column.duplicated().any():
            raise SafeError("FULL_MODEL_FEATURE_AUDIT_INVALID")
        targets = audit.loc[audit.disposition.eq("TARGET"), "column"].tolist()
        if not targets:
            targets = list(_stage_stats(pipeline_report, "07_modeling_" + surface).get("targets", {}))
        features = audit.loc[audit.disposition.eq("KEEP"), "column"].tolist()
        if not set([*targets, *features, "partition_airfield", "delta_days"]) <= set(table):
            raise SafeError("FULL_MODEL_FEATURE_AUDIT_MISMATCH")
        for feature in features:
            _feature_guard(feature, targets)
        climate = _climate_columns(pipeline_out, surface, features)
        table["group_id"] = table.partition_airfield.astype(str)
        if table.group_id.str.strip().eq("").any() or not set(table.group_id) <= set(plan.group_id):
            raise SafeError("FULL_MODEL_GROUP_NOT_IN_SHARED_PLAN")
        if "partition_fold" in table:
            observed = pd.to_numeric(table.partition_fold, errors="coerce")
            expected = table.group_id.map(plan.set_index("group_id").fold)
            if not observed.eq(expected).all():
                raise SafeError("FULL_MODEL_PARTITION_ASSIGNMENT_MISMATCH")
        table["delta_days"] = pd.to_numeric(table.delta_days, errors="coerce")
        if not np.isfinite(table.delta_days).all() or table.delta_days.le(0).any():
            raise SafeError("FULL_MODEL_INTERVAL_WEIGHTS_INVALID")
        table["row_id"] = np.arange(next_row, next_row + len(table), dtype=int)
        next_row += len(table)
        aliases = {}
        for target in targets:
            alias = target_alias(target)
            if alias in aliases or (alias in table and alias != target):
                raise SafeError("FULL_MODEL_TARGET_ALIAS_COLLISION")
            aliases[alias] = target
            table[alias] = pd.to_numeric(table[target], errors="coerce")
            target_map.append({"material": surface, "target": alias, "original_target": target, **_target_metadata(target)})
        pair_ids = None
        if "DetRatePerYear" in aliases:
            pair_ids, evidence = _rate_benchmark(pipeline_out, surface, table, features, plan, cfg)
            benchmark.extend(evidence)
        for position, row_id in enumerate(table.row_id):
            rows.append({"row_id": int(row_id), "material": surface, "model_csv_row": position + 2,
                         "curated_row": table.iloc[position].get("_pipeline_model_row"),
                         "pair_id": pair_ids.iloc[position] if pair_ids is not None else ""})
        prepared.append((surface, table, features, climate, aliases))
        all_features.update(features)
        all_climate.update(climate)
    kinds = [entry["target_kind"] for entry in target_map]
    by_kind = {kind: fit_budget(int(plan.planned_folds.iloc[0]), options, target_kind=kind)
               for kind in sorted(set(kinds))}
    budget = {"tasks": len(kinds), "per_target_kind": by_kind,
              "fit_calls_upper_bound": sum(by_kind[k]["fit_calls_upper_bound"] for k in kinds),
              "native_fits_upper_bound": sum(by_kind[k]["native_fits_upper_bound"] for k in kinds)}
    if "deduct_total" in by_kind:
        budget["per_task_upper_bound"] = by_kind["deduct_total"]
    write_json(pipeline_out/"model_fit_budget.json", budget)
    check_budget(budget, options)
    staging = Path(tempfile.mkdtemp(prefix=".full-models-", dir=out.parent))
    try:
        write_json(staging/"model_fit_budget.json", budget)
        plan.to_csv(staging / "partitions.csv", index=False)
        pd.DataFrame(target_map, columns=["material", "target", "original_target", "target_kind", "target_units"]).to_csv(staging / "target_map.csv", index=False)
        if benchmark:
            pd.DataFrame(benchmark).to_csv(staging / "rate_benchmark.csv", index=False)
        pd.DataFrame(rows, columns=["row_id", "material", "model_csv_row", "curated_row", "pair_id"]).to_csv(staging / "row_map.csv", index=False)
        summaries = []
        interval_diagnostics = []
        for surface, table, features, climate, aliases in prepared:
            material_dir = staging / "materials" / surface
            material_dir.mkdir(parents=True)
            write_json(material_dir / "features.json", {"features": features, "climate_features": climate})
            numeric = table[features].apply(pd.to_numeric, errors="coerce")
            numeric.corr(method="spearman").to_csv(material_dir / "feature_correlations.csv")
            for alias, original_target in aliases.items():
                metadata = _target_metadata(original_target)
                target_kind = metadata["target_kind"]
                if progress is not None:
                    progress({"event": "target", "material": surface,
                              "target": original_target, "status": "started"})
                task_dir = material_dir / alias
                if features:
                    def model_progress(event):
                        if progress is not None:
                            progress({**event, "material": surface, "target": original_target})
                    def build_task(destination):
                        result = search_levels(table, features, alias, destination, partitions=plan,
                            climate_features=climate, seed=cfg.get("seed", 42), weight_column="delta_days",
                            model_options=options, progress=model_progress, target_kind=target_kind)
                        result.update(original_target=original_target, **metadata)
                        write_json(destination/"summary.json", result)
                        return result
                    key = task_key(table, plan, features, climate, alias, cfg.get("seed", 42), "delta_days", options,
                                   target_kind=target_kind, extra={"original_target": original_target})
                    cache = Path(options["cache_dir"]) if options["cache_dir"] else pipeline_out/"09_model_tasks"
                    summary = verified_task(cache, key, build_task, task_dir, model_progress)
                else:
                    summary = _slot_without_features(table, alias, original_target)
                    task_dir.mkdir()
                    write_json(task_dir / "summary.json", summary)
                if target_kind == "signed_pci_rate":
                    prediction_path = task_dir / "holdout_predictions.csv"
                    predictions = _read(prediction_path) if prediction_path.is_file() else None
                    interval_diagnostics.extend(rate_interval_diagnostics(table, plan, predictions, surface))
                summaries.append({"material": surface, **summary})
                if progress is not None:
                    progress({"event": "target", "material": surface,
                              "target": original_target, "status": "completed",
                              "result_status": summary["status"]})
        if interval_diagnostics:
            pd.DataFrame(interval_diagnostics).to_csv(staging / "rate_interval_diagnostics.csv", index=False)
        climate_stats = [_stage_stats(pipeline_report, "04_climate_" + s) for s in surfaces]
        partition_stats = _stage_stats(pipeline_report, "07_partitions")
        input_stats = _stage_stats(pipeline_report, "01_split")
        audit = {"schema_version": 1, "task_mode": "original_pipeline",
                 "target_mode": ("end_inspection_deduct_family_totals_and_pci_rate" if "signed_pci_rate" in kinds
                                 else "end_inspection_deduct_family_totals"), "metric_weighting": "interval_days",
                 "observations": {"accounting": {"input_rows": input_stats.get("input_rows"),
                                                   "accepted_rows": input_stats.get("retained_rows"),
                                                   "rejected_rows": input_stats.get("excluded_rows")}},
                 "features": sorted(all_features), "climate_features": sorted(all_climate),
                 "model_rows": next_row,
                 "climate": {"mode": "original_era5_interval_pg_aging", "rows": sum(int(v.get("rows", 0)) for v in climate_stats),
                             "rows_with_climate": sum(int(v.get("rows_with_climate", 0)) for v in climate_stats),
                             "download_start_year": cfg.get("climate_start_year"),
                             "download_end_year": cfg.get("climate_end_year"),
                             "historical_window_verified": None},
                 "partition": {"grouping": "declared_group", "validation_groups": len(plan),
                               "shared_plan": "07_modeling/partitions.csv", **partition_stats},
                 "analysis_note": "Original inspection-pair, interval climate, pavement-grade and construction-aging definitions are preserved. This is retrospective explanatory modeling, not a prospective forecast.",
                 "stages": {key: value.get("status") for key, value in pipeline_report.get("stages", {}).items()},
                 "source_report": "run_report.json"}
        write_json(staging / "sweep_summary.json", {"summaries": summaries, "audit": audit})
        staging.rename(out)
    except Exception:
        if staging.exists():
            failed = out.parent / ("09_local_models_failed_" + uuid.uuid4().hex[:12])
            staging.rename(failed)
        raise
    return summaries, audit, sorted(path for path in out.rglob("*") if path.is_file())


def consolidate_full_models(out: Path, summaries: list[dict] | None = None,
                            audit: dict | None = None, report_out: Path | None = None, *,
                            pipeline_report: dict | None = None, qualification_link: bool = False) -> list[Path]:
    """Build the report separately so its checkpoint does not rewrite model files."""
    out = Path(out)
    if summaries is None or audit is None:
        saved = read_json(out / "sweep_summary.json")
        summaries, audit = saved["summaries"], saved["audit"]
    audit = dict(audit)
    report_out = Path(report_out) if report_out is not None else out.parent / "10_report"
    if not disjoint(report_out.resolve(), out.resolve()):
        raise SafeError("FULL_REPORT_MUST_BE_SEPARATE_FROM_MODELS")
    Boundary.create(source_root(), report_out)
    if pipeline_report is not None:
        audit["stages"] = {key: value.get("status") for key, value in pipeline_report.get("stages", {}).items()}
    write_json(report_out / "report_audit.json", audit)
    build_study_report(report_out, summaries, audit, model_root=out)
    links = []
    if (out / "rate_benchmark.csv").is_file():
        benchmark = quote(Path(os.path.relpath(out / "rate_benchmark.csv", report_out)).as_posix())
        intervals = quote(Path(os.path.relpath(out / "rate_interval_diagnostics.csv", report_out)).as_posix())
        links.append(f'<section><h2>Compare PCI deterioration runs</h2><p><a href="{benchmark}">Evaluation fingerprints by surface and partition</a> · '
                     f'<a href="{intervals}">Errors and coverage by inspection interval</a></p></section>')
    if qualification_link:
        links.append('<section><h2>Run qualification</h2><p><a href="../11_qualification/qualification.html">Open integrity, saved-model and deployment evidence</a>. Cached ERA5 and observed live downloads are reported separately.</p></section>')
    if links:
        dashboard = report_out / "dashboard.html"
        html = dashboard.read_text(encoding="utf-8").replace("<footer>", "".join(links) + "<footer>", 1)
        dashboard.write_text(html, encoding="utf-8")
    return [report_out / "report_audit.json", *[report_out / name for name in (
        "dashboard.html", "target_summary.csv", "feature_importance.csv", "model_leaderboard.csv", "predictions.csv",
        "two_stage_predictions.csv", "two_stage_metrics.csv", "candidate_comparisons.csv", "family_summary.csv",
        "candidate_predictions.csv", "family_importance.csv", "model_runtime.csv")]]
