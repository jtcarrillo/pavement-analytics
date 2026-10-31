"""Run every original local stage, then compare end-inspection deduct targets.

Source exports, location maps and empirical curve profiles remain private.
Original numerical stages live in ``original``; this module provides setup,
provenance, resumption and the added multi-target analysis/report stages.
"""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
import shutil
import uuid

import pandas as pd

from .autorun import column_map, external_path, normalization_map
from .dates import normalize_dates
from .io import read_json, read_table, sha256, write_json
from .security import Boundary, SafeError, disjoint, lock, refuse_public_execution, reject_links

FORMATS = {".csv", ".tsv", ".xlsx", ".xlsm"}
CANONICAL = {"section": "SectionUniqueID", "date": "Date", "pci": "PCI",
             "surface": "Inspected Surface Type", "construction": "IsConstruction"}
SETUP_KEYS = {"columns", "date_format", "blank_deducts"}


def _notify(progress, event, **fields):
    if progress is not None:
        progress({"event": event, **fields})


def _setup_files(progress, setup):
    mapping = setup.get("mapping", {})
    cache = setup.get("climate_cache", {})
    _notify(progress, "setup_files", profile=setup.get("profile"),
            mapping=mapping.get("original_path"), location_directory=mapping.get("location_directory"),
            cache=cache.get("selected"), cache_source=cache.get("source"),
            cache_fallback_reason=cache.get("fallback_reason"))


def _input_progress(progress, source):
    _notify(progress, "input", path=source["original_path"], rows=source["rows"],
            location=source.get("location_path"), location_status=source.get("location_status", "unmapped"))


def _setup_complete(progress, setup, report):
    sources = setup.get("sources", [])
    counts = Counter(source.get("location_status", "unmapped") for source in sources)
    _notify(progress, "setup_complete", files=len(sources), rows=sum(s["rows"] for s in sources),
            mapped_files=counts["mapped"], unmapped_files=counts["unmapped"],
            missing_location_files=counts["missing_file"], report=str(report))


def _profile(data: Path, explicit: Path | None, public: Path) -> Path | None:
    if explicit:
        return external_path(explicit, public)
    folder = data if data.is_dir() else data.parent
    for name in ("pavement-full.json", "pipeline.json", "pipeline.example.json"):
        for directory in (folder, folder.parent):
            candidate = directory / name
            if candidate.is_file():
                return external_path(candidate, public)
    saved = Path.home() / ".config/pavement-analytics/full.json"
    return external_path(saved, public) if saved.is_file() else None


def _candidate_file(folders: list[Path], names: tuple[str, ...], public: Path) -> Path | None:
    for directory in folders:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return external_path(candidate, public)
    return None


def _snapshot(path: Path, destination: Path, checksums: dict) -> None:
    reject_links(path)
    digest = sha256(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
    if sha256(path) != digest or sha256(destination) != digest:
        raise SafeError("INPUT_CHANGED_DURING_READ")
    checksums[str(path)] = digest
    checksums[str(destination)] = digest


def _select_cache(boundary: Boundary, data: Path, base: Path,
                  configured: object, explicit: Path | None) -> tuple[Path, dict]:
    """Reuse safe caches, replacing only unsuitable inherited profile paths."""
    fallback = boundary.workspace.parent / "original-era5-cache"
    if explicit is not None:
        source, candidate = "command_line", explicit
    elif configured:
        if not isinstance(configured, str):
            raise SafeError("INVALID_FULL_PIPELINE_PATH")
        source, candidate = "profile", Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
    else:
        source, candidate = "default", fallback

    def validate(path: Path) -> Path:
        path = external_path(path, boundary.public)
        if not disjoint(path, data) or not disjoint(path, boundary.workspace):
            raise SafeError("CLIMATE_CACHE_MUST_BE_SEPARATE_FROM_INPUTS_AND_RUN")
        if any((p / ".git").exists() for p in (path, *path.parents)):
            raise SafeError("CLIMATE_CACHE_MUST_NOT_BE_VERSIONED")
        if path.exists() and not path.is_dir():
            raise SafeError("CLIMATE_CACHE_MUST_BE_DIRECTORY")
        return path

    requested = str(candidate.expanduser().absolute()) if source != "default" else None
    reason = None
    try:
        selected = validate(candidate)
    except SafeError as error:
        if source != "profile":
            raise
        reason = str(error)
        selected = validate(fallback)
    return selected, {"source": source, "requested": requested,
                      "selected": str(selected), "fallback_reason": reason}


def prepare_full(boundary: Boundary, data: Path, config_path: Path | None = None, *,
                 mapping: Path | None = None, csv_dir: Path | None = None,
                 cdv_curves: Path | None = None, cache_dir: Path | None = None,
                 date_format: str | None = None,
                 fit_deduct_curves: bool = False, progress=None) -> dict:
    from .original.pipeline import DEFAULTS, PATH_KEYS, validate_config
    from .full_mapping import normalize_location_mapping
    from .full_preflight import check_setup_inputs

    data = external_path(data, boundary.public)
    if not data.exists() or not disjoint(data, boundary.workspace):
        raise SafeError("FULL_INPUT_MUST_EXIST_OUTSIDE_WORKSPACE")
    profile = _profile(data, config_path, boundary.public)
    folder = data if data.is_dir() else data.parent
    base = profile.parent if profile else folder
    source_options = read_json(profile) if profile else {}
    if set(source_options) - set(DEFAULTS) - SETUP_KEYS:
        raise SafeError("UNKNOWN_FULL_PIPELINE_OPTION")
    if source_options.get("blank_deducts", "zero") != "zero":
        raise SafeError("ORIGINAL_PIPELINE_USES_ZERO_BLANK_DISTRESS_POLICY")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update({k: v for k, v in source_options.items() if k in DEFAULTS})
    from .model_runtime import model_options
    cfg["local_model_options"] = model_options(cfg.get("local_model_options"))
    if cfg["local_model_options"]["cache_dir"]:
        value = Path(cfg["local_model_options"]["cache_dir"]).expanduser()
        cfg["local_model_options"]["cache_dir"] = str(external_path(value if value.is_absolute() else base/value, boundary.public))
    for key in PATH_KEYS:
        # Cache precedence and legacy-path fallback are handled separately.
        if key == "cache_dir":
            continue
        if cfg.get(key):
            if not isinstance(cfg[key], str):
                raise SafeError("INVALID_FULL_PIPELINE_PATH")
            value = Path(cfg[key]).expanduser()
            cfg[key] = str(external_path(value if value.is_absolute() else base / value, boundary.public))
    for key, value in (("mapping", mapping), ("csv_dir", csv_dir), ("cdv_curves", cdv_curves)):
        if value is not None:
            cfg[key] = str(external_path(value, boundary.public))
    folders = list(dict.fromkeys((base, folder, folder.parent)))
    if not cfg.get("mapping"):
        candidate = _candidate_file(folders, ("data_file_mapping.json", "data/data_file_mapping.json", "mapping.json"), boundary.public)
        cfg["mapping"] = str(candidate) if candidate else None
    if not cfg.get("cdv_curves"):
        candidate = _candidate_file(folders, ("cdv_curves.json", "dr-multitarget/cdv_curves.json", "data/cdv_curves.json"), boundary.public)
        cfg["cdv_curves"] = str(candidate) if candidate else None
    cfg["csv_dir"] = cfg.get("csv_dir") or (str(Path(cfg["mapping"]).parent) if cfg.get("mapping") else None)
    selected_cache, cache_audit = _select_cache(boundary, data, base, cfg.get("cache_dir"), cache_dir)
    cfg["cache_dir"] = str(selected_cache)
    retired_backend_request = source_options.get("datarobot", "off")
    cfg["datarobot"] = "off"
    cfg["retired_backend_request"] = retired_backend_request if retired_backend_request != "off" else None
    if fit_deduct_curves:
        cfg["fit_deduct_curves"] = True
    # These are the semantics of full: an older skip flag cannot silently omit
    # a requested original stage. Curve fitting remains an explicit original option.
    enforced = {"skip_climate": False, "skip_distress": False, "skip_pci": False, "no_shapley": False}
    overrides = {k: {"previous": cfg[k], "full": v} for k, v in enforced.items() if cfg[k] != v}
    cfg.update(enforced)
    cfg["target_mode"] = source_options.get("target_mode", "deduct_families")
    if cfg["target_mode"] != "deduct_families":
        raise SafeError("FULL_MODEL_SWEEP_REQUIRES_DEDUCT_FAMILY_TARGETS")
    if not 0 < cfg["dr_holdout_pct"] < 50:
        raise SafeError("FULL_COMPARISON_REQUIRES_HOLDOUT_BETWEEN_ZERO_AND_FIFTY_PERCENT")
    missing = [key for key in ("mapping", "cdv_curves") if not cfg.get(key)]
    setup = {"schema_version": 1, "mode": "original_full_pipeline", "profile": str(profile) if profile else None,
             "original_data": str(data), "missing_required": missing, "forced_stage_settings": overrides,
             "blank_distress_policy": "Original structural-zero assumption for exported blank distress cells",
             "target_mode": cfg["target_mode"], "model_backend": "local",
             "retired_backend_request_ignored": retired_backend_request,
             "climate_cache": cache_audit,
             "mapping": {"original_path": cfg.get("mapping"), "location_directory": cfg.get("csv_dir")},
             "sources": [], "date_parsing": [], "column_mapping": [],
             "checksums": {}}
    write_json(boundary.output("full_setup_report.json"), setup)
    _setup_files(progress, setup)
    if missing:
        raise SafeError("FULL_SETUP_REQUIRES_LOCATION_MAPPING_AND_PRIVATE_CDV_CURVES")
    setup["input_checks"] = check_setup_inputs(cfg)
    write_json(boundary.output("full_setup_report.json"), setup)
    problems = [item for item in setup["input_checks"] if item["status"] != "ready"]
    for item in problems:
        _notify(progress, "setup_issue", **item)
    if problems:
        raise SafeError("FULL_SETUP_INPUT_PATHS_INVALID")
    cfg["input_dir"] = str(boundary.output("inputs"))
    cfg["output_dir"] = str(boundary.output("pipeline"))
    validate_config(cfg)
    checksums = setup["checksums"]
    if profile:
        _snapshot(profile, boundary.output("originals/full_profile.json"), checksums)
    source_mapping = read_json(Path(cfg["mapping"]))
    location_root = Path(cfg["csv_dir"])
    if data.is_file():
        files = [data] if data.suffix.lower() in FORMATS else []
    else:
        candidates = data.rglob("*") if cfg["recursive"] else data.iterdir()
        files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in FORMATS
                       and not p.name.startswith("~$"))
    source_mapping, mapping_audit = normalize_location_mapping(source_mapping, location_root, files, boundary.public)
    setup["mapping_entries"] = mapping_audit
    write_json(boundary.output("full_setup_report.json"), setup)
    for item in mapping_audit:
        if item["status"] != "valid":
            _notify(progress, "mapping_entry", **item)
    if any(item["status"] == "invalid" for item in mapping_audit):
        raise SafeError("INVALID_LOCATION_MAPPING")
    location_paths = {external_path(location_root / name, boundary.public) for name in source_mapping.values()}
    files = [path for path in files if path.resolve() not in location_paths]
    if not files:
        raise SafeError("NO_INSPECTION_INPUT_FILES")
    if len({p.name for p in files}) != len(files):
        raise SafeError("DUPLICATE_INPUT_BASENAMES")
    snapshot_map = {}
    normalized_root = boundary.output("inputs")
    normalized_root.mkdir()
    explicit_format = date_format or source_options.get("date_format")
    if explicit_format is not None and (not isinstance(explicit_format, str) or not explicit_format.strip()):
        raise SafeError("INVALID_DATE_FORMAT")
    for index, source in enumerate(files, 1):
        source = external_path(source, boundary.public)
        _notify(progress, "input_started", path=str(source), index=index, files=len(files))
        snapshot = boundary.output(f"originals/input_{index:04d}{source.suffix.lower()}")
        _snapshot(source, snapshot, checksums)
        frame = read_table(snapshot)
        column_audit = {"original_filename": source.name}
        setup["column_mapping"].append(column_audit)
        chosen = column_map(list(frame), source_options.get("columns", {}), diagnostics=column_audit)
        expected = {CANONICAL[role]: name for role, name in chosen.items() if role in CANONICAL}
        frame = frame.rename(columns=normalization_map(list(frame), expected))
        date_audit = {"original_filename": source.name, "mode": "explicit" if explicit_format else "automatic"}
        setup["date_parsing"].append(date_audit)
        if explicit_format:
            dates = pd.to_datetime(frame.Date, format=explicit_format, errors="coerce")
            frame["Date"] = dates.dt.strftime("%Y-%m-%d").fillna("")
            date_audit.update(format=explicit_format, valid_rows=int(dates.notna().sum()), invalid_rows=int(dates.isna().sum()))
        else:
            frame["Date"] = normalize_dates(frame.Date, diagnostics=date_audit)
        output = normalized_root / f"input_{index:04d}.csv"
        frame.to_csv(output, index=False)
        checksums[str(output)] = sha256(output)
        source_report = {"original_path": str(source), "snapshot": str(snapshot),
                         "normalized": str(output), "rows": len(frame),
                         "observed_deduct_columns": [c for c in frame if c.endswith(" deduct")],
                         "location_path": None, "location_status": "unmapped"}
        setup["sources"].append(source_report)
        if source.name in source_mapping:
            original_location = external_path(location_root / source_mapping[source.name], boundary.public)
            source_report["location_path"] = str(original_location)
            if original_location.is_file():
                location_snapshot = boundary.output(f"location_inputs/location_{index:04d}.csv")
                _snapshot(original_location, location_snapshot, checksums)
                snapshot_map[output.name] = location_snapshot.name
                source_report.update(location_status="mapped", location_snapshot=str(location_snapshot))
            else:
                source_report["location_status"] = "missing_file"
                setup.setdefault("missing_location_files", []).append(str(original_location))
        write_json(boundary.output("full_setup_report.json"), setup)
        _input_progress(progress, source_report)
    _snapshot(Path(cfg["mapping"]), boundary.output("originals/location_mapping.json"), checksums)
    mapping_snapshot = boundary.output("location_inputs/mapping.json")
    mapping_snapshot.parent.mkdir(exist_ok=True)
    write_json(mapping_snapshot, snapshot_map)
    checksums[str(mapping_snapshot)] = sha256(mapping_snapshot)
    cfg.update(mapping=str(mapping_snapshot), csv_dir=str(mapping_snapshot.parent), recursive=False)
    for key in ("cdv_curves", "deduct_curves", "families"):
        if cfg.get(key):
            destination = boundary.output(f"profiles/{key}.json")
            _snapshot(Path(cfg[key]), destination, checksums)
            cfg[key] = str(destination)
    write_json(boundary.output("full_setup_report.json"), setup)
    write_json(boundary.output("full_config.json"), cfg)
    _setup_complete(progress, setup, boundary.path("full_setup_report.json"))
    return cfg


def _verify_setup(boundary: Boundary) -> dict:
    setup = read_json(boundary.path("full_setup_report.json"))
    for path, digest in setup.get("checksums", {}).items():
        resolved = external_path(Path(path), boundary.public)
        if not resolved.is_file() or sha256(resolved) != digest:
            raise SafeError("FULL_PIPELINE_INPUT_OR_SNAPSHOT_CHANGED")
    return setup


def full(boundary: Boundary, data: Path | None = None, config_path: Path | None = None, *,
         resume: bool = False, mapping: Path | None = None, csv_dir: Path | None = None,
         cdv_curves: Path | None = None, cache_dir: Path | None = None,
         date_format: str | None = None,
         fit_deduct_curves: bool = False, progress=None) -> str:
    from .original.pipeline import Pipeline, discover, validate_config
    from .original.common import Run
    from .full_modeling import run_full_models, consolidate_full_models

    refuse_public_execution()
    with lock(boundary.output(".full.lock")):
        _notify(progress, "setup_started", workspace=str(boundary.workspace),
                data=str(data) if data is not None else None, resume=resume)
        try:
            if resume:
                if any(v is not None for v in (config_path, mapping, csv_dir, cdv_curves, cache_dir, date_format)) or fit_deduct_curves:
                    raise SafeError("FULL_RESUME_REQUIRES_UNCHANGED_CONFIGURATION")
                cfg = read_json(boundary.path("full_config.json"))
                setup = _verify_setup(boundary)
                if data is not None and str(external_path(data, boundary.public)) != setup["original_data"]:
                    raise SafeError("FULL_RESUME_DATA_PATH_CHANGED")
                _setup_files(progress, setup)
                for source in setup["sources"]:
                    _input_progress(progress, source)
                _setup_complete(progress, setup, boundary.path("full_setup_report.json"))
            else:
                if data is None:
                    raise SafeError("FULL_PIPELINE_DATA_PATH_REQUIRED")
                cfg = prepare_full(boundary, data, config_path, mapping=mapping, csv_dir=csv_dir,
                                   cdv_curves=cdv_curves, cache_dir=cache_dir, date_format=date_format,
                                   fit_deduct_curves=fit_deduct_curves, progress=progress)
            validate_config(cfg)
            # Sources/artifacts stay inside the run; the reusable cache is external.
            for key in ("input_dir", "output_dir", "mapping", "csv_dir", "cdv_curves", "deduct_curves", "families"):
                if cfg.get(key):
                    boundary.path(cfg[key])
            files, mapping_data, inputs, fingerprint = discover(cfg)
            run = Run(cfg["output_dir"], fingerprint, cfg, inputs, resume, progress=progress)
        except BaseException as error:
            _notify(progress, "setup_failed", code=str(error) if isinstance(error, SafeError) else
                    "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "PRIVATE_SETUP_FAILED",
                    log=str(boundary.path("logs/runtime.log")))
            raise
        run.data["config"] = cfg
        try:
            code = Pipeline(cfg, run, files, mapping_data).execute()
            if code:
                raise SafeError("FULL_PIPELINE_WARNING_GATE_CHECK_PRIVATE_REPORT")
            pipeline_out = boundary.path(cfg["output_dir"])
            models_out = pipeline_out / "09_local_models"
            if run.data["stages"].get("09_local_models", {}).get("status") != "completed" and models_out.exists():
                failed = boundary.output("failed")
                failed.mkdir(exist_ok=True)
                models_out.rename(failed / ("local-models-" + uuid.uuid4().hex[:8]))
            def model_stage():
                summaries, audit, artifacts = run_full_models(pipeline_out, run.data, cfg, out=models_out, progress=progress)
                return {"targets": len(summaries), "status_counts": dict(Counter(s["status"] for s in summaries)),
                        "output": str(models_out.relative_to(pipeline_out))}, artifacts
            run.stage("09_local_models", model_stage)
            saved = read_json(models_out / "sweep_summary.json")
            report_out = pipeline_out / "10_report"
            report_key = "10_report"
            if run.data["stages"].get(report_key, {}).get("status") != "completed" and report_out.exists():
                failed = boundary.output("failed")
                failed.mkdir(exist_ok=True)
                report_out.rename(failed / ("report-" + uuid.uuid4().hex[:8]))
            def report_stage():
                artifacts = consolidate_full_models(models_out, saved["summaries"], saved["audit"], report_out=report_out,
                                                    pipeline_report=run.data, qualification_link=True)
                return {"dashboard": str((report_out / "dashboard.html").relative_to(pipeline_out)),
                        "local_models": "completed", "model_backend": "local"}, artifacts
            run.stage(report_key, report_stage)
            _verify_setup(boundary)
            qualification_out = pipeline_out/"11_qualification"
            if run.data["stages"].get("11_qualification", {}).get("status") != "completed" and qualification_out.exists():
                failed = boundary.output("failed")
                failed.mkdir(exist_ok=True)
                qualification_out.rename(failed/("qualification-"+uuid.uuid4().hex[:8]))
            def qualification_stage():
                from .run_qualification import audit_full_run
                evidence, artifacts = audit_full_run(pipeline_out, run.data, cfg, _verify_setup(boundary), qualification_out,
                                                    progress=progress)
                return {"prediction_checks": evidence["prediction_checks"], "verified_artifacts": evidence["verified_artifacts"],
                        "pending_release_gates": len(evidence["remaining_release_gates"])}, artifacts
            run.stage("11_qualification", qualification_stage)
            write_json(boundary.output("full_config.json"), cfg)
            write_json(boundary.output("full_result.json"), {"status": "completed", "model_backend": "local",
                                                            "dashboard": str(report_out / "dashboard.html"),
                                                            "qualification": str(qualification_out/"qualification.html"),
                                                            "run_report": str(run.path)})
            _notify(progress, "output", dashboard=str(report_out / "dashboard.html"),
                    models=str(models_out), report=str(run.path), qualification=str(qualification_out/"qualification.html"))
            run.finish("completed_with_warnings" if run.data["warnings"] else "completed")
        except BaseException as error:
            run.finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", str(error))
            if isinstance(error, (SafeError, KeyboardInterrupt)):
                raise
            raise SafeError("ORIGINAL_PIPELINE_FAILED_CHECK_PRIVATE_STAGE_LOG") from error
    return "PRIVATE_FULL_PIPELINE_COMPLETE"
