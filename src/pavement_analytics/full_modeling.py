"""Compare deduct targets after every original curation stage has completed.

This adapter consumes the original feature audit and shared partition plan. It
does not re-derive climate, PCI, distress families, or inspection eligibility.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import tempfile
import uuid

import numpy as np
import pandas as pd

from .io import read_json, write_json
from .security import Boundary, SafeError, disjoint, reject_links, source_root
from .study_modeling import search_levels
from .study_reporting import build_study_report


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


def _slot_without_features(table, target, original_target):
    valid = np.isfinite(table[target])
    return {"target": target, "original_target": original_target, "status": "no_features",
            "candidate": None, "valid_target_rows": int(valid.sum()),
            "valid_target_groups": int(table.loc[valid, "group_id"].nunique()),
            "excluded_missing_target": int((~valid).sum()), "importance_reliable": False,
            "nonzero_fraction": float(table.loc[valid, target].gt(0).mean()) if valid.any() else None,
            "selection_metric": "mean_interval_weighted_cv_MAE", "weight_column": "delta_days"}


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
        for position, row_id in enumerate(table.row_id):
            rows.append({"row_id": int(row_id), "material": surface, "model_csv_row": position + 2,
                         "curated_row": table.iloc[position].get("_pipeline_model_row")})
        aliases = {}
        for target in targets:
            alias = target_alias(target)
            if alias in aliases or (alias in table and alias != target):
                raise SafeError("FULL_MODEL_TARGET_ALIAS_COLLISION")
            aliases[alias] = target
            table[alias] = pd.to_numeric(table[target], errors="coerce")
            target_map.append({"material": surface, "target": alias, "original_target": target})
        prepared.append((surface, table, features, climate, aliases))
        all_features.update(features)
        all_climate.update(climate)
    staging = Path(tempfile.mkdtemp(prefix=".full-models-", dir=out.parent))
    try:
        plan.to_csv(staging / "partitions.csv", index=False)
        pd.DataFrame(target_map, columns=["material", "target", "original_target"]).to_csv(staging / "target_map.csv", index=False)
        pd.DataFrame(rows, columns=["row_id", "material", "model_csv_row", "curated_row"]).to_csv(staging / "row_map.csv", index=False)
        summaries = []
        for surface, table, features, climate, aliases in prepared:
            material_dir = staging / "materials" / surface
            material_dir.mkdir(parents=True)
            write_json(material_dir / "features.json", {"features": features, "climate_features": climate})
            numeric = table[features].apply(pd.to_numeric, errors="coerce")
            numeric.corr(method="spearman").to_csv(material_dir / "feature_correlations.csv")
            for alias, original_target in aliases.items():
                if progress is not None:
                    progress({"event": "target", "material": surface,
                              "target": original_target, "status": "started"})
                task_dir = material_dir / alias
                if features:
                    summary = search_levels(table, features, alias, task_dir, partitions=plan,
                                            climate_features=climate, seed=cfg.get("seed", 42),
                                            weight_column="delta_days")
                    summary["original_target"] = original_target
                    write_json(task_dir / "summary.json", summary)
                else:
                    summary = _slot_without_features(table, alias, original_target)
                    task_dir.mkdir()
                    write_json(task_dir / "summary.json", summary)
                summaries.append({"material": surface, **summary})
                if progress is not None:
                    progress({"event": "target", "material": surface,
                              "target": original_target, "status": "completed",
                              "result_status": summary["status"]})
        climate_stats = [_stage_stats(pipeline_report, "04_climate_" + s) for s in surfaces]
        partition_stats = _stage_stats(pipeline_report, "07_partitions")
        input_stats = _stage_stats(pipeline_report, "01_split")
        audit = {"schema_version": 1, "task_mode": "original_pipeline",
                 "target_mode": "end_inspection_deduct_family_totals", "metric_weighting": "interval_days",
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
                            pipeline_report: dict | None = None) -> list[Path]:
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
    return [report_out / "report_audit.json", *[report_out / name for name in (
        "dashboard.html", "target_summary.csv", "feature_importance.csv", "model_leaderboard.csv", "predictions.csv",
        "two_stage_predictions.csv", "two_stage_metrics.csv")]]
