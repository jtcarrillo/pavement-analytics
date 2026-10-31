"""Private full-run integrity and saved-prediction evidence; no promotion claims."""
from __future__ import annotations

from collections import Counter
from html import escape
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .autorun import external_path
from .io import read_json, sha256, write_json
from .model_registry import ID_PATTERN, REGISTRY_VERSION, environment_versions
from .model_runtime import ModelRuntime, model_options, runtime_context
from .pipeline import code_digest
from .security import Boundary, SafeError, disjoint, reject_links, source_root


CHECK_COLUMNS = ["material", "target", "candidate", "feature_set", "rows", "groups", "max_absolute_error", "status"]


def _read(path, **kwargs):
    reject_links(Path(path))
    return pd.read_csv(path, keep_default_na=False, **kwargs)


def verify_artifacts(pipeline, report):
    boundary = Boundary(source_root().resolve(), Path(pipeline).resolve())
    verified = {}
    for stage in report.get("stages", {}).values():
        if stage.get("status") not in {"completed", "skipped"}:
            continue
        for item in stage.get("artifacts", []):
            path = boundary.path(item["path"])
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise SafeError("QUALIFICATION_ARTIFACT_CHANGED")
            verified[path] = item["sha256"]
    if not verified:
        raise SafeError("QUALIFICATION_NO_VERIFIED_ARTIFACTS")
    return verified


def _identity(frame, source, plan, target_column, partition, expected_fold=False):
    required = {"row_id", "group_id", "actual", "weight", "prediction"}
    if expected_fold:
        required.add("fold")
    if not required <= set(frame):
        raise SafeError("QUALIFICATION_PREDICTION_SCHEMA_INVALID")
    ids = pd.to_numeric(frame.row_id, errors="coerce")
    if ids.isna().any() or (ids % 1 != 0).any() or not set(ids.astype(int)) <= set(source.index):
        raise SafeError("QUALIFICATION_ROW_ID_MISMATCH")
    selected = source.loc[ids.astype(int)]
    groups = frame.group_id.astype(str).to_numpy()
    if not np.array_equal(groups, selected.partition_airfield.astype(str).to_numpy()):
        raise SafeError("QUALIFICATION_GROUP_ID_MISMATCH")
    assignments = plan.set_index("group_id")
    if not assignments.loc[groups, "partition"].eq(partition).all():
        raise SafeError("QUALIFICATION_PARTITION_LEAKAGE")
    if expected_fold and not np.array_equal(pd.to_numeric(frame.fold).to_numpy(), assignments.loc[groups, "fold"].to_numpy()):
        raise SafeError("QUALIFICATION_FOLD_MISMATCH")
    actual = pd.to_numeric(selected[target_column], errors="coerce").to_numpy(float)
    weights = pd.to_numeric(selected.delta_days, errors="coerce").to_numpy(float)
    if not np.isfinite(actual).all() or (actual < 0).any() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise SafeError("QUALIFICATION_TARGET_OR_WEIGHT_INVALID")
    if not np.allclose(actual, frame.actual.to_numpy(float), rtol=1e-12, atol=1e-12):
        raise SafeError("QUALIFICATION_TARGET_MISMATCH")
    if not np.array_equal(weights, frame.weight.to_numpy(float)):
        raise SafeError("QUALIFICATION_WEIGHT_MISMATCH")
    points = frame.prediction.to_numpy(float)
    if not np.isfinite(points).all() or (points < 0).any():
        raise SafeError("QUALIFICATION_INVALID_PREDICTIONS")
    return selected


def _prediction_check(runtime, artifact, saved, source, plan, target_column, expected_features, target, material, candidate, feature_set):
    selected = _identity(saved, source, plan, target_column, "holdout")
    if saved.row_id.duplicated().any():
        raise SafeError("QUALIFICATION_DUPLICATE_PREDICTIONS")
    numeric_target = pd.to_numeric(source[target_column], errors="coerce")
    valid = np.isfinite(numeric_target) & source.partition_airfield.map(plan.set_index("group_id").partition).eq("holdout")
    if set(saved.row_id) != set(source.index[valid]):
        raise SafeError("QUALIFICATION_HOLDOUT_COVERAGE_MISMATCH")
    if artifact.get("fit_partition") != "development" or artifact.get("target") != target:
        raise SafeError("QUALIFICATION_MODEL_PARTITION_OR_TARGET_MISMATCH")
    if artifact.get("features") != expected_features:
        raise SafeError("QUALIFICATION_FEATURE_MISMATCH")
    from .full_modeling import _feature_guard
    for name in expected_features:
        _feature_guard(name, [target_column])
    development = set(source.loc[np.isfinite(numeric_target) & source.partition_airfield.map(
        plan.set_index("group_id").partition).eq("development"), "partition_airfield"])
    details = artifact.get("training_details")
    if details and (set(details.get("refit_groups", [])) != development or
                    not set(details.get("stopping_groups", [])) <= development):
        raise SafeError("QUALIFICATION_TRAINING_GROUP_MISMATCH")
    runtime_context(candidate=candidate, feature_set=feature_set, phase="qualification", fold=-1)
    predicted, _ = runtime.call("predict", artifact["estimator"], selected[expected_features])
    if not np.allclose(predicted, saved.prediction.to_numpy(float), rtol=1e-12, atol=1e-12):
        raise SafeError("QUALIFICATION_SAVED_PREDICTION_MISMATCH")
    return {"material": material, "target": target, "candidate": candidate, "feature_set": feature_set,
            "rows": len(saved), "groups": int(saved.group_id.nunique()),
            "max_absolute_error": float(np.max(abs(predicted-saved.prediction.to_numpy(float)))) if len(saved) else None,
            "status": "passed"}


def _render(out, report, checks):
    rows = ''.join('<tr>'+''.join('<td>'+escape(str(row.get(c, '')))+' </td>' for c in CHECK_COLUMNS)+'</tr>' for row in checks)
    gates = ''.join('<li>'+escape(name)+'</li>' for name in report["remaining_release_gates"])
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Run qualification</title>'
        '<style>body{font:16px system-ui;max-width:1150px;margin:28px auto;padding:16px;color:#18314c}'
        'table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccd8e4}'
        '.notice{background:#fff4dc;padding:16px}.scroll{overflow:auto}</style></head><body>'
        '<h1>Full-run qualification evidence</h1><p>Technical audit completed. Saved models, outcomes, weights and shared groups were checked.</p>'
        '<p class="notice">This report records evidence for release review. Synthetic checks and score gains do not establish operational accuracy or climate causation.</p>'
        '<p>Platform: '+escape(report["platform"]["system"]+' / '+report["platform"]["machine"])+
        '. Model prediction checks: '+str(len(checks))+'. Verified artifacts: '+str(report["verified_artifacts"])+'.</p>'
        '<h2>Remaining release evidence</h2><ul>'+gates+'</ul>'
        '<p>Live ERA5 means an observed successful download from a completed worker. Cached data remain separately identified.</p>'
        '<p><a href="qualification.json">Complete qualification JSON</a> · <a href="prediction_checks.csv">Prediction checks CSV</a></p>'
        '<h2>Saved prediction checks</h2><div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(c)+'</th>' for c in CHECK_COLUMNS)+
        '</tr></thead><tbody>'+rows+'</tbody></table></div></body></html>')
    (out/"qualification.html").write_text(html, encoding="utf-8")


def audit_full_run(pipeline, report, cfg, setup, out, *, progress=None, private_pilot=False):
    """Recheck a verified run; leave scientific/private release review pending."""
    pipeline, out = Path(pipeline), Path(out)
    reject_links(pipeline); reject_links(out)
    if out.exists():
        raise SafeError("QUALIFICATION_OUTPUT_EXISTS")
    Boundary.create(source_root(), out.parent)
    out.mkdir()
    stages = ["01_split", "07_partitions", "08_model_backend", "09_local_models", "10_report"]
    stages += [f"{prefix}_{surface}" for surface in cfg["surfaces"] for prefix in
               ("02_pairs", "03_location_rate", "04_climate", "05_targets", "06_pci", "07_modeling")]
    if any(report.get("stages", {}).get(name, {}).get("status") != "completed" for name in stages):
        raise SafeError("QUALIFICATION_PIPELINE_INCOMPLETE")
    verified = verify_artifacts(pipeline, report)
    if not setup.get("checksums"):
        raise SafeError("QUALIFICATION_INPUT_EVIDENCE_MISSING")
    for path, digest in setup.get("checksums", {}).items():
        actual = external_path(Path(path), source_root())
        if not actual.is_file() or sha256(actual) != digest:
            raise SafeError("QUALIFICATION_INPUT_CHANGED")
    if environment_versions() != report.get("model_environment"):
        raise SafeError("QUALIFICATION_ENVIRONMENT_CHANGED")
    recorded = report.get("execution_environment", {})
    if recorded.get("code_sha256") != code_digest():
        raise SafeError("QUALIFICATION_CODE_EVIDENCE_MISSING_OR_CHANGED")
    from .original.pipeline import discover
    if discover(cfg)[3] != report.get("fingerprint"):
        raise SafeError("QUALIFICATION_FINGERPRINT_CHANGED")
    models = pipeline/"09_local_models"
    plan = _read(models/"partitions.csv", dtype={"group_id": str})
    if plan.group_id.duplicated().any():
        raise SafeError("QUALIFICATION_PARTITION_PLAN_INVALID")
    mapping = _read(models/"row_map.csv")
    saved = read_json(models/"sweep_summary.json")
    target_map = _read(models/"target_map.csv")
    checks, task_support, oof_counts = [], [], Counter()
    with ModelRuntime(out, model_options({**cfg.get("local_model_options", {}), "comparisons": False}), progress) as runtime:
        for surface in cfg["surfaces"]:
            source = _read(pipeline/"07_modeling/partitioned"/(surface+".csv"), dtype=str)
            rows = mapping.loc[mapping.material.eq(surface)]
            positions = pd.to_numeric(rows.model_csv_row, errors="raise").astype(int)-2
            if len(rows) != len(source) or set(positions) != set(range(len(source))) or rows.row_id.duplicated().any():
                raise SafeError("QUALIFICATION_ROW_MAP_INVALID")
            source.index = rows.set_index(positions).loc[range(len(source)), "row_id"].to_numpy(int)
            for summary in [s for s in saved["summaries"] if s["material"] == surface]:
                target = summary["target"]
                if not ID_PATTERN.fullmatch(target):
                    raise SafeError("QUALIFICATION_TARGET_INVALID")
                root = models/"materials"/surface/target
                matches = target_map.loc[target_map.material.eq(surface) & target_map.target.eq(target)]
                if len(matches) != 1:
                    raise SafeError("QUALIFICATION_TARGET_MAP_INVALID")
                column = matches.original_target.iloc[0]
                task_support.append({"material": surface, **{k: summary.get(k) for k in ("target", "status", "valid_target_rows",
                    "valid_target_groups", "holdout_rows", "holdout_groups", "nonzero_fraction", "mae_skill", "importance_reliable")}})
                if not (root/"model.joblib").is_file():
                    continue
                declared = read_json(models/"materials"/surface/"features.json")
                sets = {"all": declared["features"], "nonclimate": [c for c in declared["features"] if c not in declared["climate_features"]]}
                predictions_path = root/"candidate_predictions.csv"
                if predictions_path.is_file():
                    predictions = _read(predictions_path, dtype={"group_id": str})
                    comparisons = _read(root/"candidate_comparisons.csv")
                    candidates = [(r.candidate, r.feature_set, r.model_path) for r in comparisons.itertuples() if r.comparison_status == "completed"]
                else:
                    predictions = _read(root/"holdout_predictions.csv", dtype={"group_id": str})
                    candidates = [(summary["candidate"], "all", "model.joblib"),
                                  (summary["nonclimate_candidate"], "nonclimate", "model.joblib")]
                for candidate, feature_set, relative in candidates:
                    if not ID_PATTERN.fullmatch(candidate) or feature_set not in sets:
                        raise SafeError("QUALIFICATION_CANDIDATE_INVALID")
                    path = Boundary(source_root().resolve(), root.resolve()).path(relative)
                    if path not in verified:
                        raise SafeError("QUALIFICATION_MODEL_NOT_CHECKPOINTED")
                    artifact = joblib.load(path)
                    if relative == "model.joblib" and feature_set == "nonclimate":
                        artifact = {**artifact, "candidate": artifact["nonclimate_candidate"],
                            "features": artifact["nonclimate_features"], "estimator": artifact["nonclimate_estimator"],
                            "training_details": artifact.get("nonclimate_training_details")}
                    if artifact.get("candidate") != candidate:
                        raise SafeError("QUALIFICATION_CANDIDATE_MISMATCH")
                    subset = predictions.loc[predictions.candidate.eq(candidate) & predictions.feature_set.eq(feature_set)] if "candidate" in predictions else predictions.copy()
                    if "candidate" not in predictions and feature_set == "nonclimate":
                        subset["prediction"] = subset.nonclimate_prediction
                    checks.append(_prediction_check(runtime, artifact, subset, source, plan, column, sets[feature_set],
                                                    target, surface, candidate, feature_set))
                oof_path = root/"oof_predictions.csv"
                if oof_path.is_file():
                    for chunk in pd.read_csv(oof_path, keep_default_na=False, dtype={"group_id": str}, chunksize=20000):
                        _identity(chunk, source, plan, column, "development", expected_fold=True)
                        oof_counts[surface] += len(chunk)
    climate = {}
    for surface in cfg["surfaces"]:
        stats = report["stages"]["04_climate_"+surface]["stats"]
        receipt_path = pipeline/"04_climate"/(surface+"_acquisition.json")
        observed = 0
        if receipt_path.is_file():
            if receipt_path not in verified:
                raise SafeError("QUALIFICATION_CLIMATE_NOT_CHECKPOINTED")
            receipt = read_json(receipt_path)
            observed = sum(int(worker.get("successful_downloads", 0)) for worker in receipt.get("workers", [])
                if worker.get("complete") is True and worker.get("returncode") == 0)
            if observed < 0 or observed != receipt.get("successful_downloads"):
                raise SafeError("QUALIFICATION_CLIMATE_EVIDENCE_MISMATCH")
        if observed != stats.get("successful_downloads", 0):
            raise SafeError("QUALIFICATION_CLIMATE_EVIDENCE_MISMATCH")
        climate[surface] = dict(stats, successful_downloads=observed)
    downloads = sum(int(item.get("successful_downloads", 0)) for item in climate.values())
    facts = {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()}
    remaining = ["private_pilot_review"]
    if facts["system"] != "Darwin" or recorded.get("system") != "Darwin" or recorded.get("machine") != facts["machine"]:
        remaining.append("macos_deployment")
    if downloads < 1: remaining.append("observed_live_era5_access")
    if not private_pilot: remaining.append("private_data_declaration")
    if not checks: remaining.append("supported_model_tasks")
    if not cfg.get("local_model_options", {}).get("comparisons", True): remaining.append("all_candidate_comparisons")
    result = {"schema_version": 1, "status": "technical_checks_passed", "release_gate": "pending_private_review",
        "evidence_kind": "declared_private_pilot" if private_pilot else "full_pipeline_run",
        "package_version": __version__, "registry_version": REGISTRY_VERSION, "platform": facts, "fitting_platform": recorded,
        "environment": environment_versions(), "code_sha256": code_digest(), "verified_artifacts": len(verified),
        "prediction_checks": len(checks), "oof_rows_checked": dict(oof_counts), "climate": climate,
        "observed_successful_downloads": downloads, "task_support": task_support,
        "remaining_release_gates": remaining,
        "note": "Technical evidence does not establish operational accuracy, independent real-data acceptance or climate causation."}
    pd.DataFrame(checks, columns=CHECK_COLUMNS).to_csv(out/"prediction_checks.csv", index=False)
    write_json(out/"qualification.json", result)
    _render(out, result, checks)
    return result, sorted(p for p in out.iterdir() if p.is_file())


def qualify_saved_run(workspace, out, *, private_pilot=False, progress=None):
    workspace = external_path(Path(workspace), source_root())
    reject_links(Path(out))
    out = Path(out).expanduser().resolve()
    if not disjoint(workspace, out) or not disjoint(workspace, source_root()):
        raise SafeError("QUALIFICATION_OUTPUT_MUST_BE_SEPARATE")
    boundary = Boundary(source_root().resolve(), workspace)
    result = read_json(boundary.path("full_result.json"))
    pipeline = boundary.path(read_json(boundary.path("full_config.json"))["output_dir"])
    report = read_json(pipeline/"run_report.json")
    if result.get("status") != "completed" or report.get("status") not in {"completed", "completed_with_warnings"}:
        raise SafeError("QUALIFICATION_PIPELINE_INCOMPLETE")
    return audit_full_run(pipeline, report, read_json(boundary.path("full_config.json")),
                          read_json(boundary.path("full_setup_report.json")), out,
                          progress=progress, private_pilot=private_pilot)
