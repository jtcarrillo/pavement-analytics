"""Reproducible offline release matrix, separate from deployment/pilot evidence."""
from __future__ import annotations

import gc
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .io import sha256, write_json
from .model_registry import candidate_catalog, environment_versions, REGISTRY_VERSION
from .pipeline import code_digest
from .security import Boundary, SafeError, reject_links, source_root
from .study_modeling import search_levels
from .study_reporting import build_study_report
from .synthetic_benchmark import (DEFAULT_SEED, FEATURES, CLIMATE_FEATURES, TARGETS, MATERIALS,
    PROFILES, create_bundle, validate_bundle)


EDGE_STATUSES = {"level_all_missing": "no_valid_target", "level_development_constant": "constant_target",
    "level_positive_only_holdout": "constant_target", "level_no_holdout_labels": "insufficient_groups",
    "level_one_positive_group": None}


def _usage():
    try:
        import resource
    except ImportError:
        return {"cpu": time.process_time(), "peak_sum_mib": None}
    parent, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return {"cpu": time.process_time()+child.ru_utime+child.ru_stime,
            "peak_sum_mib": (parent.ru_maxrss+child.ru_maxrss)/divisor}


def _parity(root, data, plan):
    comparisons = pd.read_csv(root/"candidate_comparisons.csv", keep_default_na=False)
    if comparisons.empty:
        return {"candidate_models_checked": 0, "family_rows": 0}
    bad = comparisons.comparison_status.astype(str).str.startswith(("failed_", "timed_out"))
    if bad.any():
        raise SafeError("RELEASE_CANDIDATE_FAILURE")
    predictions = pd.read_csv(root/"candidate_predictions.csv", dtype={"group_id": str}, keep_default_na=False)
    indexed = data.set_index("row_id")
    development = set(plan.loc[plan.partition.eq("development"), "group_id"])
    count = 0
    for row in comparisons.loc[comparisons.comparison_status.eq("completed")].itertuples():
        artifact = joblib.load(root/row.model_path)
        saved = predictions.loc[predictions.candidate.eq(row.candidate) & predictions.feature_set.eq(row.feature_set)]
        X = indexed.loc[saved.row_id, artifact["features"]]
        actual = artifact["estimator"].predict(X)
        if (not np.isfinite(actual).all() or (actual < 0).any() or not np.allclose(actual, saved.prediction, rtol=1e-12, atol=1e-12)
                or artifact["fit_partition"] != "development" or set(saved.group_id)&development
                or not np.array_equal(saved.weight, indexed.loc[saved.row_id, "delta_days"])):
            raise SafeError("RELEASE_SAVED_MODEL_PARITY_FAILED")
        count += 1
    families = pd.read_csv(root/"family_summary.csv", keep_default_na=False)
    for row in families.loc[families.comparison_status.eq("completed")].itertuples():
        choices = comparisons.loc[comparisons.family.eq(row.family) & comparisons.feature_set.eq(row.feature_set)
                                  & comparisons.comparison_status.eq("completed")]
        if row.candidate != choices.sort_values(["cv_mae", "candidate"]).candidate.iloc[0]:
            raise SafeError("RELEASE_FAMILY_SELECTION_MISMATCH")
    importance = pd.read_csv(root/"family_importance.csv", keep_default_na=False)
    if "reliable" in importance or (not importance.empty and not importance.repeats.eq(5).all()):
        raise SafeError("RELEASE_IMPORTANCE_SCHEMA_INVALID")
    return {"candidate_models_checked": count, "family_rows": len(families)}


def release_fixture(out, *, profile="smoke", candidate_ids=None, progress=None):
    """Exercise every case; generated data and outcomes stay outside source."""
    out = Path(out)
    reject_links(out)
    if profile not in PROFILES:
        raise SafeError("INVALID_RELEASE_PROFILE")
    if out.exists():
        raise SafeError("QUALIFICATION_OUTPUT_EXISTS")
    Boundary.create(source_root(), out.parent)
    out.mkdir()
    bundle = create_bundle(out/"fixture", profile=profile, seed=DEFAULT_SEED)
    validate_bundle(bundle)
    table = pd.read_csv(bundle/"model_table.csv", dtype={"group_id": str})
    stress = pd.read_csv(bundle/"stress_model_table.csv", dtype={"group_id": str})
    edges = pd.read_csv(bundle/"controls/edge_cases.csv", dtype={"group_id": str})
    plan = pd.read_csv(bundle/"partitions.csv", dtype={"group_id": str})
    cases, summaries = [], []
    options = {"candidate_ids": candidate_ids}
    report = {"schema_version": 1, "status": "running", "evidence_kind": "synthetic_model_stage",
        "package_version": __version__, "registry_version": REGISTRY_VERSION, "seed": DEFAULT_SEED,
        "profile": profile, "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
        "environment": environment_versions(), "code_sha256": code_digest(), "catalog": candidate_catalog(DEFAULT_SEED),
        "budgets": {"wall_seconds": 300, "cpu_seconds": 300, "peak_sum_mib": 1024}, "cases": cases,
        "remaining_release_gates": ["deployment_macos_private_run", "observed_live_era5_access", "private_pilot_review"]}
    write_json(out/"release_report.json", report)
    try:
        for material in MATERIALS:
            clean = table.loc[table.material.eq(material)]
            population = [(target, clean, FEATURES, None, "patterns") for target in TARGETS]
            population += [(target, edges.loc[edges.material.eq(material)], FEATURES, expected, "edges") for target, expected in EDGE_STATUSES.items()]
            population += [("level_zero_inflated", stress.loc[stress.material.eq(material)], [*FEATURES, "all_missing_predictor"], None, "stress")]
            for target, data, features, expected, kind in population:
                context = {"event": "target", "material": material, "target": target}
                if progress: progress({**context, "status": "started"})
                path = out/"models"/kind/"materials"/material/target
                started, before = time.perf_counter(), _usage()
                summary = search_levels(data, features, target, path, plan, CLIMATE_FEATURES,
                    seed=DEFAULT_SEED, weight_column="delta_days", model_options=options, progress=progress)
                if expected is not None and summary["status"] != expected:
                    raise SafeError("RELEASE_EDGE_STATUS_MISMATCH")
                if summary["candidate_failure_count"] or summary.get("comparison_failure_count", 0):
                    raise SafeError("RELEASE_CANDIDATE_FAILURE")
                evidence = _parity(path, data, plan)
                after = _usage()
                case = {"material": material, "target": target, "case": kind, "status": summary["status"],
                    "wall_seconds": time.perf_counter()-started, "cpu_seconds": after["cpu"]-before["cpu"],
                    "peak_sum_mib": after["peak_sum_mib"], **evidence}
                if case["wall_seconds"] > 300 or case["cpu_seconds"] > 300 or (case["peak_sum_mib"] is not None and case["peak_sum_mib"] > 1024):
                    raise SafeError("RELEASE_RESOURCE_BUDGET_EXCEEDED")
                cases.append(case)
                if kind == "patterns": summaries.append({"material": material, **summary})
                write_json(out/"release_report.json", report)
                if progress: progress({**context, "status": "completed", "result_status": summary["status"]})
                gc.collect()
        guards = []
        for name, data, features, target in [
            ("invalid_weights", pd.read_csv(bundle/"controls/invalid_weights.csv"), FEATURES, "level_linear"),
            ("duplicate_ids", pd.read_csv(bundle/"controls/duplicate_row_ids.csv"), FEATURES, "level_linear"),
            ("target_as_feature", table, [*FEATURES, "level_linear"], "level_linear"),
            ("identity_as_feature", table, [*FEATURES, "group_id"], "level_linear")]:
            try:
                search_levels(data, features, target, out/"guards"/name, plan, CLIMATE_FEATURES,
                              seed=DEFAULT_SEED, weight_column="delta_days", candidate_ids=candidate_ids)
            except SafeError as error:
                guards.append({"case": name, "status": "rejected", "error_code": str(error)})
            else:
                raise SafeError("RELEASE_UNSAFE_INPUT_ACCEPTED")
        report_out = out/"report"; report_out.mkdir()
        build_study_report(report_out, summaries, {}, model_root=out/"models/patterns")
        report.update(status="software_checks_passed", guards=guards,
            candidate_models_checked=sum(c["candidate_models_checked"] for c in cases),
            note="Invented model-stage data verifies software, not private pavement accuracy or live weather acquisition.")
    except BaseException as error:
        report.update(status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_code=str(error) if isinstance(error, SafeError) else "RELEASE_CHECK_FAILED")
        write_json(out/"release_report.json", report)
        raise
    pd.DataFrame(cases).to_csv(out/"release_cases.csv", index=False)
    write_json(out/"release_report.json", report)
    (out/"qualification.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Release checks</title>'
        '<body style="font:16px system-ui;margin:32px"><h1>Offline release checks passed</h1><p>'+
        str(len(cases))+' cases; '+str(report["candidate_models_checked"])+
        ' saved-model prediction checks.</p><p>These are synthetic model-stage checks. Deployment Mac, live ERA5 and private pilot evidence remain separate.</p>'
        '<p><a href="report/dashboard.html">Explore target and model-family results</a></p>'
        '<p><a href="release_report.json">Release JSON</a> · <a href="release_cases.csv">Case metrics CSV</a></p></body></html>', encoding="utf-8")
    manifest = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out/"qualification_manifest.json", {"schema_version": 1, "outputs": manifest, "code_sha256": code_digest()})
    return report
