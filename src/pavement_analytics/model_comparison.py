"""Development-selected family representatives and held-out model comparisons."""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
import pandas as pd

from .io import write_json
from .model_registry import REGISTRY_VERSION, ID_PATTERN, build_candidates, environment_versions
from .model_runtime import ACTIVE, ModelTimeout, runtime_context
from .security import SafeError


COMPARISON_COLUMNS = ["candidate", "family", "feature_set", "candidate_status", "comparison_status", "error_code",
    "parameters", "cv_mae", "cv_mae_std", "cv_rmse", "cv_mae_skill", "folds_completed", "feature_count",
    "development_rows", "development_groups", "holdout_rows", "holdout_groups", "cv_fit_wall_seconds",
    "refit_wall_seconds", "prediction_wall_seconds", "iterations", "holdout_mae", "holdout_rmse", "holdout_r2",
    "baseline_mae", "mae_skill", "primary_selected", "family_selected", "prediction_parity", "model_path", "training",
    "importance_status", "importance_error_code", "refit_source"]
FAMILY_COLUMNS = [*COMPARISON_COLUMNS, "nonclimate_candidate", "nonclimate_mae", "climate_added_skill"]
PREDICTION_COLUMNS = ["candidate", "family", "feature_set", "row_id", "group_id", "actual", "prediction", "weight"]
IMPORTANCE_COLUMNS = ["candidate", "family", "feature_set", "feature", "kind", "columns", "is_climate", "metric",
    "importance_mean", "importance_std", "repeats", "heuristic_support", "reliability_rule", "baseline_loss", "reference_loss"]


def empty_comparisons(out):
    for name, columns in (("candidate_comparisons.csv", COMPARISON_COLUMNS), ("family_summary.csv", FAMILY_COLUMNS),
                          ("candidate_predictions.csv", PREDICTION_COLUMNS), ("family_importance.csv", IMPORTANCE_COLUMNS)):
        pd.DataFrame(columns=columns).to_csv(Path(out)/name, index=False)


def correlated_blocks(development, features, threshold=.8):
    """Training-only absolute Spearman components; never inspect holdout labels."""
    numeric = development[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    correlation = numeric.corr(method="spearman", min_periods=3).abs()
    unseen, result = set(features), []
    for first in sorted(features):
        if first not in unseen:
            continue
        component, pending = [], [first]
        unseen.remove(first)
        while pending:
            current = pending.pop()
            component.append(current)
            neighbours = {name for name in unseen if correlation.loc[current, name] >= threshold}
            unseen -= neighbours
            pending.extend(sorted(neighbours))
        if len(component) > 1:
            result.append(sorted(component))
    return result


def _importance(model, development, heldout, columns, climate, target, weights, seed, row, baseline_prediction):
    from .study_modeling import _prediction_bundle, _score
    predicted, components = _prediction_bundle(model, heldout[columns])
    y = heldout[target].to_numpy(float)
    reference = _score(y, predicted, weights)["mae"]
    baseline = _score(y, baseline_prediction, weights)["mae"]
    reliable = (row["candidate"] != "median" and row["cv_mae_skill"] is not None and row["cv_mae_skill"] > 0
                and row["mae_skill"] is not None and row["mae_skill"] > 0 and row["holdout_r2"] is not None
                and row["holdout_r2"] > 0 and heldout.group_id.nunique() >= 2)
    losses = {"mae": (reference, baseline, reliable)}
    positive = y > 0
    if components is not None:
        brier = float(np.average((positive-components["p_positive"])**2, weights=weights))
        prior = float(np.average((positive-components["baseline_probability"])**2, weights=weights))
        losses["occurrence_brier"] = (brier, prior, bool(positive.any() and (~positive).any() and brier < prior
                                                        and heldout.group_id.nunique() >= 2))
        if positive.any():
            severity = float(np.average(abs(y[positive]-components["positive_mean"][positive]), weights=weights[positive]))
            prior = float(np.average(abs(y[positive]-components["baseline_positive_mean"][positive]), weights=weights[positive]))
            losses["positive_severity_mae"] = (severity, prior, bool(severity < prior and
                                                        heldout.loc[positive, "group_id"].nunique() >= 2))
    blocks = [(name, "feature", [name]) for name in columns]
    climate = [x for x in climate if x in columns]
    if climate:
        blocks.append(("All climate features", "climate_block", climate))
    for i, block in enumerate(correlated_blocks(development, columns), 1):
        blocks.append((f"Correlated block {i}", "correlated_block", block))
    rng = np.random.default_rng(seed+1701)
    orders = [rng.permutation(len(heldout)) for _ in range(5)]
    result = []
    for feature, kind, block in blocks:
        changes = {metric: [] for metric in losses}
        for order in orders:
            perturbed = heldout[columns].copy()
            perturbed.loc[:, block] = heldout.iloc[order][block].to_numpy()
            prediction, components = _prediction_bundle(model, perturbed)
            changes["mae"].append(_score(y, prediction, weights)["mae"]-reference)
            if components is not None:
                changes["occurrence_brier"].append(float(np.average((positive-components["p_positive"])**2,
                                                                                   weights=weights))-losses["occurrence_brier"][0])
                if positive.any():
                    changes["positive_severity_mae"].append(float(np.average(abs(y[positive]-components["positive_mean"][positive]),
                                                                         weights=weights[positive]))-losses["positive_severity_mae"][0])
        for metric, delta in changes.items():
            reference_loss, baseline_loss, supported = losses[metric]
            result.append({"candidate": row["candidate"], "family": row["family"], "feature_set": row["feature_set"],
                "feature": feature, "kind": kind, "columns": json.dumps(block), "is_climate": bool(set(block)&set(climate)),
                "metric": metric, "importance_mean": float(np.mean(delta)), "importance_std": float(np.std(delta)),
                "repeats": 5, "heuristic_support": supported, "baseline_loss": baseline_loss, "reference_loss": reference_loss,
                "reliability_rule": "heuristic_support_only; exploratory predictive reliance; repeat_SD_not_confidence"})
    return result


def build_comparisons(out, development, heldout, features, climate, target, seed, weight_column, requested):
    from .study_modeling import MedianRegressor, _fit, _prediction_bundle, _weights, _score, candidates
    out = Path(out)
    leaders = pd.read_csv(out/"leaderboard.csv", keep_default_na=False)
    primary = joblib.load(out/"model.joblib")
    pool = candidates(seed) if requested is None else build_candidates(seed, requested)
    pool = {"median": MedianRegressor(), **pool}
    sets = {"all": features, "nonclimate": [x for x in features if x not in climate]}
    weights, train_weights = _weights(heldout, weight_column), _weights(development, weight_column)
    y = heldout[target].to_numpy(float)
    baseline_prediction = primary["baseline_estimator"].predict(heldout[features])
    baseline_mae = _score(y, baseline_prediction, weights)["mae"]
    rows, predictions = [], []
    (out/"candidate_models").mkdir()
    runtime = ACTIVE.get()
    for item in leaders.to_dict("records"):
        name, feature_set, family = item["candidate"], item["feature_set"], item["family"]
        if not ID_PATTERN.fullmatch(name) or feature_set not in sets:
            raise SafeError("INVALID_COMPARISON_CANDIDATE")
        columns = sets[feature_set]
        baseline_cv = float(leaders.loc[leaders.candidate.eq("median") & leaders.feature_set.eq(feature_set), "cv_mae"].iloc[0])
        cv = float(item["cv_mae"]) if item["cv_mae"] != "" else None
        row = {**item, "comparison_status": item["candidate_status"], "family_selected": False,
            "primary_selected": item["selected"], "feature_count": len(columns), "development_rows": len(development),
            "development_groups": int(development.group_id.nunique()), "holdout_rows": len(heldout),
            "holdout_groups": int(heldout.group_id.nunique()), "baseline_mae": baseline_mae,
            "cv_mae_skill": 1-cv/baseline_cv if cv is not None and baseline_cv > 0 else None,
            "cv_fit_wall_seconds": sum(r["wall_seconds"] for r in runtime.records if r.get("phase") == "cv"
                and r.get("candidate") == name and r.get("feature_set") == feature_set and r["operation"] == "fit") if runtime else None}
        rows.append(row)
        if item["candidate_status"] != "completed" or name not in pool:
            continue
        runtime_context(candidate=name, family=family, feature_set=feature_set, phase="comparison_refit", fold=-1)
        start = time.perf_counter()
        refit_source = "comparison"
        try:
            if name == primary["candidate" if feature_set == "all" else "nonclimate_candidate"]:
                model = primary["estimator" if feature_set == "all" else "nonclimate_estimator"]
                refit_source = "primary_refit"
            elif family == "two_stage" and (out/("two_stage_"+feature_set+"_model.joblib")).exists():
                artifact = joblib.load(out/("two_stage_"+feature_set+"_model.joblib"))
                if artifact["candidate"] == name:
                    refit_source = "two_stage_refit"
                model = artifact["estimator"] if artifact["candidate"] == name else _fit(
                    pool[name], development[columns], development[target].to_numpy(float), train_weights, development.group_id)
            else:
                model = _fit(pool[name], development[columns], development[target].to_numpy(float), train_weights, development.group_id)
            row["refit_wall_seconds"] = time.perf_counter()-start
            row["refit_source"] = refit_source
            if runtime and refit_source != "comparison":
                row["refit_wall_seconds"] = sum(r["wall_seconds"] for r in runtime.records if r.get("phase") == refit_source
                    and r.get("candidate") == name and r.get("feature_set") == feature_set and r["operation"] == "fit")
            start = time.perf_counter()
            predicted, _ = _prediction_bundle(model, heldout[columns])
            row["prediction_wall_seconds"] = time.perf_counter()-start
        except Exception as error:
            row.update(comparison_status="timed_out" if isinstance(error, ModelTimeout) else "failed_refit_or_predict",
                       error_code="MODEL_OPERATION_TIMED_OUT" if isinstance(error, ModelTimeout) else "COMPARISON_REFIT_OR_PREDICTION_FAILED")
            with (out/"comparison_errors.log").open("a") as log:
                log.write(f"{name} / {feature_set}\n"+traceback.format_exc())
            continue
        relative = f"candidate_models/{feature_set}__{name}.joblib"
        artifact = {"format_version": 1, "candidate_registry_version": REGISTRY_VERSION,
            "candidate": name, "candidate_family": family, "candidate_parameters": json.loads(item["parameters"]),
            "target": target, "features": columns, "climate_features": [x for x in climate if x in columns],
            "estimator": model, "fit_partition": "development", "weight_column": weight_column,
            "weighting": "equal_total_per_group" if weight_column is None else "interval_days",
            "selection_metric": "mean_development_cv_MAE", "environment": environment_versions(),
            "training_details": getattr(model, "training_details_", None)}
        joblib.dump(artifact, out/relative)
        # Verify the exact artifact that downstream importance and report
        # readers use. Filesystem/serialization failures abort the target.
        restored = joblib.load(out/relative)
        parity, _ = _prediction_bundle(restored["estimator"], heldout[columns])
        np.testing.assert_allclose(parity, predicted, rtol=1e-12, atol=1e-12)
        measured = _score(y, predicted, weights)
        details = getattr(model, "training_details_", None)
        row.update(comparison_status="completed", prediction_parity=True, model_path=relative,
            holdout_mae=measured["mae"], holdout_rmse=measured["rmse"], holdout_r2=measured["r2"],
            mae_skill=1-measured["mae"]/baseline_mae if baseline_mae > 0 else None,
            iterations=details.get("iterations") if details else None, training=json.dumps(details, sort_keys=True))
        for row_id, group_id, actual, point, weight in zip(heldout.row_id, heldout.group_id, y, predicted, weights):
            predictions.append({"candidate": name, "family": family, "feature_set": feature_set, "row_id": row_id,
                                "group_id": group_id, "actual": actual, "prediction": point, "weight": weight})
    families, importance = [], []
    for (feature_set, family), block in pd.DataFrame(rows).groupby(["feature_set", "family"], sort=True):
        eligible = [r for r in rows if r["feature_set"] == feature_set and r["family"] == family and r["comparison_status"] == "completed"]
        if not eligible:
            families.append({"family": family, "feature_set": feature_set, "comparison_status": "no_completed_candidate",
                             "candidate_status": str(block.candidate_status.iloc[0]), "error_code": str(block.error_code.iloc[0])})
            continue
        selected = min(eligible, key=lambda r: (float(r["cv_mae"]), r["candidate"]))
        selected["family_selected"] = True
        family_row = dict(selected)
        families.append(family_row)
        if selected["candidate"] == "median":
            selected["importance_status"] = family_row["importance_status"] = "baseline_not_explained"
            continue
        artifact = joblib.load(out/selected["model_path"])
        runtime_context(candidate=selected["candidate"], family=family, feature_set=feature_set, phase="family_importance", fold=-1)
        try:
            values = _importance(artifact["estimator"], development, heldout, sets[feature_set], climate, target,
                                 weights, seed, selected, baseline_prediction)
        except Exception as error:
            selected["importance_status"] = family_row["importance_status"] = "timed_out" if isinstance(error, ModelTimeout) else "failed"
            selected["importance_error_code"] = family_row["importance_error_code"] = "FAMILY_IMPORTANCE_FAILED"
            with (out/"comparison_errors.log").open("a") as log:
                log.write(f"{selected['candidate']} / {feature_set} / importance\n"+traceback.format_exc())
            continue
        importance.extend(values)
        selected["importance_status"] = family_row["importance_status"] = "completed"
    for row in families:
        if row["feature_set"] != "all" or row.get("comparison_status") != "completed":
            continue
        nonclimate = next((r for r in families if r["family"] == row["family"] and r["feature_set"] == "nonclimate"
                          and r.get("comparison_status") == "completed"), None)
        if nonclimate:
            row.update(nonclimate_candidate=nonclimate["candidate"], nonclimate_mae=nonclimate["holdout_mae"],
                       climate_added_skill=1-row["holdout_mae"]/nonclimate["holdout_mae"] if nonclimate["holdout_mae"] > 0 else None)
    pd.DataFrame(rows, columns=COMPARISON_COLUMNS).to_csv(out/"candidate_comparisons.csv", index=False)
    pd.DataFrame(families, columns=FAMILY_COLUMNS).to_csv(out/"family_summary.csv", index=False)
    pd.DataFrame(predictions, columns=PREDICTION_COLUMNS).to_csv(out/"candidate_predictions.csv", index=False)
    pd.DataFrame(importance, columns=IMPORTANCE_COLUMNS).to_csv(out/"family_importance.csv", index=False)
    return {"comparison_failure_count": sum(r["comparison_status"].startswith(("failed_", "timed_out")) for r in rows),
            "comparison_completed_count": sum(r["comparison_status"] == "completed" for r in rows)}
