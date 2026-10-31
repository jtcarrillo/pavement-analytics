"""Deduct-level searches on one shared, group-disjoint evaluation plan.

By default errors and permutations give each evaluation group equal total weight.
The original-pipeline adapter may explicitly supply positive interval weights.
The saved estimators are fitted on development data only. Feature importance
describes predictive reliance, not a causal effect; repeat variability is not
a sampling confidence interval.
"""
from __future__ import annotations

import math
import json
from pathlib import Path
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io import write_json
from .model_registry import (REGISTRY_VERSION, build_candidates, candidate_catalog,
                             environment_versions, metadata_for)
from .security import SafeError
from .model_runtime import ACTIVE, ModelRuntime, ModelTimeout, model_options as normalize_model_options, fit_budget, runtime_context
from .two_stage import (STAGE_PREDICTION_COLUMNS, STAGE_METRIC_COLUMNS,
                        checked_components, distribution_metrics)


def group_weights(groups) -> np.ndarray:
    """Return weights with mean one and equal total weight per group."""
    values = pd.Series(groups, dtype=str).reset_index(drop=True)
    counts = values.value_counts()
    weights = values.map(lambda group: 1.0 / counts[group]).to_numpy(float)
    return weights / weights.mean()


def _weights(table: pd.DataFrame, weight_column: str | None) -> np.ndarray:
    if weight_column is None:
        return group_weights(table.group_id)
    if not isinstance(weight_column, str) or weight_column not in table:
        raise SafeError("INVALID_STUDY_WEIGHT_COLUMN")
    weights = pd.to_numeric(table[weight_column], errors="coerce").to_numpy(float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise SafeError("INVALID_STUDY_WEIGHTS")
    return weights


def _groups(table: pd.DataFrame) -> pd.Series:
    if "group_id" not in table or table.group_id.isna().any():
        raise SafeError("STUDY_VALID_GROUPS_REQUIRED")
    groups = table.group_id.astype(str)
    if groups.str.strip().eq("").any():
        raise SafeError("STUDY_VALID_GROUPS_REQUIRED")
    return groups


def make_partitions(table: pd.DataFrame, folds: int = 5, seed: int = 42,
                    holdout_fraction: float = 0.2) -> pd.DataFrame:
    """Assign groups once, before material/target availability is inspected.

    Small populations still receive a plan. Individual searches report lack
    of coverage; they never redraw a favorable holdout or combine CV folds.
    """
    if (not isinstance(folds, int) or isinstance(folds, bool) or folds < 2
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            or not isinstance(holdout_fraction, (int, float))
            or isinstance(holdout_fraction, bool) or not 0 < holdout_fraction < 0.5):
        raise SafeError("INVALID_STUDY_PARTITION_OPTIONS")
    groups = _groups(table)
    counts = groups.value_counts()
    unique = np.sort(groups.unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_hold = max(1, math.ceil(len(unique) * holdout_fraction))
    held = set(shuffled[:n_hold])
    # Balance development row counts without using any target or feature.
    development = list(shuffled[n_hold:])
    development.sort(key=lambda group: -counts[group])
    loads = np.zeros(folds, dtype=int)
    assignment = {}
    for group in development:
        fold = int(np.argmin(loads))
        assignment[group] = fold
        loads[fold] += int(counts[group])
    result = pd.DataFrame([
        {"group_id": group, "partition": "holdout" if group in held else "development",
         "fold": -1 if group in held else assignment[group]}
        for group in unique], columns=["group_id", "partition", "fold"])
    # A repeated column survives CSV round trips and records empty global folds.
    result["planned_folds"] = folds
    return result


def weighted_median(y, weights) -> float:
    values, weights = np.asarray(y, float), np.asarray(weights, float)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, cumulative[-1] / 2, side="left")]])


class MedianRegressor(RegressorMixin, BaseEstimator):
    """An intercept-only MAE baseline fitted with group-balanced weights."""

    def fit(self, X, y, sample_weight=None):
        weights = np.ones(len(y)) if sample_weight is None else sample_weight
        self.median_ = weighted_median(y, weights)
        return self

    def predict(self, X):
        return np.full(len(X), self.median_, dtype=float)


class FeatureCleaner(TransformerMixin, BaseEstimator):
    """Learn numeric/categorical roles on a training fold, then clean values."""

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.numeric_ = []
        for column in self.columns_:
            values = X[column].replace(r"^\s*$", np.nan, regex=True)
            present = values.dropna()
            if pd.api.types.is_numeric_dtype(values) or pd.to_numeric(present, errors="coerce").notna().all():
                self.numeric_.append(column)
        return self

    def transform(self, X):
        result = X[self.columns_].copy()
        for column in self.columns_:
            if column in self.numeric_:
                result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
                result[column] = result[column].replace([np.inf, -np.inf], np.nan)
            else:
                result[column] = result[column].map(
                    lambda value: np.nan if pd.isna(value) or not str(value).strip() else str(value))
                result[column] = result[column].astype(object)
        return result


class NonnegativeRegressor(RegressorMixin, BaseEstimator):
    """Keep the physical lower bound in both evaluation and saved prediction."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, model__sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.maximum(0, self.estimator_.predict(X))


def _estimator(model):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=64)),
    ])
    return NonnegativeRegressor(Pipeline([
        ("clean", FeatureCleaner()),
        ("prepare", ColumnTransformer([
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
        ], sparse_threshold=0)),
        ("model", model),
    ]))


def candidates(seed: int) -> dict:
    """Compatibility adapter for the implemented, versioned registry."""
    return build_candidates(seed)


def _score(y, prediction, weights) -> dict:
    y, prediction, weights = (np.asarray(v, float) for v in (y, prediction, weights))
    if not len(y) or not np.isfinite(prediction).all():
        raise SafeError("INVALID_STUDY_PREDICTIONS")
    error = y - prediction
    mse = np.average(error ** 2, weights=weights)
    variance = np.average((y - np.average(y, weights=weights)) ** 2, weights=weights)
    return {"mae": float(np.average(np.abs(error), weights=weights)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(1 - mse / variance) if variance > 0 else None}


def _fit(estimator, X, y, weights, groups=None):
    if ACTIVE.get() is not None:
        return ACTIVE.get().call("fit", estimator, X, y, weights, groups)
    # A constant training fold has no learnable relation. This also avoids
    # meaningless coordinate-descent convergence warnings on a zero target.
    if np.ptp(y) == 0 and not getattr(estimator, "handles_constant_targets", False):
        return MedianRegressor().fit(X, y, sample_weight=weights)
    fitted = clone(estimator)
    if callable(getattr(fitted, "fit_grouped", None)):
        return fitted.fit_grouped(X, y, sample_weight=weights, groups=groups)
    return fitted.fit(X, y, sample_weight=weights)


def _training_details(fitted, estimator, groups):
    details = getattr(fitted, "training_details_", None)
    if details is not None:
        return details
    if isinstance(fitted, MedianRegressor) and callable(getattr(estimator, "fit_grouped", None)):
        return {"iteration_policy": "constant_training_target", "iterations": 0,
                "probe_iterations": 0, "fitting_groups": sorted(set(groups.astype(str))),
                "stopping_groups": [], "refit_groups": sorted(set(groups.astype(str)))}
    return None


IMPORTANCE_COLUMNS = ["feature", "kind", "is_climate", "importance_mean", "importance_std",
                      "repeats", "reliable"]
PREDICTION_COLUMNS = ["row_id", "group_id", "actual", "prediction", "baseline_prediction",
                      "nonclimate_prediction", "weight"]
FOLD_COLUMNS = ["candidate", "feature_set", "fold", "mae", "rmse", "r2"]
LEADERBOARD_COLUMNS = ["candidate", "feature_set", "cv_mae", "cv_mae_std", "cv_rmse",
                       "cv_wins_vs_baseline", "family", "candidate_status", "error_code",
                       "folds_completed", "selected", "parameters", "cv_iterations",
                       "cv_training", "final_iterations", "final_training", "two_stage_diagnostic", "diagnostic_training"]


def _write_empty(out, summary):
    from .model_comparison import empty_comparisons
    empty_comparisons(out)
    for name, columns in [("importance.csv", IMPORTANCE_COLUMNS),
                          ("holdout_predictions.csv", PREDICTION_COLUMNS),
                          ("fold_scores.csv", FOLD_COLUMNS),
                          ("leaderboard.csv", LEADERBOARD_COLUMNS),
                          ("two_stage_predictions.csv", STAGE_PREDICTION_COLUMNS),
                          ("two_stage_metrics.csv", STAGE_METRIC_COLUMNS)]:
        pd.DataFrame(columns=columns).to_csv(out / name, index=False)
    write_json(out / "candidate_audit.json", {"registry_version": REGISTRY_VERSION,
               "task_status": summary["status"], "attempts": [],
               "catalog": candidate_catalog(summary["seed"])})
    write_json(out / "summary.json", summary)
    return summary


def _checked_prediction(estimator, X):
    prediction = np.asarray(estimator.predict(X), dtype=float)
    if prediction.shape != (len(X),) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("MODEL_PREDICTIONS_MUST_BE_FINITE_NONNEGATIVE_VECTOR")
    return prediction


def _candidate_failure(out, attempt, status, code):
    import sys
    if isinstance(sys.exc_info()[1], ModelTimeout):
        status, code = status.replace("failed_", "timed_out_", 1), "MODEL_OPERATION_TIMED_OUT"
    attempt.update(candidate_status=status, error_code=code, selected=False)
    with (out / "candidate_errors.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{attempt['candidate']} / {attempt['feature_set']} / {code}\n")
        stream.write(traceback.format_exc() + "\n")


def _prediction_bundle(estimator, X):
    if ACTIVE.get() is not None:
        return ACTIVE.get().call("predict", estimator, X)
    prediction = _checked_prediction(estimator, X)
    method = getattr(estimator, "predict_occurrence_severity", None)
    components = checked_components(method(X), len(X)) if callable(method) else None
    if components is not None and not np.allclose(prediction, components["prediction"], rtol=1e-12, atol=1e-12):
        raise SafeError("INCONSISTENT_TWO_STAGE_POINT_PREDICTION")
    return prediction, components


def _component_rows(estimator, components, table, target, weights, candidate, feature_set, partition, fold):
    if components is None:
        return []
    rows = table[["row_id", "group_id", target]].rename(columns={target: "actual"}).copy()
    rows["candidate"], rows["feature_set"], rows["partition"], rows["fold"] = candidate, feature_set, partition, fold
    for key, values in components.items():
        rows[key] = values
    rows["weight"] = weights
    details = getattr(estimator, "training_details_", {})
    rows["occurrence_status"] = details.get("occurrence_status", "custom")
    rows["severity_status"] = details.get("severity_status", "custom")
    return rows.to_dict("records")


def _write_stage_metrics(out, stage_rows, attempts, planned_folds):
    predictions = pd.DataFrame(stage_rows, columns=STAGE_PREDICTION_COLUMNS)
    predictions.to_csv(out/"two_stage_predictions.csv", index=False)
    lookup = {(r["candidate"], r["feature_set"]): r for r in attempts}
    rows = []
    for (candidate, feature_set, partition), frame in predictions.groupby(["candidate", "feature_set", "partition"], sort=True):
        attempt = lookup[(candidate, feature_set)]
        rows.append({"candidate": candidate, "feature_set": feature_set, "partition": partition,
                     "candidate_status": attempt["candidate_status"],
                     "complete_cv": attempt["folds_completed"] == planned_folds,
                     "diagnostic_selection": "mean_development_cv_MAE" if partition == "holdout" else "all_attempted_folds",
                     **distribution_metrics(frame)})
    pd.DataFrame(rows, columns=STAGE_METRIC_COLUMNS).to_csv(out/"two_stage_metrics.csv", index=False)


def _evaluate_candidates(development, heldout, columns, target, feature_set,
                         planned_folds, seed, weight_column, out, constant, requested):
    """Select on complete CV results; isolate construction/fit/predict failures."""
    y = development[target].to_numpy(float)
    estimators = {"median": MedianRegressor()}
    if columns and not constant:
        pool = candidates(seed) if requested is None else build_candidates(seed, requested)
        estimators.update(pool)
        entries = getattr(pool, "entries", [dict(metadata_for(name, seed), candidate_status="ready", error_code=None)
                                          for name in pool])
        diagnostics = getattr(pool, "diagnostics", {})
    else:
        entries = [dict(entry, candidate_status="not_attempted_constant_target" if constant else "not_attempted_no_features",
                        error_code=None) for entry in candidate_catalog(seed) if entry["candidate"] != "median"
                   and (requested is None or entry["candidate"] in requested)]
        diagnostics = {}
    entries = [dict(metadata_for("median", seed), candidate_status="ready", error_code=None), *entries]
    attempts = {entry["candidate"]: dict(entry, feature_set=feature_set, folds_completed=0, selected=False,
                                        cv_training=[], final_training=None) for entry in entries}
    for name, diagnostic in diagnostics.items():
        with (out / "candidate_errors.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{name} / {feature_set} / MODEL_CONSTRUCTION_FAILED\n{diagnostic}\n")
    results, oof_rows, stage_rows = [], [], []
    for fold in range(planned_folds):
        train = development.fold.ne(fold).to_numpy()
        test = ~train
        train_weights, test_weights = (_weights(development.loc[mask], weight_column) for mask in (train, test))
        for name, estimator in estimators.items():
            attempt = attempts[name]
            if attempt["candidate_status"] != "ready":
                continue
            runtime_context(candidate=name, family=attempt["family"], feature_set=feature_set, phase="cv", fold=fold)
            try:
                fitted = _fit(estimator, development.loc[train, columns], y[train], train_weights,
                              development.loc[train, "group_id"])
                details = _training_details(fitted, estimator, development.loc[train, "group_id"])
                if details is not None:
                    attempt["cv_training"].append(dict(fold=fold, **details))
            except Exception:
                if name == "median":
                    raise
                _candidate_failure(out, attempt, "failed_fit", "CANDIDATE_FIT_FAILED")
                continue
            try:
                prediction, components = _prediction_bundle(fitted, development.loc[test, columns])
                component_rows = _component_rows(fitted, components, development.loc[test], target, test_weights,
                                                 name, feature_set, "development_oof", fold)
            except Exception:
                if name == "median":
                    raise
                _candidate_failure(out, attempt, "failed_predict", "CANDIDATE_PREDICTION_FAILED")
                continue
            results.append({"candidate": name, "feature_set": feature_set, "fold": fold,
                            **_score(y[test], prediction, test_weights)})
            attempt["folds_completed"] += 1
            stage_rows.extend(component_rows)
            identities = development.loc[test, ["row_id", "group_id", target]].itertuples(index=False, name=None)
            oof_rows.extend({"row_id": row_id, "group_id": group_id, "candidate": name,
                             "feature_set": feature_set, "fold": fold, "actual": float(actual),
                             "prediction": float(predicted), "weight": float(weight)}
                            for (row_id, group_id, actual), predicted, weight in zip(identities, prediction, test_weights))
    scores = pd.DataFrame(results)
    baseline = scores.loc[scores.candidate.eq("median")].set_index("fold").mae
    leaders = []
    for name, attempt in attempts.items():
        row = {"candidate": name, "feature_set": feature_set, "family": attempt["family"],
               "cv_mae": None, "cv_mae_std": None, "cv_rmse": None, "cv_wins_vs_baseline": None,
               "parameters": json.dumps(attempt["parameters"], sort_keys=True), "folds_completed": attempt["folds_completed"],
               "cv_iterations": json.dumps([fit["iterations"] for fit in attempt["cv_training"]]),
               "cv_training": json.dumps(attempt["cv_training"], sort_keys=True),
               "final_iterations": None, "final_training": None, "two_stage_diagnostic": False, "diagnostic_training": None}
        if attempt["candidate_status"] == "ready" and attempt["folds_completed"] == planned_folds:
            attempt["candidate_status"] = "completed"
            block = scores.loc[scores.candidate.eq(name)].sort_values("fold")
            row.update(cv_mae=float(block.mae.mean()), cv_mae_std=float(block.mae.std(ddof=0)),
                       cv_rmse=float(block.rmse.mean()),
                       cv_wins_vs_baseline=int((block.mae.to_numpy() < baseline.loc[block.fold].to_numpy()).sum()))
        row.update(candidate_status=attempt["candidate_status"], error_code=attempt["error_code"], selected=False)
        leaders.append(row)
    # Missing/partial results can never become an attractive CV score. The
    # baseline wins ties, as it did before the registry refactor.
    eligible = sorted([row for row in leaders if row["candidate_status"] == "completed"],
                      key=lambda row: (row["cv_mae"], row["candidate"] != "median", row["candidate"]))
    primary = None
    for best in eligible:
        name = best["candidate"]
        runtime_context(candidate=name, family=best["family"], feature_set=feature_set, phase="primary_refit", fold=-1)
        try:
            fitted = _fit(estimators[name], development[columns], y, _weights(development, weight_column),
                          development.group_id)
            details = _training_details(fitted, estimators[name], development.group_id)
            attempts[name]["final_training"] = details
            if details is not None:
                best.update(final_iterations=details["iterations"], final_training=json.dumps(details, sort_keys=True))
        except Exception:
            if name == "median":
                raise
            _candidate_failure(out, attempts[name], "failed_refit", "CANDIDATE_REFIT_FAILED")
            best.update(candidate_status=attempts[name]["candidate_status"], error_code=attempts[name]["error_code"])
            continue
        try:
            predicted, components = _prediction_bundle(fitted, heldout[columns])
        except Exception:
            if name == "median":
                raise
            _candidate_failure(out, attempts[name], "failed_refit_predict", "CANDIDATE_REFIT_PREDICTION_FAILED")
            best.update(candidate_status=attempts[name]["candidate_status"], error_code=attempts[name]["error_code"])
            continue
        best["selected"] = attempts[name]["selected"] = True
        primary = (best, fitted, predicted)
        primary_components = components
        break
    if primary is None:
        raise SafeError("NO_VALID_MODEL_BASELINE")
    # One within-family diagnostic fit per feature set, selected on the same
    # CV MAE, preserves component evidence even when the median wins overall.
    for diagnostic in [row for row in eligible if row["family"] == "two_stage" and row["candidate_status"] == "completed"]:
        name = diagnostic["candidate"]
        runtime_context(candidate=name, family="two_stage", feature_set=feature_set, phase="two_stage_refit", fold=-1)
        try:
            if name == primary[0]["candidate"]:
                diagnosed = primary[1]
                components = primary_components
            else:
                diagnosed = _fit(estimators[name], development[columns], y, _weights(development, weight_column), development.group_id)
                _, components = _prediction_bundle(diagnosed, heldout[columns])
            if components is None:
                raise SafeError("TWO_STAGE_COMPONENTS_REQUIRED")
            details = _training_details(diagnosed, estimators[name], development.group_id)
        except Exception:
            _candidate_failure(out, attempts[name], "failed_diagnostic_refit", "TWO_STAGE_DIAGNOSTIC_REFIT_FAILED")
            diagnostic.update(candidate_status=attempts[name]["candidate_status"], error_code=attempts[name]["error_code"],
                              two_stage_diagnostic=False)
            continue
        joblib.dump({"format_version": 1, "candidate": name, "candidate_family": "two_stage",
                     "candidate_registry_version": REGISTRY_VERSION, "features": columns, "target": target,
                     "estimator": diagnosed, "training_details": details, "fit_partition": "development",
                     "selection_metric": "mean_development_cv_MAE", "weight_column": weight_column,
                     "environment": environment_versions()}, out/("two_stage_"+feature_set+"_model.joblib"))
        diagnostic.update(two_stage_diagnostic=True, diagnostic_training=json.dumps(details, sort_keys=True))
        attempts[name].update(two_stage_diagnostic=True, diagnostic_training=details)
        stage_rows.extend(_component_rows(diagnosed, components, heldout, target, _weights(heldout, weight_column),
                                          name, feature_set, "holdout", -1))
        break
    return (*primary, results, oof_rows, leaders, list(attempts.values()), stage_rows)


def _validate_plan(table, partitions):
    if not {"group_id", "partition", "fold"} <= set(partitions):
        raise SafeError("INVALID_SHARED_PARTITIONS")
    plan = partitions.copy()
    plan["group_id"] = _groups(plan)
    if (plan.group_id.duplicated().any()
            or not plan.partition.isin(["development", "holdout"]).all()
            or not set(_groups(table)) <= set(plan.group_id)):
        raise SafeError("INVALID_SHARED_PARTITIONS")
    folds = pd.to_numeric(plan.fold, errors="coerce")
    if (not np.isfinite(folds).all() or (folds % 1 != 0).any()
            or not folds[plan.partition.eq("holdout")].eq(-1).all()
            or not folds[plan.partition.eq("development")].ge(0).all()):
        raise SafeError("INVALID_SHARED_PARTITIONS")
    plan["fold"] = folds.astype(int)
    planned = int(plan.loc[plan.partition.eq("development"), "fold"].max() + 1) if (plan.partition == "development").any() else 0
    if "planned_folds" in plan and len(plan):
        values = pd.to_numeric(plan.planned_folds, errors="coerce")
        if not np.isfinite(values).all() or values.nunique() != 1 or (values % 1 != 0).any() or values.iloc[0] < max(2, planned):
            raise SafeError("INVALID_SHARED_PARTITIONS")
        planned = int(values.iloc[0])
    return plan, planned


def _search_levels(table: pd.DataFrame, features: list[str], target: str, out: Path,
                  partitions: pd.DataFrame, climate_features: list[str], seed: int = 42,
                  weight_column: str | None = None, candidate_ids: list[str] | None = None) -> dict:
    """Fit one target, selecting both full and climate-ablation models by CV.

    Material filtering belongs to the caller, but the supplied partition plan
    must be the same for every call. Missing target values are excluded only
    after joining that plan. A skipped task remains an explicit report row.
    """
    if (not features or len(features) != len(set(features)) or target in features
            or not set(features) <= set(table) or not set(climate_features) <= set(features)
            or not {"row_id", "group_id", target} <= set(table)
            or table.row_id.isna().any() or table.row_id.duplicated().any()):
        raise SafeError("INVALID_STUDY_MODEL_TABLE")
    plan, planned_folds = _validate_plan(table, partitions)
    if weight_column is not None:
        if weight_column in features or weight_column == target:
            raise SafeError("STUDY_WEIGHT_MUST_NOT_BE_FEATURE_OR_TARGET")
        _weights(table, weight_column)
    forbidden = {"row_id", "group_id", "section_id", "start_date", "end_date", "partition", "fold", "delta_days"}
    if any(not isinstance(name, str) or name in forbidden or name.endswith(("_end", "_delta"))
           or name.startswith(("level_", "oracle_", "truth_", "_source_")) or "__conditional_" in name
           or name.endswith(("__p_positive", "__positive_mean"))
           for name in features):
        raise SafeError("UNSAFE_MODEL_FEATURE")
    if candidate_ids is not None:
        # Explicit selections are required, rather than silently skipped when a
        # package or supported weighted implementation is unavailable.
        build_candidates(seed, candidate_ids)
    data = table.copy()
    data["group_id"] = _groups(data)
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.loc[np.isfinite(data[target])].copy()
    if data[target].lt(0).any():
        raise SafeError("NEGATIVE_DEDUCT_TARGET")
    data = data.merge(plan[["group_id", "partition", "fold"]], on="group_id", how="left", validate="many_to_one")
    development = data.loc[data.partition.eq("development")].reset_index(drop=True)
    heldout = data.loc[data.partition.eq("holdout")].reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=False)
    plan.to_csv(out / "folds.csv", index=False)
    summary = {"target": target, "status": "no_valid_target", "candidate": None,
               "cv_mae": None, "cv_mae_std": None, "holdout_mae": None,
               "holdout_rmse": None, "holdout_r2": None, "baseline_mae": None,
               "mae_skill": None, "climate_added_skill": None, "nonclimate_mae": None,
               "nonclimate_candidate": None, "importance_reliable": False,
               "valid_target_rows": len(data), "valid_target_groups": int(data.group_id.nunique()),
               "excluded_missing_target": len(table) - len(data),
               "development_rows": len(development), "development_groups": int(development.group_id.nunique()),
               "holdout_rows": len(heldout), "holdout_groups": int(heldout.group_id.nunique()),
               "nonzero_fraction": float(data[target].gt(0).mean()) if len(data) else None,
               "target_std": float(data[target].std(ddof=0)) if len(data) else None,
               "folds": planned_folds, "seed": seed,
               "candidate_registry_version": REGISTRY_VERSION, "candidate_family": None,
               "candidate_failure_count": 0, "candidate_unavailable_count": 0,
               "selection_metric": "mean_group_balanced_cv_MAE" if weight_column is None else "mean_interval_weighted_cv_MAE",
               "weight_column": weight_column,
               "weighting": "Each group receives equal total weight within each evaluation set" if weight_column is None
                            else "Each observation is weighted by its positive inspection interval in days",
               "importance_note": "Predictive reliance, not causation. Repeat SD is permutation variability, not a confidence interval. Correlated predictors can share importance."}
    if data.empty:
        return _write_empty(out, summary)
    if heldout.empty or development.group_id.nunique() < 2:
        summary["status"] = "insufficient_groups"
        return _write_empty(out, summary)
    if planned_folds < 2 or set(development.fold.unique()) != set(range(planned_folds)):
        summary["status"] = "insufficient_fold_coverage"
        return _write_empty(out, summary)
    y = development[target].to_numpy(float)
    training_weights = _weights(development, weight_column)
    held_weights = _weights(heldout, weight_column)
    held_y = heldout[target].to_numpy(float)
    nonclimate = [feature for feature in features if feature not in climate_features]
    constant = np.ptp(y) == 0
    if ACTIVE.get() is not None:
        ACTIVE.get().budget(fit_budget(planned_folds, ACTIVE.get().options, constant=bool(constant)))
    sets = {"all": features, "nonclimate": nonclimate}
    fold_scores, leaders, fitted_sets, attempts = [], [], {}, []
    oof_rows, stage_rows = [], []
    for feature_set, columns in sets.items():
        best, selected, predicted, results, oof, candidate_rows, audit, stages = _evaluate_candidates(
            development, heldout, columns, target, feature_set, planned_folds, seed, weight_column, out, constant, candidate_ids)
        fitted_sets[feature_set] = (best, selected, predicted)
        fold_scores.extend(results)
        leaders.extend(candidate_rows)
        oof_rows.extend(oof)
        attempts.extend(audit)
        stage_rows.extend(stages)
    best, fitted, predicted = fitted_sets["all"]
    nonclimate_best, nonclimate_fitted, nonclimate_predicted = fitted_sets["nonclimate"]
    baseline_fitted = MedianRegressor().fit(development[features], y, sample_weight=training_weights)
    baseline_predicted = baseline_fitted.predict(heldout[features])
    measured = _score(held_y, predicted, held_weights)
    baseline_mae = _score(held_y, baseline_predicted, held_weights)["mae"]
    nonclimate_mae = _score(held_y, nonclimate_predicted, held_weights)["mae"]
    skill = float(1 - measured["mae"] / baseline_mae) if baseline_mae > 0 else None
    reliable = (best["candidate"] != "median" and skill is not None and skill > 0
                and measured["r2"] is not None and measured["r2"] > 0
                and heldout.group_id.nunique() >= 2)
    summary.update(status="constant_target" if constant else ("baseline_selected" if best["candidate"] == "median" else "modeled"),
                   candidate=best["candidate"], cv_mae=best["cv_mae"], cv_mae_std=best["cv_mae_std"],
                   candidate_family=best["family"], candidate_parameters=json.loads(best["parameters"]),
                   candidate_iterations=best.get("final_iterations"),
                   candidate_failure_count=sum(a["candidate_status"].startswith(("failed_", "timed_out_")) for a in attempts),
                   candidate_unavailable_count=sum(a["candidate_status"] in {"unavailable", "unsupported"} for a in attempts),
                   holdout_mae=measured["mae"], holdout_rmse=measured["rmse"], holdout_r2=measured["r2"],
                   baseline_mae=baseline_mae, mae_skill=skill,
                   climate_added_skill=float(1 - measured["mae"] / nonclimate_mae) if climate_features and nonclimate_mae > 0 else None,
                   nonclimate_mae=nonclimate_mae, nonclimate_candidate=nonclimate_best["candidate"],
                   importance_reliable=bool(reliable), importance_repeats=5,
                   constant_target_scope="development" if constant else None)
    importance = []
    if best["candidate"] != "median":
        runtime_context(candidate=best["candidate"], family=best["family"], feature_set="all", phase="primary_importance", fold=-1)
        blocks = [(feature, "feature", [feature], feature in climate_features) for feature in features]
        if climate_features:
            blocks.append(("All climate features", "climate_block", climate_features, True))
        rng = np.random.default_rng(seed + 1701)
        # Reuse permutation orders for each feature for more comparable noise.
        orders = [rng.permutation(len(heldout)) for _ in range(5)]
        for name, kind, columns, is_climate in blocks:
            changes = []
            try:
                for order in orders:
                    perturbed = heldout[features].copy()
                    perturbed.loc[:, columns] = heldout.iloc[order][columns].to_numpy()
                    error = _score(held_y, _prediction_bundle(fitted, perturbed)[0], held_weights)["mae"]
                    changes.append(error - measured["mae"])
            except Exception as error:
                if ACTIVE.get() is None:
                    raise
                summary.update(importance_reliable=False, importance_status="timed_out" if isinstance(error, ModelTimeout) else "failed")
                with (out/"candidate_errors.log").open("a") as log:
                    log.write("PRIMARY_IMPORTANCE_FAILED\n"+traceback.format_exc())
                importance = []
                break
            importance.append({"feature": name, "kind": kind, "is_climate": is_climate,
                               "importance_mean": float(np.mean(changes)),
                               "importance_std": float(np.std(changes, ddof=0)),
                               "repeats": 5, "reliable": bool(reliable)})
    pd.DataFrame(importance, columns=IMPORTANCE_COLUMNS).to_csv(out / "importance.csv", index=False)
    pd.DataFrame(fold_scores, columns=FOLD_COLUMNS).to_csv(out / "fold_scores.csv", index=False)
    pd.DataFrame(leaders, columns=LEADERBOARD_COLUMNS).sort_values(["feature_set", "cv_mae", "candidate"]).to_csv(out / "leaderboard.csv", index=False)
    write_json(out / "candidate_audit.json", {"registry_version": REGISTRY_VERSION, "seed": seed,
               "requested_candidates": candidate_ids, "environment": environment_versions(), "attempts": attempts})
    pd.DataFrame(oof_rows).to_csv(out / "oof_predictions.csv", index=False)
    _write_stage_metrics(out, stage_rows, attempts, planned_folds)
    predictions = heldout[["row_id", "group_id"]].copy()
    predictions["actual"] = held_y
    predictions["prediction"] = predicted
    predictions["baseline_prediction"] = baseline_predicted
    predictions["nonclimate_prediction"] = nonclimate_predicted
    predictions["weight"] = held_weights
    predictions.to_csv(out / "holdout_predictions.csv", index=False)
    joblib.dump({"format_version": 1, "candidate_registry_version": REGISTRY_VERSION,
                 "candidate_family": best["family"], "candidate_parameters": json.loads(best["parameters"]),
                 "training_details": getattr(fitted, "training_details_", None),
                 "nonclimate_training_details": getattr(nonclimate_fitted, "training_details_", None),
                 "environment": environment_versions(), "target": target, "features": features,
                 "climate_features": climate_features, "candidate": best["candidate"],
                 "estimator": fitted, "baseline_estimator": baseline_fitted,
                 "nonclimate_features": nonclimate, "nonclimate_candidate": nonclimate_best["candidate"],
                 "nonclimate_estimator": nonclimate_fitted,
                 "fit_partition": "development", "weight_column": weight_column,
                 "weighting": "equal_total_per_group" if weight_column is None else "interval_days"}, out / "model.joblib")
    write_json(out / "summary.json", summary)
    if ACTIVE.get() is not None and ACTIVE.get().options["comparisons"]:
        from .model_comparison import build_comparisons
        summary.update(build_comparisons(out, development, heldout, features, climate_features, target,
                                         seed, weight_column, candidate_ids))
        write_json(out/"summary.json", summary)
    return summary


def search_levels(table: pd.DataFrame, features: list[str], target: str, out: Path,
                  partitions: pd.DataFrame, climate_features: list[str], seed: int = 42,
                  weight_column: str | None = None, candidate_ids: list[str] | None = None,
                  model_options: dict | None = None, progress=None) -> dict:
    """Run one target. Full/study enable bounded R7 comparisons explicitly.

    Omitting model_options preserves the established in-process library path.
    """
    if model_options is None and progress is None:
        return _search_levels(table, features, target, out, partitions, climate_features, seed, weight_column, candidate_ids)
    options = normalize_model_options(model_options)
    if candidate_ids is not None and options["candidate_ids"] is not None and candidate_ids != options["candidate_ids"]:
        raise SafeError("DUPLICATE_MODEL_CANDIDATE_SELECTION")
    options["candidate_ids"] = candidate_ids if candidate_ids is not None else options["candidate_ids"]
    with ModelRuntime(out, options, progress):
        return _search_levels(table, features, target, out, partitions, climate_features, seed, weight_column, options["candidate_ids"])
