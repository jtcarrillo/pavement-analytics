"""Deduct-level searches on one shared, group-disjoint evaluation plan.

All errors and permutations give each evaluation group equal total weight.
The saved estimators are fitted on development data only. Feature importance
describes predictive reliance, not a causal effect; repeat variability is not
a sampling confidence interval.
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io import write_json
from .security import SafeError


def group_weights(groups) -> np.ndarray:
    """Return weights with mean one and equal total weight per group."""
    values = pd.Series(groups, dtype=str).reset_index(drop=True)
    counts = values.value_counts()
    weights = values.map(lambda group: 1.0 / counts[group]).to_numpy(float)
    return weights / weights.mean()


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
    """Small fixed search space, with no holdout-driven tuning."""
    return {
        **{f"enet_{alpha}": _estimator(ElasticNet(
            alpha=alpha, l1_ratio=0.5, max_iter=20000, random_state=seed))
           for alpha in (0.01, 0.1)},
        **{f"rf_{leaf}": _estimator(RandomForestRegressor(
            n_estimators=100, min_samples_leaf=leaf, max_depth=12,
            max_features=0.8, n_jobs=1, random_state=seed)) for leaf in (5, 20)},
    }


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


def _fit(estimator, X, y, weights):
    # A constant training fold has no learnable relation. This also avoids
    # meaningless coordinate-descent convergence warnings on a zero target.
    if np.ptp(y) == 0:
        return MedianRegressor().fit(X, y, sample_weight=weights)
    return clone(estimator).fit(X, y, sample_weight=weights)


IMPORTANCE_COLUMNS = ["feature", "kind", "is_climate", "importance_mean", "importance_std",
                      "repeats", "reliable"]
PREDICTION_COLUMNS = ["row_id", "group_id", "actual", "prediction", "baseline_prediction",
                      "nonclimate_prediction", "weight"]
FOLD_COLUMNS = ["candidate", "feature_set", "fold", "mae", "rmse", "r2"]
LEADERBOARD_COLUMNS = ["candidate", "feature_set", "cv_mae", "cv_mae_std", "cv_rmse",
                       "cv_wins_vs_baseline"]


def _write_empty(out, summary):
    for name, columns in [("importance.csv", IMPORTANCE_COLUMNS),
                          ("holdout_predictions.csv", PREDICTION_COLUMNS),
                          ("fold_scores.csv", FOLD_COLUMNS),
                          ("leaderboard.csv", LEADERBOARD_COLUMNS)]:
        pd.DataFrame(columns=columns).to_csv(out / name, index=False)
    write_json(out / "summary.json", summary)
    return summary


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


def search_levels(table: pd.DataFrame, features: list[str], target: str, out: Path,
                  partitions: pd.DataFrame, climate_features: list[str], seed: int = 42) -> dict:
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
               "selection_metric": "mean_group_balanced_cv_MAE",
               "weighting": "Each group receives equal total weight within each evaluation set",
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
    training_weights = group_weights(development.group_id)
    held_weights = group_weights(heldout.group_id)
    held_y = heldout[target].to_numpy(float)
    nonclimate = [feature for feature in features if feature not in climate_features]
    constant = np.ptp(y) == 0
    sets = {"all": features, "nonclimate": nonclimate}
    fold_scores, leaders, fitted_sets = [], [], {}
    oof_rows = []
    for feature_set, columns in sets.items():
        estimators = {"median": MedianRegressor()}
        if columns and not constant:
            estimators.update(candidates(seed))
        results = []
        for fold in range(planned_folds):
            train = development.fold.ne(fold).to_numpy()
            test = ~train
            fold_train_weights = group_weights(development.loc[train, "group_id"])
            fold_test_weights = group_weights(development.loc[test, "group_id"])
            for name, estimator in estimators.items():
                fitted = _fit(estimator, development.loc[train, columns], y[train], fold_train_weights)
                prediction = fitted.predict(development.loc[test, columns])
                record = {"candidate": name, "feature_set": feature_set, "fold": fold,
                          **_score(y[test], prediction, fold_test_weights)}
                results.append(record)
                identities = development.loc[test, ["row_id", "group_id", target]].itertuples(index=False, name=None)
                for (row_id, group_id, actual), predicted, weight in zip(identities, prediction, fold_test_weights):
                    oof_rows.append({"row_id": row_id, "group_id": group_id,
                                     "candidate": name, "feature_set": feature_set, "fold": fold,
                                     "actual": float(actual),
                                     "prediction": float(predicted), "weight": float(weight)})
        scores = pd.DataFrame(results)
        baseline = scores.loc[scores.candidate.eq("median")].set_index("fold").mae
        candidate_rows = []
        for name, block in scores.groupby("candidate", sort=True):
            block = block.sort_values("fold")
            candidate_rows.append({"candidate": name, "feature_set": feature_set,
                                   "cv_mae": float(block.mae.mean()), "cv_mae_std": float(block.mae.std(ddof=0)),
                                   "cv_rmse": float(block.rmse.mean()),
                                   "cv_wins_vs_baseline": int((block.mae.to_numpy() < baseline.loc[block.fold].to_numpy()).sum())})
        # Prefer the baseline if its CV score ties a more complex model.
        best = min(candidate_rows, key=lambda row: (row["cv_mae"], row["candidate"] != "median", row["candidate"]))
        selected = _fit(estimators[best["candidate"]], development[columns], y, training_weights)
        predicted = selected.predict(heldout[columns])
        fitted_sets[feature_set] = (best, selected, predicted)
        fold_scores.extend(results)
        leaders.extend(candidate_rows)
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
                   holdout_mae=measured["mae"], holdout_rmse=measured["rmse"], holdout_r2=measured["r2"],
                   baseline_mae=baseline_mae, mae_skill=skill,
                   climate_added_skill=float(1 - measured["mae"] / nonclimate_mae) if climate_features and nonclimate_mae > 0 else None,
                   nonclimate_mae=nonclimate_mae, nonclimate_candidate=nonclimate_best["candidate"],
                   importance_reliable=bool(reliable), importance_repeats=5,
                   constant_target_scope="development" if constant else None)
    importance = []
    if best["candidate"] != "median":
        blocks = [(feature, "feature", [feature], feature in climate_features) for feature in features]
        if climate_features:
            blocks.append(("All climate features", "climate_block", climate_features, True))
        rng = np.random.default_rng(seed + 1701)
        # Reuse permutation orders for each feature for more comparable noise.
        orders = [rng.permutation(len(heldout)) for _ in range(5)]
        for name, kind, columns, is_climate in blocks:
            changes = []
            for order in orders:
                perturbed = heldout[features].copy()
                perturbed.loc[:, columns] = heldout.iloc[order][columns].to_numpy()
                error = _score(held_y, fitted.predict(perturbed), held_weights)["mae"]
                changes.append(error - measured["mae"])
            importance.append({"feature": name, "kind": kind, "is_climate": is_climate,
                               "importance_mean": float(np.mean(changes)),
                               "importance_std": float(np.std(changes, ddof=0)),
                               "repeats": 5, "reliable": bool(reliable)})
    pd.DataFrame(importance, columns=IMPORTANCE_COLUMNS).to_csv(out / "importance.csv", index=False)
    pd.DataFrame(fold_scores, columns=FOLD_COLUMNS).to_csv(out / "fold_scores.csv", index=False)
    pd.DataFrame(leaders, columns=LEADERBOARD_COLUMNS).sort_values(["feature_set", "cv_mae", "candidate"]).to_csv(out / "leaderboard.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out / "oof_predictions.csv", index=False)
    predictions = heldout[["row_id", "group_id"]].copy()
    predictions["actual"] = held_y
    predictions["prediction"] = predicted
    predictions["baseline_prediction"] = baseline_predicted
    predictions["nonclimate_prediction"] = nonclimate_predicted
    predictions["weight"] = held_weights
    predictions.to_csv(out / "holdout_predictions.csv", index=False)
    joblib.dump({"format_version": 1, "target": target, "features": features,
                 "climate_features": climate_features, "candidate": best["candidate"],
                 "estimator": fitted, "baseline_estimator": baseline_fitted,
                 "nonclimate_features": nonclimate, "nonclimate_candidate": nonclimate_best["candidate"],
                 "nonclimate_estimator": nonclimate_fitted,
                 "fit_partition": "development", "weighting": "equal_total_per_group"}, out / "model.joblib")
    write_json(out / "summary.json", summary)
    return summary
