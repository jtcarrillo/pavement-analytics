"""Small reproducible model search with grouped CV and an untouched group holdout."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io import write_json
from .metrics import condition_predict, grouped_folds, score
from .security import SafeError


def candidates(seed: int):
    for alpha in [0.01, 0.1]:
        yield f"enet_{alpha}", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=20000, random_state=seed)),
        ])
    for leaf in [2, 5]:
        yield f"rf_{leaf}", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", RandomForestRegressor(n_estimators=100, min_samples_leaf=leaf,
                                             max_features=1.0, random_state=seed, n_jobs=1)),
        ])


def search(table: pd.DataFrame, features: list[str], target: str, out: Path, *,
           folds: int = 5, seed: int = 42, holdout_fraction: float = 0.2) -> dict:
    """Select on CV only; evaluate the selected estimator on reserved groups once.

    The chosen estimator is fitted on development groups and stays that way for
    the saved model. The holdout is never included in the saved model's fitting.
    """
    if "PCI_start" not in features or not 0 < holdout_fraction < 0.5:
        raise SafeError("CONDITION_FEATURE_AND_GROUP_HOLDOUT_REQUIRED")
    data = table.loc[np.isfinite(table[target])].reset_index(drop=True)
    groups = np.unique(data.group_id.astype(str))
    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    count = max(1, int(math.ceil(len(groups) * holdout_fraction)))
    if len(groups) - count < folds:
        raise SafeError("INSUFFICIENT_GROUPS_FOR_MODEL_AND_HOLDOUT")
    hold = data.group_id.isin(groups[:count]).to_numpy()
    development = data.loc[~hold].reset_index(drop=True)
    heldout = data.loc[hold].reset_index(drop=True)
    X = development[features]
    y = development[target].to_numpy(float)
    weights = development.delta_days.to_numpy(float)
    splits = list(grouped_folds(development.group_id, folds, seed))
    estimators = dict(candidates(seed))
    model_names = ["mean", "condition", *estimators]
    scores, predictions, fold_records = [], [], []
    for fold, (train, test) in enumerate(splits):
        for group in development.iloc[test].group_id.unique():
            fold_records.append({"group_id": group, "partition": "development", "fold": fold})
        for name in model_names:
            if name == "mean":
                predicted = np.full(len(test), np.average(y[train], weights=weights[train]))
            elif name == "condition":
                predicted = condition_predict(X.iloc[train].PCI_start, y[train], weights[train], X.iloc[test].PCI_start)
            else:
                fitted = clone(estimators[name]).fit(X.iloc[train], y[train], model__sample_weight=weights[train])
                predicted = fitted.predict(X.iloc[test])
            scores.append({"candidate": name, "fold": fold, **score(y[test], predicted, weights[test])})
            for index, value in zip(test, predicted):
                predictions.append({"pair_id": int(development.iloc[index].pair_id),
                                    "group_id": development.iloc[index].group_id,
                                    "fold": fold, "candidate": name, "actual": y[index],
                                    "prediction": float(value), "weight": weights[index]})
    fold_scores = pd.DataFrame(scores)
    means = fold_scores.groupby("candidate").agg(wMAE=("wMAE", "mean"), wRMSE=("wRMSE", "mean"))
    baseline = fold_scores.loc[fold_scores.candidate.eq("condition")].sort_values("fold").wMAE.to_numpy()
    wins = {}
    for name in model_names:
        actual = fold_scores.loc[fold_scores.candidate.eq(name)].sort_values("fold").wMAE.to_numpy()
        wins[name] = int(np.sum(actual < baseline))
    means["wins_vs_condition"] = pd.Series(wins)
    eligible = [name for name in estimators if means.loc[name, "wMAE"] < means.loc["condition", "wMAE"]
                and wins[name] >= math.ceil(0.8 * folds)]
    chosen = min(eligible, key=lambda n: (means.loc[n, "wMAE"], n)) if eligible else None
    bundle = {"features": features, "target": target, "candidate": chosen or "condition",
              "status": "champion" if chosen else "no_model", "format_version": 1}
    if chosen:
        fitted = clone(estimators[chosen]).fit(X, y, model__sample_weight=weights)
        bundle["estimator"] = fitted
        predicted = fitted.predict(heldout[features])
    else:
        bundle["condition_means"] = condition_predict(X.PCI_start, y, weights, [20, 45, 60, 75, 90])
        predicted = condition_predict(X.PCI_start, y, weights, heldout.PCI_start)
    baseline_predicted = condition_predict(X.PCI_start, y, weights, heldout.PCI_start)
    for group in heldout.group_id.unique():
        fold_records.append({"group_id": group, "partition": "holdout", "fold": -1})
    summary = {"status": bundle["status"], "candidate": bundle["candidate"], "target": target,
               "selection_metric": "mean_grouped_cv_wMAE", "seed": seed, "folds": folds,
               "development_rows": len(development), "holdout_rows": len(heldout),
               "holdout": score(heldout[target], predicted, heldout.delta_days),
               "holdout_condition": score(heldout[target], baseline_predicted, heldout.delta_days),
               "excluded_missing_target": len(table) - len(data)}
    out.mkdir(parents=True, exist_ok=False)
    fold_scores.to_csv(out / "fold_scores.csv", index=False)
    means.reset_index().sort_values(["wMAE", "candidate"]).to_csv(out / "leaderboard.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_records).to_csv(out / "folds.csv", index=False)
    held_predictions = heldout[["pair_id", "group_id", "delta_days", target]].copy()
    held_predictions["prediction"] = predicted
    held_predictions["condition_prediction"] = baseline_predicted
    held_predictions.to_csv(out / "holdout_predictions.csv", index=False)
    joblib.dump(bundle, out / "model.joblib")
    write_json(out / "summary.json", summary)
    return summary
