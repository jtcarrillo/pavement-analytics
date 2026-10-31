"""Weighted occurrence/positive severity regression and component diagnostics.

The final point estimate is a conditional mean, p * positive_mean. It is not
the mixture median and need not minimize absolute error on zero-heavy targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .security import SafeError


STAGE_PREDICTION_COLUMNS = ["candidate", "feature_set", "partition", "fold", "row_id", "group_id",
    "actual", "p_positive", "positive_mean", "prediction", "baseline_probability", "baseline_positive_mean",
    "baseline_mean_prediction", "weight", "occurrence_status", "severity_status"]
STAGE_METRIC_COLUMNS = ["candidate", "feature_set", "partition", "candidate_status", "complete_cv",
    "diagnostic_selection", "rows", "groups", "positive_rows", "positive_fraction", "mean_prediction_mae",
    "mean_prediction_rmse", "baseline_mean_rmse", "mean_rmse_skill", "occurrence_brier", "baseline_brier",
    "brier_skill", "occurrence_log_loss", "pr_auc_trapezoid", "average_precision", "ranking_status",
    "positive_severity_mae", "positive_severity_rmse", "positive_baseline_rmse", "positive_support_status"]
COMPONENT_NAMES = ["p_positive", "positive_mean", "prediction", "baseline_probability",
                   "baseline_positive_mean", "baseline_mean_prediction"]


def checked_components(values, size):
    result = {key: np.asarray(values[key], dtype=float) for key in COMPONENT_NAMES}
    if any(x.shape != (size,) or not np.isfinite(x).all() or (x < 0).any() for x in result.values()):
        raise SafeError("INVALID_TWO_STAGE_COMPONENT_PREDICTIONS")
    if (result["p_positive"] > 1).any() or (result["baseline_probability"] > 1).any():
        raise SafeError("INVALID_TWO_STAGE_COMPONENT_PREDICTIONS")
    if not np.allclose(result["prediction"], result["p_positive"]*result["positive_mean"], rtol=1e-12, atol=1e-12):
        raise SafeError("INVALID_TWO_STAGE_MEAN_PREDICTION")
    if not np.allclose(result["baseline_mean_prediction"], result["baseline_probability"]*result["baseline_positive_mean"], rtol=1e-12, atol=1e-12):
        raise SafeError("INVALID_TWO_STAGE_MEAN_PREDICTION")
    return result


class OccurrenceSeverityRegressor(RegressorMixin, BaseEstimator):
    """Learn probability on all training rows and magnitude only on positives.

    Group context is required to enforce sparse-class/group fallbacks. Both
    native estimators are cloned and fit on their own training-only feature path.
    """
    handles_constant_targets = True

    def __init__(self, occurrence_estimator, severity_estimator, min_class_rows=5,
                 min_class_groups=2, min_positive_rows=10, min_positive_groups=2):
        self.occurrence_estimator, self.severity_estimator = occurrence_estimator, severity_estimator
        self.min_class_rows, self.min_class_groups = min_class_rows, min_class_groups
        self.min_positive_rows, self.min_positive_groups = min_positive_rows, min_positive_groups

    def fit_grouped(self, X, y, sample_weight=None, groups=None):
        return self.fit(X, y, sample_weight=sample_weight, groups=groups)

    def fit(self, X, y, sample_weight=None, groups=None):
        y = np.asarray(y, dtype=float)
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if (not isinstance(X, pd.DataFrame) or not len(y) or y.shape != (len(X),) or weights.shape != y.shape
                or not np.isfinite(y).all() or (y < 0).any() or not np.isfinite(weights).all() or (weights <= 0).any()):
            raise SafeError("INVALID_WEIGHTED_MODEL_INPUT")
        if groups is None:
            raise SafeError("MODEL_GROUP_CONTEXT_REQUIRED")
        groups = pd.Series(groups).reset_index(drop=True)
        if len(groups) != len(y) or groups.isna().any() or groups.astype(str).str.strip().eq("").any():
            raise SafeError("MODEL_VALID_TRAINING_GROUPS_REQUIRED")
        groups = groups.astype(str)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in
               (self.min_class_rows, self.min_class_groups, self.min_positive_rows, self.min_positive_groups)):
            raise SafeError("INVALID_TWO_STAGE_SUPPORT_POLICY")
        positive = y > 0
        self.__dict__.pop("training_details_", None)
        n_positive, n_negative = int(positive.sum()), int((~positive).sum())
        positive_groups, negative_groups = sorted(groups[positive].unique()), sorted(groups[~positive].unique())
        self.prior_probability_ = float(np.average(positive, weights=weights))
        self.positive_baseline_ = float(np.average(y[positive], weights=weights[positive])) if n_positive else 0.
        self.occurrence_estimator_, self.severity_estimator_ = None, None
        self.constant_probability_ = self.prior_probability_
        self.constant_positive_mean_ = self.positive_baseline_
        if not n_positive:
            occurrence_status = "constant_zero"
        elif not n_negative:
            occurrence_status = "constant_one"
        elif (min(n_positive, n_negative) < self.min_class_rows
              or min(len(positive_groups), len(negative_groups)) < self.min_class_groups):
            occurrence_status = "sparse_class_weighted_prevalence"
        else:
            self.occurrence_estimator_ = clone(self.occurrence_estimator)
            if isinstance(self.occurrence_estimator_, Pipeline):
                step = self.occurrence_estimator_.steps[-1][0]
                self.occurrence_estimator_.fit(X, positive.astype(int), **{step+"__sample_weight": weights})
            else:
                self.occurrence_estimator_.fit(X, positive.astype(int), sample_weight=weights)
            occurrence_status = "fitted_classifier"
        if not n_positive:
            severity_status = "no_positive_training"
        elif np.ptp(y[positive]) == 0:
            severity_status = "constant_positive_target"
        elif n_positive < self.min_positive_rows or len(positive_groups) < self.min_positive_groups:
            severity_status = "sparse_positive_weighted_mean"
        else:
            self.severity_estimator_ = clone(self.severity_estimator).fit(
                X.iloc[positive], y[positive], sample_weight=weights[positive])
            severity_status = "fitted_positive_regressor"
        members = sorted(groups.unique())
        classifier = (self.occurrence_estimator_.steps[-1][1]
                      if isinstance(self.occurrence_estimator_, Pipeline) else self.occurrence_estimator_)
        iterations = getattr(classifier, "n_iter_", [])
        self.training_details_ = {"iteration_policy": "weighted_occurrence_and_positive_severity",
            "iterations": None, "probe_iterations": 0, "prediction_meaning": "conditional_mean_p_times_positive_mean",
            "fitting_groups": members, "refit_groups": members, "stopping_groups": [],
            "fitting_rows": len(y), "refit_rows": len(y), "fitting_weight_sum": float(weights.sum()),
            "refit_weight_sum": float(weights.sum()), "positive_rows": n_positive, "negative_rows": n_negative,
            "positive_groups": positive_groups, "negative_groups": negative_groups,
            "occurrence_training_groups": members, "severity_training_groups": positive_groups,
            "positive_weight_sum": float(weights[positive].sum()), "negative_weight_sum": float(weights[~positive].sum()),
            "prior_probability": self.prior_probability_, "positive_baseline": self.positive_baseline_,
            "occurrence_status": occurrence_status, "severity_status": severity_status,
            "occurrence_iterations": int(np.max(iterations)) if len(iterations) else 0,
            "min_class_rows": self.min_class_rows, "min_class_groups": self.min_class_groups,
            "min_positive_rows": self.min_positive_rows, "min_positive_groups": self.min_positive_groups}
        self.n_features_in_, self.feature_names_in_ = X.shape[1], np.asarray(X.columns, dtype=object)
        return self

    def predict_occurrence_severity(self, X):
        check_is_fitted(self, "training_details_")
        probability = np.full(len(X), self.constant_probability_)
        if self.occurrence_estimator_ is not None:
            classes = np.asarray(self.occurrence_estimator_.classes_)
            positive = np.flatnonzero(classes == 1)
            matrix = np.asarray(self.occurrence_estimator_.predict_proba(X), dtype=float)
            if (len(positive) != 1 or matrix.shape != (len(X), len(classes)) or not np.isfinite(matrix).all()
                    or (matrix < 0).any() or (matrix > 1).any()
                    or not np.allclose(matrix.sum(axis=1), 1, rtol=0, atol=1e-10)):
                raise SafeError("INVALID_OCCURRENCE_CLASS_SCHEMA")
            probability = matrix[:, positive[0]]
        severity = (np.full(len(X), self.constant_positive_mean_) if self.severity_estimator_ is None
                    else np.asarray(self.severity_estimator_.predict(X), dtype=float))
        return checked_components({"p_positive": probability, "positive_mean": severity,
            "prediction": probability*severity, "baseline_probability": np.full(len(X), self.prior_probability_),
            "baseline_positive_mean": np.full(len(X), self.positive_baseline_),
            "baseline_mean_prediction": np.full(len(X), self.prior_probability_*self.positive_baseline_)}, len(X))

    def predict(self, X):
        return self.predict_occurrence_severity(X)["prediction"]


def distribution_metrics(frame):
    """Pooled saved evaluation weights; ranking metrics need both classes.

    Log loss clips probabilities to [1e-15, 1-1e-15]. PR AUC uses trapezoids;
    average precision is reported separately and is not the same quantity.
    """
    actual, point, probability, severity, weights = (frame[key].to_numpy(float) for key in
        ("actual", "prediction", "p_positive", "positive_mean", "weight"))
    if not len(actual) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise SafeError("INVALID_TWO_STAGE_EVALUATION_WEIGHTS")
    observed = actual > 0
    average = lambda x: float(np.average(x, weights=weights))
    rmse = float(np.sqrt(average((actual-point)**2)))
    baseline_rmse = float(np.sqrt(average((actual-frame.baseline_mean_prediction.to_numpy(float))**2)))
    brier = average((observed-probability)**2)
    baseline_brier = average((observed-frame.baseline_probability.to_numpy(float))**2)
    p = np.clip(probability, 1e-15, 1-1e-15)
    result = {"rows": len(frame), "groups": int(frame.group_id.nunique()), "positive_rows": int(observed.sum()),
        "positive_fraction": average(observed), "mean_prediction_mae": average(np.abs(actual-point)),
        "mean_prediction_rmse": rmse, "baseline_mean_rmse": baseline_rmse,
        "mean_rmse_skill": 1-rmse/baseline_rmse if baseline_rmse > 0 else None,
        "occurrence_brier": brier, "baseline_brier": baseline_brier,
        "brier_skill": 1-brier/baseline_brier if baseline_brier > 0 else None,
        "occurrence_log_loss": average(-(observed*np.log(p)+(~observed)*np.log1p(-p))),
        "pr_auc_trapezoid": None, "average_precision": None,
        "ranking_status": "both_classes" if observed.any() and (~observed).any() else "single_class_evaluation",
        "positive_severity_mae": None, "positive_severity_rmse": None, "positive_baseline_rmse": None,
        "positive_support_status": "positive_evaluation_rows" if observed.any() else "no_positive_evaluation_rows"}
    if result["ranking_status"] == "both_classes":
        precision, recall, _ = precision_recall_curve(observed, probability, sample_weight=weights)
        result.update(pr_auc_trapezoid=float(auc(recall, precision)),
                      average_precision=float(average_precision_score(observed, probability, sample_weight=weights)))
    if observed.any():
        error = actual[observed]-severity[observed]
        baseline_error = actual[observed]-frame.baseline_positive_mean.to_numpy(float)[observed]
        result.update(positive_severity_mae=float(np.average(np.abs(error), weights=weights[observed])),
                      positive_severity_rmse=float(np.sqrt(np.average(error**2, weights=weights[observed]))),
                      positive_baseline_rmse=float(np.sqrt(np.average(baseline_error**2, weights=weights[observed]))))
    return result
