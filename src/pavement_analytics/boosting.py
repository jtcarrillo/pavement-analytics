"""Optional CPU boosting with training-only grouped stopping and refitting."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .model_registry import XGBOOST_MAX_VERSION, XGBOOST_MIN_VERSION, dependency_version
from .security import SafeError


def require_xgboost():
    installed, supported = dependency_version("xgboost", XGBOOST_MIN_VERSION, XGBOOST_MAX_VERSION)
    if installed is None:
        raise SafeError("MODEL_DEPENDENCY_UNAVAILABLE")
    if not supported:
        raise SafeError("MODEL_DEPENDENCY_VERSION_UNSUPPORTED")
    from xgboost import XGBRegressor
    return XGBRegressor


def grouped_stopping_split(groups, fraction, seed):
    """Choose whole groups using identities and seed, never features or labels."""
    series = pd.Series(groups).reset_index(drop=True)
    if series.empty or series.isna().any() or series.astype(str).str.strip().eq("").any():
        raise SafeError("MODEL_VALID_TRAINING_GROUPS_REQUIRED")
    series = series.astype(str)
    unique = np.sort(series.unique())
    if len(unique) < 2:
        raise SafeError("MODEL_STOPPING_GROUPS_REQUIRED")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 0.5:
        raise SafeError("INVALID_MODEL_STOPPING_FRACTION")
    count = min(len(unique)-1, max(1, math.ceil(len(unique)*fraction)))
    stopping = set(np.random.default_rng(seed).permutation(unique)[:count])
    stop = series.isin(stopping).to_numpy()
    return ~stop, stop


class GroupedXGBoostRegressor(RegressorMixin, BaseEstimator):
    """Cloneable wrapper; raw group IDs are supplied separately from predictors.

    A stopped fit first learns preprocessing and an iteration count on a
    group-disjoint split of its own training population. It then refits fresh
    preprocessing and a fixed-count model on that entire training population.
    Prediction has no access to stopping, outer-validation or holdout labels.
    """

    def __init__(self, parameters, early_stopping_rounds=None, stopping_fraction=0.2):
        self.parameters = parameters
        self.early_stopping_rounds = early_stopping_rounds
        self.stopping_fraction = stopping_fraction

    def fit(self, X, y, sample_weight=None):
        if self.early_stopping_rounds is not None:
            raise SafeError("MODEL_GROUP_CONTEXT_REQUIRED")
        return self.fit_grouped(X, y, sample_weight=sample_weight)

    def fit_grouped(self, X, y, sample_weight=None, groups=None):
        from .study_modeling import _estimator
        model_class = require_xgboost()
        y = np.asarray(y, dtype=float)
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if (not isinstance(X, pd.DataFrame) or not len(y) or y.shape != (len(X),)
                or weights.shape != y.shape or not np.isfinite(y).all() or (y < 0).any()
                or not np.isfinite(weights).all() or (weights <= 0).any()):
            raise SafeError("INVALID_WEIGHTED_MODEL_INPUT")
        if groups is not None:
            group_series = pd.Series(groups).reset_index(drop=True)
            if len(group_series) != len(y) or group_series.isna().any() or group_series.astype(str).str.strip().eq("").any():
                raise SafeError("MODEL_VALID_TRAINING_GROUPS_REQUIRED")
            group_series = group_series.astype(str)
        else:
            group_series = None
        parameters = self.parameters.copy()
        if (parameters.get("tree_method") != "hist" or parameters.get("device") != "cpu"
                or parameters.get("n_jobs") != 1 or parameters.get("objective") != "reg:squarederror"
                or not isinstance(parameters.get("n_estimators"), int) or isinstance(parameters["n_estimators"], bool)
                or parameters["n_estimators"] < 1 or "callbacks" in parameters
                or "early_stopping_rounds" in parameters):
            raise SafeError("UNSUPPORTED_CPU_BOOSTING_CONFIGURATION")
        iterations = parameters["n_estimators"]
        groups_all = sorted(group_series.unique()) if group_series is not None else []
        details = {"iteration_policy": "fixed_budget", "iterations": iterations,
                   "probe_iterations": 0, "fitting_groups": groups_all, "stopping_groups": [],
                   "refit_groups": groups_all, "fitting_rows": len(y), "stopping_rows": 0,
                   "fitting_weight_sum": float(weights.sum()), "stopping_weight_sum": 0.,
                   "refit_rows": len(y), "refit_weight_sum": float(weights.sum())}
        # Repeated fits must not retain stopping state from a previous fit.
        for attribute in ("stopping_preprocessor_", "stopping_evaluations_", "training_details_", "estimator_"):
            self.__dict__.pop(attribute, None)
        if self.early_stopping_rounds is not None:
            if group_series is None:
                raise SafeError("MODEL_GROUP_CONTEXT_REQUIRED")
            if (isinstance(self.early_stopping_rounds, bool) or not isinstance(self.early_stopping_rounds, int)
                    or self.early_stopping_rounds < 1):
                raise SafeError("INVALID_MODEL_STOPPING_ROUNDS")
            fitting, stopping = grouped_stopping_split(group_series, self.stopping_fraction, parameters["random_state"])
            # Reuse precisely the same feature path, fitting it only on the
            # inner fitting groups. The stopping table is transformed once.
            preprocessor = _estimator(model_class(**parameters)).estimator[:-1]
            train_X = preprocessor.fit_transform(X.iloc[fitting], y[fitting])
            stop_X = preprocessor.transform(X.iloc[stopping])
            probe = model_class(**parameters, early_stopping_rounds=self.early_stopping_rounds)
            probe.fit(train_X, y[fitting], sample_weight=weights[fitting],
                      eval_set=[(stop_X, y[stopping])], sample_weight_eval_set=[weights[stopping]], verbose=False)
            iterations = int(probe.best_iteration)+1
            details.update(iteration_policy="inner_grouped_best_iteration_then_refit_all_training",
                           iterations=iterations, probe_iterations=int(probe.get_booster().num_boosted_rounds()),
                           fitting_groups=sorted(group_series.loc[fitting].unique()),
                           stopping_groups=sorted(group_series.loc[stopping].unique()),
                           fitting_rows=int(fitting.sum()), stopping_rows=int(stopping.sum()),
                           fitting_weight_sum=float(weights[fitting].sum()),
                           stopping_weight_sum=float(weights[stopping].sum()))
            self.stopping_preprocessor_ = preprocessor
            self.stopping_evaluations_ = probe.evals_result()
        parameters["n_estimators"] = iterations
        self.estimator_ = _estimator(model_class(**parameters)).fit(X, y, sample_weight=weights)
        self.training_details_ = details
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)
