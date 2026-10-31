"""Optional encoded LightGBM regression with training-only grouped stopping."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .boosting import grouped_stopping_split
from .model_registry import LIGHTGBM_MAX_VERSION, LIGHTGBM_MIN_VERSION, dependency_version
from .security import SafeError


def require_lightgbm():
    installed, supported = dependency_version("lightgbm", LIGHTGBM_MIN_VERSION, LIGHTGBM_MAX_VERSION)
    if installed is None:
        raise SafeError("MODEL_DEPENDENCY_UNAVAILABLE")
    if not supported:
        raise SafeError("MODEL_DEPENDENCY_VERSION_UNSUPPORTED")
    import lightgbm
    return lightgbm


def _encoded(preprocessor, frame, *, fitting=False, y=None):
    values = preprocessor.fit_transform(frame, y) if fitting else preprocessor.transform(frame)
    # Neutral names avoid native JSON-name restrictions on raw category labels
    # and preserve sklearn feature validation at prediction time.
    return pd.DataFrame(values, columns=[f"encoded_{i}" for i in range(values.shape[1])])


def _schema(preprocessor, encoded):
    prepare = preprocessor.named_steps["prepare"]
    categorical = prepare.named_transformers_["categorical"]
    columns = prepare.transformers_[1][2]
    encoder = categorical.named_steps["encode"]
    categories = {name: values.tolist() for name, values in zip(columns, getattr(encoder, "categories_", []))}
    infrequent = {name: values.tolist() if values is not None else []
                  for name, values in zip(columns, getattr(encoder, "infrequent_categories_", []))}
    return {"encoded_columns": list(encoded.columns),
            "transformed_columns": prepare.get_feature_names_out().tolist(), "category_mapping": categories,
            "infrequent_category_mapping": infrequent}


class GroupedLightGBMRegressor(RegressorMixin, BaseEstimator):
    """Use the shared feature pipeline and nested grouped iteration policy."""

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
        backend = require_lightgbm()
        y = np.asarray(y, dtype=float)
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if (not isinstance(X, pd.DataFrame) or not len(y) or y.shape != (len(X),)
                or weights.shape != y.shape or not np.isfinite(y).all() or (y < 0).any()
                or not np.isfinite(weights).all() or (weights <= 0).any()):
            raise SafeError("INVALID_WEIGHTED_MODEL_INPUT")
        group_series = None if groups is None else pd.Series(groups).reset_index(drop=True)
        if group_series is not None:
            if len(group_series) != len(y) or group_series.isna().any() or group_series.astype(str).str.strip().eq("").any():
                raise SafeError("MODEL_VALID_TRAINING_GROUPS_REQUIRED")
            group_series = group_series.astype(str)
        parameters = self.parameters.copy()
        forbidden = {"callbacks", "early_stopping_rounds", "early_stopping_round", "device", "num_threads", "nthread"}
        if (parameters.get("device_type") != "cpu" or parameters.get("n_jobs") != 1
                or parameters.get("objective") != "regression" or parameters.get("boosting_type") != "gbdt"
                or parameters.get("metric") != "l1" or not parameters.get("deterministic")
                or not parameters.get("force_col_wise") or forbidden & set(parameters)
                or not isinstance(parameters.get("n_estimators"), int) or isinstance(parameters["n_estimators"], bool)
                or parameters["n_estimators"] < 1):
            raise SafeError("UNSUPPORTED_CPU_BOOSTING_CONFIGURATION")
        requested = parameters["n_estimators"]
        groups_all = sorted(group_series.unique()) if group_series is not None else []
        details = {"iteration_policy": "fixed_budget", "probe_iterations": 0,
                   "fitting_groups": groups_all, "stopping_groups": [], "refit_groups": groups_all,
                   "fitting_rows": len(y), "stopping_rows": 0, "refit_rows": len(y),
                   "fitting_weight_sum": float(weights.sum()), "stopping_weight_sum": 0.,
                   "refit_weight_sum": float(weights.sum()), "feature_order": list(X.columns)}
        for attribute in ("stopping_preprocessor_", "stopping_evaluations_", "training_details_", "estimator_", "preprocessor_"):
            self.__dict__.pop(attribute, None)
        if self.early_stopping_rounds is not None:
            if group_series is None:
                raise SafeError("MODEL_GROUP_CONTEXT_REQUIRED")
            if (isinstance(self.early_stopping_rounds, bool) or not isinstance(self.early_stopping_rounds, int)
                    or self.early_stopping_rounds < 1):
                raise SafeError("INVALID_MODEL_STOPPING_ROUNDS")
            fitting, stopping = grouped_stopping_split(group_series, self.stopping_fraction, parameters["random_state"])
            preprocessor = _estimator(backend.LGBMRegressor(**parameters)).estimator[:-1]
            train_X = _encoded(preprocessor, X.iloc[fitting], fitting=True, y=y[fitting])
            stop_X = _encoded(preprocessor, X.iloc[stopping])
            probe = backend.LGBMRegressor(**parameters)
            probe.fit(train_X, y[fitting], sample_weight=weights[fitting],
                      eval_set=[(stop_X, y[stopping])], eval_sample_weight=[weights[stopping]],
                      callbacks=[backend.early_stopping(self.early_stopping_rounds, first_metric_only=True, verbose=False)])
            requested = int(probe.best_iteration_)
            if not 1 <= requested <= parameters["n_estimators"]:
                raise SafeError("INVALID_MODEL_STOPPING_ITERATION")
            details.update(iteration_policy="inner_grouped_best_iteration_then_refit_all_training",
                           probe_iterations=len(probe.evals_result_["valid_0"]["l1"]),
                           probe_model_iterations=int(probe.booster_.current_iteration()),
                           fitting_groups=sorted(group_series.loc[fitting].unique()),
                           stopping_groups=sorted(group_series.loc[stopping].unique()),
                           fitting_rows=int(fitting.sum()), stopping_rows=int(stopping.sum()),
                           fitting_weight_sum=float(weights[fitting].sum()), stopping_weight_sum=float(weights[stopping].sum()),
                           stopping_feature_schema=_schema(preprocessor, train_X))
            self.stopping_preprocessor_ = preprocessor
            self.stopping_evaluations_ = probe.evals_result_
        parameters["n_estimators"] = requested
        self.preprocessor_ = _estimator(backend.LGBMRegressor(**parameters)).estimator[:-1]
        train_X = _encoded(self.preprocessor_, X, fitting=True, y=y)
        self.estimator_ = backend.LGBMRegressor(**parameters).fit(train_X, y, sample_weight=weights)
        details.update(iterations=int(self.estimator_.n_estimators_), requested_iterations=requested,
                       final_feature_schema=_schema(self.preprocessor_, train_X))
        self.training_details_ = details
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        check_is_fitted(self, ("estimator_", "preprocessor_"))
        return np.maximum(0, self.estimator_.predict(_encoded(self.preprocessor_, X)))
