"""Gaussian identity-link additive splines with a weighted Ridge penalty.

This is an additive spline implementation, not a proprietary GAM blueprint or
pyGAM smoothing model. Feature components describe the margin before clipping.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.utils.validation import check_is_fitted

from .security import SafeError


def _categorical_role(name, series, declared):
    token = re.sub(r"[^a-z0-9]", "", str(name).lower()).removesuffix("start")
    return (name in declared or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or token in {"subgradecategory", "inspectedbranchuse", "branchuse", "isconstruction"}
            or token.endswith("code"))


class AdditiveSplineFeatures(TransformerMixin, BaseEstimator):
    """Learn one independent feature block per raw predictor, within training."""

    def __init__(self, n_knots=5, degree=3, extrapolation="constant", knots="uniform", categorical_features=()):
        self.n_knots, self.degree = n_knots, degree
        self.extrapolation, self.knots = extrapolation, knots
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        from .study_modeling import FeatureCleaner
        if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
            raise SafeError("INVALID_SPLINE_FEATURE_TABLE")
        self.cleaner_ = FeatureCleaner().fit(X)
        self.cleaner_.numeric_ = [name for name in self.cleaner_.numeric_
                                 if not _categorical_role(name, X[name], self.categorical_features)]
        cleaned = self.cleaner_.transform(X)
        self.columns_ = list(X.columns)
        self.numeric_ = self.cleaner_.numeric_.copy()
        self.categorical_ = [name for name in self.columns_ if name not in self.numeric_]
        self.imputer_ = None
        if self.numeric_:
            self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(cleaned[self.numeric_])
            imputed = self.imputer_.transform(cleaned[self.numeric_])
        self.splines_, self.encoders_, self.bounds_, self.constant_numeric_ = {}, {}, {}, []
        self.feature_slices_, self.feature_names_ = {}, []
        for name in self.columns_:
            start = len(self.feature_names_)
            if name in self.numeric_:
                index = self.numeric_.index(name)
                values = imputed[:, index:index+1]
                self.bounds_[name] = [float(values.min()), float(values.max())]
                if np.ptp(values) > 0:
                    spline = SplineTransformer(n_knots=self.n_knots, degree=self.degree, knots=self.knots,
                                               extrapolation=self.extrapolation, include_bias=False)
                    self.splines_[name] = spline.fit(values)
                    self.feature_names_.extend(f"{name}__spline_{i}" for i in range(spline.n_features_out_))
                else:
                    self.constant_numeric_.append(name)
                # Always a separate, linear missingness term, never splined.
                self.feature_names_.append(f"{name}__missing")
            else:
                encoder = Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
                    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=64)),
                ]).fit(cleaned[[name]])
                self.encoders_[name] = encoder
                width = len(encoder.named_steps["encode"].get_feature_names_out([name]))
                self.feature_names_.extend(f"{name}__category_{i}" for i in range(width))
            self.feature_slices_[name] = [start, len(self.feature_names_)]
        self.n_features_in_, self.n_features_out_ = len(self.columns_), len(self.feature_names_)
        self.feature_names_in_ = np.asarray(self.columns_, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_slices_")
        cleaned = self.cleaner_.transform(X)
        imputed = self.imputer_.transform(cleaned[self.numeric_]) if self.numeric_ else None
        parts = []
        for name in self.columns_:
            if name in self.numeric_:
                if name in self.splines_:
                    index = self.numeric_.index(name)
                    parts.append(self.splines_[name].transform(imputed[:, index:index+1]))
                parts.append(cleaned[name].isna().to_numpy(float).reshape(-1, 1))
            else:
                parts.append(self.encoders_[name].transform(cleaned[[name]]))
        return np.column_stack(parts)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)

    def schema(self):
        check_is_fitted(self, "feature_slices_")
        categories, infrequent = {}, {}
        for name, pipeline in self.encoders_.items():
            encoder = pipeline.named_steps["encode"]
            categories[name] = encoder.categories_[0].tolist()
            rare = encoder.infrequent_categories_[0]
            infrequent[name] = [] if rare is None else rare.tolist()
        return {"feature_order": self.columns_, "numeric_features": self.numeric_,
                "categorical_features": self.categorical_, "constant_numeric_features": self.constant_numeric_,
                "imputation_values": dict(zip(self.numeric_, self.imputer_.statistics_.tolist())) if self.numeric_ else {},
                "training_bounds": self.bounds_, "encoded_columns": self.feature_names_,
                "feature_slices": self.feature_slices_, "category_mapping": categories,
                "infrequent_category_mapping": infrequent,
                "spline_knots": {name: spline.bsplines_[0].t.tolist() for name, spline in self.splines_.items()}}


class AdditiveSplineRegressor(RegressorMixin, BaseEstimator):
    """Nonnegative clipped sum of univariate splines and additive category terms."""

    def __init__(self, n_knots=5, degree=3, alpha=0.1, extrapolation="constant", knots="uniform",
                 solver="cholesky", categorical_features=()):
        self.n_knots, self.degree, self.alpha = n_knots, degree, alpha
        self.extrapolation, self.knots, self.solver = extrapolation, knots, solver
        self.categorical_features = categorical_features

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if (not isinstance(X, pd.DataFrame) or not len(y) or y.shape != (len(X),) or weights.shape != y.shape
                or not np.isfinite(y).all() or (y < 0).any() or not np.isfinite(weights).all() or (weights <= 0).any()):
            raise SafeError("INVALID_WEIGHTED_MODEL_INPUT")
        if (isinstance(self.n_knots, bool) or not isinstance(self.n_knots, int) or self.n_knots < 2
                or self.degree != 3 or self.extrapolation != "constant" or self.knots != "uniform"
                or self.solver != "cholesky" or isinstance(self.alpha, bool)
                or not isinstance(self.alpha, (int, float)) or not np.isfinite(self.alpha) or self.alpha <= 0
                or not isinstance(self.categorical_features, (list, tuple))
                or any(not isinstance(name, str) for name in self.categorical_features)):
            raise SafeError("UNSUPPORTED_ADDITIVE_SPLINE_CONFIGURATION")
        self.features_ = AdditiveSplineFeatures(self.n_knots, self.degree, self.extrapolation,
                                               self.knots, self.categorical_features).fit(X)
        encoded = self.features_.transform(X)
        self.estimator_ = Ridge(alpha=self.alpha, solver=self.solver).fit(encoded, y, sample_weight=weights)
        self.n_features_in_, self.feature_names_in_ = X.shape[1], np.asarray(X.columns, dtype=object)
        self.training_details_ = {"iteration_policy": "closed_form_weighted_ridge", "iterations": None,
                                  "probe_iterations": 0, "fitting_rows": len(y), "refit_rows": len(y),
                                  "fitting_weight_sum": float(weights.sum()), "refit_weight_sum": float(weights.sum()),
                                  "stopping_groups": [], "stopping_rows": 0, "alpha": self.alpha,
                                  "n_knots": self.n_knots, "degree": self.degree, "extrapolation": self.extrapolation,
                                  "knots": self.knots, "feature_schema": self.features_.schema()}
        return self

    def fit_grouped(self, X, y, sample_weight=None, groups=None):
        if groups is not None:
            series = pd.Series(groups).reset_index(drop=True)
            if len(series) != len(X) or series.isna().any() or series.astype(str).str.strip().eq("").any():
                raise SafeError("MODEL_VALID_TRAINING_GROUPS_REQUIRED")
            members = sorted(series.astype(str).unique())
        else:
            members = []
        self.fit(X, y, sample_weight=sample_weight)
        self.training_details_.update(fitting_groups=members, refit_groups=members)
        return self

    def predict_margin(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(self.features_.transform(X))

    def predict(self, X):
        return np.maximum(0, self.predict_margin(X))

    def predict_components(self, X):
        """Private additive terms; centering/association does not imply causality."""
        check_is_fitted(self, "estimator_")
        encoded = self.features_.transform(X)
        terms = {name: encoded[:, start:stop] @ self.estimator_.coef_[start:stop]
                 for name, (start, stop) in self.features_.feature_slices_.items()}
        return {"intercept": float(self.estimator_.intercept_), "terms": pd.DataFrame(terms, index=X.index)}
