"""Versioned, cloneable candidate definitions for observed deduct regression.

The registry describes implemented models only. Optional dependency discovery
does not install packages, load data or probe an external training service.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib.metadata
import importlib.util
import json
import re
import traceback
from typing import Callable

from packaging.version import InvalidVersion, Version

from .security import SafeError


REGISTRY_VERSION = 8
XGBOOST_MIN_VERSION, XGBOOST_MAX_VERSION = "3.1.3", "3.2"
LIGHTGBM_MIN_VERSION, LIGHTGBM_MAX_VERSION = "4.6.0", "4.7"
BASELINE_ID = "median"
TARGET_KINDS = ("deduct_total", "signed_pci_rate")
BASELINE_IDS = {"median", "mean"}
ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,79}")


def validate_target_kind(target_kind):
    if not isinstance(target_kind, str) or target_kind not in TARGET_KINDS:
        raise SafeError("INVALID_MODEL_TARGET_KIND")
    return target_kind


def _point_estimator(model_class, parameters):
    from .study_modeling import _estimator
    parameters = parameters.copy()
    target_kind = parameters.pop("target_kind", "deduct_total")
    return _estimator(model_class(**parameters), target_kind=target_kind)


def _elastic(parameters):
    from sklearn.linear_model import ElasticNet
    return _point_estimator(ElasticNet, parameters)


def _forest(parameters):
    from sklearn.ensemble import RandomForestRegressor
    return _point_estimator(RandomForestRegressor, parameters)


def _extra_trees(parameters):
    from sklearn.ensemble import ExtraTreesRegressor
    return _point_estimator(ExtraTreesRegressor, parameters)


def _xgboost(parameters):
    from .boosting import GroupedXGBoostRegressor, require_xgboost
    require_xgboost()
    parameters = parameters.copy()
    target_kind = parameters.pop("target_kind", "deduct_total")
    rounds = parameters.pop("early_stopping_rounds")
    fraction = parameters.pop("stopping_fraction")
    return GroupedXGBoostRegressor(parameters, rounds, fraction, target_kind=target_kind)


def dependency_version(package, minimum=None, maximum=None):
    """Return installed version and range support without importing a backend."""
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None, False
    try:
        version = Version(installed)
        supported = (not version.is_prerelease and not version.is_devrelease
                     and (minimum is None or version >= Version(minimum))
                     and (maximum is None or version < Version(maximum)))
    except InvalidVersion:
        supported = False
    return installed, supported


def _lightgbm(parameters):
    from .lightgbm_model import GroupedLightGBMRegressor, require_lightgbm
    require_lightgbm()
    parameters = parameters.copy()
    target_kind = parameters.pop("target_kind", "deduct_total")
    rounds = parameters.pop("early_stopping_rounds")
    fraction = parameters.pop("stopping_fraction")
    return GroupedLightGBMRegressor(parameters, rounds, fraction, target_kind=target_kind)


def _additive_spline(parameters):
    from .spline_model import AdditiveSplineRegressor
    return AdditiveSplineRegressor(**parameters)


def _two_stage(parameters):
    from sklearn.linear_model import LogisticRegression
    from .study_modeling import _estimator
    from .two_stage import OccurrenceSeverityRegressor
    occurrence = _estimator(LogisticRegression(C=parameters["occurrence_C"], solver=parameters["occurrence_solver"],
                            max_iter=parameters["occurrence_max_iter"], tol=parameters["occurrence_tol"],
                            random_state=parameters["random_state"])).estimator
    severity = _extra_trees({"n_estimators": parameters["severity_estimators"], "criterion": parameters["severity_criterion"],
                            "min_samples_leaf": parameters["severity_leaf"], "max_depth": parameters["severity_depth"],
                            "max_features": parameters["severity_max_features"], "bootstrap": parameters["severity_bootstrap"],
                            "n_jobs": parameters["n_jobs"],
                            "random_state": parameters["random_state"]})
    return OccurrenceSeverityRegressor(occurrence, severity,
        min_class_rows=parameters["min_class_rows"], min_class_groups=parameters["min_class_groups"],
        min_positive_rows=parameters["min_positive_rows"], min_positive_groups=parameters["min_positive_groups"])


@dataclass(frozen=True)
class CandidateSpec:
    candidate: str
    family: str
    parameters: dict
    factory: Callable = field(repr=False, compare=False)
    required_package: str = "scikit-learn"
    import_name: str = "sklearn"
    supports_sample_weight: bool = True
    prediction_meaning: str = "nonnegative regression point estimate"
    max_workers: int = 1
    minimum_version: str | None = None
    maximum_version: str | None = None
    supported_target_kinds: tuple[str, ...] = TARGET_KINDS

    def metadata(self):
        if (not isinstance(self.candidate, str) or not ID_PATTERN.fullmatch(self.candidate)
                or self.candidate in BASELINE_IDS or not isinstance(self.family, str) or not self.family
                or not isinstance(self.supports_sample_weight, bool) or not callable(self.factory)
                or isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int) or self.max_workers < 1
                or not isinstance(self.import_name, str) or not self.import_name
                or not isinstance(self.supported_target_kinds, tuple) or not self.supported_target_kinds
                or not set(self.supported_target_kinds) <= set(TARGET_KINDS)):
            raise SafeError("INVALID_MODEL_CANDIDATE_DEFINITION")
        try:
            for bound in (self.minimum_version, self.maximum_version):
                if bound is not None:
                    if not isinstance(bound, str):
                        raise TypeError
                    Version(bound)
            if self.minimum_version and self.maximum_version and Version(self.minimum_version) >= Version(self.maximum_version):
                raise InvalidVersion("invalid version interval")
        except (InvalidVersion, TypeError):
            raise SafeError("INVALID_MODEL_CANDIDATE_DEFINITION") from None
        try:
            available = importlib.util.find_spec(self.import_name) is not None
        except (ValueError, ImportError, AttributeError):
            available = False
        # Metadata is JSON rather than repr(estimator), which can contain
        # arbitrary data or private objects in a custom implementation.
        try:
            parameters = json.loads(json.dumps(self.parameters, allow_nan=False))
        except (TypeError, ValueError):
            raise SafeError("INVALID_MODEL_CANDIDATE_PARAMETERS") from None
        if not isinstance(parameters, dict):
            raise SafeError("INVALID_MODEL_CANDIDATE_PARAMETERS")
        installed, supported = dependency_version(self.required_package, self.minimum_version, self.maximum_version)
        availability = "available" if available else "unavailable"
        if available and (self.minimum_version or self.maximum_version) and not supported:
            availability = "unsupported"
        return {"candidate": self.candidate, "family": self.family, "parameters": parameters,
                "required_package": self.required_package, "supports_sample_weight": self.supports_sample_weight,
                "prediction_meaning": self.prediction_meaning, "max_workers": self.max_workers,
                "availability": availability, "installed_version": installed,
                "minimum_version": self.minimum_version, "maximum_version": self.maximum_version,
                "supported_target_kinds": list(self.supported_target_kinds)}


def candidate_specs(seed: int = 42, target_kind="deduct_total"):
    """Fresh definitions preserve old candidates and bound each new grid."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SafeError("INVALID_MODEL_SEED")
    validate_target_kind(target_kind)
    definitions = (
        *(CandidateSpec(f"enet_{alpha}", "elastic_net", {"alpha": alpha, "l1_ratio": 0.5,
           "max_iter": 20000, "random_state": seed}, _elastic) for alpha in (0.01, 0.1)),
        *(CandidateSpec(f"rf_{leaf}", "random_forest", {"n_estimators": 100, "min_samples_leaf": leaf,
           "max_depth": 12, "max_features": 0.8, "n_jobs": 1, "random_state": seed}, _forest)
          for leaf in (5, 20)),
        *(CandidateSpec(f"et_leaf{leaf}_depth{depth if depth is not None else 'none'}", "extra_trees",
           {"n_estimators": 200, "criterion": "squared_error", "min_samples_leaf": leaf,
            "max_depth": depth, "max_features": 1.0, "bootstrap": False,
            "n_jobs": 1, "random_state": seed}, _extra_trees)
          for leaf in (2, 10) for depth in (12, None)),
        *(CandidateSpec(f"xgb_depth{depth}_lr{rate}{'_es' if stopping else ''}", "xgboost",
           {"n_estimators": 300, "max_depth": depth, "learning_rate": rate,
            "objective": "reg:squarederror", "eval_metric": "mae", "tree_method": "hist",
            "device": "cpu", "subsample": 1.0, "colsample_bytree": 1.0,
            "n_jobs": 1, "random_state": seed, "verbosity": 0,
            "early_stopping_rounds": 20 if stopping else None, "stopping_fraction": 0.2},
           _xgboost, required_package="xgboost", import_name="xgboost",
           minimum_version=XGBOOST_MIN_VERSION, maximum_version=XGBOOST_MAX_VERSION)
          for stopping in (False, True) for depth in (3, 6) for rate in (0.05, 0.1)),
        *(CandidateSpec(f"lgbm_leaves{leaves}_min{minimum}{'_es' if stopping else ''}", "lightgbm",
           {"n_estimators": 300, "num_leaves": leaves, "min_child_samples": minimum,
            "max_depth": -1, "learning_rate": 0.05, "objective": "regression", "metric": "l1",
            "boosting_type": "gbdt", "device_type": "cpu", "n_jobs": 1,
            "deterministic": True, "force_col_wise": True, "subsample": 1.0,
            "colsample_bytree": 1.0, "random_state": seed, "verbosity": -1,
            "early_stopping_rounds": 20 if stopping else None, "stopping_fraction": 0.2},
           _lightgbm, required_package="lightgbm", import_name="lightgbm",
           minimum_version=LIGHTGBM_MIN_VERSION, maximum_version=LIGHTGBM_MAX_VERSION)
          for stopping in (False, True) for leaves in (15, 31) for minimum in (10, 30)),
        *(CandidateSpec(f"spline_knots{knots}_alpha{alpha}", "additive_spline",
           {"n_knots": knots, "degree": 3, "alpha": alpha, "extrapolation": "constant",
            "knots": "uniform", "solver": "cholesky"}, _additive_spline,
           prediction_meaning="nonnegative clipped additive Gaussian point estimate")
          for knots in (5, 8) for alpha in (0.1, 1.0)),
        *(CandidateSpec(f"two_stage_c{c}_leaf{leaf}", "two_stage",
           {"occurrence_C": c, "severity_leaf": leaf, "random_state": seed,
            "occurrence_solver": "lbfgs", "occurrence_max_iter": 1000, "occurrence_tol": 0.0001,
            "severity_estimators": 200, "severity_depth": 12, "severity_max_features": 1.0,
            "severity_criterion": "squared_error", "severity_bootstrap": False, "n_jobs": 1,
            "min_class_rows": 5, "min_class_groups": 2, "min_positive_rows": 10, "min_positive_groups": 2},
           _two_stage, prediction_meaning="conditional mean: P(Y>0|X) times positive conditional mean; not a median",
           supported_target_kinds=("deduct_total",))
          for c in (0.1, 1.0) for leaf in (2, 10)),
    )
    if target_kind == "deduct_total":
        return definitions
    signed = []
    for spec in definitions:
        if target_kind not in spec.supported_target_kinds:
            signed.append(spec)
            continue
        parameters = {**spec.parameters, "target_kind": target_kind}
        if spec.family == "xgboost":
            parameters["eval_metric"] = "rmse"
        elif spec.family == "lightgbm":
            parameters["metric"] = "rmse"
        signed.append(replace(spec, parameters=parameters,
                      prediction_meaning="signed PCI change per year; negative means deterioration"))
    return tuple(signed)


def baseline_metadata(target_kind="deduct_total"):
    validate_target_kind(target_kind)
    signed = target_kind == "signed_pci_rate"
    return {"candidate": "mean" if signed else BASELINE_ID,
            "family": "weighted_mean" if signed else "weighted_median", "parameters": {},
            "required_package": "base", "supports_sample_weight": True,
            "prediction_meaning": "training weighted-mean constant" if signed else "training weighted-median constant",
            "max_workers": 1, "availability": "available", "target_kind": target_kind,
            "supported_target_kinds": list(TARGET_KINDS)}


def candidate_catalog(seed=42, target_kind="deduct_total"):
    return [baseline_metadata(target_kind), *[dict(spec.metadata(), target_kind=target_kind)
            for spec in candidate_specs(seed, target_kind)]]


def metadata_for(candidate: str, seed=42, target_kind="deduct_total"):
    for entry in candidate_catalog(seed, target_kind):
        if entry["candidate"] == candidate:
            return entry
    return {"candidate": candidate, "family": "custom", "parameters": {},
            "required_package": "custom", "supports_sample_weight": True,
            "prediction_meaning": "custom nonnegative point estimate", "max_workers": 1,
            "availability": "available", "target_kind": target_kind}


class CandidateCollection(dict):
    """A compatible estimator mapping plus structured construction outcomes."""
    def __init__(self):
        super().__init__()
        self.entries = []
        self.diagnostics = {}


def build_candidates(seed=42, requested=None, *, specs=None, target_kind="deduct_total"):
    validate_target_kind(target_kind)
    definitions = tuple(candidate_specs(seed, target_kind) if specs is None else specs)
    names = [entry.candidate for entry in definitions]
    if len(set(names)) != len(names):
        raise SafeError("DUPLICATE_MODEL_CANDIDATE_ID")
    if requested is not None:
        if (not isinstance(requested, (list, tuple)) or any(not isinstance(n, str) for n in requested)
                or len(requested) != len(set(requested))
                or not set(requested) <= set(names)|{baseline_metadata(target_kind)["candidate"]}):
            raise SafeError("UNKNOWN_OR_DUPLICATE_MODEL_CANDIDATE")
        definitions = tuple(spec for spec in definitions if spec.candidate in requested)
    result = CandidateCollection()
    for spec in definitions:
        record = dict(spec.metadata(), target_kind=target_kind)
        if target_kind not in spec.supported_target_kinds:
            record.update(candidate_status="ineligible_target", error_code="MODEL_TARGET_KIND_UNSUPPORTED")
        elif not spec.supports_sample_weight:
            record.update(candidate_status="unsupported", error_code="MODEL_SAMPLE_WEIGHTS_REQUIRED")
        elif record["availability"] == "unsupported":
            record.update(candidate_status="unsupported", error_code="MODEL_DEPENDENCY_VERSION_UNSUPPORTED")
        elif record["availability"] != "available":
            record.update(candidate_status="unavailable", error_code="MODEL_DEPENDENCY_UNAVAILABLE")
        else:
            try:
                result[spec.candidate] = spec.factory(record["parameters"])
                record.update(candidate_status="ready", error_code=None)
            except Exception:
                result.diagnostics[spec.candidate] = traceback.format_exc()
                record.update(candidate_status="failed_construct", error_code="MODEL_CONSTRUCTION_FAILED")
        result.entries.append(record)
        if requested is not None and record["candidate_status"] not in {"ready", "ineligible_target"}:
            raise SafeError(record["error_code"])
    return result


def environment_versions():
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "openpyxl", "packaging", "threadpoolctl", "xgboost", "lightgbm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions
