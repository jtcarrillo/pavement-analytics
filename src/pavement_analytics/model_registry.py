"""Versioned, cloneable candidate definitions for observed deduct regression.

The registry describes implemented models only. Optional dependency discovery
does not install packages, load data or probe an external training service.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
import importlib.util
import json
import re
import traceback
from typing import Callable

from .security import SafeError


REGISTRY_VERSION = 2
BASELINE_ID = "median"
ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,79}")


def _elastic(parameters):
    from sklearn.linear_model import ElasticNet
    from .study_modeling import _estimator
    return _estimator(ElasticNet(**parameters))


def _forest(parameters):
    from sklearn.ensemble import RandomForestRegressor
    from .study_modeling import _estimator
    return _estimator(RandomForestRegressor(**parameters))


def _extra_trees(parameters):
    from sklearn.ensemble import ExtraTreesRegressor
    from .study_modeling import _estimator
    return _estimator(ExtraTreesRegressor(**parameters))


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

    def metadata(self):
        if (not isinstance(self.candidate, str) or not ID_PATTERN.fullmatch(self.candidate)
                or self.candidate == BASELINE_ID or not isinstance(self.family, str) or not self.family
                or not isinstance(self.supports_sample_weight, bool) or not callable(self.factory)
                or isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int) or self.max_workers < 1
                or not isinstance(self.import_name, str) or not self.import_name):
            raise SafeError("INVALID_MODEL_CANDIDATE_DEFINITION")
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
        return {"candidate": self.candidate, "family": self.family, "parameters": parameters,
                "required_package": self.required_package, "supports_sample_weight": self.supports_sample_weight,
                "prediction_meaning": self.prediction_meaning, "max_workers": self.max_workers,
                "availability": "available" if available else "unavailable"}


def candidate_specs(seed: int = 42):
    """Fresh definitions preserve old candidates and bound each new grid."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SafeError("INVALID_MODEL_SEED")
    return (
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
    )


def baseline_metadata():
    return {"candidate": BASELINE_ID, "family": "weighted_median", "parameters": {},
            "required_package": "base", "supports_sample_weight": True,
            "prediction_meaning": "training weighted-median constant", "max_workers": 1,
            "availability": "available"}


def candidate_catalog(seed=42):
    return [baseline_metadata(), *[spec.metadata() for spec in candidate_specs(seed)]]


def metadata_for(candidate: str, seed=42):
    for entry in candidate_catalog(seed):
        if entry["candidate"] == candidate:
            return entry
    return {"candidate": candidate, "family": "custom", "parameters": {},
            "required_package": "custom", "supports_sample_weight": True,
            "prediction_meaning": "custom nonnegative point estimate", "max_workers": 1,
            "availability": "available"}


class CandidateCollection(dict):
    """A compatible estimator mapping plus structured construction outcomes."""
    def __init__(self):
        super().__init__()
        self.entries = []
        self.diagnostics = {}


def build_candidates(seed=42, requested=None, *, specs=None):
    definitions = tuple(candidate_specs(seed) if specs is None else specs)
    names = [entry.candidate for entry in definitions]
    if len(set(names)) != len(names):
        raise SafeError("DUPLICATE_MODEL_CANDIDATE_ID")
    if requested is not None:
        if (not isinstance(requested, (list, tuple)) or any(not isinstance(n, str) for n in requested)
                or len(requested) != len(set(requested)) or not set(requested) <= set(names)|{BASELINE_ID}):
            raise SafeError("UNKNOWN_OR_DUPLICATE_MODEL_CANDIDATE")
        definitions = tuple(spec for spec in definitions if spec.candidate in requested)
    result = CandidateCollection()
    for spec in definitions:
        record = spec.metadata()
        if not spec.supports_sample_weight:
            record.update(candidate_status="unsupported", error_code="MODEL_SAMPLE_WEIGHTS_REQUIRED")
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
        if requested is not None and record["candidate_status"] != "ready":
            raise SafeError(record["error_code"])
    return result


def environment_versions():
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "joblib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions
