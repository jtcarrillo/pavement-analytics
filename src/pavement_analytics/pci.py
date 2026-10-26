"""Curve-driven PCI calculation; no curve data or licensed tables are bundled.

The caller supplies an approved profile and validates it against its reference
implementation. Curves must cover every evaluated total; no extrapolation or
assumption that independent deducts can simply be added is permitted.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from .security import SafeError


def validate_profile(profile: dict) -> None:
    if set(profile) != {"name", "q_threshold", "m_denominator", "m_cap", "curves"}:
        raise SafeError("INVALID_CURVE_PROFILE")
    for key in ["q_threshold", "m_denominator", "m_cap"]:
        if not isinstance(profile[key], (float, int)) or not math.isfinite(profile[key]):
            raise SafeError("INVALID_CURVE_PROFILE")
    if not (0 < profile["q_threshold"] < 100 and profile["m_denominator"] > 0
            and 1 <= profile["m_cap"] <= 10 and float(profile["m_cap"]).is_integer()):
        raise SafeError("INVALID_CURVE_PROFILE")
    if not isinstance(profile["curves"], dict) or not profile["name"]:
        raise SafeError("INVALID_CURVE_PROFILE")
    for key, points in profile["curves"].items():
        if not key.isdigit() or not 2 <= int(key) <= 10:
            raise SafeError("INVALID_CURVE_Q")
        try:
            data = np.asarray(points, dtype=float)
        except (TypeError, ValueError):
            raise SafeError("INVALID_CURVE_POINTS") from None
        if (data.ndim != 2 or data.shape[1] != 2 or len(data) < 2
                or not np.isfinite(data).all() or (data < 0).any()
                or not (np.diff(data[:, 0]) > 0).all()
                or not (np.diff(data[:, 1]) >= 0).all() or (data[:, 1] > 100).any()):
            raise SafeError("INVALID_CURVE_POINTS")


def calculate(deducts, profile: dict | None = None) -> float:
    values = np.asarray(deducts, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
        raise SafeError("INVALID_DEDUCT_VALUES")
    values = sorted(values[values > 0], reverse=True)
    if not values:
        return 100.0
    if len(values) == 1:
        return 100.0 - float(values[0])
    if profile is None:
        raise SafeError("APPROVED_CDV_PROFILE_REQUIRED")
    validate_profile(profile)
    threshold = profile["q_threshold"]
    if sum(v > threshold for v in values) <= 1:
        return max(0.0, 100.0 - sum(values))
    m = min(profile["m_cap"], 1 + 9 * (100 - values[0]) / profile["m_denominator"])
    integer = int(math.floor(m))
    current = values[:integer]
    if integer < len(values) and m > integer:
        current.append(values[integer] * (m - integer))
    maximum = 0.0
    while True:
        q = sum(v > threshold for v in current)
        total = sum(current)
        if q <= 1:
            corrected = min(100.0, total)
        else:
            if str(q) not in profile["curves"]:
                raise SafeError("CURVE_Q_NOT_COVERED")
            points = np.asarray(profile["curves"][str(q)], dtype=float)
            if not points[0, 0] <= total <= points[-1, 0]:
                raise SafeError("CURVE_DOMAIN_NOT_COVERED")
            corrected = float(np.interp(total, points[:, 0], points[:, 1]))
        maximum = max(maximum, corrected)
        if q <= 1:
            return max(0.0, 100.0 - maximum)
        index = min((i for i, v in enumerate(current) if v > threshold), key=lambda i: current[i])
        current[index] = threshold


def shapley_loss(families: dict[str, list[float]], profile: dict | None = None) -> dict[str, float]:
    """Exact Shapley decomposition of PCI loss, limited to twelve active families."""
    active = [k for k, values in families.items() if any(v > 0 for v in values)]
    if len(active) > 12:
        raise SafeError("TOO_MANY_FAMILIES_FOR_EXACT_SHAPLEY")
    result = dict.fromkeys(families, 0.0)
    n = len(active)
    loss = {}
    for mask in range(1 << n):
        values = [v for i, key in enumerate(active) if mask & (1 << i) for v in families[key]]
        loss[mask] = 100.0 - calculate(values, profile)
    for i, key in enumerate(active):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for subset in combinations(others, size):
                mask = sum(1 << j for j in subset)
                result[key] += weight * (loss[mask | (1 << i)] - loss[mask])
    return result
