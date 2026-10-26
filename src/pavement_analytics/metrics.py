"""Interval-weighted errors and deterministic group-disjoint folds."""

from __future__ import annotations

import numpy as np

from .security import SafeError


def score(y, prediction, weights) -> dict[str, float | None]:
    y, prediction, weights = (np.asarray(v, float) for v in (y, prediction, weights))
    if (y.ndim != 1 or y.shape != prediction.shape or y.shape != weights.shape or not len(y)
            or not all(np.isfinite(v).all() for v in (y, prediction, weights))
            or (weights <= 0).any()):
        raise SafeError("INVALID_METRIC_INPUT")
    error = y - prediction
    variance = np.average((y - np.average(y, weights=weights)) ** 2, weights=weights)
    mse = np.average(error ** 2, weights=weights)
    return {"wMAE": float(np.average(np.abs(error), weights=weights)),
            "wRMSE": float(np.sqrt(mse)),
            "weighted_R2": float(1 - mse / variance) if variance > 0 else None}


def grouped_folds(groups, folds: int = 5, seed: int = 42):
    values = np.asarray(groups, dtype=str)
    unique, counts = np.unique(values, return_counts=True)
    if folds < 2 or len(unique) < folds or np.any(unique == ""):
        raise SafeError("INSUFFICIENT_VALID_GROUPS")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-counts[order], kind="stable")]
    loads = np.zeros(folds, dtype=int)
    allocation = {}
    for index in order:
        slot = int(np.argmin(loads))
        allocation[unique[index]] = slot
        loads[slot] += counts[index]
    assignment = np.array([allocation[g] for g in values])
    for slot in range(folds):
        yield np.flatnonzero(assignment != slot), np.flatnonzero(assignment == slot)


def condition_bins(pci):
    values = np.asarray(pci, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
        raise SafeError("INVALID_START_CONDITION")
    # [0,40), [40,55), [55,70), [70,85), [85,100].
    return np.searchsorted([40, 55, 70, 85], values, side="right")


def condition_predict(train_pci, y, weights, test_pci):
    y, weights = np.asarray(y, float), np.asarray(weights, float)
    bins = condition_bins(train_pci)
    fallback = np.average(y, weights=weights)
    means = [np.average(y[bins == b], weights=weights[bins == b]) if np.any(bins == b) else fallback
             for b in range(5)]
    return np.asarray(means)[condition_bins(test_pci)]
