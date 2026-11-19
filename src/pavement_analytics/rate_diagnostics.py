"""Describe signed PCI rates and saved holdout errors by inspection duration.

This module does not fit models, change observations or select an interval cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .security import SafeError


DURATION_BANDS = ("<=1 day", ">1 to <30 days", "30 to <90 days", "90 to <365 days", "365+ days")


def _row_ids(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if (not np.isfinite(numeric).all() or (numeric % 1 != 0).any()
            or numeric.duplicated().any()):
        raise SafeError("RATE_DIAGNOSTICS_ROW_IDS_INVALID")
    return numeric


def rate_interval_diagnostics(
    table: pd.DataFrame,
    plan: pd.DataFrame,
    predictions: pd.DataFrame | None,
    surface: str,
) -> list[dict]:
    """Return ten descriptive records: two partitions by five duration bands.

    ``table`` is one surface's modeling table, with row_id, group_id,
    delta_days and DetRatePerYear. ``plan`` maps unique group_id values to
    development/holdout partitions. ``predictions`` is the selected model's
    saved holdout table with row_id, actual, weight and prediction; None or an
    empty table is allowed. Inputs are never modified.

    Finite rates are eligible; intervals must already be positive and finite.
    The bands cover (0, 1], (1, 30), [30, 90), [90, 365) and [365, infinity).
    Actual-rate medians are unweighted descriptive medians. All scores use the
    saved interval weights, checked against delta_days. A partial prediction
    set leaves every holdout score unreported. Weighted-SSE shares use the
    complete surface holdout as denominator and are undefined for zero error.
    """
    required = {"row_id", "group_id", "delta_days", "DetRatePerYear"}
    if not required <= set(table) or not {"group_id", "partition"} <= set(plan):
        raise SafeError("RATE_DIAGNOSTICS_SCHEMA_INVALID")
    if (plan.group_id.isna().any() or plan.group_id.astype(str).str.strip().eq("").any()
            or plan.group_id.duplicated().any()
            or not plan.partition.isin(["development", "holdout"]).all()):
        raise SafeError("RATE_DIAGNOSTICS_PARTITIONS_INVALID")
    data = table.loc[:, sorted(required)].copy()
    data["row_id"] = _row_ids(data.row_id)
    data["partition"] = data.group_id.map(plan.set_index("group_id").partition)
    if data.partition.isna().any():
        raise SafeError("RATE_DIAGNOSTICS_PARTITIONS_INVALID")
    data["delta_days"] = pd.to_numeric(data.delta_days, errors="coerce")
    if not np.isfinite(data.delta_days).all() or data.delta_days.le(0).any():
        raise SafeError("RATE_DIAGNOSTICS_INTERVALS_INVALID")
    data["DetRatePerYear"] = pd.to_numeric(data.DetRatePerYear, errors="coerce")
    data = data.loc[np.isfinite(data.DetRatePerYear)].copy()
    data["duration_band"] = np.select(
        [data.delta_days.le(1), data.delta_days.lt(30), data.delta_days.lt(90), data.delta_days.lt(365)],
        DURATION_BANDS[:-1], default=DURATION_BANDS[-1],
    )
    holdout = data.loc[data.partition.eq("holdout")].set_index("row_id")
    saved = pd.DataFrame(columns=["row_id", "actual", "weight", "prediction"])
    status = "no_predictions"
    total_sse = None
    if predictions is not None and not predictions.empty:
        columns = {"row_id", "actual", "weight", "prediction"}
        if not columns <= set(predictions):
            raise SafeError("RATE_DIAGNOSTICS_PREDICTIONS_SCHEMA_INVALID")
        saved = predictions.loc[:, sorted(columns)].copy()
        saved["row_id"] = _row_ids(saved.row_id)
        if not saved.row_id.isin(holdout.index).all():
            raise SafeError("RATE_DIAGNOSTICS_PREDICTION_ROWS_INVALID")
        for column in ("actual", "weight", "prediction"):
            saved[column] = pd.to_numeric(saved[column], errors="coerce")
        if (not np.isfinite(saved[["actual", "weight", "prediction"]]).all().all()
                or saved.weight.le(0).any()):
            raise SafeError("RATE_DIAGNOSTICS_PREDICTIONS_INVALID")
        original = holdout.loc[saved.row_id]
        if not np.allclose(saved.actual, original.DetRatePerYear, rtol=1e-12, atol=1e-12):
            raise SafeError("RATE_DIAGNOSTICS_ACTUAL_MISMATCH")
        if not np.allclose(saved.weight, original.delta_days, rtol=1e-12, atol=1e-12):
            raise SafeError("RATE_DIAGNOSTICS_WEIGHT_MISMATCH")
        status = "scored" if len(saved) == len(holdout) else "incomplete_predictions"
        if status == "scored":
            saved["weighted_squared_error"] = saved.weight * (saved.prediction - saved.actual) ** 2
            saved["weighted_absolute_error"] = saved.weight * abs(saved.prediction - saved.actual)
            total_sse = float(saved.weighted_squared_error.sum())
            if not np.isfinite(total_sse):
                raise SafeError("RATE_DIAGNOSTICS_ERRORS_NONFINITE")
    records = []
    for partition in ("development", "holdout"):
        for band in DURATION_BANDS:
            subset = data.loc[data.partition.eq(partition) & data.duration_band.eq(band)]
            scored = saved.loc[saved.row_id.isin(subset.row_id)] if partition == "holdout" else saved.iloc[:0]
            record = {
                "material": surface, "partition": partition, "duration_band": band,
                "eligible_rows": len(subset), "groups": int(subset.group_id.nunique()),
                "min_actual_rate": float(subset.DetRatePerYear.min()) if len(subset) else None,
                "median_actual_rate": float(subset.DetRatePerYear.median()) if len(subset) else None,
                "weight_sum": float(subset.delta_days.sum()),
                "prediction_rows": len(scored),
                "prediction_status": "no_eligible_rows" if subset.empty else
                    "not_evaluated" if partition == "development" else status,
                "weighted_rmse": None, "weighted_mae": None,
                "weighted_sse": None, "weighted_sse_share": None,
            }
            if partition == "holdout" and status == "scored" and len(scored):
                sse = float(scored.weighted_squared_error.sum())
                record.update({
                    "weighted_rmse": float(np.sqrt(sse / scored.weight.sum())),
                    "weighted_mae": float(scored.weighted_absolute_error.sum() / scored.weight.sum()),
                    "weighted_sse": sse,
                    "weighted_sse_share": sse / total_sse if total_sse else None,
                })
            records.append(record)
    return records
