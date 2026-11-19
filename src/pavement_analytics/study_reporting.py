"""Offline comparison of inspection-level deduct targets and their predictors.

This report reads prediction tables, never serialized estimators. Identities stay
in the private CSV exports and are not displayed in the HTML dashboard.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .security import SafeError, reject_links
from .two_stage import STAGE_PREDICTION_COLUMNS, STAGE_METRIC_COLUMNS
from .model_comparison import (COMPARISON_COLUMNS, FAMILY_COLUMNS,
    PREDICTION_COLUMNS as CANDIDATE_PREDICTION_COLUMNS, IMPORTANCE_COLUMNS as FAMILY_IMPORTANCE_COLUMNS)
from .model_runtime import TIMING_COLUMNS

SUMMARY_COLUMNS = [
    "material", "target", "original_target", "target_kind", "target_units", "selection_metric",
    "status", "candidate", "cv_mae", "cv_mae_std", "cv_rmse", "cv_rmse_std",
    "holdout_mae", "holdout_rmse", "holdout_r2", "baseline_mae", "baseline_rmse", "baseline_kind",
    "baseline_candidate", "baseline_family", "mae_skill", "rmse_skill",
    "valid_target_rows", "valid_target_groups", "development_rows", "development_groups",
    "holdout_rows", "holdout_groups", "nonzero_fraction", "nonclimate_mae", "nonclimate_rmse",
    "nonclimate_candidate", "climate_added_skill", "climate_added_rmse_skill", "importance_reliable",
    "constant_target_scope", "mostly_zero_target",
    "candidate_family", "candidate_registry_version", "candidate_failure_count", "candidate_unavailable_count", "candidate_ineligible_count",
    "candidate_iterations",
    "comparison_failure_count", "comparison_completed_count",
    "importance_status", "importance_metric",
]
IMPORTANCE_COLUMNS = ["material", "target", "feature", "kind", "is_climate", "metric",
                      "importance_mean", "importance_std", "repeats", "reliable",
                      "baseline_relative_importance"]
PREDICTION_COLUMNS = ["material", "target", "row_id", "group_id", "actual", "prediction",
                      "baseline_prediction", "nonclimate_prediction", "weight"]
LEADER_COLUMNS = ["material", "target", "candidate", "feature_set", "cv_mae", "cv_mae_std",
                  "cv_rmse", "cv_rmse_std", "target_kind", "selection_metric", "baseline_candidate",
                  "cv_wins_vs_baseline", "family", "candidate_status", "error_code",
                  "folds_completed", "selected", "parameters", "cv_iterations", "cv_training",
                  "final_iterations", "final_training", "two_stage_diagnostic", "diagnostic_training"]
SCATTER_LIMIT = 600
SPARSE_NONZERO_THRESHOLD = .05


def _path(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts)
    reject_links(path)
    if root.resolve() not in path.resolve().parents:
        raise SafeError("INVALID_REPORT_PATH")
    return path


def _text(value) -> str:
    return escape(str(value), quote=True)


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _number(value, decimals=3) -> str:
    return f"{float(value):,.{decimals}f}" if _finite(value) else "—"


def _percent(value) -> str:
    return f"{100 * float(value):,.1f}%" if _finite(value) else "—"


def _true(value) -> bool:
    return value is True or str(value).lower() in {"true", "1"}


def _target_label(row: dict) -> str:
    original = row.get("original_target")
    return original if isinstance(original, str) and original else str(row.get("target", ""))


def _is_signed_rate(row: dict) -> bool:
    return (row.get("target_kind") == "signed_pci_rate"
            or row.get("target") == "DetRatePerYear"
            or row.get("original_target") == "DetRatePerYear")


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(
        f'<th scope="col">{_text(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{_text(value)}</td>' for value in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        # Identity tokens such as NA are valid group names, not missing values.
        # A converter preserves them without changing missing numeric metrics.
        return pd.read_csv(path, converters={"group_id": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _metrics(row: dict, frame: pd.DataFrame) -> None:
    """Recompute all available holdout metrics using the saved observation weights."""
    needed = {"actual", "prediction", "baseline_prediction", "weight"}
    if frame.empty or not needed <= set(frame):
        return
    values = frame[list(needed)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all() or (values.weight <= 0).any():
        raise SafeError("INVALID_STUDY_REPORT_PREDICTIONS")
    y, pred, weights = values.actual.to_numpy(), values.prediction.to_numpy(), values.weight.to_numpy()
    error = y - pred
    mae = float(np.average(np.abs(error), weights=weights))
    baseline_error = y - values.baseline_prediction.to_numpy()
    baseline = float(np.average(np.abs(baseline_error), weights=weights))
    baseline_rmse = float(np.sqrt(np.average(baseline_error ** 2, weights=weights)))
    variance = float(np.average((y - np.average(y, weights=weights)) ** 2, weights=weights))
    mse = float(np.average(error ** 2, weights=weights))
    row.update(holdout_mae=mae, holdout_rmse=float(np.sqrt(mse)),
               holdout_r2=1 - mse / variance if variance > 0 else None,
               baseline_mae=baseline, baseline_rmse=baseline_rmse,
               mae_skill=1 - mae / baseline if baseline > 0 else None,
               rmse_skill=1 - np.sqrt(mse) / baseline_rmse if baseline_rmse > 0 else None,
               holdout_rows=len(frame))
    if "group_id" in frame:
        row["holdout_groups"] = frame.group_id.nunique()
    if "nonclimate_prediction" in frame:
        other = pd.to_numeric(frame.nonclimate_prediction, errors="coerce").to_numpy()
        if np.isfinite(other).all():
            nonclimate = float(np.average(np.abs(y - other), weights=weights))
            nonclimate_rmse = float(np.sqrt(np.average((y - other) ** 2, weights=weights)))
            row.update(nonclimate_mae=nonclimate, nonclimate_rmse=nonclimate_rmse,
                       climate_added_skill=1 - mae / nonclimate if nonclimate > 0 else None,
                       climate_added_rmse_skill=1 - np.sqrt(mse) / nonclimate_rmse if nonclimate_rmse > 0 else None)


def _scatter(frame: pd.DataFrame, *, signed_rate: bool = False) -> str:
    if frame.empty or not {"actual", "prediction"} <= set(frame):
        return '<p class="empty">No independent holdout predictions available.</p>'
    frame = frame.loc[np.isfinite(frame[["actual", "prediction"]].to_numpy(float)).all(axis=1)]
    if frame.empty:
        return '<p class="empty">No finite holdout predictions available.</p>'
    values = frame[["actual", "prediction"]].to_numpy(float)
    low, high = float(values.min()), float(values.max())
    pad = .06 * (high - low) if high > low else max(1., abs(low) * .1)
    low, high = low - pad, high + pad
    sampled = frame.iloc[np.linspace(0, len(frame) - 1, min(SCATTER_LIMIT, len(frame)), dtype=int)]
    x = lambda v: 72 + (float(v) - low) / (high - low) * 370
    y = lambda v: 276 - (float(v) - low) / (high - low) * 230
    description = 'Actual versus predicted signed PCI rate' if signed_rate else 'Actual versus predicted deduct total'
    chunks = [f'<svg viewBox="0 0 500 330" role="img" aria-label="{description}">',
              '<rect x="72" y="46" width="370" height="230" fill="#f0f5f8"/>']
    for value in np.linspace(low, high, 5):
        px, py = x(value), y(value)
        chunks += [f'<line x1="72" y1="{py:.2f}" x2="442" y2="{py:.2f}" stroke="#dbe4ed"/>',
                   f'<text x="62" y="{py + 4:.2f}" text-anchor="end">{_number(value, 1)}</text>',
                   f'<text x="{px:.2f}" y="297" text-anchor="middle">{_number(value, 1)}</text>']
    chunks.append('<line x1="72" y1="276" x2="442" y2="46" stroke="#64748b" stroke-dasharray="5 4"/>')
    for point in sampled.itertuples():
        chunks.append(f'<circle cx="{x(point.actual):.2f}" cy="{y(point.prediction):.2f}" r="3" '
                      f'fill="#117c83" opacity=".5"><title>Actual {_number(point.actual)}; '
                      f'predicted {_number(point.prediction)}</title></circle>')
    observed = 'Observed PCI rate (points/year)' if signed_rate else 'Observed deduct-family total'
    predicted = 'Predicted PCI rate (points/year)' if signed_rate else 'Predicted total'
    chunks += [f'<text x="257" y="323" text-anchor="middle">{observed}</text>',
               f'<text transform="translate(17 162) rotate(-90)" text-anchor="middle">{predicted}</text>',
               '</svg>', f'<p class="caption">{len(sampled):,} of {len(frame):,} holdout points shown. '
               'Dashed line: perfect prediction. Complete predictions are exported.</p>']
    return ''.join(chunks)


def _importance_chart(frame: pd.DataFrame, *, metric: str = "mae") -> str:
    if frame.empty or not {"feature", "importance_mean", "kind"} <= set(frame):
        return '<p class="empty">No feature-importance estimates available.</p>'
    subset = frame.loc[frame.kind.eq("feature")].copy()
    subset["importance_mean"] = pd.to_numeric(subset.importance_mean, errors="coerce")
    subset = subset.loc[np.isfinite(subset.importance_mean)].sort_values(
        ["importance_mean", "feature"], ascending=[False, True]).head(10)
    if subset.empty:
        return '<p class="empty">No individual-feature estimates available.</p>'
    std = pd.to_numeric(subset.get("importance_std", pd.Series(0., index=subset.index)), errors="coerce").fillna(0)
    low = min(0., float((subset.importance_mean - std).min()))
    high = max(0., float((subset.importance_mean + std).max()))
    if high == low:
        high = 1.
    span = high - low
    x = lambda v: 210 + (float(v) - low) / span * 270
    zero = x(0)
    height = len(subset) * 42 + 62
    chunks = [f'<svg viewBox="0 0 510 {height}" role="img" aria-label="Held-out permutation feature importance">',
              f'<line x1="{zero:.2f}" y1="8" x2="{zero:.2f}" y2="{height - 52}" stroke="#8da1b2"/>']
    for i, (index, row) in enumerate(subset.iterrows()):
        top = 12 + i * 42
        value, uncertainty = float(row.importance_mean), float(std.loc[index])
        label = str(row.feature)
        short = label if len(label) <= 26 else label[:24] + '…'
        climate = _true(row.get("is_climate", False))
        colour = '#117c83' if climate else '#6979a2'
        chunks += [f'<text x="198" y="{top + 18}" text-anchor="end"><title>{_text(label)}</title>{_text(short)}</text>',
                   f'<rect x="{min(zero, x(value)):.2f}" y="{top}" width="{abs(x(value) - zero):.2f}" '
                   f'height="25" rx="3" fill="{colour}"><title>{_text(label)}: {_number(value)} '
                   f'± {_number(uncertainty)}; {"climate" if climate else "other predictor"}</title></rect>',
                   f'<line x1="{x(value - uncertainty):.2f}" y1="{top + 12}" '
                   f'x2="{x(value + uncertainty):.2f}" y2="{top + 12}" stroke="#1b3148"/>']
    chunks += [f'<text x="210" y="{height - 29}">{_number(low)}</text>',
               f'<text x="480" y="{height - 29}" text-anchor="end">{_number(high)}</text>',
               f'<text x="345" y="{height - 8}" text-anchor="middle">Increase in holdout {metric.upper()}</text></svg>',
               '<p class="caption">Top 10 predictors. Teal: climate; violet: other inputs. Lines show '
               'repeat standard deviation, not confidence intervals. Values near or below zero show '
               'little measured contribution in this model.</p>']
    return ''.join(chunks)


def _coverage(audit: dict, *, has_rate: bool = False, has_deduct: bool = True) -> str:
    original = audit.get("task_mode") == "original_pipeline"
    climate = audit.get("climate", {})
    if not isinstance(climate, dict):
        climate = {}
    observations = audit.get("observations", audit.get("accounting", {}))
    if not isinstance(observations, dict):
        observations = {}
    accounting = observations.get("accounting", observations)
    if not isinstance(accounting, dict):
        accounting = {}
    rows = []
    for key, label in [("input_rows", "Input inspections"), ("accepted_rows", "Accepted inspections"),
                       ("rejected_rows", "Rejected inspections"), ("duplicate_rows", "Duplicate records")]:
        if _finite(accounting.get(key)):
            rows.append([label, _number(accounting[key], 0)])
    for key, label in [("rows", "Climate rows"), ("eligible_rows", "Rows with complete eligible climate"),
                       ("rows_with_climate", "Rows with climate"),
                       ("missing_location_rows", "Rows missing a location"),
                       ("missing_climate_rows", "Rows missing climate"),
                       ("lookback_days", "Historical climate window (days)")]:
        if _finite(climate.get(key)):
            rows.append([label, _number(climate[key], 0)])
    climate_counts = climate.get("counts", {})
    if isinstance(climate_counts, dict):
        for key, label in [("missing_location", "Missing location"),
                           ("incomplete_hourly_coverage", "Incomplete hourly climate coverage"),
                           ("window_before_era5_coverage", "Window before ERA5 coverage"),
                           ("invalid_existing_features", "Invalid supplied climate features")]:
            if _finite(climate_counts.get(key)):
                rows.append([label, _number(climate_counts[key], 0)])
    for key, label in [("model_rows", "Inspection pairs available to modeling" if original else "Inspections available to modeling"),
                       ("excluded_climate_rows", "Inspections excluded for climate coverage")]:
        if _finite(audit.get(key)):
            rows.append([label, _number(audit[key], 0)])
    result = '<section><h2>Coverage and interpretation</h2>'
    if rows:
        result += _table(["Measure", "Count / value"], rows)
    features = audit.get("features", [])
    if not features and isinstance(audit.get("predictors"), list):
        features = [v.get("feature") for v in audit["predictors"]
                    if isinstance(v, dict) and v.get("feature") and v.get("selected", True)]
    if isinstance(features, list) and features:
        result += '<p>Predictors: ' + _text(', '.join(str(v) for v in features)) + '.</p>'
    if climate.get("mode"):
        result += '<p>Climate source mode: <strong>' + _text(climate["mode"]) + '</strong>.'
        if climate.get("historical_window_verified") is True and _finite(climate.get("lookback_days", audit.get("lookback_days"))):
            result += ' Historical window: ' + _number(climate.get("lookback_days", audit.get("lookback_days")), 0) + ' days.'
        result += '</p>'
    if climate.get("historical_window_verified") is False:
        result += ('<p class="notice">Historical timing of supplied climate features was not independently '
                   'verified. Confirm that they use no information after the inspection before interpreting '
                   'predictive performance.</p>')
    partition = audit.get("partition", {})
    if not isinstance(partition, dict):
        partition = {}
    grouping = partition.get("grouping")
    group_labels = {"climate_location": "Groups joined by shared ERA5 grid locations",
                    "declared_group": "Declared network/group identities",
                    "section_identity": "Section identities"}
    if original and isinstance(grouping, str) and grouping not in group_labels:
        group_labels[grouping] = _text(grouping)
    if grouping in group_labels:
        result += '<p>Validation grouping: <strong>' + group_labels[grouping] + '</strong>.'
        if _finite(partition.get("declared_groups")):
            result += ' Declared groups: ' + _number(partition["declared_groups"], 0) + '.'
        if _finite(partition.get("validation_groups")):
            result += ' Final validation groups: ' + _number(partition["validation_groups"], 0) + '.'
        result += '</p>'
    if original and partition.get("grid_limitation"):
        result += '<p class="notice">' + _text(partition["grid_limitation"]) + '</p>'
    if grouping == "section_identity":
        result += ('<p class="notice">Validation groups identify sections. Related sections may share a '
                   'network or climate location across partitions; this run does not establish performance '
                   'at new independent locations.</p>')
    if original:
        if has_deduct:
            result += ('<p>Deduct-family targets are observed totals at the <strong>end inspection of each '
                   'eligible original-pipeline pair</strong>. The original pairing, distress, PCI and '
                   'Shapley preparation stages determine this population. These targets are not annual '
                   'deduct changes. Target-specific missing values remain excluded only for that target.</p>')
        if has_rate:
            result += ('<p>The separate <strong>DetRatePerYear</strong> target is signed annual PCI change: '
                       '(PCI_end − PCI_start) × 365 / delta_days. Negative values mean deterioration; '
                       'zero means no recorded PCI change. Its errors are in PCI points/year, while '
                       'deduct-total errors are in deduct points. These targets are reported separately.</p>')
        result += ('<p class="notice">The original climate definitions are preserved: interval exposures, '
                   'pavement-grade statistics and construction-aging calculations use their original '
                   'windows. Some statistics use weather outside the inspection interval. This is '
                   '<strong>retrospective explanatory modeling</strong>, not a prospective forecast. '
                   'Importance measures predictive association, not causal impact.</p>'
                   '<p>Starting-inspection predictors admitted by the original feature audit are retained. '
                   'End-inspection outcomes and other target columns are excluded from predictors. '
                   'Different target availability and export coverage can affect comparisons.</p>')
    else:
        result += ('<p>Targets are observed deduct-family totals at each inspection, summed from exported '
                   'deduct channels. An unexported channel is not a confirmed zero. Different target populations '
                   'and export coverage can affect comparisons; inspect row and group counts alongside scores.</p>'
                   '<p>The intended climate window ends before the inspection. This is an '
                   'observational prediction study; importance measures predictive association, not causal impact. '
                   'Same-inspection PCI, other deduct outcomes, identities, and distress quantities are excluded '
                   'from the predictors.</p>')
    weight_note = ('Each observation is weighted by its inspection interval in days, preserving the original '
                   'pipeline weight role.' if audit.get("metric_weighting") == "interval_days" else
                   'Each validation group receives equal total weight.')
    result += ('<p>Groups retain the same development/holdout assignment across targets. Model selection '
               'uses development cross-validation; holdout data are reserved for evaluation. Repeated '
               'inspections are grouped. ' + weight_note + ' '
               'Comparing many targets on this holdout is exploratory; confirm the strongest results on '
               'additional independent data.</p></section>')
    return result


def build_study_report(out: Path, summaries: list[dict], audit: dict,
                       model_root: Path | None = None) -> None:
    """Consolidate model tables and write a self-contained private HTML report."""
    out = Path(out)
    reject_links(out)
    if not out.is_dir():
        raise SafeError("REPORT_RUN_DIRECTORY_REQUIRED")
    model_root = Path(model_root) if model_root is not None else out
    reject_links(model_root)
    if not model_root.is_dir():
        raise SafeError("REPORT_MODEL_DIRECTORY_REQUIRED")
    seen, records, importance_tables, prediction_tables, leader_tables, cards = set(), [], [], [], [], []
    stage_prediction_tables, stage_metric_tables = [], []
    comparison_specs = {"candidate_comparisons.csv": COMPARISON_COLUMNS, "family_summary.csv": FAMILY_COLUMNS,
        "candidate_predictions.csv": CANDIDATE_PREDICTION_COLUMNS, "family_importance.csv": FAMILY_IMPORTANCE_COLUMNS,
        "model_runtime.csv": TIMING_COLUMNS}
    comparison_tables = {name: [] for name in comparison_specs}
    for summary in summaries:
        material, target = str(summary.get("material", "")), str(summary.get("target", ""))
        if (not re.fullmatch(r"[A-Za-z0-9_-]+", material)
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", target)):
            raise SafeError("INVALID_REPORT_TASK")
        if (material, target) in seen:
            raise SafeError("DUPLICATE_REPORT_TASK")
        seen.add((material, target))
        row = {key: summary.get(key) for key in SUMMARY_COLUMNS}
        signed_rate = _is_signed_rate(row)
        row["target_kind"] = "signed_pci_rate" if signed_rate else "deduct_total"
        row["target_units"] = "PCI points/year" if signed_rate else "deduct points"
        row["baseline_kind"] = "weighted_mean" if signed_rate else "weighted_median"
        row["importance_metric"] = "rmse" if signed_rate else "mae"
        prevalence = row.get("nonzero_fraction")
        row["mostly_zero_target"] = (float(prevalence) < SPARSE_NONZERO_THRESHOLD
                                     if _finite(prevalence) else None)
        root = _path(model_root, "materials", material, target)
        predictions = _read(_path(root, "holdout_predictions.csv"))
        importance = _read(_path(root, "importance.csv"))
        leaders = _read(_path(root, "leaderboard.csv"))
        stage_predictions = _read(_path(root, "two_stage_predictions.csv"))
        stage_metrics = _read(_path(root, "two_stage_metrics.csv"))
        task_comparisons = {name: _read(_path(root, name)) for name in comparison_specs}
        for name, frame in task_comparisons.items():
            if not frame.empty:
                frame.insert(0, "target", target)
                frame.insert(0, "material", material)
                comparison_tables[name].append(frame)
        _metrics(row, predictions)
        records.append(row)
        for frame, collection in [(predictions, prediction_tables), (importance, importance_tables), (leaders, leader_tables),
                                  (stage_predictions, stage_prediction_tables), (stage_metrics, stage_metric_tables)]:
            if not frame.empty:
                frame.insert(0, "target", target)
                frame.insert(0, "material", material)
                collection.append(frame)
        if not importance.empty:
            importance["metric"] = row["importance_metric"]
            denominator = row.get("baseline_rmse" if signed_rate else "baseline_mae")
            importance["baseline_relative_importance"] = (
                pd.to_numeric(importance.importance_mean, errors="coerce") / float(denominator)
                if _finite(denominator) and float(denominator) > 0 else np.nan)
        reliable = _true(row.get("importance_reliable"))
        label = _target_label(row)
        detail = '<section class="target"><h2>' + _text(material + ' · ' + label) + '</h2>'
        detail += '<p><span class="badge">' + _text(row.get("status") or 'unknown') + '</span> Model: <strong>'
        detail += _text(row.get("candidate") or '—') + '</strong>.</p>'
        if signed_rate:
            detail += ('<p>Signed annual PCI change: negative values mean deterioration and zero means '
                       'no recorded PCI change. Model selection uses development weighted RMSE. '
                       'MAE skill compares with the training-derived weighted mean and remains a separate diagnostic.</p>')
        detail += '<p>Weighted holdout MAE and RMSE are in ' + _text(row["target_units"]) + '.</p>'
        failed, unavailable = row.get("candidate_failure_count"), row.get("candidate_unavailable_count")
        if (_finite(failed) and failed > 0) or (_finite(unavailable) and unavailable > 0):
            detail += ('<p class="notice">Candidate attempts with errors: ' + _number(failed, 0) +
                       '; unavailable or unsupported: ' + _number(unavailable, 0) +
                       '. See the model comparisons CSV for each status.</p>')
        if _finite(row.get("comparison_failure_count")) and row["comparison_failure_count"] > 0:
            detail += '<p class="notice">Comparison attempts with errors: ' + _number(row["comparison_failure_count"], 0) + '. See all candidate scores for statuses.</p>'
        if row.get("importance_status") in {"failed", "timed_out"}:
            detail += '<p class="notice">Primary feature explanation status: ' + _text(row["importance_status"]) + '.</p>'
        detail += _table(["MAE skill", "Holdout R²", "Weighted holdout MAE", "Weighted holdout RMSE",
                          "Baseline MAE", "Baseline RMSE", "Holdout rows / groups"], [[
            _percent(row.get("mae_skill")), _number(row.get("holdout_r2")),
            _number(row.get("holdout_mae")), _number(row.get("holdout_rmse")), _number(row.get("baseline_mae")),
            _number(row.get("baseline_rmse")),
            _number(row.get("holdout_rows"), 0) + ' / ' + _number(row.get("holdout_groups"), 0)]])
        if row["mostly_zero_target"]:
            detail += ('<p class="notice">Mostly zero target: ' + _percent(prevalence) +
                       ' of eligible observations are nonzero (flag threshold: below 5%). '
                       'Small overall error can hide poor predictions of the uncommon nonzero values.</p>')
        if row.get("status") == "constant_target":
            if row.get("constant_target_scope") == "development":
                detail += ('<p>The target is constant in the development data. The median baseline is '
                           'evaluated on holdout observations, which may have different values; learned '
                           'feature importance is unavailable.</p>')
            else:
                detail += '<p>The modeling stage flagged a constant target; learned feature importance is unavailable.</p>'
        if not reliable:
            detail += ('<p class="notice">Feature rankings are exploratory: this target has not demonstrated '
                       'reliable improvement over the baseline. Do not interpret low importance as evidence '
                       'that a feature has no relationship with the target.</p>')
        climate_skill = "climate_added_rmse_skill" if signed_rate else "climate_added_skill"
        climate_metric = "RMSE" if signed_rate else "MAE"
        if _finite(row.get(climate_skill)):
            detail += ('<p>Adding climate changed holdout ' + climate_metric + ' by <strong>' +
                       _percent(row[climate_skill]) + '</strong> relative to the model selected '
                       'using other predictors alone. Positive means lower error; this comparison includes '
                       'model selection and is not a causal effect.</p>')
        if not signed_rate and not stage_metrics.empty:
            component_holdout = stage_metrics.loc[stage_metrics.partition.eq("holdout") & stage_metrics.feature_set.eq("all")]
            if not component_holdout.empty:
                component = component_holdout.iloc[0]
                detail += '<h3>Two-stage conditional-mean diagnostics</h3>'
                detail += ('<p>Occurrence probability times positive mean estimates the overall mean. '
                           'It need not minimize MAE on zero-heavy data. The diagnostic model is selected '
                           'within its family using development CV MAE; the primary selection is unchanged.</p>')
                detail += _table(["Diagnostic model", "Mean RMSE", "Occurrence Brier", "Log loss", "Positive severity RMSE"], [[
                    str(component.candidate), _number(component.mean_prediction_rmse),
                    _number(component.occurrence_brier), _number(component.occurrence_log_loss),
                    _number(component.positive_severity_rmse)]])
        families = task_comparisons["family_summary.csv"]
        if not families.empty:
            detail += '<details><summary>Model-family comparisons and explanations</summary>'
            choice_metric = 'weighted RMSE' if signed_rate else 'MAE'
            choice_column = 'cv_rmse' if signed_rate else 'cv_mae'
            detail += ('<p>Representatives are selected independently per family and feature set using development CV '
                       + choice_metric + '. '
                       'Holdout scores evaluate those choices. Importance is exploratory predictive reliance; '
                       'repeat SD is permutation variability, not a confidence interval. Climate comparisons are not causal.</p>')
            all_families = families.loc[families.feature_set.eq("all")]
            detail += _table(["Family", "CV choice", "Status", "CV " + choice_metric,
                              "Weighted holdout RMSE", "Holdout MAE skill", "Climate added " + climate_metric + " skill", "Explanation status"], [
                [str(r.get("family", "")), str(r.get("candidate", "")), str(r.get("comparison_status", "")),
                 _number(r.get(choice_column)), _number(r.get("holdout_rmse")),
                 _percent(r.get("mae_skill")), _percent(r.get(climate_skill)), str(r.get("importance_status", ""))]
                for r in all_families.to_dict("records")])
            family_importance = task_comparisons["family_importance.csv"]
            if not family_importance.empty:
                for family, values in family_importance.loc[family_importance.feature_set.eq("all")].groupby("family", sort=True):
                    for metric, block in values.groupby("metric", sort=True):
                        strongest = block.sort_values("importance_mean", ascending=False).head(8)
                        detail += '<h3>' + _text(str(family)+' · '+str(metric)) + '</h3>'
                        support_column = "heuristic_support" if "heuristic_support" in block else "reliable"
                        if not block[support_column].map(_true).all():
                            detail += '<p class="notice">Exploratory rankings: this score does not meet the support/improvement rule.</p>'
                        detail += _table(["Predictor / block", "Columns", "Error increase", "Repeat SD"], [
                            [str(r["feature"]), str(r["columns"]), _number(r["importance_mean"]), _number(r["importance_std"])]
                            for r in strongest.to_dict("records")])
            detail += '</details>'
        if not importance.empty and "kind" in importance:
            blocks = importance.loc[importance.kind.eq("climate_block")]
            if len(blocks):
                block = blocks.iloc[0]
                detail += ('<p>Permuting the whole climate block increased ' + row["importance_metric"].upper() + ' by <strong>' +
                           _number(block.get("importance_mean")) + '</strong> (repeat SD ' +
                           _number(block.get("importance_std")) + ').</p>')
        detail += '<div class="charts"><div><h3>Observed versus predicted</h3>' + _scatter(predictions, signed_rate=signed_rate) + '</div>'
        detail += '<div><h3>Strongest predictors</h3>' + _importance_chart(importance, metric=row["importance_metric"]) + '</div></div>'
        if not importance.empty:
            feature_rows = [[str(v.get("feature", "")), 'climate' if _true(v.get("is_climate")) else 'other',
                             _number(v.get("importance_mean")), _number(v.get("importance_std"))]
                            for v in importance.to_dict("records")]
            detail += '<details><summary>All importance estimates</summary>'
            detail += _table(["Feature / block", "Type", row["importance_metric"].upper() + " increase", "Repeat SD"], feature_rows) + '</details>'
        detail += '</section>'
        cards.append(detail)
    summary_table = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    # Keep different target kinds separate. Missing skill values sort last;
    # raw errors are deliberately not used to rank targets with different scales.
    if not summary_table.empty:
        summary_table = summary_table.sort_values(["target_kind", "mae_skill", "holdout_r2", "material", "target"],
                                                  ascending=[True, False, False, True, True], na_position="last")
    importance_table = pd.concat(importance_tables, ignore_index=True).reindex(columns=IMPORTANCE_COLUMNS) if importance_tables else pd.DataFrame(columns=IMPORTANCE_COLUMNS)
    prediction_table = pd.concat(prediction_tables, ignore_index=True).reindex(columns=PREDICTION_COLUMNS) if prediction_tables else pd.DataFrame(columns=PREDICTION_COLUMNS)
    leader_table = pd.concat(leader_tables, ignore_index=True).reindex(columns=LEADER_COLUMNS) if leader_tables else pd.DataFrame(columns=LEADER_COLUMNS)
    stage_prediction_columns = ["material", "target", *STAGE_PREDICTION_COLUMNS]
    stage_metric_columns = ["material", "target", *STAGE_METRIC_COLUMNS]
    stage_prediction_table = pd.concat(stage_prediction_tables, ignore_index=True).reindex(columns=stage_prediction_columns) if stage_prediction_tables else pd.DataFrame(columns=stage_prediction_columns)
    stage_metric_table = pd.concat(stage_metric_tables, ignore_index=True).reindex(columns=stage_metric_columns) if stage_metric_tables else pd.DataFrame(columns=stage_metric_columns)
    for filename, frame in [("target_summary.csv", summary_table), ("feature_importance.csv", importance_table),
                            ("predictions.csv", prediction_table), ("model_leaderboard.csv", leader_table),
                            ("two_stage_predictions.csv", stage_prediction_table), ("two_stage_metrics.csv", stage_metric_table)]:
        frame.to_csv(_path(out, filename), index=False)
    for name, columns in comparison_specs.items():
        tables = comparison_tables[name]
        combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        if name == "family_importance.csv" and "heuristic_support" not in combined and "reliable" in combined:
            combined = combined.rename(columns={"reliable": "heuristic_support"})
        combined.reindex(columns=["material", "target", *columns]).to_csv(_path(out, name), index=False)
    rate_table = summary_table.loc[summary_table.target_kind.eq("signed_pci_rate")].sort_values(["material", "target"])
    deduct_table = summary_table.loc[summary_table.target_kind.ne("signed_pci_rate")]
    has_rate, has_deduct = not rate_table.empty, not deduct_table.empty
    main = ''
    for table, signed_rate in [(rate_table, True), (deduct_table, False)]:
        if table.empty:
            continue
        heading = 'Signed PCI rate models' if signed_rate else 'Which deduct families are most predictable?'
        baseline_name = 'weighted mean' if signed_rate else 'median'
        main += '<section><h2>' + heading + '</h2>'
        if signed_rate:
            main += ('<p>DetRatePerYear is signed annual PCI change in <strong>PCI points/year</strong>. '
                     'Negative values mean deterioration. Models are selected by development weighted RMSE '
                     'and shown by surface, separately from deduct totals. MAE skill is a secondary diagnostic.</p>')
        else:
            main += '<p>Ranked by holdout <strong>MAE skill</strong>. Errors are in <strong>deduct points</strong>.</p>'
        main += ('<p>MAE skill is 1 − model MAE / baseline MAE. Positive values improve on the training-derived '
                 + baseline_name + ' baseline; zero means no improvement; negative means worse. '
                 'An undefined score is shown as — when baseline error is zero. '
                 'R² compares predictions with the holdout mean; negative R² indicates poor fit. '
                 'MAE and RMSE use the saved observation weights; RMSE gives larger errors more influence.</p>')
        overview_rows = [[str(v["material"]) + ' · ' + _target_label(v), str(v.get("status") or '—'),
                          _number(v.get("holdout_mae")), _number(v.get("holdout_rmse")),
                          _percent(v.get("mae_skill")), _number(v.get("holdout_r2")),
                          _number(v.get("valid_target_rows"), 0) + ' / ' + _number(v.get("valid_target_groups"), 0),
                          _percent(v.get("nonzero_fraction")) + (' · mostly zero' if _true(v.get("mostly_zero_target")) else '')]
                         for v in table.to_dict("records")]
        main += _table(["Surface · target", "Status", "Weighted holdout MAE", "Weighted holdout RMSE",
                        "MAE skill", "R²", "Valid rows / groups", "Nonzero"], overview_rows)
        main += ('<p class="caption">Raw errors are not used to rank targets with different scales. '
                 'A mostly zero target can have small error while remaining difficult to predict when '
                 'nonzero values occur. “Mostly zero” flags targets with fewer than 5% nonzero eligible '
                 'observations; it is descriptive, not an accuracy judgment.</p></section>')
    if summary_table.empty:
        main += '<section><p>No target tasks were recorded.</p></section>'
    for table, signed_rate in [(rate_table, True), (deduct_table, False)]:
        strongest = []
        for row in table.to_dict("records"):
            if importance_table.empty:
                break
            subset = importance_table.loc[(importance_table.material == row["material"]) &
                                         (importance_table.target == row["target"]) &
                                         (importance_table.kind == "feature")].sort_values(
                                             ["importance_mean", "feature"], ascending=[False, True])
            labels = []
            for feature in subset.head(3).to_dict("records"):
                labels.append(str(feature["feature"]) + (' [climate]' if _true(feature["is_climate"]) else '')
                              + ' (' + _percent(feature.get("baseline_relative_importance")) + ')')
            if labels:
                strongest.append([str(row["material"]) + ' · ' + _target_label(row),
                                  'positive holdout skill' if _true(row.get("importance_reliable")) else 'exploratory',
                                  '; '.join(labels)])
        if strongest:
            metric = 'RMSE' if signed_rate else 'MAE'
            heading = 'Predictors of signed PCI rate' if signed_rate else 'Predictors across deduct families'
            main += '<section><h2>' + heading + '</h2>'
            main += _table(["Surface · target", "Interpretation", "Top predictors (" + metric + " increase / baseline " + metric + ")"], strongest)
            main += ('<p class="caption">Percentages describe permutation error increases relative to each '
                     'target’s baseline error. They are not percentages of explained variance and need not '
                     'sum to 100%. Correlated inputs can share or mask importance; the climate-block result '
                     'helps assess their combined contribution.</p></section>')
    main += _coverage(audit, has_rate=has_rate, has_deduct=has_deduct) + ''.join(cards)
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>Deduct predictability study</title><style>
:root{color-scheme:light;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1c3048;background:#edf2f6}
body{margin:0 auto;padding:28px;max-width:1250px;line-height:1.55}h1{font-size:2rem;margin:0}h2{font-size:1.35rem;line-height:1.3}h3{font-size:1.05rem}p{margin:.8rem 0}
header{padding:12px 0 26px}section{background:white;border:1px solid #d9e4ee;border-radius:12px;padding:22px;margin-bottom:22px}a{color:#066f80}nav{display:flex;flex-wrap:wrap;gap:16px}.caption,.empty{font-size:.86rem;color:#51657c}.notice{background:#fff6e6;border-left:4px solid #bc7b13;padding:10px 14px}.badge{display:inline-block;border-radius:5px;background:#e9f0f6;padding:3px 8px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:9px;text-align:left;vertical-align:top;border-bottom:1px solid #e0e8ef;overflow-wrap:anywhere}th{background:#f0f5f8;font-weight:650}tbody tr:nth-child(even){background:#f8fafc}.charts{display:grid;grid-template-columns:1fr 1fr;gap:28px}svg{width:100%;height:auto}svg text{font-family:inherit;font-size:12px}summary{cursor:pointer;font-weight:650;margin:14px 0}footer{font-size:.84rem;color:#51657c}
@media(max-width:760px){body{padding:12px}section{padding:14px}.charts{grid-template-columns:1fr}table{font-size:.8rem}}
@media print{@page{size:landscape;margin:12mm}body{max-width:none;padding:0;background:white;font-size:10pt}section{border-radius:0;padding:10pt;margin:0 0 12pt;break-inside:auto}h1,h2,h3{break-after:avoid}.target{break-before:page}.charts{grid-template-columns:1fr 1fr;gap:16pt}.charts>div{break-inside:avoid}svg{max-height:155mm}table{font-size:8pt;table-layout:fixed}th,td{padding:5pt}thead{display:table-header-group}tr{break-inside:avoid}.table-wrap{overflow:visible}details>*{display:block}summary{display:none}nav{display:none}a{color:inherit;text-decoration:none}}
</style></head><body><header><h1>Deduct predictability study</h1>
<p>Observed family totals · grouped validation · climate and available covariates</p>
<nav aria-label="Download complete results"><a href="target_summary.csv">Target summary CSV</a><a href="feature_importance.csv">Feature importance CSV</a><a href="model_leaderboard.csv">Model comparisons CSV</a><a href="predictions.csv">Predictions CSV</a><a href="family_summary.csv">Family comparisons CSV</a><a href="family_importance.csv">Family explanations CSV</a><a href="candidate_comparisons.csv">All candidate scores CSV</a><a href="candidate_predictions.csv">All candidate predictions CSV</a><a href="model_runtime.csv">Runtime CSV</a><a href="two_stage_metrics.csv">Two-stage metrics CSV</a><a href="two_stage_predictions.csv">Two-stage components CSV</a></nav></header>'''
    if has_rate:
        html = html.replace('Deduct predictability study', 'Pavement target predictability study')
        subtitle = 'Signed PCI rate and observed family totals' if has_deduct else 'Signed PCI rate'
        html = html.replace('Observed family totals ·', subtitle + ' ·')
    html += main + '<footer>This report is self-contained and makes no network requests. Outputs remain in the private run directory.</footer></body></html>'
    _path(out, "dashboard.html").write_text(html, encoding="utf-8")
