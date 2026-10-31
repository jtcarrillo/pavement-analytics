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

SUMMARY_COLUMNS = [
    "material", "target", "original_target", "status", "candidate", "cv_mae", "cv_mae_std",
    "holdout_mae", "holdout_rmse", "holdout_r2", "baseline_mae", "mae_skill",
    "valid_target_rows", "valid_target_groups", "development_rows", "development_groups",
    "holdout_rows", "holdout_groups", "nonzero_fraction", "nonclimate_mae",
    "nonclimate_candidate", "climate_added_skill", "importance_reliable",
    "constant_target_scope", "mostly_zero_target",
    "candidate_family", "candidate_registry_version", "candidate_failure_count", "candidate_unavailable_count",
    "candidate_iterations",
]
IMPORTANCE_COLUMNS = ["material", "target", "feature", "kind", "is_climate",
                      "importance_mean", "importance_std", "repeats", "reliable",
                      "baseline_relative_importance"]
PREDICTION_COLUMNS = ["material", "target", "row_id", "group_id", "actual", "prediction",
                      "baseline_prediction", "nonclimate_prediction", "weight"]
LEADER_COLUMNS = ["material", "target", "candidate", "feature_set", "cv_mae", "cv_mae_std",
                  "cv_rmse", "cv_wins_vs_baseline", "family", "candidate_status", "error_code",
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
    """Recompute all available holdout metrics using saved group-balanced weights."""
    needed = {"actual", "prediction", "baseline_prediction", "weight"}
    if frame.empty or not needed <= set(frame):
        return
    values = frame[list(needed)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all() or (values.weight <= 0).any():
        raise SafeError("INVALID_STUDY_REPORT_PREDICTIONS")
    y, pred, weights = values.actual.to_numpy(), values.prediction.to_numpy(), values.weight.to_numpy()
    error = y - pred
    mae = float(np.average(np.abs(error), weights=weights))
    baseline = float(np.average(np.abs(y - values.baseline_prediction.to_numpy()), weights=weights))
    variance = float(np.average((y - np.average(y, weights=weights)) ** 2, weights=weights))
    mse = float(np.average(error ** 2, weights=weights))
    row.update(holdout_mae=mae, holdout_rmse=float(np.sqrt(mse)),
               holdout_r2=1 - mse / variance if variance > 0 else None,
               baseline_mae=baseline, mae_skill=1 - mae / baseline if baseline > 0 else None,
               holdout_rows=len(frame))
    if "group_id" in frame:
        row["holdout_groups"] = frame.group_id.nunique()
    if "nonclimate_prediction" in frame:
        other = pd.to_numeric(frame.nonclimate_prediction, errors="coerce").to_numpy()
        if np.isfinite(other).all():
            nonclimate = float(np.average(np.abs(y - other), weights=weights))
            row.update(nonclimate_mae=nonclimate,
                       climate_added_skill=1 - mae / nonclimate if nonclimate > 0 else None)


def _scatter(frame: pd.DataFrame) -> str:
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
    chunks = ['<svg viewBox="0 0 500 330" role="img" aria-label="Actual versus predicted deduct total">',
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
    chunks += ['<text x="257" y="323" text-anchor="middle">Observed deduct-family total</text>',
               '<text transform="translate(17 162) rotate(-90)" text-anchor="middle">Predicted total</text>',
               '</svg>', f'<p class="caption">{len(sampled):,} of {len(frame):,} holdout points shown. '
               'Dashed line: perfect prediction. Complete predictions are exported.</p>']
    return ''.join(chunks)


def _importance_chart(frame: pd.DataFrame) -> str:
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
               f'<text x="345" y="{height - 8}" text-anchor="middle">Increase in holdout MAE</text></svg>',
               '<p class="caption">Top 10 predictors. Teal: climate; violet: other inputs. Lines show '
               'repeat standard deviation, not confidence intervals. Values near or below zero show '
               'little measured contribution in this model.</p>']
    return ''.join(chunks)


def _coverage(audit: dict) -> str:
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
        result += ('<p>Targets are observed deduct-family totals at the <strong>end inspection of each '
                   'eligible original-pipeline pair</strong>. The original pairing, distress, PCI and '
                   'Shapley preparation stages determine this population. These targets are not annual '
                   'deduct changes. Target-specific missing values remain excluded only for that target.</p>'
                   '<p class="notice">The original climate definitions are preserved: interval exposures, '
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
    for summary in summaries:
        material, target = str(summary.get("material", "")), str(summary.get("target", ""))
        if (not re.fullmatch(r"[A-Za-z0-9_-]+", material)
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", target)):
            raise SafeError("INVALID_REPORT_TASK")
        if (material, target) in seen:
            raise SafeError("DUPLICATE_REPORT_TASK")
        seen.add((material, target))
        row = {key: summary.get(key) for key in SUMMARY_COLUMNS}
        prevalence = row.get("nonzero_fraction")
        row["mostly_zero_target"] = (float(prevalence) < SPARSE_NONZERO_THRESHOLD
                                     if _finite(prevalence) else None)
        root = _path(model_root, "materials", material, target)
        predictions = _read(_path(root, "holdout_predictions.csv"))
        importance = _read(_path(root, "importance.csv"))
        leaders = _read(_path(root, "leaderboard.csv"))
        stage_predictions = _read(_path(root, "two_stage_predictions.csv"))
        stage_metrics = _read(_path(root, "two_stage_metrics.csv"))
        _metrics(row, predictions)
        records.append(row)
        for frame, collection in [(predictions, prediction_tables), (importance, importance_tables), (leaders, leader_tables),
                                  (stage_predictions, stage_prediction_tables), (stage_metrics, stage_metric_tables)]:
            if not frame.empty:
                frame.insert(0, "target", target)
                frame.insert(0, "material", material)
                collection.append(frame)
        if not importance.empty:
            denominator = row.get("baseline_mae")
            importance["baseline_relative_importance"] = (
                pd.to_numeric(importance.importance_mean, errors="coerce") / float(denominator)
                if _finite(denominator) and float(denominator) > 0 else np.nan)
        reliable = _true(row.get("importance_reliable"))
        label = _target_label(row)
        detail = '<section class="target"><h2>' + _text(material + ' · ' + label) + '</h2>'
        detail += '<p><span class="badge">' + _text(row.get("status") or 'unknown') + '</span> Model: <strong>'
        detail += _text(row.get("candidate") or '—') + '</strong>.</p>'
        failed, unavailable = row.get("candidate_failure_count"), row.get("candidate_unavailable_count")
        if (_finite(failed) and failed > 0) or (_finite(unavailable) and unavailable > 0):
            detail += ('<p class="notice">Candidate attempts with errors: ' + _number(failed, 0) +
                       '; unavailable or unsupported: ' + _number(unavailable, 0) +
                       '. See the model comparisons CSV for each status.</p>')
        detail += _table(["MAE skill", "Holdout R²", "MAE", "Baseline MAE", "Holdout rows / groups"], [[
            _percent(row.get("mae_skill")), _number(row.get("holdout_r2")),
            _number(row.get("holdout_mae")), _number(row.get("baseline_mae")),
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
        if _finite(row.get("climate_added_skill")):
            detail += ('<p>Adding climate changed holdout MAE by <strong>' +
                       _percent(row["climate_added_skill"]) + '</strong> relative to the model selected '
                       'using other predictors alone. Positive means lower error; this comparison includes '
                       'model selection and is not a causal effect.</p>')
        if not stage_metrics.empty:
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
        if not importance.empty and "kind" in importance:
            blocks = importance.loc[importance.kind.eq("climate_block")]
            if len(blocks):
                block = blocks.iloc[0]
                detail += ('<p>Permuting the whole climate block increased MAE by <strong>' +
                           _number(block.get("importance_mean")) + '</strong> (repeat SD ' +
                           _number(block.get("importance_std")) + ').</p>')
        detail += '<div class="charts"><div><h3>Observed versus predicted</h3>' + _scatter(predictions) + '</div>'
        detail += '<div><h3>Strongest predictors</h3>' + _importance_chart(importance) + '</div></div>'
        if not importance.empty:
            feature_rows = [[str(v.get("feature", "")), 'climate' if _true(v.get("is_climate")) else 'other',
                             _number(v.get("importance_mean")), _number(v.get("importance_std"))]
                            for v in importance.to_dict("records")]
            detail += '<details><summary>All importance estimates</summary>'
            detail += _table(["Feature / block", "Type", "MAE increase", "Repeat SD"], feature_rows) + '</details>'
        detail += '</section>'
        cards.append(detail)
    summary_table = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    # Missing skill values sort last; raw MAE is deliberately not a ranking key.
    if not summary_table.empty:
        summary_table = summary_table.sort_values(["mae_skill", "holdout_r2", "material", "target"],
                                                  ascending=[False, False, True, True], na_position="last")
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
    overview_rows = [[str(v["material"]) + ' · ' + _target_label(v), str(v.get("status") or '—'),
                      _percent(v.get("mae_skill")), _number(v.get("holdout_r2")),
                      _number(v.get("valid_target_rows"), 0) + ' / ' + _number(v.get("valid_target_groups"), 0),
                      _percent(v.get("nonzero_fraction")) + (' · mostly zero' if _true(v.get("mostly_zero_target")) else '')]
                     for v in summary_table.to_dict("records")]
    main = '<section><h2>Which deduct families are most predictable?</h2>'
    main += ('<p>Ranked by holdout <strong>MAE skill</strong>: 1 − model error / baseline error. '
             'Positive values improve on the training-derived median baseline; zero means no improvement; '
             'negative means worse. An undefined score is shown as — when baseline error is zero. '
             'R² compares predictions with the holdout mean; negative R² indicates poor fit.</p>')
    main += _table(["Surface · target", "Status", "MAE skill", "R²", "Valid rows / groups", "Nonzero"], overview_rows) if overview_rows else '<p>No target tasks were recorded.</p>'
    main += ('<p class="caption">Raw errors are shown within each target below. They are not used to rank '
             'families with different scales. A mostly zero target can have small error while remaining '
             'difficult to predict when distress occurs. “Mostly zero” flags targets with fewer than '
             '5% nonzero eligible observations; it is descriptive, not an accuracy judgment.</p></section>')
    strongest = []
    if not importance_table.empty:
        for row in summary_table.to_dict("records"):
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
        main += '<section><h2>Predictors across deduct families</h2>'
        main += _table(["Surface · target", "Interpretation", "Top predictors (MAE increase / baseline MAE)"], strongest)
        main += ('<p class="caption">Percentages describe permutation error increases relative to each '
                 'target’s baseline error. They are not percentages of explained variance and need not '
                 'sum to 100%. Correlated inputs can share or mask importance; the climate-block result '
                 'helps assess their combined contribution.</p></section>')
    main += _coverage(audit) + ''.join(cards)
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
<nav aria-label="Download complete results"><a href="target_summary.csv">Target summary CSV</a><a href="feature_importance.csv">Feature importance CSV</a><a href="model_leaderboard.csv">Model comparisons CSV</a><a href="predictions.csv">Predictions CSV</a><a href="two_stage_metrics.csv">Two-stage metrics CSV</a><a href="two_stage_predictions.csv">Two-stage components CSV</a></nav></header>'''
    html += main + '<footer>This report is self-contained and makes no network requests. Outputs remain in the private run directory.</footer></body></html>'
    _path(out, "dashboard.html").write_text(html, encoding="utf-8")
