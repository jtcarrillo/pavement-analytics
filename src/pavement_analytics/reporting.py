"""Consolidated tables and an offline dashboard for a private analysis run.

Only generated outputs are read. No serialized estimator is loaded, and the
HTML has no network dependencies. Call before the run's checksum manifest is
written so these reports are covered by immutable-run verification.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .io import read_json
from .metrics import score
from .security import SafeError, reject_links

SUMMARY_COLUMNS = [
    "material", "target", "status", "candidate", "available_rows", "valid_target_rows",
    "development_rows", "holdout_rows", "excluded_missing_target", "holdout_wMAE",
    "holdout_wRMSE", "holdout_weighted_R2", "condition_wMAE", "condition_wRMSE",
    "improvement_percent",
]
LEADER_COLUMNS = ["material", "target", "candidate", "cv_mean_wMAE", "cv_mean_wRMSE",
                  "wins_vs_condition", "selected"]
PREDICTION_COLUMNS = ["material", "target", "partition", "candidate", "pair_id", "group_id",
                      "fold", "actual", "prediction", "condition_prediction", "weight"]
SCATTER_LIMIT = 600


def _file(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts)
    reject_links(path)
    if root.resolve() not in path.resolve().parents:
        raise SafeError("INVALID_REPORT_PATH")
    return path


def _text(value) -> str:
    return escape(str(value), quote=True)


def _number(value, decimals=3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def _scatter(frame: pd.DataFrame, target: str) -> str:
    if frame.empty:
        return '<p class="empty">No holdout predictions available.</p>'
    values = frame[["actual", "prediction"]].to_numpy(float)
    low, high = float(values.min()), float(values.max())
    pad = (high - low) * .06 if high > low else max(1., abs(low) * .1)
    low, high = low - pad, high + pad
    # All predictions are exported. Only the HTML display is capped.
    sampled = frame.iloc[np.linspace(0, len(frame) - 1, min(SCATTER_LIMIT, len(frame)), dtype=int)]
    def x(value):
        return 68 + (float(value) - low) / (high - low) * 390
    def y(value):
        return 286 - (float(value) - low) / (high - low) * 244
    chunks = ['<svg viewBox="0 0 510 340" role="img" aria-label="Actual versus predicted holdout values">',
              '<rect x="68" y="42" width="390" height="244" fill="#f3f6fa"/>']
    for value in np.linspace(low, high, 5):
        px, py = x(value), y(value)
        chunks.extend([f'<line x1="68" y1="{py:.2f}" x2="458" y2="{py:.2f}" stroke="#dce4ef"/>',
                       f'<text x="58" y="{py+4:.2f}" text-anchor="end">{_number(value, 2)}</text>',
                       f'<text x="{px:.2f}" y="307" text-anchor="middle">{_number(value, 2)}</text>'])
    chunks.append('<line x1="68" y1="286" x2="458" y2="42" stroke="#64748b" stroke-dasharray="6 5"/>')
    for row in sampled.itertuples():
        chunks.append(f'<circle cx="{x(row.actual):.2f}" cy="{y(row.prediction):.2f}" r="3.2" '
                      f'fill="#137a77" opacity=".55"><title>Actual {_number(row.actual)}; '
                      f'predicted {_number(row.prediction)}</title></circle>')
    chunks.extend(['<text x="263" y="334" text-anchor="middle">Actual rate (points/year)</text>',
                   '<text transform="translate(16 164) rotate(-90)" text-anchor="middle">Predicted rate</text>',
                   '</svg>'])
    chunks.append(f'<p class="caption">{len(sampled):,} of {len(frame):,} holdout points shown; '
                  'dashed line indicates a perfect prediction. All predictions are in the CSV.</p>')
    return "".join(chunks)


def _bars(row: dict) -> str:
    actual, baseline = row.get("holdout_wMAE"), row.get("condition_wMAE")
    if actual is None or baseline is None or pd.isna(actual) or pd.isna(baseline):
        return '<p class="empty">No independent holdout score for this task.</p>'
    maximum = max(actual, baseline, .000001)
    chunks = ['<svg viewBox="0 0 510 160" role="img" aria-label="Holdout weighted mean absolute error">']
    for name, value, fill, top in [("Selected", actual, "#137a77", 30),
                                   ("Condition", baseline, "#a1afc3", 86)]:
        width = value / maximum * 284
        chunks.extend([f'<text x="10" y="{top+22}">{name}</text>',
                       f'<rect x="100" y="{top}" width="{width:.2f}" height="32" rx="4" fill="{fill}"/>',
                       f'<text x="{110+width:.2f}" y="{top+22}">{_number(value)}</text>'])
    chunks.append('<text x="100" y="149">wMAE (points/year); lower is better</text></svg>')
    return "".join(chunks)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(
        f'<th scope="col">{_text(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{_text(cell)}</td>' for cell in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'



def _setup_notes(root: Path) -> str:
    path = _file(root, "setup_report.json")
    if not path.is_file():
        return ""
    setup = read_json(path)
    features = setup.get("features", [])
    notes = '<section class="overview"><h2>Inputs and modeling assumptions</h2>'
    notes += '<p>Predictors used: <strong>' + _text(', '.join(map(str, features)) or 'See run configuration') + '</strong>.</p>'
    if features == ["PCI_start"]:
        notes += ('<p>This run uses starting PCI alone. It compares condition-based predictive models; '
                  'it does not test climate, loading, material properties, or other independent predictors. '
                  'Additional justified predictors can be declared in private options.</p>')
    notes += ('<p>Family totals sum the observed exported code/severity columns. Different source inventories '
              'may represent different coverage; review the inventory before comparing family results. '
              'An unexported severity is not assumed to be an observed zero.</p>')
    groups = setup.get("grouping", [])
    group_rows = [[str(g.get("input", "")), str(g.get("strategy", "")),
                   str(g.get("source_column") or '—'), str(g.get("limitation") or '—')] for g in groups]
    if any(g.get("strategy") == "section_identity" for g in groups):
        notes += ('<p><strong>Section-level grouping was used for at least one source.</strong> '
                  'Repeated inspections of a section stay together, but correlated sections may remain '
                  'in different partitions. This does not establish generalization to new independent networks.</p>')
    inventory = []
    for source in setup.get("sources", []):
        deducts = source.get("deduct_columns") or sorted(set(source.get("deduct_header_mapping", {}).values()))
        inventory.append([str(source.get("original_path", source.get("snapshot", ""))),
                          str(source.get("rows", "")), ', '.join(map(str, deducts))])
    notes += '<details><summary>Source coverage and validation groups</summary>'
    notes += _table(['Source', 'Rows', 'Exported deduct columns'], inventory)
    notes += _table(['Input', 'Grouping strategy', 'Group column', 'Limitation'], group_rows)
    notes += '<p><a class="private-link" href="setup_report.json">Full private setup report</a></p></details></section>'
    return notes


def _deduct_notes(root: Path) -> str:
    path = _file(root, "deduct_audit.json")
    if not path.is_file():
        return ""
    audit = read_json(path)
    notes = '<section class="overview"><h2>Deduct target availability</h2>'
    policy = audit.get("blank_deducts", "missing")
    explanation = {
        "zero": "Empty exported deduct cells were explicitly treated as zero.",
        "missing": "Empty exported deduct cells were kept as missing and can exclude deduct targets.",
    }.get(policy, "See the deduct audit for the blank-cell policy.")
    notes += '<p><strong>Blank deduct policy: ' + _text(policy) + '.</strong> ' + _text(explanation) + '</p>'
    targets = audit.get("targets", [])
    if targets and all(target.get("valid_target_rows", 0) == 0 for target in targets):
        notes += ('<p class="empty"><strong>All deduct-family tasks have zero valid target rows.</strong> '
                  'Model fitting was skipped for these tasks. Review the blank-cell policy and audit counts below.</p>')
    rows = []
    for key, label in (
        ("blank_deduct_cells", "Empty deduct cells"),
        ("zero_filled_deduct_cells", "Empty deduct cells treated as zero"),
        ("invalid_deduct_pairs", "Pairs with invalid deducts"),
        ("ambiguous_deduct_pairs", "Pairs with overlapping deduct representations"),
        ("export_gap_pairs", "Pairs with a deduct export gap"),
    ):
        counts = audit.get(key, {})
        rows.append([label, _number(counts.get("start"), 0), _number(counts.get("end"), 0)])
    notes += _table(["Measure", "Start endpoint", "End endpoint"], rows)
    notes += ('<p class="caption">Counts describe endpoints of the retained inspection pairs. '
              'An inspection shared by consecutive pairs is counted again. '
              'A blank-as-zero policy does not fill unexported columns or repair malformed values.</p>')
    ungraded = audit.get("ungraded_deduct_columns", [])
    if ungraded:
        notes += ('<p>Recognized deduct columns without an L/M/H severity: '
                  + _text(', '.join(map(str, ungraded))) + '. '
                  'Positive ungraded and severity values for the same code require reconciliation; '
                  'affected pairs are excluded rather than adding potentially overlapping values.</p>')
    ignored = audit.get("ignored_ungraded_deduct_columns", [])
    if ignored:
        notes += ('<p>Deduct columns without an L/M/H severity were excluded from family sums: '
                  + _text(', '.join(map(str, ignored))) + '.</p>')
    notes += '<p><a class="private-link" href="deduct_audit.json">Full private deduct audit</a></p></section>'
    return notes


def build_report(run_dir: Path, slots: list[dict], accounting: dict) -> None:
    """Write four consolidated artifacts inside an existing private run directory.

    Holdout scores are recomputed from saved prediction rows, not averaged across
    tasks. Different target units/populations are never pooled into one score.
    Every configured slot remains visible, including unavailable and skipped ones.
    """
    run_dir = Path(run_dir)
    reject_links(run_dir)
    if not run_dir.is_dir():
        raise SafeError("REPORT_RUN_DIRECTORY_REQUIRED")
    for slot in slots:
        if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(slot.get("target", "")))
                or not re.fullmatch(r"[A-Za-z0-9_-]+", str(slot.get("material", "")))):
            raise SafeError("INVALID_REPORT_TASK")
    wanted = {"material", *(s["target"] for s in slots)}
    dataset_path = _file(run_dir, "dataset.csv")
    dataset = (pd.read_csv(dataset_path, usecols=lambda c: c in wanted)
               if dataset_path.is_file() else pd.DataFrame())
    summaries, leaders, predictions, cards = [], [], [], []
    seen = set()
    for slot in slots:
        material, target = slot["material"], slot["target"]
        if (material, target) in seen:
            raise SafeError("DUPLICATE_REPORT_TASK")
        seen.add((material, target))
        row = {column: slot.get(column) for column in SUMMARY_COLUMNS}
        if "material" in dataset:
            subset = dataset.loc[dataset.material.eq(material)]
            valid = int(np.isfinite(pd.to_numeric(subset[target], errors="coerce")).sum()) if target in subset else 0
            row.update(available_rows=len(subset), valid_target_rows=valid,
                       excluded_missing_target=len(subset) - valid)
        holdout = pd.DataFrame()
        task_leaders = pd.DataFrame()
        # Only completed modeling slots have task files. Stale files must not turn
        # a skipped slot into a successful one in a regenerated report.
        if slot.get("status") in {"champion", "no_model"}:
            holdout_path = _file(run_dir, "materials", material, target, "holdout_predictions.csv")
            if not holdout_path.is_file():
                raise SafeError("REPORT_PREDICTIONS_MISSING")
            holdout = pd.read_csv(holdout_path, dtype={"group_id": str})
            required = {"pair_id", "group_id", "delta_days", target, "prediction", "condition_prediction"}
            if not required <= set(holdout):
                raise SafeError("REPORT_PREDICTIONS_INVALID")
            holdout = holdout.rename(columns={target: "actual", "delta_days": "weight"})
            selected = score(holdout.actual, holdout.prediction, holdout.weight)
            baseline = score(holdout.actual, holdout.condition_prediction, holdout.weight)
            row.update(holdout_rows=len(holdout), holdout_wMAE=selected["wMAE"],
                       holdout_wRMSE=selected["wRMSE"], holdout_weighted_R2=selected["weighted_R2"],
                       condition_wMAE=baseline["wMAE"], condition_wRMSE=baseline["wRMSE"],
                       improvement_percent=(100 * (baseline["wMAE"] - selected["wMAE"]) / baseline["wMAE"]
                                            if baseline["wMAE"] > 0 else None))
            held_export = holdout.assign(material=material, target=target, partition="holdout",
                                         candidate=slot.get("candidate", "condition"), fold=-1)
            predictions.append(held_export.reindex(columns=PREDICTION_COLUMNS))
            leaderboard_path = _file(run_dir, "materials", material, target, "leaderboard.csv")
            if leaderboard_path.is_file():
                task_leaders = pd.read_csv(leaderboard_path).rename(columns={"wMAE": "cv_mean_wMAE", "wRMSE": "cv_mean_wRMSE"})
                task_leaders["material"], task_leaders["target"] = material, target
                task_leaders["selected"] = task_leaders.candidate.eq(slot.get("candidate"))
                leaders.append(task_leaders.reindex(columns=LEADER_COLUMNS))
            oof_path = _file(run_dir, "materials", material, target, "oof_predictions.csv")
            if oof_path.is_file():
                oof = pd.read_csv(oof_path, dtype={"group_id": str})
                oof["material"], oof["target"], oof["partition"] = material, target, "development_oof"
                predictions.append(oof.reindex(columns=PREDICTION_COLUMNS))
        summaries.append(row)
        status = str(row.get("status", "unknown"))
        candidate = str(row.get("candidate") or "—")
        explanation = {
            "no_model": "No candidate met the development-CV improvement rule. The condition baseline was retained.",
            "no_valid_target": "Model fitting was skipped because no valid observations remained for this target; check target availability and deduct audit flags.",
            "insufficient_groups": "Too few independent groups for both development folds and a reserved holdout.",
            "modeling_disabled": "Model fitting was disabled for this run.",
            "champion": "Candidate selected using development cross-validation; assessed on reserved groups below.",
        }.get(status, "See the task status and run report for details.")
        card = (f'<article class="task" data-material="{_text(material)}" data-target="{_text(target)}">'
                f'<div class="task-heading"><div><p class="eyebrow">{_text(material)}</p>'
                f'<h2>{_text(target)}</h2></div><span class="badge">{_text(status)}</span></div>'
                f'<p>{_text(explanation)}</p><p class="caption">Selected candidate: {_text(candidate)}. '
                f'Valid rows: {_number(row.get("valid_target_rows"), 0)}; '
                f'missing target: {_number(row.get("excluded_missing_target"), 0)}; '
                f'development: {_number(row.get("development_rows"), 0)}; holdout: {_number(row.get("holdout_rows"), 0)}.</p>'
                '<div class="charts"><section><h3>Independent holdout error</h3>' + _bars(row) +
                f'<p class="caption">Improvement over condition: {_number(row.get("improvement_percent"), 1)}%. '
                'Negative means worse; undefined when baseline error is zero.</p></section>'
                '<section><h3>Holdout predictions</h3>' + _scatter(holdout, target) + '</section></div>')
        if len(task_leaders):
            table_rows = [[r.candidate, _number(r.cv_mean_wMAE), _number(r.cv_mean_wRMSE),
                           _number(r.wins_vs_condition, 0), "Yes" if r.selected else ""]
                          for r in task_leaders.itertuples()]
            card += '<details><summary>Development cross-validation leaderboard</summary><p class="caption">'
            card += 'Mean fold scores used for selection; these are not independent holdout scores.</p>'
            card += _table(["Candidate", "CV wMAE", "CV wRMSE", "Wins vs condition", "Selected"], table_rows) + '</details>'
        cards.append(card + '</article>')
    summary_frame = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    leader_frame = pd.concat(leaders, ignore_index=True) if leaders else pd.DataFrame(columns=LEADER_COLUMNS)
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=PREDICTION_COLUMNS)
    summary_frame.to_csv(_file(run_dir, "target_summary.csv"), index=False)
    leader_frame.to_csv(_file(run_dir, "model_leaderboard.csv"), index=False)
    prediction_frame.to_csv(_file(run_dir, "predictions.csv"), index=False)
    materials = sorted({s["material"] for s in slots})
    options = '<option value="">All materials</option>' + ''.join(
        f'<option value="{_text(m)}">{_text(m)}</option>' for m in materials)
    status_counts = pd.Series([s.get("status", "unknown") for s in slots], dtype=str).value_counts()
    counts_html = ''.join(f'<span><strong>{int(count):,}</strong> {_text(status)}</span>' for status, count in status_counts.items())
    accounting_rows = [[str(key).replace("_", " "), str(value)] for key, value in accounting.items()]
    table_rows = [[str(r.get("material")), str(r.get("target")), str(r.get("status")),
                   _number(r.get("valid_target_rows"), 0), _number(r.get("holdout_wMAE")),
                   _number(r.get("condition_wMAE")), _number(r.get("improvement_percent"), 1)] for r in summaries]
    setup_notes = _setup_notes(run_dir)
    deduct_notes = _deduct_notes(run_dir)
    html = _HTML_START + f'''<header><p class="eyebrow">PRIVATE ANALYSIS · OFFLINE REPORT</p>
<h1>Deduct family results</h1><p class="intro">Compare each material and target, inspect independent holdout accuracy,
and trace every result to the consolidated tables.</p><nav><a href="target_summary.csv">Target summary CSV</a>
<a href="model_leaderboard.csv">CV leaderboard CSV</a><a href="predictions.csv">Full predictions CSV</a>
<a href="run_report.json">Run report</a></nav></header><main>
<section class="overview"><h2>Run overview</h2><div class="counts">{counts_html or '<span>No target tasks</span>'}</div>
<p class="caption"><strong>no_valid_target:</strong> skipped before fitting.
<strong>no_model:</strong> condition baseline retained.
<strong>champion:</strong> candidate selected by development cross-validation; inspect its holdout results.</p>
<p>Deduct targets measure annual change in summed family deducts (positive means growth). PCI-rate deterioration is negative.
Errors are weighted by interval days. Scores from different families or heldout populations are not directly interchangeable.</p>
<p>Outputs remain private. This report makes no network requests.</p>
<details><summary>Inspection accounting</summary>{_table(['Measure', 'Count'], accounting_rows)}</details>
<details><summary>All target metrics</summary>{_table(['Material', 'Target', 'Status', 'Valid rows', 'Holdout wMAE', 'Condition wMAE', 'Improvement %'], table_rows)}</details></section>
{deduct_notes}{setup_notes}<div class="filters"><label>Material <select id="material">{options}</select></label>
<label>Find target <input id="target" type="search" placeholder="Filter by target name"></label>
<span id="visible" aria-live="polite"></span></div><div id="tasks">{''.join(cards) or '<p class="empty">No target tasks were recorded.</p>'}</div>
<p id="no-match" hidden>No targets match these filters.</p>
<footer>Open individual task folders for fold assignments, prediction tables, and the saved estimator.
The saved estimator uses development groups only. Do not tune models against this holdout repeatedly.</footer></main>''' + _HTML_END
    _file(run_dir, "dashboard.html").write_text(html, encoding="utf-8")


_HTML_START = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; img-src data:; base-uri 'none'; form-action 'none'">
<title>Deduct family results</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf2f7;color:#1b2c43;font:16px/1.55 system-ui,-apple-system,sans-serif}
header{background:#112d43;color:#f5f9ff;padding:52px max(6vw,24px) 38px}h1{font-size:clamp(32px,4vw,48px);margin:8px 0}
h2{font-size:23px;margin:0 0 12px;overflow-wrap:anywhere}h3{font-size:16px;margin:16px 0 0}.intro{max-width:760px;color:#d4e2ee}
.eyebrow{font-size:12px;font-weight:750;letter-spacing:.14em;margin:0;color:#4b8f98}header .eyebrow{color:#7fddd4}
nav{display:flex;flex-wrap:wrap;gap:12px 22px;margin-top:24px}a.private-link{color:#137a77}a{color:#a6eee4;text-decoration:underline;text-underline-offset:4px}
main{max-width:1240px;margin:auto;padding:26px 24px}.overview,.task{background:white;border:1px solid #dce5ef;border-radius:14px;padding:26px;margin-bottom:22px;box-shadow:0 5px 18px #12304a05}
.counts{display:flex;gap:12px;flex-wrap:wrap}.counts span{background:#eef4f8;padding:9px 14px;border-radius:6px}.counts strong{font-size:22px;margin-right:6px}
.task-heading{display:flex;justify-content:space-between;align-items:start;gap:18px}.badge{background:#e7f3ef;color:#186254;border-radius:20px;padding:5px 12px;font-size:13px;white-space:nowrap}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:30px}svg{width:100%;height:auto;display:block}svg text{font:12px system-ui,sans-serif;fill:#485c75}
.caption{font-size:13px;color:#597089}.empty{padding:25px;background:#f5f8fb;color:#597089;border-radius:8px}.filters{display:flex;gap:20px;align-items:end;flex-wrap:wrap;margin:28px 0}
label{display:grid;gap:6px;font-weight:650;font-size:14px}select,input{font:inherit;border:1px solid #bccbdc;border-radius:6px;padding:9px 12px;background:white;color:#1b2c43;min-width:180px}
#visible{font-size:14px;color:#597089;align-self:center}details{margin-top:18px}summary{cursor:pointer;font-weight:650}.table-wrap{overflow:auto;margin-top:14px}table{border-collapse:collapse;width:100%;font-size:13px;text-align:left}
th,td{padding:10px 12px;border-bottom:1px solid #dfe7f0;white-space:nowrap}th{background:#f3f7fa}footer{font-size:13px;color:#597089;padding:12px 0 30px}
[hidden]{display:none!important}@media(max-width:700px){.charts{grid-template-columns:1fr;gap:0}.task,.overview{padding:18px}.task-heading{display:block}.badge{display:inline-block;margin-bottom:6px}}
@media print{body{background:white}.filters,nav{display:none}.task{break-inside:avoid;box-shadow:none}header{background:white;color:#1b2c43;padding:16px}header .intro{color:#597089}main{padding:0}}
</style></head><body>'''

_HTML_END = '''<script>
const material = document.getElementById('material');
const target = document.getElementById('target');
const cards = Array.from(document.querySelectorAll('.task'));
function filterTasks() {
  const query = target.value.trim().toLowerCase();
  let shown = 0;
  for (const card of cards) {
    card.hidden = !!((material.value && card.dataset.material !== material.value) ||
      (query && !card.dataset.target.toLowerCase().includes(query)));
    if (!card.hidden) shown++;
  }
  document.getElementById('visible').textContent = shown + ' of ' + cards.length + ' tasks';
  document.getElementById('no-match').hidden = shown > 0 || cards.length === 0;
}
material.addEventListener('change', filterTasks);
target.addEventListener('input', filterTasks);
filterTasks();
</script></body></html>'''
