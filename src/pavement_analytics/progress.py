"""Readable, bounded progress events for an explicitly requested private run.

Only named paths, fixed event fields and selected aggregate statistics reach
the console. Raw subprocess output, exception text and configuration stay in
the detailed private logs.
"""
from __future__ import annotations

import json
import math
from numbers import Real
import re
from typing import TextIO

from .climate_progress import sanitize_climate_event


MODEL_STATUSES = ("modeled", "baseline_selected", "constant_target", "no_valid_target",
                  "insufficient_groups", "insufficient_fold_coverage", "no_features")
MATERIALS = ("PCC", "APC", "AC_AAC_ACC")
STAGE_LABELS = {"started": "Started", "completed": "Completed", "skipped": "Skipped",
                "reused": "Reused", "failed": "Failed", "interrupted": "Interrupted"}
RUN_LABELS = {"completed": "Completed", "completed_with_warnings": "Completed with warnings",
              "failed": "Failed", "interrupted": "Interrupted", "quality_gate_failed": "Quality gate failed"}
CLIMATE_LABELS = {
    "initializing": "initializing weather retrieval",
    "loading_location": "loading location",
    "cache_check": "checking weather cache",
    "cache_hit": "reusing cached weather",
    "requesting": "waiting for CDS / downloading",
    "downloaded": "weather download received",
    "parsing": "reading weather data",
    "preparing": "preparing hourly weather",
    "location_complete": "location ready",
    "location_failed": "location has no usable weather",
    "calculating": "calculating inspection climate features",
    "writing": "writing climate results",
    "complete": "climate processing complete",
    "fallback": "trying weather fallback",
}


def _quoted(value: object) -> str:
    # ensure_ascii also escapes bidi controls and terminal newlines/ESC bytes.
    return json.dumps(str(value), ensure_ascii=True)


def _number(value: object, places: int | None = None) -> str | None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return None
    if places is not None:
        return f"{value:,.{places}f}"
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,.3f}"


def _code(value: object, fallback: str) -> str:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}", value) else fallback


def _duration(value: object) -> str | None:
    if _number(value) is None or value < 0:
        return None
    minutes, seconds = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    return (f"{hours}h " if hours else "") + (f"{minutes}m " if minutes or hours else "") + f"{seconds}s"


def _climate_lines(event: dict) -> list[str]:
    safe = sanitize_climate_event(event)
    if safe is None:
        return []
    label = CLIMATE_LABELS.get(safe.get("phase"))
    if label is None:
        return []
    stage = event.get("stage")
    line = "Climate" + (f" {_quoted(stage)}" if isinstance(stage, str) and stage else "") + ": " + label
    values = []
    if "location_index" in safe and "total_locations" in safe:
        values.append(f"location={safe['location_index']:,}/{safe['total_locations']:,}")
    if "locations_completed" in safe and "total_locations" in safe:
        values.append(f"completed={safe['locations_completed']:,}/{safe['total_locations']:,}")
    if "rows_completed" in safe and "total_rows" in safe and safe["phase"] in {"calculating", "writing", "complete"}:
        values.append(f"rows={safe['rows_completed']:,}/{safe['total_rows']:,}")
    values += _counts(safe, (("cache_hits", "cache hits"), ("requests", "requests")))
    if safe["phase"] in {"location_complete", "location_failed", "complete"}:
        values += _counts(safe, (("usable_locations", "usable locations"), ("missing_locations", "missing locations")))
    if "start_year" in safe and "end_year" in safe:
        values.append(f"years={safe['start_year']}-{safe['end_year']}")
    if safe.get("dataset") in {"era5-land", "era5"}:
        values.append("source=" + {"era5-land": "ERA5-Land", "era5": "ERA5"}[safe["dataset"]])
    return [line + (" | " + "; ".join(values) if values else "")]


def _counts(stats: dict, fields: tuple[tuple[str, str], ...]) -> list[str]:
    result = []
    for key, label in fields:
        value = _number(stats.get(key))
        if value is not None:
            result.append(f"{label}={value}")
    return result


def _stage_stats(stage: str, stats: dict) -> list[str]:
    """Select numeric summaries; never dump arbitrary stage dictionaries."""
    if stage == "01_split":
        result = _counts(stats, (("input_rows", "input rows"), ("retained_rows", "accepted"),
                                 ("excluded_rows", "rejected")))
        surfaces = stats.get("surfaces", {})
        if isinstance(surfaces, dict):
            result += _counts(surfaces, tuple((s, s) for s in MATERIALS))
        return result
    if stage.startswith("02_pairs_"):
        return _counts(stats, (("total_pairs", "pairs"), ("sections_total", "histories"),
                               ("total_increases_skipped", "PCI increases skipped"),
                               ("total_duplicates_collapsed", "equal PCI collapsed")))
    if stage.startswith("03_location_rate_"):
        return _counts(stats, (("rows", "pairs"), ("location_count", "locations"),
                               ("missing_location_rows", "missing location pairs")))
    if stage.startswith("04_climate_"):
        result = _counts(stats, (("rows", "pairs"), ("rows_with_climate", "pairs with climate"),
                                 ("columns_added", "climate fields")))
        if stats.get("enabled") is False:
            result.append("climate not executed for this population")
        return result
    if stage.startswith("05_targets_"):
        return _counts(stats, (("rows", "pairs"), ("quarantined_pairs", "quarantined"),
                               ("blank_cells_zero_filled", "blank cells filled with zero"),
                               ("derived_columns", "derived columns")))
    if stage == "05_fit_curves":
        return _counts(stats, (("fitted_curves", "fitted curves"),))
    if stage.startswith("06_pci_"):
        result = _counts(stats, (("rows", "pairs"),))
        result.append("PCI calculated" if stats.get("calculated") is True else "PCI not calculated")
        for endpoint in ("start", "end"):
            values = stats.get(endpoint, {})
            if isinstance(values, dict):
                mae = _number(values.get("mae"), 3)
                if mae is not None:
                    result.append(f"{endpoint} PCI MAE={mae}")
                result += _counts(values, (("export_gaps", f"{endpoint} export gaps"),))
        return result
    if stage.startswith("07_modeling_"):
        result = _counts(stats, (("rows", "model rows"), ("rows_excluded", "excluded rows"),
                                 ("features", "features"), ("airfield_groups", "groups")))
        if isinstance(stats.get("targets"), dict):
            result.append(f"target tasks={len(stats['targets']):,}")
        if isinstance(stats.get("target_trainable"), dict):
            result.append(f"eligible target tasks={sum(v is True for v in stats['target_trainable'].values()):,}")
        return result
    if stage == "07_partitions":
        result = _counts(stats, (("declared_groups", "declared groups"),
                                 ("validation_groups", "validation groups"), ("folds", "folds")))
        holdout = stats.get("holdout_fraction")
        if _number(holdout) is not None:
            result.append(f"holdout={100 * holdout:.1f}%")
        return result
    if stage == "08_model_backend":
        return ["local models"] if stats.get("backend") == "local" else []
    if stage == "09_local_models":
        result = _counts(stats, (("targets", "target tasks"),))
        statuses = stats.get("status_counts", {})
        if isinstance(statuses, dict):
            result += _counts(statuses, tuple((s, s.replace("_", " ")) for s in MODEL_STATUSES))
        return result
    return []


class FullProgress:
    """Write each progress event immediately to the selected private sinks.

    A broken pipe or closed stream disables that sink only. Progress display
    must not abort the data pipeline; the independent sink can continue.
    """
    def __init__(self, stream: TextIO | None, log_stream: TextIO | None = None):
        self._streams = []
        for candidate in (stream, log_stream):
            if candidate is not None and all(candidate is not current for current in self._streams):
                self._streams.append(candidate)

    def __call__(self, event: dict) -> None:
        lines = self._render(event)
        if not lines:
            return
        text = "\n".join(lines) + "\n"
        for stream in self._streams[:]:
            try:
                stream.write(text)
                stream.flush()
            except (OSError, ValueError):
                self._streams.remove(stream)

    @staticmethod
    def _render(event: dict) -> list[str]:
        kind = event.get("event")
        if kind == "model":
            from .model_runtime import model_progress_text
            line = model_progress_text(event)
            # Prediction operations are frequent; the browser retains them,
            # while console/log milestones focus on fit work and exceptions.
            return [line] if line and (event.get("operation") != "predict" or event.get("status") in {"timed_out", "failed", "cancelled"}) else []
        if kind == "progress_window":
            return [f"Progress window: {_quoted(event.get('path', ''))}"]
        if kind == "climate":
            return _climate_lines(event)
        if kind == "heartbeat":
            stage = event.get("stage")
            line = "Still running" + (f" {_quoted(stage)}" if isinstance(stage, str) and stage else "")
            elapsed, idle = _duration(event.get("elapsed_seconds")), _duration(event.get("idle_seconds"))
            values = []
            if elapsed is not None:
                values.append("elapsed=" + elapsed)
            if idle is not None:
                values.append("last subprocess output=" + idle + " ago")
            return [line + (" | " + "; ".join(values) if values else "")]
        if kind == "setup_started":
            lines = ["Verifying setup for resume" if event.get("resume") else "Started setup",
                     f"  Workspace: {_quoted(event.get('workspace', ''))}"]
            if event.get("data") is not None:
                lines.append(f"  Input path: {_quoted(event['data'])}")
            return lines
        if kind == "setup_files":
            lines = []
            for key, label in (("profile", "Configuration"), ("mapping", "Mapping file"),
                               ("location_directory", "Location CSV directory"), ("cache", "ERA5 cache")):
                lines.append(f"  {label}: {_quoted(event[key]) if event.get(key) is not None else '(none found)'}")
            if event.get("cache_source") in {"default", "profile", "command_line"}:
                lines.append(f"  Cache setting: {event['cache_source'].replace('_', ' ')}")
            if event.get("cache_fallback_reason"):
                lines.append("  Cache fallback: " + _code(event["cache_fallback_reason"], "UNSUITABLE_INHERITED_CACHE"))
            return lines
        if kind == "input_started":
            index, total = _number(event.get("index")), _number(event.get("files"))
            position = f" ({index}/{total})" if index is not None and total is not None else ""
            return [f"Reading input{position}: {_quoted(event.get('path', ''))}"]
        if kind == "input":
            rows = _number(event.get("rows"))
            line = f"  Input file: {_quoted(event.get('path', ''))}"
            if rows is not None:
                line += f" | rows={rows}"
            status = event.get("location_status")
            if event.get("location") is not None:
                line += f" | location file={_quoted(event['location'])}"
            line += " | " + {"mapped": "mapped", "unmapped": "UNMAPPED", "missing_file": "LOCATION FILE MISSING"}.get(status, "mapping status unknown")
            return [line]
        if kind == "setup_complete":
            stats = _counts(event, (("files", "input files"), ("rows", "input rows"),
                                    ("mapped_files", "mapped files"), ("unmapped_files", "unmapped files"),
                                    ("missing_location_files", "missing location files")))
            lines = ["Completed setup" + (" | " + "; ".join(stats) if stats else "")]
            if event.get("report") is not None:
                lines.append(f"  Setup report: {_quoted(event['report'])}")
            return lines
        if kind == "setup_issue":
            roles = {"mapping": "Mapping JSON", "csv_dir": "Location CSV directory", "cdv_curves": "CDV profile",
                     "deduct_curves": "Deduct-curve profile", "families": "Family profile"}
            states = {"missing": "not found", "not_file": "not a file", "not_directory": "not a directory",
                      "unreadable": "cannot be read"}
            if event.get("role") not in roles or event.get("status") not in states:
                return []
            return [f"Setup issue: {roles[event['role']]} {states[event['status']]}: {_quoted(event.get('path', ''))}"]
        if kind == "mapping_entry":
            status = event.get("status")
            if status not in {"invalid", "normalized"}:
                return []
            label = "Invalid mapping entry" if status == "invalid" else "Normalized mapping entry"
            source = event.get("source_key")
            line = f"{label}: source={_quoted(source if isinstance(source, str) else '(non-string)')}"
            if isinstance(event.get("location_value"), str):
                line += f" | location={_quoted(event['location_value'])}"
            line += " | " + _code(event.get("reason"), "INVALID_LOCATION_MAPPING" if status == "invalid" else "PATH_NORMALIZED")
            if status == "normalized":
                line += f" | {_quoted(event.get('source_filename', ''))} -> {_quoted(event.get('relative_location', ''))}"
            return [line]
        if kind == "setup_failed":
            return ["Failed setup: " + _code(event.get("code"), "PRIVATE_SETUP_FAILED"),
                    f"  Details: {_quoted(event.get('log', ''))}"]
        if kind == "stage":
            status = event.get("status")
            if status not in STAGE_LABELS:
                return []
            stage = event.get("stage", "")
            line = f"{STAGE_LABELS[status]} {_quoted(stage)}"
            seconds = _number(event.get("seconds"), 1)
            if seconds is not None and status != "started":
                line += f" ({seconds}s)"
            stats = event.get("stats")
            if status in {"completed", "skipped", "reused"} and isinstance(stats, dict):
                if status == "reused" and stats.get("skipped"):
                    line += " | previously skipped"
                values = _stage_stats(stage, stats)
                if values:
                    line += " | " + "; ".join(values)
            lines = [line]
            if status in {"failed", "interrupted"} and event.get("log") is not None:
                lines.append(f"  Stage log: {_quoted(event['log'])}")
            return lines
        if kind == "warning":
            line = f"Warning {_quoted(event.get('stage', ''))}: {_code(event.get('code'), 'STAGE_WARNING')}"
            count = _number(event.get("count"))
            if count is not None:
                line += f" (count={count})"
            return [line]
        if kind == "target":
            status = event.get("status")
            if status not in {"started", "completed"}:
                return []
            line = f"{STAGE_LABELS[status]} target {_quoted(event.get('material', ''))} / {_quoted(event.get('target', ''))}"
            if status == "completed" and event.get("result_status") in MODEL_STATUSES:
                line += f" | {event['result_status'].replace('_', ' ')}"
            return [line]
        if kind == "run":
            label = RUN_LABELS.get(event.get("status"))
            if label is None:
                return []
            line = label + " run"
            warnings = _number(event.get("warnings"))
            if warnings is not None:
                line += f" | warnings={warnings}"
            lines = [line]
            summary = event.get("summary", {})
            if isinstance(summary, dict):
                for material in MATERIALS:
                    metrics = summary.get(material)
                    if isinstance(metrics, dict):
                        values = _counts(metrics, (("eligible_inspections", "accepted inspections"),
                                                  ("pairs", "pairs"), ("model_rows", "model rows"),
                                                  ("features", "features"), ("airfield_groups", "groups")))
                        if values:
                            lines.append(f"  {material}: " + "; ".join(values))
            if event.get("report") is not None:
                lines.append(f"  Run report: {_quoted(event['report'])}")
            return lines
        if kind == "output":
            return [f"  {label}: {_quoted(event[key])}" for key, label in
                    (("dashboard", "Dashboard"), ("models", "Models"), ("report", "Run report"), ("qualification", "Qualification"))
                    if event.get(key) is not None]
        return []
