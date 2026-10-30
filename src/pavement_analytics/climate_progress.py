"""Bounded worker progress records without coordinates or raw diagnostics."""
from __future__ import annotations

import json
from numbers import Integral
import sys


PROGRESS_PREFIX = "PAVEMENT_PROGRESS "
PHASES = frozenset({"initializing", "loading_location", "cache_check", "cache_hit", "requesting",
                    "downloaded", "parsing", "preparing", "location_complete", "location_failed",
                    "calculating", "writing", "complete", "fallback"})
INTEGER_FIELDS = frozenset({"location_index", "total_locations", "locations_completed", "usable_locations",
                            "missing_locations", "cache_hits", "requests", "rows_completed", "total_rows",
                            "start_year", "end_year"})


def sanitize_climate_event(event: object) -> dict | None:
    """Drop unknown fields and reject malformed known fields before display."""
    if not isinstance(event, dict) or event.get("event") != "climate":
        return None
    phase = event.get("phase")
    if not isinstance(phase, str) or phase not in PHASES:
        return None
    safe = {"event": "climate", "phase": phase}
    for key in INTEGER_FIELDS:
        if key in event:
            value = event[key]
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 2 ** 53:
                return None
            safe[key] = int(value)
    if "dataset" in event:
        if not isinstance(event["dataset"], str) or event["dataset"] not in {"era5-land", "era5"}:
            return None
        safe["dataset"] = event["dataset"]
    return safe


def parse_progress_line(line: str) -> dict | None:
    """Read only our prefixed bounded JSON records, never infer raw log text."""
    if not isinstance(line, str) or len(line) > 8192 or not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        event = json.loads(line[len(PROGRESS_PREFIX):])
    except (ValueError, RecursionError):
        return None
    return sanitize_climate_event(event)


class ClimateProgress:
    """Emit cumulative counters to the worker's captured standard output."""

    def __init__(self, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.state = {key: 0 for key in ("locations_completed", "usable_locations", "missing_locations",
                                        "cache_hits", "requests", "rows_completed")}

    def increment(self, field: str) -> int:
        if field not in INTEGER_FIELDS:
            raise ValueError("UNKNOWN_CLIMATE_PROGRESS_COUNTER")
        self.state[field] = self.state.get(field, 0) + 1
        return self.state[field]

    def update(self, phase: str, **fields) -> None:
        safe = sanitize_climate_event({**self.state, **fields, "event": "climate", "phase": phase})
        if safe is None:
            raise ValueError("INVALID_CLIMATE_PROGRESS_EVENT")
        self.state = safe
        print(PROGRESS_PREFIX + json.dumps(safe, allow_nan=False, separators=(",", ":")),
              file=self.stream, flush=True)
