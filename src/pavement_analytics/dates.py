"""Deterministic, source-wide inspection date normalization for automatic setup."""

from __future__ import annotations

from collections import Counter
from datetime import date
import re

import pandas as pd

from .security import SafeError


_CLOCK = (r"(?:[ T](?P<hour>\d{1,2}):(?P<minute>\d{2})"
          r"(?::(?P<second>\d{2})(?:\.(?P<fraction>\d{1,9}))?)?"
          r"(?: ?(?P<ampm>AM|PM))?)?")
_YEAR_FIRST = re.compile(
    r"(?P<year>\d{4})(?P<sep>[-/])(?P<month>\d{1,2})(?P=sep)(?P<day>\d{1,2})"
    + _CLOCK, re.I)
_YEAR_LAST = re.compile(
    r"(?P<first>\d{1,2})(?P<sep>[-/])(?P<second_date>\d{1,2})(?P=sep)(?P<year>\d{4})"
    + _CLOCK, re.I)


def _calendar(year: int, month: int, day: int) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return ""


def _valid_clock(match: re.Match) -> bool:
    if match["hour"] is None:
        return True
    hour, minute = int(match["hour"]), int(match["minute"])
    second = int(match["second"] or 0)
    return (0 <= minute <= 59 and 0 <= second <= 59
            and (1 <= hour <= 12 if match["ampm"] else 0 <= hour <= 23))


def _interpret(value: str) -> tuple[str, str, str]:
    """Return a fixed classification and the possible MDY/DMY ISO results."""
    if not value:
        return "blank", "", ""
    match = _YEAR_FIRST.fullmatch(value)
    if match and _valid_clock(match):
        result = _calendar(int(match["year"]), int(match["month"]), int(match["day"]))
        return ("year_first", result, result) if result else ("invalid", "", "")
    match = _YEAR_LAST.fullmatch(value)
    if match and _valid_clock(match):
        year, first, second = int(match["year"]), int(match["first"]), int(match["second_date"])
        mdy, dmy = _calendar(year, first, second), _calendar(year, second, first)
        if mdy and dmy:
            return ("same_in_both_orders" if mdy == dmy else "ambiguous"), mdy, dmy
        if mdy:
            return "month_first_only", mdy, ""
        if dmy:
            return "day_first_only", "", dmy
    return "invalid", "", ""


def normalize_dates(values: pd.Series, *, diagnostics: dict) -> pd.Series:
    """Normalize recognized dates without guessing an unresolved numeric order.

    Calendar-valid numeric dates that resolve only one way provide source-wide
    MDY/DMY evidence. Conflicting evidence or unresolved ambiguity stops setup.
    Unsupported values (including offsets, serial numbers and two-digit years)
    become blank so existing row validation rejects them. Valid clock fields are
    discarded: an inspection is a calendar date, not a timestamp. Time zones are
    deliberately unsupported, so no conversion can shift the inspection date.
    """
    # Parse unique strings once; inspection exports often repeat dates thousands
    # of times. The normalized input never changes the original source snapshot.
    def clean(value):
        return "" if pd.isna(value) else " ".join(str(value).split())

    cleaned = values.map(clean)
    interpreted = {value: _interpret(value) for value in cleaned.unique()}
    counts = Counter()
    for value, count in cleaned.value_counts(dropna=False).items():
        counts[interpreted[value][0]] += int(count)
    mdy_only, dmy_only = counts["month_first_only"], counts["day_first_only"]
    date_order = "month_first" if mdy_only else "day_first" if dmy_only else "not_required"
    status, error = "normalized", None
    if mdy_only and dmy_only:
        date_order, status = "conflicting", "failed"
        error = "MIXED_DATE_FORMATS_USE_DATE_FORMAT"
    elif counts["ambiguous"] and not (mdy_only or dmy_only):
        date_order, status = "unresolved", "failed"
        error = "AMBIGUOUS_DATE_FORMAT_USE_DATE_FORMAT"
    diagnostics.update({
        "policy": "strict_source_wide_calendar_dates",
        "status": status,
        "date_order": date_order,
        "timestamp_policy": "Validate clock fields and preserve the printed calendar date; time zones unsupported",
        "input_rows": len(values),
        "blank_rows": counts["blank"],
        "invalid_rows": counts["invalid"],
        "ambiguous_rows": counts["ambiguous"],
        "parsed_rows": 0 if error else len(values) - counts["blank"] - counts["invalid"],
        "format_counts": {key: counts[key] for key in (
            "year_first", "same_in_both_orders", "month_first_only", "day_first_only", "ambiguous")},
        "error_code": error,
    })
    if error:
        raise SafeError(error)
    selected = {value: parsed[2] if date_order == "day_first" else parsed[1]
                for value, parsed in interpreted.items()}
    return cleaned.map(selected)
