#!/usr/bin/env python3
"""Original interval ERA5 worker, retained for numerical compatibility.

This module preserves the earlier local-solar-day statistics, pavement-grade
calculations, interval cooling/rainflow exposures, construction-age thermal
integral, nearby-land snapping and ERA5 fallback. These are retrospective
interval features, not a forecast-time feature contract. All input, cache,
output and worker logs belong in an external private workspace.

Use through the full pipeline, which captures detailed worker output privately.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import time
import zipfile
from functools import reduce
from pathlib import Path

try:
    import cdsapi
except ImportError:  # Pure calculation functions remain usable without service extras.
    cdsapi = None
import numpy as np
import pandas as pd

from ..security import SafeError, disjoint, refuse_public_execution, reject_links
from ..climate_progress import ClimateProgress
from . import check_cli_paths


DATASET = "reanalysis-era5-land-timeseries"
DATASET_ERA5 = "reanalysis-era5-single-levels-timeseries"
DEFAULT_START_YEAR = 1980
ROUND_DECIMALS = 4  # ~11 m at the equator; plenty for ERA5-Land's ~9 km grid

# ERA5-Land grid is 0.1 deg; ERA5 is 0.25 deg.
ERA5_LAND_GRID_DEG = 0.1
ERA5_GRID_DEG = 0.25

EXTREME_NDAYS_HIGH = 7  # Eq. 8 uses the 7-day mean of daily highs per year
RAINFLOW_THRESHOLDS_C = (10.0, 20.0)  # cycle-range bins for the count features

# Arrhenius constants for cumulative thermal aging.
GAS_CONSTANT_J_PER_MOL_K = 8.314
THERMAL_AGING_EA_DEFAULT = 75000.0  # J/mol; mid-range asphalt oxidative aging

# Stat name -> output column label. Final column = prefix + label + suffix.
STAT_COLUMNS = [
    ("interval_start",    "IntervalStart"),
    ("interval_end",      "IntervalEnd"),
    ("interval_years",    "IntervalYears"),
    ("mean_f",            "MeanTempF"),
    ("daily_min_avg_f",   "DailyMinAvgF"),
    ("daily_max_avg_f",   "DailyMaxAvgF"),
    ("daily_range_avg_f", "DailyRangeAvgF"),
    ("abs_min_f",         "AbsMinF"),
    ("abs_max_f",         "AbsMaxF"),
    ("pav_min_f",         "PavMinF"),
    ("cooling_3h_f",      "Cooling3hF"),
    ("cooling_6h_f",      "Cooling6hF"),
    ("cooling_12h_f",     "Cooling12hF"),
    ("cool_events_6h",    "CoolEvents6h"),         # count over interval
    ("cool_events_6h_per_year", "CoolEvents6hPerYear"),  # rate (count / years)
    ("rainflow_cycles_10c",         "RainflowCycles10C"),
    ("rainflow_cycles_10c_per_year","RainflowCycles10CPerYear"),
    ("rainflow_cycles_20c",         "RainflowCycles20C"),
    ("rainflow_cycles_20c_per_year","RainflowCycles20CPerYear"),
    ("rainflow_damage",             "RainflowDamage"),
    ("rainflow_damage_per_year",    "RainflowDamagePerYear"),
    ("max_cooling_excursion_f",     "MaxCoolingExcursionF"),
    ("thermal_aging_phi",           "ThermalAgingPhi"),
    ("thermal_aging_teff_f",        "ThermalAgingTeffF"),
    ("thermal_aging_missing_years", "ThermalAgingMissingYears"),
    ("ltpp_air_low_c",    "LTPPAirLowC"),    # mean annual 1-day min, deg C
    ("ltpp_air_high_c",   "LTPPAirHighC"),   # mean annual 7-day max, deg C
    ("pav_design_low_c",  "PavDesignLowC"),  # LTPP Eq.7 result
    ("pav_design_high_c", "PavDesignHighC"), # LTPP Eq.8 result
    ("pav_pg_grade",      "PavPGGrade"),     # e.g. "PG 64-22"
    ("pg_low_normal_f",   "PGLowNormalF"),   # 20-yr P_0.02 of daily SHRP pav-min, deg F
    ("pg_low_interval_f", "PGLowIntervalF"), # coldest interval pav-min, deg F
    ("delta_pg_low_f",    "DeltaPGLowF"),    # interval − normal; negative = exceedance
    ("pg_window_years",   "PGWindowYears"),  # actual years used for the percentile
    ("data_source",       "DataSource"),     # "era5-land" | "era5"
    ("snap_distance_cells", "SnapDistanceCells"),  # 0 if original point used
    ("snap_distance_km",  "SnapDistanceKm"), # 0.0 if original; ~11 km per cell at equator
]

# Variables to request from CDS. Both stay in one request -> one ZIP per
# location. open_dataset/merge_frames already handles the multi-file ZIP.
ERA5_VARIABLES = ["2m_temperature", "skin_temperature"]

# Cooling-rate windows (hours). Each becomes a column above.
COOLING_WINDOWS_HRS = [3, 6, 12]

# SHRP pavement min-temperature relation (SUPERPAVE / LTPPBind family):
#     T_pav_min = 0.859 * T_air_min + 1.7    (both in deg C)
# T_air_min is the daily minimum air temperature in *local* time. Apply to
# each local-time day, then take the min over the interval.
PAV_MIN_SLOPE_C = 0.859
PAV_MIN_INTERCEPT_C = 1.7

# LTPP pavement temperature models (deg C). Inputs: T_air in deg C, latitude
# in degrees, depth H in mm.
#   Low  (Eq. 7):  T_pav = -1.56 + 0.72*T_air - 0.004*Lat^2 + 6.26*log10(H+25)
#   High (Eq. 8):  T_pav = 54.32 + 0.78*T_air - 0.0025*Lat^2 - 15.14*log10(H+25)
# Calibration RMSEs: low 2.1 C, high 3.0 C.
LTPP_LOW = dict(intercept=-1.56, t_air=0.72, lat2=-0.004, log_h=6.26, rmse=2.1)
LTPP_HIGH = dict(intercept=54.32, t_air=0.78, lat2=-0.0025, log_h=-15.14, rmse=3.0)

# 98% reliability shifts the prediction by ~2 sigma. The LTPPBind tool uses
# a 2-sided shift: low gets colder, high gets hotter, both by 2 * RMSE.
LTPP_RELIABILITY_SIGMA = {50: 0.0, 98: 2.0}

MERGE_KEYS_CANDIDATES = [
    ["valid_time", "latitude", "longitude"],
    ["time", "latitude", "longitude"],
    ["valid_time"],
    ["time"],
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Input CSV path")
    p.add_argument("--output", type=str, required=True, help="Output CSV path")
    p.add_argument(
        "--lat-col",
        type=str,
        default="Latitude",
        help="Name of the latitude column in the input CSV",
    )
    p.add_argument(
        "--lon-col",
        type=str,
        default="Longitude",
        help="Name of the longitude column in the input CSV",
    )
    p.add_argument(
        "--dates-col",
        type=str,
        default="Dates",
        help="Column containing a list of inspection dates per row. "
             "The interval for stats is min(dates) to max(dates).",
    )
    p.add_argument(
        "--min-ltpp-years",
        type=float,
        default=3.0,
        help="Minimum interval length (years) for LTPP design-temp stats; "
             "shorter intervals get NaN for those columns. Default 3.",
    )
    p.add_argument(
        "--col-prefix",
        type=str,
        default="",
        help="Optional prefix for every new stat column, e.g. 'ERA5_'",
    )
    p.add_argument(
        "--col-suffix",
        type=str,
        default="_1980_present",
        help="Suffix for every new stat column (default '_1980_present')",
    )
    p.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help=f"Start year for the averaging window (default {DEFAULT_START_YEAR})",
    )
    p.add_argument(
        "--end-year",
        type=int,
        default=dt.date.today().year - 1,
        help="End year (inclusive) for the averaging window. "
             "Defaults to last complete calendar year.",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="era5land_meantemp_cache",
        help="Directory to cache per-location downloads and parsed values",
    )
    p.add_argument(
        "--round-decimals",
        type=int,
        default=ROUND_DECIMALS,
        help=f"Decimal places for dedup rounding (default {ROUND_DECIMALS})",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cached downloads and re-fetch from CDS",
    )
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached stats (but reuse downloads) and recompute from ZIPs",
    )
    p.add_argument(
        "--pav-depth-low-mm",
        type=float,
        default=20.0,
        help="Depth H (mm) for LTPP low-temp equation (Eq. 7). LTPPBind's "
             "convention is 20 mm — binder failure in tension is a "
             "near-surface phenomenon at the standard reference depth. "
             "Default 20.",
    )
    p.add_argument(
        "--pav-depth-high-mm",
        type=float,
        default=0.0,
        help="Depth H (mm) for LTPP high-temp equation (Eq. 8). LTPPBind's "
             "convention is the surface (0 mm) — rutting is driven by the "
             "hottest layer. Default 0.",
    )
    p.add_argument(
        "--pav-reliability",
        type=int,
        choices=[50, 98],
        default=50,
        help="LTPPBind reliability level: 50 (raw equation) or 98 (with ±2σ "
             "shift). Default 50.",
    )
    p.add_argument(
        "--construction-year-col",
        type=str,
        default="Original Construction Year",
        help="Column with the original construction year (integer). Used as "
             "the start of the cumulative thermal aging integral. If absent, "
             "thermal-aging columns will be NaN.",
    )
    p.add_argument(
        "--thermal-aging-ea",
        type=float,
        default=THERMAL_AGING_EA_DEFAULT,
        help="Activation energy E_a (J/mol) for the Arrhenius thermal aging "
             "integral. Asphalt oxidative aging is typically 70-90 kJ/mol; "
             f"default {THERMAL_AGING_EA_DEFAULT/1000:.0f} kJ/mol.",
    )
    p.add_argument(
        "--rainflow-fatigue-beta",
        type=float,
        default=3.5,
        help="Power-law exponent for the damage-weighted rainflow sum "
             "Σ ΔT^β. Asphalt fatigue literature uses 3-4; default 3.5. "
             "Cycle-count thresholds are fixed at 10 and 20 deg C per the "
             "feature spec.",
    )
    p.add_argument(
        "--cool-event-threshold-c",
        type=float,
        default=10.0,
        help="Cooling-rate threshold (deg C over 6 h) above which a cooling "
             "event is counted. SHRP convention is 10. Default 10.",
    )
    p.add_argument(
        "--cool-event-refractory-hours",
        type=int,
        default=24,
        help="Hours after a counted cooling event during which further events "
             "are suppressed (so one storm doesn't count multiple times). "
             "Default 24.",
    )
    p.add_argument(
        "--pg-window-years",
        type=int,
        default=20,
        help="Years in the rolling window used to compute the PG-low climate "
             "normal (P_0.02 of daily pav-min). Default 20 per LTPPBind.",
    )
    p.add_argument(
        "--min-pg-window-years",
        type=int,
        default=10,
        help="Minimum years available in the PG window for the percentile to "
             "be reported. Below this, PGLowNormalF and DeltaPGLowF are NaN. "
             "Default 10.",
    )
    p.add_argument(
        "--snap-radius",
        type=int,
        default=2,
        help="If a location returns all-NaN ERA5-Land data (ocean mask), "
             "search up to this many 0.1-deg cells outward for a land cell. "
             "Set to 0 to disable. Default 2.",
    )
    p.add_argument(
        "--era5-fallback",
        action="store_true",
        default=True,
        help="If ERA5-Land snap fails, fall back to ERA5 (0.25 deg, defined "
             "globally including ocean). Default on.",
    )
    p.add_argument(
        "--no-era5-fallback",
        dest="era5_fallback",
        action="store_false",
        help="Disable the ERA5 backstop; affected sites get NaN.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# CDS download + parsing
# ---------------------------------------------------------------------------

def download_timeseries(
    client: cdsapi.Client,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    target: Path,
    force_download: bool = False,
    dataset: str = DATASET,
    progress: ClimateProgress | None = None,
) -> None:
    if target.exists() and not force_download:
        if progress is not None:
            progress.update("cache_hit", cache_hits=progress.increment("cache_hits"))
        print(f"[INFO] Reusing existing file: {target.name}")
        return

    request = {
        "variable": ERA5_VARIABLES,
        "location": {"latitude": lat, "longitude": lon},
        "date": f"{start_year}-01-01/{end_year}-12-31",
        "data_format": "csv",
    }
    if dataset == DATASET_ERA5:
        # ERA5 (not Land) requires product_type.
        request["product_type"] = ["reanalysis"]
    print(f"[INFO] Requesting {dataset} {ERA5_VARIABLES} for ({lat}, {lon}) {start_year}-{end_year}")
    if progress is not None:
        progress.update("requesting", requests=progress.increment("requests"),
                        dataset="era5" if dataset == DATASET_ERA5 else "era5-land")
    client.retrieve(dataset, request, str(target))
    if progress is not None:
        progress.update("downloaded")


def read_magic_bytes(path: Path, n: int = 8) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def choose_merge_keys(frames: list[pd.DataFrame]) -> list[str]:
    for keys in MERGE_KEYS_CANDIDATES:
        if all(all(k in df.columns for k in keys) for df in frames):
            return keys
    raise ValueError(
        "Could not find common merge keys across all CSVs. "
        f"Columns were: {[list(df.columns) for df in frames]}"
    )


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No data frames to merge")
    if len(frames) == 1:
        return frames[0]
    merge_keys = choose_merge_keys(frames)

    def _merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(left, right, on=merge_keys, how="outer")

    return reduce(_merge, frames)


def open_dataset(data_file: Path) -> pd.DataFrame:
    magic = read_magic_bytes(data_file)

    if magic.startswith(b"PK"):
        with zipfile.ZipFile(data_file, "r") as zf:
            members = [n for n in zf.namelist() if not n.endswith("/")]
            if not members:
                raise ValueError("ZIP archive is empty")
            csv_members = [n for n in members if n.lower().endswith(".csv")]
            if not csv_members:
                raise ValueError(f"No CSV found inside ZIP. Members: {members}")
            frames = []
            for member in csv_members:
                with zf.open(member) as f:
                    frames.append(pd.read_csv(f))
            return merge_frames(frames)

    if magic.startswith(b"\x1f\x8b"):
        with gzip.open(data_file, "rt", encoding="utf-8", newline="") as f:
            return pd.read_csv(f)

    return pd.read_csv(data_file)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(
        f"Could not find any of {candidates} in columns: {list(df.columns)}"
    )


def parse_dates_cell(cell) -> list[pd.Timestamp]:
    """Parse a Dates column cell into a list of timestamps. Handles JSON-style
    arrays with quoted strings, bracketed comma-separated lists without quotes,
    and pipe/semicolon-separated forms. Returns [] for empty/NaN cells.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        items = cell
    else:
        s = str(cell).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            return []
        # Strip outer brackets if present
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        # Split on the first separator that produces multiple parts
        for sep in (",", "|", ";"):
            if sep in s:
                items = s.split(sep)
                break
        else:
            items = [s]

    out: list[pd.Timestamp] = []
    for raw in items:
        if isinstance(raw, str):
            piece = raw.strip().strip("\"'")
        else:
            piece = raw
        if piece == "" or (isinstance(piece, float) and pd.isna(piece)):
            continue
        ts = pd.to_datetime(piece, errors="coerce", utc=True)
        if pd.notna(ts):
            out.append(ts)
    return out


def prepare_hourly_df(df: pd.DataFrame, lon: float) -> pd.DataFrame:
    """Build a clean hourly DataFrame from a raw ERA5-Land CSV/ZIP load.

    Returns a DataFrame indexed by UTC time, with columns:
      - temp_c        : 2m temperature in deg C
      - temp_f        : 2m temperature in deg F
      - skt_c         : skin temperature in deg C (may be NaN if missing)
      - skt_f         : skin temperature in deg F
      - local_time    : timestamp shifted by longitude-based offset
      - local_date    : date in local solar time

    Local time uses a longitude-based offset (15 deg/hour, no DST). This is
    appropriate for multi-decade climatologies; DST would move the daily
    extreme by ±1h depending on year, which is just noise here.
    """
    t2m_col = find_column(df, ["2m_temperature", "t2m"])
    skt_col = find_column(df, ["skin_temperature", "skt"])
    time_col = find_column(df, ["time", "valid_time", "date", "datetime"])

    utc_time = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    out = pd.DataFrame({
        "temp_k":     pd.to_numeric(df[t2m_col], errors="coerce"),
        "skt_k":      pd.to_numeric(df[skt_col], errors="coerce"),
    })
    out.index = utc_time
    out.index.name = "utc_time"
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out = out.dropna(subset=["temp_k"])  # drop hours with no 2m temp

    out["temp_c"] = out["temp_k"] - 273.15
    out["temp_f"] = out["temp_c"] * 9.0 / 5.0 + 32.0
    out["skt_c"] = out["skt_k"] - 273.15
    out["skt_f"] = out["skt_c"] * 9.0 / 5.0 + 32.0

    # Longitude-based local time: 15 deg per hour, wrapped to [-180, 180].
    lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
    local_offset = pd.Timedelta(hours=lon_wrapped / 15.0)
    out["local_time"] = out.index + local_offset
    out["local_date"] = out["local_time"].dt.date

    return out[["temp_c", "temp_f", "skt_c", "skt_f", "local_time", "local_date"]]


def build_daily_pav_min_c(hourly: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by local_date (datetime64) with the SHRP
    pavement-min in deg C for each local-time day.

    pav_min_c(d) = 0.859 * T_air_min(d) + 1.7

    This is the per-day quantity used by the PG-low percentile calculation.
    """
    daily_min_c = (
        hourly.groupby("local_date")["temp_c"].min()
    )
    pav_min_c = PAV_MIN_SLOPE_C * daily_min_c + PAV_MIN_INTERCEPT_C
    pav_min_c.index = pd.to_datetime(pav_min_c.index)
    pav_min_c.name = "pav_min_c"
    return pav_min_c.sort_index()


def compute_pg_low_stats(
    daily_pav_min_c: pd.Series,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
    window_years: int,
    min_window_years: int,
) -> dict[str, float]:
    """Compute the three PG-low columns for one row.

    - pg_low_normal_f: P_0.02 of daily pav-min over a `window_years` span
      ending at the interval midpoint, in deg F. NaN if fewer than
      `min_window_years` of data are available in that window.
    - pg_low_interval_f: coldest daily pav-min observed in the row's
      interval, in deg F.
    - delta_pg_low_f: interval - normal (negative = exceedance).
    - pg_window_years: actual years of data used for the percentile.
    """
    nan_out = {
        "pg_low_normal_f":   float("nan"),
        "pg_low_interval_f": float("nan"),
        "delta_pg_low_f":    float("nan"),
        "pg_window_years":   float("nan"),
    }
    if daily_pav_min_c.empty:
        return nan_out

    # Strip tz for the date math against the (tz-naive) daily index.
    if interval_start.tzinfo is not None:
        interval_start = interval_start.tz_convert("UTC").tz_localize(None)
    if interval_end.tzinfo is not None:
        interval_end = interval_end.tz_convert("UTC").tz_localize(None)

    # Interval coldest pav-min (deg F).
    interval_slice = daily_pav_min_c.loc[interval_start:interval_end]
    if interval_slice.empty:
        return nan_out
    interval_min_c = float(interval_slice.min())
    interval_min_f = interval_min_c * 9.0 / 5.0 + 32.0

    # Normal: P_0.02 of pav-min in [midpoint − window, midpoint], shrunk to
    # what's available.
    midpoint = interval_start + (interval_end - interval_start) / 2
    window_start = midpoint - pd.Timedelta(days=int(window_years * 365.25))
    window_end = midpoint
    window_slice = daily_pav_min_c.loc[window_start:window_end].dropna()

    actual_years = float("nan")
    if not window_slice.empty:
        span = (window_slice.index.max() - window_slice.index.min())
        actual_years = span.total_seconds() / (365.25 * 86400.0)

    if window_slice.empty or actual_years < min_window_years:
        return {
            "pg_low_normal_f":   float("nan"),
            "pg_low_interval_f": interval_min_f,
            "delta_pg_low_f":    float("nan"),
            "pg_window_years":   actual_years if not pd.isna(actual_years) else float("nan"),
        }

    normal_c = float(window_slice.quantile(0.02))
    normal_f = normal_c * 9.0 / 5.0 + 32.0
    return {
        "pg_low_normal_f":   normal_f,
        "pg_low_interval_f": interval_min_f,
        "delta_pg_low_f":    interval_min_f - normal_f,
        "pg_window_years":   actual_years,
    }


def build_skin_cooling_6h_c(hourly: pd.DataFrame) -> pd.Series:
    """Return a tz-aware Series indexed by UTC hour with the 6-hour forward
    cooling rate of skin temperature in deg C: R_6h(t) = T(t) - T(t+6h).
    Positive values mean cooling; negative means warming. NaN for the last
    6 hours (no future data) and for any input gaps.

    Built once per location so per-row event counting is just a sliced
    indexing + thresholding operation.
    """
    skt_c = hourly["skt_c"].dropna()
    if skt_c.empty:
        return pd.Series(dtype=float)
    # Strict hourly grid so .shift(-6) means 6 hours, not 6 samples.
    skt_c = skt_c.resample("1h").mean()
    r_6h = skt_c - skt_c.shift(-6)
    r_6h.name = "cooling_6h_c"
    return r_6h


def count_cool_events(
    cooling_series: pd.Series,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
    threshold_c: float,
    refractory_hours: int,
) -> int:
    """Count distinct cooling events with peak inside [interval_start,
    interval_end] whose 6-hour cooling exceeds threshold_c, suppressing
    further events within `refractory_hours` after each counted peak.

    Algorithm: find local maxima of the cooling series above threshold;
    walk through them in time order, accept each maximum only if it's
    at least refractory_hours after the most recently accepted one.
    """
    if cooling_series is None or cooling_series.empty:
        return 0

    # Normalize timestamps to match the series index timezone.
    if cooling_series.index.tz is not None:
        if interval_start.tzinfo is None:
            interval_start = interval_start.tz_localize("UTC")
        if interval_end.tzinfo is None:
            interval_end = interval_end.tz_localize("UTC")
    else:
        if interval_start.tzinfo is not None:
            interval_start = interval_start.tz_convert("UTC").tz_localize(None)
        if interval_end.tzinfo is not None:
            interval_end = interval_end.tz_convert("UTC").tz_localize(None)

    sliced = cooling_series.loc[interval_start:interval_end].dropna()
    if sliced.empty:
        return 0

    # A "peak" is a point that is greater than its immediate neighbors AND
    # above threshold. Endpoints count if they're above threshold and above
    # their one available neighbor.
    above = sliced[sliced > threshold_c]
    if above.empty:
        return 0

    # Use a simple peak-detection: at hour t, peak iff
    # series[t] > series[t-1] AND series[t] >= series[t+1]. The >= on the
    # right side handles plateaus (counts the first hour of a flat top).
    vals = sliced.values
    idx = sliced.index
    peaks: list[pd.Timestamp] = []
    for i in range(len(vals)):
        if vals[i] <= threshold_c:
            continue
        left_ok = (i == 0) or (vals[i] > vals[i - 1])
        right_ok = (i == len(vals) - 1) or (vals[i] >= vals[i + 1])
        if left_ok and right_ok:
            peaks.append(idx[i])

    if not peaks:
        return 0

    # Apply refractory: walk peaks in time order, accept if far enough from
    # the previous accepted peak.
    refractory = pd.Timedelta(hours=refractory_hours)
    count = 0
    last_accepted: pd.Timestamp | None = None
    for t in peaks:
        if last_accepted is None or (t - last_accepted) >= refractory:
            count += 1
            last_accepted = t
    return count


def _extract_turning_points(values: np.ndarray) -> np.ndarray:
    """Return the sequence of local extrema (peaks and valleys) from a 1D
    series. Endpoints are kept. This is the standard preprocessing for
    rainflow: only turning points contribute to cycles.
    """
    if len(values) < 2:
        return values.copy()
    # Direction of change at each step
    diffs = np.diff(values)
    # Mark a turning point where the sign of the derivative changes
    # (or where it transitions from/to zero, which we treat as continuation).
    keep = np.zeros(len(values), dtype=bool)
    keep[0] = True
    keep[-1] = True
    # Use sign(diffs); collapse zeros into the previous nonzero direction.
    sign = np.sign(diffs)
    last_dir = 0.0
    for i, s in enumerate(sign):
        if s != 0:
            if last_dir != 0 and s != last_dir:
                keep[i] = True
            last_dir = s
    return values[keep]


def rainflow_cycles(values: np.ndarray) -> list[tuple[float, float]]:
    """Decompose a time series into rainflow cycles. Returns a list of
    (range, count) tuples where:
      - range is the absolute peak-to-peak amplitude of the cycle (deg C)
      - count is 1.0 for full cycles, 0.5 for half cycles (residuals)

    ASTM E1049-85 4-point rainflow counting. A closed cycle is extracted
    when, looking at the last 4 turning points S0,S1,S2,S3:
        |S2-S1| <= |S1-S0|  AND  |S2-S1| <= |S3-S2|
    The cycle (S1,S2) is then removed and counting resumes.

    Equivalent to the output of standard rainflow libraries.
    """
    tps = list(_extract_turning_points(np.asarray(values, dtype=float)))
    if len(tps) < 4:
        # Not enough points for any closed cycle — everything is half-cycles.
        return [(abs(tps[i + 1] - tps[i]), 0.5) for i in range(len(tps) - 1)]

    cycles: list[tuple[float, float]] = []
    stack: list[float] = []

    for v in tps:
        stack.append(v)
        # Repeatedly check the last 4 points for a closed cycle.
        while len(stack) >= 4:
            S0, S1, S2, S3 = stack[-4], stack[-3], stack[-2], stack[-1]
            r01 = abs(S1 - S0)
            r12 = abs(S2 - S1)
            r23 = abs(S3 - S2)
            if r12 <= r01 and r12 <= r23:
                # Extract closed cycle (S1, S2). Remove S1 and S2.
                cycles.append((r12, 1.0))
                # Delete stack[-3] (S1) and stack[-2] (S2); keep S0 and S3.
                del stack[-3:-1]
            else:
                break

    # Residuals: each adjacent pair in the leftover stack is a half-cycle.
    for i in range(len(stack) - 1):
        rng = abs(stack[i + 1] - stack[i])
        cycles.append((rng, 0.5))

    return cycles


def compute_rainflow_stats(
    skt_c_slice: pd.Series,
    fatigue_beta: float,
) -> dict[str, float]:
    """Compute rainflow-based stats for one interval slice of skin
    temperature in deg C. Returns:
      - rainflow_cycles_10c: count of cycles with range >= 10 deg C
      - rainflow_cycles_20c: count of cycles with range >= 20 deg C
      - rainflow_damage:     Σ (range^beta * count) over all cycles
    Raw counts (NOT per-year). The caller normalizes by interval years.
    """
    out = {
        "rainflow_cycles_10c": 0.0,
        "rainflow_cycles_20c": 0.0,
        "rainflow_damage":     0.0,
    }
    if skt_c_slice is None or len(skt_c_slice) < 3:
        return out
    values = skt_c_slice.dropna().values
    if len(values) < 3:
        return out

    cycles = rainflow_cycles(values)
    for rng, cnt in cycles:
        if rng >= RAINFLOW_THRESHOLDS_C[0]:
            out["rainflow_cycles_10c"] += cnt
        if rng >= RAINFLOW_THRESHOLDS_C[1]:
            out["rainflow_cycles_20c"] += cnt
        out["rainflow_damage"] += (rng ** fatigue_beta) * cnt
    return out


def compute_max_cooling_excursion_c(skt_c_slice: pd.Series) -> float:
    """Largest peak-to-subsequent-trough drop (deg C) in a skin-temperature
    series. Walks turning points left-to-right; for each peak, tracks the
    minimum trough that follows it and computes peak − trough. Returns the
    max over all (peak, later_trough) pairs.

    Returns NaN if the slice is too short or has no turning points.

    Implementation note: the naive O(n²) approach is wasteful. We use a
    single forward pass tracking running min of troughs after each peak by
    walking from right to left first to precompute suffix-min, then
    left to right to compare each peak to the suffix-min after it.
    """
    if skt_c_slice is None or len(skt_c_slice) < 2:
        return float("nan")
    values = skt_c_slice.dropna().values.astype(float)
    if len(values) < 2:
        return float("nan")

    tps = _extract_turning_points(values)
    if len(tps) < 2:
        # Too few turning points (degenerate input).
        return 0.0

    # Suffix min of turning points: suffix_min[i] = min(tps[i:])
    suffix_min = np.minimum.accumulate(tps[::-1])[::-1]
    # For each turning point t at position i, the largest subsequent drop
    # has trough equal to suffix_min[i+1] (the smallest value strictly after
    # position i). The drop equals tps[i] - suffix_min[i+1].
    drops = tps[:-1] - suffix_min[1:]
    # Clamp at 0: a strictly ascending series has all-negative "drops" and
    # should report 0 cooling rather than a negative number.
    return float(max(0.0, drops.max()))


def compute_thermal_aging(
    hourly: pd.DataFrame,
    construction_year: int | float | None,
    inspect_end: pd.Timestamp,
    ea_j_per_mol: float = THERMAL_AGING_EA_DEFAULT,
) -> dict[str, float]:
    """Compute the cumulative Arrhenius thermal aging integral for a row.

    Φ_thermal = sum over hourly t in [construct_start, inspect_end] of
                exp(-Ea / (R · T_skin(t)))
    where T_skin is in Kelvin.

    `construction_year` is converted to a date by anchoring to July 1 of the
    given year (mid-year, the unbiased choice when only a year is known).
    If construction is before ERA5 coverage starts, the integral truncates
    to whatever's available, and the missing-years gap is reported.

    Returns a dict with three keys:
      - thermal_aging_phi: the raw integral (very small number, ~1e-9 scale)
      - thermal_aging_teff_f: the constant-temperature equivalent that would
        produce the same total reaction extent over the same hours; in deg F.
        Interpretable directly.
      - thermal_aging_missing_years: how many years between construction date
        and ERA5 start (zero for newer pavements).
    """
    nan_out = {
        "thermal_aging_phi":           float("nan"),
        "thermal_aging_teff_f":        float("nan"),
        "thermal_aging_missing_years": float("nan"),
    }
    if (construction_year is None
            or (isinstance(construction_year, float) and pd.isna(construction_year))
            or hourly is None or hourly.empty):
        return nan_out

    try:
        c_year = int(construction_year)
    except (TypeError, ValueError):
        return nan_out
    if c_year < 1900 or c_year > 2100:  # obviously bogus
        return nan_out

    construction_date = pd.Timestamp(year=c_year, month=7, day=1, tz="UTC")
    if inspect_end.tzinfo is None:
        inspect_end = inspect_end.tz_localize("UTC")
    if construction_date >= inspect_end:
        # Construction after last inspection — nothing to integrate.
        return {
            "thermal_aging_phi":           0.0,
            "thermal_aging_teff_f":        float("nan"),
            "thermal_aging_missing_years": 0.0,
        }

    # Effective integration start = max(construction, first ERA5 timestamp)
    era5_start = hourly.index.min()
    eff_start = max(construction_date, era5_start)
    missing_years = 0.0
    if construction_date < era5_start:
        missing_years = (era5_start - construction_date).total_seconds() / (365.25 * 86400.0)

    integration_slice = hourly.loc[eff_start:inspect_end]
    skt_k = (integration_slice["skt_c"].dropna() + 273.15).values
    if skt_k.size == 0:
        return {
            "thermal_aging_phi":           float("nan"),
            "thermal_aging_teff_f":        float("nan"),
            "thermal_aging_missing_years": missing_years,
        }

    arrhenius = np.exp(-ea_j_per_mol / (GAS_CONSTANT_J_PER_MOL_K * skt_k))
    phi = float(arrhenius.sum())
    n_hours = float(skt_k.size)

    # Effective constant temperature: solve phi/n = exp(-Ea/(R·T_eff))
    # → T_eff = -Ea / (R · ln(phi / n))
    ratio = phi / n_hours
    if ratio <= 0.0 or not np.isfinite(ratio):
        teff_f = float("nan")
    else:
        teff_k = -ea_j_per_mol / (GAS_CONSTANT_J_PER_MOL_K * math.log(ratio))
        teff_c = teff_k - 273.15
        teff_f = teff_c * 9.0 / 5.0 + 32.0

    return {
        "thermal_aging_phi":           phi,
        "thermal_aging_teff_f":        float(teff_f),
        "thermal_aging_missing_years": float(missing_years),
    }


def compute_temp_stats_from_hourly(
    hourly: pd.DataFrame,
    lat: float,
    pav_depth_low_mm: float,
    pav_depth_high_mm: float,
    pav_reliability: int,
    min_ltpp_years: float,
    source_info: dict[str, float | str | int] | None = None,
    daily_pav_min_c: pd.Series | None = None,
    pg_window_years: int = 20,
    min_pg_window_years: int = 10,
    cooling_6h_c: pd.Series | None = None,
    cool_event_threshold_c: float = 10.0,
    cool_event_refractory_hours: int = 24,
    rainflow_fatigue_beta: float = 3.5,
    full_hourly: pd.DataFrame | None = None,
    construction_year: int | float | None = None,
    thermal_aging_ea_j_per_mol: float = THERMAL_AGING_EA_DEFAULT,
) -> dict[str, float | str]:
    """Compute all temperature stats from a *sliced* hourly DataFrame produced
    by `prepare_hourly_df`. The slice defines the interval; everything reduces
    over it. Returns dict keyed by the keys in STAT_COLUMNS.
    """
    if hourly.empty:
        raise ValueError("Empty hourly slice — no data in interval")

    interval_start = hourly.index.min()
    interval_end = hourly.index.max()
    interval_years = (interval_end - interval_start).total_seconds() / (365.25 * 86400.0)

    # ---- Daily aggregates (local-time days) -----------------------------
    daily = hourly.groupby("local_date").agg(
        daily_min_c=("temp_c", "min"),
        daily_max_c=("temp_c", "max"),
        daily_min_f=("temp_f", "min"),
        daily_max_f=("temp_f", "max"),
    ).reset_index()
    daily["daily_range_f"] = daily["daily_max_f"] - daily["daily_min_f"]
    daily["pav_min_c"] = PAV_MIN_SLOPE_C * daily["daily_min_c"] + PAV_MIN_INTERCEPT_C
    daily["pav_min_f"] = daily["pav_min_c"] * 9.0 / 5.0 + 32.0

    # ---- Cooling rates (skin temp, hourly) ------------------------------
    cooling_stats = _cooling_rates_from_hourly(hourly)

    # ---- LTPP design temps ----------------------------------------------
    if interval_years >= min_ltpp_years and not daily["daily_min_c"].dropna().empty:
        air_low_c, air_high_c = compute_ltpp_air_extremes_from_daily(daily)
        pav_low_c, pav_high_c = ltpp_pavement_temps(
            air_low_c=air_low_c,
            air_high_c=air_high_c,
            lat=lat,
            depth_low_mm=pav_depth_low_mm,
            depth_high_mm=pav_depth_high_mm,
            reliability=pav_reliability,
        )
        pg = pg_grade_label(pav_low_c, pav_high_c)
    else:
        air_low_c = air_high_c = pav_low_c = pav_high_c = float("nan")
        pg = ""

    src = source_info or {}

    # PG-low percentile/exceedance against location's full daily pav-min series.
    if daily_pav_min_c is not None and not daily_pav_min_c.empty:
        pg_low_stats = compute_pg_low_stats(
            daily_pav_min_c=daily_pav_min_c,
            interval_start=interval_start,
            interval_end=interval_end,
            window_years=pg_window_years,
            min_window_years=min_pg_window_years,
        )
    else:
        pg_low_stats = {
            "pg_low_normal_f":   float("nan"),
            "pg_low_interval_f": float("nan"),
            "delta_pg_low_f":    float("nan"),
            "pg_window_years":   float("nan"),
        }

    # Cooling-event count over the interval (uses location-wide R_6h series).
    cool_events = count_cool_events(
        cooling_series=cooling_6h_c,
        interval_start=interval_start,
        interval_end=interval_end,
        threshold_c=cool_event_threshold_c,
        refractory_hours=cool_event_refractory_hours,
    ) if cooling_6h_c is not None else 0
    cool_events_per_year = (
        float(cool_events) / interval_years if interval_years > 0 else float("nan")
    )

    # Rainflow stats on the skin-temperature series within the interval.
    rainflow_stats = compute_rainflow_stats(
        skt_c_slice=hourly["skt_c"],
        fatigue_beta=rainflow_fatigue_beta,
    )
    if interval_years > 0:
        rf_10_per_year = rainflow_stats["rainflow_cycles_10c"] / interval_years
        rf_20_per_year = rainflow_stats["rainflow_cycles_20c"] / interval_years
        rf_dmg_per_year = rainflow_stats["rainflow_damage"] / interval_years
    else:
        rf_10_per_year = rf_20_per_year = rf_dmg_per_year = float("nan")

    # Max single peak-to-trough cooling excursion in the interval.
    max_excursion_c = compute_max_cooling_excursion_c(hourly["skt_c"])
    max_excursion_f = (
        float(max_excursion_c) * 9.0 / 5.0  # delta-C → delta-F (no +32 offset)
        if not (isinstance(max_excursion_c, float) and pd.isna(max_excursion_c))
        else float("nan")
    )

    # Cumulative thermal aging since construction. Uses the LOCATION-wide
    # hourly series (not the interval slice) because the integral starts at
    # the construction date, which may pre-date the row's first inspection.
    thermal_aging = compute_thermal_aging(
        hourly=full_hourly if full_hourly is not None else hourly,
        construction_year=construction_year,
        inspect_end=interval_end,
        ea_j_per_mol=thermal_aging_ea_j_per_mol,
    )

    return {
        "interval_start":    interval_start.strftime("%Y-%m-%d"),
        "interval_end":      interval_end.strftime("%Y-%m-%d"),
        "interval_years":    float(interval_years),
        "mean_f":            float(hourly["temp_f"].mean()),
        "daily_min_avg_f":   float(daily["daily_min_f"].mean()),
        "daily_max_avg_f":   float(daily["daily_max_f"].mean()),
        "daily_range_avg_f": float(daily["daily_range_f"].mean()),
        "abs_min_f":         float(hourly["temp_f"].min()),
        "abs_max_f":         float(hourly["temp_f"].max()),
        "pav_min_f":         float(daily["pav_min_f"].min()),
        **cooling_stats,
        "cool_events_6h":          int(cool_events) if cool_events is not None else 0,
        "cool_events_6h_per_year": float(cool_events_per_year),
        "rainflow_cycles_10c":          float(rainflow_stats["rainflow_cycles_10c"]),
        "rainflow_cycles_10c_per_year": float(rf_10_per_year),
        "rainflow_cycles_20c":          float(rainflow_stats["rainflow_cycles_20c"]),
        "rainflow_cycles_20c_per_year": float(rf_20_per_year),
        "rainflow_damage":              float(rainflow_stats["rainflow_damage"]),
        "rainflow_damage_per_year":     float(rf_dmg_per_year),
        "max_cooling_excursion_f":      max_excursion_f,
        **thermal_aging,
        "ltpp_air_low_c":    float(air_low_c),
        "ltpp_air_high_c":   float(air_high_c),
        "pav_design_low_c":  float(pav_low_c),
        "pav_design_high_c": float(pav_high_c),
        "pav_pg_grade":      pg,
        **pg_low_stats,
        "data_source":          src.get("data_source", ""),
        "snap_distance_cells":  src.get("snap_distance_cells", float("nan")),
        "snap_distance_km":     src.get("snap_distance_km", float("nan")),
    }


def _cooling_rates_from_hourly(hourly: pd.DataFrame) -> dict[str, float]:
    """Cooling-rate stats from skin_temperature on a hourly-indexed DataFrame.
    Resamples to a strict 1h grid first to handle gaps cleanly.
    """
    skt_f = hourly["skt_f"].dropna()
    if skt_f.empty:
        return {f"cooling_{h}h_f": float("nan") for h in COOLING_WINDOWS_HRS}

    # Strict hourly grid so .shift(-N) means N hours, not N samples.
    skt_f = skt_f.resample("1h").mean()

    out: dict[str, float] = {}
    for hrs in COOLING_WINDOWS_HRS:
        future = skt_f.shift(-hrs)
        delta = (skt_f - future).dropna()
        out[f"cooling_{hrs}h_f"] = float(delta.max()) if not delta.empty else float("nan")
    return out


def compute_ltpp_air_extremes_from_daily(
    daily: pd.DataFrame, n_high_days: int = EXTREME_NDAYS_HIGH
) -> tuple[float, float]:
    """Compute the LTPPBind air-temp inputs (deg C):
       - low:  mean across years of (each year's coldest 1-day air min)
       - high: mean across years of (each year's hottest n-day rolling
               average of daily highs).
    """
    daily = daily.copy()
    daily["year"] = pd.to_datetime(daily["local_date"]).dt.year

    yearly_min = daily.groupby("year")["daily_min_c"].min()
    air_low_c = float(yearly_min.mean())

    daily_sorted = daily.sort_values("local_date").reset_index(drop=True)
    rolling_high = (
        daily_sorted["daily_max_c"]
        .rolling(window=n_high_days, min_periods=n_high_days)
        .mean()
    )
    daily_sorted["rolling_high_c"] = rolling_high
    yearly_rolling_max = daily_sorted.groupby("year")["rolling_high_c"].max()
    air_high_c = float(yearly_rolling_max.mean())

    return air_low_c, air_high_c


def ltpp_pavement_temps(
    air_low_c: float,
    air_high_c: float,
    lat: float,
    depth_low_mm: float,
    depth_high_mm: float,
    reliability: int,
) -> tuple[float, float]:
    """Apply LTPP Eqs. 7 and 8 to get pavement low/high design temps (deg C).
    `reliability` ∈ {50, 98}; 98 shifts low colder and high hotter by 2*RMSE.
    The two depths follow LTPPBind convention: low at 20 mm, high at surface.
    """
    log_h_low = math.log10(depth_low_mm + 25.0)
    log_h_high = math.log10(depth_high_mm + 25.0)
    sigma = LTPP_RELIABILITY_SIGMA[reliability]

    low = (
        LTPP_LOW["intercept"]
        + LTPP_LOW["t_air"] * air_low_c
        + LTPP_LOW["lat2"] * lat * lat
        + LTPP_LOW["log_h"] * log_h_low
        - sigma * LTPP_LOW["rmse"]  # colder for higher reliability
    )
    high = (
        LTPP_HIGH["intercept"]
        + LTPP_HIGH["t_air"] * air_high_c
        + LTPP_HIGH["lat2"] * lat * lat
        + LTPP_HIGH["log_h"] * log_h_high
        + sigma * LTPP_HIGH["rmse"]  # hotter for higher reliability
    )
    return float(low), float(high)


def pg_grade_label(pav_low_c: float, pav_high_c: float) -> str:
    """Snap pavement low/high to the standard PG bins and format the label.
    High temps are named in 6 C bins from 46 upward; low temps in 6 C bins
    starting at -10 going colder. The PG label uses the high-temp bin and
    the absolute value of the low-temp bin: e.g. 'PG 64-22'.
    """
    # High: round UP to next 6 C bin starting from 46.
    high_bin = 46 + 6 * max(0, math.ceil((pav_high_c - 46) / 6))
    # Low: round DOWN to next 6 C bin starting at -10 going colder.
    # Bins are -10, -16, -22, -28, -34, -40, -46.
    low_bin = -10 - 6 * max(0, math.ceil((-10 - pav_low_c) / 6))
    return f"PG {high_bin}-{abs(low_bin)}"


def location_cache_key(lat: float, lon: float, start_year: int, end_year: int) -> str:
    lat_str = f"{lat:+08.4f}".replace("+", "p").replace("-", "m").replace(".", "_")
    lon_str = f"{lon:+09.4f}".replace("+", "p").replace("-", "m").replace(".", "_")
    return f"{lat_str}__{lon_str}__{start_year}_{end_year}"


def _snap_candidates(lat: float, lon: float, radius_cells: int, grid_deg: float):
    """Generate snap candidates spiraling outward from (lat, lon) on a
    grid_deg lattice. Yields (lat, lon, dlat_cells, dlon_cells) tuples,
    starting with the original point (0, 0) then rings of increasing
    chebyshev distance up to radius_cells, sorted within each ring by
    euclidean distance.
    """
    yield (lat, lon, 0, 0)
    for r in range(1, radius_cells + 1):
        ring: list[tuple[int, int]] = []
        for dlat in range(-r, r + 1):
            for dlon in range(-r, r + 1):
                if max(abs(dlat), abs(dlon)) == r:
                    ring.append((dlat, dlon))
        ring.sort(key=lambda d: d[0] * d[0] + d[1] * d[1])
        for dlat, dlon in ring:
            yield (
                lat + dlat * grid_deg,
                lon + dlon * grid_deg,
                dlat,
                dlon,
            )


def _snap_distance_km(lat: float, dlat_cells: int, dlon_cells: int, grid_deg: float) -> float:
    """Approximate distance from origin cell to (dlat_cells, dlon_cells) at
    the given latitude. Uses small-angle local Earth radius — accurate to
    sub-percent for the ~0.2 deg displacements we deal with.
    """
    if dlat_cells == 0 and dlon_cells == 0:
        return 0.0
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))
    dy = dlat_cells * grid_deg * km_per_deg_lat
    dx = dlon_cells * grid_deg * km_per_deg_lon
    return float(math.hypot(dx, dy))


def _load_one_attempt(
    client: cdsapi.Client,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    cache_dir: Path,
    force_download: bool,
    dataset: str,
    cache_suffix: str,
    progress: ClimateProgress | None = None,
) -> pd.DataFrame | None:
    """Download (or reuse) a single ERA5(-Land) request at (lat, lon), parse
    it, return the prepared hourly DataFrame. Returns None if the result is
    empty (ocean-masked) or if data fundamentally couldn't be parsed.

    cache_suffix lets the caller distinguish multiple attempts at related
    lat/lons (e.g. the snap candidates) without colliding cache filenames.
    """
    key = location_cache_key(lat, lon, start_year, end_year)
    if cache_suffix:
        key = f"{key}__{cache_suffix}"
    parquet_file = cache_dir / f"{key}.hourly.parquet"
    raw_file = cache_dir / f"{key}.zip"
    # Preserve the original cache layout while refusing redirected cache files.
    for path in (parquet_file, raw_file):
        reject_links(path)

    if progress is not None:
        progress.update("cache_check", dataset="era5" if dataset == DATASET_ERA5 else "era5-land")

    if parquet_file.exists() and not force_download:
        try:
            cached = pd.read_parquet(parquet_file)
            if progress is not None:
                progress.update("cache_hit", cache_hits=progress.increment("cache_hits"))
            if len(cached) > 0:
                return cached
            # Empty parquet means a previous attempt confirmed this point is
            # ocean-masked. Don't retry the download.
            return None
        except Exception as e:
            print(f"[WARN] Could not read parquet cache ({e}); rebuilding.")

    download_attempts = 0
    while True:
        try:
            download_timeseries(
                client=client,
                lat=lat,
                lon=lon,
                start_year=start_year,
                end_year=end_year,
                target=raw_file,
                force_download=force_download,
                dataset=dataset,
                progress=progress,
            )
        except Exception as e:
            print(f"[ERROR] Download failed for ({lat}, {lon}) on {dataset}: {e}")
            return None
        if progress is not None:
            progress.update("parsing")
        df = open_dataset(raw_file)
        try:
            if progress is not None:
                progress.update("preparing")
            hourly = prepare_hourly_df(df, lon=lon)
            break
        except KeyError as e:
            if download_attempts >= 1 or force_download:
                raise
            print(f"[WARN] Cached ZIP missing expected column ({e}); re-downloading.")
            raw_file.unlink(missing_ok=True)
            force_download = True
            download_attempts += 1

    # Persist parquet (even if empty — that's a positive result that this
    # cell is ocean-masked, so we don't re-download next time).
    try:
        hourly.to_parquet(parquet_file)
    except Exception as e:
        print(f"[WARN] Could not write parquet cache: {e}")

    return hourly if len(hourly) > 0 else None


def get_hourly_for_location(
    client: cdsapi.Client,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    cache_dir: Path,
    force_download: bool,
    snap_radius: int,
    era5_fallback: bool,
    progress: ClimateProgress | None = None,
) -> tuple[pd.DataFrame | None, dict[str, float | str | int]]:
    """Return (hourly_df, source_info) for one location.

    Resolution strategy:
      1. Try ERA5-Land at the original point.
      2. If empty (ocean-masked), spiral outward up to snap_radius cells.
      3. If still empty and era5_fallback is on, try ERA5 at original point
         (0.25 deg, defined globally including ocean).

    source_info is a dict with three keys:
      - data_source: "era5-land" or "era5"
      - snap_distance_cells: 0 if original cell used; >0 for snap hits.
        Always 0 for ERA5 fallback (no snap performed there).
      - snap_distance_km: physical distance from the requested point.

    Returns (None, empty-dict-style) if everything fails.
    """
    empty_source: dict[str, float | str | int] = {
        "data_source": "",
        "snap_distance_cells": float("nan"),
        "snap_distance_km": float("nan"),
    }

    # Attempt 1: ERA5-Land at the original point and snap candidates.
    candidates = (
        list(_snap_candidates(lat, lon, snap_radius, ERA5_LAND_GRID_DEG))
        if snap_radius > 0
        else [(lat, lon, 0, 0)]
    )

    for cand_lat, cand_lon, dlat, dlon in candidates:
        if progress is not None and (dlat != 0 or dlon != 0):
            progress.update("fallback", dataset="era5-land")
        suffix = "" if (dlat == 0 and dlon == 0) else f"snap_{dlat:+d}_{dlon:+d}".replace("+", "p").replace("-", "m")
        hourly = _load_one_attempt(
            client=client,
            lat=cand_lat,
            lon=cand_lon,
            start_year=start_year,
            end_year=end_year,
            cache_dir=cache_dir,
            force_download=force_download,
            dataset=DATASET,
            cache_suffix=suffix,
            progress=progress,
        )
        if hourly is not None:
            cells = max(abs(dlat), abs(dlon))
            return hourly, {
                "data_source": "era5-land",
                "snap_distance_cells": cells,
                "snap_distance_km": _snap_distance_km(lat, dlat, dlon, ERA5_LAND_GRID_DEG),
            }
        if not (dlat == 0 and dlon == 0):
            print(f"[INFO]   snap (+{dlat},+{dlon}): ocean-masked, trying next cell")

    # Attempt 2: ERA5 fallback at the original point.
    if era5_fallback:
        if progress is not None:
            progress.update("fallback", dataset="era5")
        print(f"[INFO]   ERA5-Land snap exhausted; falling back to ERA5 (0.25 deg)")
        hourly = _load_one_attempt(
            client=client,
            lat=lat,
            lon=lon,
            start_year=start_year,
            end_year=end_year,
            cache_dir=cache_dir,
            force_download=force_download,
            dataset=DATASET_ERA5,
            cache_suffix="era5",
            progress=progress,
        )
        if hourly is not None:
            return hourly, {
                "data_source": "era5",
                "snap_distance_cells": 0,
                "snap_distance_km": 0.0,
            }

    return None, empty_source


# (no _short_label needed — snap suffix is now built directly from cell offsets)


def slice_to_interval(
    hourly: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Slice a hourly DataFrame (UTC-indexed) to [start, end] inclusive.
    Both timestamps should be tz-aware (UTC); if not, they're assumed UTC.
    """
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    if start > end:
        start, end = end, start
    return hourly.loc[start:end]


def nan_stats_dict() -> dict[str, float | str]:
    out: dict[str, float | str] = {k: float("nan") for k, _ in STAT_COLUMNS}
    out["pav_pg_grade"] = ""
    out["interval_start"] = ""
    out["interval_end"] = ""
    out["data_source"] = ""
    # snap_distance_cells / snap_distance_km already NaN from the dict comp
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _create_cds_client():
    if cdsapi is None:
        raise SafeError("ERA5_REQUIRES_CDSAPI_INSTALL_ORIGINAL_EXTRA")
    return cdsapi.Client()


def main(argv=None) -> int:
    refuse_public_execution()
    args = parse_args(argv)

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    for path in (input_path, output_path, cache_dir):
        reject_links(path)
    input_path, output_path, cache_dir = (path.resolve() for path in (input_path, output_path, cache_dir))
    check_cli_paths(inputs=(input_path,), outputs=(output_path, cache_dir))
    if (not disjoint(input_path, output_path) or not disjoint(input_path, cache_dir)
            or not disjoint(output_path, cache_dir)):
        raise SafeError("ERA5_INPUT_OUTPUT_CACHE_MUST_BE_SEPARATE")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.end_year < args.start_year:
        raise ValueError("--end-year must be >= --start-year")

    progress = ClimateProgress()
    progress.update("initializing", start_year=args.start_year, end_year=args.end_year)

    # Build the final column names from prefix/suffix.
    out_cols = {key: f"{args.col_prefix}{label}{args.col_suffix}" for key, label in STAT_COLUMNS}

    print(f"[INFO] Reading input CSV: {input_path}")
    df_in = pd.read_csv(input_path)
    progress.update("initializing", total_rows=len(df_in))

    for required in (args.lat_col, args.lon_col, args.dates_col):
        if required not in df_in.columns:
            raise KeyError(
                f"Input CSV must contain column {required!r}. "
                f"Found: {list(df_in.columns)}"
            )

    # Parse Dates -> per-row interval (utc-aware Timestamps).
    parsed_dates = df_in[args.dates_col].apply(parse_dates_cell)
    df_in["_interval_start"] = parsed_dates.apply(lambda lst: min(lst) if lst else pd.NaT)
    df_in["_interval_end"]   = parsed_dates.apply(lambda lst: max(lst) if lst else pd.NaT)
    df_in["_n_dates"]        = parsed_dates.apply(len)

    n_no_dates = int((df_in["_n_dates"] == 0).sum())
    if n_no_dates:
        print(f"[WARN] {n_no_dates} rows have no parseable dates; new columns will be NaN")

    # Parse construction year (used for cumulative thermal aging integral).
    if args.construction_year_col in df_in.columns:
        df_in["_construction_year"] = pd.to_numeric(
            df_in[args.construction_year_col], errors="coerce"
        )
        n_no_year = int(df_in["_construction_year"].isna().sum())
        if n_no_year:
            print(
                f"[WARN] {n_no_year} rows have missing/invalid "
                f"{args.construction_year_col!r}; thermal-aging columns will be NaN"
            )
    else:
        print(
            f"[WARN] Construction-year column {args.construction_year_col!r} not "
            f"found in input. Thermal-aging columns will be NaN for all rows."
        )
        df_in["_construction_year"] = float("nan")

    # Round to a stable precision for dedup, but keep the originals for output.
    df_in["_lat_key"] = pd.to_numeric(df_in[args.lat_col], errors="coerce").round(args.round_decimals)
    df_in["_lon_key"] = pd.to_numeric(df_in[args.lon_col], errors="coerce").round(args.round_decimals)

    bad = df_in[df_in["_lat_key"].isna() | df_in["_lon_key"].isna()]
    if not bad.empty:
        print(f"[WARN] {len(bad)} rows have non-numeric lat/lon and will get NaN in new columns")

    # Sanity-check intervals against the download window.
    valid_dates = df_in["_interval_end"].dropna()
    if not valid_dates.empty:
        latest_end = valid_dates.max()
        latest_end_year = int(latest_end.year)
        if latest_end_year > args.end_year:
            print(
                f"[WARN] Some rows have interval end years up to {latest_end_year} "
                f"but --end-year={args.end_year}. Consider raising --end-year."
            )
    valid_starts = df_in["_interval_start"].dropna()
    if not valid_starts.empty:
        earliest_start_year = int(valid_starts.min().year)
        if earliest_start_year < args.start_year:
            print(
                f"[WARN] Some rows have interval start years as early as "
                f"{earliest_start_year} but --start-year={args.start_year}. "
                f"Those intervals will be truncated to the download window."
            )

    unique_locs = (
        df_in.dropna(subset=["_lat_key", "_lon_key"])
             .drop_duplicates(subset=["_lat_key", "_lon_key"])
             [["_lat_key", "_lon_key"]]
             .reset_index(drop=True)
    )
    print(f"[INFO] Input rows: {len(df_in)}, unique locations: {len(unique_locs)}")

    progress.update("initializing", total_locations=len(unique_locs))

    client = _create_cds_client()

    # Stage 1: download + cache hourly DataFrame for each unique location,
    # plus build the location-wide daily SHRP pav-min series (used by the
    # PG-low percentile calculation, which needs the *full* climate window).
    location_data: dict[
        tuple[float, float],
        tuple[
            pd.DataFrame | None,
            dict[str, float | str | int],
            pd.Series | None,  # daily_pav_min_c
            pd.Series | None,  # cooling_6h_c
        ]
    ] = {}
    empty_source: dict[str, float | str | int] = {
        "data_source": "",
        "snap_distance_cells": float("nan"),
        "snap_distance_km": float("nan"),
    }
    for i, loc_row in unique_locs.iterrows():
        progress.update("loading_location", location_index=int(i) + 1)
        lat = float(loc_row["_lat_key"])
        lon = float(loc_row["_lon_key"])
        print(f"[INFO] ({i + 1}/{len(unique_locs)}) loading lat={lat}, lon={lon}")
        try:
            hourly, source_info = get_hourly_for_location(
                client=client,
                lat=lat,
                lon=lon,
                start_year=args.start_year,
                end_year=args.end_year,
                cache_dir=cache_dir,
                force_download=args.force_download,
                snap_radius=args.snap_radius,
                era5_fallback=args.era5_fallback,
                progress=progress,
            )
            if hourly is None:
                print(f"[ERROR] No usable data for ({lat}, {lon}); will get NaN")
                pav_min_series = None
                cooling_6h_series = None
            else:
                progress.update("preparing")
                cells = source_info.get("snap_distance_cells", 0)
                km = source_info.get("snap_distance_km", 0.0)
                print(
                    f"[INFO]   resolved via: {source_info.get('data_source', '?')}"
                    f"  (snap {cells} cells / {km:.1f} km)"
                )
                pav_min_series = build_daily_pav_min_c(hourly)
                cooling_6h_series = build_skin_cooling_6h_c(hourly)
            location_data[(lat, lon)] = (
                hourly, source_info, pav_min_series, cooling_6h_series
            )
        except Exception as e:
            print(f"[ERROR] Failed to load ({lat}, {lon}): {e}")
            location_data[(lat, lon)] = (None, dict(empty_source), None, None)

        usable = location_data[(lat, lon)][0] is not None
        progress.increment("usable_locations" if usable else "missing_locations")
        progress.update("location_complete" if usable else "location_failed",
                        locations_completed=int(i) + 1)

    # Stage 2: slice each row's interval and compute stats.
    row_results: list[dict[str, float | str]] = []
    nan_template = nan_stats_dict()
    progress.update("calculating", rows_completed=0)
    last_progress = time.monotonic()

    for idx, row in df_in.iterrows():
        # Report already finished rows, including every missing-data branch.
        # A slow single row remains visible through the parent's heartbeat.
        now = time.monotonic()
        if row_results and (len(row_results) % 100 == 0 or now - last_progress >= 1):
            progress.update("calculating", rows_completed=len(row_results))
            last_progress = now
        lat_key = row["_lat_key"]
        lon_key = row["_lon_key"]
        start = row["_interval_start"]
        end = row["_interval_end"]

        if pd.isna(lat_key) or pd.isna(lon_key) or pd.isna(start) or pd.isna(end):
            row_results.append(dict(nan_template))
            continue

        loc_key = (float(lat_key), float(lon_key))
        hourly, source_info, pav_min_series, cooling_6h_series = location_data.get(
            loc_key, (None, dict(empty_source), None, None)
        )
        if hourly is None:
            row_results.append(dict(nan_template))
            continue

        try:
            sliced = slice_to_interval(hourly, start, end)
            if sliced.empty:
                print(f"[WARN] Row {idx}: no ERA5 data in interval {start.date()}..{end.date()}")
                row_results.append(dict(nan_template))
                continue
            stats = compute_temp_stats_from_hourly(
                sliced,
                lat=loc_key[0],
                pav_depth_low_mm=args.pav_depth_low_mm,
                pav_depth_high_mm=args.pav_depth_high_mm,
                pav_reliability=args.pav_reliability,
                min_ltpp_years=args.min_ltpp_years,
                source_info=source_info,
                daily_pav_min_c=pav_min_series,
                pg_window_years=args.pg_window_years,
                min_pg_window_years=args.min_pg_window_years,
                cooling_6h_c=cooling_6h_series,
                cool_event_threshold_c=args.cool_event_threshold_c,
                cool_event_refractory_hours=args.cool_event_refractory_hours,
                rainflow_fatigue_beta=args.rainflow_fatigue_beta,
                full_hourly=hourly,
                construction_year=row.get("_construction_year"),
                thermal_aging_ea_j_per_mol=args.thermal_aging_ea,
            )
            row_results.append(stats)
        except Exception as e:
            print(f"[ERROR] Row {idx} ({loc_key}, {start.date()}..{end.date()}): {e}")
            row_results.append(dict(nan_template))

    # Stage 3: assemble output DataFrame.
    progress.update("calculating", rows_completed=len(row_results))
    progress.update("writing")
    results_df = pd.DataFrame(row_results)
    for stat_key, _ in STAT_COLUMNS:
        df_in[out_cols[stat_key]] = results_df[stat_key].values

    df_out = df_in.drop(columns=[
        "_lat_key", "_lon_key", "_interval_start", "_interval_end",
        "_n_dates", "_construction_year"
    ])
    df_out.to_csv(output_path, index=False)
    new_col_names = [out_cols[k] for k, _ in STAT_COLUMNS]
    print(f"[INFO] Wrote {output_path} ({len(df_out)} rows)")
    print(f"[INFO] New columns: {new_col_names}")
    progress.update("complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
