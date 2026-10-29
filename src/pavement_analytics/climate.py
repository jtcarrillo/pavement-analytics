"""Historical ERA5-Land exposures joined to individual inspection identities.

Every feature uses only the complete UTC days before the inspection date. The
CDS time-series product supplies Kelvin temperatures and de-accumulated hourly
precipitation in metres. Daily statistics here use UTC, not local solar days.
This intentionally does not reuse whole-history PG or thermal-aging estimates.
"""

from __future__ import annotations

from collections import Counter
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np
import pandas as pd

from .io import read_json, sha256, write_json
from .security import Boundary, SafeError, inside, lock, reject_links, source_root

DATASET = "reanalysis-era5-land-timeseries"
VARIABLES = ["2m_temperature", "skin_temperature", "total_precipitation"]
FEATURES = [
    "era5_air_mean_c", "era5_air_min_c", "era5_air_max_c",
    "era5_air_daily_range_mean_c", "era5_freezing_hours", "era5_freeze_thaw_days",
    "era5_hot_hours_30c", "era5_skin_mean_c", "era5_skin_min_c", "era5_skin_max_c",
    "era5_skin_daily_range_mean_c", "era5_skin_cooling_max_6h_c",
    "era5_skin_degree_hours_above_40c", "era5_precipitation_total_mm",
    "era5_precipitation_max_daily_mm", "era5_wet_days_1mm",
    "era5_air_daily_min_mean_c", "era5_air_daily_max_mean_c",
    "era5_skin_cooling_max_3h_c", "era5_skin_cooling_max_12h_c",
    "era5_skin_cooling_events_6h_10c", "era5_skin_max_cooling_excursion_c",
    "era5_skin_rainflow_cycles_10c", "era5_skin_rainflow_cycles_20c",
    "era5_skin_rainflow_range_squared_c2",
]
_ALIASES = {
    "air_k": ("t2m", "2m_temperature"),
    "skin_k": ("skt", "skin_temperature"),
    "precip_m": ("tp", "total_precipitation"),
}
_EARLIEST = pd.Timestamp("1950-01-02", tz="UTC")
_CONFIG_KEYS = {"mode", "features", "mapping", "csv_dir", "cache_dir", "lookback_days",
                "latitude_column", "longitude_column"}


def parse_location(value) -> tuple[float, float] | None:
    """Read decimal-degree pairs, optionally with N/S/E/W hemispheres."""
    if not isinstance(value, str) or not value.strip():
        return None
    tokens = re.findall(r"[NSEW]|[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value.upper())
    remainder = re.sub(r"[NSEW]|[+-]?(?:\d+(?:\.\d*)?|\.\d+)|[\s,;°]", "", value.upper())
    if remainder or len(tokens) not in {2, 4}:
        return None
    try:
        if len(tokens) == 2:
            lat, lon = map(float, tokens)
        else:
            pairs = [tokens[:2], tokens[2:]]
            numbers = []
            for pair, allowed in zip(pairs, ("NS", "EW")):
                hemisphere = [s for s in pair if s in allowed]
                numeric = [s for s in pair if s not in "NSEW"]
                if len(hemisphere) != 1 or len(numeric) != 1:
                    return None
                number = float(numeric[0])
                # A signed number inconsistent with its hemisphere is ambiguous.
                if number < 0 and hemisphere[0] in "NE":
                    return None
                numbers.append(abs(number) * (-1 if hemisphere[0] in "SW" else 1))
            lat, lon = numbers
        if np.isfinite([lat, lon]).all() and -90 <= lat <= 90 and -180 <= lon <= 180:
            return float(lat), float(lon)
    except (TypeError, ValueError):
        pass
    return None


def _location_csv(path: Path) -> tuple[tuple[float, float] | None, str]:
    reject_links(path)
    try:
        before = sha256(path)
        frame = None
        for encoding in ("utf-8-sig", "cp1252", "latin1"):
            try:
                frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
    except (OSError, ValueError, pd.errors.ParserError):
        raise SafeError("LOCATION_CSV_UNREADABLE") from None
    if frame is None or "LOCATION(DEGREES)" not in frame.columns:
        raise SafeError("LOCATION_COLUMN_MISSING")
    if sha256(path) != before:
        raise SafeError("INPUT_CHANGED_DURING_READ")
    values = {parse_location(v) for v in frame["LOCATION(DEGREES)"] if str(v).strip()}
    if None in values:
        raise SafeError("LOCATION_VALUE_INVALID")
    if len(values) > 1:
        raise SafeError("AMBIGUOUS_LOCATION_CSV")
    return next(iter(values), None), before


def _locations(table: pd.DataFrame, config: dict, source_map: dict) -> tuple[list, dict]:
    lower = {str(c).strip().lower(): c for c in table.columns}
    lat_col = config.get("latitude_column", lower.get("latitude", lower.get("lat")))
    lon_col = config.get("longitude_column", lower.get("longitude", lower.get("lon")))
    if ("latitude_column" in config or "longitude_column" in config) and (
        lat_col not in table or lon_col not in table
    ):
        raise SafeError("LOCATION_COLUMNS_MISSING")
    locations = []
    for _, row in table.iterrows():
        if lat_col in table and lon_col in table:
            locations.append(parse_location(f"{row[lat_col]}, {row[lon_col]}"))
        else:
            locations.append(None)
    audit = {"mapping_sha256": None, "location_sources": []}
    if not config.get("mapping"):
        return locations, audit
    if not config.get("csv_dir") or "_source_file" not in table:
        raise SafeError("LOCATION_MAPPING_REQUIRES_CSV_DIR_AND_SOURCE")
    mapping_path = Path(config["mapping"])
    reject_links(mapping_path)
    try:
        mapping_hash = sha256(mapping_path)
    except OSError:
        raise SafeError("LOCATION_MAPPING_UNREADABLE") from None
    mapping = read_json(mapping_path)
    if sha256(mapping_path) != mapping_hash:
        raise SafeError("INPUT_CHANGED_DURING_READ")
    audit["mapping_sha256"] = mapping_hash
    root = Path(config["csv_dir"]).expanduser()
    reject_links(root)
    root = root.resolve()
    loaded = {}
    for position, source in enumerate(table["_source_file"].astype(str)):
        if locations[position] is not None:
            continue
        original = source_map.get(source, source)
        name = Path(original).name
        relative = mapping.get(original, mapping.get(name))
        if relative is None:
            continue
        if not isinstance(relative, str) or not relative:
            raise SafeError("LOCATION_MAPPING_INVALID")
        path = root / relative
        reject_links(path)
        path = path.resolve()
        if not inside(path, root) or path == root:
            raise SafeError("LOCATION_MAPPING_PATH_OUTSIDE_CSV_DIR")
        if path not in loaded:
            location, digest = _location_csv(path)
            loaded[path] = location
            audit["location_sources"].append({"file": str(path.relative_to(root)), "sha256": digest})
        locations[position] = loaded[path]
    return locations, audit


def _read_hourly(path: Path) -> pd.DataFrame:
    """Read the CDS CSV or grouped CSV archive without extracting archive paths."""
    reject_links(path)
    try:
        with path.open("rb") as stream:
            magic = stream.read(4)
        if magic.startswith(b"PK"):
            with zipfile.ZipFile(path) as archive:
                entries = [i for i in archive.infolist() if i.filename.lower().endswith(".csv")]
                if not entries or sum(i.file_size for i in entries) > 2_000_000_000:
                    raise SafeError("ERA5_ARCHIVE_INVALID")
                frames = []
                for entry in entries:
                    with archive.open(entry) as stream:
                        frames.append(pd.read_csv(stream))
        elif magic.startswith(b"\x1f\x8b"):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                frames = [pd.read_csv(stream)]
        else:
            frames = [pd.read_csv(path)]
        series = {}
        grid = {}
        for frame in frames:
            lower = {str(c).lower(): c for c in frame.columns}
            time_col = next((lower[k] for k in ("time", "valid_time", "datetime", "date") if k in lower), None)
            if time_col is None:
                raise SafeError("ERA5_TIME_COLUMN_MISSING")
            times = pd.DatetimeIndex(pd.to_datetime(frame[time_col], utc=True, errors="coerce"))
            if times.isna().any() or times.duplicated().any() or not (times == times.floor("h")).all():
                raise SafeError("ERA5_TIMESTAMPS_INVALID")
            for coord, aliases in {"latitude": ("latitude", "lat"), "longitude": ("longitude", "lon")}.items():
                col = next((lower[a] for a in aliases if a in lower), None)
                if col is not None:
                    unique = pd.to_numeric(frame[col], errors="coerce").unique()
                    if len(unique) != 1 or not np.isfinite(unique[0]):
                        raise SafeError("ERA5_MULTIPLE_OR_INVALID_GRID_POINTS")
                    if coord in grid and grid[coord] != float(unique[0]):
                        raise SafeError("ERA5_GRID_POINT_MISMATCH")
                    grid[coord] = float(unique[0])
            for name, aliases in _ALIASES.items():
                col = next((lower[a] for a in aliases if a in lower), None)
                if col is not None:
                    if name in series:
                        raise SafeError("ERA5_DUPLICATE_VARIABLE")
                    series[name] = pd.Series(pd.to_numeric(frame[col], errors="coerce").to_numpy(), index=times)
        if set(series) != set(_ALIASES):
            raise SafeError("ERA5_REQUIRED_VARIABLE_MISSING")
        result = pd.DataFrame(series).sort_index()
        result.index.name = "time"
        result = result.replace([np.inf, -np.inf], np.nan)
        # Invalid physical values count as absent coverage, never as zero weather.
        for col in ("air_k", "skin_k"):
            result.loc[~result[col].between(100, 400), col] = np.nan
        result.loc[result["precip_m"] < -1e-8, "precip_m"] = np.nan
        result["precip_m"] = result["precip_m"].clip(lower=0)
        result.attrs["grid"] = grid
        return result
    except SafeError:
        raise
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, pd.errors.ParserError):
        raise SafeError("ERA5_RESPONSE_INVALID") from None


def _download(request: dict, target: Path) -> None:
    """The only network boundary; tests replace this function."""
    try:
        import cdsapi
    except ImportError:
        raise SafeError("ERA5_REQUIRES_CDSAPI_INSTALL_CLIMATE_EXTRA") from None
    # Suppress client messages containing coordinates, URLs or account details.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            client = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", quiet=True,
                                   debug=False, retry_max=3, sleep_max=10, timeout=120)
            client.retrieve(DATASET, request, str(target))
        except Exception:
            raise SafeError("ERA5_DOWNLOAD_FAILED_CHECK_CDS_SETUP_LICENCE_AND_DATES") from None


def _fetch_chunk(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp,
                 cache_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Get at most one calendar year, validating cached content hashes."""
    Boundary.create(source_root(), cache_dir)
    request = {"variable": VARIABLES, "location": {"latitude": lat, "longitude": lon},
               "date": f"{start.date().isoformat()}/{end.date().isoformat()}", "data_format": "csv"}
    identity = {"dataset": DATASET, "request": request, "cache_version": 1}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    raw, manifest = cache_dir / f"{key}.data", cache_dir / f"{key}.json"
    for path in (raw, manifest):
        reject_links(path)
    with lock(cache_dir / f"{key}.lock"):
        if raw.exists() or manifest.exists():
            if not raw.is_file() or not manifest.is_file():
                raise SafeError("ERA5_CACHE_INCOMPLETE")
            record = read_json(manifest)
            if record.get("identity") != identity or record.get("sha256") != sha256(raw):
                raise SafeError("ERA5_CACHE_HASH_MISMATCH")
            hourly = _read_hourly(raw)
            return hourly, {**record, "cache_hit": True, "cache_key": key}
        with tempfile.TemporaryDirectory(dir=cache_dir) as temporary:
            target = Path(temporary) / "download.data"
            _download(request, target)
            hourly = _read_hourly(target)
            record = {"identity": identity, "sha256": sha256(target), "grid": hourly.attrs.get("grid", {})}
            target.replace(raw)
            write_json(manifest, record)
        return hourly, {**record, "cache_hit": False, "cache_key": key}


def _combine_hourly(parts: list[tuple[pd.DataFrame, dict]]) -> tuple[pd.DataFrame, dict]:
    """Concatenate disjoint windows only when their actual grid point agrees."""
    if len(parts) == 1:
        return parts[0]
    grid, chunks = {}, []
    for _, record in parts:
        for coordinate, value in record.get("grid", {}).items():
            if coordinate in grid and grid[coordinate] != value:
                raise SafeError("ERA5_GRID_POINT_CHANGED_BETWEEN_REQUESTS")
            grid[coordinate] = value
        chunks.extend(record.get("chunks", [record]))
    hourly = pd.concat([frame for frame, _ in parts]).sort_index()
    if hourly.index.duplicated().any():
        raise SafeError("ERA5_OVERLAPPING_RESPONSE_HOURS")
    hourly.attrs["grid"] = grid
    key = hashlib.sha256(json.dumps([chunk["cache_key"] for chunk in chunks]).encode()).hexdigest()
    return hourly, {"cache_key": key, "cache_hit": all(chunk["cache_hit"] for chunk in chunks),
                    "grid": grid, "chunks": chunks}


def fetch_hourly(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp,
                 cache_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Fetch an inclusive date range in independently cached calendar-year parts.

    The CDS limits CSV requests to smaller downloads. Interior historical years
    have stable full-year cache keys; edge years use the exact requested bounds.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start > end:
        raise SafeError("ERA5_REQUEST_DATE_RANGE_INVALID")
    parts = []
    for year in range(start.year, end.year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1, tz=start.tz)
        year_end = pd.Timestamp(year=year, month=12, day=31, tz=end.tz)
        first, last = max(start, year_start), min(end, year_end)
        frame, record = _fetch_chunk(lat, lon, first, last, cache_dir)
        lower = pd.Timestamp(first.date(), tz="UTC")
        upper = pd.Timestamp(last.date(), tz="UTC") + pd.Timedelta(days=1)
        if ((frame.index < lower) | (frame.index >= upper)).any():
            raise SafeError("ERA5_RESPONSE_OUTSIDE_REQUESTED_DATES")
        parts.append((frame, record))
    return _combine_hourly(parts)


def _required_windows(inspections: list[tuple[int, pd.Timestamp]], lookback: int) -> list:
    """Union exposure windows so unused years between inspections are skipped."""
    ranges = sorted({(date - pd.Timedelta(days=lookback), date - pd.Timedelta(days=1))
                     for _, date in inspections})
    merged = []
    for first, last in ranges:
        if merged and first <= merged[-1][1] + pd.Timedelta(days=1):
            merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
        else:
            merged.append((first, last))
    return merged


def window_features(hourly: pd.DataFrame, inspection: pd.Timestamp,
                    lookback_days: int) -> tuple[dict, float]:
    """Slice BEFORE all calculations, including rolling cooling and daily totals."""
    end = inspection.normalize()
    start = end - pd.Timedelta(days=lookback_days)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    block = hourly.reindex(hours)[list(_ALIASES)]
    valid = np.isfinite(block.to_numpy(dtype=float)).all(axis=1)
    coverage = float(valid.mean())
    if not valid.all():
        return {}, coverage
    air, skin = block["air_k"] - 273.15, block["skin_k"] - 273.15
    rain = block["precip_m"] * 1000
    daily_air = air.resample("D").agg(["min", "max"])
    daily_skin = skin.resample("D").agg(["min", "max"])
    daily_rain = rain.resample("D").sum()
    cooling = skin.shift(6) - skin
    cycles = rainflow_cycles(skin.to_numpy())
    values = [
        air.mean(), air.min(), air.max(), (daily_air["max"] - daily_air["min"]).mean(),
        (air < 0).sum(), ((daily_air["min"] < 0) & (daily_air["max"] > 0)).sum(),
        (air > 30).sum(), skin.mean(), skin.min(), skin.max(),
        (daily_skin["max"] - daily_skin["min"]).mean(),
        max(0., float(cooling.max())), (skin - 40).clip(lower=0).sum(),
        rain.sum(), daily_rain.max(), (daily_rain >= 1).sum(),
        daily_air["min"].mean(), daily_air["max"].mean(),
        max(0., float((skin.shift(3) - skin).max())),
        max(0., float((skin.shift(12) - skin).max())),
        cooling_event_count(cooling), max(0., float((skin.cummax() - skin).max())),
        sum(count for extent, count in cycles if extent >= 10),
        sum(count for extent, count in cycles if extent >= 20),
        sum(extent ** 2 * count for extent, count in cycles),
    ]
    return {name: float(value) for name, value in zip(FEATURES, values)}, coverage


def cooling_event_count(cooling: pd.Series) -> int:
    """Count >10 C cooling peaks, separated by at least 24 hours."""
    values = cooling.dropna()
    count, last = 0, None
    array = values.to_numpy()
    for i, value in enumerate(array):
        if value <= 10 or (i and value <= array[i - 1]):
            continue
        if i < len(array) - 1 and value < array[i + 1]:
            continue
        time = values.index[i]
        if last is None or time - last >= pd.Timedelta(hours=24):
            count += 1
            last = time
    return count


def rainflow_cycles(values: np.ndarray) -> list[tuple[float, float]]:
    """Four-point thermal range counting; residual ranges count as half cycles.

    The squared-range aggregate is an exposure proxy, not calibrated pavement
    damage. Endpoints and all residuals are confined to this inspection window.
    """
    values = np.asarray(values, dtype=float)
    if not len(values):
        return []
    # Collapse equal neighbors, then retain direction reversals and endpoints.
    values = values[np.r_[True, np.diff(values) != 0]]
    if len(values) > 2:
        turn = np.diff(values[:-1]) * np.diff(values[1:]) < 0
        values = values[np.r_[True, turn, True]]
    stack, cycles = [], []
    for value in values:
        stack.append(float(value))
        while len(stack) >= 4:
            a, b, c, d = stack[-4:]
            extent = abs(c - b)
            if extent > abs(b - a) or extent > abs(d - c):
                break
            cycles.append((extent, 1.0))
            del stack[-3:-1]
    cycles.extend((abs(b - a), 0.5) for a, b in zip(stack, stack[1:]))
    return cycles


def _guard_existing(features: list[str]) -> None:
    for feature in features:
        compact = re.sub(r"[^a-z0-9]", "", feature.lower())
        separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", feature)
        words = re.split(r"[^a-z0-9]+", separated.lower())
        if (re.match(r"\d{2}[lmh]?(deduct|slabs|sqft|ft|count)", compact)
                or any(token in compact for token in ("deduct", "target", "date", "sectionid", "conditionindex",
                                                       "foreignobjectdamage", "networkid", "branchid", "latitude", "longitude"))
                or compact.startswith(("pci", "sci", "fod", "level", "rate", "section", "branch", "network", "rowid", "groupid",
                                       "family", "invalid", "exportgap", "delta", "inspection"))
                or compact.endswith(("pci", "sci", "fod"))
                or compact in {"year", "climatestatus", "climatecoverage"}
                or any(word in {"id", "date", "time", "lat", "lon", "row", "group", "section", "pci", "sci", "fod",
                               "branch", "network", "source", "material", "surface", "construction", "condition", "end"}
                       for word in words)):
            raise SafeError("UNSAFE_EXISTING_CLIMATE_FEATURE")


def _location_id(lat: float, lon: float) -> str:
    grid_lat = round(float(lat), 1)
    grid_lon = round(((float(lon) + 180) % 360) - 180, 1)
    grid_lat, grid_lon = grid_lat or 0.0, grid_lon or 0.0
    return hashlib.sha256(f"{DATASET}:{grid_lat:.1f}:{grid_lon:.1f}".encode()).hexdigest()[:24]


def enrich(table: pd.DataFrame, config: dict, out: Path,
           source_map: dict | None = None) -> tuple[pd.DataFrame, list[str], dict]:
    """Add climate features without dropping, duplicating or reordering rows.

    Only rows with ``climate_status == 'ok'`` are eligible for modeling. Every
    inspection gets a private audit row, including failed location/date/coverage
    checks. Existing mode accepts only explicitly declared numeric features;
    callers remain responsible for confirming their historical provenance.
    """
    Boundary.create(source_root(), out)
    for filename in ("climate_rows.csv", "climate_audit.json"):
        reject_links(out / filename)
    if "row_id" not in table or table["row_id"].isna().any() or table["row_id"].duplicated().any():
        raise SafeError("CLIMATE_REQUIRES_UNIQUE_ROW_ID")
    if not isinstance(config, dict) or set(config) - _CONFIG_KEYS:
        raise SafeError("CLIMATE_CONFIGURATION_INVALID")
    for name in ("latitude_column", "longitude_column"):
        if name in config and (not isinstance(config[name], str) or not config[name].strip()):
            raise SafeError("LOCATION_COLUMN_CONFIGURATION_INVALID")
    for name in ("mapping", "csv_dir", "cache_dir"):
        if name in config and (not isinstance(config[name], (str, Path)) or not str(config[name]).strip()):
            raise SafeError("CLIMATE_PATH_CONFIGURATION_INVALID")
    if "mode" in config and not isinstance(config["mode"], str):
        raise SafeError("CLIMATE_MODE_INVALID")
    result = table.copy()
    result["climate_location_id"] = ""
    mode = config.get("mode", "era5")
    records, requests = [], []
    audit = {"mode": mode, "rows": len(table), "requests": requests}

    def finish(features, failure):
        result["climate_status"] = [record["status"] for record in records]
        result["climate_coverage"] = [record["coverage"] for record in records]
        audit["counts"] = dict(Counter(record["status"] for record in records))
        audit["eligible_rows"] = sum(record["status"] == "ok" for record in records)
        audit["features"] = features
        pd.DataFrame(records).to_csv(out / "climate_rows.csv", index=False)
        write_json(out / "climate_audit.json", audit)
        if not audit["eligible_rows"]:
            raise SafeError(failure)
        return result, features, audit

    if mode == "existing":
        features = config.get("features")
        if not isinstance(features, list) or not features or any(not isinstance(f, str) for f in features):
            raise SafeError("EXISTING_CLIMATE_REQUIRES_EXPLICIT_FEATURES")
        if len(set(features)) != len(features) or not set(features) <= set(table):
            raise SafeError("EXISTING_CLIMATE_FEATURES_INVALID")
        _guard_existing(features)
        # Names alone cannot certify provenance; caller validates feature safety.
        for name in features:
            result[name] = pd.to_numeric(result[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for position, row_id in enumerate(table["row_id"]):
            valid = np.isfinite(result[features].iloc[position].to_numpy(dtype=float))
            records.append({"row_id": row_id, "status": "ok" if valid.all() else "invalid_existing_features",
                            "coverage": float(valid.mean())})
        audit["historical_window_verified"] = False
        locations, provenance = _locations(table, config, source_map or {})
        result["climate_location_id"] = [_location_id(*location) if location is not None else "" for location in locations]
        audit.update(provenance)
        audit["location_grouping"] = "coordinates_when_available" if any(locations) else "declared_input_groups_only"
        return finish(features, "NO_VALID_EXISTING_CLIMATE_ROWS")
    if mode != "era5":
        raise SafeError("CLIMATE_MODE_INVALID")
    lookback = config.get("lookback_days", 365)
    if type(lookback) is not int or not 1 <= lookback <= 36525:
        raise SafeError("CLIMATE_LOOKBACK_DAYS_INVALID")
    if "inspection_date" not in table:
        raise SafeError("CLIMATE_INSPECTION_DATE_MISSING")
    if any(name in table for name in FEATURES):
        raise SafeError("ERA5_FEATURE_COLUMNS_ALREADY_EXIST_USE_EXISTING_MODE")
    locations, provenance = _locations(table, config, source_map or {})
    audit.update(provenance)
    audit.update({"dataset": DATASET, "lookback_days": lookback, "day_basis": "UTC",
                  "window": "[inspection_date-lookback_days, inspection_date)",
                  "coverage_requirement": 1.0, "historical_window_verified": True,
                  "location_grouping": "era5_grid_points"})
    dates = pd.to_datetime(table["inspection_date"], format="ISO8601", errors="coerce", utc=True)
    today = pd.Timestamp.now(tz="UTC").normalize()
    groups = {}
    for position, (row_id, date, location) in enumerate(zip(table["row_id"], dates, locations)):
        record = {"row_id": row_id, "status": "pending", "coverage": 0.0}
        if pd.isna(date) or date.normalize() > today:
            record["status"] = "invalid_inspection_date"
        else:
            end = date.normalize()
            start = end - pd.Timedelta(days=lookback)
            record.update({"window_start": start.date().isoformat(),
                           "window_end_exclusive": end.date().isoformat(), "expected_hours": 24 * lookback})
            if start < _EARLIEST:
                record["status"] = "window_before_era5_coverage"
            elif location is None:
                record["status"] = "missing_location"
            else:
                groups.setdefault(location, []).append((position, end))
        records.append(record)
    for name in FEATURES:
        result[name] = np.nan
    if not groups:
        return finish(FEATURES.copy(), "NO_VALID_CLIMATE_LOCATIONS_OR_DATES")
    cache = Path(config.get("cache_dir", out / "era5_cache"))
    Boundary.create(source_root(), cache)
    # Cache calendar-year chunks for the union of needed exposure windows only.
    # Repeated inspection dates at a location share their derived metrics.
    for (lat, lon), inspections in groups.items():
        try:
            parts = [fetch_hourly(lat, lon, first, last, cache)
                     for first, last in _required_windows(inspections, lookback)]
            hourly, request_audit = _combine_hourly(parts)
        except SafeError as exc:
            for record in records:
                if record["status"] == "pending":
                    record["status"] = "not_processed_after_era5_error"
            for position, _ in inspections:
                records[position]["status"] = "era5_request_failed"
            audit["failure_code"] = str(exc)
            try:
                finish(FEATURES.copy(), str(exc))
            except SafeError:
                pass
            raise
        requests.append(request_audit)
        grid = request_audit.get("grid", {})
        # Use the returned grid point where available. Conservative 0.1-degree
        # grouping otherwise prevents near-identical requested points leaking
        # across holdouts; longitude wraps make +180 and -180 equivalent.
        location_id = _location_id(grid.get("latitude", lat), grid.get("longitude", lon))
        calculated = {}
        for position, date in inspections:
            if date not in calculated:
                calculated[date] = window_features(hourly, date, lookback)
            values, coverage = calculated[date]
            records[position].update({"coverage": coverage,
                                      "valid_hours": int(round(coverage * lookback * 24)),
                                      "cache_key": request_audit["cache_key"],
                                      "status": "ok" if values else "incomplete_hourly_coverage"})
            result.iat[position, result.columns.get_loc("climate_location_id")] = location_id
            for name, value in values.items():
                result.iat[position, result.columns.get_loc(name)] = value
    return finish(FEATURES.copy(), "NO_VALID_ERA5_CLIMATE_ROWS")
