"""
Enrich a pairs xlsx (output of build_pci_pairs.py) with the first
'LOCATION(DEGREES)' value from a mapped CSV file.

The mapping JSON has the form:
    {
        "source_a.xlsx": "location_a.csv",
        "source_b.xlsx": "location_b.csv"
    }

Keys are matched against the '_source_file_start' column in the pairs file.
The corresponding CSV is opened, the FIRST non-null value of the
'LOCATION(DEGREES)' column is taken, and that value is added as a new
'LOCATION(DEGREES)' column on every matching row.

Each CSV is read once and cached (~10k+ rows can have repeats).

Usage:
    python enrich_with_location.py PAIRS.xlsx MAPPING.json
                                   [--csv-dir DIR]
                                   [-o OUTPUT.xlsx]
                                   [--source-col _source_file_start]
                                   [--location-col "LOCATION(DEGREES)"]
                                   [--on-missing {warn,blank,fail}]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from . import check_cli_paths


_HEMI_LETTERS = {"N", "S", "E", "W"}


def _extract_signed(token: str) -> float | None:
    """Parse one token that may have N/S/E/W as prefix or suffix.

    Examples that all resolve to -2.0:
        "-2.0"   "2.0W"   "W2.0"   "w2.0"
    Returns None if the token can't be parsed as a number.
    """
    t = token.strip()
    if not t:
        return None
    hemi: str | None = None

    # Leading hemisphere?
    if t[0].upper() in _HEMI_LETTERS:
        hemi = t[0].upper()
        t = t[1:].strip()
    # Trailing hemisphere?
    elif t[-1].upper() in _HEMI_LETTERS:
        hemi = t[-1].upper()
        t = t[:-1].strip()

    try:
        val = float(t)
    except ValueError:
        return None

    if hemi in {"S", "W"}:
        val = -abs(val)
    elif hemi in {"N", "E"}:
        val = abs(val)
    return val


def parse_lat_lon(value: str | None) -> tuple[float | None, float | None]:
    """Parse a LOCATION(DEGREES) string into (lat, lon) decimal degrees.

    Handles:
        "1.0 -2.0"          (signed, space-separated)
        "1.0, -2.0"         (signed, comma-separated)
        "1.0N 2.0W"         (hemisphere suffix attached)
        "1.0 N 2.0 W"       (hemisphere suffix as separate token)
        "N 1.0 W 2.0"       (hemisphere prefix as separate token)
        "N1.0 W2.0"         (hemisphere prefix attached)
    Returns (None, None) if it can't be parsed.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None
    s = str(value).strip()
    if not s:
        return None, None

    # Split on commas / whitespace into tokens, then merge any standalone
    # hemisphere letter ("N") with its neighboring number token.
    raw = [t for t in re.split(r"[,\s]+", s) if t]

    merged: list[str] = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        is_solo_hemi = len(tok) == 1 and tok.upper() in _HEMI_LETTERS

        if is_solo_hemi:
            # Prefer attaching to a bare-number previous token (suffix form
            # "1.0 N"). Fall back to attaching to next (prefix form "N 1.0").
            attached = False
            if merged:
                prev = merged[-1]
                # "Bare number" = parses as float AND has no hemisphere letter yet.
                no_existing_hemi = not any(c.upper() in _HEMI_LETTERS for c in prev)
                if no_existing_hemi:
                    try:
                        float(prev)
                        merged[-1] = prev + tok
                        attached = True
                    except ValueError:
                        pass
            if not attached and i + 1 < len(raw):
                merged.append(tok + raw[i + 1])
                i += 2
                continue
            if attached:
                i += 1
                continue
            # Lone hemisphere with nowhere to attach — give up on it.
            i += 1
            continue

        merged.append(tok)
        i += 1

    if len(merged) < 2:
        return None, None

    lat = _extract_signed(merged[0])
    lon = _extract_signed(merged[1])

    if lat is None or lon is None:
        return None, None
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None, None
    return lat, lon


def load_location_from_csv(csv_path: Path, location_col: str) -> str | None:
    """Return the first non-null LOCATION(DEGREES) value from a CSV, or None."""
    # Try a small read first for speed; fall back to full read if needed.
    try:
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                df_head = pd.read_csv(csv_path, nrows=50, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
    except Exception as e:
        raise RuntimeError(f"failed to read {csv_path.name}: {e}") from e

    if location_col not in df_head.columns:
        raise RuntimeError(
            f"{csv_path.name} is missing column '{location_col}' "
            f"(found: {list(df_head.columns)})"
        )

    series = df_head[location_col].dropna()
    if not series.empty:
        return str(series.iloc[0])

    # Empty in first 50; scan the whole file.
    df_full = pd.read_csv(csv_path, usecols=[location_col], encoding=encoding)
    series = df_full[location_col].dropna()
    return str(series.iloc[0]) if not series.empty else None


def build_location_map(
    mapping: dict[str, str],
    csv_dir: Path,
    location_col: str,
    on_missing: str,
) -> dict[str, str | None]:
    """Resolve {source_xlsx: location_string} for every entry in mapping."""
    resolved: dict[str, str | None] = {}
    cache: dict[Path, str | None] = {}
    missing_files: list[str] = []
    missing_values: list[str] = []
    errors: list[str] = []

    for source_xlsx, csv_name in mapping.items():
        csv_path = csv_dir / csv_name
        if not csv_path.is_file():
            missing_files.append(f"{source_xlsx} -> {csv_name}")
            resolved[source_xlsx] = None
            continue

        if csv_path in cache:
            resolved[source_xlsx] = cache[csv_path]
            continue

        try:
            value = load_location_from_csv(csv_path, location_col)
        except RuntimeError as e:
            errors.append(str(e))
            resolved[source_xlsx] = None
            cache[csv_path] = None
            continue

        cache[csv_path] = value
        resolved[source_xlsx] = value
        if value is None:
            missing_values.append(f"{source_xlsx} -> {csv_name}")

    # Reporting.
    if missing_files:
        msg = f"{len(missing_files)} CSV file(s) missing on disk"
        print(f"  ! {msg}", file=sys.stderr)
        for line in missing_files[:5]:
            print(f"      {line}", file=sys.stderr)
        if len(missing_files) > 5:
            print(f"      ... ({len(missing_files) - 5} more)", file=sys.stderr)
        if on_missing == "fail":
            raise SystemExit(f"ERROR: {msg} and --on-missing=fail")

    if missing_values:
        print(f"  ! {len(missing_values)} CSV file(s) had no '{location_col}' value",
              file=sys.stderr)

    if errors:
        for e in errors:
            print(f"  ! {e}", file=sys.stderr)
        if on_missing == "fail":
            raise SystemExit("ERROR: CSV read errors and --on-missing=fail")

    return resolved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("pairs", type=Path, help="Pairs xlsx (input)")
    p.add_argument("mapping", type=Path, help="Mapping JSON file")
    p.add_argument("--csv-dir", type=Path, default=None,
                   help="Directory holding the mapped CSV files "
                        "(default: same dir as the mapping JSON)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output xlsx (default: <pairs_stem>_with_location.xlsx)")
    p.add_argument("--source-col", default="_source_file_start",
                   help="Column in pairs file holding the source xlsx name")
    p.add_argument("--location-col", default="LOCATION(DEGREES)",
                   help="Column in the CSVs to pull the first value from")
    p.add_argument("--on-missing", choices=["warn", "blank", "fail"], default="warn",
                   help="Behavior when a source has no mapping / CSV / value")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    check_cli_paths(inputs=(args.pairs, args.mapping, args.csv_dir))

    if not args.pairs.is_file():
        print(f"ERROR: {args.pairs} not found", file=sys.stderr)
        return 1
    if not args.mapping.is_file():
        print(f"ERROR: {args.mapping} not found", file=sys.stderr)
        return 1

    csv_dir = args.csv_dir or args.mapping.parent
    if not csv_dir.is_dir():
        print(f"ERROR: csv dir {csv_dir} is not a directory", file=sys.stderr)
        return 1

    out_path = args.output or args.pairs.with_name(
        f"{args.pairs.stem}_with_location.xlsx"
    )
    check_cli_paths(outputs=(out_path,))

    print(f"Reading {args.mapping}...")
    with args.mapping.open() as fh:
        mapping = json.load(fh)
    print(f"  {len(mapping)} mapping entries.")

    print(f"Resolving locations from CSVs in {csv_dir}...")
    location_map = build_location_map(
        mapping, csv_dir, args.location_col, args.on_missing
    )
    n_resolved = sum(v is not None for v in location_map.values())
    print(f"  Resolved {n_resolved}/{len(location_map)} mappings.")

    print(f"Reading {args.pairs}...")
    pairs_df = pd.read_excel(args.pairs)
    print(f"  {len(pairs_df)} rows, {len(pairs_df.columns)} columns.")

    if args.source_col not in pairs_df.columns:
        print(f"ERROR: pairs file is missing column '{args.source_col}'",
              file=sys.stderr)
        return 1

    pairs_df[args.location_col] = pairs_df[args.source_col].map(location_map)

    # Parse LOCATION(DEGREES) into latitude / longitude columns.
    parsed = pairs_df[args.location_col].map(parse_lat_lon)
    pairs_df["latitude"] = parsed.map(lambda t: t[0])
    pairs_df["longitude"] = parsed.map(lambda t: t[1])

    # Diagnostics on the merge.
    n_with_location = pairs_df[args.location_col].notna().sum()
    n_with_parsed = pairs_df["latitude"].notna().sum()
    n_total = len(pairs_df)
    print(f"  {n_with_location}/{n_total} rows got a raw location string "
          f"({n_with_location / n_total * 100:.1f}%).")
    print(f"  {n_with_parsed}/{n_total} rows parsed to valid lat/lon "
          f"({n_with_parsed / n_total * 100:.1f}%).")

    # Flag any location strings that came through but failed to parse.
    parse_failed = pairs_df[
        pairs_df[args.location_col].notna() & pairs_df["latitude"].isna()
    ]
    if len(parse_failed):
        bad = parse_failed[[args.source_col, args.location_col]].drop_duplicates()
        print(f"  ! {len(bad)} unique location string(s) failed to parse:",
              file=sys.stderr)
        for _, row in bad.head(10).iterrows():
            print(f"      {row[args.source_col]} -> {row[args.location_col]!r}",
                  file=sys.stderr)
        if len(bad) > 10:
            print(f"      ... ({len(bad) - 10} more)", file=sys.stderr)

    # Unmapped source files actually present in pairs:
    sources_in_pairs = set(pairs_df[args.source_col].dropna().unique())
    sources_in_mapping = set(mapping.keys())
    unmapped_present = sorted(sources_in_pairs - sources_in_mapping)
    if unmapped_present:
        print(f"  ! {len(unmapped_present)} source file(s) in pairs have no "
              f"JSON mapping entry:", file=sys.stderr)
        for s in unmapped_present[:10]:
            print(f"      {s}", file=sys.stderr)
        if len(unmapped_present) > 10:
            print(f"      ... ({len(unmapped_present) - 10} more)", file=sys.stderr)
        if args.on_missing == "fail":
            return 1

    pairs_df.to_excel(out_path, index=False)
    print(f"Wrote {out_path}  ({len(pairs_df)} rows, "
          f"{len(pairs_df.columns)} columns).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
