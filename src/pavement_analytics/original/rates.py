"""
Add a deterioration-rate column and ERA5-Land climate metrics to a
pairs-with-location file (output of 3_enrich_with_location.py).

Steps:
  1. Read the input (xlsx or csv). Must contain delta_PCI, delta_days,
     latitude, longitude, start_date, end_date.
  2. Add DetRatePerYear = delta_PCI / (delta_days / 365).
     (delta_PCI <= 0, so the rate is negative: PCI lost per year.)
     Pairs with delta_days <= 0 or missing get NaN (counted in a warning).
  3. Build a 'Dates' column = "[start_date, end_date]" per row — the format
     era5land_mean_temp_from_csv.py expects. Each pair therefore receives
     climate stats computed over its own deterioration interval.
  4. Invoke era5land_mean_temp_from_csv.py as a subprocess (path via
     --era5-script) on an intermediate CSV.
  5. Read the ERA5 output back and write the final file (.xlsx or .csv to
     match the requested output name).

The ERA5 stage requires the cdsapi package and CDS API credentials
(~/.cdsapirc) and may download data on first run; downloads are cached in
--cache-dir, so re-runs are fast. Use --skip-climate to only add the
deterioration rate.

Usage:
    python 4_add_rate_and_climate.py PAIRS_WITH_LOCATION.xlsx \
        --era5-script era5land_mean_temp_from_csv.py \
        [-o OUTPUT.xlsx] \
        [--cache-dir era5land_meantemp_cache] \
        [--lat-col latitude] [--lon-col longitude] \
        [--start-date-col start_date] [--end-date-col end_date] \
        [--construction-year-col COL] \
        [--skip-climate] \
        [--era5-extra-args "--end-year 2025 --pav-reliability 98"]
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from . import check_cli_paths


DAYS_PER_YEAR = 365.0  # per the requested formula: delta_PCI / (delta_days / 365)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def add_deterioration_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add DetRatePerYear = delta_PCI / (delta_days / 365)."""
    for required in ("delta_PCI", "delta_days"):
        if required not in df.columns:
            raise ValueError(f"Input is missing required column: '{required}'")

    delta_pci = pd.to_numeric(df["delta_PCI"], errors="coerce")
    delta_days = pd.to_numeric(df["delta_days"], errors="coerce")

    years = delta_days / DAYS_PER_YEAR
    rate = delta_pci / years
    # Guard: zero/negative/missing day spans produce inf or nonsense.
    invalid = delta_days.isna() | (delta_days <= 0)
    rate[invalid] = float("nan")

    df = df.copy()
    df["DetRatePerYear"] = rate

    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"  ! {n_invalid} pair(s) have delta_days <= 0 or missing — "
              f"DetRatePerYear set to NaN for those.", file=sys.stderr)
    valid = rate.dropna()
    if len(valid):
        print(f"  DetRatePerYear: n={len(valid)}  mean={valid.mean():.2f}  "
              f"median={valid.median():.2f}  p10={valid.quantile(0.1):.2f}  "
              f"min={valid.min():.2f}  (PCI/yr; negative = deterioration)")
    return df


def build_dates_column(
    df: pd.DataFrame, start_col: str, end_col: str
) -> pd.Series:
    """Format '[start, end]' strings the ERA5 script's parse_dates_cell expects."""
    start = pd.to_datetime(df[start_col], errors="coerce")
    end = pd.to_datetime(df[end_col], errors="coerce")

    def fmt(s, e):
        if pd.isna(s) or pd.isna(e):
            return ""
        return f"[{s.date().isoformat()}, {e.date().isoformat()}]"

    return pd.Series(
        [fmt(s, e) for s, e in zip(start, end)], index=df.index
    )


def run_era5_stage(
    df: pd.DataFrame,
    era5_script: Path,
    lat_col: str,
    lon_col: str,
    cache_dir: Path,
    construction_year_col: str | None,
    extra_args: str,
    workdir: Path,
) -> pd.DataFrame:
    """Write intermediate CSV, invoke the ERA5 script, read result back."""
    interim_in = workdir / "era5_input.csv"
    interim_out = workdir / "era5_output.csv"
    df.to_csv(interim_in, index=False)

    bundled_worker = Path(__file__).with_name("era5.py").resolve()
    worker = ([sys.executable, "-m", "pavement_analytics.original.era5"]
              if era5_script.resolve() == bundled_worker else [sys.executable, str(era5_script)])
    cmd = [
        *worker,
        "--input", str(interim_in),
        "--output", str(interim_out),
        "--lat-col", lat_col,
        "--lon-col", lon_col,
        "--dates-col", "Dates",
        "--cache-dir", str(cache_dir),
    ]
    if construction_year_col:
        cmd += ["--construction-year-col", construction_year_col]
    if extra_args:
        cmd += shlex.split(extra_args)

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"ERA5 script exited with code {result.returncode}. "
            f"Intermediate input preserved at {interim_in} for debugging."
        )
    if not interim_out.is_file():
        raise RuntimeError(f"ERA5 script did not produce {interim_out}")

    return pd.read_csv(interim_out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path,
                   help="Pairs-with-location file (.xlsx or .csv)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output file (default: <input_stem>_rate_climate.xlsx)")
    p.add_argument("--era5-script", type=Path,
                   default=Path(__file__).with_name("era5.py"),
                   help="Path to era5land_mean_temp_from_csv.py")
    p.add_argument("--cache-dir", type=Path,
                   default=None,
                   help="ERA5 cache directory (default: era5_cache beside the output; reused across runs)")
    p.add_argument("--lat-col", default="latitude")
    p.add_argument("--lon-col", default="longitude")
    p.add_argument("--start-date-col", default="start_date")
    p.add_argument("--end-date-col", default="end_date")
    p.add_argument("--construction-year-col", default=None,
                   help="Optional column with construction year, passed through "
                        "to the ERA5 script for the thermal-aging integral")
    p.add_argument("--skip-climate", action="store_true",
                   help="Only add DetRatePerYear; skip the ERA5 stage")
    p.add_argument("--era5-extra-args", default="",
                   help="Extra flags passed verbatim to the ERA5 script, e.g. "
                        "\"--end-year 2025 --pav-reliability 98\"")
    p.add_argument("--keep-interim", action="store_true",
                   help="Keep the intermediate CSVs next to the output "
                        "instead of using a temp directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    check_cli_paths(inputs=(args.input,))

    if not args.input.is_file():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        return 1
    if not args.skip_climate and not args.era5_script.is_file():
        print(f"ERROR: ERA5 script not found at {args.era5_script} "
              f"(pass --era5-script or --skip-climate)", file=sys.stderr)
        return 1

    out_path = args.output or args.input.with_name(
        f"{args.input.stem}_rate_climate.xlsx"
    )
    args.cache_dir = args.cache_dir or out_path.parent / "era5_cache"
    check_cli_paths(outputs=(out_path, args.cache_dir))

    print(f"Reading {args.input}...")
    df = read_table(args.input)
    print(f"  {len(df)} rows, {len(df.columns)} columns.")

    # Step 1: deterioration rate.
    try:
        df = add_deterioration_rate(df)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Step 2: climate metrics via the ERA5 script.
    if args.skip_climate:
        print("  Skipping climate stage (--skip-climate).")
    else:
        for required in (args.lat_col, args.lon_col,
                         args.start_date_col, args.end_date_col):
            if required not in df.columns:
                print(f"ERROR: input is missing column '{required}' "
                      f"needed for the climate stage", file=sys.stderr)
                return 1

        df["Dates"] = build_dates_column(
            df, args.start_date_col, args.end_date_col
        )
        n_no_interval = int((df["Dates"] == "").sum())
        if n_no_interval:
            print(f"  ! {n_no_interval} row(s) lack a valid date interval — "
                  f"climate columns will be NaN for those.", file=sys.stderr)

        n_no_coords = int(
            pd.to_numeric(df[args.lat_col], errors="coerce").isna().sum()
        )
        if n_no_coords:
            print(f"  ! {n_no_coords} row(s) lack valid coordinates — "
                  f"climate columns will be NaN for those.", file=sys.stderr)

        if args.keep_interim:
            workdir = out_path.parent
            df_result = run_era5_stage(
                df, args.era5_script, args.lat_col, args.lon_col,
                args.cache_dir, args.construction_year_col,
                args.era5_extra_args, workdir,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="rate_climate_", dir=out_path.parent.resolve()) as tmp:
                df_result = run_era5_stage(
                    df, args.era5_script, args.lat_col, args.lon_col,
                    args.cache_dir, args.construction_year_col,
                    args.era5_extra_args, Path(tmp),
                )
        df = df_result.drop(columns=["Dates"], errors="ignore")

    write_table(df, out_path)
    print(f"Wrote {out_path}  ({len(df)} rows, {len(df.columns)} columns).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
