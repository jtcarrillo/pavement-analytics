"""
Split a directory of .xlsx files into four categorized output files based on
the value in the "Inspected Surface Type" column.

Categories:
    1) PCC          -> rows where surface type == "PCC"
    2) APC          -> rows where surface type == "APC"
    3) AC_AAC_ACC   -> rows where surface type is one of {"AC", "AAC", "ACC"}
    4) OTHER        -> everything else (including blanks / unrecognized values)

In addition to the four combined files, each category is also split by the
"Date" column into two more files:
    <name>_pre1980.xlsx   -> Date strictly before Jan 1 of the cutoff year
    <name>_post1980.xlsx  -> Date on/after Jan 1 of the cutoff year
Rows whose Date is missing or unparseable cannot be assigned to either side;
they are written to <name>_undated.xlsx (only created if non-empty).

Usage:
    python split_by_surface_type.py INPUT_DIR [-o OUTPUT_DIR] [--recursive]
                                    [--column "Inspected Surface Type"]
                                    [--case-sensitive]
                                    [--date-col Date]
                                    [--cutoff-year 1980]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import check_cli_paths


COLUMN_DEFAULT = "Inspected Surface Type"

# (output_filename_stem, set of accepted values)
CATEGORIES: list[tuple[str, set[str]]] = [
    ("PCC", {"PCC"}),
    ("APC", {"APC"}),
    ("AC_AAC_ACC", {"AC", "AAC", "ACC"}),
]


def find_xlsx_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.xlsx" if recursive else "*.xlsx"
    # Filter out Excel lock/temp files (start with ~$).
    return sorted(p for p in input_dir.glob(pattern) if not p.name.startswith("~$"))


def load_and_tag(files: list[Path], column: str) -> pd.DataFrame:
    """Read every xlsx, concatenate, and add a _source_file column."""
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_excel(f)
        except Exception as e:
            print(f"  ! Skipping {f.name}: {e}", file=sys.stderr)
            continue

        if column not in df.columns:
            print(
                f"  ! Skipping {f.name}: missing column '{column}' "
                f"(found: {list(df.columns)})",
                file=sys.stderr,
            )
            continue

        df = df.copy()
        df["_source_file"] = f.name
        frames.append(df)
        print(f"  + {f.name}: {len(df)} rows")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def categorize(
    combined: pd.DataFrame, column: str, case_sensitive: bool
) -> dict[str, pd.DataFrame]:
    """Return a dict mapping output stem -> dataframe of matching rows."""
    values = combined[column].astype("string").str.strip()
    if not case_sensitive:
        values_cmp = values.str.upper()
        categories_cmp = [(name, {v.upper() for v in vals}) for name, vals in CATEGORIES]
    else:
        values_cmp = values
        categories_cmp = CATEGORIES

    assigned = pd.Series(False, index=combined.index)
    result: dict[str, pd.DataFrame] = {}

    for name, accepted in categories_cmp:
        mask = values_cmp.isin(accepted)
        result[name] = combined.loc[mask].copy()
        assigned |= mask

    result["OTHER"] = combined.loc[~assigned].copy()
    return result


def write_outputs(
    buckets: dict[str, pd.DataFrame],
    output_dir: Path,
    date_col: str,
    cutoff_year: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(year=cutoff_year, month=1, day=1)
    print("\nWriting outputs:")
    for name, df in buckets.items():
        # Combined file (unchanged behavior).
        out_path = output_dir / f"{name}.xlsx"
        df.to_excel(out_path, index=False)
        print(f"  -> {out_path}  ({len(df)} rows)")

        # Date-based split.
        if date_col not in df.columns:
            print(f"     ! No '{date_col}' column — skipping pre/post split for {name}.",
                  file=sys.stderr)
            continue

        dates = pd.to_datetime(df[date_col], errors="coerce")
        pre_mask = dates < cutoff
        post_mask = dates >= cutoff
        undated_mask = dates.isna()

        pre_path = output_dir / f"{name}_pre{cutoff_year}.xlsx"
        post_path = output_dir / f"{name}_post{cutoff_year}.xlsx"
        df.loc[pre_mask].to_excel(pre_path, index=False)
        df.loc[post_mask].to_excel(post_path, index=False)
        print(f"     -> {pre_path.name}  ({int(pre_mask.sum())} rows)")
        print(f"     -> {post_path.name}  ({int(post_mask.sum())} rows)")

        n_undated = int(undated_mask.sum())
        if n_undated:
            undated_path = output_dir / f"{name}_undated.xlsx"
            df.loc[undated_mask].to_excel(undated_path, index=False)
            print(f"     -> {undated_path.name}  ({n_undated} rows with "
                  f"missing/unparseable {date_col})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", type=Path, help="Directory containing .xlsx files")
    p.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Directory to write the 4 output files (default: <input_dir>/split_output)",
    )
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    p.add_argument(
        "--column", default=COLUMN_DEFAULT,
        help=f"Column name to split on (default: '{COLUMN_DEFAULT}')",
    )
    p.add_argument(
        "--case-sensitive", action="store_true",
        help="Match category values case-sensitively (default: case-insensitive)",
    )
    p.add_argument(
        "--date-col", default="Date",
        help="Date column used for the pre/post split (default: 'Date')",
    )
    p.add_argument(
        "--cutoff-year", type=int, default=1980,
        help="Cutoff year: rows before Jan 1 of this year go to the "
             "pre file, the rest to the post file (default: 1980)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    check_cli_paths(inputs=(args.input_dir,))

    if not args.input_dir.is_dir():
        print(f"ERROR: {args.input_dir} is not a directory", file=sys.stderr)
        return 1

    output_dir = args.output_dir or (args.input_dir / "split_output")
    check_cli_paths(outputs=(output_dir,))

    print(f"Scanning {args.input_dir} ({'recursive' if args.recursive else 'top-level'})...")
    files = find_xlsx_files(args.input_dir, args.recursive)
    if not files:
        print("No .xlsx files found.", file=sys.stderr)
        return 1
    print(f"Found {len(files)} file(s).\n")

    combined = load_and_tag(files, args.column)
    if combined.empty:
        print("\nERROR: no readable rows found.", file=sys.stderr)
        return 1
    print(f"\nTotal rows combined: {len(combined)}")

    buckets = categorize(combined, args.column, args.case_sensitive)
    write_outputs(buckets, output_dir, args.date_col, args.cutoff_year)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
