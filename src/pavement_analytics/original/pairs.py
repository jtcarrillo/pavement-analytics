"""
Build PCI deterioration pairs from a categorized xlsx file (output of
split_by_surface_type.py, or any xlsx with the required columns).

For each source workbook and SectionUniqueID, sort rows by Date and emit pairs of consecutive
inspections where PCI does NOT increase. Specifically:

  - Missing identities, invalid dates, nonnumeric PCI, and PCI outside 0..100
    are dropped (with a warning). The Python API accepts group_cols to override
    the default source-workbook/section grouping.
  - Same-day candidates are counted and skipped; only positive-duration pairs
    are emitted.
  - Within a section, rows are ordered by Date ascending. If two rows share
    the same Date, the one with the higher PCI is treated as earlier
    (deterioration goes downward).
  - Equal-PCI handling (conditional dedup):
        When two consecutive rows have equal PCI, the SECOND row is examined:
          * If the second row has IsConstruction == FALSE AND the two Dates
            differ, BOTH rows are kept and a flat pair (delta_PCI = 0) is
            emitted — a genuine re-inspection showing no deterioration.
          * Otherwise (second row is a construction record, or the dates are
            identical), the EARLIER row is dropped (keep the later one).
            This preserves the old behavior for true duplicates:
            [100, 100, 50] -> [100, 50].
        If the IsConstruction column is absent, all equal-PCI runs collapse
        (the original unconditional behavior) and a warning is printed.
  - For each consecutive (prev, curr) pair after the cleanup:
        if curr <= prev  -> emit pair (prev -> curr), then advance prev = curr
        if curr >  prev  -> skip the pair (it's an increase / M&R event), but
                            still advance prev = curr so curr can pair with
                            the next row ("reset at the increase").
    This reproduces: [100, 75, 50, 75, 50] -> (100,75), (75,50), (75,50).

Output columns:
    SectionUniqueID
    start_date, end_date
    PCI_start, PCI_end
    delta_PCI      (= PCI_end - PCI_start, <= 0 by construction)
    delta_days     (= end_date - start_date in days)
    <every other column>_start, <every other column>_end

Also writes <output_stem>_summary.xlsx with run diagnostics.

Usage:
    python build_pci_pairs.py INPUT.xlsx [-o OUTPUT.xlsx]
                              [--id-col SectionUniqueID]
                              [--pci-col PCI]
                              [--date-col Date]
                              [--construction-col IsConstruction]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from . import check_cli_paths


_FALSY = {"FALSE", "F", "NO", "N", "0", "0.0"}


def is_construction_false(value) -> bool:
    """Return True only if the value clearly indicates 'not construction'.

    Conservative: missing/unrecognized values are treated as NOT-False
    (i.e., the row will be collapsed like the old behavior).
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value is False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    s = str(value).strip().upper()
    return s in _FALSY


def build_pairs_for_section(
    section_df: pd.DataFrame,
    id_col: str,
    pci_col: str,
    date_col: str,
    construction_col: str | None,
) -> tuple[list[dict], dict]:
    """Return (pair_dicts, stats_dict) for one section."""
    raw_rows = len(section_df)

    # Sort: Date asc, then PCI desc so same-date rows go high -> low.
    s = section_df.sort_values(
        [date_col, pci_col], ascending=[True, False]
    ).reset_index(drop=True)

    # Conditional collapse of equal-PCI duplicates.
    # Walk consecutive rows; drop row i when row i+1 has the same PCI UNLESS
    # row i+1 qualifies as a genuine flat re-inspection:
    #   IsConstruction == FALSE on row i+1  AND  Date differs from row i.
    keep_mask = [True] * len(s)
    flat_pairs_preserved = 0
    for i in range(len(s) - 1):
        if s.loc[i, pci_col] != s.loc[i + 1, pci_col]:
            continue
        second_row = s.loc[i + 1]
        dates_differ = s.loc[i, date_col] != second_row[date_col]
        second_not_construction = (
            construction_col is not None
            and is_construction_false(second_row[construction_col])
        )
        if second_not_construction and dates_differ:
            flat_pairs_preserved += 1  # keep both rows
        else:
            keep_mask[i] = False  # collapse: drop earlier, keep later
    s = s.loc[keep_mask].reset_index(drop=True)
    rows_after_dedup = len(s)
    duplicates_collapsed = raw_rows - rows_after_dedup

    stats = {
        id_col: section_df[id_col].iloc[0],
        "raw_rows": raw_rows,
        "rows_after_dedup": rows_after_dedup,
        "duplicates_collapsed": duplicates_collapsed,
        "flat_pairs_preserved": flat_pairs_preserved,
        "pairs_emitted": 0,
        "increases_skipped": 0,
        "nonpositive_intervals_skipped": 0,
        "first_date": s[date_col].min() if rows_after_dedup else pd.NaT,
        "last_date": s[date_col].max() if rows_after_dedup else pd.NaT,
        "max_pci": s[pci_col].max() if rows_after_dedup else pd.NA,
        "min_pci": s[pci_col].min() if rows_after_dedup else pd.NA,
    }

    if len(s) < 2:
        return [], stats

    other_cols = [c for c in s.columns if c not in {id_col, pci_col, date_col}]

    pairs: list[dict] = []
    prev_idx = 0
    for curr_idx in range(1, len(s)):
        prev_pci = s.loc[prev_idx, pci_col]
        curr_pci = s.loc[curr_idx, pci_col]

        if curr_pci > prev_pci:
            stats["increases_skipped"] += 1

        days = (s.loc[curr_idx, date_col] - s.loc[prev_idx, date_col]).days
        if days <= 0:
            stats["nonpositive_intervals_skipped"] += 1

        if curr_pci <= prev_pci and days > 0:
            prev_row = s.loc[prev_idx]
            curr_row = s.loc[curr_idx]
            pair = {
                id_col: prev_row[id_col],
                "start_date": prev_row[date_col],
                "end_date": curr_row[date_col],
                f"{pci_col}_start": prev_pci,
                f"{pci_col}_end": curr_pci,
                "delta_PCI": curr_pci - prev_pci,
                "delta_days": (curr_row[date_col] - prev_row[date_col]).days
                if pd.notna(curr_row[date_col]) and pd.notna(prev_row[date_col])
                else pd.NA,
            }
            for c in other_cols:
                pair[f"{c}_start"] = prev_row[c]
                pair[f"{c}_end"] = curr_row[c]
            pairs.append(pair)

        # In both cases (deterioration or increase), advance the anchor.
        prev_idx = curr_idx

    stats["pairs_emitted"] = len(pairs)
    return pairs, stats


def build_pairs(
    df: pd.DataFrame,
    id_col: str,
    pci_col: str,
    date_col: str,
    construction_col: str | None = "IsConstruction",
    group_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (pairs_df, per_section_stats_df, run_summary_dict)."""
    for required in (id_col, pci_col, date_col):
        if required not in df.columns:
            raise ValueError(f"Input is missing required column: '{required}'")

    # Construction column is optional but recommended.
    if construction_col is not None and construction_col not in df.columns:
        print(
            f"  ! Column '{construction_col}' not found — equal-PCI rows will "
            f"always be collapsed (old behavior).",
            file=sys.stderr,
        )
        construction_col = None

    # Parse dates (idempotent if already datetime).
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[pci_col] = pd.to_numeric(df[pci_col], errors="coerce")
    # Section IDs need not be globally unique across source workbooks.
    group_cols = group_cols or (["_source_file", id_col] if "_source_file" in df else [id_col])
    for col in group_cols:
        if col not in df:
            raise ValueError(f"Missing grouping column: {col}")
    valid_id = df[group_cols].notna().all(axis=1)
    for col in group_cols:
        valid_id &= df[col].astype("string").str.strip().ne("").fillna(False)

    # Drop rows missing PCI or Date.
    initial = len(df)
    valid = valid_id & df[date_col].notna() & np.isfinite(df[pci_col]) & df[pci_col].between(0, 100)
    df = df.loc[valid]
    dropped = initial - len(df)
    if dropped:
        print(f"  Dropped {dropped} row(s) with invalid identity, date, or PCI (must be 0..100).",
              file=sys.stderr)

    all_pairs: list[dict] = []
    section_stats: list[dict] = []
    for _, section_df in df.groupby(group_cols, sort=False):
        pairs, stats = build_pairs_for_section(
            section_df, id_col, pci_col, date_col, construction_col
        )
        stats.update({col: section_df[col].iloc[0] for col in group_cols})
        section_stats.append(stats)
        all_pairs.extend(pairs)

    stats_df = pd.DataFrame(section_stats)
    n_sections = len(stats_df)
    n_with_pairs = int((stats_df["pairs_emitted"] > 0).sum()) if n_sections else 0

    print(f"  Processed {n_sections} section(s); "
          f"{n_with_pairs} produced at least one pair.")
    print(f"  Total pairs emitted: {len(all_pairs)}")

    run_summary = {
        "input_rows_after_dropna": initial - dropped,
        "rows_dropped_missing_pci_or_date": dropped,
        "rows_dropped_invalid": dropped,
        "group_columns": group_cols,
        "sections_total": n_sections,
        "sections_with_pairs": n_with_pairs,
        "sections_without_pairs": n_sections - n_with_pairs,
        "total_pairs": len(all_pairs),
        "total_increases_skipped": int(stats_df["increases_skipped"].sum()) if n_sections else 0,
        "total_nonpositive_intervals_skipped": int(stats_df["nonpositive_intervals_skipped"].sum()) if n_sections else 0,
        "total_duplicates_collapsed": int(stats_df["duplicates_collapsed"].sum()) if n_sections else 0,
        "total_flat_pairs_preserved": int(stats_df["flat_pairs_preserved"].sum()) if n_sections else 0,
        "construction_col_used": construction_col or "(absent)",
    }

    if not all_pairs:
        columns = [id_col, "start_date", "end_date", f"{pci_col}_start", f"{pci_col}_end", "delta_PCI", "delta_days"]
        columns += [f"{c}_{when}" for c in df.columns if c not in {id_col, pci_col, date_col}
                    for when in ("start", "end")]
        return pd.DataFrame(columns=columns), stats_df, run_summary

    out = pd.DataFrame(all_pairs)
    # Nice column order: id, dates, PCI, deltas, then *_start / *_end interleaved.
    front = [id_col, "start_date", "end_date",
             f"{pci_col}_start", f"{pci_col}_end",
             "delta_PCI", "delta_days"]
    rest = [c for c in out.columns if c not in front]
    bases: list[str] = []
    for c in rest:
        base = c[:-6] if c.endswith("_start") else c[:-4] if c.endswith("_end") else None
        if base and base not in bases:
            bases.append(base)
    interleaved = []
    for b in bases:
        if f"{b}_start" in out.columns: interleaved.append(f"{b}_start")
        if f"{b}_end"   in out.columns: interleaved.append(f"{b}_end")
    leftover = [c for c in rest if c not in interleaved]
    return out[front + interleaved + leftover], stats_df, run_summary


def make_distribution_table(series: pd.Series, bins: list, labels: list[str], name: str) -> pd.DataFrame:
    """Bucket a numeric series and return a tidy count/percent table."""
    cuts = pd.cut(series.dropna(), bins=bins, labels=labels, include_lowest=True, right=True)
    counts = cuts.value_counts().reindex(labels, fill_value=0)
    total = int(counts.sum())
    pct = (counts / total * 100).round(1) if total else counts * 0
    return pd.DataFrame({name: counts.index, "count": counts.values, "pct": pct.values})


def build_summary_sheets(
    pairs_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    run_summary: dict,
    pci_col: str,
) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {}

    sheets["run_summary"] = pd.DataFrame(
        {"metric": list(run_summary.keys()), "value": list(run_summary.values())}
    )

    if pairs_df.empty:
        return sheets

    # Pairs-per-section distribution.
    pps = stats_df["pairs_emitted"]
    pps_dist = pps.value_counts().sort_index().rename_axis("pairs_per_section").reset_index(name="n_sections")
    pps_dist["pct_of_sections"] = (pps_dist["n_sections"] / len(stats_df) * 100).round(1)
    sheets["pairs_per_section"] = pps_dist

    # Delta-PCI distribution (drops; values are <= 0). The "0 (flat)" bucket
    # includes the preserved flat re-inspection pairs.
    drops = -pairs_df["delta_PCI"]
    sheets["delta_pci_distribution"] = make_distribution_table(
        drops,
        bins=[-0.001, 0, 5, 10, 20, 40, 100, float("inf")],
        labels=["0 (flat)", "0-5", "5-10", "10-20", "20-40", "40-100", ">100"],
        name="drop_in_PCI",
    )

    # Time-gap distribution (years).
    years = pairs_df["delta_days"].astype("float") / 365.25
    sheets["delta_years_distribution"] = make_distribution_table(
        years,
        bins=[-0.001, 1, 2, 3, 5, 10, 20, float("inf")],
        labels=["<1", "1-2", "2-3", "3-5", "5-10", "10-20", ">20"],
        name="gap_years",
    )

    sheets["top_sections_by_pair_count"] = (
        stats_df.sort_values("pairs_emitted", ascending=False).head(25).reset_index(drop=True)
    )

    largest = pairs_df.assign(drop=-pairs_df["delta_PCI"]).nlargest(
        25, "drop"
    )[[c for c in pairs_df.columns if c in {
        pairs_df.columns[0], "start_date", "end_date",
        f"{pci_col}_start", f"{pci_col}_end", "delta_PCI", "delta_days",
    }]].reset_index(drop=True)
    sheets["top_pairs_by_drop"] = largest

    return sheets


def print_report(run_summary: dict, stats_df: pd.DataFrame, pairs_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in run_summary.items():
        print(f"  {k:40s} {v}")

    if pairs_df.empty:
        return

    pps = stats_df["pairs_emitted"]
    print(f"\n  pairs/section  mean={pps.mean():.2f}  "
          f"median={pps.median():.0f}  max={pps.max()}")

    no_pair = stats_df[stats_df["pairs_emitted"] == 0]
    if len(no_pair):
        only_one_row = int((no_pair["rows_after_dedup"] <= 1).sum())
        had_only_increases = len(no_pair) - only_one_row
        print(f"  sections with no pairs: {len(no_pair)} "
              f"(single row after dedup: {only_one_row}; "
              f"monotone-increasing or no drops: {had_only_increases})")

    total_transitions = run_summary["total_pairs"] + run_summary["total_increases_skipped"]
    if total_transitions:
        pct = run_summary["total_increases_skipped"] / total_transitions * 100
        print(f"  increases skipped: {run_summary['total_increases_skipped']} "
              f"of {total_transitions} transitions ({pct:.1f}%) — "
              f"likely M&R events worth examining separately")

    n_flat = run_summary.get("total_flat_pairs_preserved", 0)
    if n_flat:
        print(f"  flat re-inspection pairs preserved: {n_flat} "
              f"(equal PCI, IsConstruction=FALSE, different dates)")

    drops = -pairs_df["delta_PCI"]
    print(f"\n  PCI drop per pair: mean={drops.mean():.1f}  "
          f"median={drops.median():.1f}  p90={drops.quantile(0.9):.1f}  max={drops.max():.1f}")
    years = pairs_df["delta_days"].astype("float") / 365.25
    print(f"  years per pair:    mean={years.mean():.2f}  "
          f"median={years.median():.2f}  p90={years.quantile(0.9):.2f}  max={years.max():.2f}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Input .xlsx file")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output .xlsx (default: <input_stem>_pairs.xlsx)")
    p.add_argument("--id-col", default="SectionUniqueID")
    p.add_argument("--pci-col", default="PCI")
    p.add_argument("--date-col", default="Date")
    p.add_argument("--construction-col", default="IsConstruction",
                   help="Boolean-ish column marking construction records. "
                        "Pass an empty string to disable the conditional "
                        "flat-pair rule entirely.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    check_cli_paths(inputs=(args.input,))
    if not args.input.is_file():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        return 1

    out_path = args.output or args.input.with_name(f"{args.input.stem}_pairs.xlsx")
    summary_path = out_path.with_name(f"{out_path.stem}_summary.xlsx")
    check_cli_paths(outputs=(out_path, summary_path))

    construction_col = args.construction_col or None

    print(f"Reading {args.input}...")
    df = pd.read_excel(args.input)
    print(f"  {len(df)} rows, {len(df.columns)} columns.")

    try:
        pairs_df, stats_df, run_summary = build_pairs(
            df, args.id_col, args.pci_col, args.date_col, construction_col
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print_report(run_summary, stats_df, pairs_df)

    if pairs_df.empty:
        print("\nNo pairs generated; nothing to write.", file=sys.stderr)
        return 1

    pairs_df.to_excel(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(pairs_df)} rows, {len(pairs_df.columns)} columns).")

    sheets = build_summary_sheets(pairs_df, stats_df, run_summary, args.pci_col)
    with pd.ExcelWriter(summary_path) as xw:
        for name, sdf in sheets.items():
            sdf.to_excel(xw, sheet_name=name[:31], index=False)
    print(f"Wrote {summary_path}  ({len(sheets)} sheets: {', '.join(sheets)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
