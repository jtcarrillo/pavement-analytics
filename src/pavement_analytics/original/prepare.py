"""
Prepare a pairs-with-rate-and-climate file for local modeling: remove
leakage-prone, identifier, and junk columns, and emit an audit of every
column decision.

Tuned to the actual PAVER-derived column inventory (~460 columns).

Removal categories (each reported in the audit):
  LEAK_END      : *_end columns and endpoint-derived fields (PCI_end,
                  delta_PCI, end_date, IntervalEnd_*) — describe the outcome.
  LEAK_DENOM    : IntervalYears_* — the target's own denominator in years
                  (the same quantity as the weight column). A perfect shortcut.
  LEAK_WEIGHT   : delta_days — target denominator; passed through as the
                  sample-weight column, never a feature.
  IDENTIFIER    : Branch/Network/Section IDs and names, Section From/To,
                  source-file columns. All are proxies for "which airfield
                  and which pavement is this" — memorizable group identity.
                  The airfield and section columns are KEPT (renamed
                  partition_airfield / partition_section) for local grouped
                  validation; use one to assign groups and exclude both from
                  the feature list.
  RAW_DATE      : *Date_start and start_date — high-cardinality pseudo-IDs.
                  Their numeric twins (Years since * MR_start) are KEPT.
  TEXT_CODE     : ACN/PCN string encodings — free-form rating strings whose
                  numeric equivalents (ACN/PCN, Allowable *) are KEPT.
  ARTIFACT      : pipeline intermediates (LOCATION(DEGREES), Dates,
                  DataSource_*, SnapDistance* grid-snap QC metadata).
  GEO           : latitude/longitude — a geographic shortcut around the
                  climate features. Removed by default; --keep-latlon keeps.
  DISTRESS      : the ~400 per-distress quantity/deduct columns (e.g.
                  "74L deduct_start", "48M (Ft)_start"). PCI is computed
                  from these, so they are near-collinear with PCI_start and
                  extremely sparse. Removed by default for a lean,
                  interpretable model; --keep-distress retains them so the
                  lean and full models can be compared.
  CONSTANT      : single-valued or all-null — this is what automatically
                  removes the other materials' distress families and
                  Inspected Surface Type within a per-material file.
  HIGH_CARD_TEXT: free-text columns with near-unique values (pseudo-IDs).

Everything else is kept as a candidate feature. The audit CSV lists every
input column, its disposition, and the reason.

Usage:
    python -m pavement_analytics.original.prepare PAIRS_RATE_CLIMATE.xlsx
        [-o OUTPUT.csv] [--audit AUDIT.csv]
        [--target DetRatePerYear] [--weight delta_days]
        [--airfield-col _source_file_start] [--section-col SectionUniqueID]
        [--keep-distress] [--keep-latlon]
        [--max-delta-days N]
        [--high-card-threshold 0.95]
        [--extra-drop COL [COL ...]]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from . import check_cli_paths


# Endpoint-derived fields that are leakage regardless of suffix.
EXPLICIT_LEAK_END = {"PCI_end", "delta_PCI", "end_date", "DetRatePerYear", "delta_days"}

# Climate-block columns describing the interval itself rather than climate.
LEAK_INTERVAL_END = re.compile(r"^IntervalEnd(_|$)")
LEAK_INTERVAL_YEARS = re.compile(r"^IntervalYears(_|$)")
ARTIFACT_INTERVAL_START = re.compile(r"^IntervalStart(_|$)")

# Identity proxies: which airfield / which pavement.
IDENTIFIER_STEMS = {
    "Branch Name", "BranchID", "Network Name", "NetworkID", "SectionID",
    "Section From", "Section To",
}

# Free-form rating strings; numeric equivalents are kept.
TEXT_CODE_STEMS = {
    "ACN String", "ACN String Frost", "PCN String", "PCN String Frost",
}

# Pipeline intermediates / climate-run metadata.
ARTIFACT_EXACT = {"LOCATION(DEGREES)", "Dates"}
ARTIFACT_PREFIXES = ("DataSource", "SnapDistance", "_source_row", "_pipeline_", "export_gap_", "pci_diff_")

GEO_COLS = {"latitude", "longitude"}

# Per-distress inventory: stem begins with a distress code such as "74L",
# "48", "63H", followed by a unit or "deduct".
DISTRESS_RE = re.compile(
    r"^\d{2}[LMH]?\s+(deduct|\(Slabs\)|\(SqFt\)|\(Ft\)|\(Count\))", re.IGNORECASE
)

# Raw PAVER date columns (their "Years since ..." twins survive).
RAW_DATE_RE = re.compile(r"Date$", re.IGNORECASE)


def stem(col: str) -> str:
    """Column name with a trailing _start removed."""
    return col[:-6] if col.endswith("_start") else col


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def classify(
    df: pd.DataFrame,
    target: str,
    weight: str,
    airfield_col: str,
    section_col: str,
    keep_latlon: bool,
    keep_distress: bool,
    high_card_threshold: float,
    extra_drop: set[str],
) -> pd.DataFrame:
    """Return audit frame: column, disposition, reason."""
    rows = []

    for col in df.columns:
        s = stem(col)

        # --- reserved roles first -------------------------------------------
        if col == target:
            rows.append((col, "TARGET", "modeling target"))
            continue
        if col == weight:
            rows.append((col, "WEIGHT", "sample-weight column (target "
                                        "denominator; never a feature)"))
            continue
        if col == airfield_col:
            rows.append((col, "PARTITION", "airfield identity kept ONLY for group "
                                           "partitioning (-> partition_airfield)"))
            continue
        if col == section_col:
            rows.append((col, "PARTITION", "section identity kept ONLY for optional "
                                           "section-level grouping "
                                           "(-> partition_section)"))
            continue

        # --- explicit user drops ---------------------------------------------
        if col in extra_drop:
            rows.append((col, "DROP", "user-specified --extra-drop"))
            continue

        # --- leakage -----------------------------------------------------------
        if col in EXPLICIT_LEAK_END:
            rows.append((col, "DROP", "LEAK_END: endpoint-derived (encodes outcome)"))
            continue
        if col.endswith(("_end", "_delta")) or re.search(r" (emerged|regressed|resolved|persistent)$", col):
            rows.append((col, "DROP", "LEAK_END: ending-inspection attribute "
                                      "(unknown at prediction time)"))
            continue
        if LEAK_INTERVAL_END.match(col):
            rows.append((col, "DROP", "LEAK_END: interval end date "
                                      "(endpoint knowledge)"))
            continue
        if LEAK_INTERVAL_YEARS.match(col):
            rows.append((col, "DROP", "LEAK_DENOM: interval length in years = the "
                                      "target's denominator (duplicate of weight)"))
            continue

        # --- identifiers --------------------------------------------------------
        if col.startswith("_source_file"):
            rows.append((col, "IDENTIFIER-DROP", "source-workbook identity "
                                                 "(airfield identity is the "
                                                 "partition column)"))
            continue
        if s in IDENTIFIER_STEMS:
            rows.append((col, "IDENTIFIER-DROP", "location/asset identity proxy — "
                                                 "model would memorize the airfield"))
            continue

        # --- artifacts / metadata ------------------------------------------------
        if col in ARTIFACT_EXACT or col.startswith(ARTIFACT_PREFIXES):
            rows.append((col, "DROP", "ARTIFACT: pipeline or climate-run metadata"))
            continue
        if ARTIFACT_INTERVAL_START.match(col):
            rows.append((col, "DROP", "ARTIFACT: interval start date (raw date)"))
            continue
        if col == "start_date" or RAW_DATE_RE.search(s):
            rows.append((col, "DROP", "RAW_DATE: high-cardinality date; numeric "
                                      "'Years since ...' twin retained where present"))
            continue
        if s in TEXT_CODE_STEMS:
            rows.append((col, "DROP", "TEXT_CODE: rating string; numeric equivalent "
                                      "(ACN/PCN, Allowable *) retained"))
            continue

        # --- optional families ----------------------------------------------------
        if col in GEO_COLS and not keep_latlon:
            rows.append((col, "DROP", "GEO: geographic shortcut around the climate "
                                      "features (--keep-latlon to retain)"))
            continue
        if (DISTRESS_RE.match(s) or s.startswith(("fam_", "loss_shap_", "loss_loo_", "pci_only_"))
                or re.match(r"^\d{2}[LMH]? (density|sevindex)", s)) and not keep_distress:
            rows.append((col, "DROP", "DISTRESS: per-distress quantity/deduct; PCI is "
                                      "computed from these (collinear with PCI_start, "
                                      "very sparse). --keep-distress to retain"))
            continue

        # --- data-driven drops -----------------------------------------------------
        series = df[col]
        nunique = series.nunique(dropna=True)
        if nunique <= 1:
            rows.append((col, "DROP", "CONSTANT: single-valued or all-null"))
            continue
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            non_null = int(series.notna().sum())
            if non_null > 0 and nunique / non_null >= high_card_threshold:
                rows.append((col, "DROP", f"HIGH_CARD_TEXT: {nunique} distinct in "
                                          f"{non_null} non-null rows (pseudo-ID)"))
                continue

        rows.append((col, "KEEP", "candidate feature"))

    return pd.DataFrame(rows, columns=["column", "disposition", "reason"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", type=Path, help="Pairs-with-rate-and-climate file")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Local modeling CSV (default: <stem>_modeling.csv)")
    p.add_argument("--audit", type=Path, default=None,
                   help="Audit CSV of every column decision "
                        "(default: <stem>_modeling_audit.csv)")
    p.add_argument("--target", default="DetRatePerYear")
    p.add_argument("--weight", default="delta_days")
    p.add_argument("--airfield-col", default="_source_file_start")
    p.add_argument("--section-col", default="SectionUniqueID")
    p.add_argument("--keep-distress", action="store_true",
                   help="Retain the ~400 per-distress quantity/deduct columns "
                        "(default: dropped for a lean, interpretable model)")
    p.add_argument("--keep-latlon", action="store_true",
                   help="Retain latitude/longitude as features")
    p.add_argument("--max-delta-days", type=int, default=None,
                   help="Optional cap on interval length; rows with a longer "
                        "delta_days are excluded (e.g. 10958 for ~30 years)")
    p.add_argument("--high-card-threshold", type=float, default=0.95,
                   help="Text columns with distinct/non-null ratio >= this are "
                        "dropped as pseudo-identifiers (default 0.95)")
    p.add_argument("--extra-drop", nargs="*", default=[],
                   help="Additional column names to drop")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    check_cli_paths(inputs=(args.input,))
    if not args.input.is_file():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        return 1

    out_path = args.output or args.input.with_name(
        f"{args.input.stem}_modeling.csv")
    audit_path = args.audit or args.input.with_name(
        f"{args.input.stem}_modeling_audit.csv")
    check_cli_paths(outputs=(out_path, audit_path))

    print(f"Reading {args.input}...")
    df = read_table(args.input)
    print(f"  {len(df)} rows, {len(df.columns)} columns.")

    for required in (args.target, args.weight):
        if required not in df.columns:
            print(f"ERROR: required column '{required}' not found.", file=sys.stderr)
            return 1
    for wanted, flag in ((args.airfield_col, "--airfield-col"),
                         (args.section_col, "--section-col")):
        if wanted not in df.columns:
            print(f"  ! Partition column '{wanted}' not found (set {flag}); "
                  f"grouped validation will not be possible downstream.",
                  file=sys.stderr)

    audit = classify(
        df, args.target, args.weight, args.airfield_col, args.section_col,
        args.keep_latlon, args.keep_distress, args.high_card_threshold,
        set(args.extra_drop),
    )

    keep_cols = audit.loc[
        audit["disposition"].isin(["TARGET", "WEIGHT", "PARTITION", "KEEP"]),
        "column"].tolist()
    model_df = df[keep_cols].copy()
    renames = {}
    if args.airfield_col in model_df.columns:
        renames[args.airfield_col] = "partition_airfield"
    if args.section_col in model_df.columns:
        renames[args.section_col] = "partition_section"
    model_df = model_df.rename(columns=renames)

    # Row filters.
    n0 = len(model_df)
    w = pd.to_numeric(model_df[args.weight], errors="coerce")
    y = pd.to_numeric(model_df[args.target], errors="coerce")
    usable = np.isfinite(y) & np.isfinite(w) & (w > 0)
    model_df[args.target], model_df[args.weight] = y, w
    model_df = model_df.loc[usable]
    if len(model_df) < n0:
        print(f"  Dropped {n0 - len(model_df)} row(s) with missing target or "
              f"missing/non-positive weight.")
    if args.max_delta_days is not None:
        n1 = len(model_df)
        keep = pd.to_numeric(model_df[args.weight], errors="coerce") \
            <= args.max_delta_days
        model_df = model_df.loc[keep]
        print(f"  Dropped {n1 - len(model_df)} row(s) exceeding "
              f"--max-delta-days {args.max_delta_days} "
              f"({args.max_delta_days / 365.25:.1f} years).")

    model_df.to_csv(out_path, index=False)
    audit.to_csv(audit_path, index=False)

    # Report.
    counts = audit["disposition"].value_counts()
    print("\nColumn dispositions:")
    for disp in ["TARGET", "WEIGHT", "PARTITION", "KEEP", "IDENTIFIER-DROP", "DROP"]:
        if disp in counts:
            print(f"  {disp:<16} {counts[disp]}")
    drops = audit.loc[audit.disposition.str.contains("DROP"), "reason"]
    breakdown = drops.str.split(":").str[0].value_counts()
    if len(breakdown):
        print("  Drop breakdown:")
        for reason, cnt in breakdown.items():
            print(f"    {reason[:48]:<48} {cnt}")

    kept = audit.loc[audit.disposition == "KEEP", "column"].tolist()
    print(f"\nWrote modeling file: {out_path}  "
          f"({len(model_df)} rows, {len(model_df.columns)} columns; "
          f"{len(kept)} candidate features)")
    print(f"Wrote audit:         {audit_path}")
    print("\nCandidate features:")
    for c in kept:
        print(f"    {c}")
    print("\nLocal modeling roles:")
    print(f"  * Target    = {args.target}")
    print(f"  * Weight    = {args.weight}   (sample weights, NOT a feature)")
    print("  * Partition = group partitioning on 'partition_airfield' "
          "(or 'partition_section')")
    print("  * Exclude both partition_* columns from the feature list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
