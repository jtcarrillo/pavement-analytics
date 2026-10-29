#!/usr/bin/env python3
"""
prep_distress_targets.py -- turn sparse per-distress PAVER columns into modelable targets.

Reads a deterioration-pairs CSV with columns shaped like
    "74L deduct_start", "74L deduct_end", "74L (Slabs)_start", "74L (Slabs)_end"
Input may be .xlsx / .xlsm / .xls or .csv / .tsv / .txt. Output is always CSV,
for the later PCI calculations and local modeling stages.

and writes
    <out>.csv                 the input plus derived target columns
    <out>_prevalence.csv      one row per distress (code, severity, measure) --
                              how sparse it is, and how many airfields show it

Derived columns (each as _start / _end, plus _delta where noted):
    <code> deduct_total       deduct summed over severities (74L+74M+74H)      [+delta]
    <code><sev> density       raw quantity / InspectedSlabs (PCC) or
                              / InspectedArea (SqFt) (AC); Count columns skipped
    <code> density_total      density summed over severities                  [+delta]
    <code> sevindex           severity-weighted density (L=1, M=2, H=3)
    fam_<family> deduct       deduct_total summed over a mechanism family     [+delta]
    fam_<family> density      density_total summed over a mechanism family    [+delta]
    <code> present_end        1 if deduct_total_end > 0
    <code> emerged            1 if deduct_total_start == 0 and _end > 0
    <code> regressed          1 if deduct_total_end < _start   (usually M&R between inspections)

Usage:
    python -m pavement_analytics.original.distress pairs.xlsx --list-sheets
    python -m pavement_analytics.original.distress pairs.xlsx --report-only
    python -m pavement_analytics.original.distress pairs.xlsx --sheet Pairs --out pairs_targets
    python -m pavement_analytics.original.distress pairs.csv --report-only
    python -m pavement_analytics.original.distress pairs.csv --out pairs_targets
    python -m pavement_analytics.original.distress pairs.csv --out pairs_targets --families my_families.json
    python -m pavement_analytics.original.distress pairs.csv --group-col "Network Name_start"

Absent distresses are zero-filled (they are structural zeros in a PAVER export,
not missing data). Pass --keep-nan if that assumption is wrong for your export.
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from . import check_cli_paths

EXCEL_EXT = {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}
# PAVER exports are frequently Windows-encoded rather than UTF-8
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")

COL_RE = re.compile(
    r"^(?P<code>\d{2})(?P<sev>[LMH]?) "
    r"(?:(?P<deduct>deduct)|\((?P<unit>Slabs|SqFt|Ft|Count)\))"
    r"_(?P<when>start|end)$"
)

# ASTM D5340 distress codes grouped by the mechanism most plausibly tied to climate.
# A starting point, not a verdict -- edit here or pass --families <json>.
FAMILIES = {
    "pcc_freeze_thaw":  ["64", "70", "74", "75"],        # D-cracking, scaling, joint/corner spalling
    "pcc_thermal":      ["61", "63", "65"],              # blow-up, linear cracking, joint seal damage
    "pcc_moisture":     ["69", "71", "76"],              # pumping, faulting, ASR
    "pcc_load":         ["62", "72"],                    # corner break, shattered slab
    "pcc_other":        ["66", "67", "68", "73"],        # patches, popouts, shrinkage
    "ac_thermal_crack": ["43", "47", "48"],              # block, joint reflection, L&T cracking
    "ac_aging":         ["52", "57"],                    # raveling, weathering
    "ac_high_temp":     ["42", "53", "54"],              # bleeding, rutting, shoving
    "ac_load":          ["41", "45"],                    # alligator, depression
    "ac_subgrade":      ["44", "56"],                    # corrugation, swell
    "ac_other":         ["46", "49", "50", "51", "55"],  # jet blast, oil, patching, polished agg, slippage
}

SEV_WEIGHT = {"": 1.0, "L": 1.0, "M": 2.0, "H": 3.0}
DENOM = {"Slabs": "InspectedSlabs", "SqFt": "InspectedArea (SqFt)", "Ft": "InspectedArea (SqFt)"}
GROUP_CANDIDATES = ["Network Name_start", "NetworkID_start", "partition_airfield"]


def is_excel(path):
    return os.path.splitext(path)[1].lower() in EXCEL_EXT


def excel_engine(path):
    """Return the engine name, failing with an actionable message if it is missing."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xls":
        try:
            import xlrd  # noqa: F401
        except ImportError:
            sys.exit("reading .xls needs xlrd:  pip install 'xlrd<2'   "
                     "(or re-save as .xlsx, which is better supported)")
        return "xlrd"
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit(f"reading {ext} needs openpyxl:  pip install openpyxl")
    return "openpyxl"


def list_sheets(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return list(wb.sheetnames)
    finally:
        wb.close()


def read_header(path, sheet=None):
    """Column names only -- cheap, so validation can fail before a full load."""
    if is_excel(path):
        eng = excel_engine(path)
        try:
            hdr = pd.read_excel(path, sheet_name=sheet or 0, nrows=0, engine=eng)
        except ValueError as e:
            if "not found" in str(e) or "Worksheet" in str(e):
                names = list_sheets(path) if eng == "openpyxl" else []
                avail = ", ".join(repr(n) for n in names) or "(unknown)"
                sys.exit(f"sheet {sheet!r} not in {os.path.basename(path)}. "
                         f"Available: {avail}")
            raise
        return [str(c) for c in hdr.columns]
    for enc in CSV_ENCODINGS:
        try:
            with open(path, newline="", encoding=enc) as fh:
                return next(csv.reader(fh))
        except UnicodeDecodeError:
            continue
    sys.exit(f"could not decode {path} as any of {', '.join(CSV_ENCODINGS)}")


def load_table(path, sheet=None):
    """Read the whole table. Excel via pandas+openpyxl, CSV with encoding fallback."""
    if is_excel(path):
        eng = excel_engine(path)
        print(f"reading {os.path.basename(path)}"
              f"{f' [sheet: {sheet}]' if sheet else ''} -- Excel parsing is slow "
              "for wide files; the CSV this writes will load much faster next time")
        return pd.read_excel(path, sheet_name=sheet or 0, engine=eng)
    sep = "\t" if os.path.splitext(path)[1].lower() == ".tsv" else ","
    last = None
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc, sep=sep)
        except UnicodeDecodeError as e:
            last = e
    sys.exit(f"could not decode {path} as any of {', '.join(CSV_ENCODINGS)}: {last}")


def parse_columns(columns):
    rows = []
    for c in columns:
        m = COL_RE.match(c)
        if not m:
            continue
        d = m.groupdict()
        rows.append({
            "col": c, "code": d["code"], "sev": d["sev"] or "",
            "measure": "deduct" if d["deduct"] else d["unit"], "when": d["when"],
        })
    meta = pd.DataFrame(rows)
    if meta.empty:
        sys.exit("no columns matched the distress pattern, e.g. '74L deduct_start'")
    return meta


def pick_group_col(df, requested):
    if requested:
        if requested not in df.columns:
            sys.exit(f"--group-col '{requested}' not in file")
        return requested
    for c in GROUP_CANDIDATES:
        if c in df.columns:
            return c
    return None


def prevalence(df, meta, group_col):
    out = []
    for (code, sev, measure), g in meta.groupby(["code", "sev", "measure"]):
        cols = dict(zip(g["when"], g["col"]))
        s = df[cols["start"]] if "start" in cols else None
        e = df[cols["end"]] if "end" in cols else None
        rec = {"code": code, "sev": sev, "measure": measure, "n_rows": len(df)}
        if s is not None:
            rec["pct_nonzero_start"] = round((s > 0).mean() * 100, 2)
        if e is not None:
            rec["pct_nonzero_end"] = round((e > 0).mean() * 100, 2)
            pos = e[e > 0]
            rec["n_nonzero_end"] = int(len(pos))
            rec["median_nonzero_end"] = pos.median() if len(pos) else np.nan
            rec["n_distinct_nonzero_end"] = int(pos.round(2).nunique())
            if group_col:
                grp = df.loc[e > 0, group_col]
                rec["n_groups_nonzero_end"] = int(grp.nunique())
                rec["groups_if_trace"] = ("; ".join(sorted(map(str, grp.unique()))[:5])
                                          if 0 < grp.nunique() <= 5 else "")
        if s is not None and e is not None:
            rec["pct_emerged"] = round(((s == 0) & (e > 0)).mean() * 100, 2)
            rec["pct_progressed"] = round((e > s).mean() * 100, 2)
            rec["pct_regressed"] = round((e < s).mean() * 100, 2)
        out.append(rec)
    rep = pd.DataFrame(out)
    if "pct_nonzero_end" in rep.columns:
        rep = rep.sort_values(["pct_nonzero_end", "n_groups_nonzero_end"]
                              if "n_groups_nonzero_end" in rep.columns else ["pct_nonzero_end"],
                              ascending=False)
    return rep.reset_index(drop=True)


def summarize(rep, meta, families, group_col, min_groups):
    """What is actually present in THIS population -- printed before the table."""
    cnt = rep["n_nonzero_end"] if "n_nonzero_end" in rep.columns else rep.get("pct_nonzero_end", 0)
    nonzero = rep[cnt > 0]
    empty = rep[cnt == 0]
    has_groups = group_col and "n_groups_nonzero_end" in rep.columns
    trace = nonzero[nonzero["n_groups_nonzero_end"] < min_groups] if has_groups else nonzero.iloc[0:0]
    solid = nonzero.drop(trace.index)

    print(f"\n{len(solid)} of {len(rep)} distress series carry data at >= {min_groups} groups; "
          f"{len(trace)} are trace (< {min_groups} groups); {len(empty)} are all-zero.")

    if not empty.empty:
        by_measure = empty.groupby("measure").size().to_dict()
        print("  all-zero by measure: "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_measure.items())))
        dead = sorted(set(empty["code"]) - set(nonzero["code"]))
        if dead:
            print(f"  codes with no data at all ({len(dead)}): {', '.join(dead)}")

    if not trace.empty:
        nets = set()
        for g in trace["groups_if_trace"]:
            nets.update(x for x in g.split("; ") if x)
        print(f"\n  trace series -- present at fewer than {min_groups} groups, "
              f"likely cross-type coding or one network's export quirk:")
        for _, r in trace.head(12).iterrows():
            print(f"    {r['code']}{r['sev']:<2} {r['measure']:<7} "
                  f"{r['n_groups_nonzero_end']} group(s): {r['groups_if_trace']}")
        if len(trace) > 12:
            print(f"    ... {len(trace) - 12} more in the prevalence CSV")
        if nets:
            print(f"  -> {len(nets)} network(s) account for all trace series: "
                  + ", ".join(sorted(nets)[:8]) + (" ..." if len(nets) > 8 else ""))
            print("     check their surface-type assignment in the stratification step")

    ded = solid[(solid["measure"] == "deduct") & (solid["n_distinct_nonzero_end"] <= 3)
                & (solid["n_nonzero_end"] >= 30)]        # rare series look step-like by accident
    if not ded.empty:
        names = ", ".join(f"{r['code']}{r['sev']}" for _, r in ded.iterrows())
        print(f"\n  step-function deducts (<= 3 distinct values -- severity-only, no quantity "
              f"information; use the raw measure for magnitude): {names}")

    live_codes = set(solid["code"])
    viable, dead_fams = [], []
    for fam, codes in families.items():
        hit = sorted(set(codes) & live_codes)
        (viable if hit else dead_fams).append((fam, hit))
    if viable:
        print("\n  viable families here:")
        for fam, hit in sorted(viable, key=lambda x: -len(x[1])):
            print(f"    {fam:<22} codes {', '.join(hit)}")
    if dead_fams:
        print(f"  families with no data here (do not model): "
              + ", ".join(f for f, _ in dead_fams))

    if has_groups and not solid.empty:
        print(f"\n  between-group units available: max {int(solid['n_groups_nonzero_end'].max())} "
              f"{group_col!r} values for the most common distress")
    return solid, trace, empty


def derive(df, meta, families, group_col=None, min_groups=1):
    new = {}
    ded = meta[meta["measure"] == "deduct"]
    raw = meta[meta["measure"].isin(DENOM)]

    # -- deduct totals per code (pools severities and the 42 / 42L export duplicates)
    for (code, when), g in ded.groupby(["code", "when"]):
        new[f"{code} deduct_total_{when}"] = df[g["col"].tolist()].sum(axis=1)

    # -- densities per (code, sev) and totals / severity index per code
    dens_by_code = {}
    for _, r in raw.iterrows():
        den_col = f"{DENOM[r['measure']]}_{r['when']}"
        if den_col not in df.columns:
            continue
        den = pd.to_numeric(df[den_col], errors="coerce")
        val = pd.to_numeric(df[r["col"]], errors="coerce")
        dens = np.where(den > 0, val / den, np.nan)
        name = f"{r['code']}{r['sev']} density_{r['when']}"
        new[name] = dens
        dens_by_code.setdefault((r["code"], r["when"]), []).append((r["sev"], name))

    for (code, when), items in dens_by_code.items():
        block = pd.DataFrame({n: new[n] for _, n in items})
        new[f"{code} density_total_{when}"] = block.sum(axis=1, min_count=1)
        weights = np.array([SEV_WEIGHT[s] for s, _ in items])
        new[f"{code} sevindex_{when}"] = (block * weights).sum(axis=1, min_count=1)

    # -- family sums
    codes_present = set(meta["code"])
    for fam, codes in families.items():
        codes = [c for c in codes if c in codes_present]
        if not codes:
            continue
        for when in ("start", "end"):
            d_cols = [f"{c} deduct_total_{when}" for c in codes if f"{c} deduct_total_{when}" in new]
            if d_cols:
                new[f"fam_{fam} deduct_{when}"] = pd.DataFrame({c: new[c] for c in d_cols}).sum(axis=1)
            n_cols = [f"{c} density_total_{when}" for c in codes if f"{c} density_total_{when}" in new]
            if n_cols:
                new[f"fam_{fam} density_{when}"] = pd.DataFrame({c: new[c] for c in n_cols}).sum(axis=1, min_count=1)

    # -- deltas and indicators
    for name in list(new):
        if name.endswith("_end") and (" deduct_total_" in name or " density_total_" in name
                                      or name.startswith("fam_")):
            start = name[:-4] + "_start"
            if start in new:
                new[name[:-4] + "_delta"] = pd.Series(new[name]) - pd.Series(new[start])

    for code in sorted(codes_present):
        s, e = f"{code} deduct_total_start", f"{code} deduct_total_end"
        if s in new and e in new:
            new[f"{code} present_end"] = (new[e] > 0).astype(int)
            new[f"{code} emerged"] = ((new[s] == 0) & (new[e] > 0)).astype(int)
            new[f"{code} regressed"] = (new[e] < new[s]).astype(int)

    block = pd.DataFrame(new, index=df.index)
    keep = []
    for c in block.columns:
        v = pd.to_numeric(block[c], errors="coerce").fillna(0)
        nz = v != 0
        if not nz.any():
            continue
        if group_col and min_groups > 1 and df.loc[nz, group_col].nunique() < min_groups:
            continue
        keep.append(c)
    dropped = len(block.columns) - len(keep)
    if dropped:
        print(f"  dropped {dropped} derived columns that are all-zero or present at < {min_groups} groups")
    return pd.concat([df, block[keep]], axis=1), dropped


def main():
    ap = argparse.ArgumentParser(description="Sparsity report and derived targets for PAVER distress columns.")
    ap.add_argument("csv")
    ap.add_argument("--out", default=None,
                    help="output stem; writes <out>.csv and <out>_prevalence.csv")
    ap.add_argument("--report-only", action="store_true", help="print the prevalence table, write nothing")
    ap.add_argument("--families", default=None, help="JSON {family: [codes]} to replace the built-in mapping")
    ap.add_argument("--group-col", default=None, help="airfield/network column for the n_groups count")
    ap.add_argument("--keep-nan", action="store_true", help="do not zero-fill absent distresses")
    ap.add_argument("--top", type=int, default=25, help="rows of the prevalence table to print")
    ap.add_argument("--measure", default=None,
                    choices=["deduct", "Slabs", "SqFt", "Ft", "Count"],
                    help="show only this measure in the printed table")
    ap.add_argument("--show-empty", action="store_true",
                    help="include all-zero series in the printed table")
    ap.add_argument("--min-groups", type=int, default=5,
                    help="a series/family counts as present only if seen at >= this many groups "
                         "(default 5); series below it are listed as trace and hidden from the table")
    ap.add_argument("--show-trace", action="store_true",
                    help="include trace series in the printed table")
    ap.add_argument("--sheet", default=None, help="worksheet name or index for Excel input (default: first)")
    ap.add_argument("--list-sheets", action="store_true", help="print worksheet names and exit")
    args = ap.parse_args()

    check_cli_paths(inputs=[args.csv, args.families],
                    outputs=[args.out + ".csv", args.out + "_prevalence.csv"] if args.out else [])

    if not os.path.exists(args.csv):
        sys.exit(f"input not found: {args.csv}")

    if args.list_sheets:
        if not is_excel(args.csv):
            sys.exit("--list-sheets only applies to Excel input")
        excel_engine(args.csv)
        for i, name in enumerate(list_sheets(args.csv)):
            print(f"  [{i}] {name}")
        return

    if not args.report_only and not args.out:
        ap.error("--out is required unless --report-only")
    if args.out:
        args.out = os.path.expanduser(args.out)

    sheet = args.sheet
    if sheet is not None and sheet.isdigit():
        sheet = int(sheet)

    # fail on a bad sheet or a column-pattern mismatch before the slow full read
    parse_columns(read_header(args.csv, sheet))

    df = load_table(args.csv, sheet)
    df.columns = [str(c) for c in df.columns]
    meta = parse_columns(df.columns)
    if not args.keep_nan:
        df[meta["col"].tolist()] = df[meta["col"].tolist()].fillna(0)
    group_col = pick_group_col(df, args.group_col)

    families = FAMILIES
    if args.families:
        with open(args.families) as fh:
            families = json.load(fh)

    n_codes = meta["code"].nunique()
    print(f"{len(df):,} rows; {len(meta)} distress columns over {n_codes} codes; "
          f"group column: {group_col or '(none found -- pass --group-col)'}")

    rep = prevalence(df, meta, group_col)
    solid, trace, empty = summarize(rep, meta, families, group_col, args.min_groups)

    if args.show_empty:
        view = rep
    elif args.show_trace:
        view = pd.concat([solid, trace])
    else:
        view = solid
    if args.measure:
        view = view[view["measure"] == args.measure]
    view = view.drop(columns=["groups_if_trace"], errors="ignore")

    label = args.measure or "all measures"
    hidden = "" if args.show_empty else (" (all-zero hidden)" if args.show_trace
                                          else " (trace and all-zero hidden)")
    print(f"\n  {label}, by prevalence{hidden}:")
    with pd.option_context("display.width", 170, "display.max_columns", 20):
        print(view.head(args.top).to_string(index=False))
    if len(view) > args.top:
        print(f"... {len(view) - args.top} more rows"
              + (f" (full table written to <out>_prevalence.csv)" if args.out else ""))

    if args.report_only:
        return

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    full, _ = derive(df, meta, families, group_col, args.min_groups)
    full.to_csv(f"{args.out}.csv", index=False)
    rep.to_csv(f"{args.out}_prevalence.csv", index=False)
    added = len(full.columns) - len(df.columns)
    out_csv = os.path.abspath(f"{args.out}.csv")
    print(f"\nwrote {out_csv}  (+{added} derived columns)")
    print(f"wrote {os.path.abspath(args.out + '_prevalence.csv')}")
    print("\nThe full local workflow continues with PCI calculations and modeling preparation.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:      # output piped into head/less
        try:
            sys.stdout.close()
        finally:
            os._exit(0)
