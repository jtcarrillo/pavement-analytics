#!/usr/bin/env python3
"""
fit_deduct_curves.py -- recover the ASTM D5340 deduct curves from the export itself.

Every row of a pairs file carries, per distress code and severity, a quantity
(Slabs / SqFt / Ft), a denominator (InspectedSlabs or InspectedArea) and the
deduct PAVER assigned. Density = quantity / denominator x 100 is the x-axis of
the standard's deduct curve and the deduct is its y-axis, so the export contains
thousands of samples along every curve. This script fits a monotone curve per
(code, severity) and writes deduct_curves.json for calc_pci.py's multi-severity
correction (D5340 9.6.2).

The fit tightness is itself a finding: deducts that sit on a sharp curve are
section-level lookups; a wide band means they are averaged from sample units.

Usage:
    python -m pavement_analytics.original.fit_curves AC_pairs.xlsx APC_pairs.xlsx PCC_pairs.xlsx --out deduct_curves.json
    python -m pavement_analytics.original.fit_curves data/PCC_targets.csv --out deduct_curves_pcc.json
    python -m pavement_analytics.original.fit_curves ... --report-only
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from . import check_cli_paths

DED_RE = re.compile(r"^(?P<code>\d{2})(?P<sev>[LMH]?) deduct_(?P<when>start|end)$")
QTY_RE = re.compile(r"^(?P<code>\d{2})(?P<sev>[LMH]?) \((?P<unit>Slabs|SqFt|Ft)\)_(?P<when>start|end)$")
DENOM = {"Slabs": "InspectedSlabs", "SqFt": "InspectedArea (SqFt)", "Ft": "InspectedArea (SqFt)"}


def load(path, sheet=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xlsm", ".xls"}:
        print(f"reading {os.path.basename(path)} ...")
        df = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, low_memory=False, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            sys.exit(f"could not decode {path}")
    df.columns = [str(c) for c in df.columns]
    return df


def collect_samples(df):
    """{(surface, code, sev): (density array, deduct array, unit)} plus inconsistency counts."""
    ded = {}
    for c in df.columns:
        m = DED_RE.match(c)
        if m:
            ded[(m["code"], m["sev"], m["when"])] = c
    out, bad = {}, {"qty_no_deduct": 0, "deduct_no_qty": 0}
    for c in df.columns:
        m = QTY_RE.match(c)
        if not m:
            continue
        code, sev, unit, when = m["code"], m["sev"], m["unit"], m["when"]
        dcol = ded.get((code, sev, when))
        denom_col = f"{DENOM[unit]}_{when}"
        if dcol is None or denom_col not in df.columns:
            continue
        q = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        d = pd.to_numeric(df[dcol], errors="coerce").to_numpy(float)
        a = pd.to_numeric(df[denom_col], errors="coerce").to_numpy(float)
        ok = np.isfinite(q) & np.isfinite(d) & np.isfinite(a) & (a > 0)
        bad["qty_no_deduct"] += int((ok & (q > 0) & (d <= 0)).sum())
        bad["deduct_no_qty"] += int((ok & (q <= 0) & (d > 0)).sum())
        use = ok & (q > 0) & (d > 0)
        if not use.any():
            continue
        surface = "asphalt" if 41 <= int(code) <= 57 else "concrete"
        key = (surface, code, sev)
        dens = q[use] / a[use] * 100.0
        prev = out.get(key)
        if prev:
            out[key] = (np.concatenate([prev[0], dens]), np.concatenate([prev[1], d[use]]), unit)
        else:
            out[key] = (dens, d[use], unit)
    return out, bad


def pava(y, w):
    """Pool-adjacent-violators: weighted non-decreasing fit."""
    y, w = list(map(float, y)), list(map(float, w))
    blocks = [[y[i], w[i], 1] for i in range(len(y))]   # mean, weight, count
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m0, w0, n0 = blocks[i]; m1, w1, n1 = blocks[i + 1]
            blocks[i] = [(m0 * w0 + m1 * w1) / (w0 + w1), w0 + w1, n0 + n1]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fit = []
    for m, _, n in blocks:
        fit.extend([m] * n)
    return np.array(fit)


def fit_curve(dens, ded, bins=40, min_per_bin=20):
    """Quantile-bin on log10(density), median per bin, isotonic in x. Returns knots + stats."""
    x = np.log10(np.clip(dens, 1e-6, None))
    order = np.argsort(x)
    x, y = x[order], ded[order]
    n = len(x)
    nb = max(1, min(bins, n // min_per_bin))
    edges = np.linspace(0, n, nb + 1).astype(int)
    xs, ys, ws = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        xs.append(np.median(x[a:b])); ys.append(np.median(y[a:b])); ws.append(b - a)
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)
    # merge bins that landed on the same x (many identical densities)
    ux, inv = np.unique(np.round(xs, 6), return_inverse=True)
    if len(ux) < len(xs):
        ys = np.array([np.average(ys[inv == k], weights=ws[inv == k]) for k in range(len(ux))])
        ws = np.array([ws[inv == k].sum() for k in range(len(ux))])
        xs = ux
    yf = pava(ys, ws)
    pred = np.interp(x, xs, yf, left=yf[0], right=yf[-1])
    res = y - pred
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (res ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return {
        "knots": [[float(10 ** a), float(b)] for a, b in zip(xs, yf)],
        "n": int(n),
        "density_range": [float(dens.min()), float(dens.max())],
        "deduct_range": [float(ded.min()), float(ded.max())],
        "rmse": float(np.sqrt((res ** 2).mean())),
        "mad": float(np.median(np.abs(res))),
        "within1_pct": float((np.abs(res) <= 1.0).mean() * 100),
        "r2": float(r2),
        "flat": bool(yf[-1] - yf[0] < 0.5),
    }


def main():
    ap = argparse.ArgumentParser(description="Fit deduct curves from the export's quantity/deduct columns.")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None, help="deduct_curves.json to write")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--min-samples", type=int, default=40, help="skip curves with fewer samples")
    ap.add_argument("--bins", type=int, default=40)
    args = ap.parse_args()
    if not args.report_only and not args.out:
        ap.error("--out is required unless --report-only")

    check_cli_paths(inputs=args.paths, outputs=[args.out])

    samples, bad = {}, {"qty_no_deduct": 0, "deduct_no_qty": 0}
    for p in args.paths:
        s, b = collect_samples(load(p, args.sheet))
        for k, (d, y, u) in s.items():
            if k in samples:
                pd_, py_, pu_ = samples[k]
                samples[k] = (np.concatenate([pd_, d]), np.concatenate([py_, y]), pu_)
            else:
                samples[k] = (d, y, u)
        for k in bad:
            bad[k] += b[k]

    curves, rows = {}, []
    for (surface, code, sev), (dens, ded, unit) in sorted(samples.items()):
        if len(dens) < args.min_samples:
            rows.append((surface, code, sev, unit, len(dens), None))
            continue
        fit = fit_curve(dens, ded, args.bins)
        fit["unit"] = unit
        curves.setdefault(surface, {})[f"{code}{sev}"] = fit
        rows.append((surface, code, sev, unit, len(dens), fit))

    print(f"\n{sum(len(v[0]) for v in samples.values()):,} (density, deduct) samples across "
          f"{len(samples)} code/severity series; {bad['qty_no_deduct']:,} rows with quantity but no deduct, "
          f"{bad['deduct_no_qty']:,} with deduct but no quantity\n")
    print("surface   code sev unit    n     r2   rmse   mad  within1%  density range        deduct range   note")
    for surface, code, sev, unit, n, fit in rows:
        if fit is None:
            print(f"{surface:<9} {code}  {sev or '-':<3} {unit:<5} {n:6,}   (below --min-samples, skipped)")
            continue
        note = "flat (severity-only deduct)" if fit["flat"] else ""
        print(f"{surface:<9} {code}  {sev or '-':<3} {unit:<5} {fit['n']:6,} {fit['r2']:6.3f} {fit['rmse']:6.2f} "
              f"{fit['mad']:5.2f} {fit['within1_pct']:7.1f}%  {fit['density_range'][0]:7.3f}-{fit['density_range'][1]:<7.1f} "
              f"{fit['deduct_range'][0]:5.1f}-{fit['deduct_range'][1]:<5.1f}  {note}")

    good = [f for f in curves.get("asphalt", {}).values()] + [f for f in curves.get("concrete", {}).values()]
    well = [f for f in good if f["n"] >= 500 and not f["flat"]]
    if well:
        w1 = np.median([f["within1_pct"] for f in well])
        print(f"\nAcross the {len(well)} well-sampled non-flat curves, median {w1:.0f}% of samples lie within 1 deduct "
              f"of the fitted curve.")
        if w1 >= 90:
            print("=> The export's deducts are section-level curve lookups. These fitted curves are the deduct curves.")
        elif w1 >= 60:
            print("=> Mostly section-level, with some averaging or rounding. Curves usable; expect residual noise.")
        else:
            print("=> Deducts do not sit on a single curve: they are averaged from sample units (or densities use\n"
                  "   a different denominator). Curves are approximations; the 9.6.2 correction will be too.")

    if args.report_only:
        return
    out = {
        "source": "fitted from export (quantity / denominator x 100 vs recorded deduct), quantile-binned medians with "
                  "isotonic (non-decreasing) smoothing; interpolate linearly in log10(density), hold beyond the ends",
        "inputs": [os.path.basename(p) for p in args.paths],
        "curves": curves,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.abspath(args.out)}: {sum(len(v) for v in curves.values())} curves")


if __name__ == "__main__":
    main()
