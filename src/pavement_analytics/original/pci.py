#!/usr/bin/env python3
"""
calc_pci.py -- recompute PCI from per-distress deduct values, and split the PCI
loss across distress families.

Implements the maximum-CDV procedure using a privately supplied, reviewed
CDV curve profile. No reference curve table is distributed in this package:

    1. drop deducts <= 0
    2. if 0 or 1 deducts exceed the q-threshold, CDV = sum of all deducts
    3. else  m = 1 + (9/D)(100 - HDV), capped at 10; keep the m largest deducts,
       the (floor(m)+1)-th scaled by the fractional part of m
    4. iterate: TDV = sum, q = count > threshold, CDV = curve(q, TDV);
       reduce the smallest deduct above the threshold to the threshold; repeat to q = 1
    5. PCI = 100 - max CDV

The preserved defaults select D5340 curves and D6433 counting rules (q counts
deducts above 2, D = 98). These are compatibility settings, not validation of
a standards implementation or a claim of agreement with operational records. The D5340 9.6.2 multi-severity correction (a distress at several severities
is also evaluated with the severities combined upward; the highest PCI wins) is
applied when --deduct-curves points at a deduct_curves.json from
fit_deduct_curves.py, since it needs density -> deduct curves. --sev-mode picks
how several multi-severity codes combine: greedy per code, or joint over all.

Per row and per _start/_end, writes:
    pci_calc, pci_diff (calc - recorded), tdv, hdv, n_deduct, q0, m, maxcdv, surface_used
and per family F:
    pci_only_<F>        PCI if only F's distresses existed  (same construct as SCI / FOD Index)
    loss_loo_<F>        PCI points recovered if F were removed, others held  (not additive)
    loss_shap_<F>       Shapley share of the total PCI loss  (additive: sums to 100 - pci_calc)
plus _delta columns for the Shapley shares.

Usage:
    python -m pavement_analytics.original.pci data/PCC_targets.csv --out data/PCC_pci
    python -m pavement_analytics.original.pci PCC_pairs.xlsx --sheet Pairs --out data/PCC_pci
    python -m pavement_analytics.original.pci data/PCC_targets.csv --check-only          # agreement stats, no file
    python -m pavement_analytics.original.pci ... --families my_families.json             # {family: [codes]}
    python -m pavement_analytics.original.pci ... --family-compare sci=SCI_end            # pci_only_sci vs a column
    python -m pavement_analytics.original.pci ... --deduct-curves deduct_curves.json          # enable 9.6.2
    python -m pavement_analytics.original.pci ... --deduct-curves deduct_curves.json --sev-mode joint
    python -m pavement_analytics.original.pci ... --standard D6433 --x-correction

Codes not covered by the family map fall into 'other_unmapped' so the
decomposition is always complete. Surface (asphalt/concrete curves) is
inferred per row from which distress codes carry deducts; --surface overrides.
"""

import argparse
import bisect
import itertools
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

from . import check_cli_paths

DEDUCT_RE = re.compile(r"^(?P<code>\d{2})(?P<sev>[LMH]?) deduct_(?P<when>start|end)$")

FAMILIES = {
    "pcc_freeze_thaw":  ["64", "70", "74", "75"],
    "pcc_thermal":      ["61", "63", "65"],
    "pcc_moisture":     ["69", "71", "76"],
    "pcc_load":         ["62", "72"],
    "pcc_other":        ["66", "67", "68", "73"],
    "ac_thermal_crack": ["43", "47", "48"],
    "ac_aging":         ["52", "57"],
    "ac_high_temp":     ["42", "53", "54"],
    "ac_load":          ["41", "45"],
    "ac_subgrade":      ["44", "56"],
    "ac_other":         ["46", "49", "50", "51", "55"],
}
UNMAPPED = "other_unmapped"


# ---------------------------------------------------------------------------
# CDV curves
# ---------------------------------------------------------------------------

class CDVCurves:
    """Piecewise-cubic CDV(TDV, q) for one standard, both surfaces."""

    def __init__(self, path, standard="D5340", x_correction=False, rules=None):
        """standard picks the curves; rules (default = standard) picks the q threshold
        and m divisor. Curve and counting-rule choices remain separately configurable."""
        with open(path) as fh:
            j = json.load(fh)
        if standard not in j["curves"]:
            sys.exit(f"{path} has no curves for {standard}")
        self.standard = standard
        self.rules = rules or standard
        self.q_threshold = j["rules"][self.rules]["q_threshold"]
        self.m_divisor = j["rules"][self.rules]["m_divisor"]
        self.scale = j["q1_scale_error_s"][standard] if x_correction else {}
        self.pieces = {}      # surface -> q -> (lo_list, pieces)
        self.drawn = {}       # surface -> sorted drawn q (excluding 1)
        for surf, byq in j["curves"][standard].items():
            self.pieces[surf] = {}
            for q, pcs in byq.items():
                q = int(q)
                if q == 1:
                    continue          # identity by rule; the drawn line is a calibration check
                pcs = sorted(pcs, key=lambda p: p[0])
                self.pieces[surf][q] = ([p[0] for p in pcs], pcs)
            self.drawn[surf] = sorted(self.pieces[surf])

    def _curve(self, surf, q, tdv):
        """Evaluate a drawn curve; below its domain fall back to the next-lower drawn q."""
        x = tdv / (1.0 + self.scale.get(surf, 0.0)) if self.scale else tdv
        los, pcs = self.pieces[surf][q]
        if x < los[0]:
            lower = [d for d in self.drawn[surf] if d < q]
            return self._curve(surf, lower[-1], tdv) if lower else tdv
        last_hi = pcs[-1][1]
        if x >= last_hi:
            x = last_hi                              # hold the end value
        i = bisect.bisect_right(los, x) - 1
        lo, hi, c0, c1, c2, c3 = pcs[i]
        u = x - lo
        return c0 + c1 * u + c2 * u * u + c3 * u * u * u

    def cdv(self, surf, q, tdv):
        if tdv <= 0:
            return 0.0
        if q <= 1:
            return min(tdv, 100.0)
        drawn = self.drawn[surf]
        if q in self.pieces[surf]:
            v = self._curve(surf, q, tdv)
        elif q > drawn[-1]:
            v = self._curve(surf, drawn[-1], tdv)
        else:                                        # undrawn q: interpolate in q
            lo = max(d for d in drawn if d < q)
            hi = min(d for d in drawn if d > q)
            a, b = self._curve(surf, lo, tdv), self._curve(surf, hi, tdv)
            v = a + (b - a) * (q - lo) / (hi - lo)
        return min(max(v, 0.0), 100.0)


QTY_RE = re.compile(r"^(?P<code>\d{2})(?P<sev>[LMH]?) \((?P<unit>Slabs|SqFt|Ft)\)_(?P<when>start|end)$")
DENOM = {"Slabs": "InspectedSlabs", "SqFt": "InspectedArea (SqFt)", "Ft": "InspectedArea (SqFt)"}
SEV_ORDER = {"": 0, "L": 0, "M": 1, "H": 2}


class DeductCurves:
    """Fitted density -> deduct curves from fit_deduct_curves.py."""

    def __init__(self, path):
        with open(path) as fh:
            j = json.load(fh)
        self.curves = {}
        for surf, byname in j["curves"].items():
            for name, fit in byname.items():
                kx = np.log10([k[0] for k in fit["knots"]])
                ky = np.array([k[1] for k in fit["knots"]])
                # slope of the upper part of the curve (deduct per decade of density), for
                # combined densities beyond anything a single severity was observed at
                top = max(2, len(kx) // 4)
                if len(kx) >= 2 and kx[-1] > kx[-top]:
                    slope = np.polyfit(kx[-top:], ky[-top:], 1)[0]
                else:
                    slope = 0.0
                self.curves[(surf, name)] = (kx, ky, fit["density_range"][1], max(slope, 0.0))
        self.extrapolated = 0

    def has(self, surf, code, sev):
        return (surf, f"{code}{sev}") in self.curves

    def deduct(self, surf, code, sev, density):
        kx, ky, dmax, slope = self.curves[(surf, f"{code}{sev}")]
        x = math.log10(max(density, 1e-6))
        if x > kx[-1]:
            self.extrapolated += 1
            return float(min(100.0, ky[-1] + slope * (x - kx[-1])))
        return float(np.interp(x, kx, ky, left=ky[0]))


def severity_alternatives(entries, surf, dcurves):
    """entries: [(sev, density, deduct)] for one code, 2-3 distinct severities.
    Returns the D5340 9.6.2 candidate configurations, each a list of (sev, deduct).
    First candidate is always 'as recorded'."""
    by = {}
    for sev, dens, ded, _ in entries:
        lvl = "L" if sev == "" else sev
        d0, x0 = by.get(lvl, (0.0, 0.0))
        by[lvl] = (d0 + dens, x0 + ded)          # '42' and '42L' fold together
    levels = sorted(by, key=lambda k: SEV_ORDER[k])
    if len(levels) < 2:
        return None
    code = entries[0][3]
    def look(lvl, dens):
        return dcurves.deduct(surf, code, lvl, dens)
    recorded = [(l, by[l][1]) for l in levels]
    cands = [recorded]
    if len(levels) == 2:
        lo, hi = levels
        if dcurves.has(surf, code, hi):
            cands.append([(hi, look(hi, by[lo][0] + by[hi][0]))])
    else:
        L, M, H = levels
        dL, dM, dH = by[L][0], by[M][0], by[H][0]
        if dcurves.has(surf, code, M):
            cands.append([(M, look(M, dL + dM)), (H, by[H][1])])            # (0, L+M, H)
        if dcurves.has(surf, code, H):
            cands.append([(L, by[L][1]), (H, look(H, dM + dH))])            # (L, 0, M+H)
            cands.append([(M, by[M][1]), (H, look(H, dL + dH))])            # (0, M, L+H)
            cands.append([(H, look(H, dL + dM + dH))])                      # (0, 0, L+M+H)
    return cands


def apply_severity_correction(present, surf, curves, dcurves, mode="sequential", joint_cap=512):
    """present: list of (code, sev, fam, deduct, density). Returns (new_present, gain, n_codes_corrected).
    Chooses, per D5340 9.6.2, the severity combination that yields the highest PCI."""
    bycode = {}
    for code, sev, fam, ded, dens in present:
        bycode.setdefault(code, []).append((sev, dens, ded, code, fam))
    multi = {c: v for c, v in bycode.items() if len({("L" if e[0] == "" else e[0]) for e in v}) >= 2}
    if not multi:
        return present, 0.0, 0
    fixed = [(c, s, f, d) for c, s, f, d, _ in present if c not in multi]
    options = {}
    for c, entries in multi.items():
        cands = severity_alternatives([(s, dn, dd, cd) for s, dn, dd, cd, _ in entries], surf, dcurves)
        if cands and len(cands) > 1:
            options[c] = (entries[0][4], cands)                # (family, candidates)
    if not options:
        base = fixed + [(c, s, f, d) for c, s, f, d, _ in present if c in multi]
        return [(c, s, f, d, None) for c, s, f, d in base], 0.0, 0

    def pci_of(choice):
        ded = [d for _, _, _, d in fixed]
        for c, (fam, cands) in options.items():
            ded += [d for _, d in cands[choice[c]]]
        for c, entries in multi.items():
            if c not in options:
                ded += [e[2] for e in entries]
        return pci_from(ded, surf, curves)

    codes = list(options)
    choice = {c: 0 for c in codes}
    base_pci = pci_of(choice)
    n_combos = 1
    for c in codes:
        n_combos *= len(options[c][1])
    if mode == "joint" and n_combos <= joint_cap:
        best, best_choice = base_pci, dict(choice)
        for combo in itertools.product(*[range(len(options[c][1])) for c in codes]):
            ch = dict(zip(codes, combo))
            v = pci_of(ch)
            if v > best + 1e-9:
                best, best_choice = v, ch
        choice = best_choice
    else:                                              # sequential (greedy), largest deduct mass first
        for c in sorted(codes, key=lambda c: -sum(d for _, d in options[c][1][0])):
            best, best_i = pci_of(choice), choice[c]
            for i in range(len(options[c][1])):
                if i == choice[c]:
                    continue
                trial = dict(choice); trial[c] = i
                v = pci_of(trial)
                if v > best + 1e-9:
                    best, best_i = v, i
            choice[c] = best_i
    final_pci = pci_of(choice)
    out = [(c, s, f, d, None) for c, s, f, d in fixed]
    for c, (fam, cands) in options.items():
        out += [(c, s, fam, d, None) for s, d in cands[choice[c]]]
    for c, entries in multi.items():
        if c not in options:
            out += [(e[3], e[0], e[4], e[2], None) for e in entries]
    n_changed = sum(1 for c in codes if choice[c] != 0)
    return out, final_pci - base_pci, n_changed


def max_cdv(deducts, surf, curves):
    """ASTM maximum corrected deduct value. Returns (maxcdv, hdv, m, q0, n)."""
    d = sorted((float(x) for x in deducts if x is not None and x > 0), reverse=True)
    if not d:
        return 0.0, 0.0, 0.0, 0, 0
    thr = curves.q_threshold
    q0 = sum(1 for x in d if x > thr)
    if q0 <= 1:
        return min(sum(d), 100.0), d[0], float("nan"), q0, len(d)
    hdv = d[0]
    m = min(1.0 + (9.0 / curves.m_divisor) * (100.0 - hdv), 10.0)
    k, frac = int(math.floor(m)), m - math.floor(m)
    if len(d) > k:
        cur = d[:k] + ([d[k] * frac] if frac > 1e-9 else [])
    else:
        cur = d[:]
    best = 0.0
    while True:
        tdv = sum(cur)
        q = sum(1 for x in cur if x > thr)
        best = max(best, curves.cdv(surf, q, tdv))
        if q <= 1:
            break
        idx = max(i for i, x in enumerate(cur) if x > thr)   # smallest deduct above threshold
        cur[idx] = thr
    return best, hdv, m, q0, len(d)


def pci_from(deducts, surf, curves):
    return max(0.0, 100.0 - max_cdv(deducts, surf, curves)[0])


# ---------------------------------------------------------------------------
# family decomposition (Shapley over families present in the row)
# ---------------------------------------------------------------------------

def decompose(fam_deducts, surf, curves):
    """fam_deducts: {family: [deducts]} for families with any deduct > 0.
    Returns (pci_calc, {fam: (pci_only, loss_loo, loss_shap)})."""
    fams = sorted(fam_deducts)
    n = len(fams)
    memo = {}

    def v(mask):                      # PCI loss for the coalition in mask
        if mask in memo:
            return memo[mask]
        ded = [x for i, f in enumerate(fams) if mask >> i & 1 for x in fam_deducts[f]]
        val = 100.0 - pci_from(ded, surf, curves)
        memo[mask] = val
        return val

    full = (1 << n) - 1
    total_loss = v(full)
    fact = [math.factorial(i) for i in range(n + 1)]
    out = {}
    for i, f in enumerate(fams):
        bit = 1 << i
        shap = 0.0
        for mask in range(1 << n):
            if mask & bit:
                continue
            s = bin(mask).count("1")
            w = fact[s] * fact[n - s - 1] / fact[n]
            shap += w * (v(mask | bit) - v(mask))
        only = 100.0 - v(bit)
        loo = total_loss - v(full & ~bit)
        out[f] = (only, loo, shap)
    return 100.0 - total_loss, out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_table(path, sheet=None):
    identity_types = {c: "string" for c in ("SectionUniqueID", "_source_file_start", "_source_file_end",
                                           "partition_airfield", "partition_section")}
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xlsm", ".xls"}:
        print(f"reading {os.path.basename(path)} (Excel is slow for wide files) ...")
        return pd.read_excel(path, sheet_name=sheet or 0, dtype=identity_types)
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc, dtype=identity_types)
        except UnicodeDecodeError:
            continue
    sys.exit(f"could not decode {path}")


def infer_surface(codes_with_mass):
    """asphalt if deduct mass sits in 41-57, concrete if in 61-76."""
    a = sum(v for c, v in codes_with_mass if 41 <= int(c) <= 57)
    p = sum(v for c, v in codes_with_mass if 61 <= int(c) <= 76)
    if a == 0 and p == 0:
        return None, False
    return ("asphalt" if a >= p else "concrete"), (a > 0 and p > 0)


def agreement(calc, rec, label):
    ok = calc.notna() & rec.notna()
    if ok.sum() == 0:
        print(f"  {label}: no recorded PCI to compare against")
        return
    d = (calc[ok] - rec[ok])
    ad = d.abs()
    print(f"  {label}: n={ok.sum():,}  bias={d.mean():+.2f}  MAE={ad.mean():.2f}  "
          f"median|d|={ad.median():.2f}  within 1/2/5 pts: "
          f"{(ad <= 1).mean() * 100:.0f}% / {(ad <= 2).mean() * 100:.0f}% / {(ad <= 5).mean() * 100:.0f}%"
          f"  worst={ad.max():.1f}")


def _mae(calc, rec):
    ok = ~np.isnan(calc) & ~np.isnan(rec)
    d = calc[ok] - rec[ok]
    return d.mean(), np.abs(d).mean()


def diagnose(df, new, kept, cols, args):
    print("\nDiagnosis of calc - recorded:")
    for when in cols:
        if f"PCI_{when}" not in df.columns:
            continue
        rec = pd.to_numeric(df[f"PCI_{when}"], errors="coerce").to_numpy(dtype=float)
        calc = new[f"pci_calc_{when}"]
        tdv, nded = new[f"tdv_{when}"], new[f"n_deduct_{when}"]
        ok = ~np.isnan(calc) & ~np.isnan(rec)
        d = calc - rec
        if not ok.any():
            print(f"  [{when}]: no finite PCI comparisons available")
            continue
        print(f"\n  [{when}]  quantiles of (calc - recorded):")
        qs = np.percentile(d[ok], [1, 5, 25, 50, 75, 95, 99])
        print("     p1 %.1f  p5 %.1f  p25 %.1f  p50 %.1f  p75 %.1f  p95 %.1f  p99 %.1f" % tuple(qs))

        a = ok & (rec >= 99.5) & (tdv > 0)
        b = ok & (tdv == 0) & (rec < 99.5)
        print(f"     recorded PCI=100 but deducts present : {a.sum():6,} rows  (data inconsistency; mean calc {calc[a].mean() if a.any() else float('nan'):.1f})")
        print(f"     no deducts but recorded PCI < 100     : {b.sum():6,} rows  (distress missing from export; mean recorded {rec[b].mean() if b.any() else float('nan'):.1f})")

        print("     by number of distresses present:")
        print("        n_deduct   rows    bias     MAE   within2")
        for lo, hi, lab in [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 7, "5-7"), (8, 99, "8+")]:
            sel = ok & (nded >= lo) & (nded <= hi)
            if sel.any():
                dd = d[sel]
                print(f"        {lab:<9} {sel.sum():6,}  {dd.mean():+6.2f}  {np.abs(dd).mean():6.2f}   {(np.abs(dd) <= 2).mean() * 100:5.0f}%")

        area_col = next((c for c in (f"InspectedArea (SqFt)_{when}", f"InspectedSlabs_{when}") if c in df.columns), None)
        if area_col:
            area = pd.to_numeric(df[area_col], errors="coerce").to_numpy(dtype=float)
            sel = ok & ~np.isnan(area) & (nded >= 2)
            if sel.sum() > 50:
                qcut = np.percentile(area[sel], [25, 50, 75])
                print(f"     by section size ({area_col}), sections with 2+ distresses:")
                edges = [-np.inf] + list(qcut) + [np.inf]
                for j in range(4):
                    s2 = sel & (area > edges[j]) & (area <= edges[j + 1])
                    if s2.any():
                        print(f"        quartile {j + 1}:  rows {s2.sum():6,}   bias {d[s2].mean():+6.2f}   MAE {np.abs(d[s2]).mean():6.2f}")
                print("        (|error| growing with size = more sample units = more room for spatial concentration)")

        # alternative rule sets on the same deducts
        base_bias, base_mae = _mae(calc, rec)
        print(f"     alternative rule sets (baseline bias {base_bias:+.2f}, MAE {base_mae:.2f}):")
        variants = []
        for std, rules in (("D5340", "D5340"), ("D5340", "D6433"), ("D6433", "D6433")):
            if std == args.standard and rules == args.rules:
                continue                                   # that is the baseline
            variants.append((f"{std} curves, {rules} rules", CDVCurves(args.curves, std, rules=rules), True))
        variants.append((f"{args.standard} curves with q=1 x-correction",
                         CDVCurves(args.curves, args.standard, x_correction=True, rules=args.rules), True))
        for label, cv, use_m in variants:
            alt = np.full(len(calc), np.nan)
            for i, item in enumerate(kept[when]):
                if item is None:
                    alt[i] = 100.0 if not np.isnan(calc[i]) else np.nan
                    continue
                surf, ded, _ = item
                alt[i] = 100.0 - max_cdv(ded, surf, cv)[0]
            bb, mm = _mae(alt, rec)
            print(f"        {label:<42} bias {bb:+6.2f}   MAE {mm:6.2f}")
        # no-truncation variant: monkeypatch m by feeding at most 10 deducts is the standard; here keep all
        alt = np.full(len(calc), np.nan)
        cv = CDVCurves(args.curves, args.standard, rules=args.rules)
        for i, item in enumerate(kept[when]):
            if item is None:
                alt[i] = 100.0 if not np.isnan(calc[i]) else np.nan
                continue
            surf, ded, _ = item
            dd = sorted((x for x in ded if x > 0), reverse=True)
            if not dd:
                alt[i] = 100.0; continue
            if sum(1 for x in dd if x > cv.q_threshold) <= 1:
                alt[i] = 100.0 - min(sum(dd), 100.0); continue
            cur, best = dd[:], 0.0
            while True:
                q = sum(1 for x in cur if x > cv.q_threshold)
                best = max(best, cv.cdv(surf, q, sum(cur)))
                if q <= 1:
                    break
                cur[max(k for k, x in enumerate(cur) if x > cv.q_threshold)] = cv.q_threshold
            alt[i] = 100.0 - best
        bb, mm = _mae(alt, rec)
        print(f"        {'baseline without the m-truncation':<42} bias {bb:+6.2f}   MAE {mm:6.2f}")
        print("     (a variant that beats the baseline is a rule-set mismatch; rerun with --rules / --standard to adopt it)")

        # Examine residuals by whether a code has multiple severities.
        multi = np.array([bool(item) and len(item[2]) != len(set(item[2])) for item in kept[when]])
        for label, sel in (("same code at 2+ severities", ok & multi & (nded >= 2)),
                           ("all codes at one severity ", ok & ~multi & (nded >= 2))):
            if sel.any():
                print(f"     {label}: rows {sel.sum():6,}   bias {d[sel].mean():+6.2f}   MAE {np.abs(d[sel]).mean():6.2f}")
        print("     (multi-severity residuals warrant checking the configured severity-correction curves and mode)")


def parity_plot(df, new, cols, path, title_note):
    """Parity (calc vs recorded) and residual (calc - recorded vs recorded) per _start/_end,
    coloured by number of distresses; export-gap rows marked separately."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except ImportError:
        print("  --plot needs matplotlib:  pip install matplotlib   (skipped)")
        return
    whens = [w for w in ("start", "end") if f"pci_diff_{w}" in new]
    if not whens:
        print("  --plot: no recorded PCI column to compare against (skipped)")
        return
    fig, axes = plt.subplots(len(whens), 2, figsize=(12, 5.2 * len(whens)), squeeze=False)
    bins = [0, 1, 3, 5, 8, 999]
    labels = ["0", "1-2", "3-4", "5-7", "8+"]
    cmap = ListedColormap(["#b8c2c8", "#3d7a93", "#2b3a42", "#b5551f", "#7a1f1f"])
    norm = BoundaryNorm(bins, cmap.N)
    for r, when in enumerate(whens):
        rec = pd.to_numeric(df[f"PCI_{when}"], errors="coerce").to_numpy(float)
        calc = new[f"pci_calc_{when}"]
        nded = new[f"n_deduct_{when}"]
        gap = new.get(f"export_gap_{when}", np.zeros(len(rec))).astype(bool)
        ok = np.isfinite(rec) & np.isfinite(calc) & ~gap
        d = calc - rec
        ax = axes[r, 0]
        sc = ax.scatter(rec[ok], calc[ok], c=nded[ok], cmap=cmap, norm=norm, s=5, alpha=0.35, linewidths=0)
        if gap.any():
            ax.scatter(rec[gap], calc[gap], marker="x", s=18, color="#c00000", linewidths=0.8,
                       label=f"export gap ({gap.sum():,})")
        ax.plot([0, 100], [0, 100], color="black", lw=1)
        for off, ls in ((2, ":"), (5, "--")):
            ax.plot([0, 100], [off, 100 + off], color="grey", lw=0.7, ls=ls)
            ax.plot([0, 100], [-off, 100 - off], color="grey", lw=0.7, ls=ls)
        ax.set_xlim(0, 102); ax.set_ylim(0, 102); ax.set_aspect("equal")
        ax.set_xlabel(f"recorded PCI_{when}"); ax.set_ylabel("recomputed PCI")
        ad = np.abs(d[ok])
        ax.set_title(f"{when}: bias {d[ok].mean():+.2f}, MAE {ad.mean():.2f}, "
                     f"within 2 pts {(ad <= 2).mean() * 100:.0f}%  (n={ok.sum():,})", fontsize=10)
        if gap.any():
            ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax = axes[r, 1]
        ax.scatter(rec[ok], d[ok], c=nded[ok], cmap=cmap, norm=norm, s=5, alpha=0.35, linewidths=0)
        ax.axhline(0, color="black", lw=1)
        for off in (2, 5):
            ax.axhline(off, color="grey", lw=0.7, ls=":" if off == 2 else "--")
            ax.axhline(-off, color="grey", lw=0.7, ls=":" if off == 2 else "--")
        # running median of the residual across recorded PCI
        edges = np.arange(0, 101, 5)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = ok & (rec >= lo) & (rec < hi)
            if sel.sum() >= 20:
                xs.append((lo + hi) / 2); ys.append(np.median(d[sel]))
        if xs:
            ax.plot(xs, ys, color="black", lw=1.5, label="running median")
            ax.legend(loc="lower left", fontsize=8, frameon=False)
        lim = max(10, np.percentile(np.abs(d[ok]), 99.5))
        ax.set_ylim(-lim, lim); ax.set_xlim(0, 102)
        ax.set_xlabel(f"recorded PCI_{when}"); ax.set_ylabel("recomputed - recorded")
        ax.set_title("residual by recorded PCI; colour = distresses present", fontsize=10)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, ticks=[0.5, 2, 4, 6.5, 500])
    cbar.ax.set_yticklabels(labels); cbar.set_label("number of distresses")
    fig.suptitle(title_note, fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.abspath(path)}")


def main():
    ap = argparse.ArgumentParser(description="Recompute PCI from deducts; decompose PCI loss by family.")
    ap.add_argument("path")
    ap.add_argument("--out", default=None, help="output stem; writes <out>.csv")
    ap.add_argument("--check-only", action="store_true", help="print agreement stats only")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--curves", required=True, help="Private reviewed CDV profile JSON; no standard tables are bundled")
    ap.add_argument("--standard", default="D5340", choices=["D5340", "D6433"],
                    help="which CDV curves to use")
    ap.add_argument("--rules", default="D6433", choices=["D5340", "D6433"],
                    help="q threshold / m divisor: D5340 = >5, 9/95; D6433 = >2, 9/98. Default D6433: "
                         "the preserved compatibility default is D6433; validate rule choices against the approved reference.")
    ap.add_argument("--x-correction", action="store_true",
                    help="evaluate curves at TDV/(1+s) using the q=1 scale error (see digitization summary 4.1)")
    ap.add_argument("--surface", default="auto", choices=["auto", "asphalt", "concrete"])
    ap.add_argument("--families", default=None, help="JSON {family: [codes]} replacing the built-in map")
    ap.add_argument("--family-compare", action="append", default=[], metavar="FAMILY=COLUMN",
                    help="compare pci_only_<FAMILY>_<when> against COLUMN (e.g. sci=SCI_end)")
    ap.add_argument("--no-shapley", action="store_true", help="skip the Shapley decomposition (faster)")
    ap.add_argument("--deduct-curves", default=None,
                    help="deduct_curves.json from fit_deduct_curves.py; enables the D5340 9.6.2 "
                         "multi-severity correction")
    ap.add_argument("--sev-mode", default="sequential", choices=["sequential", "joint"],
                    help="how 9.6.2 combines across several multi-severity codes: greedy per code, "
                         "or the joint best over all combinations (capped at 512)")
    ap.add_argument("--plot", default=None, metavar="PNG",
                    help="write a parity + residual figure of recomputed vs recorded PCI")
    ap.add_argument("--diagnose", action="store_true",
                    help="explain the disagreement: quantiles, bins by distress count and area, "
                         "inconsistent rows, and alternative rule sets")
    args = ap.parse_args()
    if not args.check_only and not args.out:
        ap.error("--out is required unless --check-only")

    check_cli_paths(inputs=[args.path, args.curves, args.deduct_curves, args.families],
                    outputs=[args.out + ".csv" if args.out else None, args.plot])

    curves = CDVCurves(args.curves, args.standard, args.x_correction, args.rules)
    dcurves = DeductCurves(args.deduct_curves) if args.deduct_curves else None
    families = FAMILIES
    if args.families:
        with open(args.families) as fh:
            families = json.load(fh)
    code2fam = {c: f for f, cs in families.items() for c in cs}

    df = load_table(args.path, args.sheet)
    df.columns = [str(c) for c in df.columns]
    cols, qcols = {}, {}
    for c in df.columns:
        m = DEDUCT_RE.match(c)
        if m:
            cols.setdefault(m["when"], []).append((c, m["code"], code2fam.get(m["code"], UNMAPPED), m["sev"]))
        mq = QTY_RE.match(c)
        if mq:
            qcols[(mq["code"], mq["sev"], mq["when"])] = (c, mq["unit"])
    if not cols:
        sys.exit("no '<code><sev> deduct_<start|end>' columns found")
    fams_all = sorted({f for lst in cols.values() for _, _, f, _ in lst})
    print(f"{len(df):,} rows; deduct columns: " + ", ".join(f"{w}={len(v)}" for w, v in cols.items())
          + f"; families in play: {len(fams_all)}; curves {args.standard}, rules {curves.rules} "
          + f"(q>{curves.q_threshold:g}, m=1+(9/{curves.m_divisor:g})(100-HDV))"
          + (", x-correction on" if args.x_correction else ""))

    new = {}
    kept = {}           # when -> list of (surface, deducts) per row, for --diagnose
    t0 = time.time()
    for when, lst in cols.items():
        names = [c for c, _, _, _ in lst]
        vals = df[names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        n = len(df)
        # densities for the multi-severity correction: quantity / denominator x 100
        dens = None
        if dcurves is not None:
            dens = np.full(vals.shape, np.nan)
            denom_cache = {}
            for j, (_, code, _, sev) in enumerate(lst):
                q = qcols.get((code, sev, when))
                if q is None:
                    continue
                qc, unit = q
                dc = f"{DENOM[unit]}_{when}"
                if dc not in df.columns:
                    continue
                if dc not in denom_cache:
                    denom_cache[dc] = pd.to_numeric(df[dc], errors="coerce").to_numpy(float)
                qv = pd.to_numeric(df[qc], errors="coerce").to_numpy(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    dens[:, j] = np.where(denom_cache[dc] > 0, qv / denom_cache[dc] * 100.0, np.nan)
        corr_gain = np.zeros(n); corr_n = np.zeros(n, dtype=int)
        pci_calc = np.full(n, np.nan); tdv = np.zeros(n); hdv = np.zeros(n)
        nded = np.zeros(n, dtype=int); q0 = np.zeros(n, dtype=int)
        mm = np.full(n, np.nan); mcdv = np.zeros(n); surf_used = np.empty(n, dtype=object)
        only = {f: np.full(n, 100.0) for f in fams_all}
        loo = {f: np.zeros(n) for f in fams_all}
        shap = {f: np.zeros(n) for f in fams_all}
        mixed = 0
        kept[when] = [None] * n

        for i in range(n):
            row = vals[i]
            present = [(lst[j][1], lst[j][3], lst[j][2], row[j],
                        dens[i, j] if dens is not None else None) for j in np.nonzero(row > 0)[0]]
            if args.surface == "auto":
                surf, is_mixed = infer_surface([(c, v) for c, _, _, v, _ in present])
                mixed += is_mixed
            else:
                surf = args.surface
            if surf is None:
                pci_calc[i] = 100.0; surf_used[i] = ""
                continue
            surf_used[i] = surf
            orig_codes = [c for c, _, _, _, _ in present]
            if dcurves is not None and all(d is not None and not np.isnan(d) for _, _, _, _, d in present):
                present, gain, nc = apply_severity_correction(present, surf, curves, dcurves, args.sev_mode)
                corr_gain[i], corr_n[i] = gain, nc
            ded = [v for _, _, _, v, _ in present]
            kept[when][i] = (surf, ded, orig_codes)
            best, h, m_, q_, k = max_cdv(ded, surf, curves)
            pci_calc[i] = max(0.0, 100.0 - best)
            tdv[i], hdv[i], nded[i], q0[i], mm[i], mcdv[i] = sum(ded), h, k, q_, m_, best
            if args.no_shapley:
                continue
            fd = {}
            for _, _, f, v, _ in present:
                fd.setdefault(f, []).append(v)
            _, parts = decompose(fd, surf, curves)
            for f, (o, l, s) in parts.items():
                only[f][i], loo[f][i], shap[f][i] = o, l, s
            if i % 5000 == 0 and i:
                print(f"  {when}: {i:,}/{n:,} rows ({time.time() - t0:.0f}s)")

        rec = pd.to_numeric(df.get(f"PCI_{when}"), errors="coerce") if f"PCI_{when}" in df.columns else None
        new[f"pci_calc_{when}"] = pci_calc
        if rec is not None:
            new[f"pci_diff_{when}"] = pci_calc - rec.to_numpy()
            # recorded PCI well below 100 with no distress rows at all = the distress
            # export is missing for this inspection, not a clean pavement
            new[f"export_gap_{when}"] = ((tdv == 0) & (rec.to_numpy() < 99.5)).astype(int)
        new[f"tdv_{when}"] = tdv; new[f"hdv_{when}"] = hdv; new[f"n_deduct_{when}"] = nded
        new[f"q0_{when}"] = q0; new[f"m_{when}"] = mm; new[f"maxcdv_{when}"] = mcdv
        new[f"surface_used_{when}"] = surf_used
        if dcurves is not None:
            new[f"sev_corr_gain_{when}"] = corr_gain
            new[f"sev_corr_codes_{when}"] = corr_n
            print(f"  {when}: 9.6.2 applied on {(corr_n > 0).sum():,} rows, mean gain {corr_gain[corr_n > 0].mean() if (corr_n > 0).any() else 0:.2f} PCI points")
        for f in fams_all:
            if not args.no_shapley:
                new[f"pci_only_{f}_{when}"] = only[f]
                new[f"loss_loo_{f}_{when}"] = loo[f]
                new[f"loss_shap_{f}_{when}"] = shap[f]
        if mixed:
            print(f"  {when}: {mixed} rows carry deducts from both surface families (surface chosen by deduct mass)")

    if dcurves is not None and dcurves.extrapolated:
        print(f"  note: {dcurves.extrapolated:,} combined-density lookups above the fitted range "
              "(extrapolated log-linearly from the curve's upper slope, capped at 100)")
    print("\nAgreement of recomputed PCI with the recorded PCI column:")
    for when in cols:
        if f"pci_diff_{when}" in new:
            calc = pd.Series(new[f"pci_calc_{when}"])
            rec = pd.to_numeric(df[f"PCI_{when}"], errors="coerce")
            agreement(calc, rec, when)
            gap = new[f"export_gap_{when}"].astype(bool)
            if gap.any():
                agreement(calc[~gap], rec[~gap], f"{when}, excluding {gap.sum()} export-gap rows")
    if args.plot:
        note = (f"{os.path.basename(args.path)} -- {args.standard} curves, {curves.rules} rules"
                + (", 9.6.2 on" if dcurves is not None else ", 9.6.2 off"))
        parity_plot(df, new, cols, args.plot, note)

    if args.diagnose:
        diagnose(df, new, kept, cols, args)

    for spec in args.family_compare:
        fam, col = spec.split("=", 1)
        for when in cols:
            key = f"pci_only_{fam}_{when}"
            if key in new and col in df.columns:
                agreement(pd.Series(new[key]), pd.to_numeric(df[col], errors="coerce"), f"pci_only_{fam} vs {col} ({when})")
            elif key not in new:
                print(f"  no family named {fam!r} in the map")

    if not args.no_shapley:
        print("\nMean share of PCI loss by family (Shapley, rows where the family is present):")
        for when in cols:
            tot = 100.0 - new[f"pci_calc_{when}"]
            rows = []
            for f in fams_all:
                s = new[f"loss_shap_{f}_{when}"]
                present = s != 0
                if present.any():
                    rows.append((f, present.mean() * 100, s[present].mean(),
                                 (s.sum() / tot.sum() * 100) if tot.sum() else float("nan")))
            rows.sort(key=lambda r: -r[3])
            print(f"  [{when}]  family                 present%   mean pts   share of all loss")
            for f, pp, mp, sh in rows:
                print(f"           {f:<22} {pp:6.1f}%   {mp:7.2f}   {sh:6.1f}%")

    if args.check_only:
        return
    both = [w for w in ("start", "end") if w in cols]
    if len(both) == 2 and not args.no_shapley:
        for f in fams_all:
            new[f"loss_shap_{f}_delta"] = new[f"loss_shap_{f}_end"] - new[f"loss_shap_{f}_start"]
        new["pci_calc_delta"] = new["pci_calc_end"] - new["pci_calc_start"]
    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_csv(f"{args.out}.csv", index=False)
    print(f"\nwrote {os.path.abspath(args.out + '.csv')}  (+{len(new)} columns, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        os._exit(0)
