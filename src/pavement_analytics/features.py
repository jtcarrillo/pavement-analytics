"""Explicit feature admission: unreviewed columns never become model inputs."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .security import SafeError

IDENTITY = re.compile(r"(?:^_|source|partition|(?:^|[_\s])(?:id|name|group|section|network|branch)(?:$|[_\s])|uniqueid)", re.I)
OUTCOME = re.compile(r"(?:_end$|_delta$|^delta|end_date|intervalend|intervalyears|emerged|regressed|resolved|persistent|^loss_|^pci_only|^pci_calc|^pci_diff|^export_gap|^invalid_deduct|^rate_|^DetRate)", re.I)
CUMULATIVE = re.compile(r"^(?:CoolEvents6h|RainflowCycles10C|RainflowCycles20C|RainflowDamage|PGWindowYears)(?:_start)?$", re.I)
GEO = re.compile(r"latitude|longitude|location|coordinate|(?:^|_)lat(?:_|$)|(?:^|_)lon(?:_|$)", re.I)
DISTRESS = re.compile(r"^\d{2}[LMH]|^family_", re.I)


def disposition(column: str, allow: set[str], targets: set[str], identity_columns: set[str]) -> str:
    if column in targets:
        return "TARGET"
    if column == "delta_days":
        return "WEIGHT"
    if column == "group_id":
        return "GROUP"
    if column == "pair_id":
        return "ROW_KEY"
    if OUTCOME.search(column):
        return "DROP_OUTCOME"
    if column in identity_columns or IDENTITY.search(column):
        return "DROP_IDENTITY"
    if CUMULATIVE.search(column):
        return "DROP_INTERVAL_TOTAL"
    if GEO.search(column):
        return "DROP_GEOGRAPHY"
    if re.search(r"date|DataSource|SnapDistance", column, re.I):
        return "DROP_PROVENANCE"
    if re.search(r"^(AbsMinF|PGLowIntervalF)(?:_start)?$|MR.*Name", column, re.I):
        return "DROP_DUPLICATE"
    if DISTRESS.search(column):
        return "DROP_DISTRESS_INVENTORY"
    if column not in allow:
        return "DROP_NOT_DECLARED"
    if column != "PCI_start" and not column.endswith("_start"):
        return "DROP_NOT_START_STATE"
    return "KEEP"


def build(pairs: pd.DataFrame, allow: list[str], targets: list[str], columns: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(allow) != len(set(allow)) or not allow or not set(allow) <= set(pairs.columns):
        raise SafeError("INVALID_FEATURE_ALLOWLIST")
    identities = {columns[k] + suffix for k in ["section", "group"] for suffix in ["_start", "_end"]}
    audit = pd.DataFrame([{"column": c, "disposition": disposition(c, set(allow), set(targets), identities)}
                          for c in pairs.columns])
    decisions = dict(zip(audit.column, audit.disposition))
    if any(decisions[c] != "KEEP" for c in allow):
        raise SafeError("FORBIDDEN_FEATURE_REQUESTED")
    if not set(targets) <= set(pairs):
        raise SafeError("TARGET_NOT_FOUND")
    table = pairs[["pair_id", "group_id", "delta_days", *targets, *allow]].copy()
    for c in [*targets, *allow]:
        table[c] = pd.to_numeric(table[c], errors="coerce")
        table[c] = table[c].replace([np.inf, -np.inf], np.nan)
    if table[allow].isna().all().any():
        raise SafeError("ALL_MISSING_FEATURE")
    return table, audit


def associations(frame: pd.DataFrame, features: list[str], threshold: float = 0.90) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = frame[features].corr(method="spearman")
    pairs = [{"first": a, "second": b, "rho": float(matrix.loc[a, b])}
             for i, a in enumerate(features) for b in features[i + 1:]
             if np.isfinite(matrix.loc[a, b]) and abs(matrix.loc[a, b]) >= threshold]
    return matrix, pd.DataFrame(pairs, columns=["first", "second", "rho"])
