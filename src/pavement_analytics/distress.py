"""Derive deduct totals and rates from generic PAVER distress columns."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .pci import calculate
from .security import SafeError

DEDUCT = re.compile(r"^(?P<code>\d{2})(?P<severity>[LMH]?)\s+deduct_(?P<endpoint>start|end)$", re.I)


def derive(pairs: pd.DataFrame, families: dict, profiles: dict | None = None,
           *, blank_deducts: str = "missing") -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(blank_deducts, str) or blank_deducts not in {"missing", "zero"}:
        raise SafeError("INVALID_BLANK_DEDUCT_POLICY")
    frame = pairs.copy()
    if not isinstance(families, dict):
        raise SafeError("INVALID_DEDUCT_FAMILY")
    claimed = []
    for name, codes in families.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", name) or not isinstance(codes, list) or not codes:
            raise SafeError("INVALID_DEDUCT_FAMILY")
        if any(not re.fullmatch(r"\d{2}", str(c)) for c in codes):
            raise SafeError("INVALID_DEDUCT_CODE")
        claimed.extend(str(c) for c in codes)
    if len(set(claimed)) != len(claimed):
        raise SafeError("OVERLAPPING_DEDUCT_FAMILIES")
    matched = {c: DEDUCT.match(c) for c in frame if DEDUCT.match(c)}
    targets = []
    for end in ["start", "end"]:
        cols = [c for c, m in matched.items() if m.group("endpoint").lower() == end]
        present = pd.DataFrame(True, index=frame.index, columns=cols)
        schema = f"_deduct_schema_{end}"
        if schema in frame:
            for col in cols:
                header = col[:-(len(end) + 1)]
                present[col] = frame[schema].map(lambda names: header in names)
        # Decide blank semantics before numeric coercion: malformed numeric text
        # must remain invalid, and source-absent channels must never become zero.
        raw_values = frame[cols].astype(object)
        blank = raw_values.isna()
        for col in cols:
            blank[col] |= raw_values[col].map(lambda value: isinstance(value, str) and not value.strip())
        blank &= present
        frame[f"blank_deduct_cells_{end}"] = blank.sum(axis=1).astype(int)
        frame[f"zero_filled_deduct_cells_{end}"] = (
            frame[f"blank_deduct_cells_{end}"] if blank_deducts == "zero" else 0)
        if blank_deducts == "zero":
            raw_values = raw_values.mask(blank, 0)
        for col in cols:
            frame[col] = pd.to_numeric(raw_values[col], errors="coerce")
        values = frame[cols]
        invalid = ((~np.isfinite(values) | ~values.ge(0) | ~values.le(100)) & present).any(axis=1)
        invalid |= ~present.any(axis=1)
        values = values.where(present)
        # An unsuffixed channel may overlap severity channels for the same code.
        # Only one representation may contribute positive values at an endpoint;
        # never infer whether simultaneous positive values should be added.
        ambiguous_codes = pd.Series(0, index=frame.index, dtype=int)
        for code in sorted({matched[col].group("code") for col in cols}):
            ungraded = [col for col in cols if matched[col].group("code") == code
                        and not matched[col].group("severity")]
            graded = [col for col in cols if matched[col].group("code") == code
                      and matched[col].group("severity")]
            if ungraded and graded:
                overlap = values[ungraded].gt(0).any(axis=1) & values[graded].gt(0).any(axis=1)
                ambiguous_codes += overlap.astype(int)
        frame[f"ambiguous_deduct_codes_{end}"] = ambiguous_codes
        frame[f"ambiguous_deduct_{end}"] = ambiguous_codes.gt(0)
        invalid |= frame[f"ambiguous_deduct_{end}"]
        frame[f"invalid_deduct_{end}"] = invalid
        total = values.sum(axis=1)
        frame[f"export_gap_{end}"] = (total.eq(0) & frame[f"PCI_{end}"].lt(99.5)) | invalid
        for name, codes in families.items():
            family_cols = [c for c in cols if matched[c].group("code") in {str(v) for v in codes}]
            frame[f"family_{name}_{end}"] = values[family_cols].sum(axis=1, min_count=1) if family_cols else np.nan
            frame.loc[invalid, f"family_{name}_{end}"] = np.nan
        if profiles:
            calculated = []
            for index, row in frame.iterrows():
                if invalid.loc[index]:
                    calculated.append(np.nan)
                else:
                    active = [c for c in cols if present.loc[index, c]]
                    calculated.append(calculate(row[active].to_numpy(float), profiles.get(row.material)))
            frame[f"pci_calc_{end}"] = calculated
            frame[f"pci_diff_{end}"] = frame[f"pci_calc_{end}"] - frame[f"PCI_{end}"]
    gap = frame.export_gap_start | frame.export_gap_end
    for name in families:
        target = "rate_" + name
        frame[target] = (frame[f"family_{name}_end"] - frame[f"family_{name}_start"]) * 365 / frame.delta_days
        frame.loc[gap, target] = np.nan
        targets.append(target)
    return frame, targets
