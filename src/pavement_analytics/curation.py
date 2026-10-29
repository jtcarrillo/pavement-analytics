"""Audited inspection filtering and conservative consecutive-interval pairing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .security import SafeError

SURFACES = {"PCC": "PCC", "APC": "APC", "AC": "AC", "AAC": "AC", "ACC": "AC"}


def build_pairs(raw: pd.DataFrame, columns: dict, *, cutoff_year: int = 1980,
                min_days: int = 365, max_days: int | None = None,
                date_format: str = "%Y-%m-%d") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if min_days < 1 or (max_days is not None and max_days < min_days):
        raise SafeError("INVALID_INTERVAL_LIMITS")
    frame = raw.copy()
    frame["_date"] = pd.to_datetime(frame[columns["date"]], format=date_format, errors="coerce")
    frame["_pci"] = pd.to_numeric(frame[columns["pci"]], errors="coerce")
    frame["_section"] = frame[columns["section"]].fillna("").astype(str).str.strip()
    frame["_group"] = frame[columns["group"]].fillna("").astype(str).str.strip()
    frame["_material"] = frame[columns["surface"]].str.strip().str.upper().map(SURFACES)
    if "construction" in columns:
        flag = frame[columns["construction"]].fillna("").astype(str).str.strip().str.lower()
        if not flag.isin(["", "0", "1", "false", "true", "no", "yes", "n", "y"]).all():
            raise SafeError("INVALID_CONSTRUCTION_FLAG")
        frame["_construction"] = flag.isin(["1", "true", "yes", "y"])
    else:
        frame["_construction"] = False
    reason = pd.Series("", index=frame.index)
    for label, mask in [
        ("invalid_date", frame._date.isna()),
        ("before_cutoff", frame._date.dt.year.lt(cutoff_year)),
        ("invalid_pci", ~np.isfinite(frame._pci) | ~frame._pci.between(0, 100)),
        ("missing_section", frame._section.eq("")),
        ("missing_group", frame._group.eq("")),
        ("unknown_surface", frame._material.isna()),
    ]:
        reason.loc[mask & reason.eq("")] = label
    rejected = raw.loc[reason.ne("")].copy()
    rejected["reason"] = reason[reason.ne("")]
    valid = frame.loc[reason.eq("")]
    pairs, transitions = [], []
    passthrough = [c for c in raw.columns if not c.startswith("_")]
    provenance = ["_source_file", "_source_row", "_source_sha256"]
    if "_deduct_schema" in raw:
        provenance.append("_deduct_schema")
    for _, block in valid.groupby(["_source_file", "_section", "_group", "_material"], sort=True):
        ordered = block.sort_values(["_date", "_source_row"], kind="stable").to_dict("records")
        anchor = ordered[0]
        for row in ordered[1:]:
            days = (row["_date"] - anchor["_date"]).days
            delta = row["_pci"] - anchor["_pci"]
            if days == 0:
                status = "same_day_collapsed" if delta == 0 else "same_day_conflict"
            elif row["_construction"]:
                status = "construction_reset"
            elif delta > 0:
                status = "increase_reset"
            elif days < min_days:
                status = "below_minimum_interval"
            elif max_days is not None and days > max_days:
                status = "above_maximum_interval"
            else:
                status = "paired"
                pair = {"group_id": row["_group"], "section_id": row["_section"],
                        "material": row["_material"], "start_date": anchor["_date"].date().isoformat(),
                        "end_date": row["_date"].date().isoformat(), "delta_days": days,
                        "PCI_start": anchor["_pci"], "PCI_end": row["_pci"],
                        "delta_PCI": delta, "DetRatePerYear": delta * 365.0 / days}
                for c in passthrough + provenance:
                    if c + "_start" not in pair:
                        pair[c + "_start"] = anchor[c]
                    if c + "_end" not in pair:
                        pair[c + "_end"] = row[c]
                pairs.append(pair)
            transitions.append({"source_file": row["_source_file"], "source_row": row["_source_row"],
                                "anchor_row": anchor["_source_row"], "reason": status})
            # Even a filtered interval advances the anchor: never bridge over an inspection.
            anchor = row
    pair_cols = ["group_id", "section_id", "material", "start_date", "end_date", "delta_days",
                 "PCI_start", "PCI_end", "delta_PCI", "DetRatePerYear"]
    result = pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=pair_cols)
    if not result.empty:
        result.insert(0, "pair_id", np.arange(len(result), dtype=int))
    else:
        result.insert(0, "pair_id", pd.Series(dtype=int))
    audit = pd.DataFrame(transitions, columns=["source_file", "source_row", "anchor_row", "reason"])
    counts = {"input_rows": len(raw), "accepted_rows": len(valid), "rejected_rows": len(rejected),
              "pair_rows": len(result), "transition_count": len(audit),
              "initial_anchors": len(valid) - len(audit)}
    assert counts["accepted_rows"] + counts["rejected_rows"] == counts["input_rows"]
    assert counts["transition_count"] + counts["initial_anchors"] == counts["accepted_rows"]
    return result, rejected, audit, counts
