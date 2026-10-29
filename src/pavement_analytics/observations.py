"""Inspection-level deduct targets and reviewed, outcome-independent covariates.

These observations do not inherit the interval-selection rules used for PCI
deterioration rates. Targets are sums of the channels actually exported for a
family; absent channels are never silently interpreted as observed zeros.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .curation import SURFACES
from .security import SafeError

DEDUCT = re.compile(r"^(?P<code>\d{2})(?P<severity>[LMH]?)\s+deduct$", re.I)


def _families(families: dict) -> dict[str, list[str]]:
    if not isinstance(families, dict) or not families:
        raise SafeError("INVALID_DEDUCT_FAMILY")
    result, claimed = {}, []
    for name, codes in families.items():
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", name)
                or not isinstance(codes, list) or not codes):
            raise SafeError("INVALID_DEDUCT_FAMILY")
        if any(not re.fullmatch(r"\d{2}", str(code)) for code in codes):
            raise SafeError("INVALID_DEDUCT_CODE")
        result[name] = [str(code) for code in codes]
        claimed.extend(result[name])
    if len(set(claimed)) != len(claimed):
        raise SafeError("OVERLAPPING_DEDUCT_FAMILIES")
    return result


def _record_value(value):
    """Hashable comparison preserving source schema, but normalizing missingness."""
    if isinstance(value, (tuple, list)):
        return tuple(value)
    if pd.isna(value):
        return None
    return str(value)


def build_observations(raw: pd.DataFrame, columns: dict, families: dict, *,
                       blank_deducts: str = "missing", date_format: str = "%Y-%m-%d",
                       cutoff_year: int = 1980) -> tuple[pd.DataFrame, list[str], pd.DataFrame, dict]:
    """Keep valid inspections and derive target-specific exported family totals.

    Exact repeated inspections collapse across source files. All conflicting
    records for the same group, section and date are excluded, instead of
    choosing whichever record happens to be first. Missing/invalid PCI is
    audited but does not discard an otherwise observed deduct value.
    """
    families = _families(families)
    if blank_deducts not in ("missing", "zero"):
        raise SafeError("INVALID_BLANK_DEDUCT_POLICY")
    required = {"section", "date", "surface", "group"}
    if not required <= set(columns) or not {columns[key] for key in required} <= set(raw):
        raise SafeError("REQUIRED_INPUT_COLUMNS_MISSING")
    if (not isinstance(cutoff_year, int) or isinstance(cutoff_year, bool)
            or not 1 <= cutoff_year <= 9999):
        raise SafeError("INVALID_CONFIGURATION_RANGE")
    frame = raw.copy().reset_index(drop=True)
    if "_source_file" not in frame:
        frame["_source_file"] = ""
    if "_source_row" not in frame:
        frame["_source_row"] = np.arange(2, len(frame) + 2)
    matched = {col: DEDUCT.fullmatch(col) for col in frame if DEDUCT.fullmatch(col)}
    channels = [(match.group("code"), match.group("severity").upper()) for match in matched.values()]
    if len(channels) != len(set(channels)):
        raise SafeError("NORMALIZED_HEADERS_COLLIDE")
    if "_deduct_schema" not in frame:
        frame["_deduct_schema"] = [tuple(matched)] * len(frame)
    dates = pd.to_datetime(frame[columns["date"]], format=date_format, errors="coerce")
    frame["inspection_date"] = dates.dt.strftime("%Y-%m-%d")
    frame["group_id"] = frame[columns["group"]].fillna("").astype(str).str.strip()
    frame["section_id"] = frame[columns["section"]].fillna("").astype(str).str.strip()
    frame["material"] = frame[columns["surface"]].fillna("").astype(str).str.strip().str.upper().map(SURFACES)
    reason = pd.Series("", index=frame.index, dtype=object)
    for label, mask in [
        ("invalid_date", dates.isna()), ("before_cutoff", dates.dt.year.lt(cutoff_year)),
        ("missing_section", frame.section_id.eq("")), ("missing_group", frame.group_id.eq("")),
        ("unknown_surface", frame.material.isna()),
    ]:
        reason.loc[mask & reason.eq("")] = label

    # Exclude provenance from equality; include all actual source fields and the
    # channel schema so an absent channel cannot equal an exported blank channel.
    compare = [col for col in raw if not col.startswith("_")]
    compare += ["_deduct_schema"]
    valid = frame.loc[reason.eq("")]
    keys = ["group_id", "section_id", "inspection_date"]
    duplicate = valid.duplicated(keys, keep=False)
    for _, block in valid.loc[duplicate].groupby(keys, sort=True):
        signatures = [tuple(_record_value(value) for value in record)
                      for record in block[compare].itertuples(index=False, name=None)]
        if len(set(signatures)) == 1:
            reason.loc[block.index[1:]] = "duplicate_inspection_collapsed"
        else:
            reason.loc[block.index] = "conflicting_inspection"
    rejected = raw.reset_index(drop=True).loc[reason.ne("")].copy()
    rejected["reason"] = reason.loc[reason.ne("")]
    table = frame.loc[reason.eq("")].copy().reset_index(drop=True)
    table["row_id"] = np.arange(len(table), dtype=int)

    deducts = list(matched)
    present = pd.DataFrame(False, index=table.index, columns=deducts)
    for col in deducts:
        present[col] = table._deduct_schema.map(lambda schema: col in schema)
    values = table[deducts].copy().astype(object)
    blank = values.isna()
    for col in deducts:
        blank[col] |= values[col].map(lambda value: isinstance(value, str) and not value.strip())
    blank &= present
    if blank_deducts == "zero":
        values = values.mask(blank, 0)
    for col in deducts:
        values[col] = pd.to_numeric(values[col], errors="coerce")
    values = values.astype(float).where(present)
    invalid = present & (~np.isfinite(values) | ~values.ge(0) | ~values.le(100))
    ambiguity = {}
    for code in sorted({match.group("code") for match in matched.values()}):
        ungraded = [col for col, match in matched.items() if match.group("code") == code
                    and not match.group("severity")]
        graded = [col for col, match in matched.items() if match.group("code") == code
                  and match.group("severity")]
        ambiguity[code] = (values[ungraded].gt(0).any(axis=1) & values[graded].gt(0).any(axis=1))
    pci_col = columns.get("pci")
    pci = pd.to_numeric(table[pci_col], errors="coerce") if pci_col in table else pd.Series(np.nan, index=table.index)
    valid_pci = np.isfinite(pci) & pci.between(0, 100)
    export_gap = (present.any(axis=1) & ~invalid.any(axis=1)
                  & values.sum(axis=1).eq(0) & valid_pci & pci.lt(99.5))
    table["blank_deduct_cells"] = blank.sum(axis=1).astype(int)
    table["zero_filled_deduct_cells"] = table.blank_deduct_cells if blank_deducts == "zero" else 0
    table["export_gap"] = export_gap
    targets, target_audit = [], []
    for name, codes in families.items():
        target = "level_" + name
        targets.append(target)
        selected = [col for col, match in matched.items() if match.group("code") in codes]
        absent = ~present[selected].any(axis=1)
        invalid_family = invalid[selected].any(axis=1)
        ambiguous_family = pd.Series(False, index=table.index)
        for code in codes:
            if code in ambiguity:
                ambiguous_family |= ambiguity[code]
        table[target] = values[selected].sum(axis=1, min_count=1)
        table.loc[absent | invalid_family | ambiguous_family | export_gap, target] = np.nan
        # Reasons are mutually exclusive, in this precedence, for exact accounting.
        target_reason = pd.Series("valid", index=table.index)
        for label, mask in [("family_not_exported", absent), ("invalid_family_value", invalid_family),
                            ("ambiguous_family_representation", ambiguous_family),
                            ("all_zero_export_gap", export_gap)]:
            target_reason.loc[mask & target_reason.eq("valid")] = label
        for material, subset in table.groupby("material", sort=True):
            mask = subset.index
            counts = target_reason.loc[mask].value_counts().to_dict()
            target_audit.append({"material": material, "target": target,
                                 "valid_target_rows": int(subset[target].notna().sum()),
                                 "excluded_missing_target": int(subset[target].isna().sum()),
                                 "valid_target_groups": int(subset.loc[subset[target].notna(), "group_id"].nunique()),
                                 "reasons": {key: int(value) for key, value in counts.items()}})
    coverage = []
    for source, block in frame.groupby("_source_file", sort=True):
        schemas = sorted({tuple(schema) for schema in block._deduct_schema})
        for schema in schemas:
            observed_codes = {DEDUCT.fullmatch(col).group("code") for col in schema if DEDUCT.fullmatch(col)}
            coverage.append({"source": source, "exported_deduct_columns": list(schema),
                             "families": {name: {"exported_codes": sorted(set(codes) & observed_codes),
                                                  "unexported_codes": sorted(set(codes) - observed_codes)}
                                          for name, codes in families.items()}})
    audit = {"schema_version": 1, "target_kind": "exported_deduct_family_total",
             "blank_deducts": blank_deducts,
             "accounting": {"input_rows": len(raw), "accepted_rows": len(table), "rejected_rows": len(rejected)},
             "rejected_reasons": {key: int(value) for key, value in rejected.reason.value_counts().items()},
             "blank_deduct_cells": int(blank.to_numpy().sum()),
             "zero_filled_deduct_cells": int(table.zero_filled_deduct_cells.sum()),
             "invalid_pci_rows_retained": int((~valid_pci).sum()),
             "export_gap_rows": int(export_gap.sum()), "targets": target_audit, "source_coverage": coverage,
             "coverage_note": "Targets sum exported channels only. Missing code/severity columns are not confirmed zeros; compare source coverage before interpreting pooled totals.",
             "selection_note": "All valid inspections are eligible, including first inspections, construction records, condition increases and short intervals."}
    return table, targets, rejected, audit


def _key(column: str) -> str:
    return re.sub(r"[^a-z0-9]", "", column.casefold())


# Only fields with an explicit role independent of the contemporaneous distress
# inventory are auto-admitted. Header normalization permits ordinary PAVER space
# variations. Numeric data are converted before fitting; raw categories remain
# categories for train-only encoding by the modeling stage.
NUMERIC = {
    "yearssincemajormr", "yearssinceglobalmr", "yearssincelocalizedmr",
    "inspectedareasqft", "inspectedslabs", "allowablegrossload", "allowablegrossloadfrost",
    "allowablepasses", "allowablepassesfrost", "acnpcn", "acnpcnfrost",
}
CATEGORICAL = {"subgradecategory", "inspectedbranchuse"}
BOOLEAN = {"isconstruction", "construction", "constructionflag"}


def select_covariates(table: pd.DataFrame, requested: list[str] | None = None) -> tuple[list[str], pd.DataFrame]:
    """Admit reviewed covariates, normalize them in-place, and audit every field.

    Additional climate fields are selected by the climate adapter under its own
    historical-window contract. Merely naming an unknown column is insufficient
    to admit it; the reviewed allowlist must be extended for that field's role.
    """
    if requested is not None and (not isinstance(requested, list)
                                  or any(not isinstance(col, str) for col in requested)
                                  or len(requested) != len(set(requested))):
        raise SafeError("INVALID_FEATURE_ALLOWLIST")
    allowed = NUMERIC | CATEGORICAL | BOOLEAN
    if requested is not None:
        if not set(requested) <= set(table):
            raise SafeError("FEATURE_NOT_FOUND")
        if any(_key(col) not in allowed for col in requested):
            raise SafeError("FEATURE_REVIEW_REQUIRED")
    chosen = set(requested) if requested is not None else None
    features, audit = [], []
    for col in table:
        key = _key(col)
        if key not in allowed:
            disposition = "TARGET" if col.startswith("level_") else "DROP_NOT_REVIEWED"
            audit.append({"column": col, "disposition": disposition, "kind": None})
            continue
        if chosen is not None and col not in chosen:
            audit.append({"column": col, "disposition": "DROP_NOT_REQUESTED", "kind": None})
            continue
        if key in NUMERIC:
            series = pd.to_numeric(table[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            table[col] = series.where(series.ge(0))
            kind = "numeric"
        elif key in BOOLEAN:
            text = table[col].fillna("").astype(str).str.strip().str.lower()
            table[col] = text.map({"1": 1.0, "1.0": 1.0, "true": 1.0, "yes": 1.0, "y": 1.0,
                                   "0": 0.0, "0.0": 0.0, "false": 0.0, "no": 0.0, "n": 0.0})
            kind = "numeric"
        else:
            text = table[col].fillna("").astype(str).str.strip()
            table[col] = text.where(text.ne(""), np.nan)
            kind = "categorical"
        if table[col].isna().all():
            disposition = "DROP_ALL_MISSING"
        else:
            disposition = "KEEP"
            features.append(col)
        audit.append({"column": col, "disposition": disposition, "kind": kind,
                      "missing_rows": int(table[col].isna().sum())})
    # Every row needs this value: omitted numeric fields become NaN when pandas
    # constructs the table, which would break the strict JSON run audit.
    for record in audit:
        record.setdefault("missing_rows", int(table[record["column"]].isna().sum()))
    return features, pd.DataFrame(audit, columns=["column", "disposition", "kind", "missing_rows"])
