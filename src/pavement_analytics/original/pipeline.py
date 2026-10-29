"""Preserved preprocessing stages feeding the full local modeling pipeline."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from .common import ROOT, Run, clean_json, numeric_stats, read_table, sha256, write_csv, write_json
from . import check_cli_paths
from ..security import reject_links, refuse_public_execution

SURFACES = ("PCC", "APC", "AC_AAC_ACC")
DEFAULTS = {
    "input_dir": None, "output_dir": None, "mapping": None, "csv_dir": None,
    "recursive": False, "cutoff_year": 1980, "surfaces": list(SURFACES),
    "section_scope": "source", "airfield_col": "_source_file_start",
    "skip_climate": False, "climate_start_year": 1980,
    "climate_end_year": dt.date.today().year - 1, "cache_dir": None,
    "climate_args": [], "construction_year_col": "Original Construction Year_start",
    "on_missing_location": "warn", "skip_distress": False, "skip_pci": False,
    "no_shapley": False, "fit_deduct_curves": False, "deduct_curves": None,
    "families": None, "cdv_curves": None, "target_mode": "deduct_families", "standard": "D5340", "rules": "D6433", "sev_mode": "sequential",
    "min_groups": 5, "keep_distress": False, "keep_latlon": False,
    "min_delta_days": 1, "max_delta_days": None, "high_card_threshold": .95,
    "extra_drop": [], "fail_on_warning": False, "pci_warning_mae": 5.0,
    "datarobot": "off", "targets": {}, "dr_mode": "quick", "dr_cv_reps": 5,
    "dr_holdout_pct": 20, "dr_name_prefix": "paver", "feature_impact": False,
    "retired_backend_request": None,
}
PATH_KEYS = ("input_dir", "output_dir", "mapping", "csv_dir", "cache_dir", "deduct_curves", "families", "cdv_curves")
RAW_DISTRESS = re.compile(r"^\d{2}[LMH]? (deduct|\((Slabs|SqFt|Ft|Count)\))$")
RAW_DEDUCT_HEADER = re.compile(r"^(?P<code>\d{2})[LMH]? deduct$")
IDENTIFIER_KEYS = {"branchname", "branchid", "networkname", "networkid", "sectionid",
                   "sectionuniqueid", "sectionname", "sectionfrom", "sectionto", "groupid",
                   "groupname", "airfieldid", "airfieldname", "siteid", "sitename"}
RESERVED = {"--input", "--output", "--lat-col", "--lon-col", "--dates-col", "--cache-dir",
            "--start-year", "--end-year", "--construction-year-col", "--col-prefix", "--col-suffix"}


def normalized_column_stem(column):
    normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
    return re.sub(r"(?:start|end|delta)$", "", normalized)


def family_schema_details(headers, codes):
    expected = set(map(str, codes))
    channels = sorted(header for header in headers
                      if RAW_DEDUCT_HEADER.fullmatch(header).group("code") in expected)
    observed = {RAW_DEDUCT_HEADER.fullmatch(header).group("code") for header in channels}
    return {"exported_channels": json.dumps(channels), "observed_codes": json.dumps(sorted(observed)),
            "missing_codes": json.dumps(sorted(expected - observed)),
            "coverage": "complete" if observed == expected else "partial" if observed else "unobserved"}


def parse_config(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=Path, help="JSON settings; relative paths are relative to this file")
    p.add_argument("--resume", action="store_true", help="Reuse verified successful stages of the same run")
    for key in PATH_KEYS:
        p.add_argument("--" + key.replace("_", "-"), default=None)
    for key in ("recursive", "skip_climate", "skip_distress", "skip_pci", "fit_deduct_curves",
                "keep_distress", "keep_latlon", "fail_on_warning"):
        p.add_argument("--" + key.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--no-shapley", action="store_true", default=None,
                   help="Skip family decomposition while retaining recomputed PCI")
    p.add_argument("--shapley", dest="no_shapley", action="store_false", default=None)
    for key in ("cutoff_year", "climate_start_year", "climate_end_year", "min_groups", "min_delta_days",
                "max_delta_days", "dr_cv_reps", "dr_holdout_pct"):
        p.add_argument("--" + key.replace("_", "-"), type=int, default=None)
    p.add_argument("--surfaces", nargs="+", choices=SURFACES, default=None)
    p.add_argument("--target-mode", choices=("deduct_families", "legacy"), default=None)
    p.add_argument("--section-scope", choices=("source", "global"), default=None)
    p.add_argument("--on-missing-location", choices=("warn", "fail"), default=None)
    p.add_argument("--standard", choices=("D5340", "D6433"), default=None)
    p.add_argument("--rules", choices=("D5340", "D6433"), default=None)
    p.add_argument("--sev-mode", choices=("sequential", "joint"), default=None)
    p.add_argument("--airfield-col", default=None)
    p.add_argument("--construction-year-col", default=None)
    p.add_argument("--pci-warning-mae", type=float, default=None)
    p.add_argument("--high-card-threshold", type=float, default=None)
    args = vars(p.parse_args(argv))
    config_path = args.pop("config")
    resume = args.pop("resume")
    cfg = dict(DEFAULTS)
    if config_path:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            p.error("Config must be a JSON object")
        unknown = set(raw) - set(DEFAULTS)
        if unknown:
            p.error(f"Unknown config keys: {sorted(unknown)}")
        for key in PATH_KEYS:
            if raw.get(key):
                path = Path(raw[key]).expanduser()
                raw[key] = str((config_path.resolve().parent / path).resolve())
        cfg.update(raw)
    cfg.update({k: v for k, v in args.items() if v is not None})
    for key in PATH_KEYS:
        if cfg[key]:
            cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    if not cfg["input_dir"] or not cfg["output_dir"]:
        p.error("Provide input_dir and output_dir in JSON or as CLI options")
    cfg["csv_dir"] = cfg["csv_dir"] or (str(Path(cfg["mapping"]).parent) if cfg["mapping"] else None)
    cfg["cache_dir"] = cfg["cache_dir"] or str(Path(cfg["output_dir"]).parent / "era5_cache")
    if cfg["datarobot"] != "off":
        cfg["retired_backend_request"] = cfg["datarobot"]
        cfg["datarobot"] = "off"
    return cfg, resume


def validate_config(c):
    check_cli_paths(inputs=[c.get(key) for key in ("input_dir", "mapping", "csv_dir", "deduct_curves", "families", "cdv_curves")],
                    outputs=[c.get("output_dir"), c.get("cache_dir")])
    for key, default in DEFAULTS.items():
        value = c[key]
        if isinstance(default, bool) and not isinstance(value, bool):
            raise ValueError(f"{key} must be a JSON boolean")
        if isinstance(default, int) and not isinstance(default, bool) and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{key} must be an integer")
    for key, choices in {"target_mode": ["deduct_families", "legacy"], "datarobot": ["off"], "section_scope": ["source", "global"],
                         "on_missing_location": ["warn", "fail"], "standard": ["D5340", "D6433"],
                         "rules": ["D5340", "D6433"], "sev_mode": ["sequential", "joint"],
                         "dr_mode": ["quick", "full"]}.items():
        if c[key] not in choices:
            raise ValueError(f"{key} must be one of {choices}")
    if not isinstance(c["surfaces"], list) or not c["surfaces"] or set(c["surfaces"]) - set(SURFACES):
        raise ValueError(f"surfaces must be a nonempty list from {SURFACES}")
    if len(c["surfaces"]) != len(set(c["surfaces"])):
        raise ValueError("Duplicate surface selections")
    for key in ("min_groups", "min_delta_days"):
        if c[key] < 1:
            raise ValueError(f"{key} must be positive")
    if c["max_delta_days"] is not None and (not isinstance(c["max_delta_days"], int) or c["max_delta_days"] < c["min_delta_days"]):
        raise ValueError("max_delta_days must be an integer >= min_delta_days")
    if not 1 <= c["cutoff_year"] <= 9999 or not 1940 <= c["climate_start_year"] <= c["climate_end_year"] <= dt.date.today().year:
        raise ValueError("Invalid cutoff or climate download years")
    if not 0 < c["high_card_threshold"] <= 1 or not np.isfinite(c["pci_warning_mae"]) or c["pci_warning_mae"] < 0:
        raise ValueError("Invalid high_card_threshold or pci_warning_mae")
    if not 2 <= c["dr_cv_reps"] or not 0 <= c["dr_holdout_pct"] < 100:
        raise ValueError("dr_cv_reps must be >=2 and dr_holdout_pct must be 0..99")
    for key in ("climate_args", "extra_drop"):
        if not isinstance(c[key], list) or not all(isinstance(v, str) for v in c[key]):
            raise ValueError(f"{key} must be a list of strings")
    if any(a.split("=", 1)[0] in RESERVED or a in ("-h", "--help") for a in c["climate_args"]):
        raise ValueError("climate_args cannot override runner-managed I/O, column names, or years")
    if not isinstance(c["targets"], dict) or set(c["targets"]) - set(SURFACES):
        raise ValueError("targets must map surface names to lists of numeric target column names")
    for names in c["targets"].values():
        if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
            raise ValueError("Each targets entry must contain unique nonempty column names")
        if set(names) & {"delta_days", "partition_airfield", "partition_section", "SectionUniqueID", c["airfield_col"]}:
            raise ValueError("Targets cannot also be weight or partition columns")
    if c["fit_deduct_curves"] and c["deduct_curves"]:
        raise ValueError("Choose fit_deduct_curves or deduct_curves, not both")
    for key in ("mapping", "deduct_curves", "families", "cdv_curves"):
        if c[key] and not Path(c[key]).is_file():
            raise ValueError(f"{key} file not found: {c[key]}")
    if not c["skip_pci"] and not c["cdv_curves"]:
        raise ValueError("PCI reconstruction requires a privately supplied cdv_curves file")
    if not c["skip_climate"] and not c["mapping"]:
        raise ValueError("Climate enrichment needs --mapping and its location CSVs; use --skip-climate for a local run")
    if c["families"]:
        fam = json.loads(Path(c["families"]).read_text(encoding="utf-8"))
        if not isinstance(fam, dict) or not fam or not all(isinstance(cs, list) and all(isinstance(x, str) and re.fullmatch(r"\d{2}", x) for x in cs) for cs in fam.values()):
            raise ValueError("families must be a nonempty {family: [two-digit code strings]} object")
        codes = [x for cs in fam.values() for x in cs]
        if len(codes) != len(set(codes)):
            raise ValueError("A distress code must belong to only one family")


def discover(c):
    root, out = Path(c["input_dir"]), Path(c["output_dir"])
    if not root.is_dir():
        raise ValueError(f"Input directory not found: {root}")
    if out == root or root.is_relative_to(out):
        raise ValueError("Output must not equal or contain the input directory")
    iterator = root.rglob("*") if c["recursive"] else root.iterdir()
    files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in {".xlsx", ".xlsm", ".csv", ".tsv"}
                   and not p.name.startswith("~$") and not p.resolve().is_relative_to(out)
                   and not p.resolve().is_relative_to(Path(c["cache_dir"])))
    if not files:
        raise ValueError("No .xlsx, .xlsm, .csv, or .tsv input files found")
    names = [p.name for p in files]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate input basenames make location mapping ambiguous. Rename the workbooks before running.")
    mapping = {}
    if c["mapping"]:
        mapping = json.loads(Path(c["mapping"]).read_text(encoding="utf-8"))
        if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
            raise ValueError("mapping must be a JSON {source_filename: location_csv_filename} object")
        if any(not value or Path(value).is_absolute() or ".." in Path(value).parts for value in mapping.values()):
            raise ValueError("Location mapping entries must stay within csv_dir")
        if not Path(c["csv_dir"]).is_dir():
            raise ValueError(f"Location CSV directory not found: {c['csv_dir']}")
    referenced = files + [Path(c[k]) for k in ("mapping", "deduct_curves", "families", "cdv_curves") if c[k]]
    referenced += [Path(c["csv_dir"]) / mapping[p.name] for p in files if p.name in mapping]
    inputs = [{"path": str(p.resolve()), "sha256": sha256(p) if p.is_file() else None} for p in sorted(set(referenced))]
    code = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(ROOT.rglob("*.py")) if "tests" not in p.parts}
    # Retired-service preferences do not change local curation results.
    stable_cfg = {k: v for k, v in c.items() if k not in {"datarobot", "feature_impact", "fail_on_warning", "dr_mode", "dr_name_prefix", "retired_backend_request"}}
    fingerprint = hashlib.sha256(json.dumps([stable_cfg, inputs, code], sort_keys=True).encode()).hexdigest()
    return files, mapping, inputs, fingerprint


class Pipeline:
    def __init__(self, c, run, files, mapping):
        self.c, self.run, self.files, self.mapping = c, run, files, mapping
        self.out = Path(c["output_dir"])
        from . import prepare, split, pairs, location, rates, distress, pci
        self.prep, self.split, self.pairs = prepare, split, pairs
        self.loc, self.rate, self.distress, self.pci = location, rates, distress, pci
        self.families = json.loads(Path(c["families"]).read_text()) if c["families"] else self.distress.FAMILIES

    def path(self, stage, surface):
        return self.out / stage / f"{surface}.csv"

    def ingest(self):
        frames, file_stats, schema_audit = [], [], []
        required = {"SectionUniqueID", "Date", "PCI", "Inspected Surface Type"}
        for path in self.files:
            df = read_table(path)
            missing = required - set(df)
            if missing:
                raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")
            if any(c.startswith(("_source_", "_pipeline_")) for c in df):
                raise ValueError(f"{path.name}: reserved provenance columns present; input must be raw inspections")
            if path.suffix.lower() in {".xlsx", ".xlsm"}:
                with pd.ExcelFile(path) as book:
                    if len(book.sheet_names) > 1:
                        self.run.warn("FIRST_WORKSHEET_ONLY", f"{path.name}: using first worksheet {book.sheet_names[0]!r}; {len(book.sheet_names)-1} other sheets are ignored.")
            exported = sorted(c for c in df if RAW_DEDUCT_HEADER.fullmatch(c))
            columns = len(df.columns)
            df["_pipeline_deduct_schema"] = json.dumps(exported)
            df["_source_file"] = path.name
            df["_source_row"] = np.arange(2, len(df) + 2)
            file_stats.append({"file": path.name, "rows": len(df), "columns": columns,
                               "exported_deduct_columns": exported})
            schema_audit.extend({"source": path.name, "family": family, "inspection_rows": len(df),
                                 **family_schema_details(exported, codes)}
                                for family, codes in sorted(self.families.items()))
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            raise ValueError("Input files contain no inspection rows")
        df["SectionUniqueID"] = df["SectionUniqueID"].astype("string").str.strip()
        dates = pd.to_datetime(df["Date"], errors="coerce")
        pci = pd.to_numeric(df["PCI"], errors="coerce")
        reasons = pd.Series("", index=df.index)
        def reject(mask, reason):
            reasons.loc[mask & reasons.eq("")] = reason
        reject(dates.isna(), "missing_or_invalid_date")
        reject(df.SectionUniqueID.isna() | df.SectionUniqueID.eq(""), "missing_section_id")
        reject(~np.isfinite(pci) | ~pci.between(0, 100), "invalid_pci")
        reject(dates < pd.Timestamp(self.c["cutoff_year"], 1, 1), "before_cutoff")
        buckets = self.split.categorize(df, "Inspected Surface Type", False)
        reject(df.index.isin(buckets["OTHER"].index), "unknown_surface")
        for surface in set(SURFACES) - set(self.c["surfaces"]):
            reject(df.index.isin(buckets[surface].index), "unselected_surface")
        excluded = df.loc[reasons.ne("")].copy()
        excluded["exclusion_reason"] = reasons[reasons.ne("")]
        rejected = self.out / "01_split/rejected_rows.csv"
        write_csv(rejected, excluded)
        counts = reasons[reasons.ne("")].value_counts().to_dict()
        for reason, count in counts.items():
            self.run.warn("INPUT_EXCLUDED", f"{count} inspection rows excluded: {reason}. See 01_split/rejected_rows.csv.", count)
        df["Date"], df["PCI"] = dates, pci
        kept = df.loc[reasons.eq("")].copy()
        reused = kept.groupby("SectionUniqueID")["_source_file"].nunique().gt(1)
        if reused.any():
            self.run.warn("SECTION_ID_REUSED", f"{reused.sum()} section IDs occur in multiple workbooks. Pairing scope is {self.c['section_scope']}.", reused.sum())
        schema_path = self.out / "01_split/source_family_schema.csv"
        write_csv(schema_path, pd.DataFrame(schema_audit))
        artifacts = [rejected, schema_path]
        surface_counts = {}
        for surface in self.c["surfaces"]:
            subset = kept.loc[kept.index.isin(buckets[surface].index)].copy()
            surface_counts[surface] = len(subset)
            if subset.empty:
                self.run.warn("EMPTY_SURFACE", f"{surface}: no eligible inspections.")
            path = self.path("01_split", surface)
            write_csv(path, subset)
            artifacts.append(path)
        return {"files": file_stats, "input_rows": len(df), "retained_rows": len(kept),
                "excluded_rows": len(excluded), "exclusion_reasons": counts, "surfaces": surface_counts,
                "source_family_schema": schema_audit}, artifacts

    def pair(self, surface):
        df = read_table(self.path("01_split", surface))
        if "IsConstruction" not in df and len(df):
            self.run.warn("MISSING_CONSTRUCTION_FLAG", f"{surface}: IsConstruction is absent; equal-PCI inspections will collapse.")
        group = ["SectionUniqueID"] if self.c["section_scope"] == "global" else ["_source_file", "SectionUniqueID"]
        pairs, sections, stats = self.pairs.build_pairs(df, "SectionUniqueID", "PCI", "Date", group_cols=group)
        for metric, code in [("total_increases_skipped", "PCI_INCREASE_SKIPPED"),
                             ("total_duplicates_collapsed", "EQUAL_PCI_COLLAPSED"),
                             ("total_nonpositive_intervals_skipped", "SAME_DAY_PAIR_SKIPPED")]:
            if stats[metric]:
                self.run.warn(code, f"{surface}: {stats[metric]} ({metric}).", stats[metric])
        if pairs.empty:
            self.run.warn("NO_PAIRS", f"{surface}: no positive-duration deterioration pairs.")
        path, summary = self.path("02_pairs", surface), self.out / "02_pairs" / f"{surface}_sections.csv"
        write_csv(path, pairs)
        write_csv(summary, sections)
        stats["delta_days"] = numeric_stats(pairs.delta_days)
        return stats, [path, summary]

    def location_rate(self, surface):
        df = read_table(self.path("02_pairs", surface))
        resolved, audit = {}, []
        for source in df["_source_file_start"].dropna().unique():
            raw, error = None, None
            if source not in self.mapping:
                error = "source_not_mapped"
            else:
                path = Path(self.c["csv_dir"]) / self.mapping[source]
                if not path.is_file():
                    error = "location_csv_missing"
                else:
                    try:
                        raw = self.loc.load_location_from_csv(path, "LOCATION(DEGREES)")
                    except RuntimeError as exc:
                        error = str(exc)
            lat, lon = self.loc.parse_lat_lon(raw)
            if lat is None:
                error = error or "missing_or_invalid_coordinates"
            resolved[source] = raw
            audit.append({"source": source, "location": raw, "latitude": lat, "longitude": lon, "issue": error})
        df["LOCATION(DEGREES)"] = df["_source_file_start"].map(resolved)
        coords = df["LOCATION(DEGREES)"].map(self.loc.parse_lat_lon)
        df["latitude"], df["longitude"] = coords.map(lambda v: v[0]), coords.map(lambda v: v[1])
        missing = int(df.latitude.isna().sum())
        if missing:
            self.run.warn("MISSING_LOCATION", f"{surface}: {missing}/{len(df)} pairs lack valid coordinates.", missing)
        audit_path = self.out / "03_location_rate" / f"{surface}_locations.csv"
        write_csv(audit_path, pd.DataFrame(audit, columns=["source", "location", "latitude", "longitude", "issue"]))
        if missing and self.c["on_missing_location"] == "fail":
            raise ValueError(f"{surface}: missing location under on_missing_location=fail")
        df = self.rate.add_deterioration_rate(df)
        path = self.path("03_location_rate", surface)
        write_csv(path, df)
        return {"rows": len(df), "missing_location_rows": missing, "location_count": df[["latitude", "longitude"]].dropna().drop_duplicates().shape[0],
                "DetRatePerYear": numeric_stats(df.DetRatePerYear)}, [path, audit_path]

    def climate(self, surface):
        df = read_table(self.path("03_location_rate", surface))
        path = self.path("04_climate", surface)
        if self.c["skip_climate"] or df.empty:
            if self.c["skip_climate"]:
                self.run.warn("CLIMATE_SKIPPED", f"{surface}: climate enrichment was explicitly disabled.")
            write_csv(path, df)
            return {"rows": len(df), "enabled": False}, [path]
        if importlib.util.find_spec("cdsapi") is None:
            raise ValueError("Climate requires requirements-climate.txt and CDS credentials, or --skip-climate")
        if df.latitude.notna().sum() == 0:
            raise ValueError(f"{surface}: climate requested but no usable coordinates")
        start, end = pd.to_datetime(df.start_date), pd.to_datetime(df.end_date)
        outside = (start.dt.year < self.c["climate_start_year"]) | (end.dt.year > self.c["climate_end_year"])
        if outside.any():
            self.run.warn("CLIMATE_WINDOW_TRUNCATED", f"{surface}: {outside.sum()} intervals extend beyond the configured download years.", outside.sum())
        work = self.out / "04_climate" / f"{surface}_work"
        work.mkdir(parents=True, exist_ok=True)
        request = df.copy()
        request["Dates"] = self.rate.build_dates_column(df, "start_date", "end_date")
        request["_pipeline_row_id"] = np.arange(len(df))
        infile, outfile = work / "input.csv", work / "output.csv"
        write_csv(infile, request)
        self.run.process([sys.executable, "-u", "-m", "pavement_analytics.original.era5",
                          "--input", infile, "--output", outfile, "--lat-col", "latitude", "--lon-col", "longitude",
                          "--dates-col", "Dates", "--cache-dir", self.c["cache_dir"],
                          "--start-year", self.c["climate_start_year"], "--end-year", self.c["climate_end_year"],
                          "--construction-year-col", self.c["construction_year_col"], *self.c["climate_args"]])
        climate = read_table(outfile)
        ids = climate.get("_pipeline_row_id", pd.Series(dtype=float))
        if len(climate) != len(df) or ids.isna().any() or ids.duplicated().any() or set(ids) != set(range(len(df))):
            raise ValueError("Climate output changed row identity or row count")
        climate = climate.set_index("_pipeline_row_id").reindex(range(len(df)))
        added = [c for c in climate if c not in request]
        if not added:
            raise ValueError("Climate stage returned no climate columns")
        for col in added:
            df[col] = climate[col].to_numpy()
        source_cols = [c for c in added if c.startswith("DataSource")]
        if not source_cols:
            raise ValueError("Climate output has no DataSource coverage column")
        coverage = df[source_cols].notna().any(axis=1)
        if not coverage.any():
            raise ValueError("Climate returned no usable observations for any pair. See the climate log.")
        if not coverage.all():
            self.run.warn("CLIMATE_MISSING", f"{surface}: {(~coverage).sum()} pairs have no climate observations.", (~coverage).sum())
        missingness = pd.DataFrame({"column": added, "missing_rows": [df[c].isna().sum() for c in added],
                                    "missing_pct": [100 * df[c].isna().mean() for c in added]})
        sparse = missingness.loc[missingness.missing_pct >= 50]
        if len(sparse):
            self.run.warn("CLIMATE_SPARSE", f"{surface}: {len(sparse)} climate columns are at least 50% missing; inspect the coverage table.", len(sparse))
        snap = [c for c in added if c.startswith("SnapDistanceKm")]
        if snap and (pd.to_numeric(df[snap[0]], errors="coerce") > 0).any():
            self.run.warn("CLIMATE_GRID_SNAPPED", f"{surface}: some coordinates used nearby land grid cells; see SnapDistanceKm in curated data.")
        if df[source_cols[0]].eq("era5").any():
            self.run.warn("CLIMATE_FALLBACK", f"{surface}: some rows used the coarser ERA5 fallback.")
        coverage_path = self.out / "04_climate" / f"{surface}_coverage.csv"
        write_csv(path, df)
        write_csv(coverage_path, missingness)
        return {"rows": len(df), "enabled": True, "columns_added": len(added),
                "rows_with_climate": int(coverage.sum())}, [path, coverage_path]

    def preserve_family_observability(self, df, surface):
        """A union-table blank is zero only when the source exported the family."""
        records, target_counts = [], {}
        for when in ("start", "end"):
            schema_column = f"_pipeline_deduct_schema_{when}"
            if schema_column in df:
                schemas = df[schema_column]
                origin = "source_export"
            else:
                # Standalone stage callers may supply an already-paired table.
                # Full runs always capture per-source schemas before concatenation.
                headers = sorted(c[:-(len(when) + 1)] for c in df
                                 if c.endswith("_" + when) and RAW_DEDUCT_HEADER.fullmatch(c[:-(len(when) + 1)]))
                schemas = pd.Series(json.dumps(headers), index=df.index)
                origin = "paired_table_header_fallback"
                if len(df):
                    self.run.warn("DEDUCT_SCHEMA_FALLBACK", f"{surface}/{when}: source schema provenance is absent; only the supplied paired table's exported headers are known.")
            parsed = {}
            for value in schemas.unique():
                try:
                    headers = json.loads(value)
                    if not isinstance(headers, list) or any(not isinstance(c, str) or not RAW_DEDUCT_HEADER.fullmatch(c) for c in headers):
                        raise ValueError
                except (TypeError, ValueError):
                    raise ValueError(f"{surface}: invalid source deduct schema provenance") from None
                parsed[value] = headers
            source_column = f"_source_file_{when}"
            sources = df[source_column].fillna("<unknown>") if source_column in df else pd.Series("<unknown>", index=df.index)
            source_schemas = pd.DataFrame({"source": sources, "schema": schemas})
            for family, codes in sorted(self.families.items()):
                name = f"fam_{family} deduct_{when}"
                details = {schema: family_schema_details(headers, codes) for schema, headers in parsed.items()}
                observed = schemas.map({schema: item["coverage"] != "unobserved" for schema, item in details.items()}).fillna(False).astype(bool)
                partial = schemas.map({schema: item["coverage"] == "partial" for schema, item in details.items()}).fillna(False).astype(bool)
                if name in df:
                    df.loc[~observed, name] = np.nan
                    values = pd.to_numeric(df[name], errors="coerce")
                else:
                    values = pd.Series(np.nan, index=df.index)
                finite = np.isfinite(values)
                target_counts[name] = {"rows": len(df), "observed_rows": int(observed.sum()),
                                       "unobserved_rows": int((~observed).sum()), "partial_family_rows": int(partial.sum()),
                                       "finite_target_rows": int(finite.sum())}
                for (source, schema), subset in source_schemas.groupby(["source", "schema"], sort=True):
                    indices = subset.index
                    records.append({"source": source, "endpoint": when, "family": family, "target": name,
                                    "schema_origin": origin, **details[schema], "rows": len(subset),
                                    "observed_rows": int(observed.loc[indices].sum()),
                                    "unobserved_rows": int((~observed.loc[indices]).sum()),
                                    "finite_target_rows": int(finite.loc[indices].sum()),
                                    "partial_family_rows": int(partial.loc[indices].sum())})
        for family in self.families:
            delta, start, end = (f"fam_{family} deduct_{suffix}" for suffix in ("delta", "start", "end"))
            if delta in df and start in df and end in df:
                df[delta] = df[end] - df[start]
        columns = ["source", "endpoint", "family", "target", "schema_origin", "exported_channels", "observed_codes",
                   "missing_codes", "coverage", "rows", "observed_rows", "unobserved_rows", "finite_target_rows", "partial_family_rows"]
        path = self.out / "05_targets" / f"{surface}_family_target_coverage.csv"
        write_csv(path, pd.DataFrame(records, columns=columns))
        partial_rows = sum(v["partial_family_rows"] for v in target_counts.values())
        if partial_rows:
            self.run.warn("PARTIAL_FAMILY_EXPORT", f"{surface}: {partial_rows} family/endpoint observations include only some configured family codes; totals use the exported channels. See the family target coverage audit.", partial_rows)
        return target_counts, path

    def derive_targets(self, surface):
        df = read_table(self.path("04_climate", surface))
        path = self.path("05_targets", surface)
        raw = [c for c in df if self.distress.COL_RE.match(c)]
        if df.empty or self.c["skip_distress"] or not raw:
            if not raw and len(df):
                self.run.warn("NO_DISTRESS_COLUMNS", f"{surface}: no raw distress columns; pooled targets and PCI may be unavailable.")
            target_counts, coverage_path = self.preserve_family_observability(df, surface)
            write_csv(path, df)
            return {"rows": len(df), "derived_columns": 0, "family_target_coverage": target_counts}, [path, coverage_path]
        invalid_count = 0
        blank_count = 0
        invalid_rows = pd.Series(False, index=df.index)
        for col in raw:
            s = pd.to_numeric(df[col], errors="coerce")
            invalid = df[col].notna() & (~np.isfinite(s) | (s < 0) | ((s > 100) if " deduct_" in col else False))
            invalid_rows |= invalid
            invalid_count += int(invalid.sum())
            blank_count += int(df[col].isna().sum())
            df[col] = s.mask(invalid).fillna(0)
        # Preserve the established ambiguity quarantine: an ungraded channel
        # cannot be inferred to be additional damage or a severity aggregate.
        ambiguous = pd.Series(False, index=df.index)
        matched = {col: self.distress.COL_RE.match(col) for col in raw}
        for when in ("start", "end"):
            for code in sorted({m.group("code") for m in matched.values()}):
                ungraded = [col for col in raw if re.fullmatch(re.escape(code) + r" deduct_" + when, col)]
                graded = [col for col in raw if re.fullmatch(re.escape(code) + r"[LMH] deduct_" + when, col)]
                if ungraded and graded:
                    ambiguous |= df[ungraded].gt(0).any(axis=1) & df[graded].gt(0).any(axis=1)
        invalid_rows |= ambiguous
        if ambiguous.any():
            self.run.warn("AMBIGUOUS_DEDUCT_REPRESENTATION", f"{surface}: {ambiguous.sum()} pairs have simultaneous positive ungraded and graded channels for a code; these pairs are quarantined.", ambiguous.sum())
        # Blank distress cells follow the legacy structural-zero assumption; malformed values are quarantined.
        if blank_count:
            self.run.warn("DISTRESS_ZERO_FILLED", f"{surface}: {blank_count} blank distress cells zero-filled for the original PCI calculations. Family targets remain unobserved where the source exported no family deduct channel; PCI export-gap checks also apply.", blank_count)
        rejected = self.out / "05_targets" / f"{surface}_invalid_distress.csv"
        write_csv(rejected, read_table(self.path("04_climate", surface)).loc[invalid_rows])
        if invalid_rows.any():
            self.run.warn("INVALID_DISTRESS", f"{surface}: {invalid_count} malformed/negative/out-of-range distress cells; {invalid_rows.sum()} pairs quarantined.", invalid_rows.sum())
            df = df.loc[~invalid_rows].reset_index(drop=True)
        if df.empty:
            target_counts, coverage_path = self.preserve_family_observability(df, surface)
            write_csv(path, df)
            return {"rows": 0, "derived_columns": 0, "quarantined_pairs": int(invalid_rows.sum()),
                    "family_target_coverage": target_counts}, [path, rejected, coverage_path]
        meta = self.distress.parse_columns(df.columns)
        group = self.c["airfield_col"]
        if group not in df:
            raise ValueError(f"airfield_col {group!r} is absent")
        prevalence = self.distress.prevalence(df, meta, group)
        # Keep nonzero derived columns even when rare; min_groups controls warnings, not silent target removal.
        derived, dropped = self.distress.derive(df, meta, self.families, group, min_groups=1)
        restored = []
        # The original derivation drops all-zero columns. Keep exported zero
        # family deducts for an explicit constant-target result in the sweep.
        # A family with no exported deduct channels is still unobserved.
        for family, codes in self.families.items():
            for when in ("start", "end"):
                name = f"fam_{family} deduct_{when}"
                channels = meta.loc[meta.code.isin(codes) & meta.measure.eq("deduct") & meta.when.eq(when), "col"].tolist()
                if name not in derived and channels:
                    derived[name] = df[channels].sum(axis=1, min_count=1)
                    restored.append(name)
        target_counts, coverage_path = self.preserve_family_observability(derived, surface)
        prevpath = self.out / "05_targets" / f"{surface}_prevalence.csv"
        write_csv(prevpath, prevalence)
        write_csv(path, derived)
        if "n_groups_nonzero_end" in prevalence:
            rare = prevalence.n_groups_nonzero_end.between(1, self.c["min_groups"] - 1)
            if rare.any():
                self.run.warn("RARE_DISTRESS", f"{surface}: {rare.sum()} distress series occur in fewer than {self.c['min_groups']} airfields.", rare.sum())
        return {"rows": len(derived), "raw_distress_columns": len(raw), "blank_cells_zero_filled": blank_count,
                "quarantined_pairs": int(invalid_rows.sum()), "derived_columns": len(derived.columns)-len(df.columns),
                "all_zero_derived_columns_dropped": dropped,
                "zero_family_deduct_columns_restored_for_sweep": restored,
                "ambiguous_deduct_pairs": int(ambiguous.sum()),
                "family_target_coverage": target_counts}, [path, prevpath, rejected, coverage_path]

    def fit_curves(self):
        inputs = [self.path("05_targets", s) for s in self.c["surfaces"]]
        inputs = [p for p in inputs if not read_table(p).empty]
        path = self.out / "deduct_curves.json"
        if not inputs:
            raise ValueError("No paired rows are available to fit deduct curves")
        self.run.process([sys.executable, "-u", "-m", "pavement_analytics.original.fit_curves", *inputs, "--out", path])
        fitted = json.loads(path.read_text())
        count = sum(len(v) for v in fitted["curves"].values())
        if not count:
            raise ValueError("No deduct curves had enough samples to fit; disable fit_deduct_curves or supply a fitted file")
        return {"fitted_curves": count}, [path]

    def calculate_pci(self, surface):
        source = self.path("05_targets", surface)
        df = read_table(source)
        path = self.path("06_curated", surface)
        raw = [c for c in df if self.pci.DEDUCT_RE.match(c)]
        if self.c["skip_pci"] or df.empty or not raw:
            if not raw and len(df) and not self.c["skip_pci"]:
                self.run.warn("PCI_UNAVAILABLE", f"{surface}: no deduct columns, so recomputed PCI is unavailable.")
            write_csv(path, df)
            return {"rows": len(df), "calculated": False}, [path]
        # Validate deducts even when pooled-target derivation was skipped.
        values = df[raw].apply(pd.to_numeric, errors="coerce")
        bad = df[raw].notna() & (~np.isfinite(values) | (values < 0) | (values > 100))
        if bad.any().any():
            raise ValueError("Invalid deduct values found; enable distress preprocessing to quarantine affected pairs")
        command = [sys.executable, "-u", "-m", "pavement_analytics.original.pci", source, "--out", path.with_suffix(""),
                   "--curves", self.c["cdv_curves"], "--standard", self.c["standard"], "--rules", self.c["rules"], "--sev-mode", self.c["sev_mode"]]
        if self.c["no_shapley"]:
            command.append("--no-shapley")
        if self.c["families"]:
            command += ["--families", self.c["families"]]
        curves = self.out / "deduct_curves.json" if self.c["fit_deduct_curves"] else self.c["deduct_curves"]
        if curves:
            command += ["--deduct-curves", curves]
        self.run.process(command)
        out = read_table(path)
        if len(out) != len(df):
            raise ValueError("PCI output changed row count")
        for identity in ("_source_file_start", "_source_file_end", "_source_row_start", "_source_row_end"):
            if identity in df:
                if identity not in out or not out[identity].astype(str).eq(df[identity].astype(str)).all():
                    raise ValueError("PCI output changed row identity or row order")
        # The numerical worker is allowed to infer table dtypes. Preserve the
        # original pass-through fields so leading-zero identities cannot change.
        for column in df:
            out[column] = df[column].to_numpy()
        write_csv(path, out)
        stats = {"rows": len(out), "calculated": True, "standard": self.c["standard"], "rules": self.c["rules"], "severity_correction": bool(curves)}
        for when in ("start", "end"):
            key = f"pci_diff_{when}"
            if key not in out:
                continue
            d = pd.to_numeric(out[key], errors="coerce").dropna()
            stats[when] = {"count": len(d), "bias": d.mean(), "mae": d.abs().mean(), "within_2_pct": 100*d.abs().le(2).mean(), "max_error": d.abs().max()}
            gap = int(out[f"export_gap_{when}"].sum())
            stats[when]["export_gaps"] = gap
            if gap:
                self.run.warn("PCI_EXPORT_GAP", f"{surface}/{when}: {gap} rows have no deducts but recorded PCI <99.5; distress-derived targets will exclude these pairs.", gap)
            if len(d) and d.abs().mean() > self.c["pci_warning_mae"]:
                self.run.warn("PCI_DISAGREEMENT", f"{surface}/{when}: PCI MAE {d.abs().mean():.2f} exceeds {self.c['pci_warning_mae']:.2f} points.")
            shares = [c for c in out if c.startswith("loss_shap_") and c.endswith("_"+when)]
            if shares:
                error = (out[shares].sum(axis=1) - (100 - out[f"pci_calc_{when}"])).abs().max()
                stats[when]["shapley_max_additivity_error"] = error
                if error > 1e-6:
                    raise ValueError("Shapley contributions do not sum to recomputed PCI loss")
        return stats, [path]

    def target_names(self, surface):
        if self.c["target_mode"] == "legacy":
            return list(self.c["targets"].get(surface, ["DetRatePerYear"]))
        return [f"fam_{family} deduct_end" for family in sorted(self.families)]

    def modeling(self, surface):
        df = read_table(self.path("06_curated", surface))
        names = self.target_names(surface)
        if self.c["target_mode"] == "deduct_families" and surface in self.c["targets"] and self.c["targets"][surface] != names:
            self.run.warn("AUTOTARGET_OVERRIDE", f"{surface}: the legacy target list was replaced by the requested deduct-family total sweep.")
        path = self.path("07_modeling", surface)
        auditpath = self.out / "07_modeling" / f"{surface}_feature_audit.csv"
        rejectedpath = self.out / "07_modeling" / f"{surface}_rejected_rows.csv"
        eligibilitypath = self.out / "07_modeling" / f"{surface}_target_eligibility.csv"
        for name in names:
            if name not in df:
                if self.c["target_mode"] == "legacy" and surface in self.c["targets"] and len(df):
                    raise ValueError(f"{surface}: explicitly requested target {name!r} is absent")
                df[name] = np.nan
        if df.empty:
            write_csv(path, pd.DataFrame(columns=[*names, "delta_days", "partition_airfield", "partition_section", "_pipeline_model_row"]))
            write_csv(auditpath, pd.DataFrame([{"column": name, "disposition": "TARGET", "reason": "declared family target"} for name in names]))
            write_csv(rejectedpath, pd.DataFrame(columns=["exclusion_reason"]))
            write_csv(eligibilitypath, pd.DataFrame(columns=["source_row", "target", "eligible", "reason"]))
            return {"rows": 0, "features": 0, "trainable": False,
                    "targets": {name: {"count": 0, "distinct": 0} for name in names},
                    "target_trainable": {name: False for name in names}}, [path, auditpath, rejectedpath, eligibilitypath]
        for col in ["delta_days", self.c["airfield_col"], "SectionUniqueID"]:
            if col not in df:
                raise ValueError(f"{surface}: required model column {col!r} absent")
        if {"partition_airfield", "partition_section", "partition_fold", "_pipeline_model_row"} & set(df):
            raise ValueError("Reserved partition/row columns already exist in curated input")
        y = df[names].apply(pd.to_numeric, errors="coerce")
        weight = pd.to_numeric(df.delta_days, errors="coerce")
        masks = {
            "invalid_interval": ~np.isfinite(weight) | (weight < self.c["min_delta_days"]),
            "missing_partition": df[self.c["airfield_col"]].isna() | df[self.c["airfield_col"]].astype("string").str.strip().eq(""),
        }
        if self.c["max_delta_days"] is not None:
            masks["interval_above_maximum"] = weight > self.c["max_delta_days"]
        reason = pd.Series("", index=df.index)
        for label, mask in masks.items():
            reason.loc[mask.fillna(True) & reason.eq("")] = label
        eligibility = []
        gapcols = [c for c in ("export_gap_start", "export_gap_end") if c in df]
        for name in names:
            target_reason = reason.copy()
            target_reason.loc[~np.isfinite(y[name]) & target_reason.eq("")] = "invalid_or_unobserved_target"
            distress_target = name.startswith(("loss_", "pci_only_", "pci_calc_", "fam_")) or bool(re.match(r"^\d{2}[LMH]? ", name))
            if distress_target:
                if not gapcols:
                    raise ValueError("Distress-derived targets require PCI export-gap checks; enable PCI calculation")
                gap = df[gapcols].eq(1).any(axis=1)
                target_reason.loc[gap & target_reason.eq("")] = "pci_export_gap"
            y.loc[target_reason.ne(""), name] = np.nan
            eligibility.extend({"source_row": int(index), "target": name, "eligible": not why,
                                "reason": why or "eligible"} for index, why in target_reason.items())
        write_csv(eligibilitypath, pd.DataFrame(eligibility))
        rejected = df.loc[reason.ne("")].copy()
        rejected["exclusion_reason"] = reason[reason.ne("")]
        write_csv(rejectedpath, rejected)
        if len(rejected):
            self.run.warn("MODEL_ROWS_EXCLUDED", f"{surface}: {len(rejected)} rows excluded for shared interval/group validity; target-specific exclusions are separate.", len(rejected))
        df[names], df["delta_days"] = y, weight
        usable = df.loc[reason.eq("")].copy()
        audit = self.prep.classify(usable, names[0], "delta_days", self.c["airfield_col"], "SectionUniqueID",
                                   self.c["keep_latlon"], self.c["keep_distress"], self.c["high_card_threshold"], set(self.c["extra_drop"]))
        audit.loc[audit.column.isin(names), ["disposition", "reason"]] = ["TARGET", "declared target; excluded from every target's feature list"]
        keys = audit.column.map(normalized_column_stem)
        predictor_role = ~audit.disposition.isin(["TARGET", "WEIGHT", "PARTITION"])
        identities = keys.isin(IDENTIFIER_KEYS) & predictor_role
        dates = keys.str.endswith(("date", "datetime")) & predictor_role
        audit.loc[identities, ["disposition", "reason"]] = ["DROP", "normalized_location_identifier"]
        audit.loc[dates, ["disposition", "reason"]] = ["DROP", "normalized_raw_date"]
        if self.c["fit_deduct_curves"]:
            # Preserve the original all-row calibration and its curated outputs,
            # but never feed its learned transformations into held-out models.
            # Severity correction also changes the CDV diagnostics, not just PCI.
            fitted_output = audit.column.str.lower().str.match(
                r"^(?:pci_calc|pci_only|pci_diff|loss_shap|loss_loo|tdv|hdv|n_deduct|q0|m|maxcdv|sev_corr_gain|sev_corr_codes|surface_used)_")
            fitted_output &= ~audit.disposition.eq("TARGET")
            admitted = int((fitted_output & audit.disposition.eq("KEEP")).sum())
            audit.loc[fitted_output, ["disposition", "reason"]] = ["DROP", "run_fitted_curve_uses_all_rows"]
            self.run.warn("RUN_FITTED_CURVE_FEATURES_EXCLUDED",
                          f"{surface}: {int(fitted_output.sum())} calibration-derived columns are excluded from predictors; {admitted} would otherwise have been retained. The fitted curves use all curated rows, including future holdout groups.",
                          int(fitted_output.sum()))
        columns = audit.loc[audit.disposition.isin(["KEEP", "TARGET", "WEIGHT", "PARTITION"]), "column"].tolist()
        model = usable[columns].rename(columns={self.c["airfield_col"]: "partition_airfield", "SectionUniqueID": "partition_section"})
        model["partition_airfield"] = model.partition_airfield.astype("string")
        model["partition_section"] = [json.dumps([str(a), str(section)], ensure_ascii=False)
                                      for a, section in zip(model.partition_airfield, model.partition_section)]
        model["_pipeline_model_row"] = usable.index.to_numpy()
        audit = pd.concat([audit, pd.DataFrame([{"column": "_pipeline_model_row", "disposition": "PARTITION", "reason": "curated row identity only; never a predictor"}])], ignore_index=True)
        write_csv(path, model)
        write_csv(auditpath, audit)
        features = audit.loc[audit.disposition.eq("KEEP"), "column"].tolist()
        group_count = model.partition_airfield.nunique()
        required_groups = self.c["dr_cv_reps"] + (1 if self.c["dr_holdout_pct"] > 0 else 0)
        if group_count < required_groups:
            self.run.warn("TOO_FEW_GROUPS", f"{surface}: {group_count} groups; configured validation needs at least {required_groups}.")
        if not features:
            self.run.warn("NO_FEATURES", f"{surface}: no varying predictor columns remain.")
        if len(model) < 100:
            self.run.warn("SMALL_MODEL_DATASET", f"{surface}: only {len(model)} usable modeling rows.", len(model))
        target_stats, target_trainable = {}, {}
        for name in names:
            target_stats[name] = stats = numeric_stats(model[name])
            finite = np.isfinite(model[name])
            stats["valid_groups"] = int(model.loc[finite, "partition_airfield"].nunique())
            stats["nonzero_groups"] = int(model.loc[finite & model[name].ne(0), "partition_airfield"].nunique())
            target_trainable[name] = bool(features and stats["distinct"] >= 2 and stats["valid_groups"] >= required_groups)
            if stats["distinct"] < 2:
                self.run.warn("CONSTANT_OR_UNOBSERVED_TARGET", f"{surface}: {name!r} has fewer than two observed values; it remains an explicit sweep task.")
            if stats["zero_pct"] is not None and stats["zero_pct"] >= 95:
                self.run.warn("SPARSE_TARGET", f"{surface}: {name!r} is {stats['zero_pct']:.1f}% zero.")
        profile = pd.DataFrame([{"column": col, "role": "target" if col in names else "feature",
                                 "missing_pct": 100*model[col].isna().mean(), "distinct": model[col].nunique()} for col in [*names, *features]])
        profilepath = self.out / "07_modeling" / f"{surface}_column_profile.csv"
        write_csv(profilepath, profile)
        sparse = profile.loc[profile.role.eq("feature") & profile.missing_pct.ge(50)]
        if len(sparse):
            self.run.warn("SPARSE_FEATURES", f"{surface}: {len(sparse)} retained features are at least 50% missing.", len(sparse))
        return {"input_rows": len(df), "rows": len(model), "rows_excluded": len(rejected),
                "exclusion_reasons": reason[reason.ne("")].value_counts().to_dict(),
                "features": len(features), "airfield_groups": group_count, "targets": target_stats,
                "target_trainable": target_trainable, "trainable": any(target_trainable.values())}, [path, auditpath, rejectedpath, profilepath, eligibilitypath]

    def partitions(self):
        """Freeze one plan before target filtering and local model fitting."""
        from ..study_modeling import make_partitions
        frames, group_parent = {}, {}
        locations = {}
        for surface in self.c["surfaces"]:
            table = read_table(self.path("07_modeling", surface))
            frames[surface] = table
            for group in table.partition_airfield.dropna().astype(str):
                group_parent[group] = group
            curated = read_table(self.path("06_curated", surface))
            groupcol = self.c["airfield_col"]
            if {groupcol, "latitude", "longitude"} <= set(curated):
                coords = curated[[groupcol, "latitude", "longitude"]].dropna()
                for group, latitude, longitude in coords.itertuples(index=False, name=None):
                    if str(group) not in group_parent:
                        continue
                    try:
                        lat, lon = float(latitude), float(longitude)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite([lat, lon]).all() and -90 <= lat <= 90 and -180 <= lon <= 180:
                        point = (lat, ((lon + 180) % 360) - 180)
                        locations.setdefault(point, set()).add(str(group))
        def find(group):
            while group_parent[group] != group:
                group_parent[group] = group_parent[group_parent[group]]
                group = group_parent[group]
            return group
        for groups in locations.values():
            members = sorted(groups)
            for group in members[1:]:
                group_parent[find(group)] = find(members[0])
        combined = []
        for surface, frame in frames.items():
            frame["partition_original_airfield"] = frame.partition_airfield.astype("string")
            frame["partition_airfield"] = frame.partition_airfield.astype(str).map(find)
            combined.append(pd.DataFrame({"group_id": frame.partition_airfield}))
        if not combined:
            combined = [pd.DataFrame(columns=["group_id"])]
        groups = pd.concat(combined, ignore_index=True)
        fraction = self.c["dr_holdout_pct"] / 100
        if not 0 < fraction < .5:
            raise ValueError("Shared target-sweep partitions require 0 < dr_holdout_pct < 50")
        plan = make_partitions(groups, folds=self.c["dr_cv_reps"], seed=42, holdout_fraction=fraction)
        path = self.out / "07_modeling" / "partitions.csv"
        write_csv(path, plan)
        artifacts = [path]
        assignments = plan.set_index("group_id").fold.to_dict()
        readiness = {}
        for surface, frame in frames.items():
            frame["partition_fold"] = frame.partition_airfield.map(assignments).astype(int)
            audit = read_table(self.out / "07_modeling" / f"{surface}_feature_audit.csv")
            for name in ("partition_original_airfield", "partition_fold"):
                audit = pd.concat([audit, pd.DataFrame([{"column": name, "disposition": "PARTITION", "reason": "shared validation metadata; never a predictor"}])], ignore_index=True)
            output = self.out / "07_modeling" / "partitioned" / f"{surface}.csv"
            auditpath = output.with_name(f"{surface}_feature_audit.csv")
            write_csv(output, frame)
            write_csv(auditpath, audit)
            artifacts.extend([output, auditpath])
            readiness[surface] = {}
            feature_columns = audit.loc[audit.disposition.eq("KEEP"), "column"].tolist()
            for target in self.target_names(surface):
                valid = frame.loc[np.isfinite(pd.to_numeric(frame[target], errors="coerce"))]
                train = valid.loc[valid.partition_fold.ne(-1)]
                hold = valid.loc[valid.partition_fold.eq(-1)]
                folds_present = set(train.partition_fold.unique())
                varying = any(train[feature].nunique(dropna=True) > 1 for feature in feature_columns)
                ready = bool(varying and len(hold) and train[target].nunique() >= 2
                             and train.partition_airfield.nunique() >= self.c["dr_cv_reps"]
                             and folds_present == set(range(self.c["dr_cv_reps"])))
                reason = ("no_valid_target" if valid.empty else "constant_target" if train[target].nunique() < 2
                          else "no_varying_features" if not varying else "missing_holdout" if hold.empty
                          else "insufficient_fold_coverage" if folds_present != set(range(self.c["dr_cv_reps"]))
                          else "insufficient_groups" if train.partition_airfield.nunique() < self.c["dr_cv_reps"]
                          else "ready")
                readiness[surface][target] = {"trainable": ready, "status": reason, "development_rows": len(train),
                                               "holdout_rows": len(hold), "development_groups": int(train.partition_airfield.nunique()),
                                               "holdout_groups": int(hold.partition_airfield.nunique())}
        auditpath = self.out / "07_modeling" / "partition_audit.json"
        payload = {"declared_groups": len(group_parent), "validation_groups": len(plan),
                   "grouping": "declared groups joined at identical input coordinates",
                   "grid_limitation": "Nearby coordinates snapping to one climate grid cell are not necessarily joined by this exact-coordinate rule.",
                   "seed": 42, "folds": self.c["dr_cv_reps"], "holdout_fraction": fraction,
                   "target_readiness": readiness}
        write_json(auditpath, payload)
        return payload, [*artifacts, auditpath]

    def execute(self):
        self.run.stage("01_split", self.ingest)
        for surface in self.c["surfaces"]:
            for stage, func in [("02_pairs", self.pair), ("03_location_rate", self.location_rate),
                                ("04_climate", self.climate), ("05_targets", self.derive_targets)]:
                self.run.stage(f"{stage}_{surface}", lambda f=func, s=surface: f(s))
        if self.c["fit_deduct_curves"] and not self.c["skip_pci"]:
            self.run.stage("05_fit_curves", self.fit_curves)
        else:
            self.run.stage("05_fit_curves", lambda: ({"skipped": "optional fitting disabled; supplied severity curves are used when configured"}, []))
        for surface in self.c["surfaces"]:
            self.run.stage(f"06_pci_{surface}", lambda s=surface: self.calculate_pci(s))
            self.run.stage(f"07_modeling_{surface}", lambda s=surface: self.modeling(s))
        self.run.stage("07_partitions", self.partitions)
        total = sum(self.run.data["stages"][f"07_modeling_{s}"]["stats"]["rows"] for s in self.c["surfaces"])
        if not total:
            raise ValueError("No usable modeling rows across the selected surfaces. Review rejected rows and pairing summaries.")
        if self.c["fail_on_warning"] and self.run.data["warnings"]:
            return 2
        def local_backend():
            requested = self.c.get("retired_backend_request")
            if requested:
                self.run.warn("RETIRED_BACKEND_IGNORED", "The legacy remote-training preference was ignored; the full command uses local models.")
            return {"backend": "local", "retired_backend": "DataRobot", "remote_training": False,
                    "retired_backend_request": requested,
                    "next_stage": "local target sweep"}, []
        self.run.stage("08_model_backend", local_backend)
        return 2 if self.c["fail_on_warning"] and self.run.data["warnings"] else 0


def main(argv=None):
    run = None
    try:
        refuse_public_execution()
        cfg, resume = parse_config(argv)
        validate_config(cfg)
        files, mapping, inputs, fingerprint = discover(cfg)
        run = Run(cfg["output_dir"], fingerprint, cfg, inputs, resume)
        run.data["config"] = cfg
        code = Pipeline(cfg, run, files, mapping).execute()
        run.finish("quality_gate_failed" if code == 2 else ("completed_with_warnings" if run.data["warnings"] else "completed"))
        return code
    except KeyboardInterrupt:
        if run:
            run.finish("interrupted", "Interrupted; use --resume to continue")
        return 130
    except Exception as exc:
        if run:
            run.finish("failed", str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
