"""Prepare a private, reproducible family analysis from a single data path."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import uuid

from .io import read_json, read_table, sha256, write_json
from .security import Boundary, SafeError, disjoint, reject_links

FORMATS = {".csv", ".tsv", ".xlsx"}
ALIASES = {
    "section": {"section", "sectionid", "sectionuniqueid"},
    "date": {"date", "inspectiondate", "surveydate"},
    "pci": {"pci", "pavementconditionindex"},
    "surface": {"surface", "surfacetype", "inspectedsurfacetype", "material"},
    "group": {"group", "groupid", "network", "networkid"},
    "construction": {"construction", "constructionflag"},
}
OPTION_KEYS = {"columns", "features", "deduct_families", "date_format", "cutoff_year",
               "min_delta_days", "max_delta_days", "folds", "seed", "holdout_fraction",
               "association_policy", "models"}
DEDUCT_HEADER = re.compile(r"^(\d{2})\s*([LMH])\s*deduct$", re.I)


def new_workspace(explicit: Path | None = None) -> Path:
    """An explicit destination must be empty; defaults never overwrite old runs."""
    workspace = explicit if explicit is not None else Path.home() / "pavement-runs" / (
        "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    reject_links(workspace)
    workspace = workspace.expanduser().absolute()
    if workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir())):
        raise SafeError("ANALYZE_REQUIRES_EMPTY_WORKSPACE")
    return workspace


def external_path(value: Path, public: Path) -> Path:
    reject_links(value)
    result = value.expanduser().resolve()
    if not disjoint(result, public.resolve()):
        raise SafeError("ANALYZE_SOURCE_MUST_BE_OUTSIDE_PUBLIC_TREE")
    return result


def input_files(data: Path, public: Path, workspace: Path) -> list[Path]:
    data = external_path(data, public)
    if not disjoint(data, workspace.resolve()):
        raise SafeError("ANALYZE_SOURCE_AND_WORKSPACE_MUST_BE_DISJOINT")
    if data.is_file():
        if data.suffix.lower() not in FORMATS:
            raise SafeError("UNSUPPORTED_INPUT_FORMAT")
        paths = [data]
    elif data.is_dir():
        # Freeze just this directory's supported files, never recurse into a checkout.
        paths = sorted(p for p in data.iterdir() if p.suffix.lower() in FORMATS and not p.name.startswith("~$"))
        if not paths:
            raise SafeError("NO_INPUT_FILES")
    else:
        raise SafeError("INPUT_PATH_NOT_FOUND")
    for path in paths:
        reject_links(path)
        if not path.is_file():
            raise SafeError("INPUT_MUST_BE_REGULAR_FILE")
    return paths


def column_map(headers: list[str], overrides: dict) -> dict[str, str]:
    if not isinstance(overrides, dict) or set(overrides) - set(ALIASES):
        raise SafeError("INVALID_COLUMN_MAPPING")
    found = {}
    for logical, names in ALIASES.items():
        if logical in overrides:
            source = overrides[logical]
            if not isinstance(source, str) or source not in headers:
                raise SafeError("REQUIRED_INPUT_COLUMNS_MISSING")
            found[logical] = source
            continue
        matches = [h for h in headers if re.sub(r"[^a-z0-9]", "", h.lower()) in names]
        if len(matches) > 1:
            raise SafeError("AMBIGUOUS_COLUMNS_USE_PRIVATE_OPTIONS")
        if matches:
            found[logical] = matches[0]
    if not {"section", "date", "pci", "surface"} <= set(found):
        raise SafeError("UNRECOGNIZED_COLUMNS_USE_PRIVATE_OPTIONS")
    if len(set(found.values())) != len(found):
        raise SafeError("DUPLICATE_COLUMN_MAPPING")
    return found


def prepare(boundary: Boundary, data: Path, options_path: Path | None = None,
            *, by_code: bool = False) -> Path:
    """Snapshot sources and write normalized inputs/config; originals stay intact."""
    paths = input_files(data, boundary.public, boundary.workspace)
    options = {}
    options_hash = None
    if options_path is not None:
        options_path = external_path(options_path, boundary.public)
        options_hash = sha256(options_path)
        options = read_json(options_path)
        if set(options) - OPTION_KEYS:
            raise SafeError("UNKNOWN_ANALYZE_OPTION")
        if sha256(options_path) != options_hash:
            raise SafeError("INPUT_CHANGED_DURING_READ")
    if by_code and "deduct_families" in options:
        raise SafeError("FAMILY_OPTIONS_CONFLICT")
    original_dir = boundary.output("originals")
    normalized_dir = boundary.output("inputs")
    original_dir.mkdir(exist_ok=False)
    normalized_dir.mkdir(exist_ok=False)
    report = {"schema_version": 1, "sources": [], "grouping": [],
              "options_sha256": options_hash, "feature_policy": "Explicit numeric start-state features only",
              "directory_policy": "Supported files directly in the supplied directory; no recursion"}
    if options_path is not None:
        # Preserve exact user options privately, including any feature/family choices.
        shutil.copyfile(options_path, boundary.output("analysis_options.json"))
        if sha256(boundary.output("analysis_options.json")) != options_hash:
            raise SafeError("INPUT_CHANGED_DURING_READ")
    codes = set()
    group_strategy = None
    normalized_names = []
    for index, original in enumerate(paths, 1):
        original_hash = sha256(original)
        snapshot = original_dir / f"input_{index:04d}{original.suffix.lower()}"
        shutil.copyfile(original, snapshot)
        if sha256(original) != original_hash or sha256(snapshot) != original_hash:
            raise SafeError("INPUT_CHANGED_DURING_READ")
        frame = read_table(snapshot)
        # Headers/paths are intentionally confined to the private log/report.
        print({"source": str(original), "headers": list(frame.columns)})
        mapping = column_map(list(frame.columns), options.get("columns", {}))
        rename = {source: logical for logical, source in mapping.items()}
        for header in frame.columns:
            match = DEDUCT_HEADER.fullmatch(header)
            if match:
                code, severity = match.groups()
                rename[header] = code + severity.upper() + " deduct"
                codes.add(code)
        normalized_headers = [rename.get(h, h) for h in frame.columns]
        if len(set(normalized_headers)) != len(normalized_headers):
            raise SafeError("NORMALIZED_HEADERS_COLLIDE")
        frame = frame.rename(columns=rename)
        group_source = mapping.get("group")
        strategy = "declared_group" if group_source else "section_identity"
        if group_strategy is not None and group_strategy != strategy:
            raise SafeError("MIXED_GROUPING_REQUIRES_PRIVATE_RECONCILIATION")
        group_strategy = strategy
        if group_source is None:
            if "group" in frame:
                raise SafeError("NORMALIZED_HEADERS_COLLIDE")
            # Shared IDs across files remain in the same split, avoiding leakage.
            frame["group"] = frame["section"]
        if "construction" not in frame:
            frame["construction"] = "no"
        report["grouping"].append({"input": snapshot.name, "source_column": group_source,
                                   "strategy": "declared_group" if group_source else "section_identity",
                                   "limitation": None if group_source else
                                   "Section-disjoint evaluation only; broader group isolation is not established."})
        filename = f"input_{index:04d}.csv"
        normalized = normalized_dir / filename
        frame.to_csv(normalized, index=False)
        normalized_names.append(filename)
        report["sources"].append({"original_path": str(original), "original_sha256": original_hash,
                                   "snapshot": str(snapshot.relative_to(boundary.workspace)),
                                   "normalized": "inputs/" + filename, "normalized_sha256": sha256(normalized),
                                   "column_mapping": mapping, "deduct_header_mapping": {
                                       key: value for key, value in rename.items() if DEDUCT_HEADER.fullmatch(key)},
                                   "observed_deduct_columns": sorted(h for h in frame.columns if DEDUCT_HEADER.fullmatch(h)),
                                   "rows": len(frame)})
    if not codes:
        raise SafeError("NO_RECOGNIZED_DEDUCT_COLUMNS")
    if by_code:
        families = {"deduct_" + code: [code] for code in sorted(codes)}
    else:
        from .families import default_families
        families = default_families(codes)
    config = {"schema_version": 1, "input_root": "inputs", "input_files": normalized_names,
              "output": "results", "setup_report": "setup_report.json", "columns": {key: key for key in ALIASES},
              "features": ["PCI_start"], "deduct_families": families,
              "models": True, "min_delta_days": 365, "folds": 5, "seed": 42,
              "holdout_fraction": 0.2, "date_format": "%Y-%m-%d", "association_policy": "fail"}
    config.update({key: value for key, value in options.items() if key != "columns"})
    report["deduct_codes"] = sorted(codes)
    report["deduct_families"] = config["deduct_families"]
    report["features"] = config["features"]
    report["date_format"] = config["date_format"]
    report["coverage_warning"] = (
        "Family totals use exported deduct columns only. A missing code/severity column is not a confirmed zero. "
        "Exports may cover different columns; inspect observed_deduct_columns for every source before comparing results.")
    report["feature_note"] = "Default PCI_start only; add scientifically justified features through private options."
    write_json(boundary.output("setup_report.json"), report)
    destination = boundary.output("run.json")
    write_json(destination, config)
    return destination


def analyze(boundary: Boundary, data: Path, options_path: Path | None = None,
            *, by_code: bool = False) -> str:
    from .pipeline import run
    config = prepare(boundary, data, options_path, by_code=by_code)
    return run(boundary, config)
