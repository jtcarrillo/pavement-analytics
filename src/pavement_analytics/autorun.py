"""Prepare a private, reproducible family analysis from a single data path."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import uuid

from .io import read_json, read_table, sha256, write_json
from .dates import normalize_dates
from .security import Boundary, SafeError, disjoint, reject_links

FORMATS = {".csv", ".tsv", ".xlsx"}
ALIASES = {
    "section": {"section", "sectionid", "sectionuniqueid"},
    "date": {"date", "inspectiondate", "surveydate"},
    "pci": {"pci", "pavementconditionindex"},
    "surface": {"surface", "surfacetype", "inspectedsurfacetype", "material"},
    "group": {"group", "groupid", "network", "networkid"},
    "construction": {"construction", "constructionflag", "isconstruction"},
}
OPTION_KEYS = {"columns", "features", "deduct_families", "date_format", "cutoff_year",
               "min_delta_days", "max_delta_days", "folds", "seed", "holdout_fraction",
               "association_policy", "models", "blank_deducts"}
DEDUCT_HEADER = re.compile(r"^(\d{2})\s*([LMH]?)\s*deduct$", re.I)


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


def _header_key(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", header.lower())


def column_map(headers: list[str], overrides: dict, *, diagnostics: dict | None = None) -> dict[str, str]:
    """Prefer known schema fields; never break an unresolved tie by column order."""
    audit = diagnostics if diagnostics is not None else {}
    candidates = {role: sorted((h for h in headers if _header_key(h) in names),
                              key=lambda h: (h.casefold(), h)) for role, names in ALIASES.items()}
    paver = (all(sum(_header_key(h) == token for h in headers) == 1 for token in
                 ["sectionuniqueid", "date", "pci", "inspectedsurfacetype"])
             and any(DEDUCT_HEADER.fullmatch(h) for h in headers))
    audit.update(profile="paver" if paver else "generic", roles={
        role: {"candidates": matches, "selected": None, "selection": "absent"}
        for role, matches in candidates.items()})
    if not isinstance(overrides, dict) or set(overrides) - set(ALIASES):
        raise SafeError("INVALID_COLUMN_MAPPING")
    found = {}
    ambiguous = False
    for logical in ALIASES:
        decision = audit["roles"][logical]
        if logical in overrides:
            source = overrides[logical]
            if not isinstance(source, str) or source not in headers:
                raise SafeError("REQUIRED_INPUT_COLUMNS_MISSING")
            found[logical] = source
            decision.update(selected=source, selection="explicit_override")
            continue
        matches = candidates[logical]
        if len(matches) > 1:
            preferred = {"section": ["sectionuniqueid"], "surface": ["inspectedsurfacetype"]}.get(logical, [])
            if logical in {"date", "pci"} and paver:
                preferred = [logical]
            if logical == "group":
                # A literal group_id declares the pipeline grouping. A network ID
                # outranks a display name only for a recognized PAVER core schema.
                preferred = (["groupid"] if any(h.casefold() == "group_id" for h in matches) else [])
                if paver:
                    preferred += ["networkid"]
            selected = []
            for token in preferred:
                selected = [h for h in matches if _header_key(h) == token]
                if selected:
                    break
            if len(selected) == 1:
                found[logical] = selected[0]
                decision.update(selected=selected[0], selection="canonical_precedence")
            else:
                decision["selection"] = "ambiguous"
                ambiguous = True
            continue
        if matches:
            found[logical] = matches[0]
            decision.update(selected=matches[0], selection="only_candidate")
    if ambiguous:
        raise SafeError("AMBIGUOUS_COLUMNS_USE_PRIVATE_OPTIONS")
    if not {"section", "date", "pci", "surface"} <= set(found):
        raise SafeError("UNRECOGNIZED_COLUMNS_USE_PRIVATE_OPTIONS")
    if len(set(found.values())) != len(found):
        raise SafeError("DUPLICATE_COLUMN_MAPPING")
    return found


def normalization_map(headers: list[str], mapping: dict[str, str]) -> dict[str, str]:
    """Preserve secondary source fields that would collide with canonical roles."""
    rename = {source: logical for logical, source in mapping.items()}
    for header in headers:
        match = DEDUCT_HEADER.fullmatch(header)
        if match:
            code, severity = match.groups()
            rename[header] = code + severity.upper() + " deduct"
    # Duplicate deduct channels are a real ambiguity, not a preservation case.
    if len(set(rename.values())) != len(rename):
        raise SafeError("NORMALIZED_HEADERS_COLLIDE")
    used = set(headers) | set(rename.values())
    for header in sorted(headers):
        if header in rename or header not in rename.values():
            continue
        base = "source_extra_" + header
        candidate, suffix = base, 2
        while candidate in used:
            candidate = base + "_" + str(suffix)
            suffix += 1
        rename[header] = candidate
        used.add(candidate)
    return rename


def prepare(boundary: Boundary, data: Path, options_path: Path | None = None,
            *, by_code: bool = False, date_format: str | None = None,
            blank_deducts: str | None = None, exclude_files: set[Path] | None = None) -> Path:
    """Snapshot sources and write normalized inputs/config; originals stay intact."""
    paths = input_files(data, boundary.public, boundary.workspace)
    if exclude_files:
        paths = [p for p in paths if p.resolve() not in exclude_files]
        if not paths:
            raise SafeError("NO_INSPECTION_INPUT_FILES")
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
    if date_format is not None:
        if "date_format" in options and options["date_format"] != date_format:
            raise SafeError("DATE_FORMAT_OPTIONS_CONFLICT")
        options["date_format"] = date_format
    if "date_format" in options and (not isinstance(options["date_format"], str)
                                       or not options["date_format"].strip()):
        raise SafeError("INVALID_DATE_FORMAT")
    if blank_deducts is not None:
        if "blank_deducts" in options and options["blank_deducts"] != blank_deducts:
            raise SafeError("BLANK_DEDUCT_OPTIONS_CONFLICT")
        options["blank_deducts"] = blank_deducts
    if options.get("blank_deducts", "missing") not in ("missing", "zero"):
        raise SafeError("INVALID_BLANK_DEDUCT_POLICY")
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
    mapping_report = {"schema_version": 1, "sources": [],
                      "resolution": "For unresolved roles, set columns in a private --options JSON to exact candidate headers."}
    date_report = {"schema_version": 1, "sources": [],
                   "resolution": "Use --date-format with an explicit strptime format if automatic parsing cannot resolve the source."}
    normalized_names = []
    for index, original in enumerate(paths, 1):
        original_hash = sha256(original)
        snapshot = original_dir / f"input_{index:04d}{original.suffix.lower()}"
        shutil.copyfile(original, snapshot)
        if sha256(original) != original_hash or sha256(snapshot) != original_hash:
            raise SafeError("INPUT_CHANGED_DURING_READ")
        frame = read_table(snapshot)
        source_headers = list(frame.columns)
        # Headers/paths are intentionally confined to the private log/report.
        print({"source": str(original), "headers": list(frame.columns)})
        diagnostic = {"input": str(snapshot.relative_to(boundary.workspace)), "original_path": str(original)}
        mapping_report["sources"].append(diagnostic)
        try:
            mapping = column_map(list(frame.columns), options.get("columns", {}), diagnostics=diagnostic)
        except SafeError as error:
            diagnostic.update(status="failed", error_code=str(error))
            diagnostic["override_template"] = {"columns": {
                role: None for role, choice in diagnostic.get("roles", {}).items()
                if choice["selection"] == "ambiguous"}}
            write_json(boundary.output("column_mapping_report.json"), mapping_report)
            raise
        diagnostic.update(status="selected", error_code=None)
        write_json(boundary.output("column_mapping_report.json"), mapping_report)
        rename = normalization_map(list(frame.columns), mapping)
        for header in frame.columns:
            match = DEDUCT_HEADER.fullmatch(header)
            if match:
                code, severity = match.groups()
                codes.add(code)
        normalized_headers = [rename.get(h, h) for h in frame.columns]
        if len(set(normalized_headers)) != len(normalized_headers):
            raise SafeError("NORMALIZED_HEADERS_COLLIDE")
        frame = frame.rename(columns=rename)
        date_diagnostic = {"input": str(snapshot.relative_to(boundary.workspace)),
                           "original_column": mapping["date"],
                           "mode": "explicit" if "date_format" in options else "automatic"}
        date_report["sources"].append(date_diagnostic)
        if "date_format" in options:
            # Keep the legacy explicit-format contract: parse these values during curation.
            date_diagnostic.update(format=options["date_format"], status="deferred_to_curation")
        else:
            original_date, suffix = "source_extra_date", 2
            while original_date in frame:
                original_date = "source_extra_date_" + str(suffix)
                suffix += 1
            frame[original_date] = frame["date"]
            date_diagnostic["preservation_column"] = original_date
            try:
                frame["date"] = normalize_dates(frame["date"], diagnostics=date_diagnostic)
            except SafeError as error:
                date_diagnostic.update(status="failed", error_code=str(error))
                write_json(boundary.output("date_parsing_report.json"), date_report)
                raise
            date_diagnostic.update(status="normalized", error_code=None)
        write_json(boundary.output("date_parsing_report.json"), date_report)
        group_source = mapping.get("group")
        strategy = "declared_group:" + _header_key(group_source) if group_source else "section_identity"
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
                                   "column_mapping_diagnostics": diagnostic,
                                   "date_parsing": date_diagnostic,
                                   "normalization_mapping": {h: rename.get(h, h) for h in source_headers},
                                   "observed_deduct_columns": sorted(h for h in frame.columns if DEDUCT_HEADER.fullmatch(h)),
                                   "ungraded_deduct_columns": sorted(h for h in frame.columns
                                       if (match := DEDUCT_HEADER.fullmatch(h)) and not match.group(2)),
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
              "holdout_fraction": 0.2, "date_format": "%Y-%m-%d", "association_policy": "fail",
              "blank_deducts": "missing"}
    config.update({key: value for key, value in options.items() if key != "columns"})
    report["deduct_codes"] = sorted(codes)
    report["deduct_families"] = config["deduct_families"]
    report["features"] = config["features"]
    report["date_format"] = config["date_format"]
    report["blank_deducts"] = config["blank_deducts"]
    report["coverage_warning"] = (
        "Family totals use exported deduct columns only. A missing code/severity column is not a confirmed zero. "
        "Exports may cover different columns; inspect observed_deduct_columns for every source before comparing results.")
    report["feature_note"] = "Default PCI_start only; add scientifically justified features through private options."
    write_json(boundary.output("setup_report.json"), report)
    destination = boundary.output("run.json")
    write_json(destination, config)
    return destination


def analyze(boundary: Boundary, data: Path, options_path: Path | None = None,
            *, by_code: bool = False, date_format: str | None = None,
            blank_deducts: str | None = None) -> str:
    from .pipeline import run
    config = prepare(boundary, data, options_path, by_code=by_code, date_format=date_format,
                     blank_deducts=blank_deducts)
    return run(boundary, config)
