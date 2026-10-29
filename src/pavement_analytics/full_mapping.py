"""Validate and normalize private source-to-location-file mappings.

Source paths are labels only: their basenames must match a selected inspection
unambiguously. Location paths retain a stricter boundary and must stay inside
the explicitly supplied location directory. No files are modified or moved.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .autorun import external_path
from .security import SafeError


def _record(source=None, location=None):
    return {"source_key": source if isinstance(source, str) else None,
            "location_value": location if isinstance(location, str) else None,
            "source_filename": None, "relative_location": None,
            "status": "invalid", "reason": None}


def _location_path(location, root: Path | None, root_error, public: Path):
    if not isinstance(location, str) or not location.strip() or "\0" in location:
        return None, "LOCATION_VALUE_MUST_BE_NONEMPTY_STRING"
    if root_error:
        return None, root_error
    try:
        location_path = Path(location).expanduser()
        if ".." in location_path.parts:
            return None, "LOCATION_PARENT_COMPONENT_FORBIDDEN"
        candidate = location_path if location_path.is_absolute() else root / location_path
        resolved = external_path(candidate, public)
        if resolved == root or not resolved.is_relative_to(root):
            return None, "LOCATION_PATH_OUTSIDE_DECLARED_DIRECTORY"
        if resolved.is_dir():
            return None, "LOCATION_PATH_MUST_IDENTIFY_FILE"
        return resolved, None
    except SafeError as error:
        return None, str(error)
    except (OSError, ValueError, RuntimeError):
        return None, "LOCATION_PATH_INVALID"


def normalize_location_mapping(payload, location_root: Path,
                               source_files: list[Path], public: Path) -> tuple[dict[str, str], list[dict]]:
    """Return normalized mappings plus every entry's validation decision.

    The caller must reject the mapping if any audit entry is ``invalid``.
    Missing location files are still valid *mapping syntax*; the setup/location
    phases diagnose their availability. No source-key path is ever opened.
    Plain filename keys for unselected inputs are retained, allowing one private
    mapping to serve several input subsets. Path keys require an actual unique
    basename match in ``source_files`` and are never guessed from other folders.
    """
    if not isinstance(payload, dict):
        item = _record()
        item["reason"] = "MAPPING_MUST_BE_FILENAME_TO_LOCATION_OBJECT"
        return {}, [item]
    normalized, audit, names = {}, [], {}
    try:
        root = external_path(Path(location_root), Path(public))
        root_error = None
    except SafeError as error:
        root, root_error = None, str(error)
    except (OSError, ValueError, RuntimeError):
        root, root_error = None, "LOCATION_DIRECTORY_PATH_INVALID"

    locations = {source: _location_path(location, root, root_error, Path(public))
                 for source, location in payload.items()}
    declared_locations = {path for path, error in locations.values() if error is None}
    counts = Counter()
    for path in source_files:
        # Recursive discovery may also find the declared location CSVs. Only
        # safely resolved mapping values can remove files from source matching.
        try:
            if external_path(Path(path), Path(public)) in declared_locations:
                continue
        except (SafeError, OSError, ValueError, RuntimeError):
            pass
        counts[Path(path).name] += 1

    for source, location in payload.items():
        item = _record(source, location)
        audit.append(item)
        if not isinstance(source, str) or not source.strip() or "\0" in source:
            item["reason"] = "SOURCE_KEY_MUST_BE_NONEMPTY_STRING"
            continue
        source_path = Path(source)
        name = source_path.name
        if not name or name in {".", ".."} or source.endswith("/"):
            item["reason"] = "SOURCE_KEY_MUST_IDENTIFY_FILENAME"
            continue
        item["source_filename"] = name
        if counts[name] > 1:
            item["reason"] = "SOURCE_BASENAME_MATCHES_MULTIPLE_INPUTS"
            continue
        source_is_path = source != name
        if source_is_path and counts[name] != 1:
            item["reason"] = "SOURCE_PATH_HAS_NO_SELECTED_INPUT_MATCH"
            continue
        if name in names:
            # Do not silently select one of two aliases, even if they happen to
            # name the same location today. Both original entries remain audited.
            item["reason"] = "SOURCE_BASENAME_COLLISION"
            first = names[name]
            first.update(status="invalid", reason="SOURCE_BASENAME_COLLISION")
            normalized.pop(name, None)
            continue
        names[name] = item
        resolved, location_error = locations[source]
        if location_error:
            item["reason"] = location_error
            continue
        relative = resolved.relative_to(root).as_posix()
        item["relative_location"] = relative
        changed = source_is_path or location != relative
        item.update(status="normalized" if changed else "valid",
                    reason="PATHS_NORMALIZED" if changed else "VALID_FILENAME_MAPPING")
        normalized[name] = relative
    return normalized, audit
