"""Run all implemented stages inside one private, immutable run directory."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
import shutil
import tempfile

import numpy as np

from . import __version__
from .curation import build_pairs
from .distress import derive
from .features import associations, build
from .io import apply_corrections, load_sources, read_json, sha256, write_json
from .modeling import search
from .ranking import rank
from .reporting import build_report
from .security import Boundary, SafeError, disjoint, lock, refuse_public_execution, reject_links, source_root

ALLOWED = {"schema_version", "input_root", "input_files", "output", "columns", "cutoff_year",
           "min_delta_days", "max_delta_days", "date_format", "corrections", "features",
           "deduct_families", "curve_profiles", "models", "folds", "seed", "holdout_fraction",
           "association_policy", "setup_report"}


def validate_config(config: dict) -> None:
    required = {"schema_version", "input_root", "input_files", "output", "columns", "features"}
    if set(config) - ALLOWED or not required <= set(config) or config["schema_version"] != 1:
        raise SafeError("INVALID_RUN_CONFIGURATION")
    if not isinstance(config["columns"], dict) or any(not isinstance(v, str) or not v for v in config["columns"].values()):
        raise SafeError("INVALID_COLUMN_MAPPING")
    for k in ["input_files", "features"]:
        if not isinstance(config[k], list) or not config[k] or any(not isinstance(v, str) or not v for v in config[k]):
            raise SafeError("INVALID_CONFIGURATION_LIST")
    for k, default in [("min_delta_days", 365), ("cutoff_year", 1980), ("folds", 5), ("seed", 42)]:
        value = config.get(k, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise SafeError("INTEGER_CONFIGURATION_REQUIRED")
    if not 1 <= config.get("cutoff_year", 1980) <= 9999 or config.get("folds", 5) < 2 or config.get("seed", 42) < 0:
        raise SafeError("INVALID_CONFIGURATION_RANGE")
    maximum = config.get("max_delta_days")
    if maximum is not None and (not isinstance(maximum, int) or isinstance(maximum, bool)):
        raise SafeError("INTEGER_CONFIGURATION_REQUIRED")
    if not isinstance(config.get("models", True), bool):
        raise SafeError("BOOLEAN_CONFIGURATION_REQUIRED")
    holdout = config.get("holdout_fraction", 0.2)
    if not isinstance(holdout, (int, float)) or isinstance(holdout, bool) or not 0 < holdout < 0.5:
        raise SafeError("INVALID_HOLDOUT_FRACTION")
    if config.get("association_policy", "fail") not in {"fail", "report"}:
        raise SafeError("INVALID_ASSOCIATION_POLICY")
    if "PCI_start" not in config["features"]:
        raise SafeError("CONDITION_FEATURE_REQUIRED")


def environment_versions() -> dict:
    return {"python": sys.version.split()[0], **{name: importlib.metadata.version(name) for name in
            ["numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "joblib"]}}


def code_digest() -> str:
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checksum_outputs(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        reject_links(path)
        if path.is_file() and path != root / "manifest.json":
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def run(boundary: Boundary, config_path: Path, *, resume: bool = False) -> str:
    refuse_public_execution()
    config_path = boundary.path(config_path)
    config_hash = sha256(config_path)
    config = read_json(config_path)
    validate_config(config)
    base = config_path.parent
    root = boundary.path(config["input_root"], base=base)
    out = boundary.output(config["output"], base=base)
    if not disjoint(root, out) or not disjoint(out, config_path):
        raise SafeError("INPUT_OUTPUT_PATHS_MUST_BE_DISJOINT")
    correction = boundary.path(config["corrections"], base=base) if config.get("corrections") else None
    curves = boundary.path(config["curve_profiles"], base=base) if config.get("curve_profiles") else None
    setup = boundary.path(config["setup_report"], base=base) if config.get("setup_report") else None
    auxiliaries = [p for p in [correction, curves, setup] if p is not None]
    if setup is not None:
        for item in read_json(setup).get("sources", []):
            snapshot = boundary.path(item["snapshot"], base=setup.parent)
            if sha256(snapshot) != item["original_sha256"]:
                raise SafeError("SOURCE_SNAPSHOT_CHANGED")
            auxiliaries.append(snapshot)
    for auxiliary in auxiliaries:
        if auxiliary is not None and not disjoint(auxiliary, out):
            raise SafeError("INPUT_OUTPUT_PATHS_MUST_BE_DISJOINT")
    auxiliary_hashes = {str(p): sha256(p) for p in auxiliaries}
    raw, sources = load_sources(root, config["input_files"], config["columns"])
    inputs = {"sources": sources, "auxiliary": auxiliary_hashes, "config_sha256": config_hash,
              "code_sha256": code_digest(), "version": __version__, "environment": environment_versions()}
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock(out.parent / ("." + out.name + ".lock")):
        if out.exists():
            if not resume:
                raise SafeError("OUTPUT_EXISTS_USE_NEW_VERSION_OR_RESUME")
            manifest = read_json(out / "manifest.json")
            if manifest.get("fingerprint") != fingerprint:
                raise SafeError("RESUME_INPUT_OR_CODE_CHANGED")
            if manifest.get("outputs") != checksum_outputs(out):
                raise SafeError("RESUME_OUTPUT_TAMPERED")
            return "VERIFIED_EXISTING_RUN"
        staging = Path(tempfile.mkdtemp(prefix=".run-", dir=out.parent))
        started = datetime.now(timezone.utc).isoformat()
        try:
            corrected, changes = apply_corrections(raw, correction)
            pairs, rejected, transitions, counts = build_pairs(
                corrected, config["columns"], cutoff_year=config.get("cutoff_year", 1980),
                min_days=config.get("min_delta_days", 365), max_days=config.get("max_delta_days"),
                date_format=config.get("date_format", "%Y-%m-%d"))
            changes.to_csv(staging / "corrections.csv", index=False)
            rejected.to_csv(staging / "rejected_rows.csv", index=False)
            transitions.to_csv(staging / "transitions.csv", index=False)
            write_json(staging / "pairing_report.json", {
                "accounting": counts,
                "rejected_reasons": {key: int(value) for key, value in rejected.reason.value_counts().items()},
                "transition_reasons": {key: int(value) for key, value in transitions.reason.value_counts().items()},
            })
            if pairs.empty:
                if len(rejected) == len(corrected) and len(rejected) and rejected.reason.eq("invalid_date").all():
                    raise SafeError("NO_VALID_DATES_CHECK_DATE_FORMAT")
                raise SafeError("NO_VALID_PAIRS")
            profiles = read_json(curves) if curves else None
            pairs, deduct_targets = derive(pairs, config.get("deduct_families", {}), profiles)
            pairs.to_csv(staging / "dataset.csv", index=False)
            rank(pairs).to_csv(staging / "rankings.csv", index=False)
            features, audit = build(pairs, config["features"], ["DetRatePerYear", *deduct_targets], config["columns"])
            features.to_csv(staging / "features.csv", index=False)
            audit.to_csv(staging / "feature_audit.csv", index=False)
            slots = []
            for material in sorted(pairs.material.unique()):
                mask = pairs.material.eq(material).to_numpy()
                subset = features.loc[mask]
                material_dir = staging / "materials" / material
                material_dir.mkdir(parents=True)
                matrix, correlated = associations(subset, config["features"])
                matrix.to_csv(material_dir / "association.csv")
                correlated.to_csv(material_dir / "association_pairs.csv", index=False)
                if len(correlated) and config.get("association_policy", "fail") == "fail":
                    raise SafeError("FEATURE_ASSOCIATION_REVIEW_REQUIRED")
                for target in ["DetRatePerYear", *deduct_targets]:
                    finite = np.isfinite(subset[target])
                    coverage = {"valid_target_rows": int(finite.sum()),
                                "excluded_missing_target": int((~finite).sum()),
                                "valid_target_groups": int(subset.loc[finite, "group_id"].nunique())}
                    if not finite.any():
                        slots.append({"material": material, "target": target, "status": "no_valid_target", **coverage})
                        continue
                    if config.get("models", True):
                        try:
                            summary = search(subset, config["features"], target, material_dir / target,
                                             folds=config.get("folds", 5), seed=config.get("seed", 42),
                                             holdout_fraction=config.get("holdout_fraction", 0.2))
                        except SafeError as error:
                            if str(error) != "INSUFFICIENT_GROUPS_FOR_MODEL_AND_HOLDOUT":
                                raise
                            summary = {"target": target, "status": "insufficient_groups"}
                    else:
                        summary = {"target": target, "status": "modeling_disabled"}
                    slots.append({"material": material, **coverage, **summary})
            write_json(staging / "run_report.json", {"accounting": counts, "slots": slots})
            if setup is not None:
                shutil.copyfile(setup, staging / "setup_report.json")
            build_report(staging, slots, counts)
            write_json(staging / "source_manifest.json", {"files": sources})
            # A changing input cannot silently certify an internally inconsistent run.
            current_inputs = all(sha256(root / s["path"]) == s["sha256"] for s in sources)
            current_auxiliary = all(sha256(Path(p)) == h for p, h in auxiliary_hashes.items())
            if not current_inputs or not current_auxiliary or sha256(config_path) != config_hash:
                raise SafeError("INPUT_CHANGED_DURING_RUN")
            manifest = {"schema_version": 1, "fingerprint": fingerprint, "inputs": inputs,
                        "config": config, "started": started,
                        "finished": datetime.now(timezone.utc).isoformat(), "outputs": checksum_outputs(staging)}
            write_json(staging / "manifest.json", manifest)
            staging.rename(out)
        except Exception:
            # Keep failure audits privately, away from the immutable output version.
            failed = boundary.output("failed")
            failed.mkdir(exist_ok=True)
            staging.rename(failed / staging.name.lstrip("."))
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return "PRIVATE_RUN_COMPLETE"
