"""Content-keyed, verified complete target tasks; partial work is never reused."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import platform
import shutil
import tempfile

import pandas as pd

from .io import read_json, write_json, sha256
from .security import Boundary, SafeError, reject_links, source_root


def task_key(table, plan, features, climate, target, seed, weight_column, options, extra=None):
    from .pipeline import code_digest
    from .model_registry import candidate_catalog, environment_versions, REGISTRY_VERSION
    content = hashlib.sha256()
    for frame in (table, plan):
        content.update(json.dumps([(str(c), str(frame[c].dtype)) for c in frame], ensure_ascii=True).encode())
        content.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    content.update(json.dumps({"features": features, "climate_features": climate, "target": target,
        "seed": seed, "weight_column": weight_column, "options": options, "extra": extra,
        "registry_version": REGISTRY_VERSION, "candidates": candidate_catalog(seed),
        "python": platform.python_version(), "platform": [platform.system(), platform.machine()],
        "code_sha256": code_digest(), "environment": environment_versions()}, sort_keys=True).encode())
    return content.hexdigest()


def _checksums(root):
    result = {}
    for path in sorted(root.rglob("*")):
        reject_links(path)
        if path.is_file() and path != root/"task_manifest.json":
            result[str(path.relative_to(root))] = sha256(path)
    return result


def verified_task(cache, key, build, destination, progress=None):
    """Build atomically, or copy an exactly verified completed task to a run.

    Only complete targets are resumable. One foreground writer per workspace
    is supported. Interrupted/failed pending directories are retained privately.
    """
    cache, destination = Path(cache), Path(destination)
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
        raise SafeError("INVALID_MODEL_TASK_KEY")
    Boundary.create(source_root(), cache)
    Boundary.create(source_root(), destination.parent)
    reject_links(destination)
    cache.mkdir(parents=True, exist_ok=True)
    ready = cache/key
    reject_links(ready)
    if ready.exists():
        reject_links(ready/"task_manifest.json")
        if not (ready/"task_manifest.json").is_file():
            raise SafeError("MODEL_TASK_CHECKPOINT_CHANGED")
        manifest = read_json(ready/"task_manifest.json")
        if manifest.get("key") != key or manifest.get("status") != "complete" or manifest.get("outputs") != _checksums(ready):
            raise SafeError("MODEL_TASK_CHECKPOINT_CHANGED")
        if progress is not None:
            progress({"event": "model", "status": "task_reused"})
        summary = read_json(ready/"summary.json")
    else:
        pending = Path(tempfile.mkdtemp(prefix=".pending-model-", dir=cache))
        work = pending/"result"
        try:
            summary = build(work)
            write_json(work/"task_manifest.json", {"schema_version": 1, "status": "complete", "key": key,
                                                  "outputs": _checksums(work)})
            work.rename(ready)
            pending.rmdir()
        except BaseException:
            # Completed siblings remain intact, while no incomplete target can
            # masquerade as a ready checkpoint after cancellation or a crash.
            raise
    if destination.exists():
        raise SafeError("MODEL_TASK_DESTINATION_EXISTS")
    shutil.copytree(ready, destination)
    if _checksums(destination) != _checksums(ready):
        raise SafeError("MODEL_TASK_COPY_CHANGED")
    return summary
