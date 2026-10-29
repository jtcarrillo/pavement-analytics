"""Preserved computational stages from the original inspection-pair workflow.

The supported end-to-end command owns execution and private logging. Original
standalone module CLIs retain their argument contracts behind the same private
execution boundary as the rest of this package.
"""
from pathlib import Path

from ..security import SafeError, disjoint, refuse_public_execution, reject_links, source_root


def check_cli_paths(*, inputs=(), outputs=()) -> None:
    """Keep standalone legacy-stage reads/writes outside the public checkout."""
    refuse_public_execution()
    public = source_root().resolve()
    for value in (*inputs, *outputs):
        if value is None:
            continue
        path = Path(value).expanduser()
        reject_links(path)
        path = path.resolve()
        if not disjoint(path, public):
            raise SafeError("ORIGINAL_STAGE_PATH_MUST_BE_OUTSIDE_PUBLIC_TREE")
    for value in outputs:
        if value is None:
            continue
        path = Path(value).expanduser().resolve()
        if any((parent / ".git").exists() for parent in (path, *path.parents)):
            raise SafeError("OUTPUT_MUST_NOT_BE_VERSIONED")


# Shared spelling used by the other preserved stage modules.
guard_paths = check_cli_paths
