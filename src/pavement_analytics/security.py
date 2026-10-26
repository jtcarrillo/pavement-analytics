"""Local path boundaries and a deliberately quiet private-source adapter."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterator


class SafeError(Exception):
    """An error code that may safely be printed without private values."""


def inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def disjoint(a: Path, b: Path) -> bool:
    return not inside(a, b) and not inside(b, a)


def reject_links(path: Path) -> None:
    """Reject existing symlinks in the complete path, including its ancestors."""
    absolute = path.expanduser().absolute()
    if any(p.is_symlink() for p in (absolute, *absolute.parents)):
        raise SafeError("SYMLINK_PATH_FORBIDDEN")


def refuse_public_execution() -> None:
    # This also forbids a private-data job on a self-hosted GitHub Actions runner.
    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        raise SafeError("GITHUB_ACTIONS_DATA_EXECUTION_FORBIDDEN")
    if os.environ.get("PAVEMENT_PRIVATE_EXECUTION") != "1":
        raise SafeError("PRIVATE_EXECUTION_OPT_IN_REQUIRED")


def source_root() -> Path:
    location = Path(__file__).resolve()
    for parent in location.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src/pavement_analytics").is_dir():
            return parent
    return location.parent


@dataclass(frozen=True)
class Boundary:
    public: Path
    workspace: Path

    @classmethod
    def create(cls, public: Path, workspace: Path) -> "Boundary":
        reject_links(workspace)
        public, workspace = public.resolve(), workspace.expanduser().resolve()
        if not disjoint(public, workspace):
            raise SafeError("WORKSPACE_MUST_BE_OUTSIDE_PUBLIC_TREE")
        # Do not accidentally put generated results in another Git checkout.
        if any((p / ".git").exists() for p in (workspace, *workspace.parents)):
            raise SafeError("WORKSPACE_MUST_NOT_BE_VERSIONED")
        workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            workspace.chmod(0o700)
        return cls(public, workspace)

    def path(self, value: str | Path, *, base: Path | None = None) -> Path:
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = (base or self.workspace) / p
        reject_links(p)
        p = p.resolve()
        if not inside(p, self.workspace) or not disjoint(p, self.public):
            raise SafeError("PATH_OUTSIDE_PRIVATE_WORKSPACE")
        return p

    def output(self, value: str | Path, *, base: Path | None = None) -> Path:
        p = self.path(value, base=base)
        if p == self.workspace:
            raise SafeError("OUTPUT_MUST_BE_A_CHILD_DIRECTORY")
        if any((a / ".git").exists() for a in (p, *p.parents)):
            raise SafeError("OUTPUT_MUST_NOT_BE_VERSIONED")
        return p


@contextlib.contextmanager
def private_umask() -> Iterator[None]:
    old = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(old)


@contextlib.contextmanager
def lock(path: Path) -> Iterator[None]:
    """Exclusive creation prevents two writers using the same run destination."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise SafeError("WORKSPACE_LOCKED") from None
    try:
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def sync_private(boundary: Boundary, commit: str) -> Path:
    """Fetch an exact commit over SSH into an external, never-reused checkout.

    The URL is read only at runtime. SSH credentials and host verification are
    delegated to the user's existing SSH configuration. No hooks or submodules
    execute; LFS must be materialized by an approved internal process.
    """
    refuse_public_execution()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SafeError("FULL_COMMIT_SHA_REQUIRED")
    remote = os.environ.get("PAVEMENT_DATA_REMOTE", "")
    # Restricted SCP-style SSH syntax, with no embedded password or command syntax.
    if not re.fullmatch(r"[A-Za-z0-9_-]+@[A-Za-z0-9_.-]+:[A-Za-z0-9_./-]+", remote):
        raise SafeError("SSH_REMOTE_REQUIRED")
    if any(part in {"..", "."} for part in remote.split(":", 1)[1].split("/")):
        raise SafeError("REMOTE_PATH_INVALID")
    destination = boundary.path(Path("sources") / commit)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        raise SafeError("SOURCE_DESTINATION_ALREADY_EXISTS")
    # Strip inherited Git overrides; preserve HOME/SSH agent for normal SSH auth.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1",
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_SSH_COMMAND": "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes"})
    import shutil
    with lock(boundary.path(".source-sync.lock")):
        staging = Path(tempfile.mkdtemp(prefix=".fetch-", dir=destination.parent))
        hooks = staging / "empty-hooks"
        hooks.mkdir()
        prefix = ["git", "-c", f"core.hooksPath={hooks}", "-c", "protocol.file.allow=never",
                  "-c", "protocol.ext.allow=never", "-c", "submodule.recurse=false"]
        def git(*args: str) -> bytes:
            try:
                return subprocess.run(prefix + list(args), cwd=staging, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      timeout=300, check=True).stdout
            except (OSError, subprocess.SubprocessError):
                raise SafeError("PRIVATE_SOURCE_FETCH_FAILED") from None
        try:
            git("init", "--quiet", "checkout")
            git("-C", "checkout", "remote", "add", "origin", remote)
            git("-C", "checkout", "fetch", "--quiet", "--no-tags", "--depth=1", "origin", commit)
            if git("-C", "checkout", "rev-parse", "FETCH_HEAD").decode().strip() != commit:
                raise SafeError("PRIVATE_REVISION_MISMATCH")
            entries = git("-C", "checkout", "ls-tree", "-rz", "FETCH_HEAD").split(b"\0")
            if any(e.startswith((b"120000 ", b"160000 ")) for e in entries):
                raise SafeError("SOURCE_LINKS_OR_SUBMODULES_FORBIDDEN")
            git("-C", "checkout", "checkout", "--quiet", "--detach", "FETCH_HEAD")
            (staging / "checkout").rename(destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return destination
