#!/usr/bin/env python3
"""Fail closed on unexpected Git paths, binary blobs, and common disclosure markers.

Messages report rule codes and counts only: offending content and filenames must
not become a second disclosure through public CI logs.
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ".publication-allowlist"
MAX_BYTES = 256 * 1024
BASE_NAMES = {MANIFEST, ".gitignore", ".gitattributes", "pyproject.toml", "requirements-tested.txt",
              "README.md", "SECURITY.md", "CONTRIBUTING.md", ".githooks/pre-commit", ".githooks/pre-push",
              ".github/workflows/ci.yml"}

# Patterns split strings where needed to avoid matching the scanner's own source.
MARKERS = [
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE" + r" KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub" + r"_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"https?://[^\s/@:]+:[^\s/@]+@", re.I),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"\b[A-Za-z0-9.-]+\.(?:mil|internal|local)\b", re.I),
    re.compile(r"\b(?:latitude|longitude)\s*[:=]\s*[-+]?\d+(?:\.\d+)?", re.I),
    re.compile(r"\b[-+]?\d{1,2}\.\d{4,}\s*[,;]\s*[-+]?\d{1,3}\.\d{4,}\b"),
    re.compile(r"\b(?:air" + r"field|installation)\s*[:=]\s*['\"][^'\"]+", re.I),
    re.compile(r"\b(?:Fort|Camp)\s+[A-Z][a-z]{2,}"),
    re.compile(r"\b[A-Z][A-Za-z-]+\s+(?:AFB|SFB|Air Force Base)\b"),
    re.compile(r"version https://git-lfs.github.com/spec/" + r"v1"),
]


class Rejected(Exception):
    pass


def git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise Rejected("GIT_INSPECTION_FAILED")
    return result.stdout


def eligible(path: str) -> bool:
    if path in BASE_NAMES:
        return True
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", path) or ".." in Path(path).parts:
        return False
    return ((path.startswith("src/pavement_analytics/") or path.startswith("tests/")
             or path.startswith("tools/")) and path.endswith(".py")) or (
                 path.startswith("docs/") and path.endswith(".md"))


def allowed(blob: bytes) -> set[str]:
    text = blob.decode("utf-8")
    paths = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if len(paths) != len(set(paths)) or any(not eligible(p) for p in paths):
        raise Rejected("INVALID_PUBLICATION_ALLOWLIST")
    return set(paths)


def inspect(path: str, mode: str, data: bytes, allow: set[str], forbidden: list[str]) -> set[str]:
    errors = set()
    if mode not in {"100644", "100755"}:
        errors.add("LINK_OR_SUBMODULE")
    if path not in allow or not eligible(path):
        errors.add("UNAPPROVED_PATH")
    if len(data) > MAX_BYTES or b"\0" in data:
        return errors | {"BINARY_OR_OVERSIZE"}
    try:
        value = data.decode("utf-8")
    except UnicodeDecodeError:
        return errors | {"NON_TEXT_FILE"}
    combined = path + "\n" + value
    if any(pattern.search(combined) for pattern in MARKERS):
        errors.add("SENSITIVE_MARKER")
    folded = combined.casefold()
    if any(term in folded for term in forbidden):
        errors.add("PRIVATE_DENYLIST_MATCH")
    return errors


def staged(forbidden: list[str]) -> set[str]:
    allow = allowed(git("show", ":" + MANIFEST))
    entries = git("ls-files", "--stage", "-z").split(b"\0")
    errors = set()
    for item in entries:
        if not item:
            continue
        meta, name = item.split(b"\t", 1)
        mode, sha, stage = meta.decode().split()
        if stage != "0":
            errors.add("UNMERGED_INDEX")
            continue
        path = name.decode("utf-8")
        data = git("cat-file", "blob", sha) if mode != "160000" else b""
        errors |= inspect(path, mode, data, allow, forbidden)
    return errors


def history(forbidden: list[str]) -> set[str]:
    # All locally reachable commits, including a file subsequently deleted.
    if git("rev-parse", "--is-shallow-repository").strip() != b"false":
        raise Rejected("FULL_HISTORY_REQUIRED")
    commits = git("rev-list", "--all").decode().splitlines()
    if not commits:
        return set()
    errors, seen = set(), set()
    for reference in git("for-each-ref", "--format=%(refname)").decode().splitlines():
        errors |= inspect("README.md", "100644", reference.encode(), {"README.md"}, forbidden)
        if git("cat-file", "-t", reference).strip() == b"tag":
            errors |= inspect("README.md", "100644", git("cat-file", "tag", reference), {"README.md"}, forbidden)
    for commit in commits:
        errors |= inspect("README.md", "100644", git("cat-file", "commit", commit), {"README.md"}, forbidden)
        try:
            allow = allowed(git("show", commit + ":" + MANIFEST))
        except Rejected:
            errors.add("HISTORY_WITHOUT_PUBLICATION_POLICY")
            continue
        for item in git("ls-tree", "-rz", commit).split(b"\0"):
            if not item:
                continue
            meta, name = item.split(b"\t", 1)
            mode, kind, sha = meta.decode().split()
            path = name.decode("utf-8")
            key = (path, mode, sha, tuple(sorted(allow)))
            if key in seen:
                continue
            seen.add(key)
            data = git("cat-file", "blob", sha) if kind == "blob" else b""
            errors |= inspect(path, mode, data, allow, forbidden)
    return errors


def working(forbidden: list[str]) -> set[str]:
    allow = allowed((ROOT / MANIFEST).read_bytes())
    errors = set()
    # This mode is for a clean export directory, not a developer venv.
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            errors.add("LINK_OR_SUBMODULE")
        elif path.is_file():
            errors |= inspect(relative.as_posix(), "100644", path.read_bytes(), allow, forbidden)
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check a source-only publication candidate")
    parser.add_argument("--working-tree", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--require-denylist", action="store_true")
    args = parser.parse_args(argv)
    try:
        forbidden = []
        deny = os.environ.get("PAVEMENT_DENYLIST_FILE")
        if deny:
            private_file = Path(deny).expanduser().resolve()
            if private_file == ROOT or ROOT in private_file.parents:
                raise Rejected("DENYLIST_MUST_BE_EXTERNAL")
            forbidden = [v.strip().casefold() for v in private_file.read_text().splitlines() if v.strip()]
        if args.require_denylist and not forbidden:
            raise Rejected("PRIVATE_DENYLIST_REQUIRED_FOR_PUSH")
        errors = working(forbidden) if args.working_tree else staged(forbidden)
        if args.history:
            errors |= history(forbidden)
        if errors:
            print("PUBLICATION_REJECTED: " + ", ".join(sorted(errors)))
            return 1
        print("PUBLICATION_CHECK_PASSED")
        return 0
    except Exception:
        # Never echo Git exceptions, denied tokens, or private file names.
        print("PUBLICATION_REJECTED: SCAN_INCOMPLETE")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
