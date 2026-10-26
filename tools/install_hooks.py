#!/usr/bin/env python3
"""Install the local, source-only pre-commit and pre-push gates."""

from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
for name in ["pre-commit", "pre-push"]:
    (root / ".githooks" / name).chmod(0o755)
result = subprocess.run(["git", "-C", str(root), "config", "--local", "core.hooksPath", ".githooks"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result.returncode:
    raise SystemExit("HOOK_SETUP_FAILED")
print("PUBLICATION_HOOKS_INSTALLED")
