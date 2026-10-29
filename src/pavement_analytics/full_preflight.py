"""Read-only availability checks for the full pipeline's selected inputs."""
from __future__ import annotations

import os
from pathlib import Path
import stat

from .security import reject_links


def check_setup_inputs(cfg: dict) -> list[dict]:
    """Describe all selected auxiliary inputs without exposing exception text.

    Paths are already selected by setup; this function neither searches for
    replacements nor modifies files. Symlink boundary failures remain fatal.
    Ordinary filesystem failures become fixed statuses so setup can show all
    unavailable inputs together.
    """
    checks = []
    for role in ("mapping", "csv_dir", "cdv_curves", "deduct_curves", "families"):
        value = cfg.get(role)
        if value is None or (role in {"deduct_curves", "families"} and not value):
            continue
        path = Path(value).expanduser()
        reject_links(path)
        result = {"role": role, "path": str(path), "status": "ready"}
        try:
            mode = path.stat().st_mode
            if role == "csv_dir":
                if not stat.S_ISDIR(mode):
                    result["status"] = "not_directory"
                else:
                    with os.scandir(path) as entries:
                        next(entries, None)
            elif not stat.S_ISREG(mode):
                result["status"] = "not_file"
            else:
                with path.open("rb") as stream:
                    stream.read(1)
        except (FileNotFoundError, NotADirectoryError):
            result["status"] = "missing"
        except OSError:
            result["status"] = "unreadable"
        checks.append(result)
    return checks
