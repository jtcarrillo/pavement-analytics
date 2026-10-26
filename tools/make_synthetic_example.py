"""Generate a deterministic example only in a new, external workspace."""

from __future__ import annotations

import csv
from datetime import date, timedelta
import json
from pathlib import Path
import random
import sys

from pavement_analytics.cli import Parser
from pavement_analytics.security import (
    Boundary, SafeError, private_umask, refuse_public_execution, reject_links,
    source_root,
)


def records() -> list[dict]:
    """Invent numeric histories and audit cases; no reference data are used."""
    rng = random.Random(42)
    rows = []
    origin = date(2015, 1, 1)
    for material_index, material in enumerate(("AC", "APC", "PCC")):
        for group in range(12):
            group_effect = rng.uniform(-0.2, 0.2)
            for section in range(4):
                driver = rng.uniform(0, 1)
                initial = rng.uniform(70, 98)
                loss = 0.8 + material_index * 0.25 + 3 * driver + group_effect
                deduct_initial = rng.uniform(3, 12)
                growth = 0.3 + 0.9 * driver + rng.uniform(-0.05, 0.05)
                history = []
                for visit in range(5):
                    row = {
                        "section": f"SYNTHETIC_{material}_{group:02d}_{section:02d}",
                        "group": f"SYNTHETIC_GROUP_{group:02d}",
                        "date": (origin + timedelta(days=730 * visit)).isoformat(),
                        "surface": material,
                        "pci": round(initial - loss * 2 * visit + rng.uniform(-0.1, 0.1), 4),
                        "driver_value": round(driver, 6),
                        "63L deduct": round(deduct_initial + growth * 2 * visit, 4),
                        "construction": "no",
                    }
                    history.append(row)
                # Exercise four distinct transition audit reasons in one group.
                if material == "AC" and group == 0:
                    if section == 0:
                        history[2]["pci"] = history[1]["pci"] + 1
                    elif section == 1:
                        history[2].update(date=history[1]["date"], pci=history[1]["pci"])
                    elif section == 2:
                        history[2]["date"] = (origin + timedelta(days=730 + 180)).isoformat()
                    else:
                        history[2]["construction"] = "yes"
                rows.extend(history)
    template = rows[-1].copy()
    for index, (column, value) in enumerate((
        ("date", "invalid"), ("date", "1970-01-01"), ("pci", 101),
        ("section", ""), ("group", ""), ("surface", "UNSUPPORTED"),
    )):
        row = dict(template, section=f"SYNTHETIC_REJECT_{index:02d}")
        row[column] = value
        rows.append(row)
    return rows


def create(workspace: Path) -> None:
    refuse_public_execution()
    reject_links(workspace)
    # Check before Boundary.create so an existing populated directory is untouched.
    if workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir())):
        raise SafeError("EXAMPLE_REQUIRES_EMPTY_WORKSPACE")
    with private_umask():
        boundary = Boundary.create(source_root(), workspace)
        inputs = boundary.output("inputs")
        inputs.mkdir(exist_ok=False)
        rows = records()
        with (inputs / "synthetic_inspections.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        config = {
            "schema_version": 1, "input_root": "inputs",
            "input_files": ["synthetic_inspections.csv"], "output": "runs/v1",
            "columns": {"section": "section", "date": "date", "pci": "pci",
                        "surface": "surface", "group": "group", "construction": "construction"},
            "features": ["PCI_start", "driver_value_start"],
            "deduct_families": {"example": [63]}, "models": True,
            "min_delta_days": 365, "folds": 3, "seed": 42,
            "holdout_fraction": 0.2, "association_policy": "fail",
        }
        with boundary.output("run.json").open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")


def main(argv=None) -> int:
    try:
        parser = Parser(description="Create invented inspections outside all Git checkouts")
        parser.add_argument("--workspace", required=True, type=Path)
        args = parser.parse_args(argv)
        create(args.workspace.expanduser())
    except SafeError as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        print("SYNTHETIC_EXAMPLE_CREATION_FAILED", file=sys.stderr)
        return 2
    print("SYNTHETIC_EXAMPLE_CREATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
