"""CLI: private paths and records never appear in terminal diagnostics."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import sys

from .io import write_json
from .security import Boundary, SafeError, private_umask, refuse_public_execution, source_root, sync_private


class Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default echoes invalid arguments, potentially containing paths.
        raise SafeError("INVALID_COMMAND_ARGUMENTS")


def parser() -> Parser:
    root = Parser(description="Generic PCI, deduct-value and PAVER analysis tools")
    root.add_argument("--workspace", type=Path, required=True, help="Private workspace outside the code checkout")
    commands = root.add_subparsers(dest="command", required=True, parser_class=Parser)
    commands.add_parser("init", help="Write an empty configuration template into the private workspace")
    fetch = commands.add_parser("sync", help="Fetch a pinned private data commit using runtime SSH configuration")
    fetch.add_argument("--commit", required=True)
    process = commands.add_parser("run", help="Process data and keep all results inside the private workspace")
    process.add_argument("--config", type=Path, required=True)
    process.add_argument("--resume", action="store_true")
    return root


def initialize(boundary: Boundary) -> None:
    destination = boundary.path("run.json")
    # Exclusive write: never replace a user's existing configuration.
    payload = {"schema_version": 1, "input_root": "inputs", "input_files": [], "output": "runs/v1",
               "columns": {"section": "SectionUniqueID", "date": "Date", "pci": "PCI",
                           "surface": "Inspected Surface Type", "group": "group_id"},
               "features": ["PCI_start"], "min_delta_days": 365, "models": True,
               "deduct_families": {}, "folds": 5, "seed": 42, "holdout_fraction": 0.2,
               "association_policy": "fail"}
    import json
    try:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError:
        raise SafeError("CONFIGURATION_ALREADY_EXISTS") from None


def main(argv=None) -> int:
    code = None
    try:
        args = parser().parse_args(argv)
        refuse_public_execution()
        with private_umask():
            boundary = Boundary.create(source_root(), args.workspace)
            log = boundary.output("logs")
            log.mkdir(exist_ok=True)
            # Capture library warnings and child-library diagnostics privately too.
            with boundary.output(log / "runtime.log").open("a", encoding="utf-8") as stream:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    try:
                        if args.command == "init":
                            initialize(boundary)
                            code = "PRIVATE_CONFIGURATION_CREATED"
                        elif args.command == "sync":
                            sync_private(boundary, args.commit)
                            code = "PRIVATE_SOURCE_READY"
                        else:
                            from .pipeline import run
                            code = run(boundary, args.config, resume=args.resume)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        raise
    except SafeError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED", file=sys.stderr)
        return 130
    except Exception:
        print("PRIVATE_OPERATION_FAILED_CHECK_PRIVATE_LOG", file=sys.stderr)
        return 2
    print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
