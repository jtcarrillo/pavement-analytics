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
    root.add_argument("--workspace", type=Path, help="Private workspace outside the code checkout")
    commands = root.add_subparsers(dest="command", required=True, parser_class=Parser)
    commands.add_parser("init", help="Write an empty configuration template into the private workspace")
    fetch = commands.add_parser("sync", help="Fetch a pinned private data commit using runtime SSH configuration")
    fetch.add_argument("--commit", required=True)
    process = commands.add_parser("run", help="Process data and keep all results inside the private workspace")
    process.add_argument("--config", type=Path, required=True)
    process.add_argument("--resume", action="store_true")
    analyze = commands.add_parser("analyze", help="Analyze all deduct families from an external file or folder")
    analyze.add_argument("data", type=Path, help="Private CSV, TSV, XLSX or folder (not recursive)")
    analyze.add_argument("--workspace", dest="analyze_workspace", type=Path,
                         help="New empty private workspace; defaults to ~/pavement-runs/run-<timestamp>-<id>")
    analyze.add_argument("--options", type=Path, help="Optional private JSON column, feature or family overrides")
    analyze.add_argument("--date-format", help="Explicit inspection-date format, for example %%m/%%d/%%Y")
    analyze.add_argument("--blank-deducts", choices=("missing", "zero"),
                         help="Use zero only when empty deduct cells mean absent distress; default missing")
    analyze.add_argument("--by-code", action="store_true", help="Use one family per observed code")
    analyze.add_argument("--open", action="store_true", help="Open the local HTML dashboard after success")
    study = commands.add_parser("study", help="Run location/ERA5 enrichment and compare observed deduct targets")
    study.add_argument("data", type=Path, help="Private inspection file or folder")
    study.add_argument("--workspace", dest="analyze_workspace", type=Path,
                       help="New empty private workspace; defaults to ~/pavement-runs/run-<timestamp>-<id>")
    study.add_argument("--options", type=Path,
                       help="Private study JSON; otherwise discover pavement-study.json beside data or its parent")
    study.add_argument("--date-format", help="Explicit inspection-date format")
    study.add_argument("--blank-deducts", choices=("missing", "zero"))
    study.add_argument("--by-code", action="store_true", help="One target per observed distress code")
    study.add_argument("--open", action="store_true", help="Open the consolidated local dashboard")
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
    previous_opt_in = None
    changed_opt_in = False
    try:
        args = parser().parse_args(argv)
        if args.command in {"analyze", "study"}:
            from .autorun import new_workspace
            if args.workspace is not None and args.analyze_workspace is not None:
                raise SafeError("DUPLICATE_WORKSPACE_ARGUMENT")
            # The analyze command is itself the explicit request for local private execution.
            # No environment mutation is retained when this function returns.
            previous_opt_in = os.environ.get("PAVEMENT_PRIVATE_EXECUTION")
            os.environ["PAVEMENT_PRIVATE_EXECUTION"] = "1"
            changed_opt_in = True
            workspace = new_workspace(args.analyze_workspace or args.workspace)
        else:
            if args.workspace is None:
                raise SafeError("WORKSPACE_REQUIRED")
            workspace = args.workspace
        refuse_public_execution()
        with private_umask():
            boundary = Boundary.create(source_root(), workspace)
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
                        elif args.command in {"analyze", "study"}:
                            if args.command == "study":
                                from .study import study as operation
                            else:
                                from .autorun import analyze as operation
                            code = operation(boundary, args.data, args.options, by_code=args.by_code,
                                             date_format=args.date_format, blank_deducts=args.blank_deducts)
                            if args.open:
                                import subprocess
                                try:
                                    subprocess.run(
                                        [sys.executable, "-c",
                                         "import sys,webbrowser; sys.exit(0 if webbrowser.open(sys.argv[1]) else 1)",
                                         boundary.output("results/dashboard.html").as_uri()],
                                        stdout=stream, stderr=stream, timeout=15, check=True)
                                except (OSError, subprocess.SubprocessError):
                                    print("DASHBOARD_OPEN_FAILED_OPEN_HTML_MANUALLY")
                        else:
                            from .pipeline import run
                            code = run(boundary, args.config, resume=args.resume)
                    except Exception as error:
                        if args.command == "study":
                            status = boundary.output("results/study_status.json")
                            if status.exists():
                                write_json(status, {"status": "failed", "error_code": str(error)
                                                    if isinstance(error, SafeError) else "PRIVATE_OPERATION_FAILED"})
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
    finally:
        if changed_opt_in:
            if previous_opt_in is None:
                os.environ.pop("PAVEMENT_PRIVATE_EXECUTION", None)
            else:
                os.environ["PAVEMENT_PRIVATE_EXECUTION"] = previous_opt_in
    print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
