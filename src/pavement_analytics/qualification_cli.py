"""Private entry point; detailed diagnostics never reach the console."""
import contextlib
from pathlib import Path
import traceback

from .security import Boundary, SafeError, disjoint, reject_links, source_root
from .io import write_json


def qualification_command(args, progress=None):
    destination = Path(args.output).expanduser()
    reject_links(destination)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise SafeError("QUALIFICATION_REQUIRES_EMPTY_OUTPUT")
    if args.run_workspace is not None:
        reject_links(Path(args.run_workspace))
        if not disjoint(destination.resolve(), Path(args.run_workspace).expanduser().resolve()):
            raise SafeError("QUALIFICATION_OUTPUT_MUST_BE_SEPARATE")
    boundary = Boundary.create(source_root(), destination)
    with boundary.output("runtime.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            if args.run_workspace is None:
                from .release_qualification import release_fixture
                report = release_fixture(boundary.output("checks"), profile=args.profile, candidate_ids=args.candidate_ids, progress=progress)
            else:
                from .run_qualification import qualify_saved_run
                report, _ = qualify_saved_run(args.run_workspace, boundary.output("checks"), private_pilot=args.private_pilot, progress=progress)
            write_json(boundary.output("result.json"), {"status": report["status"],
                "evidence_kind": report["evidence_kind"], "report": "checks/qualification.html",
                "remaining_release_gates": report["remaining_release_gates"]})
        except BaseException:
            traceback.print_exc()
            raise
    return boundary.path("checks/qualification.html")
