"""Observe an existing private run without modifying its checkpoints or logs."""
from __future__ import annotations

import json
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import uuid

from .climate_progress import parse_progress_line
from .live_progress import LiveProgress, FINAL_STATUSES
from .progress import FullProgress
from .security import Boundary, SafeError, disjoint, refuse_public_execution, reject_links, source_root


POLL_SECONDS = 2.0
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 256 * 1024
MAX_LINE_BYTES = 8192
SURFACE = r"(?:PCC|APC|AC_AAC_ACC)"
STAGE = re.compile(r"(?:01_split|(?:02_pairs|03_location_rate|04_climate|05_targets|06_pci|07_modeling)_"
                   + SURFACE + r"|05_fit_curves|07_partitions|08_model_backend|09_local_models|10_report)")


def open_local_window(path: Path) -> bool:
    """Bound the browser helper; keep its diagnostics off the console."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import sys,webbrowser; sys.exit(0 if webbrowser.open(sys.argv[1]) else 1)",
             Path(path).resolve().as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=15, check=False)
        return result.returncode == 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _existing_boundary(workspace: Path) -> Boundary:
    reject_links(workspace)
    path, public = workspace.expanduser().resolve(), source_root().resolve()
    if not path.is_dir():
        raise SafeError("PROGRESS_WORKSPACE_NOT_FOUND")
    if not disjoint(path, public):
        raise SafeError("WORKSPACE_MUST_BE_OUTSIDE_PUBLIC_TREE")
    if any((ancestor / ".git").exists() for ancestor in (path, *path.parents)):
        raise SafeError("WORKSPACE_MUST_NOT_BE_VERSIONED")
    # Boundary.create would chmod the original run. Only the new monitor folder
    # is created; observation of the existing run does not alter its metadata.
    return Boundary(public, path)


def _markers(boundary: Boundary) -> list[Path]:
    return [boundary.path(name) for name in
            ("progress.json", "full_setup_report.json", "pipeline/run_report.json")]


def _select_workspace(workspace: Path | None) -> Boundary:
    if workspace is not None:
        boundary = _existing_boundary(Path(workspace))
        if not any(path.is_file() for path in _markers(boundary)):
            raise SafeError("PROGRESS_RUN_REPORT_NOT_FOUND")
        return boundary
    root = Path.home() / "pavement-runs"
    reject_links(root)
    candidates = []
    if root.is_dir():
        for path in root.glob("run-*"):
            if path.is_symlink() or not path.is_dir():
                continue
            boundary = _existing_boundary(path)
            times = [marker.stat().st_mtime for marker in _markers(boundary) if marker.is_file()]
            if times:
                candidates.append((max(times), str(path), boundary))
    if not candidates:
        raise SafeError("NO_EXISTING_FULL_RUN_FOUND")
    return max(candidates, key=lambda item: item[:2])[2]


def _read_json(path: Path) -> dict | None:
    """A writer may be between truncate and write; retry on the next poll."""
    reject_links(path)
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            return None
        with path.open("rb") as stream:
            content = stream.read(MAX_JSON_BYTES + 1)
        if len(content) > MAX_JSON_BYTES:
            return None
        value = json.loads(content)
    except (OSError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


class _Tail:
    def __init__(self):
        self.offset, self.partial, self.discarding = 0, b"", False
        self.identity = None
        self.generation = 0

    def lines(self, path: Path) -> list[str]:
        reject_links(path)
        try:
            details = path.stat()
            if not stat.S_ISREG(details.st_mode):
                return []
            identity = (details.st_dev, details.st_ino)
            if identity != self.identity or details.st_size < self.offset:
                self.offset, self.partial, self.discarding = 0, b"", False
                self.identity = identity
                self.generation += 1
            with path.open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read(MAX_LOG_BYTES)
            self.offset += len(chunk)
        except OSError:
            return []
        pieces = (self.partial + chunk).split(b"\n")
        trailing = pieces.pop()
        lines = []
        for piece in pieces:
            if self.discarding:
                self.discarding = False
                continue
            if len(piece) <= MAX_LINE_BYTES:
                lines.append(piece.decode("utf-8", errors="replace"))
        if len(trailing) > MAX_LINE_BYTES:
            self.partial, self.discarding = b"", True
        else:
            self.partial = b"" if self.discarding else trailing
        return lines


class _LegacyClimate:
    """Extract counters from known old log messages, never expose raw text."""
    def __init__(self):
        self.state, self.outcomes = {}, {}
        self.structured = False

    def parse(self, line: str) -> dict | None:
        event = parse_progress_line(line)
        if event is not None:
            self.structured = True
            return event
        # Updated workers retain their original messages as private diagnostics;
        # do not count both representations when observing their stage log.
        if self.structured:
            return None
        phase = None
        match = re.fullmatch(r"\[INFO\] Input rows: (\d+), unique locations: (\d+)\s*", line)
        if match:
            self.state.update(total_rows=int(match[1]), total_locations=int(match[2]), locations_completed=0)
            phase = "initializing"
        match = re.match(r"\[INFO\] \((\d+)/(\d+)\) loading lat=", line)
        if match:
            self.state.update(location_index=int(match[1]), total_locations=int(match[2]))
            phase = "loading_location"
        if line.startswith("[INFO] Requesting reanalysis-era5-"):
            phase = "requesting"
        elif line.startswith("[INFO] Reusing existing file:"):
            phase = "cache_hit"
        elif line.startswith("[INFO]   resolved via:") or line.startswith(("[ERROR] No usable data for (", "[ERROR] Failed to load (")):
            index = self.state.get("location_index")
            if index is not None:
                self.outcomes[index] = line.startswith("[INFO]")
                self.state.update(locations_completed=len(self.outcomes),
                                  usable_locations=sum(self.outcomes.values()),
                                  missing_locations=sum(not value for value in self.outcomes.values()))
                phase = "location_complete" if self.outcomes[index] else "location_failed"
                if len(self.outcomes) == self.state.get("total_locations"):
                    phase = "calculating"
        if re.fullmatch(r"\[INFO\] Wrote .+ \((\d+) rows\)", line):
            phase = "complete"
        return {"event": "climate", "phase": phase, **self.state} if phase else None


def monitor_existing(workspace: Path | None = None, *, open_window=False, once=False) -> Path:
    """Open a current progress page, or observe old reports in a separate view.

    Old runs have no heartbeat. Changes in their logs are observations only;
    neither a quiet log nor an old ``running`` status proves process liveness.
    """
    refuse_public_execution()
    boundary = _select_workspace(workspace)
    print("Monitoring workspace: " + json.dumps(str(boundary.workspace)), flush=True)
    page, state = boundary.path("progress.html"), boundary.path("progress.json")
    if page.is_file() and state.is_file():
        print("Progress window: " + json.dumps(str(page)), flush=True)
        if open_window and not open_local_window(page):
            print("PROGRESS_OPEN_FAILED_OPEN_HTML_MANUALLY", flush=True)
        return page
    output = boundary.output(Path("monitors") / uuid.uuid4().hex)
    view = LiveProgress(Boundary.create(boundary.public, output), sink=FullProgress(sys.stdout))
    view({"event": "legacy_monitor"})
    view({"event": "setup_started", "workspace": str(boundary.workspace)})
    print("Progress window: " + json.dumps(str(view.path)), flush=True)
    if open_window and not open_local_window(view.path):
        print("PROGRESS_OPEN_FAILED_OPEN_HTML_MANUALLY", flush=True)
    seen_setup, seen_stages, tails, decoders = None, {}, {}, {}
    monitoring_stopped = False
    try:
        while True:
            setup = _read_json(boundary.path("full_setup_report.json"))
            if setup is not None and setup != seen_setup:
                mapping = setup.get("mapping") if isinstance(setup.get("mapping"), dict) else {}
                cache = setup.get("climate_cache") if isinstance(setup.get("climate_cache"), dict) else {}
                paths = {"profile": setup.get("profile"), "mapping": mapping.get("original_path"),
                         "location_directory": mapping.get("location_directory"), "cache": cache.get("selected")}
                view({"event": "setup_files", **{key: value for key, value in paths.items() if isinstance(value, str)}})
                seen_setup = setup
            report = _read_json(boundary.path("pipeline/run_report.json"))
            stages = report.get("stages", {}) if report else {}
            if isinstance(stages, dict):
                # Pipeline execution completes one surface before starting the
                # next. Lexical stage order would put an earlier surface's
                # completed stage 07 after the active surface's stage 04.
                # Replay completed records first so the active work stays last.
                ordered = sorted(stages.items(), key=lambda item: (
                    isinstance(item[1], dict) and item[1].get("status") == "running", item[0]))
                for stage, record in ordered:
                    if not STAGE.fullmatch(stage) or not isinstance(record, dict):
                        continue
                    status = record.get("status")
                    if status not in {"running", "completed", "skipped", "failed", "interrupted"}:
                        continue
                    observed = {"event": "stage", "stage": stage, "status": "started" if status == "running" else status,
                                "stats": record.get("stats", {}), "seconds": record.get("seconds")}
                    if seen_stages.get(stage) != observed:
                        view(observed)
                        seen_stages[stage] = observed
                    if stage.startswith("04_climate_") and status == "running":
                        tail = tails.setdefault(stage, _Tail())
                        generation = tail.generation
                        lines = tail.lines(boundary.path(Path("pipeline/logs") / (stage + ".log")))
                        if tail.generation != generation or stage not in decoders:
                            decoders[stage] = _LegacyClimate()
                        for line in lines:
                            event = decoders[stage].parse(line)
                            if event is not None:
                                view({**event, "stage": stage})
            if report and isinstance(report.get("status"), str) and report["status"] in FINAL_STATUSES:
                view.finish(report["status"])
                break
            if once:
                monitoring_stopped = True
                break
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        # Stop observing; never claim that the separately running pipeline was
        # interrupted and never signal or terminate it.
        print("PROGRESS_MONITOR_STOPPED_PIPELINE_UNCHANGED", flush=True)
        monitoring_stopped = True
    if monitoring_stopped:
        view({"event": "monitor_stopped"})
    return view.path
