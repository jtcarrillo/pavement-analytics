"""A private, self-refreshing progress page with no server or network requests.

Only structured milestones are rendered. The files are observations, never
pipeline checkpoints, so a failed display cannot invalidate completed work.
"""
from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

from .progress import FullProgress, RUN_LABELS, STAGE_LABELS
from .security import Boundary, SafeError


PHASE_LABELS = {
    "initializing": "Preparing climate inputs", "loading_location": "Loading location",
    "cache_check": "Checking the weather cache", "cache_hit": "Reusing cached weather",
    "requesting": "Waiting for CDS / downloading", "downloaded": "Weather download returned",
    "parsing": "Reading weather data", "preparing": "Preparing hourly weather",
    "location_complete": "Location ready", "location_failed": "Location has no usable weather",
    "calculating": "Calculating climate features", "writing": "Saving climate features",
    "complete": "Climate calculation finished", "fallback": "Trying another weather grid point or dataset",
}
FINAL_STATUSES = set(RUN_LABELS)
PATH_FIELDS = {"profile": "Configuration", "mapping": "Mapping JSON",
               "location_directory": "Location CSV directory", "cache": "ERA5 cache"}


def _text(value, limit=500):
    if not isinstance(value, str):
        return None
    # JSON quoting in the console also keeps terminal controls and bidi marks
    # inert; use the same readable escapes for untrusted filenames in HTML.
    return json.dumps(value[:limit], ensure_ascii=True)[1:-1]


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def _stage(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_]{1,96}", value) else None


class LiveProgress:
    """Forward console events and maintain a bounded browser view.

    ``path`` and ``state_path`` are ready immediately. Opening the browser is
    the CLI's responsibility. ``disabled`` indicates a display write failed;
    the independent console sink still receives subsequent events.
    """

    def __init__(self, boundary: Boundary, sink=None):
        self.boundary, self.sink = boundary, sink
        self.path = boundary.output("progress.html")
        self.state_path = boundary.output("progress.json")
        self.disabled = False
        now = time.time()
        self.state = {"schema_version": 1, "status": "running", "started_at": now,
                      "updated_at": now, "last_signal_at": now, "finished_at": None,
                      "stage": "setup", "stage_started_at": now, "stage_finished_at": None,
                      "stage_status": "started", "phase": "Setting up the run",
                      "paths": {}, "input_stats": {}, "climate": {}, "stages": [],
                      "recent": [], "dashboard": None, "legacy": False,
                      "monitor_stopped": False, "monitor_stopped_at": None}
        self._save()

    def __call__(self, event: dict) -> None:
        if self.sink is not None:
            try:
                self.sink(event)
            except (OSError, ValueError):
                self.sink = None
        if self.disabled or not isinstance(event, dict):
            return
        now = time.time()
        kind = event.get("event")
        recognized = False
        if kind == "legacy_monitor":
            self.state["legacy"] = True
            recognized = True
        elif kind == "monitor_stopped" and self.state["legacy"]:
            self.state["monitor_stopped"] = True
            self.state["monitor_stopped_at"] = now
            recognized = True
        elif kind == "setup_started":
            self.state.update(status="running", stage="setup", stage_status="started",
                              phase="Verifying saved setup" if event.get("resume") else "Setting up the run",
                              stage_started_at=now, stage_finished_at=None, finished_at=None)
            for key in ("workspace", "data"):
                value = _text(event.get(key), 1500)
                if value is not None:
                    self.state["paths"][key] = value
            recognized = True
        elif kind == "setup_files":
            for key in PATH_FIELDS:
                value = _text(event.get(key), 1500)
                if value is not None:
                    self.state["paths"][key] = value
            recognized = True
        elif kind == "setup_complete":
            for key in ("files", "rows", "mapped_files", "unmapped_files", "missing_location_files"):
                value = event.get(key)
                if _finite(value):
                    self.state["input_stats"][key] = value
            self.state.update(phase="Setup complete", stage_finished_at=now, stage_status="completed")
            recognized = True
        elif kind == "stage":
            stage, status = _stage(event.get("stage")), event.get("status")
            if stage and status in STAGE_LABELS:
                previous_stage = self.state["stage"]
                self.state.update(stage=stage, stage_status=status,
                                  phase=STAGE_LABELS[status] + " " + stage)
                if status == "started" or previous_stage != stage:
                    self.state.update(stage_started_at=now, stage_finished_at=None, climate={})
                    self.state.pop("elapsed_seconds", None)
                    self.state.pop("idle_seconds", None)
                if status != "started":
                    self.state["stage_finished_at"] = now
                record = {"stage": stage, "status": status}
                if _finite(event.get("seconds")):
                    record["seconds"] = event["seconds"]
                records = [r for r in self.state["stages"] if r["stage"] != stage]
                self.state["stages"] = (records + [record])[-48:]
                if stage == "01_split" and isinstance(event.get("stats"), dict):
                    for key in ("input_rows", "retained_rows", "excluded_rows"):
                        value = event["stats"].get(key)
                        if _finite(value):
                            self.state["input_stats"][key] = value
                recognized = True
        elif kind == "climate":
            from .climate_progress import sanitize_climate_event
            climate = sanitize_climate_event(event)
            if climate:
                self.state["climate"].update({k: v for k, v in climate.items() if k != "event"})
                self.state["phase"] = PHASE_LABELS[climate["phase"]]
                recognized = True
        elif kind == "heartbeat":
            # A heartbeat proves the local child is still running, not that a
            # remote request is making progress or that remaining time is known.
            for key in ("elapsed_seconds", "idle_seconds"):
                if _finite(event.get(key)):
                    self.state[key] = event[key]
            recognized = True
        elif kind == "target":
            name = _text(event.get("target"))
            material = event.get("material")
            status = event.get("status")
            if name and material in {"PCC", "APC", "AC_AAC_ACC"} and status in {"started", "completed"}:
                self.state["phase"] = f"{STAGE_LABELS[status]} target: {material} / {name}"
                recognized = True
        elif kind == "run" and event.get("status") in FINAL_STATUSES:
            self.state.update(status=event["status"], phase=RUN_LABELS[event["status"]] + " run",
                              finished_at=now, stage_finished_at=now)
            recognized = True
        elif kind == "setup_failed":
            self.state.update(status="failed", phase="Setup failed", finished_at=now, stage_finished_at=now)
            recognized = True
        elif kind == "output":
            value = event.get("dashboard")
            if isinstance(value, str):
                try:
                    self.state["dashboard"] = self.boundary.path(value).as_uri()
                except (SafeError, ValueError, OSError):
                    pass
            recognized = True
        elif kind in {"input", "input_started", "setup_issue", "mapping_entry", "warning", "progress_window"}:
            recognized = True
        if not recognized:
            return
        self.state["updated_at"] = now
        if kind != "monitor_stopped":
            self.state["last_signal_at"] = now
        # The console's renderer already selects fixed fields and aggregate
        # statistics. Do not preserve arbitrary event payloads in this page.
        try:
            lines = FullProgress._render(event)
        except (KeyError, TypeError, ValueError):
            lines = []
        if kind != "heartbeat":
            self.state["recent"] = (self.state["recent"] + [line[:2000] for line in lines])[-80:]
        self._save()

    def finish(self, status: str) -> None:
        if status not in FINAL_STATUSES:
            raise ValueError("Unknown progress completion status")
        self({"event": "run", "status": status})

    def _atomic(self, path: Path, content: str) -> None:
        destination = self.boundary.output(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".progress-",
                                             suffix=".tmp", dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            # Recheck immediately before replacing an existing output.
            self.boundary.output(destination)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _save(self):
        if self.disabled:
            return
        try:
            self._atomic(self.state_path, json.dumps(self.state, indent=2, allow_nan=False) + "\n")
            self._atomic(self.path, self._html())
        except (OSError, ValueError, SafeError):
            self.disabled = True

    def _html(self):
        state, climate = self.state, self.state["climate"]
        escape = html.escape
        monitoring_stopped = state["legacy"] and state["monitor_stopped_at"] is not None
        active = state["status"] not in FINAL_STATUSES and not monitoring_stopped
        refresh = '<meta http-equiv="refresh" content="2">' if active else ""

        def number(value):
            return f"{int(value):,}" if _finite(value) else "—"

        def meter(label, done, total, detail=""):
            if _finite(total) and total > 0 and _finite(done):
                value = min(100., 100 * done / total)
                bar = f'<progress max="100" value="{value:.3f}"></progress>'
                count = f"{number(done)} / {number(total)}"
            elif total == 0:
                bar, count = '<div class="empty-bar"></div>', "0 / 0"
            else:
                bar, count = '<div class="empty-bar"></div>', "Waiting for counts"
            return f'<section class="meter"><div class="meter-label"><h2>{label}</h2><strong>{count}</strong></div>{bar}<p>{escape(detail)}</p></section>'

        location_detail = f"Usable: {number(climate.get('usable_locations'))} · No usable weather: {number(climate.get('missing_locations'))}"
        location_index = climate.get("location_index")
        if _finite(location_index) and location_index:
            location_detail = f"Current location: {number(location_index)} · " + location_detail
        meters = meter("Weather locations", climate.get("locations_completed"), climate.get("total_locations"), location_detail)
        meters += meter("Inspection-pair climate features", climate.get("rows_completed"), climate.get("total_rows"),
                        "Processed rows include rows without usable weather; completion does not imply full climate coverage.")
        paths = {"workspace": "Workspace", "data": "Input path", **PATH_FIELDS}
        file_rows = "".join(f"<dt>{label}</dt><dd>{escape(state['paths'][key])}</dd>"
                            for key, label in paths.items() if key in state["paths"])
        counts = state["input_stats"]
        input_info = " · ".join(f"{label}: {number(counts[key])}" for key, label in
                                (("files", "Files"), ("input_rows", "Input rows"),
                                 ("retained_rows", "Accepted"), ("excluded_rows", "Rejected"),
                                 ("unmapped_files", "Unmapped files")) if key in counts)
        if "input_rows" not in counts and "rows" in counts:
            input_info += (" · " if input_info else "") + "Input rows: " + number(counts["rows"])
        stages = "".join(f"<tr><td>{escape(r['stage'])}</td><td>{STAGE_LABELS[r['status']]}</td>"
                         f"<td>{number(r.get('seconds')) + ' s' if 'seconds' in r else '—'}</td></tr>"
                         for r in state["stages"])
        recent = "\n".join(escape(line) for line in reversed(state["recent"])) or "Waiting for the first milestone."
        link = f'<a class="report-link" href="{escape(state["dashboard"], quote=True)}">Open results dashboard →</a>' if state["dashboard"] else ""
        timing = {key: state[key] for key in ("started_at", "stage_started_at", "last_signal_at", "finished_at", "stage_finished_at")}
        if monitoring_stopped:
            timing["finished_at"] = timing["finished_at"] or state["monitor_stopped_at"]
            timing["stage_finished_at"] = timing["stage_finished_at"] or state["monitor_stopped_at"]
        year_text = ""
        if "start_year" in climate and "end_year" in climate:
            year_text = f" · {climate['start_year']}–{climate['end_year']}"
        message = ("This page refreshes every 2 seconds. A heartbeat confirms the local worker is running; CDS may still be queued."
                   if active else "The run is no longer active. Progress and results remain available locally.")
        if state["legacy"]:
            message = ("Attached to an existing run. No process heartbeat is available; silence does not establish whether it is still running. "
                       "Detailed row progress requires starting a run with the updated version.")
        if monitoring_stopped:
            message = "Monitoring stopped. The pipeline was not stopped; displayed status is the last observation."
        status_label = state["status"].replace("_", " ")
        if monitoring_stopped:
            status_label = "snapshot · last observed " + status_label
        run_timer_label = "Time since attachment" if state["legacy"] else "Run elapsed"
        stage_timer_label = "Time since stage observed" if state["legacy"] else "Stage elapsed"
        signal_timer_label = "Last observed change" if state["legacy"] else "Last local signal"
        return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
{refresh}<title>Pavement pipeline progress</title>
<style>
:root{{font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#173249;background:#eef3f7}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1120px;margin:32px auto;padding:0 24px 40px}}
header{{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:22px}}
h1{{font-size:27px;margin:0;letter-spacing:-.6px}}h2{{font-size:15px;margin:0}}p{{margin:7px 0;color:#5b6c7b}}
.eyebrow{{font-size:11px;letter-spacing:.15em;font-weight:700;color:#477389;margin:0 0 6px;text-transform:uppercase}}
.badge{{font-size:12px;font-weight:700;white-space:nowrap;border:1px solid #bfd9d4;background:#e2f2ed;color:#126353;padding:7px 13px;border-radius:20px}}
.hero,.panel{{background:white;border:1px solid #d9e4eb;border-radius:14px;padding:23px;margin-bottom:18px;box-shadow:0 4px 20px #19364a04}}
.phase{{font-size:24px;font-weight:650;letter-spacing:-.4px;margin:7px 0 14px;color:#163950}}
.stage{{font:13px ui-monospace,SFMono-Regular,monospace;color:#607a8c}}
.timers,.counters{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.timers{{margin-top:18px;border-top:1px solid #e6edf1;padding-top:16px}}
.metric span{{display:block;color:#647b8b;font-size:12px}}.metric strong{{font-size:21px;font-variant-numeric:tabular-nums}}
.grid{{display:grid;grid-template-columns:1.1fr 1fr;gap:18px}}.meter+.meter{{border-top:1px solid #e5edf2;padding-top:20px;margin-top:20px}}
.meter-label{{display:flex;justify-content:space-between;gap:12px;margin-bottom:10px}}.meter p{{font-size:12px}}
progress,.empty-bar{{display:block;width:100%;height:10px;border:0;border-radius:8px;background:#e9eff3;overflow:hidden}}
progress::-webkit-progress-bar{{background:#e9eff3}}progress::-webkit-progress-value{{background:#219c89;border-radius:8px}}progress::-moz-progress-bar{{background:#219c89;border-radius:8px}}
.counters{{margin:20px 0 6px}}.note{{font-size:12px;margin-top:17px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;padding:9px 5px;border-bottom:1px solid #e9eff3}}th{{color:#6b7b89;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}td:first-child{{font-family:ui-monospace,SFMono-Regular,monospace}}
details summary{{font-weight:650;cursor:pointer}}dl{{display:grid;grid-template-columns:170px 1fr;gap:9px 14px;font-size:12px}}dt{{color:#607789}}dd{{margin:0;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;color:#435c6e;max-height:350px;overflow:auto;margin:18px 0 0}}.report-link{{display:inline-block;background:#126b61;color:white;text-decoration:none;border-radius:8px;padding:9px 14px;margin-top:12px}}
@media(max-width:760px){{main{{padding:0 14px;margin-top:18px}}.grid{{grid-template-columns:1fr;gap:0}}header{{align-items:flex-start}}h1{{font-size:23px}}.hero,.panel{{padding:18px}}.phase{{font-size:20px}}dl{{grid-template-columns:1fr;gap:3px}}dd{{margin-bottom:10px}}}}
</style></head><body><main>
<header><div><p class="eyebrow">Pavement Analytics · local run</p><h1>Pipeline progress</h1></div><span class="badge">{escape(status_label)}</span></header>
<section class="hero"><div class="stage">{escape(state['stage'])}</div><div class="phase">{escape(state['phase'])}</div>
<p>{escape(input_info) if input_info else 'Reading setup and input information…'}</p>
<div class="timers"><div class="metric"><span>{run_timer_label}</span><strong id="run-time">—</strong></div><div class="metric"><span>{stage_timer_label}</span><strong id="stage-time">—</strong></div><div class="metric"><span>{signal_timer_label}</span><strong id="signal-time">—</strong></div></div>{link}</section>
<div class="grid"><section class="panel"><p class="eyebrow">ERA5 climate{year_text}</p>{meters}
<div class="counters"><div class="metric"><span>Cache hits</span><strong>{number(climate.get('cache_hits'))}</strong></div><div class="metric"><span>Requests started</span><strong>{number(climate.get('requests'))}</strong></div><div class="metric"><span>Dataset</span><strong style="font-size:15px">{escape(climate.get('dataset', '—'))}</strong></div></div>
<p class="note">{message} Location counts are progress through the work, not an estimate of time remaining.</p></section>
<section class="panel"><h2>Pipeline stages</h2><table><thead><tr><th>Stage</th><th>Status</th><th>Time</th></tr></thead><tbody>{stages or '<tr><td colspan="3">Setup is in progress.</td></tr>'}</tbody></table></section></div>
<section class="panel"><details><summary>Input, mapping and cache paths</summary><dl>{file_rows}</dl></details></section>
<section class="panel"><details open><summary>Recent progress · newest first</summary><pre>{recent}</pre></details></section>
<p class="note">This page uses local files only and makes no network requests. Raw subprocess diagnostics stay in the run logs.</p>
</main><script>
const timing = {json.dumps(timing, allow_nan=False)};
function duration(seconds){{seconds=Math.max(0,Math.floor(seconds));const h=Math.floor(seconds/3600),m=Math.floor(seconds%3600/60),s=seconds%60;return (h?h+'h ':'')+(m?m+'m ':'')+s+'s';}}
function tick(){{const now=Date.now()/1000;document.getElementById('run-time').textContent=duration((timing.finished_at??now)-timing.started_at);document.getElementById('stage-time').textContent=duration((timing.stage_finished_at??now)-timing.stage_started_at);document.getElementById('signal-time').textContent=duration(now-timing.last_signal_at)+' ago';}}
tick();setInterval(tick,1000);
</script></body></html>'''
