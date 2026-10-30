"""Attach to existing runs without exposing raw logs or rewriting checkpoints."""
import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from pavement_analytics.monitor import _LegacyClimate, _Tail, _select_workspace, monitor_existing
from pavement_analytics.security import SafeError
from temp_support import temporary_directory


class MonitorTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "run"
        (self.work / "pipeline/logs").mkdir(parents=True)
        self.setup = self.work / "full_setup_report.json"
        self.report = self.work / "pipeline/run_report.json"
        self.setup.write_text(json.dumps({"mapping": {"original_path": "/private/generated/mapping.json",
                                                     "location_directory": "/private/generated/locations"}}))
        self.state = {"status": "running", "stages": {"04_climate_PCC": {"status": "running", "stats": {}}}}
        self.report.write_text(json.dumps(self.state))
        self.env = patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "1", "GITHUB_ACTIONS": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def monitor(self, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return monitor_existing(self.work, **options)

    def test_old_log_view_preserves_inputs_and_never_exposes_raw_diagnostics(self):
        log = self.work / "pipeline/logs/04_climate_PCC.log"
        log.write_text("\n".join([
            "[INFO] Input rows: 12, unique locations: 2",
            "[INFO] (1/2) loading lat=0, lon=1",
            "[INFO] Requesting reanalysis-era5-land-timeseries PRIVATE_REQUEST_MARKER",
            "[INFO]   resolved via: era5-land (snap 0 cells / 0 km)",
            "[INFO] (2/2) loading lat=0, lon=2",
            "[ERROR] Failed to load (0, 2): PRIVATE_EXCEPTION_MARKER",
            "PRIVATE_CREDENTIAL_MARKER", "",
        ]))
        originals = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                     for path in (self.setup, self.report, log)}
        page = self.monitor(once=True)
        self.assertTrue(page.is_relative_to(self.work / "monitors"))
        state = json.loads(page.with_suffix(".json").read_text())
        self.assertTrue(state["legacy"])
        self.assertEqual(state["climate"]["locations_completed"], 2)
        self.assertEqual(state["climate"]["usable_locations"], 1)
        self.assertEqual(state["climate"]["missing_locations"], 1)
        self.assertEqual(state["climate"]["phase"], "calculating")
        self.assertNotIn("rows_completed", state["climate"])
        self.assertNotIn("elapsed_seconds", state)
        for marker in ("PRIVATE_REQUEST_MARKER", "PRIVATE_EXCEPTION_MARKER", "PRIVATE_CREDENTIAL_MARKER", "loading lat="):
            self.assertNotIn(marker, page.read_text() + json.dumps(state))
        self.assertEqual(originals, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in originals})

    def test_existing_live_page_is_opened_without_any_writes(self):
        page, state = self.work / "progress.html", self.work / "progress.json"
        page.write_text("generated current page")
        state.write_text("{}")
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (page, state, self.report, self.setup)}
        with patch("pavement_analytics.monitor.open_local_window", return_value=True) as opened:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(monitor_existing(self.work, open_window=True), page)
        self.assertIn("Progress window: " + json.dumps(str(page)), output.getvalue())
        opened.assert_called_once_with(page)
        self.assertFalse((self.work / "monitors").exists())
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_partial_report_is_retried_and_terminal_run_ends_monitoring(self):
        self.report.write_text('{"status":')
        def complete_report(_):
            self.state.update(status="completed")
            self.state["stages"]["04_climate_PCC"]["status"] = "completed"
            self.report.write_text(json.dumps(self.state))
        with patch("pavement_analytics.monitor.time.sleep", side_effect=complete_report) as sleep:
            page = self.monitor()
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(json.loads(page.with_suffix(".json").read_text())["status"], "completed")

    def test_active_surface_remains_current_after_completed_surface_is_replayed(self):
        self.state["stages"] = {
            "04_climate_PCC": {"status": "completed", "stats": {}},
            "07_modeling_PCC": {"status": "completed", "stats": {}},
            "04_climate_APC": {"status": "running", "stats": {}},
        }
        self.report.write_text(json.dumps(self.state))
        page = self.monitor(once=True)
        state = json.loads(page.with_suffix(".json").read_text())
        self.assertEqual(state["stage"], "04_climate_APC")
        self.assertEqual(state["stage_status"], "started")
        self.assertEqual(state["status"], "running")

    def test_once_and_interrupt_stop_only_the_monitor(self):
        for mode in ("once", "interrupt"):
            with self.subTest(mode=mode):
                before = self.report.read_bytes()
                if mode == "once":
                    page = self.monitor(once=True)
                else:
                    with patch("pavement_analytics.monitor.time.sleep", side_effect=KeyboardInterrupt):
                        page = self.monitor()
                state = json.loads(page.with_suffix(".json").read_text())
                self.assertEqual(state["status"], "running")
                self.assertEqual(self.report.read_bytes(), before)
                self.assertTrue(state["monitor_stopped"])
                self.assertNotIn('<meta http-equiv="refresh"', page.read_text())

    def test_latest_selection_and_missing_run_do_not_create_directories(self):
        with patch("pavement_analytics.monitor.Path.home", return_value=self.root):
            with self.assertRaisesRegex(SafeError, "NO_EXISTING_FULL_RUN_FOUND"):
                _select_workspace(None)
            self.assertFalse((self.root / "pavement-runs").exists())
            roots = [self.root / "pavement-runs" / name for name in ("run-old", "run-new")]
            for number, root in enumerate(roots):
                root.mkdir(parents=True)
                marker = root / "full_setup_report.json"
                marker.write_text("{}")
                os.utime(marker, (100 + number, 100 + number))
            self.assertEqual(_select_workspace(None).workspace, roots[1])
        missing = self.root / "does-not-exist"
        with self.assertRaisesRegex(SafeError, "PROGRESS_WORKSPACE_NOT_FOUND"):
            _select_workspace(missing)
        self.assertFalse(missing.exists())

    def test_symlink_log_is_rejected_and_untrusted_stage_names_are_ignored(self):
        self.state["stages"]["../../PRIVATE_MARKER"] = {"status": "running"}
        self.report.write_text(json.dumps(self.state))
        outside = self.root / "outside.log"
        outside.write_text("PRIVATE_MARKER")
        log = self.work / "pipeline/logs/04_climate_PCC.log"
        log.symlink_to(outside)
        with self.assertRaisesRegex(SafeError, "SYMLINK_PATH_FORBIDDEN"):
            self.monitor(once=True)
        self.assertEqual(outside.read_text(), "PRIVATE_MARKER")

    def test_tail_waits_for_full_lines_and_structured_worker_messages_are_not_double_counted(self):
        path = self.root / "generated.log"
        path.write_bytes(b"[INFO] Input rows: 3, unique locations: 1")
        tail = _Tail()
        self.assertEqual(tail.lines(path), [])
        with path.open("ab") as stream:
            stream.write(b"\n")
        self.assertEqual(tail.lines(path), ["[INFO] Input rows: 3, unique locations: 1"])
        parser = _LegacyClimate()
        event = parser.parse('PAVEMENT_PROGRESS {"event":"climate","phase":"cache_hit","cache_hits":1}')
        self.assertEqual(event["cache_hits"], 1)
        self.assertIsNone(parser.parse("[INFO] Reusing existing file: PRIVATE_MARKER"))


if __name__ == "__main__":
    unittest.main()
