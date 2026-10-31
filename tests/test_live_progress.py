"""The progress window is a bounded local view, never an execution checkpoint."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from pavement_analytics.live_progress import LiveProgress
from pavement_analytics.security import Boundary, SafeError
from temp_support import temporary_directory


class LiveProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.public = root / "public"
        self.public.mkdir()
        self.boundary = Boundary.create(self.public, root / "run")

    def test_page_initializes_and_finishes_without_network_or_refresh_after_completion(self):
        with patch("pavement_analytics.live_progress.time.time", return_value=100):
            view = LiveProgress(self.boundary)
        self.assertEqual(view.path.name, "progress.html")
        html = view.path.read_text()
        self.assertIn('http-equiv="refresh"', html)
        for text in ("fetch(", "XMLHttpRequest", '<script src=', '<link href=', "http://", "https://"):
            self.assertNotIn(text, html)
        with patch("pavement_analytics.live_progress.time.time", return_value=120):
            view.finish("completed")
        self.assertNotIn('http-equiv="refresh"', view.path.read_text())
        state = json.loads(view.state_path.read_text())
        self.assertEqual((state["started_at"], state["finished_at"]), (100, 120))

    def test_climate_counts_are_partial_and_phase_does_not_claim_transfer_progress(self):
        view = LiveProgress(self.boundary)
        view({"event": "stage", "stage": "04_climate_PCC", "status": "started"})
        view({"event": "climate", "phase": "requesting", "location_index": 2,
              "total_locations": 5, "locations_completed": 1, "usable_locations": 1,
              "missing_locations": 0, "total_rows": 12, "rows_completed": 0,
              "cache_hits": 1, "requests": 1, "dataset": "era5-land", "secret": "PRIVATE_MARKER"})
        html = view.path.read_text()
        self.assertIn("Waiting for CDS / downloading", html)
        self.assertIn("1 / 5", html)
        self.assertIn("0 / 12", html)
        self.assertIn('value="20.000"', html)
        self.assertNotIn("PRIVATE_MARKER", html + view.state_path.read_text())
        view({"event": "climate", "phase": "calculating", "total_locations": 0,
              "locations_completed": 0, "total_rows": 0, "rows_completed": 0})
        self.assertIn("0 / 0", view.path.read_text())

    def test_paths_and_recent_lines_escape_html_and_ignore_unknown_payloads(self):
        view = LiveProgress(self.boundary)
        hostile = '/private/generated<script>alert("bad")</script>\n\u202e.csv'
        view({"event": "setup_files", "mapping": hostile, "secret": "PRIVATE_MARKER"})
        view({"event": "unknown", "message": "PRIVATE_MARKER"})
        view({"event": "warning", "stage": "01_split", "code": "GENERATED_WARNING",
              "message": "PRIVATE_MARKER", "count": 1})
        html = view.path.read_text()
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn('<script>alert(', html)
        self.assertNotIn("\u202e", html)
        self.assertNotIn("PRIVATE_MARKER", html + view.state_path.read_text())
        for _ in range(100):
            view({"event": "warning", "stage": "01_split", "code": "GENERATED_WARNING", "count": 1})
        self.assertEqual(len(view.state["recent"]), 80)

    def test_heartbeat_keeps_phase_and_marks_local_signal(self):
        view = LiveProgress(self.boundary)
        view({"event": "stage", "stage": "04_climate_PCC", "status": "started"})
        with patch("pavement_analytics.live_progress.time.time", return_value=500):
            view({"event": "heartbeat", "elapsed_seconds": 50, "idle_seconds": 45,
                  "message": "PRIVATE_MARKER"})
        state = json.loads(view.state_path.read_text())
        self.assertEqual(state["last_signal_at"], 500)
        self.assertEqual(state["phase"], "Started 04_climate_PCC")
        self.assertEqual(state["idle_seconds"], 45)
        self.assertNotIn("PRIVATE_MARKER", view.path.read_text())
        view({"event": "stage", "stage": "05_targets_PCC", "status": "started"})
        state = json.loads(view.state_path.read_text())
        self.assertNotIn("elapsed_seconds", state)
        self.assertNotIn("idle_seconds", state)

    def test_progress_window_location_is_retained_in_recent_log(self):
        view = LiveProgress(self.boundary)
        with patch("pavement_analytics.live_progress.FullProgress._render",
                   return_value=['Progress window: "/private/generated/progress.html"']):
            view({"event": "progress_window", "path": str(view.path)})
        self.assertIn("Progress window:", view.state["recent"][-1])

    def test_dashboard_link_must_remain_inside_workspace_and_failure_stops_refresh(self):
        view = LiveProgress(self.boundary)
        view({"event": "output", "dashboard": str(self.public / "outside.html")})
        self.assertIsNone(view.state["dashboard"])
        destination = self.boundary.path("pipeline/10_report/dashboard.html")
        view({"event": "output", "dashboard": str(destination)})
        self.assertEqual(view.state["dashboard"], destination.as_uri())
        for status in ("failed", "interrupted"):
            view.finish(status)
            self.assertEqual(view.state["status"], status)
            self.assertNotIn('http-equiv="refresh"', view.path.read_text())

    def test_write_failures_disable_window_but_preserve_console_sink(self):
        events = []
        view = LiveProgress(self.boundary, events.append)
        with patch.object(view, "_atomic", side_effect=OSError("generated failure")):
            view({"event": "stage", "stage": "01_split", "status": "started"})
        self.assertTrue(view.disabled)
        view({"event": "stage", "stage": "01_split", "status": "completed"})
        self.assertEqual(len(events), 2)

    def test_progress_symlink_cannot_overwrite_another_file(self):
        other = self.boundary.workspace.parent / "other.html"
        other.write_text("generated unchanged")
        (self.boundary.workspace / "progress.html").symlink_to(other)
        with self.assertRaises(SafeError):
            LiveProgress(self.boundary)
        self.assertEqual(other.read_text(), "generated unchanged")

    def test_legacy_monitor_labels_observed_times_without_claiming_heartbeat(self):
        view = LiveProgress(self.boundary)
        view({"event": "legacy_monitor"})
        view({"event": "stage", "stage": "04_climate_PCC", "status": "started"})
        html = view.path.read_text()
        self.assertIn("No process heartbeat is available", html)
        self.assertIn("Time since attachment", html)
        self.assertIn("Time since stage observed", html)
        self.assertNotIn("<span>Run elapsed</span>", html)
        self.assertNotIn("A heartbeat confirms", html)

    def test_stopping_monitor_freezes_observation_without_stopping_pipeline(self):
        view = LiveProgress(self.boundary)
        view({"event": "legacy_monitor"})
        view({"event": "stage", "stage": "04_climate_PCC", "status": "started"})
        previous_signal = view.state["last_signal_at"]
        with patch("pavement_analytics.live_progress.time.time", return_value=1000):
            view({"event": "monitor_stopped"})
        html = view.path.read_text()
        self.assertEqual(view.state["status"], "running")
        self.assertIsNone(view.state["finished_at"])
        self.assertEqual(view.state["monitor_stopped_at"], 1000)
        self.assertEqual(view.state["last_signal_at"], previous_signal)
        self.assertIn("Monitoring stopped. The pipeline was not stopped", html)
        self.assertIn("snapshot · last observed running", html)
        self.assertNotIn('http-equiv="refresh"', html)
        self.assertNotIn("interrupted", html)
        # This event cannot stop a modern run's live display.
        other = LiveProgress(self.boundary)
        other({"event": "monitor_stopped"})
        self.assertIsNone(other.state["monitor_stopped_at"])
        self.assertIn('http-equiv="refresh"', other.path.read_text())


if __name__ == "__main__":
    unittest.main()
