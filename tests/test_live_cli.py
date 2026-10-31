"""The browser opens before long work and observes durable completion or failure."""
import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from pavement_analytics.cli import main
from pavement_analytics.security import SafeError
from temp_support import temporary_directory


class LiveCliTests(unittest.TestCase):
    def setUp(self):
        temp = temporary_directory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.public = self.root / "source"
        self.public.mkdir()
        self.work = self.root / "run"
        self.raw = self.root / "raw"
        self.raw.mkdir()

    def run_full(self, flags, *, fail=False):
        order = []

        def open_window(path):
            order.append("window")
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads((self.work / "progress.json").read_text())["status"], "running")
            return True

        def operation(boundary, *args, progress, **kwargs):
            order.append("work")
            progress({"event": "setup_started", "workspace": str(boundary.workspace)})
            progress({"event": "stage", "stage": "04_climate_PCC", "status": "started"})
            progress({"event": "climate", "phase": "requesting", "location_index": 1,
                      "total_locations": 2, "locations_completed": 0})
            if fail:
                raise SafeError("GENERATED_SETUP_FAILURE")
            dashboard = boundary.output("dashboard.html")
            dashboard.write_text("<html>Generated results</html>")
            boundary.output("full_result.json").write_text(json.dumps({"dashboard": str(dashboard)}))
            progress({"event": "output", "dashboard": str(dashboard)})
            progress({"event": "run", "status": "completed"})
            return "PRIVATE_FULL_PIPELINE_COMPLETE"

        stream = io.StringIO()
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}), \
                patch("pavement_analytics.cli.source_root", return_value=self.public), \
                patch("pavement_analytics.full.full", side_effect=operation), \
                patch("pavement_analytics.monitor.open_local_window", side_effect=open_window), \
                patch("subprocess.run") as browser, \
                contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = main(["full", str(self.raw), "--workspace", str(self.work), *flags])
        return code, stream.getvalue(), order, browser

    def test_open_shows_live_window_before_work_and_final_dashboard_after_success(self):
        code, output, order, browser = self.run_full(["--open"])
        self.assertEqual(code, 0)
        self.assertEqual(order, ["window", "work"])
        browser.assert_called_once()
        self.assertIn((self.work / "dashboard.html").as_uri(), browser.call_args.args[0])
        page = (self.work / "progress.html").read_text()
        self.assertIn("Open results dashboard", page)
        self.assertNotIn('http-equiv="refresh"', page)
        self.assertIn("Progress window:", output)

    def test_progress_only_and_quiet_still_create_a_live_window(self):
        code, output, order, browser = self.run_full(["--progress", "--quiet"])
        self.assertEqual(code, 0)
        self.assertEqual(order, ["window", "work"])
        browser.assert_not_called()
        self.assertEqual(output.strip(), "PRIVATE_FULL_PIPELINE_COMPLETE")
        self.assertTrue((self.work / "progress.html").is_file())

    def test_failure_remains_visible_and_stops_browser_refresh(self):
        code, output, order, browser = self.run_full(["--progress"], fail=True)
        self.assertEqual(code, 2)
        self.assertEqual(order, ["window", "work"])
        browser.assert_not_called()
        state = json.loads((self.work / "progress.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertNotIn('http-equiv="refresh"', (self.work / "progress.html").read_text())
        self.assertIn("GENERATED_SETUP_FAILURE", output)


if __name__ == "__main__":
    unittest.main()
