"""Milestone callbacks reflect durable stage state without raw diagnostics."""
import contextlib
import io
import json
from pathlib import Path
import unittest

import numpy as np

from pavement_analytics.original.common import Run, sha256
from temp_support import temporary_directory


class OriginalProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / "run"
        self.config = {"surfaces": ["PCC"]}
        self.events = []

    def make_run(self, *, resume=False, progress=None):
        return Run(self.out, "generated-input-fingerprint", self.config, [], resume,
                   self.events.append if progress is None else progress)

    @staticmethod
    def quiet(function, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return function(*args)

    def test_stage_events_surround_work_and_follow_saved_artifact_verification(self):
        seen = []
        def observe(event):
            saved = json.loads((self.out / "run_report.json").read_text())
            seen.append((event, saved))
        run = self.make_run(progress=observe)
        artifact = self.out / "generated.txt"
        def stage():
            self.assertEqual(seen[-1][0]["status"], "started")
            self.assertEqual(seen[-1][1]["stages"]["01_split"]["status"], "running")
            run.warn("GENERATED_WARNING", "Generated private diagnostic", np.int64(2))
            print("Generated raw subprocess-like output")
            artifact.write_text("generated content")
            return {"rows": np.int64(3), "missing_metric": float("nan"),
                    "surfaces": {"PCC": np.int64(3)}}, [artifact]
        result = self.quiet(run.stage, "01_split", stage)
        self.quiet(run.finish, "completed_with_warnings")
        events = [event for event, _ in seen]
        self.assertEqual([(e["event"], e.get("status")) for e in events],
                         [("stage", "started"), ("warning", None),
                          ("stage", "completed"), ("run", "completed_with_warnings")])
        complete, snapshot = seen[2]
        self.assertEqual(complete["stats"], result)
        self.assertIsNone(complete["stats"]["missing_metric"])
        self.assertIsInstance(complete["seconds"], float)
        self.assertEqual(snapshot["stages"]["01_split"]["artifacts"],
                         [{"path": artifact.name, "sha256": sha256(artifact)}])
        self.assertEqual(complete["log"], str(self.out / "logs/01_split.log"))
        self.assertEqual(events[1], {"event": "warning", "stage": "01_split",
                                     "code": "GENERATED_WARNING", "count": 2})
        self.assertEqual(events[-1]["warnings"], 1)
        self.assertEqual(events[-1]["summary"]["PCC"]["eligible_inspections"], 3)
        self.assertEqual(events[-1]["report"], str(self.out / "run_report.json"))
        serialized = json.dumps(events, allow_nan=False)
        self.assertNotIn("Generated private diagnostic", serialized)
        self.assertNotIn("Generated raw subprocess-like output", serialized)
        log = (self.out / "logs/01_split.log").read_text()
        self.assertIn("Generated private diagnostic", log)
        self.assertIn("Generated raw subprocess-like output", log)

    def test_skipped_stage_and_verified_resume_report_reuse_without_execution(self):
        run = self.make_run()
        self.quiet(run.stage, "05_fit_curves", lambda: ({"skipped": "optional calculation disabled"}, []))
        self.assertEqual(self.events[-1]["status"], "skipped")
        self.assertEqual(self.events[-1]["stats"], {"skipped": "optional calculation disabled"})
        self.events.clear()
        resumed = self.make_run(resume=True)
        def must_not_run():
            self.fail("Completed optional stage must be reused")
        self.quiet(resumed.stage, "05_fit_curves", must_not_run)
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0]["status"], "reused")
        self.assertEqual(self.events[0]["stats"], {"skipped": "optional calculation disabled"})

    def test_tampered_checkpoint_never_emits_reuse(self):
        run = self.make_run()
        artifact = self.out / "generated.txt"
        artifact.write_text("generated original")
        self.quiet(run.stage, "01_split", lambda: ({"rows": 1}, [artifact]))
        artifact.write_text("generated altered")
        self.events.clear()
        with self.assertRaisesRegex(ValueError, "Checkpoint artifact changed"):
            self.make_run(resume=True)
        self.assertEqual(self.events, [])

    def test_failure_and_interrupt_are_reported_after_checkpoint_without_raw_error(self):
        for failure, status in ((ValueError("Generated private failure detail"), "failed"),
                                (KeyboardInterrupt("Generated interrupt detail"), "interrupted")):
            with self.subTest(status=status):
                self.out = self.root / status
                self.events.clear()
                run = self.make_run()
                def stage():
                    raise failure
                with self.assertRaises(type(failure)):
                    self.quiet(run.stage, "04_climate_PCC", stage)
                self.assertEqual([e["status"] for e in self.events], ["started", status])
                final = self.events[-1]
                self.assertIsInstance(final["seconds"], float)
                self.assertNotIn("error", final)
                self.assertNotIn(str(failure), json.dumps(self.events))
                saved = json.loads((self.out / "run_report.json").read_text())
                self.assertEqual(saved["stages"]["04_climate_PCC"]["status"], "failed")
                self.assertEqual(saved["stages"]["04_climate_PCC"]["error"], str(failure))

    def test_callback_cannot_mutate_stored_stats_and_warning_count_is_optional(self):
        def observe(event):
            self.events.append(event)
            if event.get("status") == "completed":
                event["stats"]["rows"] = 999
        run = self.make_run(progress=observe)
        self.quiet(run.warn, "GENERATED_WARNING", "Generated detail")
        self.assertNotIn("count", self.events[0])
        result = self.quiet(run.stage, "01_split", lambda: ({"rows": 4}, []))
        self.assertEqual(result["rows"], 4)
        self.assertEqual(run.data["stages"]["01_split"]["stats"]["rows"], 4)

    def test_omitting_callback_retains_standalone_output(self):
        run = Run(self.out, "generated-input-fingerprint", self.config, [])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run.stage("01_split", lambda: ({"rows": 2}, []))
            run.finish("completed")
        self.assertIn("START 01_split", output.getvalue())
        self.assertIn("DONE  01_split", output.getvalue())
        self.assertIn("COMPLETED:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
