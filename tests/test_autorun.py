import contextlib
import io
import json
import os
import subprocess
from pathlib import Path
from temp_support import temporary_directory
import unittest
from unittest.mock import patch

import pandas as pd

from pavement_analytics.autorun import column_map, new_workspace, prepare
from pavement_analytics.cli import main
from pavement_analytics.io import sha256
from pavement_analytics.pipeline import run
from pavement_analytics.security import Boundary, SafeError


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.public = self.root / "code"
        self.public.mkdir()
        self.data = self.root / "private-data"
        self.data.mkdir()
        self.boundary = Boundary.create(self.public, self.root / "work")
        self.environment = patch.dict(os.environ, {"GITHUB_ACTIONS": "false"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.rows = pd.DataFrame([
            {"SectionUniqueID": f"S{section:02d}", "Inspection Date": str(year) + "-01-01",
             "PCI": 90 - 3 * (year - 2015), "Inspected Surface Type": "AC",
             " 43 l  deduct ": 2 + (year - 2015), "43M deduct": 0,
             "unreviewed_numeric": 1000 + section}
            for section in range(8) for year in (2015, 2017, 2019)])

    def input(self, name="data.csv"):
        path = self.data / name
        if path.suffix == ".xlsx":
            self.rows.to_excel(path, index=False)
        else:
            self.rows.to_csv(path, index=False, sep="\t" if path.suffix == ".tsv" else ",")
        return path

    def test_snapshot_normalizes_and_uses_only_declared_features(self):
        source = self.input()
        before = source.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            config_path = prepare(self.boundary, source, by_code=True)
        config = json.loads(config_path.read_text())
        self.assertEqual(config["deduct_families"], {"deduct_43": ["43"]})
        self.assertEqual(config["features"], ["PCI_start"])
        normalized = pd.read_csv(self.boundary.workspace / "inputs/input_0001.csv", dtype=str)
        self.assertIn("43L deduct", normalized)
        self.assertEqual(list(normalized.group), list(normalized.section))
        self.assertEqual(source.read_bytes(), before)
        report = json.loads((self.boundary.workspace / "setup_report.json").read_text())
        self.assertEqual(report["sources"][0]["original_sha256"], sha256(source))
        self.assertEqual(report["grouping"][0]["strategy"], "section_identity")
        self.assertEqual(list(self.public.iterdir()), [])

    def test_xlsx_and_tsv_are_discovered_without_recursion(self):
        self.input("a.xlsx")
        self.input("b.tsv")
        nested = self.data / "nested"
        nested.mkdir()
        self.rows.to_csv(nested / "not-discovered.csv", index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            config = json.loads(prepare(self.boundary, self.data, by_code=True).read_text())
        self.assertEqual(config["input_files"], ["input_0001.csv", "input_0002.csv"])

    def test_options_allow_explicit_features_and_families(self):
        source = self.input()
        options = self.data / "options.json"
        options.write_text(json.dumps({"features": ["PCI_start", "unreviewed_numeric_start"],
                                       "deduct_families": {"custom": ["43"]}, "models": False}))
        with contextlib.redirect_stdout(io.StringIO()):
            config = json.loads(prepare(self.boundary, source, options).read_text())
        self.assertEqual(config["deduct_families"], {"custom": ["43"]})
        self.assertEqual(config["features"], ["PCI_start", "unreviewed_numeric_start"])
        self.assertTrue((self.boundary.workspace / "analysis_options.json").is_file())

    def test_tilde_workspace_cannot_overwrite_existing_files(self):
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "keep.txt").write_text("keep")
        with patch.dict(os.environ, {"HOME": str(self.root)}):
            with self.assertRaisesRegex(SafeError, "REQUIRES_EMPTY_WORKSPACE"):
                new_workspace(Path("~/existing"))
        self.assertEqual((existing / "keep.txt").read_text(), "keep")

    def test_mixed_grouping_strategies_are_rejected(self):
        self.input("a.csv")
        other = self.rows.copy()
        other["group_id"] = "G1"
        other.to_csv(self.data / "b.csv", index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SafeError, "MIXED_GROUPING"):
                prepare(self.boundary, self.data, by_code=True)

    def test_ambiguous_headers_require_override(self):
        headers = ["section", "SectionID", "PCI", "Date", "Surface"]
        with self.assertRaisesRegex(SafeError, "AMBIGUOUS_COLUMNS"):
            column_map(headers, {})
        self.assertEqual(column_map(headers, {"section": "section"})["section"], "section")

    def test_public_source_links_and_nonempty_destination_are_refused(self):
        source = self.input()
        public_source = self.public / "forbidden.csv"
        public_source.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(SafeError, "SOURCE_MUST_BE_OUTSIDE"):
            prepare(self.boundary, public_source, by_code=True)
        linked = self.data / "link.csv"
        linked.symlink_to(source)
        with self.assertRaisesRegex(SafeError, "SYMLINK"):
            prepare(self.boundary, linked, by_code=True)
        with self.assertRaisesRegex(SafeError, "REQUIRES_EMPTY_WORKSPACE"):
            new_workspace(self.data)

    def test_actual_cli_runs_without_environment_and_hides_paths(self):
        source = self.input()
        options = self.data / "options.json"
        options.write_text(json.dumps({"models": False}))
        workspace = self.root / "analyzed"
        stream = io.StringIO()
        with patch.dict(os.environ, {}, clear=False), patch("pavement_analytics.cli.source_root", return_value=self.public):
            os.environ.pop("PAVEMENT_PRIVATE_EXECUTION", None)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                result = main(["analyze", str(source), "--workspace", str(workspace),
                               "--options", str(options), "--by-code"])
            self.assertNotIn("PAVEMENT_PRIVATE_EXECUTION", os.environ)
        self.assertEqual(result, 0, stream.getvalue())
        self.assertEqual(stream.getvalue().strip(), "PRIVATE_RUN_COMPLETE")
        self.assertTrue((workspace / "results/run_report.json").is_file())
        report = json.loads((workspace / "results/run_report.json").read_text())
        self.assertEqual({slot["target"] for slot in report["slots"]}, {"rate_deduct_43", "DetRatePerYear"})
        saved = Boundary.create(self.public, workspace)
        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "1"}):
            self.assertEqual(run(saved, workspace / "run.json", resume=True), "VERIFIED_EXISTING_RUN")
            snapshot = workspace / "originals/input_0001.csv"
            snapshot.write_text(snapshot.read_text() + "\n")
            with self.assertRaisesRegex(SafeError, "SOURCE_SNAPSHOT_CHANGED"):
                run(saved, workspace / "run.json", resume=True)

    def test_default_workspace_and_browser_failure_preserve_completed_run(self):
        source = self.input()
        options = self.data / "options.json"
        options.write_text(json.dumps({"models": False}))
        stream = io.StringIO()
        with patch("pathlib.Path.home", return_value=self.root), patch(
                "subprocess.run", side_effect=subprocess.TimeoutExpired("browser", 15)) as browser:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                code = main(["analyze", str(source), "--options", str(options), "--by-code", "--open"])
        self.assertEqual(code, 0, stream.getvalue())
        self.assertEqual(stream.getvalue().strip(), "PRIVATE_RUN_COMPLETE")
        runs = list((self.root / "pavement-runs").iterdir())
        self.assertEqual(len(runs), 1)
        self.assertTrue((runs[0] / "results/run_report.json").is_file())
        self.assertIn("DASHBOARD_OPEN_FAILED", (runs[0] / "logs/runtime.log").read_text())
        self.assertIs(browser.call_args.kwargs["stdout"], browser.call_args.kwargs["stderr"])

    def test_github_actions_is_refused_before_workspace_creation(self):
        workspace = self.root / "forbidden"
        stream = io.StringIO()
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                code = main(["analyze", str(self.data), "--workspace", str(workspace)])
        self.assertEqual(code, 2)
        self.assertEqual(stream.getvalue().strip(), "GITHUB_ACTIONS_DATA_EXECUTION_FORBIDDEN")
        self.assertFalse(workspace.exists())

    def test_failure_is_private_and_does_not_echo_source(self):
        source = self.input()
        self.rows.drop(columns="PCI").to_csv(source, index=False)
        workspace = self.root / "failed"
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = main(["analyze", str(source), "--workspace", str(workspace), "--by-code"])
        self.assertEqual(code, 2)
        self.assertEqual(stream.getvalue().strip(), "UNRECOGNIZED_COLUMNS_USE_PRIVATE_OPTIONS")
        self.assertIn(str(source), (workspace / "logs/runtime.log").read_text())


if __name__ == "__main__":
    unittest.main()
