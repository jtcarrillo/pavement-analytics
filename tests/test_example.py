import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
from temp_support import temporary_directory
import unittest
from unittest.mock import patch

from pavement_analytics.curation import build_pairs
from pavement_analytics.io import load_sources
from pavement_analytics.security import SafeError


SPEC = importlib.util.spec_from_file_location(
    "make_synthetic_example", Path(__file__).resolve().parents[1] / "tools/make_synthetic_example.py")
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


class SyntheticExampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.public = self.root / "code"
        self.public.mkdir()
        self.environment = patch.dict(os.environ, {
            "PAVEMENT_PRIVATE_EXECUTION": "1", "GITHUB_ACTIONS": "false"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.source = patch.object(example, "source_root", return_value=self.public)
        self.source.start()
        self.addCleanup(self.source.stop)

    def test_external_example_audits_and_determinism(self):
        first, second = self.root / "first", self.root / "second"
        example.create(first)
        example.create(second)
        self.assertEqual((first / "inputs/synthetic_inspections.csv").read_bytes(),
                         (second / "inputs/synthetic_inspections.csv").read_bytes())
        config = json.loads((first / "run.json").read_text())
        raw, _ = load_sources(first / "inputs", config["input_files"], config["columns"])
        pairs, rejected, transitions, counts = build_pairs(raw, config["columns"])
        self.assertEqual(counts, {"input_rows": 726, "accepted_rows": 720,
                                 "rejected_rows": 6, "pair_rows": 572,
                                 "transition_count": 576, "initial_anchors": 144})
        self.assertEqual(set(rejected.reason), {"invalid_date", "before_cutoff", "invalid_pci",
                                               "missing_section", "missing_group", "unknown_surface"})
        self.assertEqual(set(transitions.reason) - {"paired"}, {
            "increase_reset", "same_day_collapsed", "below_minimum_interval", "construction_reset"})
        self.assertEqual(set(pairs.material), {"AC", "APC", "PCC"})
        self.assertEqual(list(self.public.iterdir()), [])

    def test_populated_destination_is_not_changed(self):
        workspace = self.root / "existing"
        workspace.mkdir()
        sentinel = workspace / "preserve.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(SafeError, "EXAMPLE_REQUIRES_EMPTY_WORKSPACE"):
            example.create(workspace)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertEqual(list(workspace.iterdir()), [sentinel])

    def test_source_and_other_git_destinations_are_refused(self):
        with self.assertRaisesRegex(SafeError, "WORKSPACE_MUST_BE_OUTSIDE_PUBLIC_TREE"):
            example.create(self.public / "generated")
        other = self.root / "other"
        (other / ".git").mkdir(parents=True)
        with self.assertRaisesRegex(SafeError, "WORKSPACE_MUST_NOT_BE_VERSIONED"):
            example.create(other / "generated")
        self.assertFalse((self.public / "generated").exists())
        self.assertFalse((other / "generated").exists())

    def test_cli_refuses_public_execution_without_echoing_paths(self):
        stream = io.StringIO()
        workspace = self.root / "unpublished-path"
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                status = example.main(["--workspace", str(workspace)])
        self.assertEqual(status, 2)
        self.assertEqual(stream.getvalue().strip(), "GITHUB_ACTIONS_DATA_EXECUTION_FORBIDDEN")
        self.assertFalse(workspace.exists())


if __name__ == "__main__":
    unittest.main()
