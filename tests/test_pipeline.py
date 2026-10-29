import contextlib
import io
import json
import os
from pathlib import Path
from temp_support import temporary_directory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from pavement_analytics.cli import main
from pavement_analytics.features import build
from pavement_analytics.curation import build_pairs
from pavement_analytics.modeling import search
from pavement_analytics.pipeline import run
from pavement_analytics.security import Boundary, SafeError
from test_analysis import COLUMNS, inspections


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.public = self.root / "code"
        self.public.mkdir()
        self.boundary = Boundary.create(self.public, self.root / "work")
        self.env = patch.dict(os.environ, {"GITHUB_ACTIONS": "false", "PAVEMENT_PRIVATE_EXECUTION": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        inputs = self.boundary.workspace / "inputs"
        inputs.mkdir()
        raw = inspections(8, 2, 4)
        raw = raw.drop(columns=[c for c in raw if c.startswith("_")])
        raw.to_csv(inputs / "generated.csv", index=False)
        self.config_path = self.boundary.workspace / "run.json"
        self.config = {"schema_version": 1, "input_root": "inputs", "input_files": ["generated.csv"],
                       "output": "runs/v1", "columns": COLUMNS, "features": ["PCI_start"],
                       "models": False}
        self.config_path.write_text(json.dumps(self.config))

    def test_complete_run_resume_and_output_tamper(self):
        self.assertEqual(run(self.boundary, self.config_path), "PRIVATE_RUN_COMPLETE")
        out = self.boundary.workspace / "runs/v1"
        report = json.loads((out / "run_report.json").read_text())
        self.assertEqual(report["accounting"]["pair_rows"], 48)
        self.assertEqual(run(self.boundary, self.config_path, resume=True), "VERIFIED_EXISTING_RUN")
        with (out / "dataset.csv").open("a") as stream:
            stream.write("changed\n")
        with self.assertRaisesRegex(SafeError, "OUTPUT_TAMPERED"):
            run(self.boundary, self.config_path, resume=True)

    def test_changed_inputs_invalidate_resume(self):
        run(self.boundary, self.config_path)
        source = self.boundary.workspace / "inputs/generated.csv"
        source.write_text(source.read_text().replace(",95,", ",94,", 1))
        with self.assertRaisesRegex(SafeError, "INPUT_OR_CODE_CHANGED"):
            run(self.boundary, self.config_path, resume=True)

    def test_input_path_escape_and_unknown_configuration_fail(self):
        self.config["unrecognized"] = True
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(SafeError, "INVALID_RUN_CONFIGURATION"):
            run(self.boundary, self.config_path)
        self.config.pop("unrecognized")
        self.config["input_files"] = ["../../code/file.csv"]
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(SafeError, "INPUT_PATH"):
            run(self.boundary, self.config_path)

    def test_actual_cli_sanitizes_failures_and_writes_private_log(self):
        self.config["input_files"] = ["generated-missing.csv"]
        self.config_path.write_text(json.dumps(self.config))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            result = main(["--workspace", str(self.boundary.workspace), "run", "--config", str(self.config_path)])
        self.assertEqual(result, 2)
        self.assertNotIn("generated-missing", stream.getvalue())
        self.assertNotIn(str(self.boundary.workspace), stream.getvalue())
        self.assertTrue((self.boundary.workspace / "logs/runtime.log").is_file())

    def test_model_split_holdout_and_saved_model(self):
        pairs = build_pairs(inspections(8, 3, 5), COLUMNS)[0]
        table, _ = build(pairs, ["PCI_start"], ["DetRatePerYear"], COLUMNS)
        out = self.boundary.workspace / "model"
        summary = search(table, ["PCI_start"], "DetRatePerYear", out, folds=3)
        fold_table = pd.read_csv(out / "folds.csv")
        self.assertEqual(fold_table.groupby("group_id").size().max(), 1)
        holdout = set(fold_table.loc[fold_table.partition.eq("holdout"), "group_id"])
        oof = pd.read_csv(out / "oof_predictions.csv")
        self.assertFalse(holdout & set(oof.group_id))
        self.assertIn(summary["status"], ["champion", "no_model"])
        saved = joblib.load(out / "model.joblib")
        self.assertEqual(saved["features"], ["PCI_start"])
        self.assertTrue(np.isfinite(summary["holdout"]["wMAE"]))

    def test_equal_baseline_selects_no_model(self):
        pairs = build_pairs(inspections(8, 1, 3), COLUMNS)[0]
        table, _ = build(pairs, ["PCI_start"], ["DetRatePerYear"], COLUMNS)
        table["DetRatePerYear"] = -1.0
        # A fixed candidate avoids unnecessary solver warnings on a constant target.
        with patch("pavement_analytics.modeling.candidates", return_value=iter([])):
            summary = search(table, ["PCI_start"], "DetRatePerYear", self.boundary.workspace / "constant", folds=3)
        self.assertEqual(summary["status"], "no_model")
        self.assertAlmostEqual(summary["holdout"]["wMAE"], 0)

    def test_complete_pipeline_models_pci_and_deduct_targets(self):
        source = self.boundary.workspace / "inputs/generated.csv"
        raw = pd.read_csv(source, dtype=str)
        raw["63L deduct"] = 100 - pd.to_numeric(raw.PCI)
        raw.to_csv(source, index=False)
        self.config.update(models=True, folds=3, deduct_families={"cracking": [63]})
        self.config_path.write_text(json.dumps(self.config))
        self.assertEqual(run(self.boundary, self.config_path), "PRIVATE_RUN_COMPLETE")
        out = self.boundary.workspace / "runs/v1"
        report = json.loads((out / "run_report.json").read_text())
        self.assertEqual({slot["target"] for slot in report["slots"]}, {"DetRatePerYear", "rate_cracking"})
        self.assertTrue(all(slot["status"] in {"champion", "no_model"} for slot in report["slots"]))
        self.assertEqual(run(self.boundary, self.config_path, resume=True), "VERIFIED_EXISTING_RUN")


if __name__ == "__main__":
    unittest.main()
