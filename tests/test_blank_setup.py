"""Sparse generated exports verify explicit blank-deduct policy end to end."""

import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from temp_support import temporary_directory
from pavement_analytics.autorun import prepare
from pavement_analytics.cli import main
from pavement_analytics.pipeline import run
from pavement_analytics.security import Boundary, SafeError


class BlankDeductSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.public = self.root / "code"
        self.public.mkdir()
        self.data = self.root / "private-data"
        self.data.mkdir()
        self.environment = patch.dict(os.environ, {"GITHUB_ACTIONS": "false"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.options = self.data / "options.json"
        self.options.write_text(json.dumps({"models": False}))

    def records(self):
        return pd.DataFrame([
            {"SectionUniqueID": f"S{section:02d}", "Date": f"{2015 + 2 * visit}-01-01",
             "PCI": 94 - section - visit * (1 + section * 0.25),
             "Inspected Surface Type": "AC", "43L deduct": 4 + 2 * visit,
             "43M deduct": "", "52L deduct": "3" if visit == 2 else ""}
            for section in range(8) for visit in range(3)])

    def input(self, rows=None):
        source = self.data / "generated.csv"
        (self.records() if rows is None else rows).to_csv(source, index=False)
        return source

    def run_cli(self, source, workspace, *extra):
        console = io.StringIO()
        with patch("pavement_analytics.cli.source_root", return_value=self.public):
            with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
                status = main(["analyze", str(source), "--workspace", str(workspace),
                               "--options", str(self.options), *extra])
        return status, console.getvalue().strip()

    def test_default_missing_policy_preserves_exclusions(self):
        source = self.input()
        workspace = self.root / "missing"
        self.assertEqual(self.run_cli(source, workspace), (0, "PRIVATE_RUN_COMPLETE"))
        config = json.loads((workspace / "run.json").read_text())
        setup = json.loads((workspace / "setup_report.json").read_text())
        self.assertEqual(config["blank_deducts"], "missing")
        self.assertEqual(setup["blank_deducts"], "missing")
        report = json.loads((workspace / "results/run_report.json").read_text())
        deduct_slots = [slot for slot in report["slots"] if slot["target"].startswith("rate_")]
        self.assertEqual(len(deduct_slots), 11)
        self.assertTrue(all(slot["status"] == "no_valid_target" for slot in deduct_slots))
        pci_slot = next(slot for slot in report["slots"] if slot["target"] == "DetRatePerYear")
        self.assertEqual(pci_slot["valid_target_rows"], 16)
        audit = json.loads((workspace / "results/deduct_audit.json").read_text())
        self.assertEqual(audit["blank_deducts"], "missing")
        self.assertEqual(audit["zero_filled_deduct_cells"], {"start": 0, "end": 0})
        self.assertEqual(audit["invalid_deduct_pairs"], {"start": 16, "end": 16})

    def test_zero_policy_restores_family_rates_and_audits_without_mutating_inputs(self):
        source = self.input()
        before = source.read_bytes()
        workspace = self.root / "zero"
        self.assertEqual(self.run_cli(source, workspace, "--blank-deducts", "zero"),
                         (0, "PRIVATE_RUN_COMPLETE"))
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual((workspace / "originals/input_0001.csv").read_bytes(), before)
        normalized = pd.read_csv(workspace / "inputs/input_0001.csv", dtype=str, keep_default_na=False)
        self.assertTrue(normalized["43M deduct"].eq("").all())
        self.assertEqual(normalized["52L deduct"].eq("").sum(), 16)
        config = json.loads((workspace / "run.json").read_text())
        setup = json.loads((workspace / "results/setup_report.json").read_text())
        self.assertEqual(config["blank_deducts"], "zero")
        self.assertEqual(setup["blank_deducts"], "zero")
        pairs = pd.read_csv(workspace / "results/dataset.csv")
        self.assertTrue(pairs["43M deduct_start"].eq(0).all())
        self.assertTrue(pairs["43M deduct_end"].eq(0).all())
        np.testing.assert_allclose(pairs.rate_ac_thermal_crack, 2 * 365 / pairs.delta_days)
        expected_aging_growth = np.where(pairs.start_date.eq("2015-01-01"), 0, 3)
        np.testing.assert_allclose(pairs.rate_ac_aging, expected_aging_growth * 365 / pairs.delta_days)
        self.assertTrue(pairs.rate_pcc_thermal.isna().all())
        self.assertTrue((workspace / "results/dashboard.html").is_file())
        self.assertTrue((workspace / "results/target_summary.csv").is_file())
        audit = json.loads((workspace / "results/deduct_audit.json").read_text())
        self.assertEqual(audit["pair_rows"], 16)
        self.assertEqual(audit["blank_deduct_cells"], {"start": 32, "end": 24})
        self.assertEqual(audit["zero_filled_deduct_cells"], {"start": 32, "end": 24})
        self.assertEqual(audit["invalid_deduct_pairs"], {"start": 0, "end": 0})
        self.assertEqual(audit["export_gap_pairs"], {"start": 0, "end": 0, "either": 0})
        targets = {slot["target"]: slot for slot in audit["targets"]}
        self.assertEqual(targets["rate_ac_thermal_crack"]["valid_target_rows"], 16)
        self.assertEqual(targets["rate_ac_aging"]["valid_target_rows"], 16)
        self.assertEqual(targets["rate_pcc_thermal"]["valid_target_rows"], 0)
        manifest = json.loads((workspace / "results/manifest.json").read_text())
        self.assertIn("deduct_audit.json", manifest["outputs"])

    def test_sparse_family_targets_reach_modeling_with_group_disjoint_holdout(self):
        self.options.write_text(json.dumps({"models": True, "folds": 2}))
        source = self.input()
        workspace = self.root / "modeled"
        self.assertEqual(self.run_cli(source, workspace, "--blank-deducts", "zero"),
                         (0, "PRIVATE_RUN_COMPLETE"))
        report = json.loads((workspace / "results/run_report.json").read_text())
        fitted = {slot["target"]: slot for slot in report["slots"]
                  if slot["status"] in {"champion", "no_model"}}
        self.assertEqual(set(fitted), {"DetRatePerYear", "rate_ac_thermal_crack", "rate_ac_aging"})
        for target in fitted:
            directory = workspace / "results/materials/AC" / target
            folds = pd.read_csv(directory / "folds.csv")
            holdout = set(folds.loc[folds.partition.eq("holdout"), "group_id"])
            development = set(folds.loc[folds.partition.eq("development"), "group_id"])
            self.assertTrue(holdout)
            self.assertTrue(development)
            self.assertFalse(holdout & development)
            self.assertTrue((directory / "model.joblib").is_file())

    def test_options_policy_and_matching_cli_policy_are_supported(self):
        source = self.input()
        self.options.write_text(json.dumps({"models": False, "blank_deducts": "zero"}))
        for name, extra in (("options-only", ()), ("matching", ("--blank-deducts", "zero"))):
            with self.subTest(name=name):
                workspace = self.root / name
                self.assertEqual(self.run_cli(source, workspace, *extra), (0, "PRIVATE_RUN_COMPLETE"))
                pairs = pd.read_csv(workspace / "results/dataset.csv")
                self.assertTrue(pairs.rate_ac_thermal_crack.notna().all())

    def test_conflicting_policy_sources_are_rejected(self):
        source = self.input()
        self.options.write_text(json.dumps({"models": False, "blank_deducts": "missing"}))
        self.assertEqual(self.run_cli(source, self.root / "conflict", "--blank-deducts", "zero"),
                         (2, "BLANK_DEDUCT_OPTIONS_CONFLICT"))

    def test_invalid_private_policy_values_are_safe_errors(self):
        source = self.input()
        for index, value in enumerate(["", "guess", None, True, 0, [], {}]):
            with self.subTest(value=value):
                self.options.write_text(json.dumps({"blank_deducts": value}))
                boundary = Boundary.create(self.public, self.root / f"bad-policy-{index}")
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(SafeError, "INVALID_BLANK_DEDUCT_POLICY"):
                        prepare(boundary, source, self.options)

    def test_malformed_values_are_still_excluded_under_zero_policy(self):
        rows = self.records()
        rows["43M deduct"] = "not-a-number"
        source = self.input(rows)
        workspace = self.root / "malformed"
        self.assertEqual(self.run_cli(source, workspace, "--blank-deducts", "zero"),
                         (0, "PRIVATE_RUN_COMPLETE"))
        audit = json.loads((workspace / "results/deduct_audit.json").read_text())
        self.assertEqual(audit["invalid_deduct_pairs"], {"start": 16, "end": 16})
        self.assertTrue(all(target["valid_target_rows"] == 0 for target in audit["targets"]))

    def test_changing_policy_invalidates_resume(self):
        source = self.input()
        workspace = self.root / "resume"
        self.assertEqual(self.run_cli(source, workspace, "--blank-deducts", "zero"),
                         (0, "PRIVATE_RUN_COMPLETE"))
        boundary = Boundary.create(self.public, workspace)
        config_path = workspace / "run.json"
        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "1"}):
            self.assertEqual(run(boundary, config_path, resume=True), "VERIFIED_EXISTING_RUN")
            config = json.loads(config_path.read_text())
            config["blank_deducts"] = "missing"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SafeError, "RESUME_INPUT_OR_CODE_CHANGED"):
                run(boundary, config_path, resume=True)

    def test_blank_audit_counts_cannot_be_requested_as_predictors(self):
        source = self.input()
        for field in ("blank_deduct_cells_start", "zero_filled_deduct_cells_start"):
            with self.subTest(field=field):
                self.options.write_text(json.dumps({"models": False,
                    "features": ["PCI_start", field]}))
                self.assertEqual(self.run_cli(source, self.root / field, "--blank-deducts", "zero"),
                                 (2, "FORBIDDEN_FEATURE_REQUESTED"))


if __name__ == "__main__":
    unittest.main()
