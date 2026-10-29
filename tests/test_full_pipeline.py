"""Run all restored stages; substitute only the external weather acquisition."""
import contextlib
import io
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pavement_analytics.cli import main
from pavement_analytics.full import prepare_full
from pavement_analytics.original import era5
from pavement_analytics.original.common import Run
from pavement_analytics.security import Boundary, SafeError
from temp_support import temporary_directory
from test_original_pci import toy_cdv_profile


class FullPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.locations = self.root / "locations"
        self.locations.mkdir()
        self.public = self.root / "source"
        self.public.mkdir()
        self.work = self.root / "run"
        mapping = {}
        for group in range(10):
            name = f"export_{group:02d}.csv"
            frame = pd.DataFrame([
                {"SectionUniqueID": f"{section:03d}", "Date": f"{year}-01-01", "PCI": 95 - 3 * offset,
                 "Inspected Surface Type": "PCC", "IsConstruction": "no",
                 "Original Construction Year": 2017,
                 "64L deduct": 2 + group + section + offset,
                 "64L (Slabs)": 2 + group + section + offset,
                 "70L deduct": 0, "70L (Slabs)": 0, "62L deduct": 0,
                 "InspectedSlabs": 100, "Years since Major MR": year - 2017}
                for section in range(2) for offset, year in enumerate((2019, 2020, 2021))])
            frame.to_csv(self.raw / name, index=False)
            locname = f"location_{group:02d}.csv"
            pd.DataFrame({"LOCATION(DEGREES)": [f"0, {group}"]}).to_csv(self.locations / locname, index=False)
            mapping[name] = locname
        self.mapping = self.root / "data_file_mapping.json"
        self.mapping.write_text(json.dumps(mapping))
        self.curves = self.root / "cdv_curves.json"
        self.curves.write_text(json.dumps(toy_cdv_profile()))
        self.profile = self.root / "pipeline.example.json"
        self.settings = {"input_dir": "raw", "output_dir": "previous-output",
                         "mapping": "data_file_mapping.json", "csv_dir": "locations",
                         "surfaces": ["PCC"], "climate_start_year": 2018, "climate_end_year": 2021,
                         "dr_cv_reps": 3, "min_groups": 2, "datarobot": "dry-run",
                         "targets": {"PCC": ["DetRatePerYear"]}}
        self.profile.write_text(json.dumps(self.settings))
        self.calls = []

    def weather(self, *args, **kwargs):
        # Original hourly derivations run on generated weather for each location.
        longitude = kwargs["lon"]
        time = pd.date_range("2018-01-01", "2021-01-02", freq="h", tz="UTC")
        phase = np.arange(len(time)) * 2 * np.pi / 24
        raw = pd.DataFrame({"time": time, "t2m": 273.15 + longitude + 10 * np.sin(phase),
                            "skt": 273.15 + longitude + 15 * np.sin(phase)})
        return era5.prepare_hourly_df(raw, longitude), {"data_source": "era5-land", "snap_distance_cells": 0,
                                                       "snap_distance_km": 0.}

    def run_cli(self, *extra, resume=False, expected=0):
        original_process = Run.process
        original_find_spec = importlib.util.find_spec
        def find_spec(name, *args, **kwargs):
            if name == "datarobot":
                raise AssertionError("Retired backend must not be probed or imported")
            return object() if name == "cdsapi" else original_find_spec(name, *args, **kwargs)
        def process(run, command):
            text = [str(v) for v in command]
            if "pavement_analytics.original.era5" in text:
                self.calls.append("climate")
                tail = text[text.index("pavement_analytics.original.era5") + 1:]
                with patch.object(era5, "_create_cds_client", return_value=object()), \
                        patch.object(era5, "get_hourly_for_location", side_effect=self.weather):
                    self.assertEqual(era5.main(tail), 0)
            else:
                self.calls.append(text[3] if "-m" in text else "other")
                original_process(run, command)
        command = ["full", "--workspace", str(self.work)]
        command += ["--resume"] if resume else [str(self.raw)]
        command += list(extra)
        stream = io.StringIO()
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}), \
                patch("pavement_analytics.cli.source_root", return_value=self.public), \
                patch.object(Run, "process", process), \
                patch("importlib.util.find_spec", side_effect=find_spec), \
                contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = main(command)
        if code != expected:
            log = self.work / "logs/runtime.log"
            self.fail(stream.getvalue() + (log.read_text() if log.exists() else ""))
        self.assertNotIn(str(self.root), stream.getvalue())
        return stream.getvalue()

    def test_every_original_stage_then_modeling_report_and_verified_resume(self):
        self.assertEqual(self.run_cli().strip(), "PRIVATE_FULL_PIPELINE_COMPLETE")
        pipeline = self.work / "pipeline"
        report = json.loads((pipeline / "run_report.json").read_text())
        for stage in ("01_split", "02_pairs_PCC", "03_location_rate_PCC", "04_climate_PCC", "05_targets_PCC",
                      "06_pci_PCC", "07_modeling_PCC", "07_partitions", "08_model_backend",
                      "09_local_models", "10_report"):
            self.assertEqual(report["stages"][stage]["status"], "completed", stage)
        self.assertIn("climate", self.calls)
        self.assertIn("pavement_analytics.original.pci", self.calls)
        self.assertFalse(any("datarobot" in call for call in self.calls))
        curated = pd.read_csv(pipeline / "06_curated/PCC.csv")
        self.assertEqual(len(curated), 40)
        self.assertIn("fam_pcc_freeze_thaw deduct_end", curated)
        self.assertIn("loss_shap_pcc_freeze_thaw_end", curated)
        self.assertTrue(any(c.startswith("ThermalAgingPhi") for c in curated))
        self.assertTrue(any(c.startswith("PGLowNormalF") for c in curated))
        self.assertTrue(any(c.startswith("PavPGGrade") for c in curated))
        for end in ("start", "end"):
            np.testing.assert_allclose(curated.filter(regex=f"^loss_shap_.*_{end}$").sum(axis=1),
                                       100 - curated[f"pci_calc_{end}"])
        summary = pd.read_csv(pipeline / "10_report/target_summary.csv")
        self.assertEqual(len(summary), 11)
        self.assertTrue(summary.original_target.str.endswith(" deduct_end").all())
        self.assertFalse(summary.original_target.eq("DetRatePerYear").any())
        html = (pipeline / "10_report/dashboard.html").read_text()
        self.assertIn("retrospective", html.lower())
        self.assertNotIn("Historical window: 365", html)
        self.assertEqual(list(self.public.iterdir()), [])
        before = len(self.calls)
        self.run_cli(resume=True)
        self.assertEqual(len(self.calls), before)
        (self.raw / "export_00.csv").write_text((self.raw / "export_00.csv").read_text() + "\n")
        self.assertIn("INPUT_OR_SNAPSHOT_CHANGED", self.run_cli(resume=True, expected=2))

    def test_full_overrides_old_skip_flags_and_snapshots_private_curve_profile(self):
        self.settings.update(skip_climate=True, skip_distress=True, skip_pci=True, no_shapley=True)
        self.profile.write_text(json.dumps(self.settings))
        boundary = Boundary.create(self.public, self.work)
        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "1", "GITHUB_ACTIONS": "false"}):
            cfg = prepare_full(boundary, self.raw)
        self.assertTrue(all(cfg[k] is False for k in ("skip_climate", "skip_distress", "skip_pci", "no_shapley")))
        self.assertEqual(Path(cfg["cdv_curves"]).read_bytes(), self.curves.read_bytes())
        self.assertNotEqual(Path(cfg["cdv_curves"]), self.curves)
        self.assertEqual(cfg["target_mode"], "deduct_families")
        self.assertEqual(cfg["datarobot"], "off")

    def test_optional_curve_fit_runs_without_using_fitted_outputs_as_predictors(self):
        self.settings.update(fit_deduct_curves=True, datarobot="train", keep_distress=True)
        self.profile.write_text(json.dumps(self.settings))
        self.run_cli()
        pipeline = self.work / "pipeline"
        report = json.loads((pipeline / "run_report.json").read_text())
        self.assertGreater(report["stages"]["05_fit_curves"]["stats"]["fitted_curves"], 0)
        self.assertEqual(report["stages"]["08_model_backend"]["stats"]["backend"], "local")
        self.assertFalse(report["stages"]["08_model_backend"]["stats"]["remote_training"])
        audit = pd.read_csv(pipeline / "07_modeling/PCC_feature_audit.csv")
        calibrated = audit.column.str.startswith(("pci_calc_", "pci_only_", "loss_shap_", "loss_loo_", "pci_diff_"))
        self.assertFalse(audit.loc[calibrated, "disposition"].eq("KEEP").any())
        self.assertTrue(audit.loc[audit.column.eq("PCI_start"), "disposition"].eq("KEEP").all())
        self.assertFalse(any("datarobot" in call for call in self.calls))

    def test_missing_curve_profile_is_an_explicit_setup_failure(self):
        self.curves.unlink()
        self.assertIn("PRIVATE_CDV_CURVES", self.run_cli(expected=2))
        self.assertEqual(json.loads((self.work / "full_setup_report.json").read_text())["missing_required"], ["cdv_curves"])

    def test_no_data_or_resume_without_workspace_has_clear_errors(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            result = main(["full", "--resume"])
        self.assertEqual(result, 2)
        self.assertIn("FULL_RESUME_REQUIRES_WORKSPACE", stream.getvalue())

    def test_full_forbidden_on_actions(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["full", str(self.raw), "--workspace", str(self.work)]), 2)
        self.assertFalse(self.work.exists())


if __name__ == "__main__":
    unittest.main()
