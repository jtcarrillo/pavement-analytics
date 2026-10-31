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
        if "--quiet" in extra:
            self.assertNotIn(str(self.root), stream.getvalue())
        return stream.getvalue()

    def test_every_original_stage_then_modeling_report_and_verified_resume(self):
        legacy_checkout = self.root / "legacy_checkout"
        legacy_checkout.mkdir()
        (legacy_checkout / ".git").mkdir()
        inherited_cache = legacy_checkout / "weather_cache"
        self.settings["cache_dir"] = str(inherited_cache)
        self.profile.write_text(json.dumps(self.settings))
        mapping = json.loads(self.mapping.read_text())
        self.mapping.write_text(json.dumps({str(self.raw / source): str(self.locations / location)
                                            for source, location in mapping.items()}))
        output = self.run_cli()
        self.assertTrue(output.rstrip().endswith("PRIVATE_FULL_PIPELINE_COMPLETE"))
        self.assertIn(str(self.mapping), output)
        self.assertIn(str(self.profile), output)
        self.assertIn("export_00.csv", output)
        self.assertIn("location_00.csv", output)
        self.assertIn("Started", output)
        self.assertIn("Completed", output)
        self.assertIn("Skipped", output)
        self.assertLess(output.index("01_split"), output.index("02_pairs_PCC"))
        self.assertIn("fam_pcc_freeze_thaw deduct_end", output)
        self.assertEqual((self.work / "logs/progress.log").read_text(),
                         output.removesuffix("PRIVATE_FULL_PIPELINE_COMPLETE\n"))
        selected_cache = self.work.parent / "original-era5-cache"
        setup = json.loads((self.work / "full_setup_report.json").read_text())
        self.assertEqual(len(setup["mapping_entries"]), 10)
        self.assertTrue(all(item["status"] == "normalized" for item in setup["mapping_entries"]))
        self.assertIn("Normalized mapping entry", output)
        self.assertEqual(setup["climate_cache"], {
            "source": "profile", "requested": str(inherited_cache), "selected": str(selected_cache),
            "fallback_reason": "CLIMATE_CACHE_MUST_NOT_BE_VERSIONED"})
        config = json.loads((self.work / "full_config.json").read_text())
        self.assertEqual(config["cache_dir"], str(selected_cache))
        self.assertFalse(inherited_cache.exists())
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
        progress_state = json.loads((self.work / "progress.json").read_text())
        self.assertIn(progress_state["status"], {"completed", "completed_with_warnings"})
        self.assertEqual(progress_state["dashboard"], (pipeline / "10_report/dashboard.html").as_uri())
        self.assertNotIn('http-equiv="refresh"', (self.work / "progress.html").read_text())
        self.assertIn("retrospective", html.lower())
        self.assertNotIn("Historical window: 365", html)
        self.assertEqual(list(self.public.iterdir()), [])
        before = len(self.calls)
        resumed = self.run_cli(resume=True)
        self.assertIn("Reused", resumed)
        self.assertIn("09_local_models", resumed)
        self.assertEqual(len(self.calls), before)
        self.assertEqual(self.run_cli("--quiet", resume=True).strip(), "PRIVATE_FULL_PIPELINE_COMPLETE")
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

    def prepare_setup(self, **options):
        self.profile.write_text(json.dumps(self.settings))
        boundary = Boundary.create(self.public, self.work)
        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "1", "GITHUB_ACTIONS": "false"}):
            config = prepare_full(boundary, self.raw, **options)
        return config, json.loads((self.work / "full_setup_report.json").read_text())

    def test_local_model_options_resolve_and_invalidate_the_original_fingerprint(self):
        self.settings["local_model_options"] = {"cache_dir": "model-cache", "candidate_ids": ["enet_0.01"],
                                               "fit_timeout_seconds": 123.}
        cfg, _ = self.prepare_setup()
        self.assertEqual(cfg["local_model_options"]["cache_dir"], str(self.root/"model-cache"))
        self.assertTrue(cfg["local_model_options"]["comparisons"])
        from pavement_analytics.original.pipeline import discover
        first = discover(cfg)[3]
        changed = json.loads(json.dumps(cfg))
        changed["local_model_options"]["fit_timeout_seconds"] = 124.
        self.assertNotEqual(first, discover(changed)[3])

    def test_inherited_git_cache_uses_external_fallback_without_moving_old_files(self):
        for marker in ("directory", "worktree_file"):
            with self.subTest(marker=marker):
                repository = self.root / ("old_checkout_" + marker)
                repository.mkdir()
                if marker == "directory":
                    (repository / ".git").mkdir()
                else:
                    (repository / ".git").write_text("gitdir: ../metadata/worktrees/generated\n")
                old_cache = repository / "cache"
                old_cache.mkdir()
                cached = old_cache / "existing.cache"
                cached.write_bytes(b"generated existing weather cache")
                self.work = self.root / ("run_" + marker)
                self.settings["cache_dir"] = str(old_cache.relative_to(self.root))
                expected_profile = json.dumps(self.settings)
                config, setup = self.prepare_setup()
                fallback = self.work.parent / "original-era5-cache"
                self.assertEqual(Path(config["cache_dir"]), fallback)
                self.assertEqual(setup["climate_cache"], {
                    "source": "profile", "requested": str(old_cache), "selected": str(fallback),
                    "fallback_reason": "CLIMATE_CACHE_MUST_NOT_BE_VERSIONED"})
                self.assertEqual(self.profile.read_text(), expected_profile)
                self.assertEqual(cached.read_bytes(), b"generated existing weather cache")
                self.assertEqual(list(old_cache.iterdir()), [cached])
                self.assertFalse(fallback.exists())

    def test_default_cache_and_safe_profile_cache_preserve_expected_selection(self):
        config, setup = self.prepare_setup()
        fallback = self.work.parent / "original-era5-cache"
        self.assertEqual(Path(config["cache_dir"]), fallback)
        self.assertEqual(setup["climate_cache"], {
            "source": "default", "requested": None, "selected": str(fallback), "fallback_reason": None})
        self.work = self.root / "run_safe_cache"
        safe_cache = self.root / "existing_external_cache"
        safe_cache.mkdir()
        (safe_cache / "existing.cache").write_bytes(b"generated weather")
        self.settings["cache_dir"] = str(safe_cache.relative_to(self.root))
        config, setup = self.prepare_setup()
        self.assertEqual(Path(config["cache_dir"]), safe_cache)
        self.assertEqual(setup["climate_cache"], {
            "source": "profile", "requested": str(safe_cache), "selected": str(safe_cache),
            "fallback_reason": None})
        self.assertEqual((safe_cache / "existing.cache").read_bytes(), b"generated weather")

    def test_explicit_cache_override_wins_over_inherited_repository_cache(self):
        (self.public / ".git").mkdir()
        self.settings["cache_dir"] = str(self.public / "old_cache")
        explicit = self.root / "explicit_external_cache"
        config, setup = self.prepare_setup(cache_dir=explicit)
        self.assertEqual(Path(config["cache_dir"]), explicit)
        self.assertEqual(setup["climate_cache"]["source"], "command_line")
        self.assertEqual(setup["climate_cache"]["selected"], str(explicit))
        self.assertIsNone(setup["climate_cache"]["fallback_reason"])
        self.assertFalse((self.public / "old_cache").exists())

    def test_explicit_repository_cache_is_still_rejected(self):
        repository = self.root / "explicit_checkout"
        repository.mkdir()
        (repository / ".git").write_text("gitdir: ../metadata/worktrees/generated\n")
        with self.assertRaisesRegex(SafeError, "CLIMATE_CACHE_MUST_NOT_BE_VERSIONED"):
            self.prepare_setup(cache_dir=repository / "cache")
        self.assertFalse((repository / "cache").exists())
        self.assertFalse((self.work / "full_config.json").exists())

    def test_inherited_cache_cannot_overlap_inputs_run_or_public_source(self):
        for location in ("inputs", "run", "public"):
            with self.subTest(location=location):
                self.work = self.root / ("run_overlap_" + location)
                base = {"inputs": self.raw, "run": self.work, "public": self.public}[location]
                inherited = base / "cache"
                self.settings["cache_dir"] = str(inherited)
                config, setup = self.prepare_setup()
                self.assertEqual(Path(config["cache_dir"]), self.work.parent / "original-era5-cache")
                self.assertEqual(setup["climate_cache"]["requested"], str(inherited))
                expected = ("ANALYZE_SOURCE_MUST_BE_OUTSIDE_PUBLIC_TREE" if location == "public"
                            else "CLIMATE_CACHE_MUST_BE_SEPARATE_FROM_INPUTS_AND_RUN")
                self.assertEqual(setup["climate_cache"]["fallback_reason"], expected)
                self.assertFalse(inherited.exists())

    def test_inherited_unusable_cache_does_not_follow_links_or_overwrite_files(self):
        target = self.root / "existing_cache_target"
        target.mkdir()
        link = self.root / "old_cache_link"
        link.symlink_to(target, target_is_directory=True)
        existing_file = self.root / "old_cache_file"
        existing_file.write_bytes(b"generated existing file")
        for inherited, reason in ((link, "SYMLINK_PATH_FORBIDDEN"),
                                  (existing_file, "CLIMATE_CACHE_MUST_BE_DIRECTORY")):
            with self.subTest(reason=reason):
                self.work = self.root / ("run_" + reason.lower())
                self.settings["cache_dir"] = str(inherited)
                config, setup = self.prepare_setup()
                self.assertEqual(Path(config["cache_dir"]), self.work.parent / "original-era5-cache")
                self.assertEqual(setup["climate_cache"]["fallback_reason"], reason)
                self.assertTrue(link.is_symlink())
                self.assertEqual(list(target.iterdir()), [])
                self.assertEqual(existing_file.read_bytes(), b"generated existing file")

    def test_unsafe_default_cache_is_rejected_instead_of_used_as_fallback(self):
        fallback = self.work.parent / "original-era5-cache"
        fallback.mkdir()
        (fallback / ".git").mkdir()
        with self.assertRaisesRegex(SafeError, "CLIMATE_CACHE_MUST_NOT_BE_VERSIONED"):
            self.prepare_setup()
        self.assertFalse((self.work / "full_config.json").exists())
        self.assertEqual(list(fallback.iterdir()), [fallback / ".git"])

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
        output = self.run_cli(expected=2)
        self.assertIn("PRIVATE_CDV_CURVES", output)
        self.assertIn("Failed", output)
        self.assertIn("runtime.log", output)
        self.assertNotIn("Started \"01_split\"", output)
        self.assertEqual(json.loads((self.work / "full_setup_report.json").read_text())["missing_required"], ["cdv_curves"])

    def test_stage_failure_reports_phase_without_echoing_raw_exception(self):
        with patch("pavement_analytics.original.pipeline.Pipeline.climate",
                   side_effect=ValueError("generated-sensitive-error-detail")):
            output = self.run_cli(expected=2)
        self.assertIn('Failed "04_climate_PCC"', output)
        self.assertIn("04_climate_PCC.log", output)
        self.assertNotIn("generated-sensitive-error-detail", output)
        self.assertNotIn('Completed "04_climate_PCC"', output)
        self.assertNotIn('Started "09_local_models"', output)
        self.assertIn("generated-sensitive-error-detail", (self.work / "logs/runtime.log").read_text())

    def test_stale_setup_paths_are_reported_together_before_reading_inputs(self):
        missing_mapping = self.root / "old/mapping.json"
        missing_locations = self.root / "old/locations"
        self.settings.update(mapping=str(missing_mapping), csv_dir=str(missing_locations))
        self.profile.write_text(json.dumps(self.settings))
        output = self.run_cli(expected=2)
        self.assertIn("FULL_SETUP_INPUT_PATHS_INVALID", output)
        self.assertIn("Mapping JSON not found", output)
        self.assertIn("Location CSV directory not found", output)
        self.assertIn(str(missing_mapping), output)
        self.assertIn(str(missing_locations), output)
        self.assertNotIn("Reading input", output)
        setup = json.loads((self.work / "full_setup_report.json").read_text())
        self.assertEqual({item["role"]: item["status"] for item in setup["input_checks"]},
                         {"mapping": "missing", "csv_dir": "missing", "cdv_curves": "ready"})
        self.assertFalse(missing_mapping.exists())

    def test_invalid_mapping_entries_are_diagnosed_without_dumping_nested_values(self):
        self.mapping.write_text(json.dumps({"export_00.csv": {"data": "generated-sensitive-value"},
                                            "export_01.csv": "../outside.csv"}))
        output = self.run_cli(expected=2)
        self.assertEqual(output.count("Invalid mapping entry:"), 2)
        for expected in ("export_00.csv", "export_01.csv", "LOCATION_VALUE_MUST_BE_NONEMPTY_STRING",
                         "LOCATION_PARENT_COMPONENT_FORBIDDEN", "INVALID_LOCATION_MAPPING"):
            self.assertIn(expected, output)
        self.assertNotIn("generated-sensitive-value", output)
        self.assertNotIn("Reading input", output)
        setup = json.loads((self.work / "full_setup_report.json").read_text())
        self.assertEqual([row["status"] for row in setup["mapping_entries"]], ["invalid", "invalid"])
        self.assertIsNone(setup["mapping_entries"][0]["location_value"])
        self.assertNotIn("generated-sensitive-value", json.dumps(setup))
        self.assertFalse((self.work / "inputs").exists())

    def test_setup_read_failure_identifies_current_input_before_stopping(self):
        failed_input = self.raw / "export_00.csv"
        failed_input.write_text("bad_column\ngenerated-sensitive-cell\n")
        output = self.run_cli(expected=2)
        self.assertIn("Reading input (1/10)", output)
        self.assertIn(str(failed_input), output)
        self.assertIn("Failed setup", output)
        self.assertNotIn('Started "01_split"', output)
        self.assertNotIn("generated-sensitive-cell", output)
        progress = json.loads((self.work / "progress.json").read_text())
        self.assertEqual(progress["status"], "failed")
        self.assertNotIn("generated-sensitive-cell", (self.work / "progress.html").read_text())

    def test_setup_summary_identifies_unmapped_and_missing_location_files(self):
        mapping = json.loads(self.mapping.read_text())
        del mapping["export_00.csv"]
        mapping["export_01.csv"] = "missing_location.csv"
        self.mapping.write_text(json.dumps(mapping))
        events = []
        boundary = Boundary.create(self.public, self.work)
        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "1", "GITHUB_ACTIONS": "false"}):
            prepare_full(boundary, self.raw, progress=events.append)
        inputs = [event for event in events if event["event"] == "input"]
        self.assertEqual(len(inputs), 10)
        self.assertEqual(inputs[0]["location_status"], "unmapped")
        self.assertEqual(inputs[1]["location_status"], "missing_file")
        summary = events[-1]
        self.assertEqual(summary["event"], "setup_complete")
        self.assertEqual((summary["files"], summary["rows"], summary["mapped_files"],
                          summary["unmapped_files"], summary["missing_location_files"]), (10, 60, 8, 1, 1))

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
