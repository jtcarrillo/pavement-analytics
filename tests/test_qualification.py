"""Generated qualification evidence, identity guards and private boundaries."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from pavement_analytics.cli import main
from pavement_analytics.full_modeling import consolidate_full_models, run_full_models
from pavement_analytics.io import sha256, write_json
from pavement_analytics.model_registry import environment_versions
from pavement_analytics.model_tasks import verified_task
from pavement_analytics.original.common import Run
from pavement_analytics.pipeline import code_digest
from pavement_analytics.qualification_cli import qualification_command
from pavement_analytics.release_qualification import _usage, release_fixture
from pavement_analytics.run_qualification import _identity, audit_full_run, qualify_saved_run, verify_artifacts
from pavement_analytics.security import SafeError, source_root
from temp_support import temporary_directory
import test_full_modeling as modeling_fixtures


class QualificationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.source = pd.DataFrame({"partition_airfield": ["001", "002"], "delta_days": [120, 800],
                                    "observed": [3., 5.]}, index=[10, 11])
        self.plan = pd.DataFrame({"group_id": ["001", "002"], "partition": ["holdout", "development"], "fold": [-1, 0]})
        self.saved = pd.DataFrame({"row_id": [10], "group_id": ["001"], "actual": [3.], "weight": [120.], "prediction": [2.]})

    def test_valid_identity_retains_text_groups_and_original_weights(self):
        result = _identity(self.saved, self.source, self.plan, "observed", "holdout")
        self.assertEqual(result.index.tolist(), [10])
        self.assertEqual(result.partition_airfield.tolist(), ["001"])

    def test_rejects_row_group_target_weight_prediction_and_partition_changes(self):
        for column, value, code in (("row_id", 10.5, "ROW_ID"), ("group_id", "002", "GROUP_ID"),
                ("actual", 4., "TARGET_MISMATCH"), ("weight", 1., "WEIGHT_MISMATCH"), ("prediction", -1., "INVALID_PREDICTIONS")):
            with self.subTest(column=column):
                changed = self.saved.copy(); changed[column] = value
                with self.assertRaisesRegex(SafeError, code):
                    _identity(changed, self.source, self.plan, "observed", "holdout")
        with self.assertRaisesRegex(SafeError, "PARTITION_LEAKAGE"):
            _identity(self.saved, self.source, self.plan, "observed", "development")

    def test_missing_schema_and_wrong_oof_fold_are_rejected(self):
        with self.assertRaisesRegex(SafeError, "SCHEMA_INVALID"):
            _identity(self.saved.drop(columns="prediction"), self.source, self.plan, "observed", "holdout")
        saved = pd.DataFrame({"row_id": [11], "group_id": ["002"], "actual": [5.], "weight": [800.], "prediction": [4.], "fold": [1]})
        with self.assertRaisesRegex(SafeError, "FOLD_MISMATCH"):
            _identity(saved, self.source, self.plan, "observed", "development", expected_fold=True)


class FullRunQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = modeling_fixtures.FullModelTests()
        fixture.setUp()
        cls.addClassCleanup(fixture.doCleanups)
        cls.template = fixture.pipeline
        cls.cfg = {**fixture.cfg, "local_model_options": {"candidate_ids": ["enet_0.01"], "isolated": False}}
        cls.report = copy.deepcopy(fixture.report)
        run_full_models(cls.template, cls.report, cls.cfg)
        consolidate_full_models(cls.template/"09_local_models", qualification_link=True)
        names = ["01_split", "07_partitions", "08_model_backend", "09_local_models", "10_report"]
        names += [prefix+"_PCC" for prefix in ("02_pairs", "03_location_rate", "04_climate", "05_targets", "06_pci", "07_modeling")]
        for name in names:
            cls.report["stages"].setdefault(name, {"stats": {}}).update(status="completed", artifacts=[])
        cls.report.update(status="completed", fingerprint="generated-fingerprint", model_environment=environment_versions(),
            execution_environment={"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(), "code_sha256": code_digest()})

    def setUp(self):
        temporary = temporary_directory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pipeline = self.root/"pipeline"
        shutil.copytree(self.template, self.pipeline)
        self.report = copy.deepcopy(type(self).report)
        self.report["stages"]["09_local_models"]["artifacts"] = [
            {"path": str(p.relative_to(self.pipeline)), "sha256": sha256(p)}
            for p in sorted(self.pipeline.rglob("*")) if p.is_file()]
        self.input = self.root/"input.txt"; self.input.write_text("generated input evidence")
        self.setup = {"checksums": {str(self.input): sha256(self.input)}}

    def audit(self, **kwargs):
        with patch("pavement_analytics.original.pipeline.discover", return_value=([], [], [], "generated-fingerprint")):
            return audit_full_run(self.pipeline, self.report, self.cfg, self.setup, self.root/"qualification", **kwargs)

    def restamp(self, path):
        for item in self.report["stages"]["09_local_models"]["artifacts"]:
            if self.pipeline/item["path"] == path:
                item["sha256"] = sha256(path)
                return
        self.fail("Missing generated artifact")

    def test_all_saved_candidates_reload_without_mutating_original_artifacts(self):
        before = verify_artifacts(self.pipeline, self.report)
        result, files = self.audit(private_pilot=True)
        self.assertGreaterEqual(result["prediction_checks"], 8)
        self.assertGreater(result["oof_rows_checked"]["PCC"], 0)
        self.assertEqual(verify_artifacts(self.pipeline, self.report), before)
        self.assertEqual(result["observed_successful_downloads"], 0)
        self.assertIn("observed_live_era5_access", result["remaining_release_gates"])
        self.assertIn("private_pilot_review", result["remaining_release_gates"])
        self.assertNotIn("private_data_declaration", result["remaining_release_gates"])
        self.assertEqual(result["release_gate"], "pending_private_review")
        self.assertTrue(all(p.is_file() for p in files))

    def test_hash_verification_precedes_model_deserialization(self):
        path = next(self.pipeline.rglob("candidate_models/*.joblib"))
        path.write_bytes(b"generated invalid artifact")
        with patch("pavement_analytics.run_qualification.joblib.load", side_effect=AssertionError("must not deserialize")):
            with self.assertRaisesRegex(SafeError, "ARTIFACT_CHANGED"):
                self.audit()

    def test_missing_stage_inputs_environment_and_code_fail_explicitly(self):
        variants = [("stage", "PIPELINE_INCOMPLETE"), ("input", "INPUT_CHANGED"),
                    ("environment", "ENVIRONMENT_CHANGED"), ("code", "CODE_EVIDENCE")]
        for case, code in variants:
            with self.subTest(case=case):
                previous, setup = copy.deepcopy(self.report), copy.deepcopy(self.setup)
                if case == "stage": self.report["stages"]["01_split"]["status"] = "failed"
                if case == "input": self.setup["checksums"][str(self.input)] = "changed"
                if case == "environment": self.report["model_environment"] = {}
                if case == "code": self.report["execution_environment"]["code_sha256"] = "changed"
                with self.assertRaisesRegex(SafeError, code): self.audit()
                self.report, self.setup = previous, setup
                shutil.rmtree(self.root/"qualification")

    def test_changed_fingerprint_is_not_accepted(self):
        self.report["fingerprint"] = "changed"
        with self.assertRaisesRegex(SafeError, "FINGERPRINT_CHANGED"): self.audit()

    def test_changed_saved_prediction_is_detected_beyond_file_hashes(self):
        path = self.pipeline/"09_local_models/materials/PCC/level_thermal/candidate_predictions.csv"
        frame = pd.read_csv(path, dtype={"group_id": str}); frame.loc[0, "prediction"] += 1
        frame.to_csv(path, index=False); self.restamp(path)
        with self.assertRaisesRegex(SafeError, "SAVED_PREDICTION_MISMATCH"): self.audit()

    def test_changed_weight_or_group_is_detected_beyond_file_hashes(self):
        path = self.pipeline/"09_local_models/materials/PCC/level_thermal/candidate_predictions.csv"
        frame = pd.read_csv(path, dtype={"group_id": str}); frame.loc[0, "weight"] += 1
        frame.to_csv(path, index=False); self.restamp(path)
        with self.assertRaisesRegex(SafeError, "WEIGHT_MISMATCH"): self.audit()

    def test_changed_feature_or_fit_partition_metadata_is_rejected(self):
        path = next((self.pipeline/"09_local_models/materials/PCC/level_thermal/candidate_models").glob("*.joblib"))
        artifact = joblib.load(path); artifact["fit_partition"] = "holdout"
        joblib.dump(artifact, path); self.restamp(path)
        with self.assertRaisesRegex(SafeError, "PARTITION_OR_TARGET_MISMATCH"): self.audit()

    def test_live_weather_claim_requires_checkpointed_completed_worker_receipt(self):
        self.report["stages"]["04_climate_PCC"]["stats"]["successful_downloads"] = 1
        with self.assertRaisesRegex(SafeError, "CLIMATE_EVIDENCE_MISMATCH"): self.audit()
        shutil.rmtree(self.root/"qualification")
        receipt = self.pipeline/"04_climate/PCC_acquisition.json"
        write_json(receipt, {"successful_downloads": 1, "workers": [
            {"complete": True, "returncode": 0, "successful_downloads": 1},
            {"complete": False, "returncode": 3, "successful_downloads": 5},
            {"complete": True, "returncode": 0, "successful_downloads": 0, "cache_hits": 4}]})
        self.report["stages"]["04_climate_PCC"]["artifacts"].append(
            {"path": str(receipt.relative_to(self.pipeline)), "sha256": sha256(receipt)})
        result, _ = self.audit()
        self.assertEqual(result["observed_successful_downloads"], 1)
        self.assertNotIn("observed_live_era5_access", result["remaining_release_gates"])
        self.assertIn("private_pilot_review", result["remaining_release_gates"])

    def test_read_only_completed_run_entry_point_retains_input_artifacts(self):
        write_json(self.root/"full_result.json", {"status": "completed"})
        write_json(self.root/"full_config.json", {**self.cfg, "output_dir": "pipeline"})
        write_json(self.root/"full_setup_report.json", self.setup)
        write_json(self.pipeline/"run_report.json", self.report)
        before = {p: sha256(p) for p in self.root.rglob("*") if p.is_file()}
        output = temporary_directory(); self.addCleanup(output.cleanup)
        with patch("pavement_analytics.original.pipeline.discover", return_value=([], [], [], "generated-fingerprint")):
            result, _ = qualify_saved_run(self.root, Path(output.name)/"audit", private_pilot=True)
        self.assertEqual(result["evidence_kind"], "declared_private_pilot")
        self.assertTrue(all(sha256(path) == digest for path, digest in before.items()))

    def test_saved_run_audit_is_separate_and_completed_runs_only(self):
        with self.assertRaisesRegex(SafeError, "OUTPUT_MUST_BE_SEPARATE"):
            qualify_saved_run(self.root, self.root/"child")
        write_json(self.root/"full_result.json", {"status": "failed"})
        write_json(self.root/"full_config.json", {"output_dir": "pipeline"})
        write_json(self.pipeline/"run_report.json", self.report)
        with self.assertRaisesRegex(SafeError, "PIPELINE_INCOMPLETE"):
            qualify_saved_run(self.root, self.root.parent/(self.root.name+"-unused"))


class ReleaseQualificationTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_complete_fixture_matrix_and_artifact_parity_with_selected_candidate(self):
        events = []
        result = release_fixture(self.root/"release", candidate_ids=["enet_0.01"], progress=events.append)
        self.assertEqual(result["status"], "software_checks_passed")
        self.assertEqual(len(result["cases"]), 45)
        self.assertEqual(len(result["guards"]), 4)
        self.assertGreater(result["candidate_models_checked"], 100)
        self.assertEqual(len([e for e in events if e.get("event") == "target" and e["status"] == "completed"]), 45)
        self.assertEqual({c["material"] for c in result["cases"]}, {"PCC", "APC", "AC_AAC_ACC"})
        manifest = json.loads((self.root/"release/qualification_manifest.json").read_text())
        self.assertTrue(all(sha256(self.root/"release"/p) == digest for p, digest in manifest["outputs"].items()))
        self.assertIn("private_pilot_review", result["remaining_release_gates"])

    def test_existing_versioned_and_linked_outputs_are_rejected(self):
        for path in (self.root, source_root()/"forbidden-release-output"):
            with self.assertRaises(SafeError): release_fixture(path)
        link = self.root/"link"; link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(SafeError, "SYMLINK"): release_fixture(link/"release")

    def test_rss_units_are_normalized_for_macos_and_linux(self):
        import resource
        usage = SimpleNamespace(ru_maxrss=1024**2, ru_utime=2., ru_stime=3.)
        with patch.object(resource, "getrusage", return_value=usage):
            with patch("pavement_analytics.release_qualification.platform.system", return_value="Darwin"):
                self.assertEqual(_usage()["peak_sum_mib"], 2.)
            with patch("pavement_analytics.release_qualification.platform.system", return_value="Linux"):
                self.assertEqual(_usage()["peak_sum_mib"], 2048.)

    def test_code_change_during_task_build_never_publishes_ready_checkpoint(self):
        def build(destination):
            destination.mkdir(); write_json(destination/"summary.json", {"status": "generated"})
            return {"status": "generated"}
        with patch("pavement_analytics.pipeline.code_digest", side_effect=["before", "after"]):
            with self.assertRaisesRegex(SafeError, "CODE_CHANGED_DURING_BUILD"):
                verified_task(self.root/"cache", "a"*64, build, self.root/"result")
        self.assertFalse((self.root/"cache"/("a"*64)).exists())
        self.assertFalse((self.root/"result").exists())
        self.assertEqual(len(list((self.root/"cache").glob(".pending-model-*"))), 1)

    def test_cli_blocks_hosted_execution_before_creating_outputs(self):
        stream = io.StringIO()
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}), contextlib.redirect_stderr(stream):
            self.assertEqual(main(["qualify", "--output", str(self.root/"blocked")]), 2)
        self.assertIn("GITHUB_ACTIONS_DATA_EXECUTION_FORBIDDEN", stream.getvalue())
        self.assertFalse((self.root/"blocked").exists())

    def test_cli_failure_details_stay_in_private_log(self):
        args = SimpleNamespace(output=self.root/"failed", run_workspace=None, profile="smoke", candidate_ids=None, private_pilot=False)
        with patch("pavement_analytics.release_qualification.release_fixture", side_effect=RuntimeError("generated-private-diagnostic")):
            with self.assertRaises(RuntimeError): qualification_command(args)
        self.assertIn("generated-private-diagnostic", (self.root/"failed/runtime.log").read_text())
        self.assertFalse((self.root/"failed/result.json").exists())

    def test_cli_success_restores_environment_and_opens_requested_report(self):
        stream = io.StringIO()
        report_path = self.root/"result/qualification.html"
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false", "PAVEMENT_PRIVATE_EXECUTION": "0"}), \
                patch("pavement_analytics.qualification_cli.qualification_command", return_value=report_path) as command, \
                patch("pavement_analytics.monitor.open_local_window", return_value=True) as window, \
                contextlib.redirect_stdout(stream):
            self.assertEqual(main(["qualify", "--output", str(self.root/"result"), "--open"]), 0)
            self.assertEqual(os.environ["PAVEMENT_PRIVATE_EXECUTION"], "0")
        command.assert_called_once(); window.assert_called_once_with(report_path)
        self.assertIn("PRIVATE_QUALIFICATION_READY", stream.getvalue())

    def test_incompatible_cli_options_are_rejected(self):
        for arguments in (["--private-pilot"], [str(self.root), "--candidate", "enet_0.01"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["qualify", *arguments, "--output", str(self.root/"invalid")]), 2)
        self.assertFalse((self.root/"invalid").exists())


class AcquisitionReceiptTests(unittest.TestCase):
    def test_download_cache_and_failed_worker_evidence_remain_distinct(self):
        temporary = temporary_directory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name, phases, code in (("cache", ["cache_hit", "complete"], 0),
                ("download", ["requesting", "downloaded", "complete"], 0),
                ("failed", ["downloaded"], 3)):
            with self.subTest(case=name):
                run = Run(root/name, "generated-fingerprint", {"surfaces": ["PCC"]}, [], progress=lambda event: None)
                lines = ["PAVEMENT_PROGRESS "+json.dumps({"event": "climate", "phase": phase, "cache_hits": int(name == "cache"), "requests": int(name != "cache")}) for phase in phases]
                script = "\n".join("print("+repr(line)+", flush=True)" for line in lines)+"\nraise SystemExit("+str(code)+")"
                def execute():
                    run.process([sys.executable, "-u", "-c", script]); return {}, []
                with contextlib.redirect_stdout(io.StringIO()):
                    if code:
                        with self.assertRaises(RuntimeError): run.stage("04_climate_PCC", execute)
                    else: run.stage("04_climate_PCC", execute)
                receipt = run.climate_evidence["04_climate_PCC"][0]
                self.assertEqual(receipt["successful_downloads"], int(name != "cache"))
                self.assertEqual(receipt["complete"], name != "failed")
                self.assertEqual(receipt["returncode"], code)


if __name__ == "__main__":
    unittest.main()
