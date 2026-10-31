"""Generated original-stage outputs verify the final target-sweep handoff."""
import json
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from pavement_analytics.full_modeling import consolidate_full_models, run_full_models
from pavement_analytics.original.common import Run
from pavement_analytics.security import SafeError
from pavement_analytics.study_modeling import _estimator, make_partitions, search_levels
from temp_support import temporary_directory


def linear_candidates(seed):
    return {"linear": _estimator(ElasticNet(alpha=.01, l1_ratio=.5, max_iter=10000,
                                           random_state=seed))}


class FullModelTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pipeline = self.root / "pipeline"
        self.modeldir = self.pipeline / "07_modeling" / "partitioned"
        self.modeldir.mkdir(parents=True)
        rng = np.random.default_rng(11)
        self.frame = pd.DataFrame({
            "partition_airfield": np.repeat([f"{i:03d}" for i in range(12)], 6),
            "partition_section": [f"section-{i}" for i in range(72)],
            "delta_days": np.tile([100, 200, 400, 800, 1600, 3200], 12),
            "PCI_start": rng.uniform(70, 100, 72),
            "MeanTempC_1980_present": rng.uniform(0, 20, 72),
        })
        self.frame["fam_thermal deduct_end"] = 2 + 3 * self.frame.MeanTempC_1980_present
        self.frame["fam_other deduct_end"] = rng.uniform(0, 20, 72)
        self.frame.loc[self.frame.index[::17], "fam_other deduct_end"] = np.nan
        self.plan = make_partitions(self.frame.rename(columns={"partition_airfield": "group_id"}), folds=3)
        self.frame["partition_fold"] = self.frame.partition_airfield.map(self.plan.set_index("group_id").fold)
        self.frame.to_csv(self.modeldir / "PCC.csv", index=False)
        self.plan.to_csv(self.pipeline / "07_modeling/partitions.csv", index=False)
        roles = {"partition_airfield": "PARTITION", "partition_section": "PARTITION",
                 "partition_fold": "PARTITION", "delta_days": "WEIGHT", "PCI_start": "KEEP",
                 "MeanTempC_1980_present": "KEEP", "fam_thermal deduct_end": "TARGET",
                 "fam_other deduct_end": "TARGET"}
        self.feature_audit = pd.DataFrame([{"column": name, "disposition": role, "reason": "generated test"}
                                           for name, role in roles.items()])
        self.feature_audit.to_csv(self.modeldir / "PCC_feature_audit.csv", index=False)
        for stage in ("03_location_rate", "04_climate"):
            (self.pipeline / stage).mkdir()
        self.frame.drop(columns="MeanTempC_1980_present").to_csv(self.pipeline / "03_location_rate/PCC.csv", index=False)
        self.frame.to_csv(self.pipeline / "04_climate/PCC.csv", index=False)
        self.report = {"stages": {"04_climate_PCC": {"status": "complete", "stats": {"rows": 72, "rows_with_climate": 72}}}}
        self.cfg = {"surfaces": ["PCC"], "dr_cv_reps": 3,
                    "climate_start_year": 1980, "climate_end_year": 2020}

    def test_original_outputs_sweep_preserves_targets_plan_weights_and_climate(self):
        with patch("pavement_analytics.study_modeling.candidates", linear_candidates):
            summaries, audit, artifacts = run_full_models(self.pipeline, self.report, self.cfg)
        out = self.pipeline / "09_local_models"
        thermal = next(s for s in summaries if s["target"] == "level_thermal")
        other = next(s for s in summaries if s["target"] == "level_other")
        self.assertGreater(thermal["mae_skill"], .95)
        self.assertEqual(other["excluded_missing_target"], 5)
        self.assertEqual(thermal["weight_column"], "delta_days")
        self.assertEqual(thermal["original_target"], "fam_thermal deduct_end")
        self.assertEqual(audit["metric_weighting"], "interval_days")
        features = json.loads((out / "materials/PCC/features.json").read_text())
        self.assertEqual(features["climate_features"], ["MeanTempC_1980_present"])
        self.assertIn("PCI_start", features["features"])
        self.assertFalse(any("deduct" in f for f in features["features"]))
        predicted = pd.read_csv(out / "materials/PCC/level_thermal/holdout_predictions.csv", dtype={"group_id": str})
        expected = self.frame.iloc[predicted.row_id]
        np.testing.assert_array_equal(predicted.weight, expected.delta_days)
        self.assertEqual(predicted.group_id.tolist(), expected.partition_airfield.tolist())
        for target in ("level_thermal", "level_other"):
            pd.testing.assert_frame_equal(self.plan, pd.read_csv(out / f"materials/PCC/{target}/folds.csv", dtype={"group_id": str}))
        self.assertTrue(all(p.is_file() for p in artifacts))
        report_artifacts = consolidate_full_models(out)
        self.assertTrue(all(p.is_file() for p in report_artifacts))
        report_out = self.pipeline / "10_report"
        html = (report_out / "dashboard.html").read_text()
        self.assertIn("retrospective explanatory modeling", html)
        self.assertIn("weighted by its inspection interval in days", html)
        self.assertIn("fam_thermal deduct_end", html)
        self.assertNotIn("climate window ends before the inspection", html)
        self.assertNotIn("each validation group receives equal total weight", html.lower())
        consolidated = pd.read_csv(report_out / "target_summary.csv")
        self.assertEqual(set(consolidated.original_target), {"fam_thermal deduct_end", "fam_other deduct_end"})
        bundle = joblib.load(out / "materials/PCC/level_thermal/model.joblib")
        self.assertEqual(bundle["weighting"], "interval_days")

    def test_progress_covers_missing_and_constant_targets_without_changing_results(self):
        extra_targets = ["fam_missing deduct_end", "fam_constant deduct_end"]
        self.frame[extra_targets[0]] = np.nan
        self.frame[extra_targets[1]] = 0.
        self.frame.to_csv(self.modeldir / "PCC.csv", index=False)
        pd.concat([self.feature_audit, pd.DataFrame([
            {"column": name, "disposition": "TARGET", "reason": "generated test"}
            for name in extra_targets])]).to_csv(self.modeldir / "PCC_feature_audit.csv", index=False)
        events = []
        original_search = search_levels
        def fitting(table, features, target, task_dir, **kwargs):
            # A started event must be visible before fitting begins.
            self.assertEqual(events[-1]["status"], "started")
            self.assertEqual(events[-1]["target"], "fam_" + target.removeprefix("level_") + " deduct_end")
            return original_search(table, features, target, task_dir, **kwargs)
        with patch("pavement_analytics.study_modeling.candidates", linear_candidates):
            plain, plain_audit, _ = run_full_models(self.pipeline, self.report, self.cfg)
            with patch("pavement_analytics.full_modeling.search_levels", side_effect=fitting):
                observed, observed_audit, _ = run_full_models(
                    self.pipeline, self.report, self.cfg, self.pipeline / "progress_models",
                    progress=events.append)
        self.assertEqual(observed, plain)
        self.assertEqual(observed_audit, plain_audit)
        expected = []
        for summary in observed:
            event = {"event": "target", "material": "PCC", "target": summary["original_target"]}
            expected.extend([{**event, "status": "started"},
                             {**event, "status": "completed", "result_status": summary["status"]}])
        self.assertEqual([e for e in events if e["event"] == "target"], expected)
        self.assertTrue(any(e["event"] == "model" and e["status"] == "task_reused" for e in events))
        statuses = {summary["original_target"]: summary["status"] for summary in observed}
        self.assertEqual(statuses[extra_targets[0]], "no_valid_target")
        self.assertEqual(statuses[extra_targets[1]], "constant_target")
        for target in ("level_thermal", "level_other"):
            filename = f"materials/PCC/{target}/holdout_predictions.csv"
            self.assertEqual((self.pipeline / "09_local_models" / filename).read_bytes(),
                             (self.pipeline / "progress_models" / filename).read_bytes())

    def test_progress_completes_every_slot_without_features(self):
        self.feature_audit.loc[self.feature_audit.disposition.eq("KEEP"), "disposition"] = "DROP"
        self.feature_audit.to_csv(self.modeldir / "PCC_feature_audit.csv", index=False)
        events = []
        with patch("pavement_analytics.full_modeling.search_levels") as fitting:
            summaries, _, _ = run_full_models(self.pipeline, self.report, self.cfg, progress=events.append)
        fitting.assert_not_called()
        self.assertEqual(len(events), 2 * len(summaries))
        self.assertEqual([event["status"] for event in events], ["started", "completed"] * len(summaries))
        self.assertTrue(all(event["result_status"] == "no_features" for event in events[1::2]))
        self.assertEqual([event["target"] for event in events[1::2]],
                         [summary["original_target"] for summary in summaries])

    def test_outcome_feature_cannot_bypass_original_audit(self):
        self.frame["PCI_end"] = 80
        self.frame.to_csv(self.modeldir / "PCC.csv", index=False)
        pd.concat([self.feature_audit, pd.DataFrame([{"column": "PCI_end", "disposition": "KEEP"}])]).to_csv(
            self.modeldir / "PCC_feature_audit.csv", index=False)
        with self.assertRaisesRegex(SafeError, "UNSAFE_ORIGINAL_MODEL_FEATURE"):
            run_full_models(self.pipeline, self.report, self.cfg)

    def test_shared_fold_mismatch_is_not_silently_redrawn(self):
        self.frame.loc[0, "partition_fold"] = 99
        self.frame.to_csv(self.modeldir / "PCC.csv", index=False)
        with self.assertRaisesRegex(SafeError, "PARTITION_ASSIGNMENT_MISMATCH"):
            run_full_models(self.pipeline, self.report, self.cfg)

    def test_invalid_interval_fails_before_any_models(self):
        self.frame.loc[0, "delta_days"] = 0
        self.frame.to_csv(self.modeldir / "PCC.csv", index=False)
        with self.assertRaisesRegex(SafeError, "INTERVAL_WEIGHTS_INVALID"):
            run_full_models(self.pipeline, self.report, self.cfg)
        self.assertFalse((self.pipeline / "09_local_models").exists())

    def test_partial_failure_keeps_audit_and_allows_stage_retry(self):
        events = []
        with patch("pavement_analytics.full_modeling.search_levels", side_effect=SafeError("TEST_MODEL_FAILURE")):
            with self.assertRaisesRegex(SafeError, "TEST_MODEL_FAILURE"):
                run_full_models(self.pipeline, self.report, self.cfg, progress=events.append)
        self.assertEqual(events, [{"event": "target", "material": "PCC",
                                   "target": "fam_thermal deduct_end", "status": "started"}])
        self.assertFalse((self.pipeline / "09_local_models").exists())
        self.assertEqual(len(list(self.pipeline.glob("09_local_models_failed_*"))), 1)
        with patch("pavement_analytics.study_modeling.candidates", linear_candidates):
            summaries, _, _ = run_full_models(self.pipeline, self.report, self.cfg)
        self.assertEqual(len(summaries), 2)

    def test_report_is_separate_and_preserves_all_model_artifacts(self):
        with patch("pavement_analytics.study_modeling.candidates", linear_candidates):
            _, _, model_artifacts = run_full_models(self.pipeline, self.report, self.cfg)
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in model_artifacts}
        model_out = self.pipeline / "09_local_models"
        report_out = self.pipeline / "10_report"
        consolidate_full_models(model_out, report_out=report_out, pipeline_report=self.report)
        html = (report_out / "dashboard.html").read_text()
        self.assertNotIn("DataRobot", html)
        self.assertFalse((model_out / "dashboard.html").exists())
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in model_artifacts})

    def test_component_report_exports_are_verified_on_full_run_resume(self):
        with patch("pavement_analytics.study_modeling.candidates", linear_candidates):
            run_full_models(self.pipeline, self.report, self.cfg)
        checkpoint = self.root/"report-checkpoint"
        run = Run(checkpoint, "generated-fingerprint", self.cfg, [])
        report_out = checkpoint/"10_report"

        def report_stage():
            artifacts = consolidate_full_models(self.pipeline/"09_local_models", report_out=report_out)
            return {}, artifacts

        run.stage("10_report", report_stage)
        recorded = {Path(item["path"]).name for item in run.data["stages"]["10_report"]["artifacts"]}
        for name in ("two_stage_predictions.csv", "two_stage_metrics.csv", "candidate_comparisons.csv",
                     "family_summary.csv", "family_importance.csv", "candidate_predictions.csv", "model_runtime.csv"):
            self.assertIn(name, recorded)
            path = report_out/name
            original = path.read_bytes()
            path.write_bytes(original+b"\n")
            with self.assertRaisesRegex(ValueError, "Checkpoint artifact changed or missing"):
                Run(checkpoint, "generated-fingerprint", self.cfg, [], resume=True)
            path.write_bytes(original)
        Run(checkpoint, "generated-fingerprint", self.cfg, [], resume=True)


    def test_interrupted_sweep_reuses_only_verified_completed_targets(self):
        self.cfg["local_model_options"] = {"isolated": False, "candidate_ids": ["enet_0.01"]}
        def interrupt(event):
            if event["event"] == "target" and event["status"] == "completed":
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            run_full_models(self.pipeline, self.report, self.cfg, progress=interrupt)
        cache = self.pipeline/"09_model_tasks"
        self.assertEqual(len(list(cache.glob("*/task_manifest.json"))), 1)
        self.assertFalse((self.pipeline/"09_local_models").exists())
        with patch("pavement_analytics.full_modeling.search_levels", wraps=search_levels) as fitting:
            summaries, _, artifacts = run_full_models(self.pipeline, self.report, self.cfg)
        self.assertEqual(fitting.call_count, 1)
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(path.is_file() for path in artifacts))
        self.assertEqual(len(list(cache.glob("*/task_manifest.json"))), 2)

    def test_fit_budget_is_saved_and_rejected_before_search(self):
        self.cfg["local_model_options"] = {"max_fit_calls": 1}
        with patch("pavement_analytics.full_modeling.search_levels") as fitting:
            with self.assertRaisesRegex(SafeError, "LOCAL_MODEL_FIT_BUDGET_EXCEEDED"):
                run_full_models(self.pipeline, self.report, self.cfg)
        fitting.assert_not_called()
        self.assertGreater(json.loads((self.pipeline/"model_fit_budget.json").read_text())["fit_calls_upper_bound"], 1)


class IntervalWeightTests(unittest.TestCase):
    def test_interval_weighted_error_uses_exact_saved_weights(self):
        table = pd.DataFrame({"row_id": range(20), "group_id": np.repeat([f"g{i}" for i in range(10)], 2),
                              "x": np.tile([0., 1.], 10), "target": np.tile([0., 100.], 10),
                              "delta_days": np.tile([1000., 1.], 10)})
        plan = make_partitions(table, folds=3)
        with temporary_directory() as root, patch("pavement_analytics.study_modeling.candidates", return_value={}):
            result = search_levels(table, ["x"], "target", Path(root) / "weighted", plan, [], weight_column="delta_days")
            self.assertAlmostEqual(result["holdout_mae"], 100 / 1001)
            standard = search_levels(table, ["x"], "target", Path(root) / "default", plan, [])
            self.assertAlmostEqual(standard["holdout_mae"], 50.)
            self.assertEqual(standard["selection_metric"], "mean_group_balanced_cv_MAE")

    def test_weight_role_cannot_become_predictor(self):
        table = pd.DataFrame({"row_id": range(10), "group_id": [f"g{i}" for i in range(10)],
                              "target": np.arange(10.), "delta_days": np.ones(10)})
        plan = make_partitions(table, folds=3)
        with temporary_directory() as root:
            with self.assertRaisesRegex(SafeError, "WEIGHT_MUST_NOT_BE_FEATURE"):
                search_levels(table, ["delta_days"], "target", Path(root) / "invalid", plan, [], weight_column="delta_days")


if __name__ == "__main__":
    unittest.main()
