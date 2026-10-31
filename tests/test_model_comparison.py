"""Family evaluation, spawned deadlines and verified complete-task reuse."""
import io
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from pavement_analytics.io import write_json
from pavement_analytics.model_comparison import correlated_blocks
from pavement_analytics.model_registry import candidate_catalog
from pavement_analytics.model_runtime import (ModelRuntime, ModelTimeout, check_budget,
    fit_budget, model_options, model_progress_text, runtime_context, sanitize_model_event)
from pavement_analytics.model_tasks import task_key, verified_task
from pavement_analytics.progress import FullProgress
from pavement_analytics.security import Boundary, SafeError, source_root
from pavement_analytics.study_modeling import MedianRegressor, search_levels
from pavement_analytics.study_reporting import build_study_report
from pavement_analytics.synthetic_benchmark import FEATURES, CLIMATE_FEATURES, records
from temp_support import temporary_directory


REPRESENTATIVES = ["enet_0.01", "rf_5", "et_leaf10_depth12", "xgb_depth3_lr0.1_es",
    "lgbm_leaves15_min10_es", "spline_knots5_alpha1.0", "two_stage_c1.0_leaf10"]
AVAILABLE = {x["candidate"] for x in candidate_catalog() if x["availability"] == "available"}
PRESENT = [x for x in REPRESENTATIVES if x in AVAILABLE]
CALIBRATION_SEEDS = (20261011, 20261012, 20261013)


class SlowModel(RegressorMixin, BaseEstimator):
    def __init__(self, mode="fit"):
        self.mode = mode

    def fit(self, X, y, sample_weight=None):
        if self.mode == "fit":
            time.sleep(2.)
        if self.mode == "error":
            raise RuntimeError("generated-private-model-detail")
        self.mean_ = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, X):
        if self.mode == "predict":
            time.sleep(2.)
        return np.full(len(X), self.mean_)


class RefitFailureModel(RegressorMixin, BaseEstimator):
    def __init__(self, limit=100):
        self.limit = limit

    def fit(self, X, y, sample_weight=None):
        if len(X) > self.limit:
            raise RuntimeError("generated-private-refit-detail")
        self.mean_ = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        table, _, _, self.plan, _, _ = records(profile="smoke")
        self.table = table.loc[table.material.eq("PCC")].copy()

    def search(self, name, target="level_zero_inflated", ids=None, frame=None, isolated=False):
        return search_levels(self.table if frame is None else frame, FEATURES, target, self.root/name,
            self.plan, CLIMATE_FEATURES, weight_column="delta_days", model_options={"isolated": isolated,
            "candidate_ids": ["enet_0.01", "two_stage_c1.0_leaf10"] if ids is None else ids})

    def test_every_present_family_has_independent_selections_and_serialized_predictions(self):
        summary = self.search("all", ids=PRESENT, isolated=True)
        root = self.root/"all"
        comparisons = pd.read_csv(root/"candidate_comparisons.csv")
        predictions = pd.read_csv(root/"candidate_predictions.csv")
        families = pd.read_csv(root/"family_summary.csv")
        self.assertEqual(len(comparisons), 2*(len(PRESENT)+1))
        self.assertTrue(comparisons.comparison_status.eq("completed").all())
        self.assertTrue(comparisons.prediction_parity.all())
        self.assertEqual(summary["comparison_failure_count"], 0)
        self.assertEqual(set(families.feature_set), {"all", "nonclimate"})
        runtime = json.loads((root/"model_runtime.json").read_text())
        self.assertLessEqual(runtime["fit_calls"], runtime["fit_calls_upper_bound"])
        self.assertEqual(runtime["estimator_threads"], 1)
        for row in comparisons.itertuples():
            saved = predictions.loc[predictions.candidate.eq(row.candidate) & predictions.feature_set.eq(row.feature_set)]
            artifact = joblib.load(root/row.model_path)
            X = self.table.set_index("row_id").loc[saved.row_id, artifact["features"]]
            np.testing.assert_allclose(artifact["estimator"].predict(X), saved.prediction, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(saved.weight, self.table.set_index("row_id").loc[saved.row_id, "delta_days"])
            self.assertEqual(artifact["fit_partition"], "development")
            self.assertFalse(set(saved.group_id)&set(self.plan.loc[self.plan.partition.eq("development"), "group_id"]))
        for row in families.itertuples():
            eligible = comparisons.loc[comparisons.family.eq(row.family) & comparisons.feature_set.eq(row.feature_set)]
            self.assertEqual(row.candidate, eligible.sort_values(["cv_mae", "candidate"]).iloc[0].candidate)
        importance = pd.read_csv(root/"family_importance.csv")
        self.assertIn("climate_block", set(importance.kind))
        self.assertTrue(importance.repeats.eq(5).all())
        self.assertIn("heuristic_support", importance)
        self.assertNotIn("reliable", importance)
        self.assertEqual(set(importance.loc[importance.family.eq("two_stage"), "metric"]),
                         {"mae", "occurrence_brier", "positive_severity_mae"})

    def test_family_choices_and_fitted_predictions_ignore_holdout_labels(self):
        self.search("before")
        changed = self.table.copy()
        held = changed.group_id.isin(self.plan.loc[self.plan.partition.eq("holdout"), "group_id"])
        changed.loc[held, "level_zero_inflated"] += 1000.
        self.search("after", frame=changed)
        frames = [pd.read_csv(self.root/n/"candidate_comparisons.csv") for n in ("before", "after")]
        for column in ("candidate", "family", "feature_set", "cv_mae", "family_selected", "primary_selected", "training"):
            pd.testing.assert_series_equal(frames[0][column], frames[1][column])
        a, b = (pd.read_csv(self.root/n/"candidate_predictions.csv") for n in ("before", "after"))
        np.testing.assert_array_equal(a.prediction, b.prediction)
        self.assertEqual((self.root/"before/candidate_audit.json").read_bytes(), (self.root/"after/candidate_audit.json").read_bytes())

    def test_calibration_selection_and_prediction_parity_with_legacy_path(self):
        for seed in CALIBRATION_SEEDS:
            with self.subTest(seed=seed):
                table, _, _, plan, _, _ = records(seed=seed, profile="smoke")
                table = table.loc[table.material.eq("PCC")]
                old = search_levels(table, FEATURES, "level_linear", self.root/f"old-{seed}", plan, CLIMATE_FEATURES,
                                    seed=seed, weight_column="delta_days", candidate_ids=PRESENT)
                new = search_levels(table, FEATURES, "level_linear", self.root/f"new-{seed}", plan, CLIMATE_FEATURES,
                    seed=seed, weight_column="delta_days", model_options={"candidate_ids": PRESENT, "comparisons": False})
                self.assertEqual(old["candidate"], new["candidate"])
                self.assertAlmostEqual(old["cv_mae"], new["cv_mae"], delta=1e-12)
                a, b = (pd.read_csv(self.root/f"{name}-{seed}"/"holdout_predictions.csv") for name in ("old", "new"))
                for column in ("prediction", "nonclimate_prediction", "baseline_prediction"):
                    np.testing.assert_allclose(a[column], b[column], rtol=1e-12, atol=1e-12)

    def test_training_only_correlated_blocks_ignore_holdout_extremes(self):
        development = pd.DataFrame({"a": np.arange(50.), "proxy": np.arange(50.)*3,
                                    "independent": np.random.default_rng(5).normal(size=50), "category": ["A", "B"]*25})
        self.assertEqual(correlated_blocks(development, list(development)), [["a", "proxy"]])
        frame = self.table.copy()
        frame["driver"] = np.arange(len(frame), dtype=float)
        frame["driver_proxy"] = frame.driver*2
        search_levels(frame, ["driver", "driver_proxy"], "level_linear", self.root/"blocks", self.plan, ["driver", "driver_proxy"],
                      model_options={"isolated": False, "candidate_ids": ["enet_0.01"]})
        importance = pd.read_csv(self.root/"blocks/family_importance.csv")
        blocks = importance.loc[importance.kind.eq("correlated_block")]
        self.assertFalse(blocks.empty)
        self.assertEqual(json.loads(blocks.iloc[0]["columns"]), ["driver", "driver_proxy"])

    def test_constant_and_missing_tasks_keep_explicit_family_statuses(self):
        _, _, _, plan, _, edges = records(profile="smoke")
        for target in ("level_constant_zero", "level_positive_only_holdout", "level_all_missing"):
            summary = search_levels(edges, FEATURES, target, self.root/target, plan, CLIMATE_FEATURES,
                model_options={"isolated": False, "candidate_ids": ["enet_0.01", "two_stage_c1.0_leaf10"]})
            self.assertFalse(summary["importance_reliable"])
            self.assertTrue(pd.read_csv(self.root/target/"family_importance.csv").empty)
            if summary["status"] == "constant_target":
                families = pd.read_csv(self.root/target/"family_summary.csv")
                learned = families.loc[families.family.ne("weighted_median")]
                self.assertTrue(learned.comparison_status.eq("no_completed_candidate").all())
                self.assertTrue(learned.candidate_status.eq("not_attempted_constant_target").all())

    def test_constant_predictors_have_zero_importance_in_every_component_loss(self):
        table = self.table.copy()
        table["constant"], table["other_constant"] = 1., 2.
        table["level_zero_inflated"] = np.tile([0., 0., 2., 8.], len(table)//4)
        search_levels(table, ["constant", "other_constant"], "level_zero_inflated", self.root/"constant-features",
            self.plan, ["constant"], weight_column="delta_days",
            model_options={"isolated": False, "candidate_ids": ["two_stage_c1.0_leaf10"]})
        importance = pd.read_csv(self.root/"constant-features/family_importance.csv")
        self.assertEqual(set(importance.metric), {"mae", "occurrence_brier", "positive_severity_mae"})
        np.testing.assert_allclose(importance.importance_mean, 0., rtol=0, atol=1e-12)

    def test_losing_candidate_refit_failure_is_reported_without_changing_primary(self):
        table = self.table.copy()
        table["level_zero_inflated"] = np.where(np.arange(len(table))%10 == 0, 10., 0.)
        limit = int(table.group_id.isin(self.plan.loc[self.plan.partition.eq("development"), "group_id"]).sum())-1
        with patch("pavement_analytics.study_modeling.candidates", return_value={"late_failure": RefitFailureModel(limit)}):
            result = search_levels(table, FEATURES, "level_zero_inflated", self.root/"late_failure", self.plan, CLIMATE_FEATURES,
                                   model_options={"isolated": False})
        self.assertEqual(result["candidate"], "median")
        self.assertEqual(result["comparison_failure_count"], 2)
        comparisons = pd.read_csv(self.root/"late_failure/candidate_comparisons.csv")
        rows = comparisons.loc[comparisons.candidate.eq("late_failure")]
        self.assertTrue(rows.candidate_status.eq("completed").all())
        self.assertTrue(rows.comparison_status.eq("failed_refit_or_predict").all())
        self.assertNotIn("generated-private-refit-detail", (self.root/"late_failure/candidate_comparisons.csv").read_text())

    def test_old_and_new_report_readers_consolidate_without_group_identity_in_html(self):
        target = "level_zero_inflated"
        summary = self.search("materials/PCC/"+target)
        build_study_report(self.root, [{"material": "PCC", **summary}], {})
        html = (self.root/"dashboard.html").read_text()
        self.assertIn("Model-family comparisons and explanations", html)
        self.assertIn("occurrence_brier", html)
        for group in self.table.group_id.unique():
            self.assertNotIn(str(group), html)
        for filename in ("candidate_comparisons.csv", "candidate_predictions.csv", "family_summary.csv", "family_importance.csv", "model_runtime.csv"):
            frame = pd.read_csv(self.root/filename)
            self.assertFalse(frame.empty)
            self.assertEqual(set(frame.material), {"PCC"})
        # Earlier diagnostic exports used a misleading flag name. Reading them
        # preserves the values under the explicitly heuristic label.
        legacy = pd.read_csv(self.root/"materials/PCC"/target/"family_importance.csv")
        legacy.rename(columns={"heuristic_support": "reliable"}).to_csv(self.root/"materials/PCC"/target/"family_importance.csv", index=False)
        build_study_report(self.root, [{"material": "PCC", **summary}], {})
        self.assertIn("heuristic_support", pd.read_csv(self.root/"family_importance.csv"))
        # An old task with no R7 sidecars remains a valid report input.
        old = search_levels(self.table, FEATURES, "level_linear", self.root/"old-models/materials/PCC/level_linear",
                            self.plan, CLIMATE_FEATURES, candidate_ids=["enet_0.01"])
        old_report = self.root/"old-report"; old_report.mkdir()
        build_study_report(old_report, [{"material": "PCC", **old}], {}, model_root=self.root/"old-models")
        self.assertTrue(pd.read_csv(old_report/"family_summary.csv").empty)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.X = pd.DataFrame({"x": np.arange(24.)})
        self.y, self.weights = np.arange(24.)+1, np.ones(24)
        self.groups = np.repeat(["SYN_A", "SYN_B", "SYN_C"], 8)

    def test_fit_and_prediction_deadlines_kill_and_restart_worker(self):
        options = model_options({"fit_timeout_seconds": .2, "prediction_timeout_seconds": .2})
        with ModelRuntime(self.root, options) as runtime:
            runtime_context(candidate="slow", family="test", feature_set="all", phase="cv", fold=0)
            with self.assertRaises(ModelTimeout):
                runtime.call("fit", SlowModel(), self.X, self.y, self.weights, self.groups)
            self.assertIsNone(runtime.process)
            fitted = runtime.call("fit", SlowModel("predict"), self.X, self.y, self.weights, self.groups)
            with self.assertRaises(ModelTimeout):
                runtime.call("predict", fitted, self.X)
            fitted = runtime.call("fit", MedianRegressor(), self.X, self.y, self.weights, self.groups)
            predicted, _ = runtime.call("predict", fitted, self.X)
            np.testing.assert_array_equal(predicted, 12.)
        events = pd.read_csv(self.root/"model_runtime.csv")
        self.assertEqual(events.status.eq("timed_out").sum(), 2)
        self.assertFalse(any(p.name.startswith("model-") for p in self.root.iterdir()))

    def test_cv_timeout_has_a_visible_status_and_cannot_win(self):
        table, _, _, plan, _, _ = records(profile="smoke")
        with patch("pavement_analytics.study_modeling.candidates", return_value={"slow": SlowModel()}):
            summary = search_levels(table, FEATURES, "level_linear", self.root/"search", plan, CLIMATE_FEATURES,
                model_options={"comparisons": False, "fit_timeout_seconds": .2})
        self.assertEqual(summary["candidate"], "median")
        leaders = pd.read_csv(self.root/"search/leaderboard.csv")
        rows = leaders.loc[leaders.candidate.eq("slow")]
        self.assertTrue(rows.candidate_status.eq("timed_out_fit").all())
        self.assertTrue(rows.cv_mae.isna().all())
        self.assertEqual(summary["candidate_failure_count"], 2)

    def test_native_failure_details_are_private_and_next_model_still_runs(self):
        with ModelRuntime(self.root, model_options()) as runtime:
            with self.assertRaisesRegex(SafeError, "MODEL_WORKER_OPERATION_FAILED") as error:
                runtime.call("fit", SlowModel("error"), self.X, self.y, self.weights, self.groups)
            self.assertNotIn("generated-private-model-detail", str(error.exception))
            fitted = runtime.call("fit", MedianRegressor(), self.X, self.y, self.weights, self.groups)
            self.assertTrue(np.isfinite(runtime.call("predict", fitted, self.X)[0]).all())
        self.assertIn("generated-private-model-detail", (self.root/"model_worker.log").read_text())

    def test_cancellation_cleans_worker_and_records_cancelled_state(self):
        with self.assertRaises(KeyboardInterrupt):
            with ModelRuntime(self.root, model_options()) as runtime:
                runtime.call("fit", MedianRegressor(), self.X, self.y, self.weights, self.groups)
                process = runtime.process
                with patch.object(runtime, "_receive", side_effect=KeyboardInterrupt):
                    runtime.call("fit", SlowModel(), self.X, self.y, self.weights, self.groups)
        self.assertFalse(process.is_alive())
        self.assertEqual(json.loads((self.root/"model_runtime.json").read_text())["status"], "cancelled")
        self.assertEqual(pd.read_csv(self.root/"model_runtime.csv").status.iloc[-1], "cancelled")

    def test_options_and_fit_budget_fail_before_training(self):
        for options in ({"fit_timeout_seconds": 0}, {"prediction_timeout_seconds": float("inf")},
                        {"isolated": "yes"}, {"candidate_ids": ["unknown"]}, {"max_fit_calls": True}, {"workers": 20}):
            with self.subTest(options=options), self.assertRaises(SafeError):
                model_options(options)
        options = model_options({"max_fit_calls": 1})
        budget = fit_budget(5, options)
        with self.assertRaisesRegex(SafeError, "FIT_BUDGET_EXCEEDED"):
            check_budget(budget, options)
        self.assertGreater(budget["native_fits_upper_bound"], budget["fit_calls_upper_bound"])
        self.assertEqual((budget["fit_workers"], budget["estimator_threads"]), (1, 1))

    def test_progress_accepts_only_bounded_known_fields(self):
        event = {"event": "model", "status": "completed", "candidate": "enet_0.01", "phase": "cv",
                 "operation": "fit", "feature_set": "all", "fold": 1, "completed_fits": 3, "planned_fits": 10,
                 "traceback": "generated-private-model-detail", "path": "/generated/private"}
        clean = sanitize_model_event(event)
        self.assertNotIn("traceback", clean)
        self.assertNotIn("path", clean)
        stream = io.StringIO(); FullProgress(stream)(event)
        self.assertIn("enet_0.01", stream.getvalue())
        self.assertNotIn("generated-private", stream.getvalue())
        self.assertIn("upper bound 10", model_progress_text(event))
        self.assertIsNone(sanitize_model_event({"event": "model", "status": []}))
        self.assertNotIn("candidate", sanitize_model_event({**event, "candidate": "../secret"}))


class TaskCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.table, _, _, self.plan, _, _ = records(profile="smoke")
        self.options = model_options({"isolated": False, "candidate_ids": ["enet_0.01"]})

    def key(self, **overrides):
        values = {"table": self.table, "plan": self.plan, "features": FEATURES, "climate": CLIMATE_FEATURES,
                  "target": "level_linear", "seed": 42, "weight_column": "delta_days", "options": self.options}
        values.update(overrides)
        return task_key(**values)

    def test_task_key_covers_data_plan_features_seed_parameters_code_and_environment(self):
        key = self.key()
        changed = self.table.copy(); changed.loc[0, "level_linear"] += 1
        plan = self.plan.copy(); plan.loc[0, "fold"] = 99
        for kwargs in ({"table": changed}, {"plan": plan}, {"features": FEATURES[::-1]}, {"climate": []},
                       {"seed": 43}, {"options": {**self.options, "fit_timeout_seconds": 2.}}):
            self.assertNotEqual(key, self.key(**kwargs))
        with patch("pavement_analytics.pipeline.code_digest", return_value="changed"):
            self.assertNotEqual(key, self.key())
        with patch("pavement_analytics.model_registry.environment_versions", return_value={"test": "changed"}):
            self.assertNotEqual(key, self.key())

    def build(self, destination):
        return search_levels(self.table, FEATURES, "level_linear", destination, self.plan, CLIMATE_FEATURES,
                             weight_column="delta_days", model_options=self.options)

    def test_complete_task_reuses_without_fit_and_rejects_tampering(self):
        key = self.key(); cache = self.root/"cache"
        first = verified_task(cache, key, self.build, self.root/"first")
        with patch("pavement_analytics.study_modeling._fit", side_effect=AssertionError("must reuse")):
            second = verified_task(cache, key, self.build, self.root/"second")
        self.assertEqual(first, second)
        path = cache/key/"candidate_predictions.csv"
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(SafeError, "MODEL_TASK_CHECKPOINT_CHANGED"):
            verified_task(cache, key, self.build, self.root/"third")
        self.assertFalse((self.root/"third").exists())

    def test_cancellation_keeps_completed_siblings_and_never_reuses_partial_target(self):
        cache = self.root/"cache"; first = self.key()
        verified_task(cache, first, self.build, self.root/"first")
        second = self.key(seed=43)
        def cancelled(destination):
            destination.mkdir()
            write_json(destination/"summary.json", {"status": "partial"})
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            verified_task(cache, second, cancelled, self.root/"partial")
        self.assertTrue((cache/first/"task_manifest.json").exists())
        self.assertFalse((cache/second).exists())
        self.assertEqual(len(list(cache.glob(".pending-model-*"))), 1)
        verified_task(cache, second, self.build, self.root/"recovered")
        self.assertTrue((cache/second/"task_manifest.json").exists())

    def test_checkpoint_paths_reject_symlinks_and_code_tree_destinations(self):
        key = self.key(); cache = self.root/"cache"
        verified_task(cache, key, self.build, self.root/"first")
        path = cache/key/"summary.json"
        path.unlink(); path.symlink_to(self.root/"first/summary.json")
        with self.assertRaises(SafeError):
            verified_task(cache, key, self.build, self.root/"second")
        with self.assertRaises(SafeError):
            verified_task(self.root/"other-cache", self.key(seed=43), self.build, source_root()/"generated-private-output")
        with self.assertRaisesRegex(SafeError, "INVALID_MODEL_TASK_KEY"):
            verified_task(cache, "../invalid", self.build, self.root/"invalid")


if __name__ == "__main__":
    unittest.main()
