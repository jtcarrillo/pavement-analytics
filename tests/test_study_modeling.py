"""Generated populations exercise shared splits, leakage guards and rankings."""
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

from pavement_analytics.security import SafeError
from pavement_analytics.study_modeling import (
    FeatureCleaner, _estimator, group_weights, make_partitions, search_levels,
)
from temp_support import temporary_directory


def records():
    rng = np.random.default_rng(28)
    size = 20 * 12
    climate = rng.uniform(0, 10, size)
    return pd.DataFrame({
        "row_id": np.arange(size), "group_id": np.repeat([f"generated-{i}" for i in range(20)], 12),
        "material": "AC", "temperature": climate, "age": rng.uniform(1, 30, size),
        "soil": np.where(rng.uniform(size=size) < .5, "type_a", "type_b"),
        "level_strong": 2 + 3 * climate + rng.normal(0, .05, size),
        "level_weak": rng.uniform(0, 30, size),
    })


def fast_candidates(seed):
    # Real estimators; fewer forest trees keep repeatability tests inexpensive.
    return {"linear": _estimator(ElasticNet(alpha=.01, l1_ratio=.5, max_iter=10000, random_state=seed)),
            "forest": _estimator(RandomForestRegressor(n_estimators=12, min_samples_leaf=3,
                                                        max_depth=8, n_jobs=1, random_state=seed))}


class PartitionTests(unittest.TestCase):
    def test_same_assignments_after_row_reordering_and_target_change(self):
        frame = records()
        first = make_partitions(frame, folds=3)
        altered = frame.sample(frac=1, random_state=8).copy()
        altered["level_strong"] = np.nan
        pd.testing.assert_frame_equal(first, make_partitions(altered, folds=3))
        self.assertFalse(first.group_id.duplicated().any())
        self.assertEqual(set(first.loc[first.partition.eq("development"), "fold"]), {0, 1, 2})
        self.assertEqual(set(first.loc[first.partition.eq("holdout"), "fold"]), {-1})
        self.assertEqual(first.planned_folds.unique().tolist(), [3])

    def test_weights_balance_groups_of_different_sizes(self):
        groups = pd.Series(["a"] * 2 + ["b"] * 10)
        weights = group_weights(groups)
        self.assertAlmostEqual(weights[groups.eq("a")].sum(), weights[groups.eq("b")].sum())
        self.assertAlmostEqual(weights.mean(), 1)

    def test_invalid_group_and_plan_are_rejected(self):
        frame = records()
        frame.loc[0, "group_id"] = None
        with self.assertRaisesRegex(SafeError, "VALID_GROUPS"):
            make_partitions(frame)
        frame = records()
        plan = make_partitions(frame)
        plan = pd.concat([plan, plan.iloc[:1]], ignore_index=True)
        with temporary_directory() as root:
            with self.assertRaisesRegex(SafeError, "INVALID_SHARED_PARTITIONS"):
                search_levels(frame, ["temperature"], "level_strong", Path(root) / "invalid", plan, ["temperature"])


class ModelingTests(unittest.TestCase):
    def test_strong_target_beats_unpredictable_target_and_reports_climate(self):
        frame = records()
        frame.loc[frame.index[::9], "age"] = np.nan
        plan = make_partitions(frame, folds=3)
        hold_groups = plan.loc[plan.partition.eq("holdout"), "group_id"]
        frame.loc[frame.group_id.isin(hold_groups), "soil"] = "unseen-category"
        with temporary_directory() as root, patch("pavement_analytics.study_modeling.candidates", fast_candidates):
            out = Path(root)
            strong = search_levels(frame, ["temperature", "age", "soil"], "level_strong", out / "strong", plan, ["temperature"])
            weak = search_levels(frame, ["temperature", "age", "soil"], "level_weak", out / "weak", plan, ["temperature"])
            self.assertGreater(strong["mae_skill"], .95)
            self.assertGreater(strong["holdout_r2"], .99)
            self.assertGreater(strong["climate_added_skill"], .95)
            self.assertLess(weak["mae_skill"], .15)
            self.assertTrue(strong["importance_reliable"])
            importance = pd.read_csv(out / "strong/importance.csv")
            individual = importance.loc[importance.kind.eq("feature")].sort_values("importance_mean", ascending=False)
            self.assertEqual(individual.feature.iloc[0], "temperature")
            self.assertEqual(importance.kind.eq("climate_block").sum(), 1)
            predictions = pd.read_csv(out / "strong/holdout_predictions.csv")
            self.assertEqual(set(predictions.group_id), set(hold_groups))
            artifact = joblib.load(out / "strong/model.joblib")
            self.assertEqual(artifact["fit_partition"], "development")
            matching = frame.set_index("row_id").loc[predictions.row_id]
            np.testing.assert_allclose(artifact["estimator"].predict(matching[artifact["features"]]), predictions.prediction)
            pd.testing.assert_frame_equal(pd.read_csv(out / "strong/folds.csv"), pd.read_csv(out / "weak/folds.csv"))
            oof = pd.read_csv(out / "strong/oof_predictions.csv")
            self.assertFalse(set(oof.group_id) & set(predictions.group_id))

    def test_holdout_outcomes_cannot_change_fitted_model_or_cv_selection(self):
        frame = records()
        plan = make_partitions(frame, folds=3)
        hold_groups = plan.loc[plan.partition.eq("holdout"), "group_id"]
        changed = frame.copy()
        changed.loc[changed.group_id.isin(hold_groups), "level_strong"] += 500
        with temporary_directory() as root, patch("pavement_analytics.study_modeling.candidates", fast_candidates):
            out = Path(root)
            before = search_levels(frame, ["temperature", "age"], "level_strong", out / "before", plan, ["temperature"])
            after = search_levels(changed, ["temperature", "age"], "level_strong", out / "after", plan, ["temperature"])
            self.assertEqual(before["candidate"], after["candidate"])
            self.assertEqual(before["cv_mae"], after["cv_mae"])
            np.testing.assert_allclose(pd.read_csv(out / "before/holdout_predictions.csv").prediction,
                                       pd.read_csv(out / "after/holdout_predictions.csv").prediction)
            self.assertFalse(after["importance_reliable"])

    def test_missing_target_fold_is_reported_without_redrawing_plan(self):
        frame = records()
        plan = make_partitions(frame, folds=3)
        missing_groups = plan.loc[plan.fold.eq(1), "group_id"]
        frame.loc[frame.group_id.isin(missing_groups), "level_strong"] = np.nan
        with temporary_directory() as root:
            out = Path(root) / "missing"
            result = search_levels(frame, ["temperature"], "level_strong", out, plan, ["temperature"])
            self.assertEqual(result["status"], "insufficient_fold_coverage")
            self.assertEqual(result["excluded_missing_target"], int(frame.level_strong.isna().sum()))
            pd.testing.assert_frame_equal(pd.read_csv(out / "folds.csv"), plan)
            self.assertFalse((out / "model.joblib").exists())

    def test_single_group_and_entirely_missing_target_remain_report_rows(self):
        frame = records().iloc[:12].copy()
        plan = make_partitions(frame, folds=5)
        with temporary_directory() as root:
            result = search_levels(frame, ["temperature"], "level_strong", Path(root) / "small", plan, ["temperature"])
            self.assertEqual(result["status"], "insufficient_groups")
            self.assertEqual(result["development_rows"], 0)
            frame["level_strong"] = np.nan
            result = search_levels(frame, ["temperature"], "level_strong", Path(root) / "missing", plan, ["temperature"])
            self.assertEqual(result["status"], "no_valid_target")
            self.assertEqual(result["excluded_missing_target"], len(frame))

    def test_constant_development_target_has_no_spurious_importance(self):
        frame = records()
        frame["level_strong"] = 0.0
        plan = make_partitions(frame, folds=3)
        with temporary_directory() as root:
            out = Path(root) / "constant"
            result = search_levels(frame, ["temperature"], "level_strong", out, plan, ["temperature"])
            self.assertEqual(result["status"], "constant_target")
            self.assertEqual(result["candidate"], "median")
            self.assertIsNone(result["mae_skill"])
            self.assertIsNone(result["holdout_r2"])
            self.assertEqual(result["holdout_mae"], 0)
            self.assertFalse(result["importance_reliable"])
            self.assertTrue(pd.read_csv(out / "importance.csv").empty)

    def test_preprocessing_types_are_learned_from_training_only(self):
        cleaner = FeatureCleaner().fit(pd.DataFrame({"numeric": ["1", "2", ""], "category": ["a", "b", None]}))
        changed = cleaner.transform(pd.DataFrame({"numeric": ["bad", "3", np.inf], "category": ["never-seen", "", None]}))
        self.assertEqual(cleaner.numeric_, ["numeric"])
        self.assertTrue(np.isnan(changed.numeric.iloc[0]))
        self.assertTrue(np.isnan(changed.numeric.iloc[2]))
        self.assertEqual(changed.category.iloc[0], "never-seen")


if __name__ == "__main__":
    unittest.main()
