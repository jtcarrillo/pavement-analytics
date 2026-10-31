"""Registry admission, old-model parity and visible candidate failures."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression

from pavement_analytics.model_registry import (
    CandidateSpec, REGISTRY_VERSION, build_candidates, candidate_catalog, candidate_specs,
)
from pavement_analytics.security import SafeError
from pavement_analytics.study_modeling import _estimator, make_partitions, search_levels
from pavement_analytics.synthetic_benchmark import create_bundle, run_smoke
from temp_support import temporary_directory


def records():
    rng = np.random.default_rng(81)
    size = 120
    x = rng.uniform(0, 10, size)
    return pd.DataFrame({"row_id": np.arange(size), "group_id": np.repeat([f"SYN{i:03d}" for i in range(15)], 8),
                         "climate": x, "age": rng.uniform(1, 25, size),
                         "soil": np.where(rng.random(size) < 0.5, "A", "B"),
                         "delta_days": rng.integers(365, 731, size),
                         "level_target": 3+2*x+rng.uniform(-0.02, 0.02, size)})


def legacy_candidates(seed):
    # Frozen pre-registry definitions are intentional: the release gate checks
    # numerical parity, rather than comparing two aliases of the new registry.
    return {
        **{f"enet_{alpha}": _estimator(ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=20000, random_state=seed))
           for alpha in (0.01, 0.1)},
        **{f"rf_{leaf}": _estimator(RandomForestRegressor(n_estimators=100, min_samples_leaf=leaf,
           max_depth=12, max_features=0.8, n_jobs=1, random_state=seed)) for leaf in (5, 20)},
    }


class ControlledRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, mode="ok", limit=80):
        self.mode, self.limit = mode, limit

    def fit(self, X, y, sample_weight=None):
        self.training_size_ = len(X)
        if self.mode == "fit" or (self.mode == "partial" and X.marker.sum() > 0):
            raise RuntimeError("generated-private-model-detail")
        if self.mode == "refit" and len(X) > self.limit:
            raise RuntimeError("generated-private-model-detail")
        self.estimator_ = LinearRegression().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.mode == "predict" or (self.mode == "refit_predict" and self.training_size_ > self.limit):
            raise RuntimeError("generated-private-model-detail")
        if self.mode == "nan":
            return np.full(len(X), np.nan)
        if self.mode == "shape":
            return np.zeros((len(X), 2))
        if self.mode == "negative":
            return np.full(len(X), -1.)
        return np.maximum(0, self.estimator_.predict(X))


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        temp = temporary_directory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.frame = records()
        self.plan = make_partitions(self.frame, folds=3, seed=42)

    def search(self, out="result", *, pool=None, frame=None, features=None, requested=None):
        data = self.frame if frame is None else frame
        chosen = ["climate", "age"] if features is None else features
        if pool is None:
            return search_levels(data, chosen, "level_target", self.root/out, self.plan, ["climate"],
                                 weight_column="delta_days", candidate_ids=requested)
        with patch("pavement_analytics.study_modeling.candidates", return_value=pool):
            return search_levels(data, chosen, "level_target", self.root/out, self.plan, ["climate"], weight_column="delta_days")

    def test_catalog_lists_only_implemented_families_and_stable_weighted_ids(self):
        catalog = candidate_catalog()
        self.assertEqual([r["candidate"] for r in catalog], ["median", "enet_0.01", "enet_0.1", "rf_5", "rf_20",
                         "et_leaf2_depth12", "et_leaf2_depthnone", "et_leaf10_depth12", "et_leaf10_depthnone",
                         "xgb_depth3_lr0.05", "xgb_depth3_lr0.1", "xgb_depth6_lr0.05", "xgb_depth6_lr0.1",
                         "xgb_depth3_lr0.05_es", "xgb_depth3_lr0.1_es", "xgb_depth6_lr0.05_es", "xgb_depth6_lr0.1_es",
                         "lgbm_leaves15_min10", "lgbm_leaves15_min30", "lgbm_leaves31_min10", "lgbm_leaves31_min30",
                         "lgbm_leaves15_min10_es", "lgbm_leaves15_min30_es", "lgbm_leaves31_min10_es", "lgbm_leaves31_min30_es",
                         "spline_knots5_alpha0.1", "spline_knots5_alpha1.0", "spline_knots8_alpha0.1", "spline_knots8_alpha1.0",
                         "two_stage_c0.1_leaf2", "two_stage_c0.1_leaf10", "two_stage_c1.0_leaf2", "two_stage_c1.0_leaf10"])
        self.assertEqual({r["family"] for r in catalog}, {"weighted_median", "elastic_net", "random_forest", "extra_trees", "xgboost", "lightgbm", "additive_spline", "two_stage"})
        self.assertTrue(all(r["supports_sample_weight"] and r["max_workers"] == 1 for r in catalog))
        json.dumps(catalog, allow_nan=False)
        specs = candidate_specs()
        specs[0].parameters["alpha"] = 99
        self.assertEqual(candidate_specs()[0].parameters["alpha"], 0.01)

    def test_all_original_factories_match_legacy_fits_and_saved_predictions(self):
        data = self.frame.copy()
        data.loc[data.index[::11], "age"] = np.nan
        train = data.iloc[:96]
        test = data.iloc[96:].copy()
        test["soil"] = "UNSEEN"
        old = legacy_candidates(42)
        new = build_candidates(42, requested=list(old))
        self.assertEqual(list(new), list(old))
        for name, model in new.items():
            with self.subTest(candidate=name):
                fit = clone(model).fit(train[["climate", "age", "soil"]], train.level_target, sample_weight=train.delta_days)
                reference = old[name].fit(train[["climate", "age", "soil"]], train.level_target, sample_weight=train.delta_days)
                actual = fit.predict(test[["climate", "age", "soil"]])
                np.testing.assert_allclose(actual, reference.predict(test[["climate", "age", "soil"]]), atol=1e-12, rtol=1e-12)
                path = self.root/(name+".joblib")
                joblib.dump(fit, path)
                np.testing.assert_allclose(joblib.load(path).predict(test[["climate", "age", "soil"]]), actual)
                self.assertEqual(type(fit).__module__, "pavement_analytics.study_modeling")

    def test_cv_selection_scores_and_predictions_match_the_frozen_old_grid(self):
        before = self.search("legacy", pool=legacy_candidates(42))
        after = self.search("registry", requested=list(legacy_candidates(42)))
        for key in ("candidate", "cv_mae", "cv_mae_std", "holdout_mae", "holdout_rmse", "holdout_r2", "climate_added_skill"):
            self.assertEqual(before[key], after[key])
        for file in ("holdout_predictions.csv", "fold_scores.csv", "oof_predictions.csv"):
            pd.testing.assert_frame_equal(pd.read_csv(self.root/"legacy"/file), pd.read_csv(self.root/"registry"/file))

    def test_explicit_selector_limits_candidates_but_always_retains_the_baseline(self):
        summary = self.search(requested=["enet_0.01"])
        leaders = pd.read_csv(self.root/"result/leaderboard.csv")
        self.assertEqual(set(leaders.candidate), {"median", "enet_0.01"})
        self.assertEqual(summary["candidate_family"], "elastic_net")
        self.assertEqual(summary["candidate_registry_version"], REGISTRY_VERSION)
        artifact = joblib.load(self.root/"result/model.joblib")
        self.assertEqual(artifact["fit_partition"], "development")
        self.assertEqual(artifact["candidate_parameters"]["alpha"], 0.01)
        self.assertIn("scikit-learn", artifact["environment"])

    def test_unknown_and_duplicate_explicit_selectors_fail_before_output_creation(self):
        for selected in (["unimplemented"], ["enet_0.01", "enet_0.01"]):
            with self.assertRaisesRegex(SafeError, "UNKNOWN_OR_DUPLICATE"):
                self.search(requested=selected)
            self.assertFalse((self.root/"result").exists())

    def test_optional_missing_dependency_is_recorded_and_required_one_fails(self):
        factory = Mock(side_effect=AssertionError("Must not construct an unavailable model"))
        spec = CandidateSpec("missing", "generated", {}, factory, required_package="generated-package",
                             import_name="generated_dependency_that_does_not_exist")
        pool = build_candidates(specs=[spec])
        self.assertEqual(dict(pool), {})
        self.assertEqual(pool.entries[0]["candidate_status"], "unavailable")
        summary = self.search(pool=pool)
        self.assertEqual(summary["candidate"], "median")
        self.assertEqual(summary["candidate_unavailable_count"], 2)
        leaders = pd.read_csv(self.root/"result/leaderboard.csv")
        self.assertTrue(leaders.loc[leaders.candidate.eq("missing"), "candidate_status"].eq("unavailable").all())
        with self.assertRaisesRegex(SafeError, "MODEL_DEPENDENCY_UNAVAILABLE"):
            build_candidates(requested=["missing"], specs=[spec])
        factory.assert_not_called()

    def test_unsupported_sample_weights_are_visible_without_fitting(self):
        spec = CandidateSpec("unweighted", "generated", {}, Mock(), supports_sample_weight=False)
        pool = build_candidates(specs=[spec])
        self.assertEqual(pool.entries[0]["candidate_status"], "unsupported")
        self.assertFalse(pool)
        with self.assertRaisesRegex(SafeError, "MODEL_SAMPLE_WEIGHTS_REQUIRED"):
            build_candidates(requested=["unweighted"], specs=[spec])

    def test_constructor_failure_has_a_fixed_code_and_private_diagnostic(self):
        spec = CandidateSpec("broken", "generated", {}, Mock(side_effect=RuntimeError("generated-private-model-detail")))
        pool = build_candidates(specs=[spec])
        self.assertEqual(pool.entries[0]["candidate_status"], "failed_construct")
        self.search(pool=pool)
        audit = (self.root/"result/candidate_audit.json").read_text()
        self.assertIn("MODEL_CONSTRUCTION_FAILED", audit)
        self.assertNotIn("generated-private-model-detail", audit)
        self.assertIn("generated-private-model-detail", (self.root/"result/candidate_errors.log").read_text())

    def test_fit_and_invalid_prediction_failures_cannot_win_or_disappear(self):
        for mode in ("fit", "predict", "nan", "shape", "negative"):
            with self.subTest(mode=mode):
                summary = self.search(mode, pool={"broken": ControlledRegressor(mode)})
                self.assertEqual(summary["candidate"], "median")
                self.assertEqual(summary["candidate_failure_count"], 2)
                leader = pd.read_csv(self.root/mode/"leaderboard.csv")
                failed = leader.loc[leader.candidate.eq("broken")]
                self.assertEqual(len(failed), 2)
                self.assertTrue(failed.cv_mae.isna().all())
                self.assertFalse(failed.selected.any())
                self.assertNotIn("generated-private-model-detail", (self.root/mode/"candidate_audit.json").read_text())

    def test_partial_fold_coverage_is_retained_as_a_failure_not_a_valid_mean(self):
        frame = self.frame.copy()
        fault_group = self.plan.loc[self.plan.fold.eq(0), "group_id"].iloc[0]
        frame["marker"] = 0.
        frame.loc[frame.group_id.eq(fault_group).idxmax(), "marker"] = 1.
        result = self.search(pool={"partial": ControlledRegressor("partial")}, frame=frame,
                             features=["climate", "age", "marker"])
        self.assertEqual(result["candidate"], "median")
        leader = pd.read_csv(self.root/"result/leaderboard.csv")
        failed = leader.loc[leader.candidate.eq("partial")]
        self.assertTrue(failed.folds_completed.eq(1).all())
        self.assertTrue(failed.cv_mae.isna().all())
        self.assertTrue(failed.candidate_status.eq("failed_fit").all())

    def test_refit_or_final_prediction_failure_uses_the_next_cv_candidate(self):
        for mode in ("refit", "refit_predict"):
            with self.subTest(mode=mode):
                result = self.search(mode, pool={"fragile": ControlledRegressor(mode)})
                self.assertEqual(result["candidate"], "median")
                leaders = pd.read_csv(self.root/mode/"leaderboard.csv")
                self.assertTrue(leaders.loc[leaders.candidate.eq("fragile"), "candidate_status"].str.startswith("failed_refit").any())
                self.assertEqual(leaders.loc[leaders.feature_set.eq("all") & leaders.selected, "candidate"].tolist(), ["median"])

    def test_holdout_label_changes_do_not_change_selection_or_registry_predictions(self):
        before = self.search("before", requested=["enet_0.01"])
        altered = self.frame.copy()
        held = altered.group_id.isin(self.plan.loc[self.plan.partition.eq("holdout"), "group_id"])
        altered.loc[held, "level_target"] += 1000
        after = self.search("after", frame=altered, requested=["enet_0.01"])
        self.assertEqual(before["candidate"], after["candidate"])
        self.assertEqual(before["cv_mae"], after["cv_mae"])
        a, b = (pd.read_csv(self.root/p/"holdout_predictions.csv") for p in ("before", "after"))
        np.testing.assert_allclose(a.prediction, b.prediction)

    def test_identity_outcome_and_oracle_columns_are_forbidden_predictors(self):
        for forbidden in ("group_id", "PCI_end", "level_other", "truth_signal", "level_x__conditional_mean", "zero__p_positive"):
            frame = self.frame.copy()
            if forbidden not in frame:
                frame[forbidden] = 1.
            with self.assertRaisesRegex(SafeError, "UNSAFE_MODEL_FEATURE"):
                self.search(frame=frame, features=["climate", forbidden])

    def test_registry_rejects_invalid_definitions_and_nonfinite_parameters(self):
        for spec in (CandidateSpec("../bad", "generated", {}, Mock()),
                     CandidateSpec("bad", "generated", {"alpha": np.nan}, Mock()),
                     CandidateSpec("bad", "generated", {}, Mock(), max_workers=0)):
            with self.assertRaises(SafeError):
                build_candidates(specs=[spec])

    def test_benchmark_selector_runs_only_the_required_installed_candidate(self):
        bundle = create_bundle(self.root/"bundle", profile="smoke")
        summaries = run_smoke(bundle, self.root/"selected", candidate_ids=["enet_0.01"])
        self.assertEqual(len(summaries), 3)
        for summary in summaries:
            frame = pd.read_csv(self.root/"selected/PCC"/summary["target"]/"leaderboard.csv")
            self.assertEqual(set(frame.candidate), {"median", "enet_0.01"})


if __name__ == "__main__":
    unittest.main()
