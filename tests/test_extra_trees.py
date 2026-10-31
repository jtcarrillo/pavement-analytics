"""Admission tests use invented observations and the production evaluation path."""
import json
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor

from pavement_analytics.model_registry import build_candidates
from pavement_analytics.study_modeling import search_levels
from pavement_analytics.synthetic_benchmark import CLIMATE_FEATURES, FEATURES, records
from temp_support import temporary_directory


EXTRA_IDS = ["et_leaf2_depth12", "et_leaf2_depthnone", "et_leaf10_depth12", "et_leaf10_depthnone"]
CALIBRATION_SEEDS = (20261011, 20261012, 20261013)


class ExtraTreesTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def search(self, frame, plan, target, name, seed=20261031):
        return search_levels(frame, FEATURES, target, self.root/name, plan,
                             CLIMATE_FEATURES, seed=seed, weight_column="delta_days",
                             candidate_ids=EXTRA_IDS)

    def test_every_configuration_uses_weights_in_the_fitted_leaf_loss(self):
        # Identical inputs force a single leaf: the squared-error optimum is
        # the weighted mean, so this checks actual forwarding, not just an API.
        X = pd.DataFrame({"constant": np.ones(48)})
        y = np.repeat([0., 10.], 24)
        weights = np.repeat([1., 9.], 24)
        pool = build_candidates(requested=EXTRA_IDS)
        for name, estimator in pool.items():
            with self.subTest(candidate=name):
                fitted = clone(estimator).fit(X, y, sample_weight=weights)
                uniform = clone(estimator).fit(X, y, sample_weight=np.ones(48))
                np.testing.assert_allclose(fitted.predict(X), 9., rtol=0, atol=1e-12)
                np.testing.assert_allclose(uniform.predict(X), 5., rtol=0, atol=1e-12)
                model = fitted.estimator_.named_steps["model"]
                self.assertIsInstance(model, ExtraTreesRegressor)
                self.assertEqual((len(model.estimators_), model.n_jobs, model.bootstrap), (200, 1, False))

    def test_every_configuration_clones_and_serializes_with_missing_and_unseen_inputs(self):
        _, stress, _, plan, _, _ = records(profile="smoke")
        frame = stress.loc[stress.material.eq("PCC")].dropna(subset=["level_linear"]).copy()
        held = frame.group_id.isin(plan.loc[plan.partition.eq("holdout"), "group_id"])
        columns = [*FEATURES, "all_missing_predictor"]
        train, test = frame.loc[~held], frame.loc[held]
        for name, estimator in build_candidates(requested=EXTRA_IDS).items():
            with self.subTest(candidate=name):
                fitted = clone(estimator).fit(train[columns], train.level_linear, sample_weight=train.delta_days)
                prediction = fitted.predict(test[columns])
                self.assertEqual(prediction.shape, (len(test),))
                self.assertTrue(np.isfinite(prediction).all() and (prediction >= 0).all())
                encoder = fitted.estimator_.named_steps["prepare"].named_transformers_["categorical"].named_steps["encode"]
                self.assertNotIn("HOLDOUT_ONLY", {value for values in encoder.categories_ for value in values})
                numeric = fitted.estimator_.named_steps["clean"].numeric_
                imputer = fitted.estimator_.named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"]
                index = numeric.index("wet_days")
                self.assertAlmostEqual(imputer.statistics_[index], float(train.wet_days.median()))
                path = self.root/(name+".joblib")
                joblib.dump(fitted, path)
                np.testing.assert_allclose(joblib.load(path).predict(test[columns]), prediction, rtol=0, atol=1e-12)

    def test_shared_group_folds_and_selection_ignore_holdout_labels(self):
        frame, _, _, plan, _, _ = records(profile="smoke")
        frame = frame.loc[frame.material.eq("PCC")].copy()
        first = self.search(frame, plan, "level_threshold", "first")
        held = frame.group_id.isin(plan.loc[plan.partition.eq("holdout"), "group_id"])
        frame.loc[held, "level_threshold"] += 1000
        second = self.search(frame, plan, "level_threshold", "second")
        for field in ("candidate", "nonclimate_candidate", "cv_mae", "cv_mae_std", "candidate_parameters"):
            self.assertEqual(first[field], second[field])
        before, after = (pd.read_csv(self.root/name/"holdout_predictions.csv") for name in ("first", "second"))
        for field in ("prediction", "baseline_prediction", "nonclimate_prediction"):
            np.testing.assert_allclose(before[field], after[field], rtol=0, atol=1e-12)
        pd.testing.assert_frame_equal(plan, pd.read_csv(self.root/"first/folds.csv"))
        leaders = pd.read_csv(self.root/"first/leaderboard.csv")
        self.assertEqual(set(leaders.candidate), {"median", *EXTRA_IDS})
        self.assertEqual(len(leaders), 10)
        self.assertTrue(leaders.candidate_status.eq("completed").all())
        self.assertTrue(leaders.folds_completed.eq(3).all())
        oof = pd.read_csv(self.root/"first/oof_predictions.csv").merge(plan, on="group_id", suffixes=("", "_plan"))
        self.assertTrue(oof.partition.eq("development").all())
        self.assertTrue(oof.fold.eq(oof.fold_plan).all())
        audit = json.loads((self.root/"first/candidate_audit.json").read_text())
        self.assertEqual({a["family"] for a in audit["attempts"]}, {"weighted_median", "extra_trees"})
        self.assertIn("Predictive reliance, not causation", first["importance_note"])

    def test_constant_and_missing_labels_retain_explicit_statuses(self):
        _, _, _, plan, _, edges = records(profile="smoke")
        edges = edges.loc[edges.material.eq("PCC")]
        for target, status, value in (("level_constant_zero", "constant_target", 0),
                                      ("level_constant_positive", "constant_target", 7),
                                      ("level_development_constant", "constant_target", 0),
                                      ("level_all_missing", "no_valid_target", None),
                                      ("level_no_holdout_labels", "insufficient_groups", None)):
            with self.subTest(target=target):
                summary = self.search(edges, plan, target, target)
                self.assertEqual(summary["status"], status)
                self.assertFalse(summary["importance_reliable"])
                if value is not None:
                    prediction = pd.read_csv(self.root/target/"holdout_predictions.csv").prediction
                    np.testing.assert_array_equal(prediction, np.full(len(prediction), value))
                    leaders = pd.read_csv(self.root/target/"leaderboard.csv")
                    unused = leaders.loc[leaders.candidate.isin(EXTRA_IDS)]
                    self.assertTrue(unused.candidate_status.eq("not_attempted_constant_target").all())

    def test_nonlinear_signals_pass_frozen_development_gates_on_three_seeds(self):
        # These seeds, grid and 20% CV improvement gate were declared in the
        # roadmap before measurement. No holdout metric decides admission.
        for seed in CALIBRATION_SEEDS:
            frame, _, _, plan, _, _ = records(seed=seed, profile="standard")
            frame = frame.loc[frame.material.eq("PCC")]
            for target in ("level_threshold", "level_interaction"):
                with self.subTest(seed=seed, target=target):
                    name = f"calibration-{seed}-{target}"
                    summary = self.search(frame, plan, target, name, seed)
                    leaders = pd.read_csv(self.root/name/"leaderboard.csv")
                    all_features = leaders.loc[leaders.feature_set.eq("all")].set_index("candidate")
                    self.assertTrue(leaders.candidate_status.eq("completed").all())
                    self.assertLessEqual(all_features.loc[EXTRA_IDS, "cv_mae"].min(),
                                         0.8*all_features.loc["median", "cv_mae"])
                    self.assertEqual(summary["candidate_family"], "extra_trees")
                    self.assertEqual(summary["candidate_failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
