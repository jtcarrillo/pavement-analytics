"""Weighted additive spline admission and independent curve/structure checks."""
import json
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from pavement_analytics.model_registry import build_candidates
from pavement_analytics.security import SafeError
from pavement_analytics.spline_model import AdditiveSplineRegressor
from pavement_analytics.study_modeling import search_levels
from pavement_analytics.synthetic_benchmark import CLIMATE_FEATURES, FEATURES, records
from temp_support import temporary_directory


SPLINE_IDS = [f"spline_knots{knots}_alpha{alpha}" for knots in (5, 8) for alpha in (0.1, 1.0)]
CALIBRATION_SEEDS = (20261011, 20261012, 20261013)


class AdditiveSplineTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_every_configuration_uses_weights_in_its_squared_error_loss(self):
        X = pd.DataFrame({"constant": np.ones(48)})
        y = np.tile([0., 10.], 24)
        weights = np.where(y > 0, 9., 1.)
        for name, model in build_candidates(requested=SPLINE_IDS).items():
            with self.subTest(candidate=name):
                fitted = clone(model).fit(X, y, sample_weight=weights)
                uniform = clone(model).fit(X, y, sample_weight=np.ones(48))
                np.testing.assert_allclose(fitted.predict(X), 9., rtol=0, atol=1e-12)
                np.testing.assert_allclose(uniform.predict(X), 5., rtol=0, atol=1e-12)

    def test_every_configuration_clones_serializes_and_handles_stress(self):
        _, stress, _, plan, _, _ = records(profile="smoke")
        data = stress.loc[stress.material.eq("PCC")].dropna(subset=["level_smooth"])
        held = data.group_id.isin(plan.loc[plan.partition.eq("holdout"), "group_id"])
        train, test = data.loc[~held], data.loc[held]
        features = [*FEATURES, "all_missing_predictor"]
        for name, model in build_candidates(requested=SPLINE_IDS).items():
            with self.subTest(candidate=name):
                fitted = clone(model).fit_grouped(train[features], train.level_smooth,
                                                 sample_weight=train.delta_days, groups=train.group_id)
                prediction = fitted.predict(test[features])
                self.assertEqual(prediction.shape, (len(test),))
                self.assertTrue(np.isfinite(prediction).all() and (prediction >= 0).all())
                path = self.root/(name+".joblib")
                joblib.dump(fitted, path)
                np.testing.assert_allclose(joblib.load(path).predict(test[features]), prediction, rtol=0, atol=1e-12)
                repeated = clone(model).fit(train[features], train.level_smooth, sample_weight=train.delta_days)
                np.testing.assert_array_equal(repeated.predict(test[features]), prediction)
                schema = fitted.features_.schema()
                self.assertIn("all_missing_predictor", schema["constant_numeric_features"])
                self.assertNotIn("HOLDOUT_ONLY", schema["category_mapping"]["subgrade_category"])
                self.assertEqual(len(fitted.estimator_.coef_), len(schema["encoded_columns"]))

    def test_preprocessing_knots_roles_and_imputation_use_training_rows_only(self):
        X = pd.DataFrame({"climate": [0., 1., np.nan, 4., 8., 10.], "soil": ["A", "B", "A", "B", "A", "B"]})
        fitted = AdditiveSplineRegressor().fit(X, [2., 3., 4., 5., 6., 7.])
        before = json.dumps(fitted.features_.schema(), sort_keys=True)
        test = pd.DataFrame({"climate": [-1000., 1000., np.nan], "soil": ["UNSEEN"]*3})
        self.assertTrue(np.isfinite(fitted.predict(test)).all())
        self.assertEqual(json.dumps(fitted.features_.schema(), sort_keys=True), before)
        self.assertEqual(fitted.features_.bounds_["climate"], [0., 10.])
        self.assertEqual(fitted.features_.schema()["imputation_values"]["climate"], 4.)
        spline = fitted.features_.splines_["climate"]
        np.testing.assert_allclose(spline.bsplines_[0].t[3:-3], np.linspace(0, 10, 5))

    def test_declared_and_reviewed_numeric_category_codes_are_not_splined(self):
        X = pd.DataFrame({"Subgrade Category_start": [1, 2, 3, 1, 2, 3],
                          "Inspected Branch Use_start": [10, 20, 10, 20, 10, 20],
                          "Most Recent Major MR Code_start": [101, 102, 101, 102, 101, 102],
                          "custom": [4, 5, 4, 5, 4, 5],
                          "typed": pd.Categorical([1, 2, 1, 2, 1, 2]),
                          "flag": [True, False]*3, "climate": np.arange(6.)})
        fitted = AdditiveSplineRegressor(categorical_features=("custom",)).fit(X, np.arange(6.)+3)
        self.assertEqual(fitted.features_.numeric_, ["climate"])
        self.assertEqual(set(fitted.features_.encoders_), set(X)-{"climate"})
        self.assertEqual(set(fitted.features_.splines_), {"climate"})

    def test_missing_indicators_are_linear_and_constant_columns_skip_expansion(self):
        rng = np.random.default_rng(72)
        size = 180
        x = rng.uniform(0, 10, size)
        missing = np.arange(size)%5 == 0
        y = 20+x+8*missing
        X = pd.DataFrame({"climate": np.where(missing, np.nan, x), "empty": np.nan, "constant": 4.})
        fitted = AdditiveSplineRegressor(n_knots=8).fit(X, y)
        schema = fitted.features_.schema()
        self.assertEqual(set(schema["constant_numeric_features"]), {"empty", "constant"})
        self.assertEqual(sum(name.endswith("__missing") for name in schema["encoded_columns"]), 3)
        self.assertFalse(any(name.startswith(("empty__spline", "constant__spline")) for name in schema["encoded_columns"]))
        start, stop = fitted.features_.feature_slices_["climate"]
        transformed = fitted.features_.transform(X)
        np.testing.assert_array_equal(transformed[:, stop-1], missing.astype(float))
        self.assertGreater(stop-start, 1)

    def test_each_configuration_has_constant_boundary_extrapolation(self):
        X = pd.DataFrame({"climate": np.linspace(0, 10, 120), "constant": 7.})
        y = 20+np.sin(X.climate.to_numpy())
        test = pd.DataFrame({"climate": [-1e9, 0, 10, 1e9], "constant": [7, 900, -900, 7]})
        for name, model in build_candidates(requested=SPLINE_IDS).items():
            with self.subTest(candidate=name):
                fitted = clone(model).fit(X, y)
                pred = fitted.predict(test)
                np.testing.assert_allclose(pred[[0, 3]], pred[[1, 2]], rtol=0, atol=1e-12)

    def test_components_reconstruct_margin_and_pure_interactions_are_absent(self):
        rng = np.random.default_rng(3)
        X = pd.DataFrame({"x": rng.uniform(-1, 1, 300), "z": rng.uniform(-1, 1, 300)})
        y = 25+10*X.x*X.z
        fitted = AdditiveSplineRegressor().fit(X, y)
        corners = pd.DataFrame({"x": [-1., -1., 1., 1.], "z": [-1., 1., -1., 1.]})
        margin = fitted.predict_margin(corners)
        self.assertAlmostEqual(margin[0]-margin[1]-margin[2]+margin[3], 0., delta=1e-10)
        # The known product has a cross difference of 40, which an additive
        # margin cannot recover. This is a structural limitation, not a win gate.
        truth = 25+10*corners.x*corners.z
        self.assertAlmostEqual(truth.iloc[0]-truth.iloc[1]-truth.iloc[2]+truth.iloc[3], 40.)
        components = fitted.predict_components(corners)
        np.testing.assert_allclose(components["intercept"]+components["terms"].sum(axis=1), margin, atol=1e-12)
        np.testing.assert_allclose(fitted.predict(corners), np.maximum(0, margin), atol=1e-12)

    def test_shared_plan_audits_and_predictions_ignore_holdout_labels(self):
        table, _, _, plan, _, _ = records(profile="smoke")
        data = table.loc[table.material.eq("PCC")].copy()
        results = []
        for name in ("before", "after"):
            if name == "after":
                held = data.group_id.isin(plan.loc[plan.partition.eq("holdout"), "group_id"])
                data.loc[held, "level_smooth"] += 1000
            results.append(search_levels(data, FEATURES, "level_smooth", self.root/name, plan,
                                         CLIMATE_FEATURES, weight_column="delta_days", candidate_ids=SPLINE_IDS))
        self.assertEqual(results[0]["candidate"], results[1]["candidate"])
        self.assertEqual(results[0]["cv_mae"], results[1]["cv_mae"])
        self.assertEqual((self.root/"before/candidate_audit.json").read_bytes(), (self.root/"after/candidate_audit.json").read_bytes())
        a, b = (pd.read_csv(self.root/name/"holdout_predictions.csv") for name in ("before", "after"))
        np.testing.assert_array_equal(a.prediction, b.prediction)
        audit = json.loads((self.root/"before/candidate_audit.json").read_text())
        development = set(plan.loc[plan.partition.eq("development"), "group_id"])
        for entry in audit["attempts"]:
            if entry["family"] != "additive_spline": continue
            self.assertEqual(entry["candidate_status"], "completed")
            for fit in entry["cv_training"]:
                self.assertEqual(set(fit["refit_groups"]), development-set(plan.loc[plan.fold.eq(fit["fold"]), "group_id"]))
                self.assertEqual(fit["stopping_groups"], [])
                self.assertIsNone(fit["iterations"])

    def test_explicit_constant_and_missing_target_statuses_remain_unchanged(self):
        _, _, _, plan, _, edges = records(profile="smoke")
        for target, status in (("level_constant_zero", "constant_target"), ("level_constant_positive", "constant_target"),
                               ("level_positive_only_holdout", "constant_target"), ("level_all_missing", "no_valid_target"),
                               ("level_no_holdout_labels", "insufficient_groups")):
            summary = search_levels(edges, FEATURES, target, self.root/target, plan, CLIMATE_FEATURES,
                                    candidate_ids=SPLINE_IDS)
            self.assertEqual(summary["status"], status)
            self.assertFalse(summary["importance_reliable"])

    def test_invalid_weights_and_unsupported_spline_settings_fail(self):
        X, y = pd.DataFrame({"x": np.arange(10.)}), np.arange(10.)
        for kwargs in ({"n_knots": 1}, {"alpha": 0}, {"degree": 2}, {"extrapolation": "linear"}, {"solver": "sag"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(SafeError, "UNSUPPORTED_ADDITIVE_SPLINE"):
                AdditiveSplineRegressor(**kwargs).fit(X, y)
        for weights in (np.zeros(10), np.full(10, np.nan)):
            with self.assertRaisesRegex(SafeError, "INVALID_WEIGHTED_MODEL_INPUT"):
                AdditiveSplineRegressor().fit(X, y, sample_weight=weights)

    def test_frozen_calibration_scores_and_centered_oracle_shapes(self):
        for seed in CALIBRATION_SEEDS:
            data, _, _, plan, _, _ = records(seed=seed, profile="standard")
            data = data.loc[data.material.eq("PCC")]
            for target in ("level_linear", "level_smooth"):
                with self.subTest(seed=seed, target=target):
                    out = self.root/f"{seed}-{target}"
                    result = search_levels(data, FEATURES, target, out, plan, CLIMATE_FEATURES,
                                           seed=seed, weight_column="delta_days", candidate_ids=SPLINE_IDS)
                    leaders = pd.read_csv(out/"leaderboard.csv")
                    self.assertTrue(leaders.candidate_status.eq("completed").all())
                    all_features = leaders.loc[leaders.feature_set.eq("all")].set_index("candidate")
                    self.assertLessEqual(all_features.loc[SPLINE_IDS, "cv_mae"].min(), .8*all_features.loc["median", "cv_mae"])
                    self.assertEqual(result["candidate_failure_count"], 0)
            development = data.loc[data.group_id.isin(plan.loc[plan.partition.eq("development"), "group_id"])]
            reference = build_candidates(seed, requested=["spline_knots8_alpha0.1"])["spline_knots8_alpha0.1"]
            reference.fit(development[FEATURES], development.level_smooth, sample_weight=development.delta_days)
            grids = {"thermal_aging_index": (np.linspace(0, 12, 101), lambda x: 9*(1-np.cos(2*np.pi*x/12))),
                     "temperature_range_c": (np.linspace(15, 55, 101), lambda x: 12*((x-15)/40-.5)**2),
                     "wet_days": (np.linspace(0, 220, 101), lambda x: 5*(x/220)**2)}
            for name, (grid, oracle) in grids.items():
                with self.subTest(seed=seed, shape=name):
                    probe = pd.concat([development[FEATURES].iloc[[0]]]*len(grid), ignore_index=True)
                    probe[name] = grid
                    shape = reference.predict_components(probe)["terms"][name].to_numpy()
                    truth = oracle(grid)
                    error = np.sqrt(np.mean(((shape-shape.mean())-(truth-truth.mean()))**2))
                    self.assertLessEqual(error, 1.0)


if __name__ == "__main__":
    unittest.main()
