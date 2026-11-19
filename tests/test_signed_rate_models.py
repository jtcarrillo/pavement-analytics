"""Signed PCI-rate selection, baseline semantics and unchanged deduct bounds."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression

from pavement_analytics.model_registry import (
    baseline_metadata, build_candidates, candidate_catalog, candidate_specs,
)
from pavement_analytics.security import SafeError
from pavement_analytics.study_modeling import (
    MeanRegressor, _checked_prediction, _estimator, make_partitions, search_levels,
)
from temp_support import temporary_directory


class SignedConstantRegressor(RegressorMixin, BaseEstimator):
    target_kind = "signed_pci_rate"

    def __init__(self, value=-1.):
        self.value = value

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def signed_records():
    count = 120
    temperature = np.tile(np.linspace(0., 9., 8), 15)
    return pd.DataFrame({
        "row_id": np.arange(count),
        "group_id": np.repeat([f"signed-{i:02d}" for i in range(15)], 8),
        "temperature": temperature,
        "age": np.tile([4., 10., 15., 7., 3., 6., 20., 8.], 15),
        "delta_days": np.tile([100., 400., 600., 300., 150., 800., 500., 350.], 15),
        "DetRatePerYear": -2.-3.*temperature,
    })


class SignedRateTests(unittest.TestCase):
    def setUp(self):
        temp = temporary_directory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.frame = signed_records()
        self.plan = make_partitions(self.frame, folds=3)

    def search(self, name="result", frame=None, **kwargs):
        return search_levels(
            self.frame if frame is None else frame, ["temperature", "age"],
            "DetRatePerYear", self.root/name, self.plan, ["temperature"],
            weight_column="delta_days", target_kind="signed_pci_rate", **kwargs)

    def test_signed_catalog_uses_mean_and_marks_occurrence_models_ineligible(self):
        self.assertEqual(baseline_metadata("signed_pci_rate")["candidate"], "mean")
        self.assertEqual(candidate_catalog(target_kind="signed_pci_rate")[0]["family"], "weighted_mean")
        specs = candidate_specs(target_kind="signed_pci_rate")
        for spec in specs:
            if spec.family == "two_stage":
                self.assertNotIn("signed_pci_rate", spec.supported_target_kinds)
            else:
                self.assertEqual(spec.parameters["target_kind"], "signed_pci_rate")
            if spec.family == "xgboost":
                self.assertEqual(spec.parameters["eval_metric"], "rmse")
            if spec.family == "lightgbm":
                self.assertEqual(spec.parameters["metric"], "rmse")
        pool = build_candidates(requested=["enet_0.01", "two_stage_c0.1_leaf2"],
                                target_kind="signed_pci_rate")
        self.assertEqual(list(pool), ["enet_0.01"])
        self.assertEqual(pool.entries[-1]["candidate_status"], "ineligible_target")
        self.assertEqual(pool.entries[-1]["error_code"], "MODEL_TARGET_KIND_UNSUPPORTED")

    def test_shared_candidate_profile_retains_deduct_two_stage_and_rate_audit(self):
        requested = ["enet_0.01", "two_stage_c0.1_leaf2"]
        deduct = build_candidates(requested=requested)
        self.assertEqual(set(deduct), set(requested))
        result = self.search(candidate_ids=requested)
        self.assertEqual(result["candidate_ineligible_count"], 2)
        self.assertEqual(result["candidate_failure_count"], 0)
        board = pd.read_csv(self.root/"result/leaderboard.csv")
        self.assertTrue(board.loc[board.family.eq("two_stage"), "candidate_status"].eq("ineligible_target").all())
        self.assertTrue(pd.read_csv(self.root/"result/two_stage_metrics.csv").empty)

    def test_only_ineligible_requested_candidates_leave_weighted_mean_baseline(self):
        result = self.search(candidate_ids=["two_stage_c0.1_leaf2"])
        self.assertEqual(result["candidate"], "mean")
        self.assertEqual(result["status"], "baseline_selected")
        self.assertEqual(result["candidate_ineligible_count"], 2)
        self.assertEqual(result["rmse_skill"], 0.)

    def test_base_regressors_keep_signed_values_and_survive_reload(self):
        X = self.frame[["temperature", "age"]]
        requested = ["enet_0.01", "rf_5", "et_leaf2_depth12"]
        for name, estimator in build_candidates(requested=requested, target_kind="signed_pci_rate").items():
            with self.subTest(candidate=name):
                fitted = clone(estimator).fit(X, self.frame.DetRatePerYear,
                                               sample_weight=self.frame.delta_days)
                prediction = _checked_prediction(fitted, X)
                self.assertTrue((prediction < 0).all())
                artifact = self.root/(name+".joblib")
                joblib.dump(fitted, artifact)
                np.testing.assert_array_equal(joblib.load(artifact).predict(X), prediction)

    def test_rate_selection_uses_rmse_even_when_mae_prefers_another_constant(self):
        frame = self.frame.copy()
        frame["delta_days"] = 365.
        frame["DetRatePerYear"] = np.tile([-1., -1., -1., -9.], 30)
        with patch("pavement_analytics.study_modeling.candidates",
                   return_value={"mae_favorite": SignedConstantRegressor(-1.)}):
            result = self.search(frame=frame)
        self.assertEqual(result["candidate"], "mean")
        self.assertEqual(result["selection_metric"], "mean_interval_weighted_cv_RMSE")
        leaders = pd.read_csv(self.root/"result/leaderboard.csv")
        all_models = leaders.loc[leaders.feature_set.eq("all")].set_index("candidate")
        self.assertLess(all_models.loc["mae_favorite", "cv_mae"], all_models.loc["mean", "cv_mae"])
        self.assertGreater(all_models.loc["mae_favorite", "cv_rmse"], all_models.loc["mean", "cv_rmse"])
        self.assertAlmostEqual(result["baseline_mae"], 3.)
        self.assertAlmostEqual(result["baseline_rmse"], np.sqrt(12.))

    def test_signed_search_has_weighted_baseline_rmse_importance_and_bundle_contract(self):
        result = self.search(candidate_ids=["enet_0.01"])
        development = self.frame.loc[self.frame.group_id.isin(
            self.plan.loc[self.plan.partition.eq("development"), "group_id"])]
        expected_mean = np.average(development.DetRatePerYear, weights=development.delta_days)
        predictions = pd.read_csv(self.root/"result/holdout_predictions.csv")
        np.testing.assert_allclose(predictions.baseline_prediction, expected_mean)
        self.assertTrue((predictions.prediction < 0).all())
        self.assertEqual(result["nonzero_fraction"], 1.)
        self.assertGreater(result["rmse_skill"], .99)
        self.assertEqual(result["importance_metric"], "rmse")
        importance = pd.read_csv(self.root/"result/importance.csv")
        self.assertTrue(importance.metric.eq("rmse").all())
        self.assertEqual(importance.loc[importance.kind.eq("feature")].sort_values(
            "importance_mean", ascending=False).feature.iloc[0], "temperature")
        bundle = joblib.load(self.root/"result/model.joblib")
        audit = json.loads((self.root/"result/candidate_audit.json").read_text())
        for record in (result, bundle, audit):
            self.assertEqual(record["target_kind"], "signed_pci_rate")
            self.assertEqual(record["baseline_candidate"], "mean")
            self.assertEqual(record["selection_metric"], "mean_interval_weighted_cv_RMSE")
        self.assertIsInstance(bundle["baseline_estimator"], MeanRegressor)

    def test_holdout_label_changes_do_not_change_rate_selection_or_predictions(self):
        before = self.search("before", candidate_ids=["enet_0.01"])
        changed = self.frame.copy()
        held = changed.group_id.isin(self.plan.loc[self.plan.partition.eq("holdout"), "group_id"])
        changed.loc[held, "DetRatePerYear"] += 1000.
        after = self.search("after", frame=changed, candidate_ids=["enet_0.01"])
        self.assertEqual(before["candidate"], after["candidate"])
        self.assertEqual(before["cv_rmse"], after["cv_rmse"])
        a = pd.read_csv(self.root/"before/holdout_predictions.csv")
        b = pd.read_csv(self.root/"after/holdout_predictions.csv")
        np.testing.assert_allclose(a.prediction, b.prediction)
        np.testing.assert_allclose(a.baseline_prediction, b.baseline_prediction)

    def test_negative_constant_target_and_empty_target_preserve_signed_semantics(self):
        frame = self.frame.copy()
        frame["DetRatePerYear"] = -4.5
        result = self.search("constant", frame=frame, candidate_ids=["enet_0.01"])
        self.assertEqual(result["candidate"], "mean")
        self.assertEqual(result["status"], "constant_target")
        self.assertEqual(result["holdout_rmse"], 0.)
        self.assertIsNone(result["rmse_skill"])
        self.assertEqual(result["nonzero_fraction"], 1.)
        frame["DetRatePerYear"] = np.nan
        empty = self.search("empty", frame=frame, candidate_ids=["enet_0.01"])
        self.assertEqual(empty["status"], "no_valid_target")
        self.assertEqual(empty["target_kind"], "signed_pci_rate")

    def test_default_deduct_target_still_rejects_negative_values(self):
        with self.assertRaisesRegex(SafeError, "NEGATIVE_DEDUCT_TARGET"):
            search_levels(self.frame, ["temperature"], "DetRatePerYear", self.root/"deduct",
                          self.plan, ["temperature"], candidate_ids=["enet_0.01"])
        X = pd.DataFrame({"x": [0., 1., 2.]})
        fitted = _estimator(LinearRegression()).fit(X, [-2., -3., -4.])
        np.testing.assert_array_equal(fitted.predict(X), np.zeros(3))
        del fitted.target_kind
        np.testing.assert_array_equal(fitted.predict(X), np.zeros(3))

    def test_target_kind_typos_fail_before_creating_output(self):
        with self.assertRaisesRegex(SafeError, "INVALID_MODEL_TARGET_KIND"):
            search_levels(self.frame, ["temperature"], "DetRatePerYear", self.root/"invalid",
                          self.plan, ["temperature"], target_kind="signed_rate")
        self.assertFalse((self.root/"invalid").exists())
        with self.assertRaisesRegex(SafeError, "INVALID_MODEL_TARGET_KIND"):
            build_candidates(target_kind=None)

    def test_pci_change_outcomes_cannot_be_predictors_for_either_target_kind(self):
        frame = self.frame.assign(delta_PCI=-10., observed=3.)
        for kind in ("deduct_total", "signed_pci_rate"):
            for feature in ("delta_PCI", "DetRatePerYear"):
                with self.subTest(target_kind=kind, feature=feature):
                    out = self.root/(kind+"_"+feature)
                    with self.assertRaisesRegex(SafeError, "UNSAFE_MODEL_FEATURE"):
                        search_levels(frame, ["temperature", feature], "observed", out,
                                      self.plan, ["temperature"], target_kind=kind)
                    self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
