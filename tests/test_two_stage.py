"""Occurrence/severity admission, independent losses and shared evaluation."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression

from pavement_analytics.model_registry import build_candidates
from pavement_analytics.security import SafeError
from pavement_analytics.study_modeling import search_levels
from pavement_analytics.study_reporting import build_study_report
from pavement_analytics.synthetic_benchmark import CLIMATE_FEATURES, FEATURES, records
from pavement_analytics.two_stage import (OccurrenceSeverityRegressor, checked_components,
                                          distribution_metrics)
from temp_support import temporary_directory


TWO_STAGE_IDS = [f"two_stage_c{c}_leaf{leaf}" for c in (0.1, 1.0) for leaf in (2, 10)]
CALIBRATION_SEEDS = (20261011, 20261012, 20261013)
REFERENCE = "two_stage_c1.0_leaf10"


class ReversedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, invalid=False):
        self.invalid = invalid

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([1, 0])
        return self

    def predict_proba(self, X):
        return np.tile([1.2, -.2] if self.invalid else [.8, .2], (len(X), 1))


class TwoStageTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def model(self):
        return build_candidates(requested=[REFERENCE])[REFERENCE]

    def fit(self, y, weights=None, groups=None, model=None):
        y = np.asarray(y, float)
        X = pd.DataFrame({"x": np.arange(len(y), dtype=float)})
        groups = np.repeat([f"SYN{i:02d}" for i in range(6)], len(y)//6) if groups is None else groups
        return (self.model() if model is None else model).fit_grouped(X, y, sample_weight=weights, groups=groups), X

    def test_every_configuration_uses_both_native_weighted_losses(self):
        # Constant features force the classifier intercept and a single tree
        # leaf. Their independent weighted optima are 8/9 and 5, respectively.
        X = pd.DataFrame({"constant": np.ones(60)})
        y = np.repeat([0., 2., 6.], 20)
        weights = np.repeat([1., 2., 6.], 20)
        groups = np.tile(["SYN_A", "SYN_B", "SYN_C", "SYN_D"], 15)
        for name, model in build_candidates(requested=TWO_STAGE_IDS).items():
            with self.subTest(candidate=name):
                fitted = clone(model).fit_grouped(X, y, sample_weight=weights, groups=groups)
                values = fitted.predict_occurrence_severity(X)
                # Match the declared lbfgs gradient tolerance (1e-4).
                np.testing.assert_allclose(values["p_positive"], 8/9, atol=1e-4, rtol=0)
                np.testing.assert_allclose(values["positive_mean"], 5., atol=1e-12, rtol=0)
                np.testing.assert_allclose(values["prediction"], values["p_positive"]*5, atol=1e-12)

    def test_native_training_receives_all_binary_rows_and_only_positive_severity_rows(self):
        y = np.tile([0., 2., 3., 7.], 15)
        weights = np.arange(1, 61, dtype=float)
        X = pd.DataFrame({"x": np.arange(60.), "soil": np.where(y > 0, "POSITIVE", "ZERO_ONLY")})
        groups = np.tile(["SYN_A", "SYN_B", "SYN_C"], 20)
        captured = {}
        occurrence_fit, severity_fit = LogisticRegression.fit, ExtraTreesRegressor.fit

        def occurrence(estimator, X, y, sample_weight=None, **kwargs):
            captured["binary"], captured["all_weights"] = np.asarray(y).copy(), sample_weight.copy()
            return occurrence_fit(estimator, X, y, sample_weight=sample_weight, **kwargs)

        def severity(estimator, X, y, sample_weight=None, **kwargs):
            captured["positive"], captured["positive_weights"] = np.asarray(y).copy(), sample_weight.copy()
            return severity_fit(estimator, X, y, sample_weight=sample_weight, **kwargs)

        with patch.object(LogisticRegression, "fit", occurrence), patch.object(ExtraTreesRegressor, "fit", severity):
            fitted = self.model().fit_grouped(X, y, sample_weight=weights, groups=groups)
        np.testing.assert_array_equal(captured["binary"], y > 0)
        np.testing.assert_array_equal(captured["all_weights"], weights)
        np.testing.assert_array_equal(captured["positive"], y[y > 0])
        np.testing.assert_array_equal(captured["positive_weights"], weights[y > 0])
        occurrence = fitted.occurrence_estimator_.named_steps["prepare"].named_transformers_["categorical"].named_steps["encode"]
        severity = fitted.severity_estimator_.estimator_.named_steps["prepare"].named_transformers_["categorical"].named_steps["encode"]
        self.assertIn("ZERO_ONLY", occurrence.categories_[0])
        self.assertNotIn("ZERO_ONLY", severity.categories_[0])

    def test_every_configuration_clones_serializes_and_handles_missing_unseen_inputs(self):
        _, stress, _, plan, _, _ = records(profile="smoke")
        data = stress.loc[stress.material.eq("PCC")].dropna(subset=["level_zero_inflated"])
        held = data.group_id.isin(plan.loc[plan.partition.eq("holdout"), "group_id"])
        train, test = data.loc[~held], data.loc[held]
        features = [*FEATURES, "all_missing_predictor"]
        for name, model in build_candidates(requested=TWO_STAGE_IDS).items():
            with self.subTest(candidate=name):
                fitted = clone(model).fit_grouped(train[features], train.level_zero_inflated,
                                                 sample_weight=train.delta_days, groups=train.group_id)
                expected = checked_components(fitted.predict_occurrence_severity(test[features]), len(test))
                path = self.root/(name+".joblib")
                joblib.dump(fitted, path)
                actual = joblib.load(path).predict_occurrence_severity(test[features])
                for component in expected:
                    np.testing.assert_array_equal(actual[component], expected[component])
                encoder = fitted.occurrence_estimator_.named_steps["prepare"].named_transformers_["categorical"].named_steps["encode"]
                self.assertNotIn("HOLDOUT_ONLY", {x for categories in encoder.categories_ for x in categories})

    def test_zero_and_all_positive_training_have_explicit_fallbacks(self):
        for y, p, severity_status in ((np.zeros(60), 0., "no_positive_training"),
                                       (np.full(60, 7.), 1., "constant_positive_target"),
                                       (np.arange(60.)+1, 1., "fitted_positive_regressor")):
            with self.subTest(probability=p, severity=severity_status):
                fitted, X = self.fit(y)
                values = fitted.predict_occurrence_severity(X)
                np.testing.assert_array_equal(values["p_positive"], p)
                self.assertIsNone(fitted.occurrence_estimator_)
                self.assertEqual(fitted.training_details_["severity_status"], severity_status)
                if not y.any():
                    np.testing.assert_array_equal(values["prediction"], 0.)
                    self.assertIsNone(fitted.severity_estimator_)
                elif np.ptp(y) == 0:
                    np.testing.assert_array_equal(values["prediction"], 7.)

    def test_sparse_classes_and_positive_groups_use_training_weighted_fallbacks(self):
        groups = np.repeat([f"SYN{i}" for i in range(6)], 10)
        cases = []
        y = np.zeros(60); y[:4] = [1, 2, 4, 8]
        cases.append((y, groups, "sparse_class_weighted_prevalence", "sparse_positive_weighted_mean"))
        y = np.zeros(60); y[:12] = np.arange(12)+1
        one_group = groups.astype(object); one_group[:12] = "SYN_POS"
        cases.append((y, one_group, "sparse_class_weighted_prevalence", "sparse_positive_weighted_mean"))
        y = np.arange(60.)+1; y[:3] = 0
        cases.append((y, groups, "sparse_class_weighted_prevalence", "fitted_positive_regressor"))
        weights = np.arange(1, 61, dtype=float)
        for y, member_groups, occurrence, severity in cases:
            with self.subTest(occurrence=occurrence, severity=severity):
                fitted, X = self.fit(y, weights, member_groups)
                values = fitted.predict_occurrence_severity(X)
                np.testing.assert_array_equal(values["p_positive"], np.average(y > 0, weights=weights))
                self.assertEqual(fitted.training_details_["occurrence_status"], occurrence)
                self.assertEqual(fitted.training_details_["severity_status"], severity)
                if fitted.severity_estimator_ is None:
                    np.testing.assert_array_equal(values["positive_mean"], np.average(y[y > 0], weights=weights[y > 0]))

    def test_refitting_clears_learned_state_and_requires_valid_training_context(self):
        fitted, X = self.fit(np.tile([0., 2., 3.], 20))
        self.assertIsNotNone(fitted.occurrence_estimator_)
        self.assertIsNotNone(fitted.severity_estimator_)
        fitted.fit_grouped(X, np.zeros(60), groups=np.repeat(["SYN_A", "SYN_B"], 30))
        self.assertIsNone(fitted.occurrence_estimator_)
        self.assertIsNone(fitted.severity_estimator_)
        for kwargs, code in (({}, "MODEL_GROUP_CONTEXT_REQUIRED"),
                              ({"groups": [None]*60}, "MODEL_VALID_TRAINING_GROUPS_REQUIRED"),
                              ({"groups": ["SYN"]*60, "sample_weight": np.zeros(60)}, "INVALID_WEIGHTED_MODEL_INPUT")):
            with self.assertRaisesRegex(SafeError, code):
                self.model().fit(X, np.zeros(60), **kwargs)
        invalid = self.model().set_params(min_class_rows=0)
        with self.assertRaisesRegex(SafeError, "INVALID_TWO_STAGE_SUPPORT_POLICY"):
            invalid.fit(X, np.zeros(60), groups=["SYN"]*60)

    def test_positive_probability_uses_class_identity_and_rejects_invalid_components(self):
        for invalid in (False, True):
            model = self.model().set_params(occurrence_estimator=ReversedClassifier(invalid))
            fitted, X = self.fit(np.tile([0., 2., 3.], 20), model=model)
            if invalid:
                with self.assertRaisesRegex(SafeError, "INVALID_OCCURRENCE_CLASS_SCHEMA"):
                    fitted.predict(X)
            else:
                np.testing.assert_array_equal(fitted.predict_occurrence_severity(X)["p_positive"], .8)
        values = {"p_positive": np.array([.5]), "positive_mean": np.array([4.]), "prediction": np.array([2.]),
                  "baseline_probability": np.array([.25]), "baseline_positive_mean": np.array([4.]),
                  "baseline_mean_prediction": np.array([1.])}
        for key in values:
            broken = {name: value.copy() for name, value in values.items()}
            broken[key][0] = np.nan if key != "prediction" else 3.
            with self.subTest(component=key), self.assertRaises(SafeError):
                checked_components(broken, 1)

    def test_independent_mixture_mean_and_median_optimize_different_losses(self):
        # Midpoint integration of the fixture's uniform positive distribution;
        # this does not use the fitted implementation or a holdout selection.
        severity = 20.
        positive = np.linspace(.6*severity, 1.4*severity, 20000, endpoint=False)+.8*severity/40000
        for p in (.25, .75):
            mean = p*severity
            median = 0. if p <= .5 else severity*(1.4-.4/p)
            mae = lambda x: (1-p)*abs(x)+p*np.mean(np.abs(positive-x))
            mse = lambda x: (1-p)*x*x+p*np.mean((positive-x)**2)
            self.assertLess(mse(mean), mse(median))
            self.assertLess(mae(median), mae(mean))
            self.assertLess(mse(mean), mse(mean+1.))
            self.assertLess(mae(median), mae(median+1.))

    def test_component_metrics_match_independent_weighted_arithmetic(self):
        frame = pd.DataFrame({"actual": [0., 0., 2., 6.], "p_positive": [.1, .8, .4, .9],
                              "positive_mean": [3., 3., 3., 5.], "baseline_probability": [.5]*4,
                              "baseline_positive_mean": [4.]*4, "baseline_mean_prediction": [2.]*4,
                              "weight": [1., 2., 3., 4.], "group_id": ["SYN_A", "SYN_A", "SYN_B", "SYN_C"]})
        frame["prediction"] = frame.p_positive*frame.positive_mean
        metrics = distribution_metrics(frame)
        observed, p, weights = np.array([0., 0., 1., 1.]), frame.p_positive.to_numpy(), frame.weight.to_numpy()
        self.assertAlmostEqual(metrics["occurrence_brier"], np.sum(weights*(observed-p)**2)/10)
        self.assertAlmostEqual(metrics["occurrence_log_loss"], -np.sum(weights*(observed*np.log(p)+(1-observed)*np.log(1-p)))/10)
        self.assertAlmostEqual(metrics["positive_severity_mae"], 1.)
        self.assertAlmostEqual(metrics["positive_severity_rmse"], 1.)
        self.assertAlmostEqual(metrics["mean_prediction_rmse"], np.sqrt(np.average((frame.actual-frame.prediction)**2, weights=weights)))
        self.assertNotAlmostEqual(metrics["pr_auc_trapezoid"], metrics["average_precision"])
        for value in (0., 7.):
            frame["actual"] = value
            metrics = distribution_metrics(frame)
            self.assertEqual(metrics["ranking_status"], "single_class_evaluation")
            self.assertIsNone(metrics["pr_auc_trapezoid"])
            self.assertIsNone(metrics["average_precision"])
            if value == 0:
                self.assertIsNone(metrics["positive_severity_rmse"])
                self.assertEqual(metrics["positive_support_status"], "no_positive_evaluation_rows")

    def test_shared_folds_diagnostics_serialization_and_selection_ignore_holdout_labels(self):
        table, _, _, plan, _, _ = records(profile="smoke")
        data = table.loc[table.material.eq("PCC")].copy()
        results = []
        for name in ("before", "after"):
            if name == "after":
                held = data.group_id.isin(plan.loc[plan.partition.eq("holdout"), "group_id"])
                data.loc[held, "level_zero_inflated"] += 1000
            results.append(search_levels(data, FEATURES, "level_zero_inflated", self.root/name, plan,
                                         CLIMATE_FEATURES, weight_column="delta_days", candidate_ids=TWO_STAGE_IDS))
        for key in ("candidate", "nonclimate_candidate", "cv_mae", "candidate_parameters"):
            self.assertEqual(results[0][key], results[1][key])
        self.assertEqual((self.root/"before/candidate_audit.json").read_bytes(), (self.root/"after/candidate_audit.json").read_bytes())
        before, after = (pd.read_csv(self.root/name/"two_stage_predictions.csv") for name in ("before", "after"))
        for key in ("candidate", "feature_set", "row_id", "group_id", "p_positive", "positive_mean", "prediction"):
            np.testing.assert_array_equal(before[key], after[key])
        oof = before.loc[before.partition.eq("development_oof")].merge(plan, on="group_id", suffixes=("", "_plan"))
        self.assertTrue(oof.partition_plan.eq("development").all())
        self.assertTrue(oof.fold.eq(oof.fold_plan).all())
        self.assertEqual(set(oof.candidate), set(TWO_STAGE_IDS))
        leaders = pd.read_csv(self.root/"before/leaderboard.csv")
        audit = json.loads((self.root/"before/candidate_audit.json").read_text())
        for attempt in audit["attempts"]:
            if attempt["family"] != "two_stage": continue
            for fit in attempt["cv_training"]:
                expected = set(plan.loc[plan.partition.eq("development") & plan.fold.ne(fit["fold"]), "group_id"])
                self.assertEqual(set(fit["refit_groups"]), expected)
                self.assertLessEqual(set(fit["severity_training_groups"]), expected)
                self.assertEqual(fit["stopping_groups"], [])
        for feature_set in ("all", "nonclimate"):
            eligible = leaders.loc[leaders.feature_set.eq(feature_set) & leaders.family.eq("two_stage")]
            best = eligible.sort_values(["cv_mae", "candidate"]).iloc[0]
            saved = before.loc[before.partition.eq("holdout") & before.feature_set.eq(feature_set)]
            self.assertEqual(set(saved.candidate), {best.candidate})
            self.assertTrue(best.two_stage_diagnostic)
            artifact = joblib.load(self.root/"before"/("two_stage_"+feature_set+"_model.joblib"))
            X = data.set_index("row_id").loc[saved.row_id, artifact["features"]]
            components = artifact["estimator"].predict_occurrence_severity(X)
            for key in ("p_positive", "positive_mean", "prediction"):
                np.testing.assert_allclose(components[key], saved[key], rtol=1e-12, atol=1e-12)
            self.assertEqual(artifact["selection_metric"], "mean_development_cv_MAE")
        self.assertEqual(results[0]["candidate_failure_count"], 0)

    def test_partial_component_failure_is_visible_and_cannot_win(self):
        data, _, _, plan, _, _ = records(profile="smoke")
        original = OccurrenceSeverityRegressor.predict_occurrence_severity

        def invalid(model, X):
            values = original(model, X)
            # Invalidate one training fold (two hook calls per prediction),
            # leaving earlier successful rows as a visible partial CV attempt.
            if set(model.training_details_["fitting_groups"]) == trigger:
                values["p_positive"] = np.full(len(X), np.nan)
            return values

        development = plan.loc[plan.partition.eq("development")]
        trigger = set(development.loc[development.fold.ne(1), "group_id"])
        with patch.object(OccurrenceSeverityRegressor, "predict_occurrence_severity", invalid):
            result = search_levels(data, FEATURES, "level_zero_inflated", self.root/"partial", plan,
                                   CLIMATE_FEATURES, candidate_ids=[REFERENCE])
        self.assertEqual(result["candidate"], "median")
        leaders = pd.read_csv(self.root/"partial/leaderboard.csv")
        failed = leaders.loc[leaders.candidate.eq(REFERENCE)]
        self.assertTrue(failed.candidate_status.eq("failed_predict").all())
        self.assertTrue(failed.cv_mae.isna().all())
        self.assertTrue(failed.folds_completed.lt(3).all())
        self.assertTrue(failed.folds_completed.eq(1).all())
        self.assertEqual(result["candidate_failure_count"], 2)

    def test_global_constant_and_missing_tasks_retain_existing_shortcuts(self):
        _, _, _, plan, _, edges = records(profile="smoke")
        for target, status in (("level_constant_zero", "constant_target"), ("level_constant_positive", "constant_target"),
                               ("level_positive_only_holdout", "constant_target"), ("level_all_missing", "no_valid_target"),
                               ("level_no_holdout_labels", "insufficient_groups")):
            result = search_levels(edges, FEATURES, target, self.root/target, plan, CLIMATE_FEATURES,
                                   candidate_ids=TWO_STAGE_IDS)
            self.assertEqual(result["status"], status)
            self.assertTrue(pd.read_csv(self.root/target/"two_stage_predictions.csv").empty)
            self.assertTrue(pd.read_csv(self.root/target/"two_stage_metrics.csv").empty)

    def test_diagnostic_refit_failures_preserve_primary_and_private_error_boundary(self):
        data, _, _, plan, _, _ = records(profile="smoke")
        data = data.loc[data.material.eq("PCC")].copy()
        data["level_zero_inflated"] = np.where(np.arange(len(data))%10 == 0, 10., 0.)
        development_rows = int(data.group_id.isin(plan.loc[plan.partition.eq("development"), "group_id"]).sum())
        original = OccurrenceSeverityRegressor.fit_grouped

        def fail_refit(model, X, y, sample_weight=None, groups=None):
            if len(X) == development_rows:
                raise RuntimeError("generated-private-model-detail")
            return original(model, X, y, sample_weight=sample_weight, groups=groups)

        path = self.root/"diagnostic_failure"
        with patch.object(OccurrenceSeverityRegressor, "fit_grouped", fail_refit):
            result = search_levels(data, FEATURES, "level_zero_inflated", path, plan, CLIMATE_FEATURES,
                                   candidate_ids=TWO_STAGE_IDS)
        self.assertEqual(result["candidate"], "median")
        self.assertEqual(result["candidate_failure_count"], 8)
        metrics = pd.read_csv(path/"two_stage_metrics.csv")
        self.assertEqual(set(metrics.partition), {"development_oof"})
        self.assertTrue(metrics.complete_cv.all())
        self.assertTrue(metrics.candidate_status.eq("failed_diagnostic_refit").all())
        self.assertFalse((path/"two_stage_all_model.joblib").exists())
        self.assertEqual(joblib.load(path/"model.joblib")["candidate"], "median")
        self.assertIn("generated-private-model-detail", (path/"candidate_errors.log").read_text())
        self.assertNotIn("generated-private-model-detail", (path/"candidate_audit.json").read_text())

    def test_consolidation_retains_component_rows_and_explains_mean_semantics(self):
        data, _, _, plan, _, _ = records(profile="smoke")
        data = data.loc[data.material.eq("PCC")]
        target = "level_zero_inflated"
        path = self.root/"materials/PCC"/target
        summary = search_levels(data, FEATURES, target, path, plan, CLIMATE_FEATURES, candidate_ids=[REFERENCE])
        build_study_report(self.root, [{"material": "PCC", **summary}], {})
        actual = pd.read_csv(self.root/"two_stage_predictions.csv")
        expected = pd.read_csv(path/"two_stage_predictions.csv")
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(set(actual.material), {"PCC"})
        self.assertEqual(set(actual.target), {target})
        self.assertEqual(len(pd.read_csv(self.root/"two_stage_metrics.csv")), 4)
        html = (self.root/"dashboard.html").read_text()
        self.assertIn("Two-stage conditional-mean diagnostics", html)
        self.assertIn("It need not minimize MAE", html)
        self.assertNotIn("SYN_G", html)

    def test_frozen_calibration_recovers_occurrence_and_overall_mean(self):
        for seed in CALIBRATION_SEEDS:
            with self.subTest(seed=seed):
                data, _, oracle, plan, _, _ = records(seed=seed, profile="standard")
                data = data.loc[data.material.eq("PCC")]
                path = self.root/str(seed)
                result = search_levels(data, FEATURES, "level_zero_inflated", path, plan, CLIMATE_FEATURES,
                                       seed=seed, weight_column="delta_days", candidate_ids=TWO_STAGE_IDS)
                metrics = pd.read_csv(path/"two_stage_metrics.csv")
                reference = metrics.loc[metrics.candidate.eq(REFERENCE) & metrics.feature_set.eq("all")
                                        & metrics.partition.eq("development_oof")].iloc[0]
                self.assertGreaterEqual(reference.brier_skill, .01)
                self.assertGreaterEqual(reference.mean_rmse_skill, .01)
                rows = pd.read_csv(path/"two_stage_predictions.csv")
                rows = rows.loc[rows.candidate.eq(REFERENCE) & rows.feature_set.eq("all") & rows.partition.eq("development_oof")]
                truth = oracle.set_index("row_id").loc[rows.row_id]
                probability_rmse = np.sqrt(np.average((rows.p_positive.to_numpy()-truth.zero__p_positive.to_numpy())**2, weights=rows.weight))
                mean_rmse = np.sqrt(np.average((rows.prediction.to_numpy()-truth.level_zero_inflated__conditional_mean.to_numpy())**2, weights=rows.weight))
                self.assertLessEqual(probability_rmse, .15)
                self.assertLessEqual(mean_rmse, 3.5)
                self.assertEqual(result["candidate_failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
