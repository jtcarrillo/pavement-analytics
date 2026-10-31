"""Optional boosting admission, nested stopping boundaries and provenance."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from pavement_analytics.boosting import GroupedXGBoostRegressor, grouped_stopping_split, require_xgboost
from pavement_analytics.model_registry import build_candidates, dependency_version
from pavement_analytics.original.common import Run
from pavement_analytics.original.pipeline import DEFAULTS, discover
from pavement_analytics.security import SafeError
from pavement_analytics.study_modeling import search_levels
from pavement_analytics.study_reporting import build_study_report
from pavement_analytics.synthetic_benchmark import CLIMATE_FEATURES, FEATURES, records
from temp_support import temporary_directory


FIXED_IDS = [f"xgb_depth{depth}_lr{rate}" for depth in (3, 6) for rate in (0.05, 0.1)]
STOPPED_IDS = [name+"_es" for name in FIXED_IDS]
XGB_IDS = [*FIXED_IDS, *STOPPED_IDS]
SUPPORTED = importlib.util.find_spec("xgboost") is not None and dependency_version("xgboost", "3.1.3", "3.2")[1]


class OptionalBoostingTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_absent_dependency_preserves_base_search_and_is_explicitly_visible(self):
        original = importlib.util.find_spec
        def available(name, *args, **kwargs):
            return None if name in {"xgboost", "lightgbm"} else original(name, *args, **kwargs)
        frame, _, _, plan, _, _ = records(profile="smoke")
        frame = frame.loc[frame.material.eq("PCC")]
        with patch("pavement_analytics.model_registry.importlib.util.find_spec", side_effect=available):
            pool = build_candidates()
            self.assertEqual(len(pool), 12)
            entries = [r for r in pool.entries if r["family"] == "xgboost"]
            self.assertEqual(len(entries), 8)
            self.assertTrue(all(r["candidate_status"] == "unavailable" for r in entries))
            with self.assertRaisesRegex(SafeError, "MODEL_DEPENDENCY_UNAVAILABLE"):
                search_levels(frame, FEATURES, "level_linear", self.root/"required", plan,
                              CLIMATE_FEATURES, candidate_ids=[FIXED_IDS[0]])
            self.assertFalse((self.root/"required").exists())
            result = search_levels(frame, FEATURES, "level_linear", self.root/"base", plan,
                                   CLIMATE_FEATURES, weight_column="delta_days")
        self.assertEqual(result["candidate_failure_count"], 0)
        self.assertEqual(result["candidate_unavailable_count"], 32)
        self.assertNotEqual(result["candidate_family"], "xgboost")

    def test_unsupported_version_has_a_distinct_status_and_required_preflight_error(self):
        original = __import__("importlib.metadata", fromlist=["version"]).version
        with patch("pavement_analytics.model_registry.importlib.util.find_spec", return_value=object()):
            for version in ("3.1.2", "3.2.0", "3.1.3rc1", "not-a-version"):
                with self.subTest(version=version), patch("pavement_analytics.model_registry.importlib.metadata.version",
                     side_effect=lambda name: version if name == "xgboost" else original(name)):
                    pool = build_candidates()
                    entries = [r for r in pool.entries if r["family"] == "xgboost"]
                    self.assertTrue(all(r["candidate_status"] == "unsupported" for r in entries))
                    with self.assertRaisesRegex(SafeError, "MODEL_DEPENDENCY_VERSION_UNSUPPORTED"):
                        build_candidates(requested=[FIXED_IDS[0]])
                    with self.assertRaisesRegex(SafeError, "MODEL_DEPENDENCY_VERSION_UNSUPPORTED"):
                        require_xgboost()

    def test_stopping_split_uses_whole_groups_and_is_invariant_to_row_order(self):
        groups = np.repeat([f"SYN{i:03d}" for i in range(12)], np.arange(1, 13))
        fit, stop = grouped_stopping_split(groups, .2, 42)
        self.assertFalse(set(groups[fit]) & set(groups[stop]))
        order = np.random.default_rng(5).permutation(len(groups))
        _, other = grouped_stopping_split(groups[order], .2, 42)
        self.assertEqual(set(groups[stop]), set(groups[order][other]))
        for invalid in (["one"]*3, [None, "a"], ["", "a"]):
            with self.assertRaises(SafeError):
                grouped_stopping_split(invalid, .2, 42)

    def test_optional_dependency_change_invalidates_full_pipeline_checkpoint(self):
        raw = self.root/"input"
        raw.mkdir()
        (raw/"generated.csv").write_text("value\n1\n")
        cfg = {**DEFAULTS, "input_dir": str(raw), "output_dir": str(self.root/"output"),
               "cache_dir": str(self.root/"cache"), "skip_climate": True, "skip_pci": True}
        with patch("pavement_analytics.original.pipeline.environment_versions", return_value={"xgboost": None}):
            first = discover(cfg)[3]
        with patch("pavement_analytics.original.pipeline.environment_versions", return_value={"xgboost": "3.1.3"}):
            second = discover(cfg)[3]
        self.assertNotEqual(first, second)
        Run(cfg["output_dir"], first, cfg, [])
        with self.assertRaisesRegex(ValueError, "changed"):
            Run(cfg["output_dir"], second, cfg, [], resume=True)


@unittest.skipUnless(SUPPORTED, "Install the supported xgboost extra for backend admission tests")
class XGBoostTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_every_configuration_forwards_weights_to_the_final_squared_error_loss(self):
        X = pd.DataFrame({"constant": np.ones(48)})
        y = np.tile([0., 10.], 24)
        weights = np.where(y > 0, 9., 1.)
        groups = np.repeat([f"SYN{i}" for i in range(12)], 4)
        for name, model in build_candidates(requested=XGB_IDS).items():
            with self.subTest(candidate=name):
                fitted = clone(model).fit_grouped(X, y, sample_weight=weights, groups=groups)
                uniform = clone(model).fit_grouped(X, y, sample_weight=np.ones(48), groups=groups)
                np.testing.assert_allclose(fitted.predict(X), 9., atol=1e-3, rtol=0)
                np.testing.assert_allclose(uniform.predict(X), 5., atol=1e-3, rtol=0)

    def test_every_configuration_handles_stress_clones_serializes_and_repeats_deterministically(self):
        _, stress, _, plan, _, _ = records(profile="smoke")
        frame = stress.loc[stress.material.eq("PCC")].dropna(subset=["level_linear"])
        hold = frame.group_id.isin(plan.loc[plan.partition.eq("holdout"), "group_id"])
        train, test = frame.loc[~hold], frame.loc[hold]
        columns = [*FEATURES, "all_missing_predictor"]
        for name, model in build_candidates(requested=XGB_IDS).items():
            with self.subTest(candidate=name):
                fitted = clone(model).fit_grouped(train[columns], train.level_linear,
                                                 sample_weight=train.delta_days, groups=train.group_id)
                prediction = fitted.predict(test[columns])
                self.assertTrue(np.isfinite(prediction).all() and (prediction >= 0).all())
                path = self.root/(name+".joblib")
                joblib.dump(fitted, path)
                np.testing.assert_allclose(joblib.load(path).predict(test[columns]), prediction, rtol=0, atol=1e-12)
                repeated = clone(model).fit_grouped(train[columns], train.level_linear,
                                                   sample_weight=train.delta_days, groups=train.group_id)
                np.testing.assert_array_equal(repeated.predict(test[columns]), prediction)
                self.assertEqual(repeated.training_details_, fitted.training_details_)
                native = fitted.estimator_.estimator_.named_steps["model"]
                self.assertEqual(native.get_booster().num_boosted_rounds(), fitted.training_details_["iterations"])
                self.assertEqual((native.n_jobs, native.device, native.tree_method), (1, "cpu", "hist"))

    def test_probe_preprocessing_and_evaluation_weights_match_the_inner_training_split(self):
        backend = require_xgboost()
        groups = np.repeat([f"SYN{i:03d}" for i in range(12)], 4)
        fit, stop = grouped_stopping_split(groups, .2, 42)
        X = pd.DataFrame({"climate": np.arange(48, dtype=float), "soil": np.where(stop, "STOP_ONLY", "FIT_ONLY")})
        X.loc[[0, 4, 12], "climate"] = np.nan
        y, weights = np.arange(48, dtype=float)+3, np.arange(48, dtype=float)+1
        model = build_candidates(requested=[STOPPED_IDS[0]])[STOPPED_IDS[0]]
        original_fit, calls = backend.fit, []
        def capture(estimator, data, labels, **kwargs):
            record = {"X": np.asarray(data).copy(), "y": np.asarray(labels).copy(), **kwargs}
            calls.append(record)
            result = original_fit(estimator, data, labels, **kwargs)
            if "eval_set" in kwargs:
                matrix, actual = kwargs["eval_set"][0]
                record["best_mae"] = float(np.average(np.abs(actual-estimator.predict(matrix)),
                                                         weights=kwargs["sample_weight_eval_set"][0]))
            return result
        with patch.object(backend, "fit", capture):
            fitted = clone(model).fit_grouped(X, y, sample_weight=weights, groups=groups)
        self.assertEqual(len(calls), 2)
        probe, final = calls
        prep = fitted.stopping_preprocessor_
        np.testing.assert_allclose(probe["X"], prep.transform(X.loc[fit]), rtol=0, atol=0)
        np.testing.assert_allclose(probe["eval_set"][0][0], prep.transform(X.loc[stop]), rtol=0, atol=0)
        np.testing.assert_array_equal(probe["sample_weight"], weights[fit])
        np.testing.assert_array_equal(probe["sample_weight_eval_set"][0], weights[stop])
        np.testing.assert_array_equal(probe["eval_set"][0][1], y[stop])
        self.assertEqual(probe["X"].shape[1], probe["eval_set"][0][0].shape[1])
        encoder = prep.named_steps["prepare"].named_transformers_["categorical"].named_steps["encode"]
        self.assertEqual(encoder.categories_[0].tolist(), ["FIT_ONLY"])
        imputer = prep.named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"]
        self.assertAlmostEqual(imputer.statistics_[0], float(X.loc[fit, "climate"].median()))
        best = fitted.training_details_["iterations"]-1
        self.assertAlmostEqual(fitted.stopping_evaluations_["validation_0"]["mae"][best], probe["best_mae"], delta=1e-5)
        self.assertNotIn("eval_set", final)
        np.testing.assert_array_equal(final["sample_weight"], weights)
        self.assertEqual(len(final["y"]), len(X))

    def test_stopping_can_shorten_the_model_and_missing_group_context_fails(self):
        model = build_candidates(requested=[STOPPED_IDS[0]])[STOPPED_IDS[0]]
        X = pd.DataFrame({"constant": np.ones(48)})
        y = np.tile([2., 8.], 24)
        with self.assertRaisesRegex(SafeError, "GROUP_CONTEXT"):
            clone(model).fit(X, y)
        with self.assertRaisesRegex(SafeError, "STOPPING_GROUPS"):
            clone(model).fit_grouped(X, y, groups=["one"]*48)
        fitted = clone(model).fit_grouped(X, y, groups=np.repeat(np.arange(12), 4))
        self.assertEqual(fitted.training_details_["iterations"], 1)
        self.assertLess(fitted.training_details_["probe_iterations"], 300)
        self.assertEqual(fitted.estimator_.estimator_.named_steps["model"].n_estimators, 1)
        for weights in (np.zeros(48), np.full(48, np.nan)):
            with self.assertRaisesRegex(SafeError, "INVALID_WEIGHTED_MODEL_INPUT"):
                clone(model).fit_grouped(X, y, sample_weight=weights, groups=np.repeat(np.arange(12), 4))

    def test_outer_and_final_audits_exclude_evaluation_groups_and_holdout_labels(self):
        frame, _, _, plan, _, _ = records(profile="smoke")
        frame = frame.loc[frame.material.eq("PCC")].copy()
        chosen = [FIXED_IDS[0], STOPPED_IDS[0]]
        summaries = []
        for name in ("first", "altered"):
            if name == "altered":
                frame.loc[frame.group_id.isin(plan.loc[plan.partition.eq("holdout"), "group_id"]), "level_linear"] += 1000
            summaries.append(search_levels(frame, FEATURES, "level_linear", self.root/name, plan,
                                           CLIMATE_FEATURES, weight_column="delta_days", candidate_ids=chosen))
        self.assertEqual(summaries[0]["candidate"], summaries[1]["candidate"])
        self.assertEqual(summaries[0]["cv_mae"], summaries[1]["cv_mae"])
        for file in ("fold_scores.csv", "candidate_audit.json"):
            self.assertEqual((self.root/"first"/file).read_bytes(), (self.root/"altered"/file).read_bytes())
        a, b = (pd.read_csv(self.root/name/"holdout_predictions.csv") for name in ("first", "altered"))
        np.testing.assert_array_equal(a.prediction, b.prediction)
        audit = json.loads((self.root/"first/candidate_audit.json").read_text())
        held = set(plan.loc[plan.partition.eq("holdout"), "group_id"])
        development = set(plan.loc[plan.partition.eq("development"), "group_id"])
        for attempt in audit["attempts"]:
            if attempt["family"] != "xgboost": continue
            self.assertEqual(len(attempt["cv_training"]), 3)
            for fit in attempt["cv_training"]:
                validation = set(plan.loc[plan.fold.eq(fit["fold"]), "group_id"])
                self.assertEqual(set(fit["refit_groups"]), development-validation)
                self.assertFalse(set(fit["stopping_groups"]) & (held|validation|set(fit["fitting_groups"])))
                self.assertTrue(1 <= fit["iterations"] <= 300)
            final = attempt["final_training"]
            if final:
                self.assertEqual(set(final["refit_groups"]), development)
                self.assertFalse(set(final["stopping_groups"]) & held)
        leaders = pd.read_csv(self.root/"first/leaderboard.csv")
        self.assertTrue(leaders.candidate_status.eq("completed").all())
        selected = leaders.loc[leaders.feature_set.eq("all") & leaders.selected].iloc[0]
        self.assertEqual(summaries[0]["candidate_iterations"], selected.final_iterations)
        task = self.root/"materials/PCC/level_linear"
        task.mkdir(parents=True)
        for name in ("holdout_predictions.csv", "leaderboard.csv", "importance.csv"):
            (task/name).write_bytes((self.root/"first"/name).read_bytes())
        build_study_report(self.root, [{"material": "PCC", **summaries[0]}], {})
        combined = pd.read_csv(self.root/"model_leaderboard.csv")
        self.assertIn("cv_training", combined)
        self.assertEqual(len(combined), 6)
        self.assertNotIn("SYN", (self.root/"dashboard.html").read_text())

    def test_constant_and_missing_target_statuses_do_not_use_holdout_support(self):
        _, _, _, plan, _, edges = records(profile="smoke")
        for target, status in (("level_constant_zero", "constant_target"), ("level_constant_positive", "constant_target"),
                               ("level_positive_only_holdout", "constant_target"), ("level_all_missing", "no_valid_target"),
                               ("level_no_holdout_labels", "insufficient_groups")):
            result = search_levels(edges, FEATURES, target, self.root/target, plan,
                                   CLIMATE_FEATURES, candidate_ids=XGB_IDS)
            self.assertEqual(result["status"], status)
            self.assertFalse(result["importance_reliable"])

    def test_direct_estimator_rejects_unbounded_or_non_cpu_parameters(self):
        source = build_candidates(requested=[FIXED_IDS[0]])[FIXED_IDS[0]]
        X = pd.DataFrame({"x": [0., 1., 2., 3.]})
        for changes in ({"n_jobs": -1}, {"device": "cuda"}, {"callbacks": []}, {"n_estimators": 0}):
            with self.subTest(changes=changes), self.assertRaisesRegex(SafeError, "UNSUPPORTED_CPU_BOOSTING"):
                GroupedXGBoostRegressor({**source.parameters, **changes}).fit(X, [1., 2., 3., 4.])

    def test_fixed_and_stopped_grids_clear_frozen_nonlinear_calibration_gates(self):
        for seed in (20261011, 20261012, 20261013):
            table, _, _, plan, _, _ = records(seed=seed, profile="standard")
            frame = table.loc[table.material.eq("PCC")]
            for target in ("level_threshold", "level_interaction"):
                with self.subTest(seed=seed, target=target):
                    out = self.root/f"{seed}-{target}"
                    summary = search_levels(frame, FEATURES, target, out, plan, CLIMATE_FEATURES,
                                            seed=seed, weight_column="delta_days", candidate_ids=XGB_IDS)
                    leaders = pd.read_csv(out/"leaderboard.csv")
                    self.assertTrue(leaders.candidate_status.eq("completed").all())
                    all_features = leaders.loc[leaders.feature_set.eq("all")].set_index("candidate")
                    baseline = all_features.loc["median", "cv_mae"]
                    self.assertLessEqual(all_features.loc[FIXED_IDS, "cv_mae"].min(), .8*baseline)
                    self.assertLessEqual(all_features.loc[STOPPED_IDS, "cv_mae"].min(), .8*baseline)
                    self.assertEqual(summary["candidate_failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
