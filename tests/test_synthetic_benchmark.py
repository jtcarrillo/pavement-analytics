"""Synthetic model fixtures protect split, target, oracle and failure contracts."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pavement_analytics.security import SafeError
from pavement_analytics.study_modeling import _estimator, search_levels
from pavement_analytics.synthetic_benchmark import (
    CLIMATE_FEATURES, FEATURES, TARGETS, create_bundle, records, run_smoke, validate_bundle,
)
from temp_support import temporary_directory


def fast_candidates(seed):
    return {"generated_rf": _estimator(RandomForestRegressor(n_estimators=5, max_depth=3, random_state=seed, n_jobs=1))}


class SyntheticBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temp = temporary_directory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bundle = self.root / "bundle"

    def test_fixed_seed_reproduces_tables_and_oracles_without_source_inputs(self):
        first, second = records(profile="smoke"), records(profile="smoke")
        for a, b in zip(first, second):
            pd.testing.assert_frame_equal(a, b)
        self.assertFalse(first[0].equals(records(seed=1234, profile="smoke")[0]))
        table, stress, oracle, plan, _, _ = first
        self.assertEqual((len(table), table.group_id.nunique(), len(TARGETS)), (180, 15, 9))
        self.assertEqual(table.groupby("group_id").material.nunique().tolist(), [3]*15)
        self.assertTrue(table.row_id.equals(oracle.row_id))
        self.assertTrue(table.row_id.equals(stress.row_id))
        self.assertEqual(set(plan.group_id), set(table.group_id))
        self.assertEqual(plan.group_id.nunique(), len(plan))
        self.assertFalse(set(FEATURES)&(set(TARGETS)|set(oracle.columns)))

    def test_partition_group_identity_and_interval_history_survive_csv_round_trip(self):
        create_bundle(self.bundle, profile="smoke")
        summary = validate_bundle(self.bundle)
        self.assertEqual((summary["development_groups"], summary["holdout_groups"]), (12, 3))
        table = pd.read_csv(self.bundle/"model_table.csv", dtype={"group_id": str})
        for _, block in table.groupby("section_id", sort=False):
            self.assertEqual(block.start_date.iloc[1:].tolist(), block.end_date.iloc[:-1].tolist())
        self.assertTrue(table.group_id.str.startswith("SYN").all())
        self.assertTrue(np.isfinite(table[TARGETS].to_numpy()).all())
        self.assertTrue(table[TARGETS].ge(0).all().all())

    def test_stress_fixture_separates_missingness_and_unseen_holdout_categories(self):
        table, stress, _, plan, _, _ = records(profile="smoke")
        held = stress.group_id.isin(plan.loc[plan.partition.eq("holdout"), "group_id"])
        self.assertNotIn("HOLDOUT_ONLY", set(stress.loc[~held, "subgrade_category"]))
        self.assertTrue(stress.loc[held, "subgrade_category"].eq("HOLDOUT_ONLY").all())
        self.assertTrue(stress.all_missing_predictor.isna().all())
        self.assertGreater(stress[FEATURES].isna().sum().sum(), 0)
        self.assertGreater(stress[TARGETS].isna().sum().sum(), 0)
        pd.testing.assert_series_equal(table.row_id, stress.row_id)

    def test_two_stage_truth_distinguishes_mean_median_and_positive_magnitude(self):
        table, _, oracle, _, _, _ = records()
        p, severity = oracle.zero__p_positive.to_numpy(), oracle.zero__positive_mean.to_numpy()
        expected = p*severity
        median = np.where(p>0.5, severity*(1.4-0.4/p), 0)
        np.testing.assert_allclose(expected, oracle.level_zero_inflated__conditional_mean)
        np.testing.assert_allclose(median, oracle.zero__conditional_median)
        self.assertTrue((expected[p<=0.5]>0).all())
        self.assertTrue((median[p<=0.5]==0).all())
        self.assertGreater(table.level_zero_inflated.eq(0).mean(), 0.5)
        positive = table.level_zero_inflated.gt(0).to_numpy()
        self.assertTrue((table.loc[positive, "level_zero_inflated"]>=0.6*severity[positive]).all())
        self.assertTrue((table.loc[positive, "level_zero_inflated"]<=1.4*severity[positive]).all())

    def test_current_models_handle_stress_and_report_explicit_edge_statuses(self):
        table, stress, _, plan, _, edges = records(profile="smoke")
        with patch("pavement_analytics.study_modeling.candidates", fast_candidates):
            summary = search_levels(stress, [*FEATURES, "all_missing_predictor"], "level_linear", self.root/"stress", plan, CLIMATE_FEATURES, weight_column="delta_days")
            self.assertIn(summary["status"], {"modeled", "baseline_selected"})
            prediction = pd.read_csv(self.root/"stress/holdout_predictions.csv").prediction
            self.assertTrue(np.isfinite(prediction).all())
            self.assertTrue(prediction.ge(0).all())
            for target, status in [("level_all_missing", "no_valid_target"), ("level_constant_zero", "constant_target"),
                                   ("level_constant_positive", "constant_target"), ("level_development_constant", "constant_target"),
                                   ("level_no_holdout_labels", "insufficient_groups")]:
                result = search_levels(edges, FEATURES, target, self.root/target, plan, CLIMATE_FEATURES, weight_column="delta_days")
                self.assertEqual(result["status"], status)

    def test_invalid_fixture_inputs_are_rejected_before_model_fitting(self):
        create_bundle(self.bundle, profile="smoke")
        plan = pd.read_csv(self.bundle/"partitions.csv")
        for name, code in [("invalid_weights", "INVALID_STUDY_WEIGHTS"), ("duplicate_row_ids", "INVALID_STUDY_MODEL_TABLE")]:
            table = pd.read_csv(self.bundle/f"controls/{name}.csv")
            with self.assertRaisesRegex(SafeError, code):
                search_levels(table, FEATURES, "level_linear", self.root/name, plan, CLIMATE_FEATURES, weight_column="delta_days")

    def test_smoke_exercises_installed_candidates_with_a_shared_plan(self):
        create_bundle(self.bundle, profile="smoke")
        result = self.root/"results"
        with patch("pavement_analytics.study_modeling.candidates", fast_candidates):
            summaries = run_smoke(self.bundle, result)
        self.assertEqual(len(summaries), 3)
        tables = [pd.read_csv(result/"PCC"/item["target"]/"folds.csv") for item in summaries]
        for frame in tables[1:]:
            pd.testing.assert_frame_equal(tables[0], frame)
        metadata = json.loads((result/"benchmark_result.json").read_text())
        self.assertTrue(metadata["synthetic"])
        self.assertEqual(metadata["candidate_source"], "pavement_analytics.study_modeling.candidates")

    def test_generated_files_are_hashed_and_existing_outputs_are_preserved(self):
        create_bundle(self.bundle, profile="smoke")
        original = (self.bundle/"model_table.csv").read_bytes()
        with self.assertRaisesRegex(SafeError, "EMPTY_DIRECTORY"):
            create_bundle(self.bundle, profile="smoke")
        self.assertEqual((self.bundle/"model_table.csv").read_bytes(), original)
        (self.bundle/"model_table.csv").write_bytes(original+b"\n")
        with self.assertRaisesRegex(SafeError, "ARTIFACT_CHANGED"):
            validate_bundle(self.bundle)

    def test_generation_refuses_source_tree_symlinks_and_overlapping_results(self):
        source = Path(__file__).resolve().parents[1]/"generated-benchmark"
        with self.assertRaisesRegex(SafeError, "OUTSIDE_PUBLIC_TREE"):
            create_bundle(source, profile="smoke")
        target = self.root/"target"
        target.mkdir()
        link = self.root/"linked"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(SafeError, "SYMLINK_PATH_FORBIDDEN"):
            create_bundle(link, profile="smoke")
        create_bundle(self.bundle, profile="smoke")
        with self.assertRaisesRegex(SafeError, "RESULTS_MUST_BE_SEPARATE"):
            run_smoke(self.bundle, self.bundle/"results")


if __name__ == "__main__":
    unittest.main()
