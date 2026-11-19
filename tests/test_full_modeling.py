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

from pavement_analytics.full_modeling import _rate_benchmark, consolidate_full_models, run_full_models
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

    def test_signed_row_map_preserves_curated_pair_identity_after_model_reordering(self):
        groups = {value: f"input_{i:04d}.csv" for i, value in enumerate(self.frame.partition_airfield.unique())}
        originals = {value: f"generated_source_{i:03d}.csv" for i, value in enumerate(groups.values())}
        frame = self.frame.copy()
        frame["partition_airfield"] = frame.partition_airfield.map(groups)
        frame["_pipeline_model_row"] = np.arange(len(frame))
        frame["DetRatePerYear"] = -frame.PCI_start / 20
        curated = pd.DataFrame({"_source_file_start": frame.partition_airfield,
            "_source_file_end": frame.partition_airfield, "SectionUniqueID": frame.partition_section,
            "start_date": "2020-01-01", "end_date": "2022-01-01"})
        curated_dir = self.pipeline/"06_curated"
        curated_dir.mkdir()
        curated.to_csv(curated_dir/"PCC.csv", index=False)
        setup = {"sources": [{"normalized": str(self.root/"inputs"/name),
                              "original_path": str(self.root/"generated"/original)}
                             for name, original in originals.items()]}
        (self.root/"full_setup_report.json").write_text(json.dumps(setup))
        plan = self.plan.copy()
        plan["group_id"] = plan.group_id.map(groups)
        plan.to_csv(self.pipeline/"07_modeling/partitions.csv", index=False)
        frame = frame.sample(frac=1, random_state=8).reset_index(drop=True)
        frame.to_csv(self.modeldir/"PCC.csv", index=False)
        additions = pd.DataFrame([
            {"column": "DetRatePerYear", "disposition": "TARGET"},
            {"column": "_pipeline_model_row", "disposition": "PARTITION"}])
        pd.concat([self.feature_audit, additions], ignore_index=True).to_csv(
            self.modeldir/"PCC_feature_audit.csv", index=False)
        cfg = {**self.cfg, "local_model_options": {"candidate_ids": ["enet_0.01"],
                                                     "comparisons": False, "isolated": False}}
        _, _, artifacts = run_full_models(self.pipeline, self.report, cfg)
        out = self.pipeline/"09_local_models"
        mapping = pd.read_csv(out/"row_map.csv")
        self.assertIn(out/"rate_benchmark.csv", artifacts)
        self.assertEqual(mapping.model_csv_row.tolist(), list(range(2, len(frame)+2)))
        self.assertEqual(mapping.curated_row.tolist(), frame._pipeline_model_row.tolist())
        for row in mapping.itertuples():
            original = curated.iloc[row.curated_row]
            identity = ["PCC", originals[original._source_file_start], originals[original._source_file_end],
                        original.SectionUniqueID, original.start_date, original.end_date]
            canonical = json.dumps(identity, ensure_ascii=True, allow_nan=False, separators=(",", ":"))+"\n"
            self.assertEqual(row.pair_id, hashlib.sha256(canonical.encode()).hexdigest())
        self.assertEqual(mapping.pair_id.nunique(), len(frame))
        saved = pd.read_csv(out/"materials/PCC/DetRatePerYear/holdout_predictions.csv")
        linked = mapping.set_index("row_id").loc[saved.row_id]
        np.testing.assert_allclose(saved.actual, frame.iloc[linked.model_csv_row.to_numpy()-2].DetRatePerYear,
                                   rtol=1e-12, atol=1e-12)


class RateBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pipeline = self.root/"pipeline"
        (self.pipeline/"06_curated").mkdir(parents=True)
        self.features = ["x", "material_name"]
        sources = [f"input_{i:04d}.csv" for i in range(3)]
        self.table = pd.DataFrame({"group_id": np.repeat(sources, 2),
            "_pipeline_model_row": range(6), "DetRatePerYear": -np.arange(1., 7.),
            "delta_days": np.arange(1., 7.)*365, "x": np.arange(6.), "material_name": ["type_a", "type_b"]*3})
        self.plan = pd.DataFrame({"group_id": sources,
            "partition": ["development", "development", "holdout"], "fold": [0, 1, -1]})
        self.curated = pd.DataFrame({"_source_file_start": self.table.group_id,
            "_source_file_end": self.table.group_id, "SectionUniqueID": ["section_0", "section_1"]*3,
            "start_date": ["2020-01-01", "2021-01-01"]*3, "end_date": ["2021-01-01", "2023-01-01"]*3})
        self.setup = {"sources": [{"normalized": str(self.root/"inputs"/name),
                                   "original_path": str(self.root/"generated"/f"source_{i}.csv")}
                                  for i, name in enumerate(sources)]}
        self.write_sources(self.curated, self.setup)

    def write_sources(self, curated, setup):
        curated.to_csv(self.pipeline/"06_curated/PCC.csv", index=False)
        (self.root/"full_setup_report.json").write_text(json.dumps(setup))

    def benchmark(self, table=None, plan=None):
        return _rate_benchmark(self.pipeline, "PCC", self.table if table is None else table,
                               self.features, self.plan if plan is None else plan, {})

    def test_source_renumbering_and_curated_and_model_reordering_preserve_evidence(self):
        first_ids, first = self.benchmark()
        renames = {name: f"input_{i+100:04d}.csv" for i, name in enumerate(self.plan.group_id)}
        setup = json.loads(json.dumps(self.setup))
        for source in setup["sources"]:
            source["normalized"] = str(self.root/"inputs"/renames[Path(source["normalized"]).name])
        order = [2, 5, 0, 4, 1, 3]
        positions = {old: new for new, old in enumerate(order)}
        curated = self.curated.replace(renames).iloc[order].reset_index(drop=True)
        table = self.table.copy()
        table["group_id"] = table.group_id.map(renames)
        table["_pipeline_model_row"] = table._pipeline_model_row.map(positions)
        table = table.iloc[[5, 2, 1, 3, 0, 4]].reset_index(drop=True)
        plan = self.plan.copy()
        plan["group_id"] = plan.group_id.map(renames)
        self.write_sources(curated, setup)
        later_ids, later = self.benchmark(table, plan)
        self.assertEqual(dict(zip(self.table.x, first_ids)), dict(zip(table.x, later_ids)))
        self.assertEqual(first, later)
        self.assertEqual([row["eligible_rows"] for row in first], [4, 2])
        self.assertTrue(all(row["identity_status"] == "ready" for row in first))

    def test_each_fingerprint_changes_only_for_its_evaluation_role(self):
        _, records = self.benchmark()
        first = records[0]
        changes = [("DetRatePerYear", -2.5, "outcomes_weights_sha256"),
                   ("delta_days", 500., "outcomes_weights_sha256"),
                   ("x", 99., "features_sha256"),
                   ("material_name", "type_c", "features_sha256")]
        hash_columns = [name for name in first if name.endswith("_sha256")]
        for column, value, changed_hash in changes:
            with self.subTest(column=column):
                table = self.table.copy()
                table.loc[0, column] = value
                _, changed = self.benchmark(table)
                self.assertEqual(records[1], changed[1])
                for key in hash_columns:
                    if key == changed_hash:
                        self.assertNotEqual(first[key], changed[0][key])
                    else:
                        self.assertEqual(first[key], changed[0][key])
        plan = self.plan.copy()
        plan.loc[:1, "fold"] = [1, 0]
        _, changed = self.benchmark(plan=plan)
        self.assertNotEqual(first["group_folds_sha256"], changed[0]["group_folds_sha256"])
        for key in set(hash_columns)-{"group_folds_sha256"}:
            self.assertEqual(first[key], changed[0][key])
        plan.loc[0, ["partition", "fold"]] = ["holdout", -1]
        _, repartitioned = self.benchmark(plan=plan)
        self.assertEqual([row["eligible_rows"] for row in repartitioned], [2, 4])
        self.assertNotEqual(records[0]["pair_ids_sha256"], repartitioned[0]["pair_ids_sha256"])
        self.assertNotEqual(records[1]["pair_ids_sha256"], repartitioned[1]["pair_ids_sha256"])

    def test_zero_targets_are_eligible_and_missing_targets_have_no_hashes(self):
        table = self.table.copy()
        table["DetRatePerYear"] = 0.
        _, zero = self.benchmark(table)
        self.assertTrue(all(row["identity_status"] == "ready" for row in zero))
        table["DetRatePerYear"] = np.nan
        _, missing = self.benchmark(table)
        self.assertTrue(all(row["identity_status"] == "no_eligible_rows" and row["eligible_rows"] == 0
                            for row in missing))
        self.assertTrue(all(value is None for row in missing for key, value in row.items()
                            if key.endswith("_sha256")))

    def test_missing_invalid_and_duplicate_pair_identities_are_unavailable(self):
        for position in (np.nan, -1, 1.5, len(self.curated)):
            with self.subTest(position=position):
                table = self.table.copy()
                table["_pipeline_model_row"] = table._pipeline_model_row.astype(float)
                table.loc[0, "_pipeline_model_row"] = position
                ids, evidence = self.benchmark(table)
                self.assertEqual(ids.iloc[0], "")
                self.assertEqual(evidence[0]["identity_status"], "identity_unavailable")
                self.assertEqual(evidence[1]["identity_status"], "ready")
                self.assertTrue(all(value is None for key, value in evidence[0].items()
                                    if key.endswith("_sha256")))
        table = self.table.copy()
        table.loc[0, "DetRatePerYear"] = np.nan
        table.loc[0, "_pipeline_model_row"] = -1
        _, evidence = self.benchmark(table)
        self.assertEqual(evidence[0]["identity_status"], "ready")
        self.assertEqual(evidence[0]["eligible_rows"], 3)
        duplicate = self.curated.copy()
        duplicate.loc[1] = duplicate.loc[0]
        self.write_sources(duplicate, self.setup)
        ids, evidence = self.benchmark()
        self.assertTrue(ids.iloc[:2].eq("").all())
        self.assertEqual(evidence[0]["identity_status"], "identity_unavailable")
        self.write_sources(self.curated.drop(columns="start_date"), self.setup)
        _, evidence = self.benchmark()
        self.assertTrue(all(row["identity_status"] == "identity_unavailable" for row in evidence))


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
