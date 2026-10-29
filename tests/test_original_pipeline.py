"""Regression coverage for source-specific targets and model feature boundaries."""
import contextlib
import io
import json
from pathlib import Path
import unittest

import pandas as pd

from pavement_analytics.original.common import Run, read_table, write_csv
from pavement_analytics.original.pipeline import DEFAULTS, Pipeline
from temp_support import temporary_directory


class OriginalPipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()

    def pipeline(self, files=(), **changes):
        config = {**DEFAULTS, "output_dir": str(self.root / "run"),
                  "surfaces": ["PCC"], "skip_climate": True,
                  "min_groups": 1, **changes}
        return Pipeline(config, Run(config["output_dir"], "generated-fixture", config, []), list(files), {})

    @staticmethod
    def quiet(function, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return function(*args)

    def test_unexported_family_is_missing_while_exported_zero_remains_observed(self):
        exports = {
            "freeze.csv": {"64L deduct": [1, 3], "64M deduct": [0, 2]},
            "load.csv": {"62L deduct": [2, 4]},
            "zero.csv": {"64L deduct": [0, 0], "69L deduct": [0, 0]},
            "quantity_only.csv": {"64L (Slabs)": [1, 2]},
        }
        files = []
        for name, measurements in exports.items():
            frame = pd.DataFrame({"SectionUniqueID": ["001", "001"],
                                  "Date": ["2019-01-01", "2020-01-01"], "PCI": [100, 90],
                                  "Inspected Surface Type": ["PCC", "PCC"],
                                  "IsConstruction": [False, False], **measurements})
            path = self.raw / name
            frame.to_csv(path, index=False)
            files.append(path)
        pipeline = self.pipeline(files)
        self.quiet(pipeline.ingest)
        inspections = read_table(pipeline.path("01_split", "PCC"))
        for name, measurements in exports.items():
            expected = sorted(c for c in measurements if c.endswith(" deduct"))
            for schema in inspections.loc[inspections._source_file.eq(name), "_pipeline_deduct_schema"]:
                self.assertEqual(json.loads(schema), expected)
        self.quiet(pipeline.pair, "PCC")
        paired = read_table(pipeline.path("02_pairs", "PCC"))
        for when in ("start", "end"):
            for row in paired.to_dict("records"):
                name = row[f"_source_file_{when}"]
                self.assertEqual(json.loads(row[f"_pipeline_deduct_schema_{when}"]),
                                 sorted(c for c in exports[name] if c.endswith(" deduct")))
        self.quiet(pipeline.location_rate, "PCC")
        self.quiet(pipeline.climate, "PCC")
        self.quiet(pipeline.derive_targets, "PCC")
        targets = read_table(pipeline.path("05_targets", "PCC")).set_index("_source_file_end")
        freeze = "fam_pcc_freeze_thaw deduct_end"
        load = "fam_pcc_load deduct_end"
        self.assertEqual(targets.loc["freeze.csv", freeze], 5)
        self.assertEqual(targets.loc["zero.csv", freeze], 0)
        self.assertTrue(targets.loc[["load.csv", "quantity_only.csv"], freeze].isna().all())
        self.assertEqual(targets.loc["load.csv", load], 4)
        self.assertTrue(targets.loc[["freeze.csv", "zero.csv", "quantity_only.csv"], load].isna().all())
        moisture = "fam_pcc_moisture deduct_end"
        self.assertEqual(targets.loc["zero.csv", moisture], 0)
        self.assertTrue(targets.loc[["freeze.csv", "load.csv", "quantity_only.csv"], moisture].isna().all())
        schema_audit = pd.read_csv(pipeline.out / "01_split/source_family_schema.csv").set_index(["source", "family"])
        partial = schema_audit.loc[("freeze.csv", "pcc_freeze_thaw")]
        self.assertEqual(partial.coverage, "partial")
        self.assertEqual(json.loads(partial.observed_codes), ["64"])
        self.assertEqual(json.loads(partial.missing_codes), ["70", "74", "75"])
        self.assertEqual(schema_audit.loc[("quantity_only.csv", "pcc_freeze_thaw"), "coverage"], "unobserved")
        coverage = pd.read_csv(pipeline.out / "05_targets/PCC_family_target_coverage.csv").set_index(["source", "endpoint", "family"])
        absent = coverage.loc[("load.csv", "end", "pcc_freeze_thaw")]
        self.assertEqual(absent.coverage, "unobserved")
        self.assertEqual((absent.observed_rows, absent.unobserved_rows, absent.finite_target_rows), (0, 1, 0))
        partial = coverage.loc[("freeze.csv", "end", "pcc_freeze_thaw")]
        self.assertEqual(partial.coverage, "partial")
        self.assertEqual((partial.observed_rows, partial.finite_target_rows, partial.partial_family_rows), (1, 1, 1))
        zero = coverage.loc[("zero.csv", "end", "pcc_moisture")]
        self.assertEqual((zero.observed_rows, zero.unobserved_rows, zero.finite_target_rows), (1, 0, 1))

        # Isolate model preparation from numerical PCI checks: source availability
        # must survive target eligibility and must not become synthetic zero labels.
        curated = targets.reset_index()
        curated["export_gap_start"] = 0
        curated["export_gap_end"] = 0
        write_csv(pipeline.path("06_curated", "PCC"), curated)
        self.quiet(pipeline.modeling, "PCC")
        model = read_table(pipeline.path("07_modeling", "PCC")).set_index("partition_airfield")
        self.assertEqual(model.loc["zero.csv", freeze], 0)
        self.assertTrue(model.loc[["load.csv", "quantity_only.csv"], freeze].isna().all())
        self.assertEqual(model.loc["zero.csv", moisture], 0)
        self.assertTrue(model.loc[["freeze.csv", "load.csv", "quantity_only.csv"], moisture].isna().all())
        eligibility = pd.read_csv(pipeline.out / "07_modeling/PCC_target_eligibility.csv")
        freeze_eligibility = eligibility.loc[eligibility.target.eq(freeze)]
        self.assertEqual(freeze_eligibility.eligible.sum(), 2)
        self.assertEqual(freeze_eligibility.loc[~freeze_eligibility.eligible, "reason"].tolist(),
                         ["invalid_or_unobserved_target", "invalid_or_unobserved_target"])

    def model_frame(self):
        return pd.DataFrame({
            "SectionUniqueID": ["001", "002", "003", "004"],
            "_source_file_start": ["a.csv", "a.csv", "b.csv", "b.csv"],
            "Network name_start": ["site-a", "site-a", "site-b", "site-b"],
            "Branch   Name_start": ["branch-a", "branch-a", "branch-b", "branch-b"],
            "network-name_start": ["network-a", "network-a", "network-b", "network-b"],
            "Section ID_start": [11, 11, 22, 22],
            "group-id_start": [101, 101, 202, 202],
            "Most recent Major MR DATE ._start": ["2017-01-01", "2017-01-01", "2018-01-01", "2018-01-01"],
            "Years since Major MR_start": [1, 2, 3, 4],
            "Inspected Branch Use_start": ["runway", "taxiway", "runway", "taxiway"],
            "delta_days": [365, 365, 365, 365], "PCI_start": [95, 90, 85, 80],
            "fam_pcc_freeze_thaw deduct_start": [1, 2, 3, 4],
            "fam_pcc_freeze_thaw deduct_end": [2, 3, 5, 8],
            "export_gap_start": [0, 0, 0, 0], "export_gap_end": [0, 0, 0, 0],
            "pci_calc_start": [95, 90, 85, 80], "pci_only_pcc_freeze_thaw_start": [96, 91, 86, 81],
            "loss_shap_pcc_freeze_thaw_start": [4, 9, 14, 19],
            "loss_loo_pcc_freeze_thaw_start": [3, 8, 13, 18],
            "tdv_start": [5, 10, 15, 20], "maxcdv_start": [5, 10, 15, 20],
        })

    def test_normalized_identity_and_date_aliases_cannot_be_model_predictors(self):
        pipeline = self.pipeline(airfield_col="Network name_start")
        write_csv(pipeline.path("06_curated", "PCC"), self.model_frame())
        self.quiet(pipeline.modeling, "PCC")
        audit = pd.read_csv(pipeline.out / "07_modeling/PCC_feature_audit.csv").set_index("column")
        for col in ("Branch   Name_start", "network-name_start", "Section ID_start", "group-id_start",
                    "Most recent Major MR DATE ._start"):
            self.assertEqual(audit.loc[col, "disposition"], "DROP", col)
        self.assertEqual(audit.loc["Network name_start", "disposition"], "PARTITION")
        for col in ("PCI_start", "Years since Major MR_start", "Inspected Branch Use_start"):
            self.assertEqual(audit.loc[col, "disposition"], "KEEP", col)
        model = read_table(pipeline.path("07_modeling", "PCC"))
        self.assertEqual(model.partition_airfield.tolist(), ["site-a", "site-a", "site-b", "site-b"])
        self.assertNotIn("Branch   Name_start", model)

    def test_run_fitted_curve_outputs_are_excluded_but_recorded_start_state_survives(self):
        pipeline = self.pipeline(fit_deduct_curves=True, keep_distress=True)
        write_csv(pipeline.path("06_curated", "PCC"), self.model_frame())
        self.quiet(pipeline.modeling, "PCC")
        audit = pd.read_csv(pipeline.out / "07_modeling/PCC_feature_audit.csv").set_index("column")
        for col in ("pci_calc_start", "pci_only_pcc_freeze_thaw_start", "loss_shap_pcc_freeze_thaw_start",
                    "loss_loo_pcc_freeze_thaw_start", "tdv_start", "maxcdv_start"):
            self.assertEqual(audit.loc[col, "disposition"], "DROP", col)
            self.assertEqual(audit.loc[col, "reason"], "run_fitted_curve_uses_all_rows", col)
        for col in ("PCI_start", "fam_pcc_freeze_thaw deduct_start"):
            self.assertEqual(audit.loc[col, "disposition"], "KEEP", col)


if __name__ == "__main__":
    unittest.main()
