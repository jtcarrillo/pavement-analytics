"""Generated inspection records exercise level targets and outcome isolation."""

import json
import unittest

import numpy as np
import pandas as pd

from pavement_analytics.observations import build_observations, select_covariates
from pavement_analytics.security import SafeError


COLUMNS = {"section": "section", "date": "date", "pci": "pci", "surface": "surface", "group": "group"}


def generated(rows=3):
    return pd.DataFrame({"section": ["synthetic-section"] * rows,
                         "date": pd.date_range("2020-01-01", periods=rows, freq="30D").strftime("%Y-%m-%d"),
                         "pci": [95] * rows, "surface": ["PCC"] * rows,
                         "group": ["synthetic-group"] * rows,
                         "construction": ["no"] * rows,
                         "63L deduct": list(range(1, rows + 1)),
                         "74L deduct": [4] * rows,
                         "_source_file": ["generated.csv"] * rows,
                         "_source_row": list(range(2, rows + 2)),
                         "_deduct_schema": [("63L deduct", "74L deduct")] * rows})


class ObservationTests(unittest.TestCase):
    def build(self, raw, **kwargs):
        return build_observations(raw, COLUMNS, {"thermal": ["63"], "freeze": ["74"]}, **kwargs)

    def test_levels_include_first_short_interval_increase_and_construction(self):
        raw = generated(3)
        raw["pci"] = [80, 90, 100]
        raw["construction"] = ["no", "yes", "no"]
        table, targets, rejected, audit = self.build(raw)
        self.assertEqual(targets, ["level_thermal", "level_freeze"])
        self.assertEqual(len(table), 3)
        self.assertEqual(table.level_thermal.tolist(), [1, 2, 3])
        self.assertEqual(table.row_id.tolist(), [0, 1, 2])
        self.assertEqual(table.inspection_date.tolist(), raw.date.tolist())
        self.assertTrue(rejected.empty)
        self.assertEqual(audit["accounting"], {"input_rows": 3, "accepted_rows": 3, "rejected_rows": 0})
        self.assertNotIn("DetRatePerYear", table)

    def test_identity_and_date_validation_account_for_every_row(self):
        raw = generated(6)
        raw.loc[0, "date"] = "unparseable"
        raw.loc[1, "date"] = "1979-01-01"
        raw.loc[2, "section"] = ""
        raw.loc[3, "group"] = " "
        raw.loc[4, "surface"] = "unknown"
        table, _, rejected, audit = self.build(raw)
        self.assertEqual(len(table), 1)
        self.assertEqual(set(rejected.reason), {"invalid_date", "before_cutoff", "missing_section", "missing_group", "unknown_surface"})
        self.assertEqual(sum(audit["rejected_reasons"].values()) + len(table), len(raw))

    def test_invalid_pci_is_audited_without_discarding_deduct_target(self):
        raw = generated(3)
        raw["pci"] = ["bad", "101", ""]
        table, _, _, audit = self.build(raw)
        self.assertEqual(table.level_thermal.tolist(), [1, 2, 3])
        self.assertEqual(audit["invalid_pci_rows_retained"], 3)

    def test_exact_duplicates_collapse_across_sources_and_conflicts_all_reject(self):
        raw = generated(2)
        duplicate = raw.iloc[[0]].copy()
        duplicate["_source_file"] = "generated-copy.csv"
        duplicate["_source_row"] = 30
        conflict = raw.iloc[[1]].copy()
        conflict["63L deduct"] = 50
        conflict["_source_file"] = "generated-copy.csv"
        table, _, rejected, audit = self.build(pd.concat([raw, duplicate, conflict], ignore_index=True))
        self.assertEqual(table.level_thermal.tolist(), [1])
        self.assertEqual(audit["rejected_reasons"], {"conflicting_inspection": 2, "duplicate_inspection_collapsed": 1})
        self.assertEqual(len(rejected), 3)

    def test_duplicate_schema_difference_is_conflict_even_when_blanks_match(self):
        raw = generated(1)
        copied = raw.copy()
        copied["_deduct_schema"] = [("63L deduct",)]
        table, _, rejected, _ = self.build(pd.concat([raw, copied], ignore_index=True))
        self.assertTrue(table.empty)
        self.assertEqual(rejected.reason.tolist(), ["conflicting_inspection"] * 2)

    def test_invalid_family_does_not_erase_other_valid_family(self):
        raw = generated(3)
        raw["63L deduct"] = ["bad", 101, -1]
        table, _, _, audit = self.build(raw)
        self.assertTrue(table.level_thermal.isna().all())
        self.assertEqual(table.level_freeze.tolist(), [4, 4, 4])
        thermal = next(item for item in audit["targets"] if item["target"] == "level_thermal")
        self.assertEqual(thermal["reasons"], {"invalid_family_value": 3})

    def test_blank_zero_never_converts_malformed_text_or_absent_channels(self):
        raw = generated(4)
        raw["63L deduct"] = ["", "NaN", "inf", None]
        raw.loc[3, "_deduct_schema"] = ("74L deduct",)
        table, _, _, audit = self.build(raw, blank_deducts="zero")
        self.assertEqual(table.level_thermal.iloc[0], 0)
        self.assertTrue(table.level_thermal.iloc[1:].isna().all())
        self.assertEqual(audit["blank_deduct_cells"], 1)
        self.assertEqual(audit["zero_filled_deduct_cells"], 1)
        self.assertEqual(table.level_freeze.tolist(), [4, 4, 4, 4])

    def test_default_blank_semantics_remain_missing(self):
        raw = generated(1)
        raw["63L deduct"] = ""
        table, _, _, audit = self.build(raw)
        self.assertTrue(table.level_thermal.isna().all())
        self.assertEqual(table.level_freeze.iloc[0], 4)
        self.assertEqual(audit["zero_filled_deduct_cells"], 0)

    def test_ungraded_and_graded_overlap_invalidates_only_affected_family(self):
        raw = generated(3)
        raw["63 deduct"] = [5, 0, 8]
        raw["63L deduct"] = [2, 3, 0]
        raw["_deduct_schema"] = [("63 deduct", "63L deduct", "74L deduct")] * 3
        table, _, _, audit = self.build(raw)
        self.assertTrue(np.isnan(table.level_thermal.iloc[0]))
        self.assertEqual(table.level_thermal.iloc[1:].tolist(), [3, 8])
        self.assertEqual(table.level_freeze.tolist(), [4, 4, 4])
        thermal = next(item for item in audit["targets"] if item["target"] == "level_thermal")
        self.assertEqual(thermal["reasons"]["ambiguous_family_representation"], 1)

    def test_all_zero_low_condition_export_gap_excludes_targets(self):
        raw = generated(3)
        raw["63L deduct"] = 0
        raw["74L deduct"] = 0
        raw["pci"] = [90, 100, "bad"]
        table, _, _, audit = self.build(raw)
        self.assertTrue(np.isnan(table.level_thermal.iloc[0]))
        self.assertEqual(table.level_thermal.iloc[1:].tolist(), [0, 0])
        self.assertEqual(audit["export_gap_rows"], 1)

    def test_source_absent_family_unknown_and_partial_coverage_audited(self):
        raw = generated(2)
        raw["_deduct_schema"] = [("63L deduct",), ("74L deduct",)]
        table, _, _, audit = build_observations(raw, COLUMNS, {"thermal": ["61", "63"], "freeze": ["74"]}, blank_deducts="zero")
        self.assertEqual(table.level_thermal.iloc[0], 1)
        self.assertTrue(np.isnan(table.level_thermal.iloc[1]))
        self.assertTrue(np.isnan(table.level_freeze.iloc[0]))
        self.assertEqual(table.level_freeze.iloc[1], 4)
        coverage = next(item for item in audit["source_coverage"] if item["exported_deduct_columns"] == ["63L deduct"])
        self.assertEqual(coverage["families"]["thermal"]["unexported_codes"], ["61"])

    def test_empty_input_and_absent_family_have_stable_schema(self):
        table, targets, rejected, audit = self.build(generated(0))
        self.assertTrue(table.empty)
        self.assertTrue(set(["row_id", "group_id", "inspection_date", *targets]) <= set(table))
        self.assertTrue(rejected.empty)
        self.assertEqual(audit["accounting"]["input_rows"], 0)

    def test_overlapping_families_rejected(self):
        with self.assertRaisesRegex(SafeError, "OVERLAPPING_DEDUCT_FAMILIES"):
            build_observations(generated(), COLUMNS, {"one": ["63"], "two": ["63"]})


class CovariateTests(unittest.TestCase):
    def test_safe_fields_converted_and_outcome_identity_geography_dropped(self):
        table = pd.DataFrame({"Years since Major   MR": ["2", "bad", "-1"],
                              "InspectedArea   (SqFt)": ["500", "600", "inf"],
                              "Subgrade Category": ["A", " ", "B"],
                              "construction": ["yes", "no", ""],
                              "PCI": [80, 90, 100], "SCI": [90, 91, 100],
                              "63L deduct": [5, 4, 0], "63L (Slabs)": [5, 4, 0],
                              "level_thermal": [5, 4, 0], "NetworkID": [1, 2, 3],
                              "latitude": [1, 2, 3], "custom_numeric": [1, 2, 3]})
        features, audit = select_covariates(table)
        self.assertEqual(features, ["Years since Major   MR", "InspectedArea   (SqFt)", "Subgrade Category", "construction"])
        self.assertEqual(table["Years since Major   MR"].iloc[0], 2)
        self.assertTrue(table["Years since Major   MR"].iloc[1:].isna().all())
        self.assertEqual(table.construction.iloc[:2].tolist(), [1, 0])
        self.assertTrue(pd.isna(table["Subgrade Category"].iloc[1]))
        self.assertEqual(audit.loc[audit.column.eq("level_thermal"), "disposition"].iloc[0], "TARGET")
        json.dumps(audit.to_dict("records"), allow_nan=False)

    def test_explicit_selection_cannot_admit_unreviewed_predictors(self):
        table = pd.DataFrame({"PCI": [80], "custom_numeric": [1], "Years Since Major MR": [2]})
        for column in ["PCI", "custom_numeric"]:
            with self.subTest(column=column), self.assertRaisesRegex(SafeError, "FEATURE_REVIEW_REQUIRED"):
                select_covariates(table, [column])
        features, _ = select_covariates(table, ["Years Since Major MR"])
        self.assertEqual(features, ["Years Since Major MR"])

    def test_all_missing_and_unrequested_features_not_admitted(self):
        table = pd.DataFrame({"InspectedSlabs": ["bad", ""], "Subgrade Category": ["A", "B"]})
        features, audit = select_covariates(table, ["InspectedSlabs"])
        self.assertEqual(features, [])
        self.assertEqual(audit.disposition.tolist(), ["DROP_ALL_MISSING", "DROP_NOT_REQUESTED"])

    def test_invalid_allowlist_types_fail_with_stable_error(self):
        for requested in ("PCI", [1], [{}], ["PCI", "PCI"]):
            with self.subTest(requested=requested), self.assertRaisesRegex(SafeError, "INVALID_FEATURE_ALLOWLIST"):
                select_covariates(pd.DataFrame({"PCI": [80]}), requested)


if __name__ == "__main__":
    unittest.main()
