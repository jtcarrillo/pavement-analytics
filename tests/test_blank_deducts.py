"""Generated pairs verify explicit blank-deduct policies and their audit counts."""

import unittest

import numpy as np
import pandas as pd

from pavement_analytics.distress import derive
from pavement_analytics.security import SafeError


class BlankDeductTests(unittest.TestCase):
    def pairs(self, start="", end="2", *, pci=90):
        return pd.DataFrame([{
            "PCI_start": pci, "PCI_end": pci - 1,
            "delta_days": 730, "DetRatePerYear": -0.5, "material": "AC",
            "43L deduct_start": start, "43L deduct_end": end,
            "43M deduct_start": "4", "43M deduct_end": "6",
        }])

    def test_default_policy_retains_missing_values(self):
        result, _ = derive(self.pairs(), {"cracking": ["43"]})
        self.assertTrue(result.invalid_deduct_start.iloc[0])
        self.assertTrue(result.rate_cracking.isna().all())
        self.assertEqual(result.blank_deduct_cells_start.iloc[0], 1)
        self.assertEqual(result.zero_filled_deduct_cells_start.iloc[0], 0)

    def test_confirmed_blank_zeros_enable_family_growth_and_are_audited(self):
        pairs = self.pairs()
        result, targets = derive(pairs, {"cracking": ["43"]}, blank_deducts="zero")
        self.assertEqual(targets, ["rate_cracking"])
        self.assertEqual(result.family_cracking_start.iloc[0], 4)
        self.assertEqual(result.family_cracking_end.iloc[0], 8)
        self.assertEqual(result.rate_cracking.iloc[0], 2)
        self.assertFalse(result.invalid_deduct_start.iloc[0])
        self.assertEqual(result.blank_deduct_cells_start.iloc[0], 1)
        self.assertEqual(result.zero_filled_deduct_cells_start.iloc[0], 1)
        self.assertEqual(result.zero_filled_deduct_cells_end.iloc[0], 0)
        self.assertEqual(pairs["43L deduct_start"].iloc[0], "")
        self.assertEqual(result.DetRatePerYear.iloc[0], -0.5)

    def test_whitespace_and_actual_nulls_are_blank_but_numeric_zero_is_not(self):
        blanks = ["", " \t\n", None, np.nan, pd.NA]
        for value in blanks:
            with self.subTest(value=value):
                result, _ = derive(self.pairs(value, value), {"cracking": ["43"]},
                                   blank_deducts="zero")
                self.assertEqual(result.rate_cracking.iloc[0], 1)
                for endpoint in ("start", "end"):
                    self.assertEqual(result[f"blank_deduct_cells_{endpoint}"].iloc[0], 1)
                    self.assertEqual(result[f"zero_filled_deduct_cells_{endpoint}"].iloc[0], 1)
        result, _ = derive(self.pairs("0", 0), {"cracking": ["43"]}, blank_deducts="zero")
        self.assertEqual(result.blank_deduct_cells_start.iloc[0], 0)
        self.assertEqual(result.blank_deduct_cells_end.iloc[0], 0)

    def test_malformed_and_out_of_range_values_are_never_filled(self):
        for value in ["NaN", "N/A", "not recorded", "inf", "-inf", np.inf, -np.inf, -1, 101]:
            with self.subTest(value=value):
                result, _ = derive(self.pairs(value, value), {"cracking": ["43"]},
                                   blank_deducts="zero")
                self.assertTrue(result.invalid_deduct_start.iloc[0])
                self.assertTrue(result.invalid_deduct_end.iloc[0])
                self.assertTrue(result.rate_cracking.isna().all())
                self.assertEqual(result.blank_deduct_cells_start.iloc[0], 0)
                self.assertEqual(result.zero_filled_deduct_cells_end.iloc[0], 0)

    def test_source_absence_never_becomes_a_zero_or_invalidates_other_sources(self):
        pairs = self.pairs()
        for endpoint in ("start", "end"):
            pairs[f"63L deduct_{endpoint}"] = np.nan
            pairs[f"_deduct_schema_{endpoint}"] = [("43L deduct", "43M deduct")]
        result, _ = derive(pairs, {"cracking": ["43"], "absent": ["63"]}, blank_deducts="zero")
        self.assertEqual(result.rate_cracking.iloc[0], 2)
        self.assertTrue(result.rate_absent.isna().all())
        self.assertTrue(result.family_absent_start.isna().all())
        self.assertTrue(result["63L deduct_start"].isna().all())
        self.assertFalse(result.invalid_deduct_start.iloc[0])
        self.assertEqual(result.blank_deduct_cells_start.iloc[0], 1)
        self.assertEqual(result.zero_filled_deduct_cells_start.iloc[0], 1)

    def test_family_without_any_exported_channel_remains_missing(self):
        result, _ = derive(self.pairs(), {"cracking": ["43"], "absent": ["63"]},
                           blank_deducts="zero")
        self.assertTrue(result.rate_absent.isna().all())
        self.assertTrue(result.family_absent_start.isna().all())

    def test_all_blank_low_pci_preserves_export_gap_exclusion(self):
        pairs = self.pairs("", "")
        pairs["43M deduct_start"] = ""
        pairs["43M deduct_end"] = ""
        result, _ = derive(pairs, {"cracking": ["43"]}, blank_deducts="zero")
        self.assertFalse(result.invalid_deduct_start.iloc[0])
        self.assertTrue(result.export_gap_start.iloc[0])
        self.assertTrue(result.export_gap_end.iloc[0])
        self.assertTrue(result.rate_cracking.isna().all())
        self.assertEqual(result.zero_filled_deduct_cells_start.iloc[0], 2)
        self.assertEqual(result.DetRatePerYear.iloc[0], -0.5)

    def test_all_blank_high_pci_allows_valid_zero_family_rate(self):
        pairs = self.pairs("", "")
        pairs["PCI_start"] = 100
        pairs["PCI_end"] = 99.5
        pairs["43M deduct_start"] = ""
        pairs["43M deduct_end"] = ""
        result, _ = derive(pairs, {"cracking": ["43"]}, blank_deducts="zero")
        self.assertFalse(result.export_gap_start.iloc[0])
        self.assertFalse(result.export_gap_end.iloc[0])
        self.assertEqual(result.rate_cracking.iloc[0], 0)

    def test_invalid_policy_is_rejected(self):
        for value in [None, True, [], "", "guess", "ZERO"]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(SafeError, "^INVALID_BLANK_DEDUCT_POLICY$"):
                    derive(self.pairs(), {"cracking": ["43"]}, blank_deducts=value)

    def test_audit_columns_are_integer_counts(self):
        result, _ = derive(self.pairs(), {"cracking": ["43"]}, blank_deducts="zero")
        for name in ["blank_deduct_cells_start", "blank_deduct_cells_end",
                     "zero_filled_deduct_cells_start", "zero_filled_deduct_cells_end"]:
            self.assertTrue(pd.api.types.is_integer_dtype(result[name].dtype))


if __name__ == "__main__":
    unittest.main()
