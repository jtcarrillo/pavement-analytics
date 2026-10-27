"""Generated records verify family sums and structural column absence."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from pavement_analytics.curation import build_pairs
from pavement_analytics.distress import derive
from pavement_analytics.families import DEFAULT_FAMILIES, default_families
from pavement_analytics.io import load_sources
from pavement_analytics.security import SafeError
from test_analysis import COLUMNS, inspections


class FamilyTests(unittest.TestCase):
    def test_default_families_partition_codes_and_preserve_unknowns(self):
        result = default_families(["63", "01"])
        codes = [c for values in result.values() for c in values]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertEqual(result["deduct_01"], ["01"])
        result["pcc_thermal"].append("00")
        self.assertNotIn("00", DEFAULT_FAMILIES["pcc_thermal"])

    def test_heterogeneous_exports_do_not_invalidate_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, code, material in [("a.csv", "63", "PCC"), ("b.csv", "43", "AC")]:
                raw = inspections(1, 1, 3)
                raw = raw.drop(columns=[c for c in raw if c.startswith("_")])
                raw[COLUMNS["surface"]] = material
                raw[code + "L deduct"] = [3, 4, 6]
                raw[code + "M deduct"] = [2, 4, 4]
                raw.to_csv(root / name, index=False)
            loaded, _ = load_sources(root, ["a.csv", "b.csv"], COLUMNS)
            pairs = build_pairs(loaded, COLUMNS)[0]
            derived, _ = derive(pairs, default_families(["43", "63"]))
            for material, own, absent in [("PCC", "pcc_thermal", "ac_thermal_crack"),
                                           ("AC", "ac_thermal_crack", "pcc_thermal")]:
                sub = derived.loc[derived.material.eq(material)]
                np.testing.assert_allclose(sub["rate_" + own], np.array([3, 2]) * 365 / 400)
                self.assertTrue(sub["rate_" + absent].isna().all())
                self.assertFalse(sub.invalid_deduct_start.any())
            # A blank in a column actually exported is missing, never structural zero.
            loaded.loc[0, "63M deduct"] = ""
            damaged, _ = derive(build_pairs(loaded, COLUMNS)[0], default_families(["43", "63"]))
            self.assertTrue(damaged.loc[damaged.material.eq("PCC"), "rate_pcc_thermal"].isna().iloc[0])
            self.assertTrue(damaged.loc[damaged.material.eq("AC"), "rate_ac_thermal_crack"].notna().all())

    def test_empty_and_overlapping_families_fail(self):
        pairs = build_pairs(inspections(), COLUMNS)[0]
        for families in [{"empty": []}, {"one": [63], "two": [63]}, []]:
            with self.subTest(families=families), self.assertRaises(SafeError):
                derive(pairs, families)


if __name__ == "__main__":
    unittest.main()
