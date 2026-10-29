"""Generated exports verify ungraded deduct coverage without double counting."""

import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from temp_support import temporary_directory
from pavement_analytics.autorun import column_map, normalization_map, prepare
from pavement_analytics.curation import build_pairs
from pavement_analytics.distress import derive
from pavement_analytics.features import build
from pavement_analytics.io import load_sources
from pavement_analytics.security import Boundary, SafeError
from test_analysis import COLUMNS, inspections


class UngradedDeductTests(unittest.TestCase):
    def pair(self, **deducts):
        row = {"pair_id": 0, "group_id": "generated", "section_id": "001",
               "material": "PCC", "PCI_start": 95, "PCI_end": 93,
               "delta_days": 730, "DetRatePerYear": -1}
        row.update(deducts)
        return pd.DataFrame([row])

    def test_ungraded_only_values_produce_a_nonzero_family_target(self):
        pairs = self.pair(**{"49 deduct_start": "4", "49 deduct_end": "10"})
        result, targets = derive(pairs, {"family": ["49"]})
        self.assertEqual(targets, ["rate_family"])
        self.assertEqual(result.family_family_start.iloc[0], 4)
        self.assertEqual(result.family_family_end.iloc[0], 10)
        self.assertEqual(result.rate_family.iloc[0], 3)
        self.assertFalse(result.invalid_deduct_start.iloc[0])
        self.assertFalse(result.ambiguous_deduct_start.iloc[0])

    def test_representation_can_switch_across_inspections_without_double_counting(self):
        pairs = self.pair(**{"73 deduct_start": "4", "73 deduct_end": "",
                             "73L deduct_start": "", "73L deduct_end": "8"})
        result, _ = derive(pairs, {"family": ["73"]}, blank_deducts="zero")
        self.assertEqual(result.rate_family.iloc[0], 2)
        for endpoint in ("start", "end"):
            self.assertFalse(result[f"ambiguous_deduct_{endpoint}"].iloc[0])
            self.assertEqual(result[f"ambiguous_deduct_codes_{endpoint}"].iloc[0], 0)
            self.assertEqual(result[f"zero_filled_deduct_cells_{endpoint}"].iloc[0], 1)

    def test_ungraded_growth_is_not_hidden_by_blank_severity_columns(self):
        pairs = self.pair(**{"73 deduct_start": "4", "73 deduct_end": "8",
                             "73L deduct_start": "", "73L deduct_end": "",
                             "63L deduct_start": "2", "63L deduct_end": "2"})
        result, _ = derive(pairs, {"ungraded": ["73"], "graded": ["63"]},
                           blank_deducts="zero")
        self.assertEqual(result.rate_ungraded.iloc[0], 2)
        self.assertEqual(result.rate_graded.iloc[0], 0)

    def test_positive_overlap_excludes_all_deduct_targets_but_preserves_pci_rate(self):
        pairs = self.pair(**{"73 deduct_start": 5, "73 deduct_end": 0,
                             "73L deduct_start": 3, "73L deduct_end": 8,
                             "63L deduct_start": 2, "63L deduct_end": 4})
        result, _ = derive(pairs, {"first": ["73"], "second": ["63"]})
        self.assertTrue(result.ambiguous_deduct_start.iloc[0])
        self.assertTrue(result.invalid_deduct_start.iloc[0])
        self.assertEqual(result.ambiguous_deduct_codes_start.iloc[0], 1)
        self.assertFalse(result.ambiguous_deduct_end.iloc[0])
        self.assertTrue(result.rate_first.isna().all())
        self.assertTrue(result.rate_second.isna().all())
        self.assertTrue(result.family_first_start.isna().all())
        self.assertEqual(result.DetRatePerYear.iloc[0], -1)

    def test_ambiguity_count_counts_codes_not_severity_columns(self):
        pairs = self.pair(**{"49 deduct_start": 4, "49 deduct_end": 4,
                             "49L deduct_start": 1, "49L deduct_end": 0,
                             "49M deduct_start": 2, "49M deduct_end": 0,
                             "73 deduct_start": 5, "73 deduct_end": 0,
                             "73H deduct_start": 2, "73H deduct_end": 8})
        result, _ = derive(pairs, {"family": ["49", "73"]})
        self.assertEqual(result.ambiguous_deduct_codes_start.iloc[0], 2)
        self.assertEqual(result.ambiguous_deduct_codes_end.iloc[0], 0)
        self.assertTrue(pd.api.types.is_integer_dtype(result.ambiguous_deduct_codes_start.dtype))

    def test_zero_alternatives_and_different_codes_are_not_ambiguous(self):
        pairs = self.pair(**{"49 deduct_start": 0, "49 deduct_end": 6,
                             "49L deduct_start": 4, "49L deduct_end": 0,
                             "73H deduct_start": 2, "73H deduct_end": 4})
        result, _ = derive(pairs, {"family": ["49", "73"]})
        self.assertEqual(result.family_family_start.iloc[0], 6)
        self.assertEqual(result.family_family_end.iloc[0], 10)
        self.assertEqual(result.rate_family.iloc[0], 2)
        self.assertFalse(result.ambiguous_deduct_start.iloc[0])
        self.assertFalse(result.ambiguous_deduct_end.iloc[0])

    def test_source_absent_channels_do_not_trigger_overlap(self):
        pairs = self.pair(**{"73 deduct_start": 4, "73 deduct_end": 8,
                             "73L deduct_start": 2, "73L deduct_end": 3,
                             "63 deduct_start": np.nan, "63 deduct_end": np.nan})
        for endpoint in ("start", "end"):
            pairs[f"_deduct_schema_{endpoint}"] = [("73 deduct",)]
        result, _ = derive(pairs, {"family": ["73"], "absent": ["63"]},
                           blank_deducts="zero")
        self.assertEqual(result.rate_family.iloc[0], 2)
        self.assertTrue(result.rate_absent.isna().all())
        self.assertFalse(result.ambiguous_deduct_start.iloc[0])
        self.assertEqual(result.zero_filled_deduct_cells_start.iloc[0], 0)

    def test_blank_ungraded_channel_keeps_missing_default(self):
        pairs = self.pair(**{"73 deduct_start": "", "73 deduct_end": "8"})
        result, _ = derive(pairs, {"family": ["73"]})
        self.assertTrue(result.invalid_deduct_start.iloc[0])
        self.assertTrue(result.rate_family.isna().all())
        self.assertEqual(result.blank_deduct_cells_start.iloc[0], 1)
        self.assertEqual(result.zero_filled_deduct_cells_start.iloc[0], 0)

    def test_ungraded_inventory_and_ambiguity_audit_cannot_be_features(self):
        pairs = self.pair(**{"73 deduct_start": 4, "73 deduct_end": 8,
                             "73 (Slabs)_start": 12})
        result, _ = derive(pairs, {"family": ["73"]})
        for forbidden in ("73 deduct_start", "73 (Slabs)_start",
                          "ambiguous_deduct_start", "ambiguous_deduct_codes_start"):
            with self.subTest(column=forbidden):
                with self.assertRaisesRegex(SafeError, "FORBIDDEN_FEATURE_REQUESTED"):
                    build(result, ["PCI_start", forbidden], ["rate_family"],
                          {"section": "section", "group": "group"})


class UngradedSourceTests(unittest.TestCase):
    def test_mixed_source_schemas_keep_ungraded_values_and_missing_families(self):
        with temporary_directory() as directory:
            root = Path(directory)
            for filename, header, values in (
                ("first.csv", "73 deduct", [4, 8, 10]),
                ("second.csv", "73L deduct", [2, 6, 9]),
            ):
                raw = inspections(1, 1, 3)
                raw = raw.drop(columns=[name for name in raw if name.startswith("_")])
                raw[header] = values
                raw.to_csv(root / filename, index=False)
            loaded, _ = load_sources(root, ["first.csv", "second.csv"], COLUMNS)
            schemas = loaded.groupby("_source_file")["_deduct_schema"].first().to_dict()
            self.assertEqual(schemas, {"first.csv": ("73 deduct",), "second.csv": ("73L deduct",)})
            result, _ = derive(build_pairs(loaded, COLUMNS)[0],
                               {"family": ["73"], "absent": ["49"]}, blank_deducts="zero")
            np.testing.assert_allclose(result.rate_family, np.array([4, 2, 4, 3]) * 365 / 400)
            self.assertTrue(result.rate_absent.isna().all())
            self.assertFalse(result.invalid_deduct_start.any())
            self.assertFalse(result.ambiguous_deduct_start.any())
            self.assertEqual(result.zero_filled_deduct_cells_start.sum(), 0)

    def test_ungraded_header_normalization_and_paver_precedence(self):
        headers = ["SectionUniqueID", "Date", "Inspection Date", "PCI",
                   "Inspected Surface Type", "Network", "NetworkID", "49   DeDuCt"]
        mapping = column_map(headers, {})
        self.assertEqual(mapping["date"], "Date")
        self.assertEqual(mapping["group"], "NetworkID")
        self.assertEqual(normalization_map(headers, mapping)["49   DeDuCt"], "49 deduct")

    def test_duplicate_ungraded_channels_fail_normalization(self):
        with self.assertRaisesRegex(SafeError, "NORMALIZED_HEADERS_COLLIDE"):
            normalization_map(["49 deduct", "49   DEDUCT"], {})

    def test_automatic_by_code_setup_accepts_an_ungraded_only_export(self):
        with temporary_directory() as directory, patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}):
            root = Path(directory)
            public = root / "code"
            public.mkdir()
            source = root / "generated.csv"
            raw = inspections(1, 1, 3)
            raw = raw.drop(columns=[name for name in raw if name.startswith("_")])
            raw[" 49   DeDuCt "] = [4, 8, 10]
            raw.to_csv(source, index=False)
            boundary = Boundary.create(public, root / "workspace")
            with contextlib.redirect_stdout(io.StringIO()):
                config_path = prepare(boundary, source, by_code=True)
            config = json.loads(config_path.read_text())
            self.assertEqual(config["deduct_families"], {"deduct_49": ["49"]})
            normalized = pd.read_csv(boundary.workspace / "inputs/input_0001.csv")
            self.assertIn("49 deduct", normalized)
            setup = json.loads((boundary.workspace / "setup_report.json").read_text())
            self.assertEqual(setup["sources"][0]["observed_deduct_columns"], ["49 deduct"])
            self.assertEqual(setup["sources"][0]["ungraded_deduct_columns"], ["49 deduct"])


if __name__ == "__main__":
    unittest.main()
