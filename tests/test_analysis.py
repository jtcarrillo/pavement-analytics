"""All records are synthesized in memory or in temporary directories."""

from datetime import date, timedelta
from temp_support import temporary_directory
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from pavement_analytics.curation import build_pairs
from pavement_analytics.distress import derive
from pavement_analytics.features import build
from pavement_analytics.io import apply_corrections, load_sources, read_table
from pavement_analytics.metrics import condition_predict, grouped_folds, score
from pavement_analytics.pci import calculate, shapley_loss
from pavement_analytics.ranking import rank
from pavement_analytics.security import SafeError

COLUMNS = {"section": "SectionUniqueID", "date": "Date", "pci": "PCI",
           "surface": "Inspected Surface Type", "group": "group_id"}


def inspections(groups=8, sections=2, steps=3):
    records = []
    for group in range(groups):
        for section in range(sections):
            for step in range(steps):
                records.append({"SectionUniqueID": f"{section:04d}",
                                "Date": (date(2000, 1, 1) + timedelta(days=400 * step)).isoformat(),
                                "PCI": str(95 - (1 + group % 3 + section) * step),
                                "Inspected Surface Type": "PCC", "group_id": f"g{group:03d}",
                                "_source_file": f"generated_{group}.csv", "_source_row": section * steps + step + 2,
                                "_source_sha256": "synthetic"})
    return pd.DataFrame(records)


class PairingTests(unittest.TestCase):
    def test_pairing_never_crosses_sources_and_reconciles(self):
        frame = inspections()
        pairs, rejected, transitions, counts = build_pairs(frame, COLUMNS)
        self.assertEqual(len(pairs), 8 * 2 * 2)
        self.assertTrue((pairs._source_file_start == pairs._source_file_end).all())
        self.assertEqual(counts["input_rows"], counts["accepted_rows"] + counts["rejected_rows"])
        self.assertEqual(counts["accepted_rows"], counts["transition_count"] + counts["initial_anchors"])
        self.assertEqual(len(rejected), 0)
        self.assertTrue(np.isfinite(pairs.PCI_start).all())

    def test_increase_resets_anchor_and_does_not_bridge(self):
        raw = inspections(1, 1, 4)
        raw["PCI"] = ["90", "85", "98", "95"]
        pairs, _, audit, _ = build_pairs(raw, COLUMNS)
        self.assertEqual(list(pairs.PCI_start), [90, 98])
        self.assertEqual(list(pairs.PCI_end), [85, 95])
        self.assertEqual(audit.reason.tolist(), ["paired", "increase_reset", "paired"])

    def test_same_day_and_short_intervals_advance_anchor(self):
        raw = inspections(1, 1, 4)
        raw["Date"] = [date(2000, 1, 1) + timedelta(days=d) for d in [0, 0, 20, 420]]
        raw["Date"] = raw.Date.astype(str)
        pairs, _, audit, _ = build_pairs(raw, COLUMNS)
        self.assertEqual(pairs._source_row_start.tolist(), [4])
        self.assertEqual(audit.reason.tolist(), ["same_day_conflict", "below_minimum_interval", "paired"])

    def test_construction_resets_and_invalid_rows_are_audited(self):
        raw = inspections(1, 1, 4)
        raw["IsConstruction"] = ["0", "1", "0", "0"]
        raw.loc[3, "PCI"] = "101"
        pairs, rejected, audit, counts = build_pairs(raw, {**COLUMNS, "construction": "IsConstruction"})
        self.assertEqual(len(pairs), 1)
        self.assertEqual(rejected.reason.tolist(), ["invalid_pci"])
        self.assertEqual(audit.reason.tolist(), ["construction_reset", "paired"])


class IOTests(unittest.TestCase):
    def test_leading_zero_identity_and_duplicate_header_rejection(self):
        with temporary_directory() as folder:
            path = Path(folder) / "generated.csv"
            inspections(1).drop(columns=[c for c in inspections(1) if c.startswith("_")]).to_csv(path, index=False)
            loaded, manifest = load_sources(Path(folder), [path.name], COLUMNS)
            self.assertEqual(loaded.SectionUniqueID.iloc[0], "0000")
            self.assertEqual(loaded._source_row.iloc[0], 2)
            path.write_text("PCI, PCI\n1,2\n")
            with self.assertRaisesRegex(SafeError, "DUPLICATE_HEADERS"):
                read_table(path)

    def test_guarded_correction_and_stale_value_rejection(self):
        raw = inspections(1, 1, 2)
        row = {"correction_id": "c1", "source_file": raw._source_file.iloc[0], "source_row": "2",
               "column": "PCI", "expected": "95", "replacement": "94", "reason": "generated test",
               "reference": "test", "author": "test", "date": "2000-01-01"}
        with temporary_directory() as folder:
            path = Path(folder) / "changes.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            corrected, log = apply_corrections(raw, path)
            self.assertEqual(corrected.PCI.iloc[0], "94")
            self.assertEqual(raw.PCI.iloc[0], "95")
            with self.assertRaisesRegex(SafeError, "EXPECTATION_MISMATCH"):
                apply_corrections(corrected, path)


class AnalysisTests(unittest.TestCase):
    def test_feature_allowlist_cannot_override_outcome_or_identity_rules(self):
        pairs = build_pairs(inspections(), COLUMNS)[0]
        for forbidden in ["PCI_end", "delta_PCI", "group_id_start", "SectionUniqueID_start"]:
            with self.subTest(column=forbidden), self.assertRaisesRegex(SafeError, "FORBIDDEN_FEATURE"):
                build(pairs, ["PCI_start", forbidden], ["DetRatePerYear"], COLUMNS)
        table, audit = build(pairs, ["PCI_start"], ["DetRatePerYear"], COLUMNS)
        self.assertEqual(set(table), {"pair_id", "group_id", "delta_days", "DetRatePerYear", "PCI_start"})
        self.assertEqual(len(audit), len(pairs.columns))

    def test_deduct_rates_and_export_gap_exclusion(self):
        pairs = build_pairs(inspections(1), COLUMNS)[0]
        pairs["63L deduct_start"] = 4.0
        pairs["63L deduct_end"] = 8.0
        derived, targets = derive(pairs, {"cracking": [63]})
        self.assertAlmostEqual(derived.rate_cracking.iloc[0], 4 * 365 / 400)
        pairs.loc[0, "63L deduct_start"] = 0
        derived, _ = derive(pairs, {"cracking": [63]})
        self.assertTrue(np.isnan(derived.rate_cracking.iloc[0]))
        self.assertTrue(np.isfinite(derived.DetRatePerYear.iloc[0]))
        pairs.loc[1, "63L deduct_end"] = -1
        derived, _ = derive(pairs, {"cracking": [63]})
        self.assertTrue(derived.invalid_deduct_end.iloc[1])

    def test_metrics_and_condition_bins_have_known_answers(self):
        scores = score([0, 10], [2, 6], [1, 3])
        self.assertAlmostEqual(scores["wMAE"], 3.5)
        self.assertAlmostEqual(scores["wRMSE"], np.sqrt(13))
        predicted = condition_predict([90, 90, 30], [-1, -5, -7], [1, 3, 1], [90, 30])
        np.testing.assert_allclose(predicted, [-4, -7])
        with self.assertRaises(SafeError):
            score([1], [1], [0])

    def test_grouped_folds_are_disjoint_repeatable_and_complete(self):
        groups = np.repeat([f"g{i}" for i in range(7)], np.arange(1, 8))
        folds = list(grouped_folds(groups, 3, 9))
        again = list(grouped_folds(groups, 3, 9))
        seen = []
        for (train, test), (_, repeated) in zip(folds, again):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            np.testing.assert_array_equal(test, repeated)
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(len(groups))))

    def test_identical_groups_tie_in_ranking(self):
        raw = inspections(2, 1, 3)
        raw["PCI"] = [95, 90, 85] * 2
        pairs = build_pairs(raw, COLUMNS)[0]
        ranks = rank(pairs)
        np.testing.assert_allclose(ranks["rank"], [1.5, 1.5])
        self.assertTrue(ranks.stable.all())

    def test_pci_requires_curves_and_shapley_is_additive(self):
        self.assertEqual(calculate([]), 100)
        self.assertEqual(calculate([20]), 80)
        with self.assertRaisesRegex(SafeError, "APPROVED_CDV"):
            calculate([10, 20])
        # Artificial mathematical curves; not a standard, not reference data.
        profile = {"name": "generated-test", "q_threshold": 2, "m_denominator": 98, "m_cap": 10,
                   "curves": {str(q): [[0, 0], [100, 100], [1000, 100]] for q in range(2, 11)}}
        losses = shapley_loss({"a": [10], "b": [20]}, profile)
        self.assertAlmostEqual(sum(losses.values()), 100 - calculate([10, 20], profile))
        profile["curves"]["2"] = [[0, 0], [5, 5]]
        with self.assertRaisesRegex(SafeError, "DOMAIN"):
            calculate([10, 20], profile)


if __name__ == "__main__":
    unittest.main()
