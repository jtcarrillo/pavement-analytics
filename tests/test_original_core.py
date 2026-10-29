"""Regression contracts for the preserved original paired-inspection stages."""
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pavement_analytics.original import check_cli_paths, location, pairs, prepare, rates, split
from pavement_analytics.security import SafeError, source_root
from temp_support import temporary_directory


def inspections(values, *, construction=None, source="source_a.csv", section="001"):
    result = pd.DataFrame({"SectionUniqueID": [section] * len(values),
                           "Date": pd.date_range("2000-01-01", periods=len(values), freq="YS"),
                           "PCI": values, "_source_file": source,
                           "64L deduct": np.arange(len(values), dtype=float)})
    if construction is not None:
        result["IsConstruction"] = construction
    return result


def run_pairs(frame, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return pairs.build_pairs(frame, "SectionUniqueID", "PCI", "Date", **kwargs)


class OriginalPairingTests(unittest.TestCase):
    def test_increase_resets_the_anchor(self):
        paired, stats, summary = run_pairs(inspections([100, 75, 50, 75, 50]))
        self.assertEqual(list(zip(paired.PCI_start, paired.PCI_end)), [(100, 75), (75, 50), (75, 50)])
        self.assertEqual(summary["total_increases_skipped"], 1)
        self.assertEqual(stats.pairs_emitted.tolist(), [3])
        self.assertEqual(paired.SectionUniqueID.unique().tolist(), ["001"])

    def test_flat_reinspection_depends_on_original_construction_rule(self):
        raw = inspections([100, 100, 50], construction=[False, False, False])
        paired, _, summary = run_pairs(raw)
        self.assertEqual(paired.delta_PCI.tolist(), [0, -50])
        self.assertEqual(summary["total_flat_pairs_preserved"], 1)
        raw["IsConstruction"] = [False, True, False]
        paired, _, summary = run_pairs(raw)
        self.assertEqual(paired.PCI_start.tolist(), [100])
        self.assertEqual(paired.start_date.dt.year.tolist(), [2001])
        self.assertEqual(summary["total_duplicates_collapsed"], 1)
        paired, _, _ = run_pairs(raw.drop(columns="IsConstruction"))
        self.assertEqual(paired.start_date.dt.year.tolist(), [2001])

    def test_same_day_pairs_are_counted_but_never_emitted(self):
        raw = inspections([90, 70, 60], construction=[False] * 3)
        raw.loc[1, "Date"] = raw.loc[0, "Date"]
        paired, _, summary = run_pairs(raw)
        self.assertEqual(paired.PCI_start.tolist(), [70])
        self.assertEqual(paired.PCI_end.tolist(), [60])
        self.assertEqual(summary["total_nonpositive_intervals_skipped"], 1)
        self.assertTrue(paired.delta_days.gt(0).all())

    def test_sources_isolate_reused_section_identifiers(self):
        raw = pd.concat([inspections([100, 90], source="source_a.csv"),
                         inspections([60, 50], source="source_b.csv")], ignore_index=True)
        paired, _, summary = run_pairs(raw)
        self.assertEqual(summary["sections_total"], 2)
        self.assertEqual(paired.delta_PCI.tolist(), [-10, -10])
        self.assertTrue(paired._source_file_start.eq(paired._source_file_end).all())

    def test_invalid_rows_and_empty_population_remain_audited(self):
        raw = inspections([90, 110, -1, "bad", 80, 70])
        raw.loc[4, "Date"] = pd.NaT
        raw.loc[5, "SectionUniqueID"] = ""
        paired, _, summary = run_pairs(raw)
        self.assertTrue(paired.empty)
        self.assertEqual(summary["rows_dropped_invalid"], 5)
        self.assertIn("64L deduct_end", paired)
        self.assertEqual(summary["sections_without_pairs"], 1)

    def test_surface_buckets_preserve_original_classification(self):
        frame = pd.DataFrame({"Inspected Surface Type": [" pcc ", "APC", "ac", "AAC", "ACC", None, "unknown"]})
        buckets = split.categorize(frame, "Inspected Surface Type", False)
        self.assertEqual({key: len(value) for key, value in buckets.items()},
                         {"PCC": 1, "APC": 1, "AC_AAC_ACC": 3, "OTHER": 2})
        self.assertEqual(sum(map(len, buckets.values())), len(frame))


class OriginalClimatePreparationTests(unittest.TestCase):
    def test_rates_and_date_windows_use_each_original_pair(self):
        frame = pd.DataFrame({"delta_PCI": [-10, 0, -5, -5], "delta_days": [730, 365, 0, None],
                              "start_date": ["2000-01-01", "2000-02-01", "bad", "2000-01-01"],
                              "end_date": ["2002-01-01", "2001-02-01", "2001-01-01", None]})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = rates.add_deterioration_rate(frame)
        self.assertEqual(result.DetRatePerYear.iloc[:2].tolist(), [-5, 0])
        self.assertTrue(result.DetRatePerYear.iloc[2:].isna().all())
        self.assertNotIn("DetRatePerYear", frame)
        self.assertEqual(rates.build_dates_column(frame, "start_date", "end_date").tolist(),
                         ["[2000-01-01, 2002-01-01]", "[2000-02-01, 2001-02-01]", "", ""])

    def test_original_location_forms_and_first_nonnull_lookup(self):
        for text in ["1.0 -2.0", "1.0, -2.0", "1.0N 2.0W", "1.0 N 2.0 W", "N 1.0 W 2.0", "N1.0 W2.0"]:
            self.assertEqual(location.parse_lat_lon(text), (1.0, -2.0), text)
        self.assertEqual(location.parse_lat_lon("unparseable"), (None, None))
        with temporary_directory() as temporary:
            path = Path(temporary) / "location.csv"
            pd.DataFrame({"LOCATION(DEGREES)": [None] * 55 + ["1.0, -2.0", "0, 0"]}).to_csv(path, index=False)
            self.assertEqual(location.load_location_from_csv(path, "LOCATION(DEGREES)"), "1.0, -2.0")
            result = location.build_location_map({"source.csv": path.name}, path.parent, "LOCATION(DEGREES)", "fail")
            self.assertEqual(result, {"source.csv": "1.0, -2.0"})

    def test_bundled_worker_preserves_interval_argument_contract(self):
        with temporary_directory() as temporary:
            work = Path(temporary)
            frame = pd.DataFrame({"pair": [2, 1], "Dates": ["[2000-01-01, 2001-01-01]"] * 2})
            worker = Path(rates.__file__).with_name("era5.py")
            def subprocess_run(command):
                self.assertEqual(command[1:3], ["-m", "pavement_analytics.original.era5"])
                self.assertEqual(command[command.index("--dates-col") + 1], "Dates")
                self.assertEqual(command[command.index("--construction-year-col") + 1], "build_year")
                self.assertEqual(command[-2:], ["--pav-reliability", "98"])
                incoming = pd.read_csv(command[command.index("--input") + 1])
                incoming["MeanTemp_C"] = [2.0, 1.0]
                incoming.to_csv(command[command.index("--output") + 1], index=False)
                return SimpleNamespace(returncode=0)
            with patch.object(rates.subprocess, "run", side_effect=subprocess_run), contextlib.redirect_stdout(io.StringIO()):
                result = rates.run_era5_stage(frame, worker, "latitude", "longitude", work / "cache",
                                             "build_year", "--pav-reliability 98", work)
            self.assertEqual(result.pair.tolist(), [2, 1])
            self.assertEqual(result.MeanTemp_C.tolist(), [2.0, 1.0])


class OriginalFeaturePreparationTests(unittest.TestCase):
    def test_original_role_and_leakage_rules_preserve_climate_families(self):
        frame = pd.DataFrame({
            "fam_pcc_load deduct_end": [2, 5, 8], "other deduct_end": [3, 5, 7],
            "delta_days": [365, 500, 700], "_source_file_start": ["a", "a", "b"],
            "SectionUniqueID": ["001", "002", "003"], "PCI_start": [90, 80, 70],
            "PCI_end": [85, 70, 60], "64L deduct_start": [2, 4, 6],
            "CoolEvents6h": [1, 3, 5], "RainflowCycles10C": [2, 4, 6],
            "PGHigh": [52, 58, 64], "ThermalAging": [1., 2., 3.],
            "IntervalYears": [1., 2., 3.], "start_date": ["2000", "2001", "2002"],
            "latitude": [0., 1., 2.], "_source_row_start": [2, 3, 4], "constant": [1, 1, 1],
        })
        audit = prepare.classify(frame, "fam_pcc_load deduct_end", "delta_days", "_source_file_start",
                                 "SectionUniqueID", False, False, .95, set()).set_index("column")
        for col in ["PCI_start", "CoolEvents6h", "RainflowCycles10C", "PGHigh", "ThermalAging"]:
            self.assertEqual(audit.loc[col, "disposition"], "KEEP", col)
        self.assertEqual(audit.loc["fam_pcc_load deduct_end", "disposition"], "TARGET")
        self.assertEqual(audit.loc["delta_days", "disposition"], "WEIGHT")
        self.assertEqual(audit.loc["SectionUniqueID", "disposition"], "PARTITION")
        for col in ["other deduct_end", "PCI_end", "64L deduct_start", "IntervalYears", "start_date",
                    "latitude", "_source_row_start", "constant"]:
            self.assertEqual(audit.loc[col, "disposition"], "DROP", col)

    def test_standalone_boundary_rejects_public_and_versioned_outputs(self):
        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "1", "GITHUB_ACTIONS": "false"}):
            with self.assertRaisesRegex(SafeError, "OUTSIDE_PUBLIC"):
                check_cli_paths(inputs=(source_root() / "README.md",))
            with temporary_directory() as temporary:
                root = Path(temporary)
                (root / ".git").mkdir()
                with self.assertRaisesRegex(SafeError, "VERSIONED"):
                    check_cli_paths(outputs=(root / "result.csv",))

    def test_standalone_boundary_requires_opt_in_and_refuses_actions(self):
        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "", "GITHUB_ACTIONS": "false"}):
            with self.assertRaisesRegex(SafeError, "OPT_IN"):
                check_cli_paths()
        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "1", "GITHUB_ACTIONS": "true"}):
            with self.assertRaisesRegex(SafeError, "GITHUB_ACTIONS"):
                check_cli_paths()


if __name__ == "__main__":
    unittest.main()
