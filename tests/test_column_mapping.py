"""Generated schemas exercise safe precedence without operational source records."""
import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from temp_support import temporary_directory
from pavement_analytics.autorun import column_map, prepare
from pavement_analytics.cli import main
from pavement_analytics.security import Boundary, SafeError


class ColumnPrecedenceTests(unittest.TestCase):
    def baseline(self, **alternatives):
        roles = {"section": ["section"], "date": ["date"], "pci": ["PCI"],
                 "surface": ["surface"], "group": ["group_id"]}
        roles.update(alternatives)
        return [header for headers in roles.values() for header in headers]

    def test_unique_section_id_wins_over_short_id_and_label(self):
        headers = self.baseline(section=["section", "SectionID", "SectionUniqueID"])
        self.assertEqual(column_map(headers, {})["section"], "SectionUniqueID")

    def test_inspected_surface_wins_over_generic_surface_and_material(self):
        headers = self.baseline(surface=["Surface", "Material", "Inspected Surface Type"])
        self.assertEqual(column_map(headers, {})["surface"], "Inspected Surface Type")

    def test_group_id_wins_over_network_id_and_label(self):
        headers = self.baseline(group=["Network", "NetworkID", "group_id"])
        self.assertEqual(column_map(headers, {})["group"], "group_id")

    def test_network_id_wins_over_network_label_in_paver_profile(self):
        headers = ["SectionUniqueID", "Date", "PCI", "Inspected Surface Type", "63L deduct",
                   "Network", "NetworkID"]
        self.assertEqual(column_map(headers, {})["group"], "NetworkID")

    def test_generic_network_id_and_label_are_ambiguous(self):
        headers = self.baseline(group=["Network", "NetworkID"])
        with self.assertRaisesRegex(SafeError, "AMBIGUOUS_COLUMNS_USE_PRIVATE_OPTIONS"):
            column_map(headers, {})

    def test_column_order_does_not_change_mapping(self):
        sections = ["SectionUniqueID", "section", "SectionID"]
        surfaces = ["Inspected Surface Type", "Surface", "Material"]
        groups = ["group_id", "NetworkID", "Network"]
        expected = {"section": "SectionUniqueID", "date": "date", "pci": "PCI",
                    "surface": "Inspected Surface Type", "group": "group_id"}
        for section_order in itertools.permutations(sections):
            for surface_order in itertools.permutations(surfaces):
                for group_order in itertools.permutations(groups):
                    headers = self.baseline(section=section_order, surface=surface_order, group=group_order)
                    self.assertEqual(column_map(headers, {}), expected)
                    self.assertEqual(column_map(list(reversed(headers)), {}), expected)

    def test_equivalent_canonical_spellings_still_fail(self):
        for role, duplicate in (
            ("section", ["SectionUniqueID", "Section Unique ID", "section"]),
            ("surface", ["Inspected Surface Type", "InspectedSurfaceType", "Surface"]),
            ("group", ["group_id", "Group ID", "NetworkID"]),
        ):
            with self.subTest(role=role):
                with self.assertRaisesRegex(SafeError, "AMBIGUOUS_COLUMNS_USE_PRIVATE_OPTIONS"):
                    column_map(self.baseline(**{role: duplicate}), {})

    def test_genuinely_ambiguous_generic_headers_still_fail(self):
        for role, headers in (("section", ["section", "SectionID"]),
                              ("surface", ["Surface", "Material"]),
                              ("date", ["Inspection Date", "Survey Date"])):
            with self.subTest(role=role):
                with self.assertRaisesRegex(SafeError, "AMBIGUOUS_COLUMNS_USE_PRIVATE_OPTIONS"):
                    column_map(self.baseline(**{role: headers}), {})

    def test_explicit_overrides_take_precedence(self):
        headers = self.baseline(section=["SectionUniqueID", "SectionID", "section"],
                                surface=["Inspected Surface Type", "Surface", "Material"],
                                group=["group_id", "NetworkID", "Network"])
        overrides = {"section": "section", "surface": "Material", "group": "Network"}
        mapping = column_map(headers, overrides)
        for role, selected in overrides.items():
            self.assertEqual(mapping[role], selected)

    def test_paver_profile_prefers_date_and_pci_with_private_diagnostics(self):
        headers = ["SectionUniqueID", "Date", "Inspection Date", "PCI", "Pavement Condition Index",
                   "Inspected Surface Type", "63L deduct"]
        diagnostics = {}
        selected = column_map(headers, {}, diagnostics=diagnostics)
        self.assertEqual(selected["date"], "Date")
        self.assertEqual(selected["pci"], "PCI")
        self.assertEqual(diagnostics["profile"], "paver")
        self.assertEqual(set(diagnostics["roles"]["date"]["candidates"]), {"Date", "Inspection Date"})
        self.assertEqual(diagnostics["roles"]["date"]["selected"], "Date")
        self.assertTrue(diagnostics["roles"]["date"]["selection"])

    def test_date_and_pci_precedence_are_not_applied_to_generic_tables(self):
        for extra in ["Inspection Date", "Pavement Condition Index"]:
            with self.subTest(extra=extra):
                headers = self.baseline() + [extra]
                with self.assertRaisesRegex(SafeError, "AMBIGUOUS_COLUMNS_USE_PRIVATE_OPTIONS"):
                    column_map(headers, {})

    def test_canonical_core_without_deduct_header_is_still_generic(self):
        headers = ["SectionUniqueID", "Date", "Inspection Date", "PCI", "Inspected Surface Type"]
        diagnostics = {}
        with self.assertRaisesRegex(SafeError, "AMBIGUOUS_COLUMNS_USE_PRIVATE_OPTIONS"):
            column_map(headers, {}, diagnostics=diagnostics)
        self.assertEqual(diagnostics["profile"], "generic")

    def test_explicit_override_resolves_equivalent_canonical_spellings(self):
        headers = self.baseline(section=["SectionUniqueID", "Section Unique ID"])
        self.assertEqual(column_map(headers, {"section": "Section Unique ID"})["section"],
                         "Section Unique ID")


class ColumnNormalizationTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.public = self.root / "code"
        self.public.mkdir()
        self.data = self.root / "private-data"
        self.data.mkdir()
        self.boundary = Boundary.create(self.public, self.root / "work")
        self.environment = patch.dict(os.environ, {"GITHUB_ACTIONS": "false"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def records(self):
        return pd.DataFrame([
            {"SectionUniqueID": "001", "section": "section label", "SectionID": "short label",
             "source_extra_section": "already occupied", "Date": year + "-01-01", "PCI": pci,
             "Inspected Surface Type": "AC", "surface": "generic label", "Material": "other label",
             "group_id": "group key", "group": "group label", "NetworkID": "network key",
             "Network": "network label", "63L deduct": deduct}
            for year, pci, deduct in [("2015", 90, 10), ("2017", 88, 12)]])

    def test_normalization_preserves_unselected_headers_and_values(self):
        original = self.data / "generated.csv"
        self.records().to_csv(original, index=False)
        before = original.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.boundary, original, by_code=True)
        normalized = pd.read_csv(self.boundary.workspace / "inputs/input_0001.csv", dtype=str)
        self.assertEqual(normalized.section.tolist(), ["001", "001"])
        self.assertEqual(normalized.source_extra_section.tolist(), ["already occupied"] * 2)
        self.assertEqual(normalized.source_extra_section_2.tolist(), ["section label"] * 2)
        self.assertEqual(normalized.source_extra_surface.tolist(), ["generic label"] * 2)
        self.assertEqual(normalized.source_extra_group.tolist(), ["group label"] * 2)
        self.assertEqual(normalized.SectionID.tolist(), ["short label"] * 2)
        self.assertEqual(normalized.Material.tolist(), ["other label"] * 2)
        self.assertEqual(normalized.NetworkID.tolist(), ["network key"] * 2)
        self.assertEqual(normalized.Network.tolist(), ["network label"] * 2)
        self.assertEqual(original.read_bytes(), before)
        report = json.loads((self.boundary.workspace / "setup_report.json").read_text())
        source = report["sources"][0]
        self.assertEqual(source["normalization_mapping"]["section"], "source_extra_section_2")
        self.assertEqual(source["normalization_mapping"]["SectionUniqueID"], "section")
        self.assertEqual(source["column_mapping_diagnostics"]["roles"]["section"]["selected"],
                         "SectionUniqueID")
        diagnostic = json.loads((self.boundary.workspace / "column_mapping_report.json").read_text())
        self.assertEqual(diagnostic["schema_version"], 1)
        self.assertEqual(diagnostic["sources"][0]["roles"]["group"]["selected"], "group_id")
        self.assertFalse(diagnostic["sources"][0].get("error_code"))
        self.assertEqual(list(self.public.iterdir()), [])

    def test_duplicate_normalized_deduct_columns_are_not_renamed_away(self):
        frame = self.records()
        frame["63 L deduct"] = frame["63L deduct"]
        original = self.data / "duplicate.csv"
        frame.to_csv(original, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SafeError, "NORMALIZED_HEADERS_COLLIDE"):
                prepare(self.boundary, original, by_code=True)

    def test_mixed_group_identity_semantics_are_rejected(self):
        self.records().to_csv(self.data / "a.csv", index=False)
        self.records().drop(columns="group_id").to_csv(self.data / "b.csv", index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SafeError, "MIXED_GROUPING_REQUIRES_PRIVATE_RECONCILIATION"):
                prepare(self.boundary, self.data, by_code=True)

    def test_complete_paver_export_runs_without_private_options(self):
        original = self.data / "generated.csv"
        rows = pd.DataFrame([
            {"SectionUniqueID": f"S{group:02d}", "SectionID": "unused-short-label",
             "section": "unused-section-label", "Date": f"{2015 + visit * 2}-01-01",
             "Inspection Date": "unused-date", "PCI": 94 - group - visit * (1 + group * 0.5),
             "Pavement Condition Index": "unused-condition", "Inspected Surface Type": "AC",
             "Surface": "unused-surface", "Material": "unused-material", "NetworkID": f"G{group:02d}",
             "Network": "unused-network-label", "43L deduct": 3 + visit * (0.5 + group * 0.2)}
            for group in range(8) for visit in range(3)])
        rows.to_csv(original, index=False)
        workspace = self.root / "complete-analysis"
        console = io.StringIO()
        with patch("pavement_analytics.cli.source_root", return_value=self.public):
            with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
                status = main(["analyze", str(original), "--workspace", str(workspace)])
        self.assertEqual(status, 0, console.getvalue())
        self.assertEqual(console.getvalue().strip(), "PRIVATE_RUN_COMPLETE")
        report = json.loads((workspace / "results/run_report.json").read_text())
        self.assertEqual(report["accounting"]["pair_rows"], 16)
        fitted = [slot for slot in report["slots"] if slot["status"] in {"champion", "no_model"}]
        self.assertEqual({slot["target"] for slot in fitted}, {"DetRatePerYear", "rate_ac_thermal_crack"})
        config = json.loads((workspace / "run.json").read_text())
        self.assertEqual(config["features"], ["PCI_start"])
        folds = pd.read_csv(workspace / "results/materials/AC/rate_ac_thermal_crack/folds.csv")
        holdout = set(folds.loc[folds.partition.eq("holdout"), "group_id"])
        development = set(folds.loc[folds.partition.eq("development"), "group_id"])
        self.assertTrue(holdout)
        self.assertTrue(development)
        self.assertFalse(holdout & development)
        self.assertTrue((workspace / "results/dashboard.html").is_file())
        self.assertTrue((workspace / "results/target_summary.csv").is_file())

    def test_later_file_failure_retains_prior_mapping_diagnostics(self):
        self.records().to_csv(self.data / "a.csv", index=False)
        self.records().drop(columns="SectionUniqueID").to_csv(self.data / "b.csv", index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SafeError, "AMBIGUOUS_COLUMNS_USE_PRIVATE_OPTIONS"):
                prepare(self.boundary, self.data, by_code=True)
        diagnostic = json.loads((self.boundary.workspace / "column_mapping_report.json").read_text())
        self.assertEqual(len(diagnostic["sources"]), 2)
        self.assertEqual(diagnostic["sources"][0]["roles"]["section"]["selected"], "SectionUniqueID")
        self.assertFalse(diagnostic["sources"][0].get("error_code"))
        self.assertEqual(diagnostic["sources"][1]["error_code"], "AMBIGUOUS_COLUMNS_USE_PRIVATE_OPTIONS")

    def test_failed_mapping_writes_structured_private_report_without_console_values(self):
        frame = self.records().drop(columns="SectionUniqueID")
        original = self.data / "private-generated.csv"
        frame.to_csv(original, index=False)
        workspace = self.root / "failed-analysis"
        console = io.StringIO()
        with patch("pavement_analytics.cli.source_root", return_value=self.public):
            with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
                status = main(["analyze", str(original), "--workspace", str(workspace), "--by-code"])
        self.assertEqual(status, 2)
        self.assertEqual(console.getvalue().strip(), "AMBIGUOUS_COLUMNS_USE_PRIVATE_OPTIONS")
        diagnostic = json.loads((workspace / "column_mapping_report.json").read_text())
        self.assertEqual(diagnostic["schema_version"], 1)
        source = diagnostic["sources"][0]
        self.assertEqual(source["original_path"], str(original))
        self.assertEqual(source["error_code"], "AMBIGUOUS_COLUMNS_USE_PRIVATE_OPTIONS")
        self.assertEqual(set(source["roles"]["section"]["candidates"]), {"section", "SectionID"})
        self.assertIsNone(source["roles"]["section"]["selected"])
        self.assertEqual(source["override_template"], {"columns": {"section": None}})
        for private_value in (str(original), str(workspace), "SectionID", "section label", "group key"):
            self.assertNotIn(private_value, console.getvalue())
        self.assertFalse((workspace / "results").exists())
        self.assertEqual(list(self.public.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
