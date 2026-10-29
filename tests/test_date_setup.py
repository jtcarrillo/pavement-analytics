"""Generated exports exercise date setup and private failure diagnostics."""

import contextlib
import datetime
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from temp_support import temporary_directory
from pavement_analytics.autorun import prepare
from pavement_analytics.cli import main
from pavement_analytics.security import Boundary, SafeError


class DateSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.public = self.root / "code"
        self.public.mkdir()
        self.data = self.root / "private-data"
        self.data.mkdir()
        self.environment = patch.dict(os.environ, {"GITHUB_ACTIONS": "false"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.options = self.data / "options.json"
        self.options.write_text(json.dumps({"models": False}))

    def records(self, dates):
        return pd.DataFrame([
            {"SectionUniqueID": f"S{section:02d}", "Date": date,
             "PCI": 90 - visit * 3, "Inspected Surface Type": "AC",
             "43L deduct": 4 + visit, "IsConstruction": "no"}
            for section in range(2) for visit, date in enumerate(dates)])

    def write_input(self, rows, name="generated.csv"):
        source = self.data / name
        if source.suffix == ".xlsx":
            rows.to_excel(source, index=False)
        else:
            rows.to_csv(source, index=False)
        return source

    def run_cli(self, source, workspace, *extra):
        console = io.StringIO()
        with patch("pavement_analytics.cli.source_root", return_value=self.public):
            with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
                status = main(["analyze", str(source), "--workspace", str(workspace),
                               "--options", str(self.options), "--by-code", *extra])
        return status, console.getvalue().strip()

    def test_common_export_dates_produce_pairs_and_complete_reports(self):
        formats = {
            "iso_timestamp": [f"{year}-12-31 00:00:00" for year in (2015, 2017, 2019)],
            "month_first": [f"12/31/{year} 12:00:00 AM" for year in (2015, 2017, 2019)],
            "day_first": [f"31/12/{year}" for year in (2015, 2017, 2019)],
        }
        for label, dates in formats.items():
            with self.subTest(label=label):
                source = self.write_input(self.records(dates), label + ".csv")
                before = source.read_bytes()
                workspace = self.root / label
                status, console = self.run_cli(source, workspace)
                self.assertEqual((status, console), (0, "PRIVATE_RUN_COMPLETE"))
                self.assertEqual(source.read_bytes(), before)
                report = json.loads((workspace / "results/run_report.json").read_text())
                self.assertEqual(report["accounting"]["accepted_rows"], 6)
                self.assertEqual(report["accounting"]["pair_rows"], 4)
                pairs = pd.read_csv(workspace / "results/dataset.csv")
                self.assertEqual(set(pairs.start_date), {"2015-12-31", "2017-12-31"})
                self.assertEqual(set(pairs.end_date), {"2017-12-31", "2019-12-31"})
                self.assertTrue((workspace / "results/dashboard.html").is_file())
                self.assertTrue((workspace / "results/target_summary.csv").is_file())
                setup = json.loads((workspace / "results/setup_report.json").read_text())
                self.assertEqual(setup["sources"][0]["date_parsing"]["mode"], "automatic")
                normalized = pd.read_csv(workspace / "inputs/input_0001.csv", dtype=str)
                preservation = setup["sources"][0]["date_parsing"]["preservation_column"]
                self.assertEqual(list(normalized[preservation]), dates * 2)

    def test_native_excel_dates_remain_accepted(self):
        dates = [datetime.datetime(year, 12, 31) for year in (2015, 2017, 2019)]
        source = self.write_input(self.records(dates), "generated.xlsx")
        workspace = self.root / "excel"
        self.assertEqual(self.run_cli(source, workspace), (0, "PRIVATE_RUN_COMPLETE"))
        report = json.loads((workspace / "results/run_report.json").read_text())
        self.assertEqual(report["accounting"]["pair_rows"], 4)

    def test_original_date_preservation_does_not_overwrite_source_fields(self):
        dates = ["2015-12-31 00:00:00", "2017-12-31 00:00:00"]
        rows = self.records(dates)
        rows["source_extra_date"] = "existing source field"
        rows["source_extra_date_2"] = "second existing source field"
        source = self.write_input(rows)
        boundary = Boundary.create(self.public, self.root / "prepared")
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(boundary, source, by_code=True)
        normalized = pd.read_csv(boundary.workspace / "inputs/input_0001.csv", dtype=str)
        self.assertEqual(list(normalized.source_extra_date), list(rows.source_extra_date))
        self.assertEqual(list(normalized.source_extra_date_2), list(rows.source_extra_date_2))
        self.assertEqual(list(normalized.source_extra_date_3), dates * 2)
        self.assertEqual(list(normalized.date), ["2015-12-31", "2017-12-31"] * 2)

    def test_ambiguous_dates_fail_with_private_report_and_no_console_values(self):
        source = self.write_input(self.records(["01/02/2015", "01/02/2017"]))
        workspace = self.root / "ambiguous"
        self.assertEqual(self.run_cli(source, workspace),
                         (2, "AMBIGUOUS_DATE_FORMAT_USE_DATE_FORMAT"))
        report = json.loads((workspace / "date_parsing_report.json").read_text())
        self.assertEqual(len(report["sources"]), 1)
        self.assertFalse((workspace / "results").exists())
        self.assertTrue((workspace / "originals/input_0001.csv").is_file())

    def test_inference_is_scoped_to_each_file(self):
        self.write_input(self.records(["12/31/2015", "12/31/2017"]), "a.csv")
        self.write_input(self.records(["01/02/2015", "01/02/2017"]), "b.csv")
        workspace = self.root / "per-file"
        self.assertEqual(self.run_cli(self.data, workspace),
                         (2, "AMBIGUOUS_DATE_FORMAT_USE_DATE_FORMAT"))
        report = json.loads((workspace / "date_parsing_report.json").read_text())
        self.assertEqual(len(report["sources"]), 2)

    def test_explicit_cli_format_resolves_ambiguous_dates(self):
        dates = ["01/02/2015", "01/02/2017", "01/02/2019"]
        source = self.write_input(self.records(dates))
        for label, date_format, expected_month_day in (
            ("month", "%m/%d/%Y", "01-02"),
            ("day", "%d/%m/%Y", "02-01"),
        ):
            with self.subTest(label=label):
                workspace = self.root / label
                self.assertEqual(self.run_cli(source, workspace, "--date-format", date_format),
                                 (0, "PRIVATE_RUN_COMPLETE"))
                config = json.loads((workspace / "run.json").read_text())
                self.assertEqual(config["date_format"], date_format)
                pairs = pd.read_csv(workspace / "results/dataset.csv")
                self.assertEqual(set(pairs.start_date),
                                 {f"2015-{expected_month_day}", f"2017-{expected_month_day}"})
                report = json.loads((workspace / "date_parsing_report.json").read_text())
                self.assertEqual(report["sources"][0]["mode"], "explicit")

    def test_options_format_remains_supported_and_identical_cli_format_is_allowed(self):
        dates = ["02.01.2015", "02.01.2017", "02.01.2019"]
        source = self.write_input(self.records(dates))
        self.options.write_text(json.dumps({"models": False, "date_format": "%d.%m.%Y"}))
        for name, extra in (("options-only", ()), ("same-format", ("--date-format", "%d.%m.%Y"))):
            with self.subTest(name=name):
                workspace = self.root / name
                self.assertEqual(self.run_cli(source, workspace, *extra), (0, "PRIVATE_RUN_COMPLETE"))
                pairs = pd.read_csv(workspace / "results/dataset.csv")
                self.assertEqual(set(pairs.start_date), {"2015-01-02", "2017-01-02"})

    def test_conflicting_explicit_formats_are_rejected(self):
        source = self.write_input(self.records(["01/02/2015", "01/02/2017"]))
        self.options.write_text(json.dumps({"models": False, "date_format": "%d/%m/%Y"}))
        self.assertEqual(self.run_cli(source, self.root / "conflict", "--date-format", "%m/%d/%Y"),
                         (2, "DATE_FORMAT_OPTIONS_CONFLICT"))

    def test_invalid_format_option_is_a_safe_error(self):
        source = self.write_input(self.records(["2015-12-31", "2017-12-31"]))
        for index, value in enumerate(["", 7, None]):
            with self.subTest(value=value):
                self.options.write_text(json.dumps({"date_format": value}))
                boundary = Boundary.create(self.public, self.root / f"bad-format-{index}")
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(SafeError, "INVALID_DATE_FORMAT"):
                        prepare(boundary, source, self.options, by_code=True)

    def test_all_invalid_dates_keep_pairing_audits_and_specific_error(self):
        rows = self.records(["invalid-date-a", "invalid-date-b"])
        source = self.write_input(rows)
        workspace = self.root / "invalid"
        self.assertEqual(self.run_cli(source, workspace), (2, "NO_VALID_DATES_CHECK_DATE_FORMAT"))
        reports = list((workspace / "failed").glob("run-*/pairing_report.json"))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text())
        self.assertEqual(report["accounting"]["input_rows"], len(rows))
        self.assertEqual(report["accounting"]["accepted_rows"], 0)
        self.assertEqual(report["rejected_reasons"], {"invalid_date": len(rows)})
        self.assertEqual(report["transition_reasons"], {})
        rejected = pd.read_csv(reports[0].with_name("rejected_rows.csv"), dtype=str)
        self.assertEqual(list(rejected.source_extra_date), list(rows.Date))

    def test_valid_single_inspection_histories_keep_no_valid_pairs_error(self):
        source = self.write_input(self.records(["2015-12-31"]))
        workspace = self.root / "singletons"
        self.assertEqual(self.run_cli(source, workspace), (2, "NO_VALID_PAIRS"))
        report_path = next((workspace / "failed").glob("run-*/pairing_report.json"))
        report = json.loads(report_path.read_text())
        self.assertEqual(report["accounting"]["accepted_rows"], 2)
        self.assertEqual(report["accounting"]["transition_count"], 0)

    def test_is_construction_resets_pairing_after_date_normalization(self):
        rows = self.records(["2015-12-31 00:00:00", "2017-12-31 00:00:00", "2019-12-31 00:00:00"])
        rows.loc[rows.Date.str.startswith("2017"), "IsConstruction"] = "yes"
        source = self.write_input(rows)
        workspace = self.root / "construction"
        self.assertEqual(self.run_cli(source, workspace), (0, "PRIVATE_RUN_COMPLETE"))
        report = json.loads((workspace / "results/pairing_report.json").read_text())
        self.assertEqual(report["accounting"]["pair_rows"], 2)
        self.assertEqual(report["transition_reasons"], {"construction_reset": 2, "paired": 2})
        pairs = pd.read_csv(workspace / "results/dataset.csv")
        self.assertEqual(set(pairs.start_date), {"2017-12-31"})


if __name__ == "__main__":
    unittest.main()
