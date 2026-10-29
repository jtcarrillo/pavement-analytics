import unittest

import pandas as pd

from pavement_analytics.dates import normalize_dates
from pavement_analytics.security import SafeError


class DateNormalizationTests(unittest.TestCase):
    def normalize(self, values):
        diagnostics = {}
        result = normalize_dates(pd.Series(values, dtype=object), diagnostics=diagnostics)
        return result.tolist(), diagnostics

    def test_year_first_dates_and_valid_timestamps(self):
        result, report = self.normalize([
            "2020-02-29", "2020/2/29", " 2020-02-29  00:00:00 ",
            "2020-02-29T23:59:59.123456789", "2020-02-29 12:30 PM",
        ])
        self.assertEqual(result, ["2020-02-29"] * 5)
        self.assertEqual(report["parsed_rows"], 5)
        self.assertEqual(report["date_order"], "not_required")

    def test_month_first_evidence_resolves_ambiguous_values(self):
        result, report = self.normalize([
            "01/02/2020", "12/31/2020 12:00:00 AM", "1-3-2020 1:59 pm", "2020-02-29",
        ])
        self.assertEqual(result, ["2020-01-02", "2020-12-31", "2020-01-03", "2020-02-29"])
        self.assertEqual(report["date_order"], "month_first")
        self.assertEqual(report["ambiguous_rows"], 2)

    def test_day_first_evidence_resolves_ambiguous_values(self):
        result, report = self.normalize(["01/02/2020", "31/12/2020 23:59", "29/02/2020"])
        self.assertEqual(result, ["2020-02-01", "2020-12-31", "2020-02-29"])
        self.assertEqual(report["date_order"], "day_first")

    def test_row_order_cannot_change_interpretation(self):
        values = ["01/02/2020", "12/31/2020", "03/04/2020", "12/31/2020"]
        forward, report = self.normalize(values)
        backward, reversed_report = self.normalize(list(reversed(values)))
        self.assertEqual(forward, list(reversed(backward)))
        self.assertEqual(report, reversed_report)

    def test_ambiguous_source_requires_explicit_format(self):
        report = {}
        with self.assertRaisesRegex(SafeError, "^AMBIGUOUS_DATE_FORMAT_USE_DATE_FORMAT$"):
            normalize_dates(pd.Series(["01/02/2020", "03/04/2020", "2020-12-31"]), diagnostics=report)
        self.assertEqual(report["date_order"], "unresolved")
        self.assertEqual(report["ambiguous_rows"], 2)
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("01/02/2020", str(report))

    def test_conflicting_evidence_fails_even_without_ambiguous_rows(self):
        report = {}
        with self.assertRaisesRegex(SafeError, "^MIXED_DATE_FORMATS_USE_DATE_FORMAT$"):
            normalize_dates(pd.Series(["12/31/2020", "31/12/2020"]), diagnostics=report)
        self.assertEqual(report["date_order"], "conflicting")
        self.assertEqual(report["parsed_rows"], 0)

    def test_same_result_in_both_orders_needs_no_guess(self):
        result, report = self.normalize(["01/01/2020", "12-12-2020"])
        self.assertEqual(result, ["2020-01-01", "2020-12-12"])
        self.assertEqual(report["date_order"], "not_required")

    def test_impossible_calendar_values_provide_no_date_order_evidence(self):
        result, report = self.normalize(["2019-02-29", "31/02/2020", "02/30/2020", "2020-13-01"])
        self.assertEqual(result, [""] * 4)
        self.assertEqual(report["invalid_rows"], 4)
        with self.assertRaisesRegex(SafeError, "AMBIGUOUS_DATE_FORMAT"):
            self.normalize(["31/02/2020", "01/02/2020"])

    def test_complete_clock_must_be_valid(self):
        values = ["2020-01-01 24:00", "2020-01-01 23:60", "2020-01-01 23:59:60",
                  "2020-01-01 00:00 AM", "2020-01-01 13:00 PM", "2020-01-01 2",
                  "2020-01-01 12:00:00.123junk", "2020-01-01 extra"]
        result, report = self.normalize(values)
        self.assertEqual(result, [""] * len(values))
        self.assertEqual(report["invalid_rows"], len(values))

    def test_timezone_serial_number_short_year_and_non_numeric_formats_unsupported(self):
        values = ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00+01:00",
                  "43831", "1577836800", "01/01/20", "January 1, 2020", "2020-01/01"]
        result, report = self.normalize(values)
        self.assertEqual(result, [""] * len(values))
        self.assertEqual(report["invalid_rows"], len(values))

    def test_blanks_and_missing_values_are_counted_separately(self):
        result, report = self.normalize([None, pd.NA, float("nan"), "", " \t ", "invalid"])
        self.assertEqual(result, [""] * 6)
        self.assertEqual(report["blank_rows"], 5)
        self.assertEqual(report["invalid_rows"], 1)

    def test_index_and_name_are_preserved(self):
        values = pd.Series(["2020-01-01", "2021-01-01"], index=[8, 4], name="Date")
        result = normalize_dates(values, diagnostics={})
        pd.testing.assert_series_equal(result, values)

    def test_empty_source_has_complete_accounting(self):
        result, report = self.normalize([])
        self.assertEqual(result, [])
        self.assertEqual(report["input_rows"], 0)
        self.assertEqual(report["parsed_rows"], 0)


if __name__ == "__main__":
    unittest.main()
