"""Generated duration-band evidence without refitting or modifying observations."""
import unittest

import numpy as np
import pandas as pd

from pavement_analytics.rate_diagnostics import DURATION_BANDS, rate_interval_diagnostics
from pavement_analytics.security import SafeError


class RateIntervalDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.plan = pd.DataFrame({"group_id": ["001", "002", "003"],
                                  "partition": ["development", "holdout", "holdout"]})
        self.table = pd.DataFrame({
            "row_id": [10, 11, 12, 13], "group_id": ["001", "001", "002", "003"],
            "delta_days": [1, 365, 1, 365], "DetRatePerYear": [-200., -1., -100., -2.],
        })
        self.saved = pd.DataFrame({"row_id": [13, 12], "actual": [-2., -100.],
                                   "weight": [365., 1.], "prediction": [-1., 0.]})

    def records(self, predictions=None, table=None):
        result = rate_interval_diagnostics(self.table if table is None else table,
                                           self.plan, predictions, "AC")
        return {(r["partition"], r["duration_band"]): r for r in result}

    def test_saved_weighted_error_and_surface_sse_shares_preserve_sign_and_inputs(self):
        table, plan, saved = self.table.copy(deep=True), self.plan.copy(deep=True), self.saved.copy(deep=True)
        records = self.records(self.saved)
        short = records["holdout", "<=1 day"]
        long = records["holdout", "365+ days"]
        self.assertEqual(len(records), 10)
        self.assertEqual(short["weighted_rmse"], 100.)
        self.assertEqual(short["weighted_mae"], 100.)
        self.assertEqual(short["weighted_sse"], 10000.)
        self.assertEqual(long["weighted_sse"], 365.)
        self.assertAlmostEqual(short["weighted_sse_share"], 10000. / 10365.)
        self.assertAlmostEqual(long["weighted_sse_share"], 365. / 10365.)
        self.assertEqual(short["min_actual_rate"], -100.)
        self.assertEqual(short["median_actual_rate"], -100.)
        self.assertEqual(short["prediction_status"], "scored")
        self.assertEqual(short["prediction_rows"], 1)
        for name, expected in (("table", table), ("plan", plan), ("saved", saved)):
            pd.testing.assert_frame_equal(getattr(self, name), expected)

    def test_weighted_errors_within_a_band_are_not_unweighted_errors(self):
        table = self.table.copy()
        table.loc[table.row_id.eq(12), "delta_days"] = 2
        table.loc[table.row_id.eq(13), "delta_days"] = 29
        saved = self.saved.assign(weight=[29., 2.])
        scored = self.records(saved, table)["holdout", ">1 to <30 days"]
        self.assertAlmostEqual(scored["weighted_rmse"], np.sqrt((2 * 10000 + 29) / 31))
        self.assertAlmostEqual(scored["weighted_mae"], (2 * 100 + 29) / 31)
        self.assertEqual(scored["weighted_sse_share"], 1.)
        self.assertEqual(scored["groups"], 2)
        self.assertEqual(scored["median_actual_rate"], -51.)

    def test_every_whole_day_boundary_and_missing_rate(self):
        days = [1, 2, 29, 30, 89, 90, 364, 365, 730, 1, 1]
        table = pd.DataFrame({"row_id": range(len(days)), "group_id": ["001"] * len(days),
                              "delta_days": days, "DetRatePerYear": [-float(x) for x in days[:-2]] + [np.nan, np.inf]})
        records = self.records(table=table)
        self.assertEqual([records["development", band]["eligible_rows"] for band in DURATION_BANDS], [1, 2, 2, 2, 2])
        for band in DURATION_BANDS:
            self.assertEqual(records["development", band]["groups"], 1)
            self.assertEqual(records["development", band]["prediction_status"], "not_evaluated")
            self.assertIsNone(records["development", band]["weighted_rmse"])
            self.assertEqual(records["holdout", band]["prediction_status"], "no_eligible_rows")

    def test_csv_text_predictions_map_to_integer_source_row_ids(self):
        expected = self.records(self.saved)
        actual = self.records(self.saved.astype(str))
        self.assertEqual(actual, expected)

    def test_fractional_positive_intervals_are_retained_at_continuous_boundaries(self):
        days = [.5, 1., 1.5, 29.5, 30., 89.5, 90., 364.5, 365., 365.5]
        table = pd.DataFrame({"row_id": range(len(days)), "group_id": ["001"] * len(days),
                              "delta_days": days, "DetRatePerYear": [-2.] * len(days)})
        records = self.records(table=table)
        self.assertEqual([records["development", band]["eligible_rows"] for band in DURATION_BANDS], [2] * 5)
        self.assertAlmostEqual(sum(r["weight_sum"] for r in records.values()), sum(days))

    def test_absent_empty_and_partial_predictions_are_explicitly_unscored(self):
        for predictions in (None, pd.DataFrame(), self.saved.iloc[:1]):
            with self.subTest(rows=None if predictions is None else len(predictions)):
                records = self.records(predictions)
                for band in ("<=1 day", "365+ days"):
                    item = records["holdout", band]
                    expected = "incomplete_predictions" if predictions is not None and len(predictions) else "no_predictions"
                    self.assertEqual(item["prediction_status"], expected)
                    for score in ("weighted_rmse", "weighted_mae", "weighted_sse", "weighted_sse_share"):
                        self.assertIsNone(item[score])
                self.assertIsNone(records["development", "<=1 day"]["weighted_sse"])

    def test_perfect_predictions_have_zero_errors_and_undefined_sse_shares(self):
        records = self.records(self.saved.assign(prediction=self.saved.actual))
        for band in ("<=1 day", "365+ days"):
            self.assertEqual(records["holdout", band]["weighted_rmse"], 0.)
            self.assertEqual(records["holdout", band]["weighted_sse"], 0.)
            self.assertIsNone(records["holdout", band]["weighted_sse_share"])

    def test_prediction_identity_actual_weight_and_finiteness_guards(self):
        cases = [
            (self.saved.assign(row_id=[10, 12]), "PREDICTION_ROWS_INVALID"),
            (self.saved.assign(row_id=[13, 13]), "ROW_IDS_INVALID"),
            (self.saved.assign(row_id=[13.5, 12]), "ROW_IDS_INVALID"),
            (self.saved.assign(actual=[-3., -100.]), "ACTUAL_MISMATCH"),
            (self.saved.assign(weight=[364., 1.]), "WEIGHT_MISMATCH"),
            (self.saved.assign(weight=[0., 1.]), "PREDICTIONS_INVALID"),
            (self.saved.assign(prediction=[np.nan, 0.]), "PREDICTIONS_INVALID"),
            (self.saved.drop(columns="prediction"), "PREDICTIONS_SCHEMA_INVALID"),
        ]
        for saved, code in cases:
            with self.subTest(code=code), self.assertRaisesRegex(SafeError, code):
                self.records(saved)

    def test_invalid_source_intervals_and_unknown_groups_are_not_silently_removed(self):
        for days in (0, -1, np.inf, np.nan):
            table = self.table.astype({"delta_days": float})
            table.loc[0, "delta_days"] = days
            with self.subTest(days=days), self.assertRaisesRegex(SafeError, "INTERVALS_INVALID"):
                self.records(table=table)
        with self.assertRaisesRegex(SafeError, "PARTITIONS_INVALID"):
            self.records(table=self.table.assign(group_id="unknown"))
        with self.assertRaisesRegex(SafeError, "ROW_IDS_INVALID"):
            self.records(table=self.table.assign(row_id=1))
        with self.assertRaisesRegex(SafeError, "SCHEMA_INVALID"):
            self.records(table=self.table.drop(columns="DetRatePerYear"))

    def test_empty_eligible_population_returns_ten_empty_bands(self):
        records = self.records(table=self.table.assign(DetRatePerYear=np.nan))
        self.assertEqual(len(records), 10)
        self.assertTrue(all(r["eligible_rows"] == 0 and r["groups"] == 0 for r in records.values()))
        self.assertTrue(all(r["min_actual_rate"] is None and r["median_actual_rate"] is None for r in records.values()))


if __name__ == "__main__":
    unittest.main()
