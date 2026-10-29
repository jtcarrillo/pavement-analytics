"""Generated regression cases for the preserved interval climate worker."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
import pandas as pd

from pavement_analytics.original import era5
from pavement_analytics.security import SafeError
from temp_support import temporary_directory


def hourly(start="2020-01-01", days=40):
    time = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    phase = np.arange(len(time)) * 2 * np.pi / 24
    raw = pd.DataFrame({"time": time, "t2m": 273.15 + 8 + 12 * np.sin(phase),
                        "skt": 273.15 + 10 + 20 * np.sin(phase)})
    return era5.prepare_hourly_df(raw, lon=0)


class OriginalClimateCalculations(unittest.TestCase):
    def test_original_complete_output_contract(self):
        labels = dict(era5.STAT_COLUMNS)
        self.assertEqual(len(labels), 37)
        self.assertEqual(len(set(labels.values())), 37)
        for name in ("PavPGGrade", "ThermalAgingPhi", "ThermalAgingTeffF",
                     "PGLowNormalF", "DeltaPGLowF", "RainflowDamagePerYear",
                     "Cooling12hF", "CoolEvents6hPerYear"):
            self.assertIn(name, labels.values())
        result = era5.compute_temp_stats_from_hourly(
            hourly(), lat=0, pav_depth_low_mm=20, pav_depth_high_mm=0,
            pav_reliability=50, min_ltpp_years=0,
            daily_pav_min_c=era5.build_daily_pav_min_c(hourly()),
            min_pg_window_years=0, cooling_6h_c=era5.build_skin_cooling_6h_c(hourly()),
            full_hourly=hourly(), construction_year=2019)
        self.assertEqual(set(result), set(labels))
        self.assertAlmostEqual(result["mean_f"], 46.4)
        self.assertAlmostEqual(result["daily_range_avg_f"], 43.2)
        self.assertEqual(result["interval_start"], "2020-01-01")
        self.assertGreater(result["thermal_aging_phi"], 0)
        self.assertTrue(result["pav_pg_grade"].startswith("PG "))
        self.assertAlmostEqual(result["rainflow_damage_per_year"],
                               result["rainflow_damage"] / result["interval_years"])

    def test_original_local_solar_day_and_temperature_units(self):
        raw = pd.DataFrame({"time": ["2020-01-01T23:00:00Z", "2020-01-02T00:00:00Z"],
                            "2m_temperature": [273.15, 283.15], "skin_temperature": [283.15, 293.15]})
        result = era5.prepare_hourly_df(raw, lon=15)
        self.assertEqual(result.temp_f.tolist(), [32, 50])
        self.assertEqual(result.skt_f.tolist(), [50, 68])
        self.assertEqual(result.local_date.nunique(), 1)
        self.assertEqual(str(result.local_date.iloc[0]), "2020-01-02")

    def test_interval_slicing_is_inclusive_and_accepts_reverse_dates(self):
        data = hourly(days=2)
        start, end = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")
        sliced = era5.slice_to_interval(data, end, start)
        self.assertEqual(len(sliced), 25)
        self.assertEqual(sliced.index.min(), start.tz_localize("UTC"))
        self.assertEqual(sliced.index.max(), end.tz_localize("UTC"))

    def test_original_forward_cooling_keeps_full_window_semantics(self):
        time = pd.date_range("2020-01-01", periods=12, freq="h", tz="UTC")
        data = pd.DataFrame({"skt_c": [20.] * 6 + [0.] * 6}, index=time)
        cooling = era5.build_skin_cooling_6h_c(data)
        # Preserve original interval compatibility: the peak can use later
        # samples from the location-wide series. This is not forecast input.
        count = era5.count_cool_events(cooling, time[0], time[2], 10., 24)
        self.assertEqual(count, 1)
        self.assertTrue(cooling.iloc[-6:].isna().all())

    def test_rainflow_damage_and_excursion_keep_original_definition(self):
        values = pd.Series([0., 20., 0.])
        result = era5.compute_rainflow_stats(values, fatigue_beta=3.5)
        self.assertEqual(result["rainflow_cycles_10c"], 1)
        self.assertEqual(result["rainflow_cycles_20c"], 1)
        self.assertAlmostEqual(result["rainflow_damage"], 20 ** 3.5)
        self.assertEqual(era5.compute_max_cooling_excursion_c(values), 20)
        self.assertEqual(era5.compute_max_cooling_excursion_c(pd.Series([0., 1., 2.])), 0)

    def test_thermal_aging_uses_construction_history_and_reports_gap(self):
        data = hourly(days=2)
        data["skt_c"] = 20.
        result = era5.compute_thermal_aging(data, 2018, data.index[-1])
        expected = len(data) * math.exp(-75000 / (8.314 * 293.15))
        self.assertAlmostEqual(result["thermal_aging_phi"], expected, places=20)
        self.assertAlmostEqual(result["thermal_aging_teff_f"], 68.)
        self.assertGreater(result["thermal_aging_missing_years"], 1)
        later = era5.compute_thermal_aging(data, 2021, data.index[-1])
        self.assertEqual(later["thermal_aging_phi"], 0)
        self.assertTrue(math.isnan(later["thermal_aging_teff_f"]))

    def test_pg_normal_uses_midpoint_anchored_twenty_year_history(self):
        daily = pd.Series(20., index=pd.date_range("1990-01-01", "2020-01-01", freq="D"))
        daily.loc["2018-01-01"] = -20.
        result = era5.compute_pg_low_stats(daily, pd.Timestamp("2018-01-01", tz="UTC"),
                                           pd.Timestamp("2019-01-01", tz="UTC"), 20, 10)
        self.assertEqual(result["pg_low_normal_f"], 68.)
        self.assertEqual(result["pg_low_interval_f"], -4.)
        self.assertEqual(result["delta_pg_low_f"], -72.)
        short = era5.compute_pg_low_stats(daily.loc["2017":], pd.Timestamp("2018-01-01"),
                                          pd.Timestamp("2019-01-01"), 20, 10)
        self.assertTrue(math.isnan(short["pg_low_normal_f"]))
        self.assertEqual(short["pg_low_interval_f"], -4.)

    def test_design_reliability_and_short_interval_rules(self):
        low, high = era5.ltpp_pavement_temps(-10., 30., 0., 20., 0., 50)
        low98, high98 = era5.ltpp_pavement_temps(-10., 30., 0., 20., 0., 98)
        self.assertAlmostEqual(low - low98, 4.2)
        self.assertAlmostEqual(high98 - high, 6.)
        result = era5.compute_temp_stats_from_hourly(hourly(days=2), 0., 20., 0., 50, 3.)
        self.assertEqual(result["pav_pg_grade"], "")
        self.assertTrue(math.isnan(result["pav_design_low_c"]))
        self.assertTrue(math.isfinite(result["mean_f"]))

    def test_date_cell_formats(self):
        for value in ('["2020-01-01", "2021-01-01"]', '[2020-01-01, 2021-01-01]',
                      '2020-01-01|2021-01-01', '2020-01-01;2021-01-01'):
            self.assertEqual([t.year for t in era5.parse_dates_cell(value)], [2020, 2021])
        self.assertEqual(era5.parse_dates_cell("not a date"), [])


class OriginalClimateIO(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_grouped_csv_archive_and_original_api_request(self):
        target = self.root / "hourly.zip"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("air.csv", "time,t2m\n2020-01-01T00:00:00Z,273.15\n")
            archive.writestr("skin.csv", "time,skt\n2020-01-01T00:00:00Z,283.15\n")
        result = era5.open_dataset(target)
        self.assertEqual(result.shape, (1, 3))
        client = Mock()
        with contextlib.redirect_stdout(io.StringIO()):
            era5.download_timeseries(client, 0., 0., 1980, 2025, self.root / "missing.zip")
        dataset, request, path = client.retrieve.call_args.args
        self.assertEqual(dataset, era5.DATASET)
        self.assertEqual(request["variable"], ["2m_temperature", "skin_temperature"])
        self.assertEqual(request["date"], "1980-01-01/2025-12-31")
        self.assertNotIn("product_type", request)
        self.assertEqual(request["data_format"], "csv")
        with contextlib.redirect_stdout(io.StringIO()):
            era5.download_timeseries(client, 0., 0., 1980, 2025, self.root / "fallback.zip",
                                     dataset=era5.DATASET_ERA5)
        self.assertEqual(client.retrieve.call_args.args[1]["product_type"], ["reanalysis"])

    def test_land_snapping_and_global_fallback_are_retained(self):
        data = hourly(days=2)
        args = dict(client=object(), lat=0., lon=0., start_year=1980, end_year=2025,
                    cache_dir=self.root, force_download=False, snap_radius=1, era5_fallback=True)
        with patch.object(era5, "_load_one_attempt", side_effect=[None, data]) as load:
            result, source = era5.get_hourly_for_location(**args)
        self.assertIs(result, data)
        self.assertEqual(source["data_source"], "era5-land")
        self.assertEqual(source["snap_distance_cells"], 1)
        self.assertEqual(load.call_count, 2)
        args["snap_radius"] = 0
        with patch.object(era5, "_load_one_attempt", side_effect=[None, data]) as load, contextlib.redirect_stdout(io.StringIO()):
            result, source = era5.get_hourly_for_location(**args)
        self.assertEqual(source["data_source"], "era5")
        self.assertEqual(load.call_args.kwargs["dataset"], era5.DATASET_ERA5)

    def test_worker_outputs_all_original_columns_without_network(self):
        source = self.root / "input.csv"
        output = self.root / "result.csv"
        pd.DataFrame({"row_key": [3, 1], "latitude": [0., 0.], "longitude": [0., 0.],
                      "Dates": ["[2020-01-01,2020-01-20]", "[2020-01-15,2020-02-01]"],
                      "Original Construction Year": [2018, 2019]}).to_csv(source, index=False)
        info = {"data_source": "era5-land", "snap_distance_cells": 0, "snap_distance_km": 0.}
        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "1", "GITHUB_ACTIONS": "false"}), \
                patch.object(era5, "_create_cds_client", return_value=object()), \
                patch.object(era5, "get_hourly_for_location", return_value=(hourly(), info)) as get, \
                contextlib.redirect_stdout(io.StringIO()):
            result = era5.main(["--input", str(source), "--output", str(output),
                                "--lat-col", "latitude", "--lon-col", "longitude",
                                "--cache-dir", str(self.root / "cache"), "--col-suffix", ""])
        self.assertEqual(result, 0)
        result = pd.read_csv(output)
        self.assertEqual(result.row_key.tolist(), [3, 1])
        self.assertEqual(get.call_count, 1)
        self.assertTrue(set(label for _, label in era5.STAT_COLUMNS) <= set(result))
        self.assertTrue((result.ThermalAgingPhi > 0).all())

    def test_cli_is_private_and_service_dependency_is_lazy(self):
        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "0"}):
            with self.assertRaisesRegex(SafeError, "PRIVATE_EXECUTION"):
                era5.main([])
        with patch.object(era5, "cdsapi", None):
            with self.assertRaisesRegex(SafeError, "REQUIRES_CDSAPI"):
                era5._create_cds_client()
            self.assertEqual(era5.pg_grade_label(-20., 60.), "PG 64-22")

    def test_cache_symlink_is_rejected_before_loading(self):
        target = self.root / "target"
        target.write_text("private")
        name = era5.location_cache_key(0., 0., 1980, 2025)
        (self.root / (name + ".zip")).symlink_to(target)
        with self.assertRaisesRegex(SafeError, "SYMLINK"):
            era5._load_one_attempt(object(), 0., 0., 1980, 2025, self.root, False, era5.DATASET, "")


if __name__ == "__main__":
    unittest.main()
