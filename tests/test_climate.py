import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from pavement_analytics import climate
from pavement_analytics.security import SafeError
from temp_support import temporary_directory


def hourly(start="2020-01-01", periods=48):
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    hours = np.arange(periods) % 24
    return pd.DataFrame({"air_k": 273.15 + hours - 12,
                         "skin_k": 273.15 + 2 * hours - 12,
                         "precip_m": 0.001}, index=index)


def source_rows():
    return pd.DataFrame({"row_id": [7, 3, 9], "inspection_date": ["2020-01-03"] * 3,
                         "_source_file": ["source-a.csv"] * 3,
                         "group_id": ["synthetic-a"] * 3,
                         "Latitude": [0, 0, ""], "Longitude": [0, 0, ""]}, index=[9, 2, 5])


def save_response(path, frame, *, archive=False):
    coordinates = {"latitude": 0.0, "longitude": 0.0}
    if archive:
        with zipfile.ZipFile(path, "w") as output:
            for internal, external in (("air_k", "t2m"), ("skin_k", "skt"), ("precip_m", "tp")):
                part = pd.DataFrame({"time": frame.index.astype(str), **coordinates,
                                     external: frame[internal].to_numpy()})
                output.writestr(f"{external}.csv", part.to_csv(index=False))
    else:
        part = frame.rename(columns={"air_k": "t2m", "skin_k": "skt", "precip_m": "tp"}).copy()
        part.index.name = "time"
        for name, value in coordinates.items():
            part[name] = value
        part.to_csv(path)


class LocationTests(unittest.TestCase):
    def test_decimal_and_hemisphere_pairs(self):
        for text in ("40.25,-120.5", "40.25N 120.5W", "N40.25 W120.5", "40.25 N 120.5 W"):
            self.assertEqual(climate.parse_location(text), (40.25, -120.5))
        self.assertEqual(climate.parse_location("0, 0"), (0., 0.))
        for text in ("91, 0", "0, 181", "nan, 0", "0 0 0", "40S 10N", "-40N 20E", "N W", "0:0"):
            self.assertIsNone(climate.parse_location(text), text)

    def test_original_source_mapping_with_identity_preserved(self):
        with temporary_directory() as directory:
            root = Path(directory)
            (root / "mapping.json").write_text(json.dumps({"synthetic.xlsx": "location.csv"}))
            (root / "location.csv").write_text('LOCATION(DEGREES)\n"0, 0"\n"0N 0E"\n')
            table = source_rows().drop(columns=["Latitude", "Longitude"])
            locations, audit = climate._locations(table, {"mapping": root / "mapping.json", "csv_dir": root},
                                                  {"source-a.csv": "synthetic.xlsx"})
            self.assertEqual(locations, [(0., 0.)] * 3)
            self.assertEqual(len(audit["location_sources"]), 1)
            self.assertEqual(len(audit["mapping_sha256"]), 64)

    def test_mapping_path_escape_is_rejected(self):
        with temporary_directory() as directory:
            root = Path(directory)
            (root / "csvs").mkdir()
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({"source-a.csv": "../location.csv"}))
            table = source_rows().drop(columns=["Latitude", "Longitude"])
            with self.assertRaisesRegex(SafeError, "PATH_OUTSIDE_CSV_DIR"):
                climate._locations(table, {"mapping": mapping, "csv_dir": root / "csvs"}, {})

    def test_ambiguous_location_csv_does_not_pick_first(self):
        with temporary_directory() as directory:
            path = Path(directory) / "location.csv"
            path.write_text('LOCATION(DEGREES)\n"0, 0"\n"0, 1"\n')
            with self.assertRaisesRegex(SafeError, "AMBIGUOUS_LOCATION_CSV"):
                climate._location_csv(path)


class WindowTests(unittest.TestCase):
    def test_features_use_only_prior_complete_days(self):
        block = hourly()
        end = pd.Timestamp("2020-01-03", tz="UTC")
        expected, coverage = climate.window_features(block, end, 2)
        outside = hourly("2019-12-31", 120)
        outside.loc[:, ["air_k", "skin_k", "precip_m"]] = [350, 350, 10]
        outside.loc[block.index] = block
        actual, other_coverage = climate.window_features(outside, end, 2)
        self.assertEqual(actual, expected)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(other_coverage, 1.0)
        self.assertEqual(set(actual), set(climate.FEATURES))
        self.assertEqual(actual["era5_freezing_hours"], 24)
        self.assertEqual(actual["era5_freeze_thaw_days"], 2)
        self.assertAlmostEqual(actual["era5_precipitation_total_mm"], 48)
        self.assertAlmostEqual(actual["era5_precipitation_max_daily_mm"], 24)
        self.assertEqual(actual["era5_wet_days_1mm"], 2)
        self.assertEqual(actual["era5_skin_cooling_max_6h_c"], 36)

    def test_missing_hours_and_missing_variables_have_no_partial_metrics(self):
        end = pd.Timestamp("2020-01-03", tz="UTC")
        for frame in (hourly().iloc[1:], hourly().assign(precip_m=np.nan)):
            values, coverage = climate.window_features(frame, end, 2)
            self.assertEqual(values, {})
            self.assertLess(coverage, 1)

    def test_cooling_events_count_peaks_with_refractory_period(self):
        index = pd.date_range("2020-01-01", periods=72, freq="h", tz="UTC")
        series = pd.Series(0., index=index)
        series.iloc[[1, 2, 12, 25, 50]] = [15, 15, 20, 15, 15]
        self.assertEqual(climate.cooling_event_count(series), 3)

    def test_rainflow_closed_and_residual_cycles(self):
        self.assertEqual(climate.rainflow_cycles(np.array([0, 0, 0])), [])
        self.assertEqual(climate.rainflow_cycles(np.array([0, 5, 10])), [(10., .5)])
        self.assertEqual(climate.rainflow_cycles(np.array([0, 10, 0, 10, 0])),
                         [(10., 1.), (10., .5), (10., .5)])
        self.assertEqual(climate.rainflow_cycles(np.array([0, 10, 10, 0, 10, 0])),
                         climate.rainflow_cycles(np.array([0, 10, 0, 10, 0])))


class CacheTests(unittest.TestCase):
    def test_multi_year_request_is_split_and_reused_by_calendar_year(self):
        with temporary_directory() as directory:
            root = Path(directory)
            seen = []
            def download(request, target):
                start, end = map(pd.Timestamp, request["date"].split("/"))
                self.assertEqual(start.year, end.year)
                seen.append((start, end))
                save_response(target, hourly(start.isoformat(), ((end - start).days + 1) * 24))
            with patch.object(climate, "_download", side_effect=download):
                args = (0., 0., pd.Timestamp("2019-12-31"), pd.Timestamp("2021-01-01"), root)
                frame, audit = climate.fetch_hourly(*args)
                second, reused = climate.fetch_hourly(*args)
            self.assertEqual(len(seen), 3)
            self.assertEqual(len(audit["chunks"]), 3)
            self.assertEqual(len(frame), (1 + 366 + 1) * 24)
            self.assertEqual((seen[1][1] - seen[1][0]).days + 1, 366)
            self.assertTrue(reused["cache_hit"])
            pd.testing.assert_frame_equal(frame, second)

    def test_grid_change_between_years_is_rejected(self):
        with temporary_directory() as directory:
            root = Path(directory)
            def download(request, target):
                start, end = map(pd.Timestamp, request["date"].split("/"))
                save_response(target, hourly(start.isoformat(), ((end - start).days + 1) * 24))
                frame = pd.read_csv(target)
                frame["latitude"] = 0.0 if start.year == 2019 else 0.1
                frame.to_csv(target, index=False)
            with patch.object(climate, "_download", side_effect=download):
                with self.assertRaisesRegex(SafeError, "GRID_POINT_CHANGED_BETWEEN_REQUESTS"):
                    climate.fetch_hourly(0., 0., pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-01"), root)

    def test_grouped_archive_and_cache_reuse_are_hashed(self):
        with temporary_directory() as directory:
            root = Path(directory)
            def download(request, target):
                self.assertEqual(request["date"], "2020-01-01/2020-01-02")
                self.assertEqual(request["variable"], climate.VARIABLES)
                save_response(target, hourly(), archive=True)
            with patch.object(climate, "_download", side_effect=download) as mocked:
                args = (0., 0., pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), root)
                first, audit = climate.fetch_hourly(*args)
                second, reused = climate.fetch_hourly(*args)
                self.assertEqual(mocked.call_count, 1)
                self.assertFalse(audit["cache_hit"])
                self.assertTrue(reused["cache_hit"])
                pd.testing.assert_frame_equal(first, second)
                self.assertEqual(first["air_k"].iloc[0], hourly()["air_k"].iloc[0])
                raw = root / f'{audit["cache_key"]}.data'
                raw.write_bytes(raw.read_bytes() + b"tampering")
                with self.assertRaisesRegex(SafeError, "CACHE_HASH_MISMATCH"):
                    climate.fetch_hourly(*args)

    def test_duplicate_hours_do_not_multiply_rows(self):
        with temporary_directory() as directory:
            path = Path(directory) / "response.csv"
            save_response(path, pd.concat([hourly(), hourly().iloc[:1]]))
            with self.assertRaisesRegex(SafeError, "TIMESTAMPS_INVALID"):
                climate._read_hourly(path)

    def test_missing_variable_and_multiple_grid_points_rejected(self):
        with temporary_directory() as directory:
            path = Path(directory) / "response.csv"
            save_response(path, hourly())
            frame = pd.read_csv(path)
            frame.drop(columns="tp").to_csv(path, index=False)
            with self.assertRaisesRegex(SafeError, "REQUIRED_VARIABLE_MISSING"):
                climate._read_hourly(path)
            frame.loc[1, "latitude"] = 1
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(SafeError, "MULTIPLE_OR_INVALID_GRID_POINTS"):
                climate._read_hourly(path)

    def test_invalid_values_are_missing_coverage(self):
        with temporary_directory() as directory:
            path = Path(directory) / "response.csv"
            frame = hourly()
            frame.iloc[0, 0] = 0
            frame.iloc[1, 2] = -1
            save_response(path, frame)
            data = climate._read_hourly(path)
            self.assertTrue(np.isnan(data.iloc[0, 0]))
            self.assertTrue(np.isnan(data.iloc[1, 2]))


class EnrichmentTests(unittest.TestCase):
    def test_decades_apart_inspections_download_only_needed_years(self):
        with temporary_directory() as directory:
            table = source_rows().iloc[:2].copy()
            table["inspection_date"] = ["2000-01-03", "2024-01-03"]
            seen = []
            def download(request, target):
                start, end = map(pd.Timestamp, request["date"].split("/"))
                days = (end - start).days + 1
                self.assertEqual(start.year, end.year)
                self.assertLessEqual(days, 366)
                seen.append(start.year)
                save_response(target, hourly(start.isoformat(), days * 24))
            with patch.object(climate, "_download", side_effect=download):
                result, _, audit = climate.enrich(table, {"lookback_days": 500}, Path(directory) / "out")
            self.assertEqual(seen, [1998, 1999, 2000, 2022, 2023, 2024])
            self.assertEqual(result["climate_status"].tolist(), ["ok", "ok"])
            self.assertEqual(len(audit["requests"][0]["chunks"]), 6)

    def test_preserves_identity_order_and_excludes_missing_location(self):
        with temporary_directory() as directory:
            root = Path(directory)
            with patch.object(climate, "_download", side_effect=lambda request, target: save_response(target, hourly())) as mock:
                table = source_rows()
                enriched, features, audit = climate.enrich(table, {"lookback_days": 2}, root / "out")
            pd.testing.assert_series_equal(enriched["row_id"], table["row_id"])
            self.assertEqual(mock.call_count, 1)
            self.assertEqual(enriched["climate_status"].tolist(), ["ok", "ok", "missing_location"])
            self.assertEqual(audit["counts"], {"ok": 2, "missing_location": 1})
            self.assertTrue(enriched.loc[5, features].isna().all())
            self.assertEqual(enriched.loc[9, "climate_location_id"], enriched.loc[2, "climate_location_id"])
            self.assertTrue(enriched.loc[9, "climate_location_id"])
            self.assertEqual(enriched.loc[5, "climate_location_id"], "")
            rows = pd.read_csv(root / "out" / "climate_rows.csv")
            self.assertEqual(rows["row_id"].tolist(), [7, 3, 9])

    def test_actual_grid_point_groups_different_requested_coordinates(self):
        with temporary_directory() as directory:
            table = source_rows().iloc[:2].copy()
            table["Latitude"] = [0.01, 0.02]
            with patch.object(climate, "_download", side_effect=lambda request, target: save_response(target, hourly())):
                result, _, _ = climate.enrich(table, {"lookback_days": 2}, Path(directory) / "out")
            self.assertEqual(result["climate_location_id"].nunique(), 1)

    def test_no_locations_produces_audit_and_never_downloads(self):
        with temporary_directory() as directory:
            out = Path(directory) / "out"
            with patch.object(climate, "_download") as mock:
                with self.assertRaisesRegex(SafeError, "NO_VALID_CLIMATE_LOCATIONS_OR_DATES"):
                    climate.enrich(source_rows().drop(columns=["Latitude", "Longitude"]), {}, out)
                mock.assert_not_called()
            self.assertEqual(json.loads((out / "climate_audit.json").read_text())["counts"], {"missing_location": 3})

    def test_no_complete_coverage_is_audited_and_fails(self):
        with temporary_directory() as directory:
            out = Path(directory) / "out"
            with patch.object(climate, "_download", side_effect=lambda request, target: save_response(target, hourly().iloc[1:])):
                with self.assertRaisesRegex(SafeError, "NO_VALID_ERA5_CLIMATE_ROWS"):
                    climate.enrich(source_rows(), {"lookback_days": 2}, out)
            audit = json.loads((out / "climate_audit.json").read_text())
            self.assertEqual(audit["counts"]["incomplete_hourly_coverage"], 2)

    def test_download_error_has_audit_without_raw_exception(self):
        with temporary_directory() as directory:
            out = Path(directory) / "out"
            with patch.object(climate, "_download", side_effect=SafeError("ERA5_TEST_FAILURE")):
                with self.assertRaisesRegex(SafeError, "ERA5_TEST_FAILURE"):
                    climate.enrich(source_rows(), {"lookback_days": 2}, out)
            audit = json.loads((out / "climate_audit.json").read_text())
            self.assertEqual(audit["failure_code"], "ERA5_TEST_FAILURE")
            self.assertEqual(audit["counts"]["era5_request_failed"], 2)

    def test_existing_features_require_explicit_names_and_finite_values(self):
        with temporary_directory() as directory:
            table = source_rows().assign(climate_temperature=[1, 2, np.inf])
            with patch.object(climate, "_download") as mock:
                result, features, audit = climate.enrich(table, {"mode": "existing", "features": ["climate_temperature"]}, Path(directory) / "out")
                mock.assert_not_called()
            self.assertEqual(features, ["climate_temperature"])
            self.assertEqual(result["climate_status"].tolist(), ["ok", "ok", "invalid_existing_features"])
            self.assertFalse(audit["historical_window_verified"])
            self.assertTrue(result["climate_location_id"].iloc[0])
            self.assertEqual(result["climate_location_id"].iloc[0], result["climate_location_id"].iloc[1])
            with self.assertRaisesRegex(SafeError, "REQUIRES_EXPLICIT_FEATURES"):
                climate.enrich(table, {"mode": "existing"}, Path(directory) / "missing")

    def test_existing_outcomes_and_identity_are_rejected(self):
        with temporary_directory() as directory:
            for name in ("PCI", "PCI_start", "SCI", "FOD Index", "74L deduct", "74L (Slabs)",
                         "target", "level_rigid_74", "inspection_date", "Latitude", "SectionID", "SectionUniqueID",
                         "Most Recent Global MR Date", "mean PCI", "MeanPCI", "Pavement Condition Index", "family_74"):
                table = source_rows().assign(**{name: 1})
                with self.subTest(name=name), self.assertRaisesRegex(SafeError, "UNSAFE_EXISTING_CLIMATE_FEATURE"):
                    climate.enrich(table, {"mode": "existing", "features": [name]}, Path(directory) / "out")

    def test_bad_identity_and_lookback_rejected(self):
        with temporary_directory() as directory:
            out = Path(directory) / "out"
            with self.assertRaisesRegex(SafeError, "UNIQUE_ROW_ID"):
                climate.enrich(source_rows().assign(row_id=1), {}, out)
            for days in (0, -1, True, 1.5, 36526):
                with self.assertRaisesRegex(SafeError, "LOOKBACK_DAYS_INVALID"):
                    climate.enrich(source_rows(), {"lookback_days": days}, out)

    def test_unknown_configuration_key_is_rejected(self):
        with temporary_directory() as directory:
            with self.assertRaisesRegex(SafeError, "CLIMATE_CONFIGURATION_INVALID"):
                climate.enrich(source_rows(), {"lookback_day": 2}, Path(directory) / "out")

    def test_column_and_path_configuration_types_are_rejected_safely(self):
        with temporary_directory() as directory:
            out = Path(directory) / "out"
            for key in ("latitude_column", "longitude_column"):
                for value in ([], {}, None, "", " "):
                    with self.subTest(key=key, value=value), self.assertRaisesRegex(SafeError, "LOCATION_COLUMN_CONFIGURATION_INVALID"):
                        climate.enrich(source_rows(), {key: value}, out)
            for value in (None, {}, [], ""):
                with self.subTest(value=value), self.assertRaisesRegex(SafeError, "CLIMATE_PATH_CONFIGURATION_INVALID"):
                    climate.enrich(source_rows(), {"cache_dir": value}, out)

    def test_audit_output_symlinks_are_rejected(self):
        with temporary_directory() as directory:
            root = Path(directory)
            out = root / "out"
            out.mkdir()
            target = root / "unchanged.txt"
            target.write_text("original")
            (out / "climate_rows.csv").symlink_to(target)
            with self.assertRaisesRegex(SafeError, "SYMLINK_PATH_FORBIDDEN"):
                climate.enrich(source_rows(), {}, out)
            self.assertEqual(target.read_text(), "original")

    def test_negative_zero_coordinates_share_same_grid_id(self):
        self.assertEqual(climate._location_id(-0.01, -0.01), climate._location_id(0.01, 0.01))

    def test_unavailable_dates_do_not_trigger_requests(self):
        with temporary_directory() as directory:
            table = source_rows().copy()
            table["inspection_date"] = ["1950-01-02", "invalid", "2200-01-01"]
            out = Path(directory) / "out"
            with patch.object(climate, "_download") as mocked:
                with self.assertRaisesRegex(SafeError, "NO_VALID_CLIMATE_LOCATIONS_OR_DATES"):
                    climate.enrich(table, {"lookback_days": 2}, out)
                mocked.assert_not_called()
            audit = json.loads((out / "climate_audit.json").read_text())
            self.assertEqual(audit["counts"], {"window_before_era5_coverage": 1, "invalid_inspection_date": 2})


if __name__ == "__main__":
    unittest.main()
