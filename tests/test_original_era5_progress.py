"""Generated progress regressions without weather service calls or private data."""
import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from pavement_analytics.climate_progress import (
    ClimateProgress, PROGRESS_PREFIX, parse_progress_line, sanitize_climate_event,
)
from pavement_analytics.original import era5
from temp_support import temporary_directory


class FlushedStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1
        super().flush()


def events(stream):
    return [event for line in stream.getvalue().splitlines()
            if (event := parse_progress_line(line)) is not None]


class ClimateProgressProtocolTests(unittest.TestCase):
    def test_known_fields_round_trip_and_unknown_details_are_removed(self):
        payload = {"event": "climate", "phase": "requesting", "location_index": 2,
                   "total_locations": 7, "requests": 3, "dataset": "era5-land",
                   "start_year": 1980, "end_year": 2025,
                   "latitude": "generated-private-coordinate", "path": "generated-private-path",
                   "error": "generated-private-message", "stage": "untrusted-stage"}
        safe = sanitize_climate_event(payload)
        self.assertEqual(set(safe), {"event", "phase", "location_index", "total_locations",
                                     "requests", "dataset", "start_year", "end_year"})
        self.assertEqual(parse_progress_line(PROGRESS_PREFIX + json.dumps(payload) + "\n"), safe)
        self.assertNotIn("generated-private", json.dumps(safe))

    def test_malformed_protocol_and_unsafe_known_fields_are_rejected(self):
        for text in ("ordinary worker output", PROGRESS_PREFIX + "{", PROGRESS_PREFIX + "[]",
                     PROGRESS_PREFIX + "null"):
            self.assertIsNone(parse_progress_line(text))
        bad = [
            {"event": "other", "phase": "requesting"},
            {"event": "climate", "phase": "generated-private-message"},
            {"event": "climate", "phase": ["requesting"]},
            {"event": "climate", "phase": "requesting", "dataset": "generated-private-service"},
        ]
        for value in (-1, True, 1.5, "3", float("nan"), float("inf"), {}):
            bad.append({"event": "climate", "phase": "calculating", "rows_completed": value})
        for payload in bad:
            with self.subTest(payload=payload):
                self.assertIsNone(sanitize_climate_event(payload))
                self.assertIsNone(parse_progress_line(PROGRESS_PREFIX + json.dumps(payload)))

    def test_emission_is_flushed_and_keeps_progress_context(self):
        stream = FlushedStream()
        reporter = ClimateProgress(stream=stream)
        reporter.update("initializing", total_locations=4, total_rows=20)
        reporter.increment("requests")
        reporter.update("requesting", location_index=1, dataset="era5-land")
        result = events(stream)
        self.assertEqual([item["phase"] for item in result], ["initializing", "requesting"])
        self.assertEqual(result[-1]["requests"], 1)
        self.assertEqual(result[-1]["total_locations"], 4)
        self.assertEqual(result[-1]["total_rows"], 20)
        self.assertGreaterEqual(stream.flushes, 2)


class OriginalClimateProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_cache_reuse_and_request_are_distinct_before_service_call(self):
        stream = FlushedStream()
        reporter = ClimateProgress(stream=stream)
        target = self.root / "generated.zip"
        target.write_bytes(b"generated cached content")
        client = Mock()
        with contextlib.redirect_stdout(io.StringIO()):
            era5.download_timeseries(client, 0., 0., 1980, 2025, target, progress=reporter)
        client.retrieve.assert_not_called()
        self.assertEqual(events(stream)[-1]["phase"], "cache_hit")
        self.assertEqual(events(stream)[-1]["cache_hits"], 1)

        def retrieve(*args):
            self.assertEqual(events(stream)[-1]["phase"], "requesting")
            self.assertEqual(events(stream)[-1]["requests"], 1)
            self.assertGreater(stream.flushes, 1)

        client.retrieve.side_effect = retrieve
        with contextlib.redirect_stdout(io.StringIO()):
            era5.download_timeseries(client, 0., 0., 1980, 2025, target,
                                     force_download=True, progress=reporter)
        client.retrieve.assert_called_once()
        result = events(stream)
        self.assertEqual(result[-1]["phase"], "downloaded")
        self.assertEqual(result[-1]["requests"], 1)
        self.assertEqual(result[-1]["cache_hits"], 1)
        self.assertNotIn(str(target), stream.getvalue())

    def test_parquet_hit_reports_reuse_without_download_or_parse(self):
        stream = io.StringIO()
        reporter = ClimateProgress(stream=stream)
        key = era5.location_cache_key(0., 0., 1980, 2025)
        (self.root / (key + ".hourly.parquet")).write_bytes(b"generated cache marker")
        cached = pd.DataFrame({"generated": [1]})
        with patch.object(pd, "read_parquet", return_value=cached), \
                patch.object(era5, "download_timeseries") as download, \
                patch.object(era5, "open_dataset") as parse:
            result = era5._load_one_attempt(client=object(), lat=0., lon=0., start_year=1980,
                                            end_year=2025, cache_dir=self.root, force_download=False,
                                            dataset=era5.DATASET, cache_suffix="", progress=reporter)
        self.assertIs(result, cached)
        download.assert_not_called()
        parse.assert_not_called()
        phases = [event["phase"] for event in events(stream)]
        self.assertIn("cache_check", phases)
        self.assertEqual(phases[-1], "cache_hit")
        self.assertEqual(events(stream)[-1]["cache_hits"], 1)

    def test_worker_progress_counts_failed_locations_and_skipped_rows(self):
        source, output = self.root / "input.csv", self.root / "output.csv"
        pd.DataFrame({"row_key": range(105), "latitude": [0.] * 104 + [np.nan],
                      "longitude": [0.] * 103 + [1., 0.],
                      "Dates": ["[2020-01-01,2020-01-02]"] * 105,
                      "Original Construction Year": [2019] * 105}).to_csv(source, index=False)
        raw = pd.DataFrame({"generated": [1, 2]},
                           index=pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC"))
        info = {"data_source": "era5-land", "snap_distance_cells": 0, "snap_distance_km": 0.}
        stats = era5.nan_stats_dict()
        stats["mean_f"] = 50.
        stream = io.StringIO()

        def acquire(**kwargs):
            return (raw, info) if kwargs["lon"] == 0. else (None, info)

        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "1", "GITHUB_ACTIONS": "false"}), \
                patch.object(era5, "_create_cds_client", return_value=object()), \
                patch.object(era5, "get_hourly_for_location", side_effect=acquire), \
                patch.object(era5, "build_daily_pav_min_c", return_value=pd.Series(dtype=float)), \
                patch.object(era5, "build_skin_cooling_6h_c", return_value=pd.Series(dtype=float)), \
                patch.object(era5, "slice_to_interval", return_value=raw), \
                patch.object(era5, "compute_temp_stats_from_hourly", return_value=stats) as compute, \
                contextlib.redirect_stdout(stream):
            result = era5.main(["--input", str(source), "--output", str(output),
                                "--lat-col", "latitude", "--lon-col", "longitude",
                                "--start-year", "1980", "--end-year", "2025",
                                "--cache-dir", str(self.root / "cache"), "--col-suffix", ""])
        self.assertEqual(result, 0)
        self.assertEqual(compute.call_count, 103)
        result = events(stream)
        self.assertEqual(result[-1]["phase"], "complete")
        self.assertEqual(result[-1]["rows_completed"], 105)
        self.assertEqual(result[-1]["total_rows"], 105)
        self.assertEqual(result[-1]["locations_completed"], 2)
        self.assertEqual(result[-1]["usable_locations"], 1)
        self.assertEqual(result[-1]["missing_locations"], 1)
        self.assertTrue(any(event["phase"] == "location_failed" for event in result))
        calculations = [event["rows_completed"] for event in result if event["phase"] == "calculating"]
        self.assertEqual(calculations, sorted(calculations))
        self.assertEqual(calculations[0], 0)
        self.assertEqual(calculations[-1], 105)
        self.assertEqual(len(pd.read_csv(output)), 105)


if __name__ == "__main__":
    unittest.main()
