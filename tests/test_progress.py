"""Progress remains readable while raw data, messages and credentials stay private."""
import io
import unittest

from pavement_analytics.progress import FullProgress


class FlushedStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1
        super().flush()


class BrokenStream:
    def __init__(self):
        self.writes = 0

    def write(self, text):
        self.writes += 1
        raise BrokenPipeError("closed output")

    def flush(self):
        raise AssertionError("A failed stream should be disabled before flushing")


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.console, self.log = FlushedStream(), FlushedStream()
        self.progress = FullProgress(self.console, self.log)

    def test_setup_inventory_paths_row_counts_and_mapping_gaps_are_visible(self):
        events = [
            {"event": "setup_started", "workspace": "/private/run", "data": "/private/raw", "resume": False},
            {"event": "setup_files", "profile": "/private/pipeline.json", "mapping": "/private/mapping.json",
             "location_directory": "/private/locations", "cache": "/private/cache", "cache_source": "profile",
             "cache_fallback_reason": "CLIMATE_CACHE_MUST_NOT_BE_VERSIONED"},
            {"event": "input_started", "path": "/private/raw/source.csv", "index": 1, "files": 3},
            {"event": "input", "path": "/private/raw/source.csv", "rows": 1200,
             "location": "/private/locations/site.csv", "location_status": "mapped"},
            {"event": "input", "path": "/private/raw/unmapped.csv", "rows": 25,
             "location": None, "location_status": "unmapped"},
            {"event": "input", "path": "/private/raw/missing.csv", "rows": 30,
             "location": "/private/locations/missing.csv", "location_status": "missing_file"},
            {"event": "setup_complete", "files": 3, "rows": 1255, "mapped_files": 1,
             "unmapped_files": 1, "missing_location_files": 1, "report": "/private/run/full_setup_report.json"},
        ]
        for event in events:
            self.progress(event)
        result = self.console.getvalue()
        for name in ("/private/pipeline.json", "/private/mapping.json", "/private/locations/site.csv",
                     "/private/raw/source.csv", "/private/cache", "rows=1,200", "input rows=1,255",
                     "UNMAPPED", "LOCATION FILE MISSING", "mapped files=1", "unmapped files=1",
                     "missing location files=1"):
            self.assertIn(name, result)
        self.assertIn("Reading input (1/3)", result)
        self.assertEqual(self.log.getvalue(), result)
        self.assertEqual(self.console.flushes, len(events))

    def test_started_is_flushed_before_completion_and_statuses_remain_distinct(self):
        self.progress({"event": "stage", "stage": "01_split", "status": "started"})
        self.assertEqual(self.console.getvalue(), 'Started "01_split"\n')
        self.assertEqual(self.console.flushes, 1)
        self.progress({"event": "stage", "stage": "01_split", "status": "completed", "seconds": 2.34,
                       "stats": {"input_rows": 1200, "retained_rows": 1100, "excluded_rows": 100,
                                 "surfaces": {"PCC": 1000, "APC": 100}}})
        self.progress({"event": "stage", "stage": "05_fit_curves", "status": "skipped",
                       "stats": {"skipped": "raw private reason"}, "seconds": 0})
        self.progress({"event": "stage", "stage": "01_split", "status": "reused", "stats": {"input_rows": 1200}})
        self.progress({"event": "stage", "stage": "05_fit_curves", "status": "reused",
                       "stats": {"skipped": "raw private reason"}})
        self.progress({"event": "stage", "stage": "04_climate_PCC", "status": "failed",
                       "log": "/private/stage.log", "seconds": 1.2, "error": "raw private exception"})
        self.progress({"event": "stage", "stage": "09_local_models", "status": "interrupted", "seconds": 1})
        result = self.console.getvalue()
        for text in ('Completed "01_split" (2.3s)', "input rows=1,200", "accepted=1,100", "rejected=100",
                     "PCC=1,000", 'Skipped "05_fit_curves"', 'Reused "01_split"',
                     'Failed "04_climate_PCC"', 'Interrupted "09_local_models"'):
            self.assertIn(text, result)
        self.assertNotIn("raw private", result)
        self.assertIn('Reused "05_fit_curves" | previously skipped', result)

    def test_setup_diagnostics_show_fixed_reasons_and_escape_mapping_paths(self):
        self.progress({"event": "setup_issue", "role": "mapping", "status": "missing",
                       "path": "/private/old\nmapping.json"})
        self.progress({"event": "mapping_entry", "status": "invalid", "source_key": "input\x1b.csv",
                       "location_value": {"secret": "PRIVATE_MARKER"},
                       "reason": "LOCATION_VALUE_MUST_BE_NONEMPTY_STRING"})
        self.progress({"event": "mapping_entry", "status": "normalized", "source_key": "/old/input.csv",
                       "location_value": "/private/site.csv", "source_filename": "input.csv",
                       "relative_location": "site.csv", "reason": "PATHS_NORMALIZED"})
        self.progress({"event": "setup_issue", "role": "unknown", "status": "missing", "path": "PRIVATE_MARKER"})
        result = self.console.getvalue()
        self.assertIn('Mapping JSON not found: "/private/old\\nmapping.json"', result)
        self.assertIn('source="input\\u001b.csv"', result)
        self.assertIn("LOCATION_VALUE_MUST_BE_NONEMPTY_STRING", result)
        self.assertIn('"input.csv" -> "site.csv"', result)
        self.assertNotIn("PRIVATE_MARKER", result)
        self.assertNotIn("\x1b", result)
        self.assertEqual(self.log.getvalue(), result)

    def test_stage_stats_whitelist_reports_scientific_support_without_raw_values(self):
        events = [
            ("02_pairs_PCC", {"total_pairs": 800, "sections_total": 90}),
            ("03_location_rate_PCC", {"rows": 800, "location_count": 10, "missing_location_rows": 5}),
            ("04_climate_PCC", {"rows": 800, "rows_with_climate": 790, "columns_added": 37, "enabled": True}),
            ("05_targets_PCC", {"rows": 790, "quarantined_pairs": 10, "blank_cells_zero_filled": 25}),
            ("05_fit_curves", {"fitted_curves": 3}),
            ("06_pci_PCC", {"rows": 790, "calculated": True, "end": {"mae": 1.12345, "export_gaps": 4}}),
            ("07_modeling_PCC", {"rows": 780, "features": 20, "airfield_groups": 10,
                                "targets": {"private family": {}}, "target_trainable": {"private family": True}}),
            ("07_partitions", {"declared_groups": 10, "validation_groups": 8, "folds": 3, "holdout_fraction": .2}),
            ("09_local_models", {"targets": 11, "status_counts": {"modeled": 8, "constant_target": 2,
                                                                   "no_valid_target": 1, "private label": 999}}),
        ]
        for stage, stats in events:
            stats["secret"] = "PRIVATE_CREDENTIAL_MARKER"
            self.progress({"event": "stage", "stage": stage, "status": "completed", "stats": stats})
        result = self.console.getvalue()
        for text in ("pairs=800", "histories=90", "locations=10", "pairs with climate=790", "climate fields=37",
                     "quarantined=10", "fitted curves=3", "end PCI MAE=1.123", "end export gaps=4",
                     "features=20", "target tasks=1", "eligible target tasks=1", "validation groups=8",
                     "holdout=20.0%", "modeled=8", "constant target=2", "no valid target=1"):
            self.assertIn(text, result)
        self.assertNotIn("private", result)
        self.assertNotIn("PRIVATE_CREDENTIAL_MARKER", result)

    def test_climate_events_explain_waiting_cache_and_feature_progress(self):
        base = {"event": "climate", "stage": "04_climate_PCC", "location_index": 2,
                "total_locations": 12, "locations_completed": 1, "cache_hits": 1,
                "requests": 1, "start_year": 2000, "end_year": 2020, "dataset": "era5-land"}
        for phase in ("cache_hit", "requesting", "downloaded", "parsing", "preparing"):
            self.progress({**base, "phase": phase})
        self.progress({**base, "phase": "location_failed", "locations_completed": 2,
                       "usable_locations": 1, "missing_locations": 1})
        self.progress({**base, "phase": "calculating", "rows_completed": 100, "total_rows": 1200})
        self.progress({**base, "phase": "complete", "rows_completed": 1200, "total_rows": 1200})
        result = self.console.getvalue()
        for value in ('Climate "04_climate_PCC"', "reusing cached weather", "waiting for CDS / downloading",
                      "location=2/12", "completed=2/12", "cache hits=1", "requests=1", "years=2000-2020",
                      "source=ERA5-Land", "location has no usable weather", "missing locations=1",
                      "rows=100/1,200", "rows=1,200/1,200", "climate processing complete"):
            self.assertIn(value, result)
        self.assertEqual(self.console.getvalue(), self.log.getvalue())
        self.assertEqual(self.console.flushes, 8)

    def test_climate_unknown_fields_and_invalid_counters_never_reach_output(self):
        self.progress({"event": "climate", "phase": "requesting", "stage": "04_climate\n\x1b",
                       "secret": "PRIVATE_MARKER", "coordinates": "PRIVATE_MARKER"})
        result = self.console.getvalue()
        self.assertIn('"04_climate\\n\\u001b"', result)
        self.assertNotIn("PRIVATE_MARKER", result)
        self.assertNotIn("\x1b", result)
        for event in ({"phase": "PRIVATE_MARKER"}, {"phase": "requesting", "cache_hits": "PRIVATE_MARKER"},
                      {"phase": "calculating", "rows_completed": float("nan")},
                      {"phase": "calculating", "rows_completed": True},
                      {"phase": "calculating", "rows_completed": -1}):
            self.progress({"event": "climate", **event})
        self.assertEqual(self.console.getvalue(), result)

    def test_heartbeat_formats_elapsed_time_and_ignores_raw_details(self):
        self.progress({"event": "heartbeat", "stage": "04_climate_PCC", "elapsed_seconds": 3725.9,
                       "idle_seconds": 65.2, "message": "PRIVATE_MARKER", "error": "PRIVATE_MARKER"})
        self.assertEqual(self.console.getvalue(),
                         'Still running "04_climate_PCC" | elapsed=1h 2m 5s; last subprocess output=1m 5s ago\n')
        self.progress({"event": "heartbeat", "stage": "stage\n\x1b", "elapsed_seconds": "PRIVATE_MARKER",
                       "idle_seconds": -1})
        result = self.console.getvalue()
        self.assertIn('Still running "stage\\n\\u001b"\n', result)
        self.assertNotIn("PRIVATE_MARKER", result)
        self.assertEqual(self.log.getvalue(), result)

    def test_terminal_controls_are_escaped_in_paths_stage_and_target_names(self):
        hostile = 'generated\n\r\x1b[2J\t\u202e"file'
        self.progress({"event": "input", "path": hostile, "rows": 1, "location": hostile, "location_status": "mapped"})
        self.progress({"event": "stage", "stage": hostile, "status": "started"})
        self.progress({"event": "target", "target": hostile, "material": hostile, "status": "started"})
        self.progress({"event": "warning", "stage": hostile, "code": "MISSING_LOCATION", "count": 1,
                       "message": "PRIVATE_CREDENTIAL_MARKER"})
        self.progress({"event": "setup_failed", "code": "bad\nPRIVATE_CREDENTIAL_MARKER", "log": hostile})
        result = self.console.getvalue()
        self.assertNotIn("\x1b", result)
        self.assertNotIn("\r", result)
        self.assertNotIn("\t", result)
        self.assertNotIn("\u202e", result)
        self.assertIn('generated\\n\\r\\u001b[2J\\t\\u202e\\"file', result)
        self.assertIn("PRIVATE_SETUP_FAILED", result)
        self.assertNotIn("PRIVATE_CREDENTIAL_MARKER", result)

    def test_run_summary_and_artifact_locations_are_visible(self):
        self.progress({"event": "target", "status": "completed", "target": "fam_freeze deduct_end",
                       "material": "PCC", "result_status": "baseline_selected"})
        self.progress({"event": "run", "status": "completed_with_warnings", "warnings": 2,
                       "summary": {"PCC": {"eligible_inspections": 20, "pairs": 10, "model_rows": 9,
                                            "features": 5, "airfield_groups": 3, "secret": "PRIVATE_MARKER"}},
                       "report": "/private/run/run_report.json"})
        self.progress({"event": "output", "dashboard": "/private/report/dashboard.html",
                       "models": "/private/models", "report": "/private/run/run_report.md"})
        result = self.console.getvalue()
        for text in ("baseline selected", "Completed with warnings run", "warnings=2", "accepted inspections=20",
                     "pairs=10", "model rows=9", "features=5", "groups=3", 'Dashboard: "/private/report/dashboard.html"',
                     'Models: "/private/models"', 'Run report: "/private/run/run_report.md"'):
            self.assertIn(text, result)
        self.assertNotIn("PRIVATE_MARKER", result)

    def test_quiet_and_broken_console_keep_log_sink_working(self):
        quiet = FullProgress(None, self.log)
        quiet({"event": "stage", "stage": "01_split", "status": "started"})
        self.assertEqual(self.console.getvalue(), "")
        self.assertIn("Started", self.log.getvalue())
        broken = BrokenStream()
        progress = FullProgress(broken, self.log)
        for status in ("started", "completed"):
            progress({"event": "stage", "stage": "02_pairs_PCC", "status": status})
        self.assertEqual(broken.writes, 1)
        self.assertIn('Completed "02_pairs_PCC"', self.log.getvalue())
        closed = io.StringIO()
        closed.close()
        FullProgress(closed, self.log)({"event": "stage", "stage": "03_location_rate_PCC", "status": "started"})
        self.assertIn('Started "03_location_rate_PCC"', self.log.getvalue())

    def test_unknown_events_and_non_numeric_stats_are_not_dumped(self):
        self.progress({"event": "unrecognized", "data": "PRIVATE_MARKER"})
        self.progress({"event": "stage", "stage": "01_split", "status": "completed",
                       "stats": {"input_rows": "PRIVATE_MARKER", "retained_rows": float("nan"),
                                 "excluded_rows": True, "surfaces": {"unknown": "PRIVATE_MARKER"}}})
        self.assertEqual(self.console.getvalue(), 'Completed "01_split"\n')
        self.assertEqual(self.console.flushes, 1)


if __name__ == "__main__":
    unittest.main()
