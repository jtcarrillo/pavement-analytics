import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pavement_analytics.cli import main
from pavement_analytics.study import isolate_climate_sites, resolve_options, snapshot_locations
from pavement_analytics.security import Boundary, SafeError
from temp_support import temporary_directory


class StudyIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.public = self.root / "source"
        self.public.mkdir()
        self.work = self.root / "work"
        self.source = self.data / "inspections.csv"
        # Each section has only one inspection: pairing would have no rows.
        pd.DataFrame([
            {"SectionUniqueID": f"S{group}-{row}", "NetworkID": f"G{group}",
             "Date": "2020-01-01", "PCI": 100, "Inspected Surface Type": "PCC",
             "64L deduct": 3 + row * 2, "70L deduct": 1,
             "historic_temperature": row + group / 100,
             "Subgrade Category": "a" if row % 2 else "b"}
            for group in range(12) for row in range(5)]).to_csv(self.source, index=False)
        self.options = self.data / "pavement-study.json"
        self.options.write_text(json.dumps({"folds": 3, "deduct_families": {"thermal": ["64", "70"]},
                                           "climate": {"mode": "existing", "features": ["historic_temperature"]}}))

    def run_study(self, *args):
        stream = io.StringIO()
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}), patch(
                "pavement_analytics.cli.source_root", return_value=self.public):
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                code = main(["study", str(self.data), "--workspace", str(self.work), *args])
        if code:
            log = self.work / "logs/runtime.log"
            self.fail(stream.getvalue() + (log.read_text() if log.exists() else ""))
        self.assertEqual(stream.getvalue().strip(), "PRIVATE_STUDY_COMPLETE")

    def test_entire_existing_climate_path_models_levels_and_consolidates(self):
        self.run_study()
        out = self.work / "results"
        dataset = pd.read_csv(out / "dataset.csv")
        self.assertEqual(len(dataset), 60)
        self.assertEqual(set(dataset.level_thermal), {4, 6, 8, 10, 12})
        slots = json.loads((out / "run_report.json").read_text())["slots"]
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["target"], "level_thermal")
        self.assertGreater(slots[0]["mae_skill"], .5)
        features = json.loads((out / "features.json").read_text())["features"]
        self.assertIn("historic_temperature", features)
        self.assertNotIn("PCI", features)
        self.assertFalse(any("deduct" in f for f in features))
        for name in ("dashboard.html", "target_summary.csv", "feature_importance.csv", "manifest.json"):
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(json.loads((out / "study_status.json").read_text())["status"], "complete")
        self.assertFalse(json.loads((out / "study_audit.json").read_text())["partition"]["shared_climate_sites_joined"])
        self.assertIsNone(json.loads((out / "study_audit.json").read_text())["lookback_days"])
        self.assertNotIn("Historical window: 365", (out / "dashboard.html").read_text())
        self.assertEqual(list(self.public.iterdir()), [])

    def test_location_csv_discovery_does_not_treat_metadata_as_inspections(self):
        self.options.write_text(json.dumps({"models": False, "folds": 3}))
        location = self.data / "location.csv"
        pd.DataFrame({"LOCATION(DEGREES)": ["0, 0"]}).to_csv(location, index=False)
        (self.data / "data_file_mapping.json").write_text(json.dumps({self.source.name: location.name}))
        def climate(table, config, out, source_map=None):
            self.assertEqual(source_map, {"input_0001.csv": self.source.name})
            self.assertTrue(Path(config["mapping"]).is_file())
            self.assertNotEqual(Path(config["mapping"]).parent, self.data)
            result = table.copy()
            result["era5_test"] = np.arange(len(result))
            result["climate_status"] = "ok"
            result["climate_location_id"] = "one-grid"
            return result, ["era5_test"], {"eligible_rows": len(result), "mode": "era5"}
        with patch("pavement_analytics.climate.enrich", side_effect=climate):
            self.run_study()
        setup = json.loads((self.work / "setup_report.json").read_text())
        self.assertEqual(len(setup["sources"]), 1)
        self.assertEqual(pd.read_csv(self.work / "results/partitions.csv").shape[0], 1)

    def test_site_union_is_transitive_and_preserves_declared_groups(self):
        frame = pd.DataFrame({"group_id": ["a", "b", "b", "c", "d"],
                              "climate_location_id": ["x", "x", "y", "y", "z"]})
        result, audit = isolate_climate_sites(frame)
        self.assertEqual(result.group_id.nunique(), 2)
        self.assertEqual(result.declared_group_id.tolist(), frame.group_id.tolist())
        self.assertEqual(audit["validation_groups"], 2)

    def test_unknown_profile_options_are_rejected(self):
        self.options.write_text(json.dumps({"climte": {}}))
        with self.assertRaisesRegex(SafeError, "UNKNOWN_STUDY_OPTION"):
            resolve_options(self.data, self.public)

    def test_null_optional_climate_paths_use_defaults(self):
        self.options.write_text(json.dumps({"climate": {"cache_dir": None, "csv_dir": None, "mapping": None}}))
        options, _ = resolve_options(self.data, self.public)
        self.assertEqual(options["climate"], {})

    def test_mapping_cannot_escape_location_directory(self):
        mapping = self.data / "mapping.json"
        mapping.write_text(json.dumps({"input.csv": "../outside.csv"}))
        boundary = Boundary.create(self.public, self.work)
        with self.assertRaisesRegex(SafeError, "INVALID_LOCATION_MAPPING"):
            snapshot_locations(boundary, {"mapping": str(mapping)})

    def test_actions_rejected_before_output_is_created(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}), contextlib.redirect_stderr(io.StringIO()):
            result = main(["study", str(self.data), "--workspace", str(self.work)])
        self.assertEqual(result, 2)
        self.assertFalse(self.work.exists())


if __name__ == "__main__":
    unittest.main()
