"""Selected-input failures are reported together without raw diagnostics."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from pavement_analytics.full_preflight import check_setup_inputs
from pavement_analytics.security import SafeError
from temp_support import temporary_directory


class FullPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_available_files_and_directory_are_checked_without_changing_them(self):
        config = {}
        for role in ("mapping", "cdv_curves", "deduct_curves", "families"):
            path = self.root / (role + ".json")
            path.write_bytes(b"{}\n")
            config[role] = str(path)
        locations = self.root / "locations"
        locations.mkdir()
        (locations / "generated.csv").write_bytes(b"generated\n")
        config["csv_dir"] = str(locations)
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        results = check_setup_inputs(config)
        self.assertEqual([row["role"] for row in results],
                         ["mapping", "csv_dir", "cdv_curves", "deduct_curves", "families"])
        self.assertTrue(all(row["status"] == "ready" for row in results))
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, before)

    def test_all_missing_paths_are_reported_and_unselected_options_are_skipped(self):
        config = {"mapping": str(self.root / "missing.json"),
                  "csv_dir": str(self.root / "missing_directory"), "cdv_curves": None}
        results = check_setup_inputs(config)
        self.assertEqual(results, [
            {"role": "mapping", "path": config["mapping"], "status": "missing"},
            {"role": "csv_dir", "path": config["csv_dir"], "status": "missing"},
        ])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_wrong_filesystem_types_are_distinguished(self):
        directory = self.root / "directory"
        directory.mkdir()
        file = self.root / "file"
        file.write_text("generated")
        results = check_setup_inputs({"mapping": directory, "csv_dir": file})
        self.assertEqual([row["status"] for row in results], ["not_file", "not_directory"])

    def test_blank_optional_paths_are_skipped_but_required_paths_are_checked(self):
        config = {role: "" for role in
                  ("mapping", "csv_dir", "cdv_curves", "deduct_curves", "families")}
        results = check_setup_inputs(config)
        self.assertEqual([row["role"] for row in results],
                         ["mapping", "csv_dir", "cdv_curves"])
        self.assertTrue(all(row["path"] == str(Path("")) for row in results))
        self.assertTrue(all(isinstance(row["status"], str) and row["status"]
                            for row in results))

    def test_unreadable_file_and_directory_do_not_leak_error_details(self):
        mapping = self.root / "mapping.json"
        mapping.write_text("{}")
        locations = self.root / "locations"
        locations.mkdir()
        curves = self.root / "curves.json"
        curves.write_text("{}")
        original_open = Path.open

        def open_file(path, *args, **kwargs):
            if path == mapping:
                raise PermissionError("generated-sensitive-file-detail")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", open_file), \
                patch.object(os, "scandir", side_effect=PermissionError("generated-sensitive-directory-detail")):
            results = check_setup_inputs({"mapping": mapping, "csv_dir": locations, "cdv_curves": curves})
        self.assertEqual([row["status"] for row in results], ["unreadable", "unreadable", "ready"])
        self.assertNotIn("generated-sensitive", str(results))
        self.assertTrue(all(set(row) == {"role", "path", "status"} for row in results))

    def test_symlink_inputs_remain_forbidden(self):
        target = self.root / "mapping.json"
        target.write_text("{}")
        link = self.root / "mapping_link.json"
        link.symlink_to(target)
        with self.assertRaisesRegex(SafeError, "SYMLINK_PATH_FORBIDDEN"):
            check_setup_inputs({"mapping": link})
        self.assertTrue(link.is_symlink())
        self.assertEqual(target.read_text(), "{}")


if __name__ == "__main__":
    unittest.main()
