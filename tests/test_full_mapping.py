"""Explicit mapping compatibility without guessing files or crossing boundaries."""
from pathlib import Path
import unittest

from pavement_analytics.full_mapping import normalize_location_mapping
from temp_support import temporary_directory


class FullMappingTests(unittest.TestCase):
    def setUp(self):
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.locations = self.root / "locations"
        self.locations.mkdir()
        self.public = self.root / "source"
        self.public.mkdir()
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.sources = [self.raw / "inspection_a.xlsx", self.raw / "inspection_b.csv"]
        for source in self.sources:
            source.write_text("generated inspection fixture")
        (self.locations / "site_a.csv").write_text('LOCATION(DEGREES)\n"0, 1"\n')
        (self.locations / "nested").mkdir()
        (self.locations / "nested/site_b.csv").write_text('LOCATION(DEGREES)\n"0, 2"\n')

    def normalize(self, payload, *, sources=None, locations=None):
        return normalize_location_mapping(payload, locations or self.locations,
                                          self.sources if sources is None else sources, self.public)

    def assert_invalid(self, payload, **kwargs):
        mapping, audit = self.normalize(payload, **kwargs)
        self.assertTrue(any(entry["status"] == "invalid" for entry in audit), audit)
        self.assertTrue(all(isinstance(entry["reason"], str) and entry["reason"] for entry in audit
                            if entry["status"] == "invalid"))
        return mapping, audit

    def test_relative_mapping_and_unused_basename_are_preserved(self):
        payload = {"inspection_a.xlsx": "site_a.csv", "inspection_b.csv": "nested/site_b.csv",
                   "unused_export.xlsx": "missing_site.csv"}
        mapping, audit = self.normalize(payload)
        self.assertEqual(mapping, payload)
        self.assertEqual(len(audit), 3)
        self.assertTrue(all(row["status"] == "valid" for row in audit))
        # Missing-file accounting belongs to the later location preflight.
        self.assertFalse((self.locations / "missing_site.csv").exists())

    def test_source_paths_and_contained_absolute_location_paths_normalize(self):
        mapping, audit = self.normalize({
            str(self.root / "old_project/inspection_a.xlsx"): str(self.locations / "site_a.csv"),
            "old_exports/inspection_b.csv": str(self.locations / "nested/site_b.csv"),
        })
        self.assertEqual(mapping, {"inspection_a.xlsx": "site_a.csv", "inspection_b.csv": "nested/site_b.csv"})
        self.assertTrue(all(row["status"] == "normalized" for row in audit))
        self.assertEqual({row["source_filename"] for row in audit}, set(mapping))
        self.assertEqual({row["relative_location"] for row in audit}, set(mapping.values()))
        self.assertFalse((self.root / "old_project").exists())

    def test_source_path_must_match_one_selected_basename(self):
        mapping, _ = self.assert_invalid({"old_exports/not_selected.xlsx": "site_a.csv"})
        self.assertEqual(mapping, {})
        duplicate_sources = [self.raw / "inspection_a.xlsx", self.root / "another/inspection_a.xlsx"]
        mapping, _ = self.assert_invalid({"inspection_a.xlsx": "site_a.csv"}, sources=duplicate_sources)
        self.assertEqual(mapping, {})

    def test_declared_location_file_does_not_make_source_basename_ambiguous(self):
        source = self.raw / "inspection.csv"
        source.write_text("generated inspection fixture")
        location = self.locations / "inspection.csv"
        location.write_text('LOCATION(DEGREES)\n"0, 1"\n')
        for key in ("inspection.csv", "old/path/inspection.csv"):
            for value in ("inspection.csv", str(location)):
                with self.subTest(key=key, value=value):
                    mapping, audit = self.normalize({key: value}, sources=[source, location])
                    self.assertEqual(mapping, {"inspection.csv": "inspection.csv"})
                    self.assertEqual(len(audit), 1)
                    self.assertIn(audit[0]["status"], {"valid", "normalized"})

    def test_unsafe_location_does_not_remove_an_ambiguous_source(self):
        source = self.raw / "inspection.csv"
        source.write_text("generated inspection fixture")
        outside = self.root / "other" / "inspection.csv"
        outside.parent.mkdir()
        outside.write_text("generated second inspection fixture")
        for value in (str(outside), "../other/inspection.csv"):
            with self.subTest(value=value):
                mapping, audit = self.assert_invalid(
                    {"unused.csv": value, "old/path/inspection.csv": "site_a.csv"},
                    sources=[source, outside])
                self.assertEqual(mapping, {})
                self.assertEqual([row["status"] for row in audit], ["invalid", "invalid"])

    def test_normalized_source_collisions_never_silently_choose_a_location(self):
        for other in ("site_a.csv", "nested/site_b.csv"):
            with self.subTest(other=other):
                mapping, audit = self.assert_invalid({"inspection_a.xlsx": "site_a.csv",
                                                     "old/inspection_a.xlsx": other})
                self.assertNotIn("inspection_a.xlsx", mapping)
                self.assertEqual([row["status"] for row in audit], ["invalid", "invalid"])

    def test_location_paths_cannot_escape_declared_directory(self):
        outside = self.root / "outside.csv"
        outside.write_text("generated outside file")
        for value in (str(outside), "../outside.csv", "nested/../../outside.csv", "nested/../site_a.csv"):
            with self.subTest(value=value):
                mapping, _ = self.assert_invalid({"inspection_a.xlsx": value})
                self.assertEqual(mapping, {})
        self.assertEqual(outside.read_text(), "generated outside file")

    def test_location_symlinks_and_public_tree_are_rejected(self):
        link = self.locations / "linked.csv"
        link.symlink_to(self.locations / "site_a.csv")
        mapping, _ = self.assert_invalid({"inspection_a.xlsx": "linked.csv"})
        self.assertEqual(mapping, {})
        public_location = self.public / "site.csv"
        public_location.write_text("generated public file")
        mapping, _ = self.assert_invalid({"inspection_a.xlsx": str(public_location)})
        self.assertEqual(mapping, {})
        self.assert_invalid({"inspection_a.xlsx": "site.csv"}, locations=self.public)
        self.assertTrue(link.is_symlink())

    def test_location_value_must_identify_a_file(self):
        for value in (".", "nested", str(self.locations), "nested/"):
            with self.subTest(value=value):
                mapping, _ = self.assert_invalid({"inspection_a.xlsx": value})
                self.assertEqual(mapping, {})

    def test_non_object_mapping_and_invalid_entry_types_have_safe_audits(self):
        for payload in (None, [], [{"inspection_a.xlsx": "site_a.csv"}], "site_a.csv"):
            with self.subTest(payload=payload):
                mapping, audit = self.assert_invalid(payload)
                self.assertEqual(mapping, {})
                self.assertTrue(all(row["source_key"] is None and row["location_value"] is None for row in audit))
        cases = [{None: "site_a.csv"}, {3: "site_a.csv"}, {"": "site_a.csv"},
                 {"   ": "site_a.csv"}, {"inspection_a.xlsx": None},
                 {"inspection_a.xlsx": ""}, {"inspection_a.xlsx": "   "},
                 {"inspection_a.xlsx": ["site_a.csv"]},
                 {"inspection_a.xlsx": {"private_marker": "site_a.csv"}}]
        for payload in cases:
            with self.subTest(payload=payload):
                mapping, audit = self.assert_invalid(payload)
                self.assertEqual(mapping, {})
                self.assertTrue(all(row["source_key"] is None or isinstance(row["source_key"], str) for row in audit))
                self.assertTrue(all(row["location_value"] is None or isinstance(row["location_value"], str) for row in audit))
                self.assertNotIn("private_marker", str(audit))

    def test_one_bad_entry_does_not_hide_other_mapping_diagnostics(self):
        mapping, audit = self.assert_invalid({"inspection_a.xlsx": "site_a.csv",
                                             "inspection_b.csv": "../outside.csv",
                                             "third.xlsx": None})
        self.assertEqual(mapping, {"inspection_a.xlsx": "site_a.csv"})
        self.assertEqual(len(audit), 3)
        self.assertEqual(sum(row["status"] == "invalid" for row in audit), 2)


if __name__ == "__main__":
    unittest.main()
