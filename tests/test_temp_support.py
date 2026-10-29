"""Reproduce an OS temporary-root alias without weakening path boundaries."""

from pathlib import Path
import unittest
from unittest.mock import patch

from pavement_analytics.security import Boundary, SafeError, reject_links
from temp_support import temporary_directory


class TemporaryRootTests(unittest.TestCase):
    def test_os_alias_is_resolved_before_fixture_creation(self):
        with temporary_directory() as directory:
            root = Path(directory)
            real = root / "real-temp"
            real.mkdir()
            alias = root / "temp-alias"
            alias.symlink_to(real, target_is_directory=True)
            with patch("tempfile.gettempdir", return_value=str(alias)):
                with temporary_directory() as fixture:
                    child = Path(fixture)
                    self.assertEqual(child.parent, real)
                    reject_links(child)
                    public = child / "code"
                    public.mkdir()
                    boundary = Boundary.create(public, child / "work")
                    nested_link = boundary.workspace / "link"
                    nested_link.symlink_to(public, target_is_directory=True)
                    with self.assertRaisesRegex(SafeError, "SYMLINK_PATH_FORBIDDEN"):
                        boundary.path(nested_link / "data.csv")
                    # Direct runtime use of the original alias stays forbidden.
                    with self.assertRaisesRegex(SafeError, "SYMLINK_PATH_FORBIDDEN"):
                        reject_links(alias / child.name)
                self.assertFalse(child.exists())

    def test_generated_root_is_canonical_and_cleaned_on_exception(self):
        child = None
        with self.assertRaisesRegex(RuntimeError, "generated failure"):
            with temporary_directory() as directory:
                child = Path(directory)
                self.assertEqual(child, child.resolve(strict=True))
                (child / "generated.txt").write_text("temporary")
                raise RuntimeError("generated failure")
        self.assertFalse(child.exists())


if __name__ == "__main__":
    unittest.main()
