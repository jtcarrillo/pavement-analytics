import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCANNER = Path(__file__).resolve().parents[1] / "tools/check_public_tree.py"
spec = importlib.util.spec_from_file_location("public_check", SCANNER)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class PublicationTests(unittest.TestCase):
    def test_forced_input_file_and_disguised_binary_rejected(self):
        self.assertIn("UNAPPROVED_PATH", check.inspect("input.csv", "100644", b"x", {"README.md"}, []))
        self.assertIn("BINARY_OR_OVERSIZE", check.inspect("README.md", "100644", b"x\0y", {"README.md"}, []))
        self.assertIn("LINK_OR_SUBMODULE", check.inspect("README.md", "120000", b"outside", {"README.md"}, []))

    def test_private_denylist_matches_content_and_filename_without_echo(self):
        sentinel = "generated" + "-restricted"
        reasons = check.inspect("README.md", "100644", sentinel.encode(), {"README.md"}, [sentinel])
        self.assertEqual(reasons, {"PRIVATE_DENYLIST_MATCH"})
        self.assertNotIn(sentinel, str(reasons))

    def test_publication_allowlist_cannot_admit_data_extensions(self):
        with self.assertRaises(check.Rejected):
            check.allowed(b"README.md\nprivate.csv\n")

    def test_deleted_data_in_history_still_blocks_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tools").mkdir()
            shutil.copy2(SCANNER, root / "tools/check_public_tree.py")
            (root / ".publication-allowlist").write_text(".publication-allowlist\nREADME.md\ntools/check_public_tree.py\n")
            (root / "README.md").write_text("Generic tools\n")
            env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
                   "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.invalid"}
            def git(*args):
                subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)
            git("init", "-q")
            git("add", ".")
            git("commit", "-qm", "Initial source")
            (root / "generated.csv").write_text("generated\n")
            git("add", ".")
            git("commit", "-qm", "Temporary generated file")
            (root / "generated.csv").unlink()
            git("add", "-u")
            git("commit", "-qm", "Remove generated file")
            result = subprocess.run([os.sys.executable, "tools/check_public_tree.py", "--history"],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("UNAPPROVED_PATH", result.stdout)
            self.assertNotIn("generated.csv", result.stdout)


if __name__ == "__main__":
    unittest.main()
