import contextlib
import io
import os
from pathlib import Path
import subprocess
from temp_support import temporary_directory
import unittest
from unittest.mock import patch

from pavement_analytics.cli import main
from pavement_analytics.security import Boundary, SafeError, sync_private


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.public = self.root / "code"
        self.public.mkdir()
        self.boundary = Boundary.create(self.public, self.root / "work")

    def test_nested_workspaces_and_parent_traversal_are_rejected(self):
        with self.assertRaises(SafeError):
            Boundary.create(self.public, self.public / "data")
        with self.assertRaises(SafeError):
            Boundary.create(self.public, self.root)
        with self.assertRaises(SafeError):
            self.boundary.path("../code/file.csv")

    def test_symlink_aliases_and_symlink_output_are_rejected(self):
        alias = self.boundary.workspace / "alias"
        alias.symlink_to(self.public, target_is_directory=True)
        with self.assertRaisesRegex(SafeError, "SYMLINK"):
            self.boundary.path(alias / "file.csv")
        log = self.boundary.workspace / "logs"
        log.mkdir()
        private_target = self.root / "sentinel"
        private_target.write_text("unchanged")
        (log / "runtime.log").symlink_to(private_target)
        with patch.dict(os.environ, {"PAVEMENT_PRIVATE_EXECUTION": "1", "GITHUB_ACTIONS": "false"}):
            with contextlib.redirect_stderr(io.StringIO()):
                result = main(["--workspace", str(self.boundary.workspace), "init"])
        self.assertEqual(result, 2)
        self.assertEqual(private_target.read_text(), "unchanged")

    def test_outputs_cannot_be_in_private_git_checkout(self):
        folder = self.boundary.workspace / "sources"
        folder.mkdir()
        (folder / ".git").mkdir()
        with self.assertRaisesRegex(SafeError, "VERSIONED"):
            self.boundary.output(folder / "results")

    def test_cli_refuses_github_actions_even_with_opt_in(self):
        stderr = io.StringIO()
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "PAVEMENT_PRIVATE_EXECUTION": "1"}):
            with contextlib.redirect_stderr(stderr):
                result = main(["--workspace", str(self.boundary.workspace), "init"])
        self.assertEqual(result, 2)
        self.assertIn("GITHUB_ACTIONS_DATA_EXECUTION_FORBIDDEN", stderr.getvalue())
        self.assertNotIn(str(self.boundary.workspace), stderr.getvalue())

    def test_unknown_cli_values_are_never_echoed(self):
        stream = io.StringIO()
        sentinel = "synthetic-private-argument"
        with contextlib.redirect_stderr(stream), contextlib.redirect_stdout(stream):
            result = main(["--workspace", str(self.boundary.workspace), sentinel])
        self.assertEqual(result, 2)
        self.assertNotIn(sentinel, stream.getvalue())

    def test_source_fetch_requires_pin_and_runtime_ssh_remote(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false", "PAVEMENT_PRIVATE_EXECUTION": "1",
                                     "PAVEMENT_DATA_REMOTE": "https://example.invalid/private.git"}):
            with patch("subprocess.run") as execute:
                with self.assertRaisesRegex(SafeError, "FULL_COMMIT"):
                    sync_private(self.boundary, "main")
                with self.assertRaisesRegex(SafeError, "SSH_REMOTE"):
                    sync_private(self.boundary, "a" * 40)
                execute.assert_not_called()

    def test_source_fetch_captures_private_git_error(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false", "PAVEMENT_PRIVATE_EXECUTION": "1",
                                     "PAVEMENT_DATA_REMOTE": "git@example.invalid:repo.git"}):
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(
                1, "git", stderr=b"synthetic-private-error")):
                with self.assertRaisesRegex(SafeError, "^PRIVATE_SOURCE_FETCH_FAILED$"):
                    sync_private(self.boundary, "a" * 40)

    def test_source_fetch_pins_revision_and_keeps_checkout_external(self):
        commit = "a" * 40
        calls = []
        def execute(command, **kwargs):
            calls.append((command, kwargs))
            root = Path(kwargs["cwd"])
            output = b""
            if "init" in command:
                (root / "checkout").mkdir()
            elif "rev-parse" in command:
                output = (commit + "\n").encode()
            elif "ls-tree" in command:
                output = b"100644 blob " + b"b" * 40 + b"\tgenerated.csv\0"
            elif "--detach" in command:
                (root / "checkout/generated.csv").write_text("generated\n")
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr=b"")
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false", "PAVEMENT_PRIVATE_EXECUTION": "1",
                                     "PAVEMENT_DATA_REMOTE": "git@example.invalid:repo.git"}):
            with patch("subprocess.run", side_effect=execute):
                destination = sync_private(self.boundary, commit)
        self.assertTrue((destination / "generated.csv").is_file())
        self.assertTrue(self.boundary.workspace in destination.parents)
        self.assertTrue(all(options["stderr"] == subprocess.PIPE for _, options in calls))
        self.assertTrue(any("fetch" in command and commit in command for command, _ in calls))
        self.assertIn("StrictHostKeyChecking=yes", calls[0][1]["env"]["GIT_SSH_COMMAND"])


if __name__ == "__main__":
    unittest.main()
