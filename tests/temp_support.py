"""Create generated test fixtures below the canonical OS temporary directory.

On macOS the default temporary path commonly traverses /var -> /private/var.
Resolve that OS-selected root before allocating a test directory. Do not resolve
later fixture paths: deliberately created symlinks must still be rejected by the
production path guards.
"""

from pathlib import Path
import tempfile


def temporary_directory():
    root = Path(tempfile.gettempdir()).resolve(strict=True)
    return tempfile.TemporaryDirectory(dir=root)
