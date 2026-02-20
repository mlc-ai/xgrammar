"""Tests that importing xgrammar does not modify the root logger (no side-effects on import)."""

import logging
import os
import subprocess
import sys


def _run_in_subprocess(code: str) -> subprocess.CompletedProcess:
    """Run code in a fresh interpreter so 'import xgrammar' happens after our setup."""
    python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
    env = os.environ.copy()
    env["PYTHONPATH"] = python_dir + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=10
    )


def test_import_xgrammar_does_not_change_root_logger_level():
    """Importing xgrammar must not change the root logger's level (reproduces the reported bug)."""
    code = """
import logging
# Default root level is WARNING (30). After the bug, it became INFO (20).
root = logging.getLogger()
assert root.level == logging.WARNING, f"expected WARNING, got {root.level}"
import xgrammar
root_after = logging.getLogger()
assert root_after.level == logging.WARNING, (
    f"import xgrammar changed root level from WARNING to {root_after.level}"
)
"""
    result = _run_in_subprocess(code)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def test_import_xgrammar_preserves_user_set_root_level():
    """If the user set the root logger to DEBUG (or any level), importing xgrammar must not override it."""
    code = """
import logging
logging.getLogger().setLevel(logging.DEBUG)
level_before = logging.getLogger().level
import xgrammar
level_after = logging.getLogger().level
assert level_before == level_after, (
    f"import xgrammar changed root level from {level_before} to {level_after}"
)
assert level_after == logging.DEBUG
"""
    result = _run_in_subprocess(code)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def test_enable_logging_only_affects_xgrammar_logger():
    """Calling enable_logging() must configure only the 'xgrammar' logger, not the root logger."""
    from xgrammar.support import logging as xgr_logging

    root = logging.getLogger()
    root_level_before = root.level
    root_handlers_before = len(root.handlers)

    xgr_logging.enable_logging()

    # Root logger must be unchanged
    assert (
        logging.getLogger().level == root_level_before
    ), "enable_logging() must not change root level"
    assert (
        len(logging.getLogger().handlers) == root_handlers_before
    ), "enable_logging() must not add root handlers"

    # xgrammar logger must be configured
    xgr_log = logging.getLogger("xgrammar")
    assert xgr_log.level == logging.INFO
    assert len(xgr_log.handlers) >= 1
