import re
import subprocess
from importlib.metadata import version as distribution_version
from pathlib import Path

import pytest

import xgrammar as xgr


def test_runtime_version_matches_distribution_metadata():
    assert xgr.__version__ == distribution_version("xgrammar")


def test_development_version_contains_git_commit():
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        pytest.skip("Git metadata is unavailable")

    exact_tag = subprocess.run(
        ["git", "describe", "--exact-match", "--tags", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if exact_tag.returncode == 0:
        pytest.skip("An exact release tag intentionally has no local commit component")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    match = re.search(r"\+g([0-9a-f]+)", xgr.__version__)
    assert match is not None
    assert commit.startswith(match.group(1))
