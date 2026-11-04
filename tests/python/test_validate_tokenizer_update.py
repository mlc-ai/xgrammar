"""Check CI gating without dispatching workflows or merging a PR."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def validator(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "validate_tokenizer_update",
        Path(__file__).resolve().parents[2] / "scripts" / "validate_tokenizer_update.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("UPDATE_SHA", "a" * 40)
    monkeypatch.setenv("UPDATE_PR", "123")
    return module


@pytest.mark.parametrize("conclusion", ["success", "failure", "cancelled"])
def test_merge_requires_all_workflows_to_pass(validator, monkeypatch, conclusion):
    calls = []
    queried = set()

    def runs(workflow, sha):
        assert sha == "a" * 40
        old = {"databaseId": 1, "status": "completed", "conclusion": "success", "url": "old"}
        if workflow not in queried:
            queried.add(workflow)
            return [old]
        return [
            {"databaseId": 2, "status": "completed", "conclusion": conclusion, "url": "new"},
            old,
        ]

    monkeypatch.setattr(validator, "runs", runs)
    monkeypatch.setattr(validator, "gh", lambda *args: calls.append(args))
    if conclusion == "success":
        validator.main()
        assert calls[-1] == (
            "pr",
            "merge",
            "123",
            "--auto",
            "--squash",
            "--match-head-commit",
            "a" * 40,
        )
    else:
        with pytest.raises(RuntimeError, match="PR remains unmerged"):
            validator.main()
        assert all(c[0] != "pr" for c in calls)
    dispatches = [c for c in calls if c[0] == "workflow"]
    assert len(dispatches) == 4
    docs = next(c for c in dispatches if c[2] == "documentation.yaml")
    assert "build_only=true" in docs
    assert "docs_ref=ci/update-tokenizer-assets" in docs


def test_ci_timeout_does_not_merge(validator, monkeypatch):
    calls = []
    clock = iter([0, 10801])
    monkeypatch.setattr(validator, "runs", lambda *args: [])
    monkeypatch.setattr(validator, "gh", lambda *args: calls.append(args))
    monkeypatch.setattr(validator.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="Timed out"):
        validator.main()
    assert all(c[0] != "pr" for c in calls)
