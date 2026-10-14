"""Validate automated pin updates without Hub requests."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def updater(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "update_tokenizer_revisions", scripts / "update_tokenizer_revisions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(tmp_path, models):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"models": models}) + "\n")
    return path


def model(repo_id, **kwargs):
    return {"repo_id": repo_id, "revision": "a" * 40, "requires_token": False, **kwargs}


def test_update_tracks_refs_and_preserves_fixed_models(updater, tmp_path):
    path = manifest(
        tmp_path,
        [model("public"), model("tagged", update_ref="v2"), model("fixed", update_ref=None)],
    )
    calls = []

    def info(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return SimpleNamespace(sha="b" * 40)

    changes = updater.update_manifest(path, SimpleNamespace(model_info=info))
    assert [change[0] for change in changes] == ["public", "tagged"]
    assert calls == [
        ("public", {"revision": "main", "expand": ["sha"]}),
        ("tagged", {"revision": "v2", "expand": ["sha"]}),
    ]
    assert json.loads(path.read_text())["models"][2]["revision"] == "a" * 40


@pytest.mark.parametrize("include,expected", [(False, []), (True, ["gated"])])
def test_token_required_assets(updater, tmp_path, include, expected):
    path = manifest(tmp_path, [model("gated", requires_token=True)])
    calls = []

    def info(repo_id, **kwargs):
        calls.append(repo_id)
        return SimpleNamespace(sha="b" * 40)

    updater.update_manifest(path, SimpleNamespace(model_info=info), include)
    assert calls == expected


def test_no_change_preserves_manifest_bytes(updater, tmp_path):
    path = manifest(tmp_path, [model("public")])
    before = path.read_bytes()
    api = SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40))
    assert updater.update_manifest(path, api) == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("failure", [ValueError("Hub lookup failed"), "invalid-sha"])
def test_failed_resolution_does_not_write_partial_updates(updater, tmp_path, failure):
    path = manifest(tmp_path, [model("first"), model("second")])
    before = path.read_bytes()

    def info(repo_id, **kwargs):
        if repo_id == "first":
            return SimpleNamespace(sha="b" * 40)
        if isinstance(failure, Exception):
            raise failure
        return SimpleNamespace(sha=failure)

    with pytest.raises(ValueError):
        updater.update_manifest(path, SimpleNamespace(model_info=info))
    assert path.read_bytes() == before
