"""Exercise rate-limit recovery without contacting the Hub or waiting."""

import importlib.util
from pathlib import Path

import pytest
from huggingface_hub.utils import HfHubHTTPError

spec = importlib.util.spec_from_file_location(
    "prepare_tokenizer_assets",
    Path(__file__).resolve().parents[2] / "scripts" / "prepare_tokenizer_assets.py",
)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class Response:
    def __init__(self, status_code, headers):
        self.status_code = status_code
        self.headers = headers
        self.request = None


def hub_error(status=429, headers=None, message="Rate limited"):
    return HfHubHTTPError(message, response=Response(status, headers or {}))


@pytest.mark.parametrize(
    "headers,message,expected",
    [
        ({"Retry-After": "39"}, "Rate limited", 40),
        ({"RateLimit": '"api";r=0;t=55'}, "Rate limited", 56),
        ({"Retry-After": "39", "RateLimit": '"api";r=0;t=55'}, "Rate limited", 56),
        ({}, "Retry after 39 seconds (0/1000 requests remaining)", 40),
        ({"Retry-After": "invalid"}, "Rate limited", 301),
    ],
)
def test_rate_limit_reset(headers, message, expected):
    assert assets.rate_limit_wait(hub_error(headers=headers, message=message)) == expected


def test_snapshot_retries_rate_limit(monkeypatch):
    calls = []
    waits = []

    def download(model_id, **kwargs):
        calls.append((model_id, kwargs))
        if len(calls) == 1:
            raise hub_error(message="Retry after 39 seconds")
        return "/cache/snapshots/pinned"

    monkeypatch.setattr(assets, "snapshot_download", download)
    monkeypatch.setattr(assets.time, "sleep", waits.append)
    assert assets.download_snapshot("example/model", "pinned") == "/cache/snapshots/pinned"
    assert calls[0] == calls[1]
    assert calls[0][1]["max_workers"] == 1
    assert waits == [40]


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_other_http_errors_are_not_retried(monkeypatch, status):
    error = hub_error(status=status)

    def download(*args, **kwargs):
        raise error

    monkeypatch.setattr(assets, "snapshot_download", download)
    with pytest.raises(HfHubHTTPError) as raised:
        assets.download_snapshot("example/model", "pinned")
    assert raised.value is error


def test_rate_limit_retries_are_bounded(monkeypatch):
    waits = []

    def download(*args, **kwargs):
        raise hub_error(headers={"Retry-After": "39"})

    monkeypatch.setattr(assets, "snapshot_download", download)
    monkeypatch.setattr(assets.time, "sleep", waits.append)
    with pytest.raises(HfHubHTTPError):
        assets.download_snapshot("example/model", "pinned", max_attempts=3)
    assert waits == [40, 40]
    waits.clear()
    with pytest.raises(HfHubHTTPError):
        assets.download_snapshot("example/model", "pinned", retry_budget=1)
    assert waits == []
