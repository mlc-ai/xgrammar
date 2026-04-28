import os
from pathlib import Path

import pytest

try:
    import pytest_run_parallel  # noqa: F401

    PARALLEL_RUN_AVAILABLE = True
except ModuleNotFoundError:
    PARALLEL_RUN_AVAILABLE = False


def _hf_token_available() -> bool:
    """Check whether a HuggingFace token is available via env vars or cached login."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    # huggingface-cli login stores token here
    return Path.home().joinpath(".cache", "huggingface", "token").is_file()


def _hf_token_explicitly_disabled(config) -> bool:
    """Return whether pytest mark expression explicitly excludes hf-token tests."""
    markexpr = getattr(config.option, "markexpr", "") or config.getoption("markexpr", "")
    return "not hf_token_required" in (markexpr or "")


def pytest_configure(config):
    if not PARALLEL_RUN_AVAILABLE:
        config.addinivalue_line(
            "markers", "thread_unsafe: mark the test function as single-threaded"
        )


def pytest_collection_modifyitems(config, items):
    if _hf_token_available():
        return
    skip_no_token = pytest.mark.skip(
        reason="HF_TOKEN not set (run `huggingface-cli login` or set HF_TOKEN env var)"
    )
    for item in items:
        if "hf_token_required" in item.keywords:
            item.add_marker(skip_no_token)


if not PARALLEL_RUN_AVAILABLE:

    @pytest.fixture
    def num_parallel_threads():
        return 1
