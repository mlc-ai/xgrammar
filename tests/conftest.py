import pytest

try:
    import pytest_run_parallel  # noqa: F401

    PARALLEL_RUN_AVALIABLE = True
except ModuleNotFoundError:
    PARALLEL_RUN_AVALIABLE = False


def pytest_configure(config):
    if not PARALLEL_RUN_AVALIABLE:
        config.addinivalue_line(
            "markers", "thread_unsafe: mark the test function as single-threaded"
        )


if not PARALLEL_RUN_AVALIABLE:

    @pytest.fixture
    def num_parallel_threads():
        return 1
