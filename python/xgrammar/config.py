"""Global configuration for XGrammar."""

from contextlib import contextmanager

from .base import _core


def get_max_recursion_depth() -> int:
    """Get the maximum allowed recursion depth. The depth is shared per process.

    The maximum recursion depth is determined in the following order:

    1. Manually set via :py:func:`set_max_recursion_depth`
    2. ``XGRAMMAR_MAX_RECURSION_DEPTH`` environment variable (if set and is a valid integer <= 1,000,000)
    3. Default value of 10,000

    Returns
    -------
    max_recursion_depth : int
        The maximum allowed recursion depth.
    """
    return _core.config.get_max_recursion_depth()


def set_max_recursion_depth(max_recursion_depth: int) -> None:
    """Set the maximum allowed recursion depth. The depth is shared per process. This method is
    thread-safe.

    Parameters
    ----------
    max_recursion_depth : int
        The maximum allowed recursion depth.
    """
    _core.config.set_max_recursion_depth(max_recursion_depth)


@contextmanager
def max_recursion_depth(temp_depth: int):
    """A context manager for temporarily setting recursion depth.

    Examples
    --------
    >>> with recursion_depth(1000):
    ...     # recursion depth is 1000 here
    ...     pass
    >>> # recursion depth is restored to original value
    """
    prev_depth = get_max_recursion_depth()
    set_max_recursion_depth(temp_depth)
    try:
        yield
    finally:
        set_max_recursion_depth(prev_depth)


def get_serialization_version() -> str:
    """Get the serialization version number. The current version is "v4".

    Returns
    -------
    serialization_version : str
        The serialization version number.
    """
    return _core.config.get_serialization_version()
