"""Global configuration for XGrammar."""

from contextlib import contextmanager
from typing import Dict

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
    """Get the serialization version number.

    Returns
    -------
    serialization_version : str
        The serialization version number.
    """
    return str(_core.config.get_serialization_version())


def get_cross_grammar_cache_max_memory() -> int:
    """Get the memory budget of the cross-grammar cache, in bytes.

    The cross-grammar cache is shared per process. It keeps the grammar converted from each JSON
    schema (together with its conversion options, such as the style, ``any_order``,
    ``excludes`` and whitespace settings), so a later conversion of the same schema reuses it.
    This mostly speeds up building structural tags whose tools repeat across requests, and
    :py:meth:`Grammar.from_json_schema`. Converted grammars are identical with and without the
    cache.

    Returns
    -------
    max_memory_bytes : int
        The budget in bytes. 0 means the cache is disabled, and -1 means it is unbounded.
    """
    return int(_core.config.get_cross_grammar_cache_max_memory())


def set_cross_grammar_cache_max_memory(max_memory_bytes: int) -> None:
    """Set the memory budget of the cross-grammar cache. Least recently used entries are evicted
    once the budget is exceeded. The default budget is 64 MiB. This method is thread-safe.

    Parameters
    ----------
    max_memory_bytes : int
        The budget in bytes. 0 disables the cache, and -1 removes the bound.
    """
    _core.config.set_cross_grammar_cache_max_memory(max_memory_bytes)


def clear_cross_grammar_cache() -> None:
    """Remove every entry from the cross-grammar cache and reset its counters."""
    _core.config.clear_cross_grammar_cache()


def get_cross_grammar_cache_stats() -> Dict[str, int]:
    """Get the counters of the cross-grammar cache.

    Returns
    -------
    stats : Dict[str, int]
        ``hits``, ``misses``, ``evictions``, ``num_entries``, ``memory_bytes`` (estimated) and
        ``max_memory_bytes``.
    """
    names = ("hits", "misses", "evictions", "num_entries", "memory_bytes", "max_memory_bytes")
    values = _core.config.get_cross_grammar_cache_stats()
    return {name: int(value) for name, value in zip(names, values)}
