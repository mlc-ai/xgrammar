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
    """Get the serialization version number.

    Returns
    -------
    serialization_version : str
        The serialization version number.
    """
    return str(_core.config.get_serialization_version())


def set_json_schema_cache_enabled(enabled: bool) -> None:
    """Turn the JSON schema grammar cache on or off. The cache is shared per process and is on
    by default. This method is thread-safe.

    The cache keeps the grammar converted from each JSON schema, together with its conversion
    options (such as the style, ``any_order``, ``excludes`` and whitespace settings), so a later
    conversion of the same schema reuses it instead of converting it again. It is used by
    :py:meth:`Grammar.from_json_schema`, by the ``json_schema`` contents of
    :py:meth:`Grammar.from_structural_tag`, and by the :py:class:`GrammarCompiler` methods that
    call them. It mostly speeds up structural tags whose tools repeat across requests. Converted
    grammars are identical with and without the cache.

    It is separate from the :py:class:`GrammarCompiler` caches: ``cache_enabled``,
    ``cache_limit_bytes`` and :py:meth:`GrammarCompiler.clear_cache` do not affect it.

    Parameters
    ----------
    enabled : bool
        Whether to use the cache. Turning it off also removes all its entries.
    """
    _core.config.set_json_schema_cache_enabled(enabled)


def get_json_schema_cache_enabled() -> bool:
    """Whether the JSON schema grammar cache is on. See :py:func:`set_json_schema_cache_enabled`.

    Returns
    -------
    enabled : bool
        Whether the cache is on.
    """
    return bool(_core.config.get_json_schema_cache_enabled())


def set_json_schema_cache_limit_bytes(limit_bytes: int) -> None:
    """Set the memory limit of the JSON schema grammar cache (see
    :py:func:`set_json_schema_cache_enabled`). Least recently used entries are evicted to stay
    within it. By default there is no limit, like the compiler's ``cache_limit_bytes``. A tool
    schema takes roughly 10 to 80 KB, so 64 MiB holds about 1,000 to 5,000 of them. This method
    is thread-safe.

    Parameters
    ----------
    limit_bytes : int
        The limit in bytes, or -1 for no limit.

    Raises
    ------
    RuntimeError
        When ``limit_bytes`` is less than -1.
    """
    _core.config.set_json_schema_cache_limit_bytes(limit_bytes)


def get_json_schema_cache_limit_bytes() -> int:
    """Get the memory limit of the JSON schema grammar cache. See
    :py:func:`set_json_schema_cache_limit_bytes`.

    Returns
    -------
    limit_bytes : int
        The limit in bytes, or -1 for no limit.
    """
    return int(_core.config.get_json_schema_cache_limit_bytes())
