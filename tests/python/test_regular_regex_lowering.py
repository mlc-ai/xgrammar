"""Formats and numeric ranges lowered to one automaton accept what their CFG expansion accepts."""

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

CASES = [
    ({"type": "string", "format": "date"}, ['"2024-02-29"'], ['"2024-13-01"', '"2024-2-9"', '""']),
    (
        {"type": "string", "format": "uuid"},
        ['"123e4567-e89b-12d3-a456-426614174000"'],
        ['"123e4567-e89b-12d3-a456-42661417400"', '"123e4567e89b12d3a456426614174000"'],
    ),
    (
        {"type": "string", "format": "ipv4"},
        ['"192.168.0.1"', '"0.0.0.0"', '"255.255.255.255"'],
        ['"256.1.1.1"', '"1.2.3"', '"1.2.3.4.5"'],
    ),
    (
        {"type": "string", "format": "hostname"},
        ['"example.com"', '"a-b.c"'],
        ['"-a.com"', '"a..b"', '"A.com"'],
    ),
    (
        {"type": "string", "format": "uri-template"},
        ['"http://example.com/{id}"', '"/a/{b}/c"'],
        ['"http://example.com/{id"', '"a b"'],
    ),
    (
        {"type": "string", "format": "date-time"},
        ['"2024-01-02T03:04:05Z"', '"2024-01-02T03:04:05.678+09:00"'],
        ['"2024-01-02 03:04:05Z"', '"2024-01-02T25:00:00Z"'],
    ),
    (
        {"type": "integer", "minimum": -3, "maximum": 120},
        ["-3", "0", "7", "99", "120"],
        ["-4", "121", "007", "1.0"],
    ),
    ({"type": "integer", "minimum": 17}, ["17", "18", "1000000"], ["16", "-17", "017"]),
    ({"type": "integer", "maximum": -5}, ["-5", "-6", "-1000"], ["-4", "0", "5"]),
    (
        {"type": "number", "minimum": 1.5, "maximum": 2.5},
        ["1.5", "2", "2.25", "2.5"],
        ["1.4", "2.6", "3"],
    ),
]


@pytest.mark.parametrize("schema,valid,invalid", CASES)
def test_lowered_regex_language(schema, valid, invalid):
    grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=False, strict_mode=True)
    for text in valid:
        assert _is_grammar_accept_string(grammar, text), text
    for text in invalid:
        assert not _is_grammar_accept_string(grammar, text), text


def test_repeated_pattern_is_stable():
    """The cached automaton of a repeated pattern must not leak state between uses."""
    pattern = {"type": "string", "pattern": "^[A-Fa-f0-9]{6}$"}
    schema = {
        "type": "object",
        "properties": {name: pattern for name in ["a", "b", "c"]},
        "required": ["a", "b", "c"],
    }
    for _ in range(3):
        grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=False, strict_mode=True)
        assert _is_grammar_accept_string(grammar, '{"a": "00ff00", "b": "ABCDEF", "c": "123abc"}')
        assert not _is_grammar_accept_string(
            grammar, '{"a": "00ff0", "b": "ABCDEF", "c": "123abc"}'
        )
        assert not _is_grammar_accept_string(
            grammar, '{"a": "00ff00", "b": "ABCDEG", "c": "123abc"}'
        )
