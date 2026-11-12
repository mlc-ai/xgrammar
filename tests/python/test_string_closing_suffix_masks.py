"""Tokens that close a string and continue are resolved per suffix and still equal eager masks."""

import itertools

import pytest
import torch

import xgrammar as xgr

SUFFIXES = [
    b'"',
    b'",',
    b'", ',
    b'", "',
    b'", "t',
    b'", "t"',
    b'", "t": ',
    b'", "t": "',
    b'", "t": "c',
    b'", "u',
    b'", "tt',
    b'"}',
    b'"}x',
    b'"]',
    b'" ,',
    b'":',
    b'": ',
    b'": 1',
    b'""',
    b'","',
]
VOCAB = [bytes([i]) for i in range(256)]
# The same suffix after different content, after no content, and after an escape.
for content in [b"", b"a", b"ab", b"abc", b"\\n", b"\\u0041", "中".encode(), b"a\\"]:
    VOCAB += [content + suffix for suffix in SUFFIXES]
VOCAB += [b'a\x01"', b"\\", b'\\x",', b"<stop>"]
STOP = len(VOCAB) - 1
# Enough accepted tokens for the bitset representation, plus duplicates.
VOCAB += ["".join(s).encode() for s in itertools.product("abcdefg", repeat=4)]
VOCAB += [b'ab",', b'", "t']

STRING = {"type": "string"}
CASES = [
    (
        {"type": "object", "properties": {"s": STRING, "t": STRING}, "required": ["s", "t"]},
        b'{"s": "ab", "t": "c"}',
    ),
    # The property after the string is optional, so both a comma and a brace may follow.
    (
        {"type": "object", "properties": {"s": STRING, "t": STRING}, "required": ["s"]},
        b'{"s": "a\\nb", "t": "c"}',
    ),
    ({"type": "array", "items": STRING}, b'["a", "ab", ""]'),
    ({"type": "array", "items": STRING, "minItems": 1, "maxItems": 2}, b'["ab", "c"]'),
    # A string key of a free-form object is followed by a colon, a string value by a comma.
    ({"type": "object", "additionalProperties": STRING}, b'{"a": "b", "t": "ab"}'),
    ({"anyOf": [STRING, {"type": "array", "items": STRING}]}, b'"abc"'),
    # A bounded string is not a plain string rule and keeps the generic path.
    (
        {"type": "object", "properties": {"s": {"type": "string", "maxLength": 3}, "t": STRING}},
        b'{"s": "ab", "t": "c"}',
    ),
]


def mask(matcher):
    result = xgr.allocate_token_bitmask(1, len(VOCAB))
    matcher.fill_next_token_bitmask(result)
    if len(VOCAB) % 32:
        result[0, -1] &= (1 << (len(VOCAB) % 32)) - 1
    return result


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("schema,text", CASES)
def test_string_closing_masks_match_eager(schema, text, cache):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])

    def compile(dynamic):
        compiler = xgr.GrammarCompiler(
            info, cache_enabled=cache, enable_dynamic_compilation=dynamic, max_threads=1
        )
        return compiler.compile_json_schema(schema, any_whitespace=False, strict_mode=True)

    eager, dynamic = compile(False), compile(True)
    for replay in range(2):
        expected, actual = xgr.GrammarMatcher(eager), xgr.GrammarMatcher(dynamic)
        for offset in range(len(text) + 1):
            torch.testing.assert_close(mask(actual), mask(expected), rtol=0, atol=0)
            if offset < len(text):
                accepted = expected.accept_token(text[offset])
                assert accepted == actual.accept_token(text[offset])
                assert accepted, f"the example must be valid at byte {offset}"


def test_string_closing_masks_shared_between_grammars():
    """Summaries live with the tokenizer: a second grammar must not inherit the first's parent."""
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    for schema, text in CASES + CASES[::-1]:
        eager = xgr.GrammarCompiler(
            info, cache_enabled=False, enable_dynamic_compilation=False, max_threads=1
        ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)
        dynamic = xgr.GrammarCompiler(
            info, cache_enabled=False, enable_dynamic_compilation=True, max_threads=1
        ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)
        expected, actual = xgr.GrammarMatcher(eager), xgr.GrammarMatcher(dynamic)
        for offset in range(len(text) + 1):
            torch.testing.assert_close(mask(actual), mask(expected), rtol=0, atol=0)
            if offset < len(text):
                assert expected.accept_token(text[offset]) and actual.accept_token(text[offset])
