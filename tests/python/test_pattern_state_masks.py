"""Masks inside regex patterns equal eager masks.

Dynamic compilation builds these masks by walking the rule automaton instead of running the
parser. Eager compilation does not and is the reference.
"""

import itertools

import pytest
import torch

import xgrammar as xgr

VOCAB = [bytes([i]) for i in range(256)]
# Words and spaces in every arrangement: enough tokens that cross several repetitions.
PIECES = ["a", "bc", "d ", " e", "f g", " ", "  ", "1", "-"]
for repeat in (1, 2, 3, 4):
    VOCAB += ["".join(parts).encode() for parts in itertools.product(PIECES, repeat=repeat)]
# Tokens that close the string, alone and after content, and bytes a JSON string cannot hold.
for content in [b"", b"a", b"a b", b"bc ", b" ", b"12", "中".encode(), "α β".encode()]:
    VOCAB += [content + suffix for suffix in [b'"', b'",', b'", "', b'"}', b'" ', b'"]']]
VOCAB += [b"a\x01", b"a\\", b'a"b', "中文".encode(), "中".encode()[:2], b"\xe4", b"<stop>"]
VOCAB = list(dict.fromkeys(VOCAB))
STOP = VOCAB.index(b"<stop>")


def pattern(regex, **keywords):
    return {"type": "string", "pattern": regex, **keywords}


WORDS = r"^(?:\S+\s+){0,%d}\S+$"
CASES = [
    # A small repetition is unrolled into the pattern automaton: every state has its own mask.
    (pattern(WORDS % 3), b'"a bc  d e"'),
    (pattern(r"^$|(^(?:\S+\s+){0,3}\S+$)"), b'""'),
    # A large repetition is counted at runtime; the text uses every repetition allowed.
    (pattern(WORDS % 150), b'"' + b" ".join([b"a"] * 151) + b'"'),
    (pattern(WORDS % 150), b'"a  bc f g d"'),
    (pattern(r"^(?:[a-c]+-){0,30}[a-c]+$"), b'"' + b"-".join([b"ab"] * 31) + b'"'),
    (pattern(r"^(?:\d{1,2} ){20,25}$"), b'"' + b"1 12 " * 11 + b'"'),
    (pattern(r"^(?:\d{1,2} ){130,140}$"), b'"' + b"1 12 " * 66 + b'"'),
    (pattern(r"^[a-g]{2,5}-\d{3}$"), b'"abc-112"'),
    (pattern(r"^(ab|a b)+c$"), b'"aba bc"'),
    (pattern("^[一-鿿]+ [α-ω]*$"), '"中文 αβ"'.encode()),
    (
        {
            "type": "object",
            "properties": {"s": pattern(WORDS % 150), "t": pattern(WORDS % 150)},
            "required": ["s", "t"],
        },
        b'{"s": "a bc", "t": "d e"}',
    ),
    ({"type": "array", "items": pattern(WORDS % 150), "maxItems": 3}, b'["a b", "bc"]'),
]


def mask(matcher):
    result = xgr.allocate_token_bitmask(1, len(VOCAB))
    matcher.fill_next_token_bitmask(result)
    if len(VOCAB) % 32:
        result[0, -1] &= (1 << (len(VOCAB) % 32)) - 1
    return result


def compile_schema(info, schema, dynamic, cache):
    compiler = xgr.GrammarCompiler(
        info, cache_enabled=cache, enable_dynamic_compilation=dynamic, max_threads=1
    )
    return compiler.compile_json_schema(schema, any_whitespace=False, strict_mode=True)


def assert_same_masks(eager, dynamic, text):
    expected, actual = xgr.GrammarMatcher(eager), xgr.GrammarMatcher(dynamic)
    for offset in range(len(text) + 1):
        torch.testing.assert_close(mask(actual), mask(expected), rtol=0, atol=0)
        if offset < len(text):
            accepted = expected.accept_token(text[offset])
            assert accepted == actual.accept_token(text[offset])
            assert accepted, f"the example must be valid at byte {offset}"


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("schema,text", CASES)
def test_pattern_state_masks_match_eager(schema, text, cache):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    eager = compile_schema(info, schema, False, cache)
    dynamic = compile_schema(info, schema, True, cache)
    for _ in range(2):
        assert_same_masks(eager, dynamic, text)


def test_pattern_state_masks_with_multibyte_tokens():
    """Replay token by token, so that masks are taken in the middle of words and repetitions."""
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    schema = pattern(WORDS % 150)
    eager = compile_schema(info, schema, False, False)
    dynamic = compile_schema(info, schema, True, False)
    tokens = [b'"', b"a", b"d ", b"bc", b"  ", b"f g", b" e", b"d  e", b"a bc", b'a"']
    expected, actual = xgr.GrammarMatcher(eager), xgr.GrammarMatcher(dynamic)
    for token in tokens:
        torch.testing.assert_close(mask(actual), mask(expected), rtol=0, atol=0)
        token_id = VOCAB.index(token)
        assert expected.accept_token(token_id) and actual.accept_token(token_id)
    torch.testing.assert_close(mask(actual), mask(expected), rtol=0, atol=0)


def test_pattern_state_masks_shared_between_grammars():
    """Compiling one schema must not change the masks of another."""
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    for schema, text in CASES + CASES[::-1]:
        eager = compile_schema(info, schema, False, False)
        dynamic = compile_schema(info, schema, True, False)
        assert_same_masks(eager, dynamic, text)
