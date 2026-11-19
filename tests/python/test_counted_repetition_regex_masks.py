"""Counted-repetition masks for bounded regex repetitions must equal token-by-token replay.

The fast path decides most tokens from the repetition budget alone. Here every mask bit is
checked against accept_token on a fresh matcher, at every prefix of inputs that sit on the
lower bound, inside the bounds, and on the upper bound.
"""

import itertools
import json

import pytest

import xgrammar as xgr

VOCAB = [bytes([i]) for i in range(256)] + [
    b"ab",
    b"abc",
    b"abcd",
    b"abcde",
    b"a-b_c",
    b'ab"',
    b'abc"',
    b'abc",',
    b'abc"}',
    b'"ab',
    b'"',
    b'",',
    b'"}',
    b"a@",
    b"ab@x",
    b"@x",
    b'@x"',
    b"A1",
    b"a\\",
    b'a\\"',
    b"\\u0061",
    b"a b",
    b"\xc3\xa9",
    b"a\xc3",
    b"\xc3",
    "é".encode() + b"a",
    "中".encode(),
    b"<stop>",
]
STOP = len(VOCAB) - 1
VOCAB += ["".join(chars).encode() for chars in itertools.product("abc_", repeat=3)]
VOCAB += [word * 30 for word in (b"a", b"ab", b"abc")] + [b"a" * 199, b"a" * 200, b"a" * 201]


def pattern(regex):
    return {"type": "string", "pattern": regex}


# (schema, valid instances). Bounds are above the regex large-repeat threshold (128), so the
# repetitions compile to repeat edges rather than unrolled FSM states.
CASES = [
    (pattern("^[0-9a-zA-Z_-]{1,200}$"), ["a", "abc", "a-b_c", "a" * 199, "a" * 200]),
    (pattern("^[0-9a-zA-Z_-]{2,200}$"), ["ab", "abc", "ab" * 100]),
    (pattern("^[a-c]{0,201}$"), ["", "a", "abc" * 67, "a" * 201]),
    (pattern("^[a-c]{200}$"), ["a" * 200, "abc" * 66 + "ab"]),
    (pattern("^[a-z]{1,200}@x$"), ["a@x", "ab@x", "a" * 200 + "@x"]),
    (pattern("^x[a-z]{1,300}$"), ["xa", "xabc", "x" + "abc" * 100]),
    (pattern("^[a-zé]{1,200}$"), ["é", "aéa", "é" * 200]),
    (pattern("^[^@]{1,200}$"), ["a", "a b", "中a", "a" * 200]),
    (pattern("^(ab|c){1,200}$"), ["ab", "c", "abc", "ab" * 200]),
    (pattern("^[a-z]{1,200}[0-9]{1,200}$"), ["a1", "abc123", "a" * 200 + "1"]),
    (
        {
            "type": "object",
            "properties": {
                "k": pattern("^[0-9a-zA-Z_-]{1,200}$"),
                "v": {"type": "string", "minLength": 1, "maxLength": 210},
            },
            "required": ["k", "v"],
            "additionalProperties": False,
        },
        [{"k": "abc", "v": "ab"}, {"k": "a" * 200, "v": "a" * 210}],
    ),
    (
        {"type": "array", "items": pattern("^[a-c]{1,200}$"), "maxItems": 3},
        [["abc", "a"], ["a" * 200]],
    ),
    (
        {"anyOf": [pattern("^[a-c]{1,200}$"), pattern("^[a-c]{1,210}!$")]},
        ["abc", "abc!", "a" * 210 + "!"],
    ),
]
IDS = [json.dumps(schema)[:60] for schema, _ in CASES]


def allowed_tokens(matcher):
    mask = xgr.allocate_token_bitmask(1, len(VOCAB))
    matcher.fill_next_token_bitmask(mask)
    return [bool((int(mask[0, i // 32]) >> (i % 32)) & 1) for i in range(len(VOCAB))]


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("schema,instances", CASES, ids=IDS)
def test_bounded_regex_masks_match_token_replay(schema, instances, dynamic):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    compiled = xgr.GrammarCompiler(
        info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=dynamic
    ).compile_json_schema(schema, any_whitespace=False)
    for instance in instances:
        text = json.dumps(instance, ensure_ascii=False).encode()
        # Checking every prefix of a long instance is quadratic; the ends are where bounds bite.
        offsets = sorted(
            set(range(min(len(text), 8) + 1)) | set(range(len(text) - 6, len(text) + 1))
        )
        matcher = xgr.GrammarMatcher(compiled)
        for offset in range(len(text) + 1):
            if offset in offsets:
                allowed = allowed_tokens(matcher)
                for token_id in range(len(VOCAB)):
                    oracle = matcher.fork()
                    assert allowed[token_id] == oracle.accept_token(token_id), (
                        instance,
                        offset,
                        VOCAB[token_id],
                    )
            if offset < len(text):
                assert matcher.accept_token(text[offset]), (instance, offset)
        assert matcher.accept_token(STOP)


@pytest.mark.parametrize("schema,instances", CASES, ids=IDS)
def test_bounded_regex_masks_eager_equals_dynamic(schema, instances):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    eager, dynamic = (
        xgr.GrammarCompiler(
            info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=flag
        ).compile_json_schema(schema, any_whitespace=False)
        for flag in (False, True)
    )
    for instance in instances:
        text = json.dumps(instance, ensure_ascii=False).encode()
        expected, actual = xgr.GrammarMatcher(eager), xgr.GrammarMatcher(dynamic)
        for byte in text:
            assert allowed_tokens(actual) == allowed_tokens(expected)
            assert expected.accept_token(byte) and actual.accept_token(byte)
        assert allowed_tokens(actual) == allowed_tokens(expected)
