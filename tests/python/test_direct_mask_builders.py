"""Direct JIT mask builders must reproduce eager masks, including under matcher state changes.

Eager compilation never uses the direct builders, so it is the oracle for every case here.
"""

import itertools
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import xgrammar as xgr

VOCAB = [bytes([i]) for i in range(256)] + [
    b"ab",
    b"abc",
    b"abd",
    b"abc!",
    b"abc!x",
    b"abx",
    b'"name"',
    b'"name":',
    b'"name":1',
    b'"na',
    b'"n',
    b'"url"',
    b'"url":"',
    b'"u',
    b'"x',
    b"AB",
    b"AB1",
    b"AB123",
    b'AB123"',
    b'AB123",',
    b"AB1234",
    b"2024",
    b"2024-",
    b"2024-01-02",
    b"-01",
    b"12",
    b"12.3",
    b"12!",
    b"hel",
    b"hello",
    b"hello!",
    b"\xe4\xb8",
    b"\xe4",
    b"\xe0\x80\x80",
    b"\xf0\x90",
    "中".encode(),
    "中文".encode(),
    "中文!".encode(),
    b"<stop>",
]
STOP = len(VOCAB) - 1
# Equal decoded strings with different ids, plus dense accepted and rejected sets so that every
# AdaptiveTokenMask storage form is produced.
VOCAB += [b"abc", b'"name"', b"AB123"]
DENSE = ["".join(chars).encode() for chars in itertools.product("abcdefghijk", repeat=3)]
VOCAB += DENSE + [b"1" + word for word in DENSE]

EBNF_CASES = [
    # Deterministic byte paths: literal tries, shared prefixes, unicode literals.
    (
        "literal-trie",
        'root ::= key ":" value\nkey ::= "\\"name\\"" | "\\"url\\""\nvalue ::= [0-9]+',
        b'"name":12',
    ),
    ("literal-shared-prefix", 'root ::= run "!"\nrun ::= "ab" | "abc" | "abd"', b"abc!"),
    ("literal-unicode", 'root ::= key "!"\nkey ::= "中文" | "hello"', "中文!".encode()),
    ("literal-then-rule", 'root ::= key\nkey ::= "ab" tail\ntail ::= [a-z] tail | "!"', b"abcd!"),
    # A one-byte boundary is not selective and must use the generic builder.
    ("literal-short", 'root ::= run "!"\nrun ::= "a" | "b"', b"a!"),
    # Small deterministic ASCII FSMs: loops, fixed width, optional parts.
    ("ascii-loop", 'root ::= run "!"\nrun ::= [a-z]+', b"abc!"),
    ("ascii-fixed", 'root ::= run "!"\nrun ::= [A-Z]{2} [0-9]{3}', b"AB123!"),
    ("ascii-date", 'root ::= d "!"\nd ::= [0-9]{4} "-" [0-9]{2} "-" [0-9]{2}', b"2024-01-02!"),
    ("ascii-number", 'root ::= n "!"\nn ::= [0-9]+ ("." [0-9]+)?', b"12.3!"),
    ("ascii-optional-end", 'root ::= run "x"\nrun ::= "ab" [a-z]*', b"abcx"),
    # Each of these has a property a direct builder must refuse.
    ("refuse-unicode-class", 'root ::= run "!"\nrun ::= [a-z中-文]+', "a中!".encode()),
    ("refuse-nondeterministic", 'root ::= run "!"\nrun ::= "ab" [a-z] | "a" [b-z] "c"', b"abc!"),
    ("refuse-lookahead", 'root ::= run "!"\nrun ::= [a-z]+ (= "!")', b"abc!"),
    ("refuse-lazy", 'root ::= run "!"\nrun[lazy] ::= [a-z]+', b"a!"),
    ("refuse-max-tokens", 'root ::= run "!"\nrun[max_tokens=2] ::= [a-z]+', b"ab!"),
    ("refuse-max-chars", 'root ::= run "!"\nrun[max_chars=2] ::= [a-z]+', b"ab!"),
    ("refuse-recursion", 'root ::= run "!"\nrun ::= "a" run | "b"', b"aab!"),
    # The same sub-rule reached from different parents must keep its caller's continuation.
    ("shared-callee", 'root ::= "<" run "!" | "[" run "x"\nrun ::= [a-z] run | [a-z]', b"[abcx"),
]

SCHEMA_CASES = [
    (
        "object",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}, "n": {"type": "integer"}},
            "required": ["name", "n"],
        },
        {"name": "abc", "n": 123},
    ),
    (
        "optional-properties",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "url": {"type": "string"},
                "n": {"type": "number"},
            },
        },
        {"url": "ab", "n": 12.3},
    ),
    ("enum", {"enum": ["hello", "hel", "abc", 12]}, "hel"),
    ("fixed-pattern", {"type": "string", "pattern": "^[A-Z]{2}[0-9]{3}$"}, "AB123"),
    ("loop-pattern", {"type": "string", "pattern": "^[a-z]+$"}, "abcdef"),
    ("bounded-length", {"type": "string", "minLength": 1, "maxLength": 4}, "abc"),
    ("date", {"type": "string", "format": "date"}, "2024-01-02"),
    ("unicode-string", {"type": "string"}, "中文a"),
    (
        "array-of-strings",
        {"type": "array", "items": {"type": "string", "pattern": "^[a-z]{1,3}$"}},
        ["ab", "abc"],
    ),
]

CASES = [(label, source, text) for label, source, text in EBNF_CASES] + [
    (label, schema, json.dumps(value, ensure_ascii=False).encode())
    for label, schema, value in SCHEMA_CASES
]
IDS = [case[0] for case in CASES]


def compile_pair(source, cache_enabled, info=None):
    info = info or xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])

    def build(dynamic):
        compiler = xgr.GrammarCompiler(
            info, max_threads=1, cache_enabled=cache_enabled, enable_dynamic_compilation=dynamic
        )
        if isinstance(source, str):
            return compiler.compile_grammar(source)
        return compiler.compile_json_schema(source, any_whitespace=False)

    return info, build(False), build(True)


def mask(matcher):
    result = xgr.allocate_token_bitmask(1, len(VOCAB))
    matcher.fill_next_token_bitmask(result)
    # Storage strategies may differ in unused padding bits; compare actual vocabulary ids.
    if len(VOCAB) % 32:
        result[0, -1] &= (1 << (len(VOCAB) % 32)) - 1
    return result


def assert_same_mask(actual, expected):
    torch.testing.assert_close(mask(actual), mask(expected), rtol=0, atol=0)


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("label,source,text", CASES, ids=IDS)
def test_direct_masks_match_eager(label, source, text, cache_enabled):
    info, eager, dynamic = compile_pair(source, cache_enabled)
    # First use, warmed masks, then masks materialized by serialization.
    for replay in range(3):
        candidate = (
            dynamic
            if replay < 2
            else xgr.CompiledGrammar.deserialize_json(dynamic.serialize_json(), info)
        )
        expected, actual = xgr.GrammarMatcher(eager), xgr.GrammarMatcher(candidate)
        for offset in range(len(text) + 1):
            assert_same_mask(actual, expected)
            if offset < len(text):
                assert expected.accept_token(text[offset])
                assert actual.accept_token(text[offset])
        assert actual.is_terminated() == expected.is_terminated()


@pytest.mark.parametrize("label,source,text", CASES, ids=IDS)
def test_direct_masks_agree_with_token_replay(label, source, text):
    """Every mask bit must equal what accept_token decides for that token."""
    if "max_tokens" in str(source) or "max_chars" in str(source):
        # accept_token deliberately relaxes exhausted budgets, whereas masks enforce them.
        pytest.skip("budgeted rules are covered by eager/dynamic mask equality")
    _, eager, dynamic = compile_pair(source, True)
    actual = xgr.GrammarMatcher(dynamic)
    for offset in range(len(text) + 1):
        result = mask(actual)
        for token_id in range(len(VOCAB)):
            oracle = xgr.GrammarMatcher(eager)
            for byte in text[:offset]:
                assert oracle.accept_token(byte)
            allowed = bool((int(result[0, token_id // 32]) >> (token_id % 32)) & 1)
            assert allowed == oracle.accept_token(token_id), (offset, VOCAB[token_id])
        if offset < len(text):
            assert actual.accept_token(text[offset])


@pytest.mark.parametrize("label,source,text", CASES, ids=IDS)
def test_direct_masks_multibyte_tokens(label, source, text):
    """Advance with the longest vocabulary token at each step instead of single bytes."""
    _, eager, dynamic = compile_pair(source, True)
    expected, actual = xgr.GrammarMatcher(eager), xgr.GrammarMatcher(dynamic)
    longest_first = sorted(range(len(VOCAB)), key=lambda i: -len(VOCAB[i]))
    offset = 0
    while offset < len(text):
        assert_same_mask(actual, expected)
        token_id = next(
            i for i in longest_first if i != STOP and VOCAB[i] and text.startswith(VOCAB[i], offset)
        )
        accepted = expected.accept_token(token_id)
        assert actual.accept_token(token_id) == accepted
        if not accepted:
            # Fall back to one byte so the walk always makes progress on both matchers.
            token_id = text[offset]
            assert expected.accept_token(token_id)
            assert actual.accept_token(token_id)
        offset += len(VOCAB[token_id])
    assert_same_mask(actual, expected)


@pytest.mark.parametrize("label,source,text", CASES, ids=IDS)
def test_direct_masks_rollback_fork_reset(label, source, text):
    _, eager, dynamic = compile_pair(source, True)
    expected = xgr.GrammarMatcher(eager, max_rollback_tokens=len(text) + 1)
    actual = xgr.GrammarMatcher(dynamic, max_rollback_tokens=len(text) + 1)
    forks = []
    for offset, byte in enumerate(text):
        assert_same_mask(actual, expected)
        if offset == len(text) // 2:
            forks.append((actual.fork(), expected.fork(), offset))
        assert expected.accept_token(byte)
        assert actual.accept_token(byte)
    assert_same_mask(actual, expected)

    # Roll back step by step: masks seen after rollback must match the eager matcher's.
    for _ in range(len(text)):
        expected.rollback(1)
        actual.rollback(1)
        assert_same_mask(actual, expected)

    # A fork taken mid-way continues independently of the rolled-back original.
    for forked_actual, forked_expected, offset in forks:
        for byte in text[offset:]:
            assert_same_mask(forked_actual, forked_expected)
            assert forked_expected.accept_token(byte)
            assert forked_actual.accept_token(byte)
        assert_same_mask(forked_actual, forked_expected)

    # Replay after rollback, then after reset.
    for _ in range(2):
        for byte in text:
            assert_same_mask(actual, expected)
            assert expected.accept_token(byte)
            assert actual.accept_token(byte)
        assert_same_mask(actual, expected)
        expected.reset()
        actual.reset()


def test_direct_masks_shared_grammar_across_threads():
    """Concurrent first fills on one JIT grammar must all publish the eager mask."""
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    compiler = xgr.GrammarCompiler(info, cache_enabled=True, enable_dynamic_compilation=True)
    eager_compiler = xgr.GrammarCompiler(info, cache_enabled=False)
    pairs = []
    for label, source, text in CASES:
        if isinstance(source, str):
            pairs.append(
                (compiler.compile_grammar(source), eager_compiler.compile_grammar(source), text)
            )
        else:
            pairs.append(
                (
                    compiler.compile_json_schema(source, any_whitespace=False),
                    eager_compiler.compile_json_schema(source, any_whitespace=False),
                    text,
                )
            )

    def replay(index):
        dynamic, eager, text = pairs[index % len(pairs)]
        expected, actual = xgr.GrammarMatcher(eager), xgr.GrammarMatcher(dynamic)
        for byte in text:
            assert_same_mask(actual, expected)
            assert expected.accept_token(byte)
            assert actual.accept_token(byte)
        assert_same_mask(actual, expected)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(replay, range(len(pairs) * 8)))


def test_direct_masks_outlive_compiler_and_tokenizer_roundtrip():
    """Tokenizer-derived tables are rebuilt on deserialization; grammars keep their summaries."""
    info = xgr.TokenizerInfo.deserialize_json(
        xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP]).serialize_json()
    )
    for label, source, text in CASES:
        _, eager, dynamic = compile_pair(source, False, info)
        expected, actual = xgr.GrammarMatcher(eager), xgr.GrammarMatcher(dynamic)
        for byte in text:
            assert_same_mask(actual, expected)
            assert expected.accept_token(byte)
            assert actual.accept_token(byte)
        assert_same_mask(actual, expected)
