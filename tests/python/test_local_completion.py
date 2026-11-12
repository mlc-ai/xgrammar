"""Local-completion masks preserve eager masks at and across rule boundaries."""

import itertools

import pytest
import torch

import xgrammar as xgr

VOCAB = [bytes([i]) for i in range(256)] + [
    b"ab",
    b"abc",
    b"abc!",
    b"abc!x",
    b"abc!wrong",
    b"!",
    b"!x",
    b"!y",
    b'hello"',
    b'hello",',
    b'hello"wrong',
    b'\\"',
    b"\\u1234",
    b"\xe0\xa0",
    b"\xe0\x80\x80",
    b"\xed\xa0\x80",
    b"\xf4\x90\x80\x80",
    b"\xf0\x90",
    "中文".encode(),
    "中文!x".encode(),
    b"<stop>",
]
STOP = len(VOCAB) - 1
# Exercise the bitset representation, LCP reuse, duplicates, and rejected subtrees.
VOCAB += ["".join(s).encode() for s in itertools.product("abcdefghijk", repeat=3)]
VOCAB += [b"0" + w for w in VOCAB[STOP + 1 :]]

CASES = [
    ('root ::= "[" run "x"\nrun ::= [a-z] run | "!"', b"[abc!x"),
    ('root ::= "[" run "x"\nrun ::= [a-z] run | "!" (= "x")', b"[abc!x"),
    ('root ::= "[" run "x"\nrun ::= [^!] run | "!"', "[中文!x".encode()),
    ('root ::= "[" run "x"\nrun ::= atom run | "!"\natom ::= [a-z]', b"[abc!x"),
    # A lookahead in a callee must prevent flattening through it.
    ('root ::= "[" run "x"\nrun ::= atom run | "!"\natom ::= [a-z] (= [a-z!])', b"[abc!x"),
    ('root ::= "[" run "x"\nrun[max_tokens=4] ::= [a-z] run | "!"', b"[ab!x"),
    ('root ::= "[" run "x"\nrun[max_chars=4] ::= [a-z] run | "!"', b"[ab!x"),
    # Multiple completion bytes and accepting states with outgoing edges fall back.
    ('root ::= "[" run "x"\nrun ::= [a-z] run | "!" | "?"', b"[abc!x"),
    ('root ::= "[" run "x"\nrun ::= [a-z]*', b"[abcx"),
    (
        {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]},
        b'{"s":"abc"}',
    ),
]


def mask(matcher):
    result = xgr.allocate_token_bitmask(1, len(VOCAB))
    matcher.fill_next_token_bitmask(result)
    if len(VOCAB) % 32:
        result[0, -1] &= (1 << (len(VOCAB) % 32)) - 1
    return result


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("source,text", CASES)
def test_local_completion_matches_eager(source, text, cache):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])

    def compile(dynamic):
        compiler = xgr.GrammarCompiler(
            info, cache_enabled=cache, enable_dynamic_compilation=dynamic, max_threads=1
        )
        return (
            compiler.compile_grammar(source)
            if isinstance(source, str)
            else compiler.compile_json_schema(source, any_whitespace=False)
        )

    eager, dynamic = compile(False), compile(True)
    # First use, warmed masks, and serialized/materialized masks must agree at every byte.
    for replay in range(3):
        candidate = (
            dynamic
            if replay < 2
            else xgr.CompiledGrammar.deserialize_json(dynamic.serialize_json(), info)
        )
        expected, actual = xgr.GrammarMatcher(eager), xgr.GrammarMatcher(candidate)
        for offset in range(len(text) + 1):
            torch.testing.assert_close(mask(actual), mask(expected), rtol=0, atol=0)
            if offset < len(text):
                assert expected.accept_token(text[offset]) == actual.accept_token(text[offset])
