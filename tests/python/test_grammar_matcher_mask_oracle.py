"""Check next-token masks against what the matcher accepts.

A token is allowed by the mask exactly when a fork of the matcher accepts its bytes, and the
stop token exactly when the grammar is completed. The vocabularies are small and synthetic, so
every token can be checked at every position.
"""

import itertools
import json
import sys
from typing import Any, Dict, List

import pytest

import xgrammar as xgr
from xgrammar.testing import _get_masked_tokens_from_bitmask


def _assert_mask_matches_matcher(matcher: xgr.GrammarMatcher, vocab: List[bytes]):
    """The last token of the vocabulary is the stop token."""
    bitmask = xgr.allocate_token_bitmask(1, len(vocab))
    matcher.fill_next_token_bitmask(bitmask)
    rejected = set(_get_masked_tokens_from_bitmask(bitmask, len(vocab)))
    for token_id, token in enumerate(vocab[:-1]):
        accepted = matcher.fork().accept_string(token)
        assert (token_id not in rejected) == accepted, (token, accepted)
    assert (len(vocab) - 1 not in rejected) == matcher.is_completed()


def _json_like_vocab() -> List[bytes]:
    alphabet = [b"a", b"b", b"c", b'"', b"\\", b"u", b"0", b"f", b",", b"}", b"[", b"]", b" ", b"x"]
    vocab = [b"".join(symbols) for n in [1, 2] for symbols in itertools.product(alphabet, repeat=n)]
    vocab += [b'"ab', b'ab"', b'a" ', b'a",', b'\\"', b'\\\\"', b"\\u0", b"u0f", b"0f0f"]
    vocab += [b"\\u00", b"\\u0fa", b'0f"', b'0f",', b'"}', b'" ,']
    vocab += ["é".encode(), "😀".encode(), '"é'.encode(), "é,".encode()]
    # Tokens that split a UTF-8 character.
    vocab += [b"\xc3", b"\xa9", b"\xf0\x9f", b"\x98\x80"]
    vocab += [b"[[[[", b"]]]]", b"[" * 20, b"]" * 20, b"a" * 9, b"a" * 12]
    return vocab + [b"</s>"]


def _check_prefixes(
    grammar: xgr.Grammar, prefixes: List[str], max_threads: int, cache_enabled: bool
):
    vocab = _json_like_vocab()
    tokenizer_info = xgr.TokenizerInfo(vocab, stop_token_ids=[len(vocab) - 1])
    compiler = xgr.GrammarCompiler(
        tokenizer_info, max_threads=max_threads, cache_enabled=cache_enabled
    )
    matcher = xgr.GrammarMatcher(compiler.compile_grammar(grammar))
    for prefix in prefixes:
        matcher.reset()
        assert matcher.accept_string(prefix), prefix
        _assert_mask_matches_matcher(matcher, vocab)


ebnf__prefixes = [
    # Lookahead assertions on helper rules, exact and inexact.
    (
        'root ::= "[" item ("," item)* "]"\nitem ::= [abc]+ (= [,\\]])',
        ["", "[", "[a", "[ab,", "[ab,c"],
    ),
    ('root ::= "[" word "]"\nword ::= w (= "]")\nw ::= [ab] | [ab] w', ["", "[", "[a", "[ab"]),
    # A JSON-like string with escapes and a lookahead assertion after the closing quote.
    (
        'root ::= "[" str ("," str)* "]"\n'
        'str ::= "\\"" chars (= [,\\]])\n'
        'chars ::= "\\"" | [^"\\\\] chars | "\\\\" esc chars\n'
        'esc ::= ["\\\\/bfnrt] | "u" [0-9a-f] [0-9a-f] [0-9a-f] [0-9a-f]',
        ["", "[", '["', '["a', '["a\\', '["a\\u0', '["a\\u0f0', '["a\\u0f0f', '["é"', '["a",'],
    ),
    # Nullable helper rules and alternatives that share a prefix.
    ('root ::= "(" opt opt ")"\nopt ::= "" | "a" | "ab"', ["", "(", "(a", "(ab", "(aba"]),
    # Character classes with multi-byte characters.
    ('root ::= "<" [é😀a]+ ">"', ["", "<", "<é", "<😀a"]),
    # Recursion. The configurations of deep positions are too large for the configuration DFA,
    # so shallow and deep positions take different paths in the same compilation.
    (
        'root ::= "[" inner "]"\ninner ::= "" | "[" inner "]" | [ab]+',
        ["", "[", "[[", "[[a", "[[a]"],
    ),
    (
        'root ::= value\nvalue ::= "[" value "]" | [abc]*',
        ["", "[" * 3, "[" * 15, "[" * 16, "[" * 17, "[" * 30 + "a", "[" * 30 + "a" + "]" * 5],
    ),
]


@pytest.mark.parametrize("max_threads, cache_enabled", [(1, False), (1, True), (4, False)])
@pytest.mark.parametrize("ebnf, prefixes", ebnf__prefixes)
def test_ebnf_grammars(ebnf: str, prefixes: List[str], max_threads: int, cache_enabled: bool):
    _check_prefixes(xgr.Grammar.from_ebnf(ebnf), prefixes, max_threads, cache_enabled)


schema__prefixes = [
    (
        {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "array", "items": {}}},
            "required": ["a"],
        },
        ["", "{", '{"a"', '{"a": "', '{"a": "x\\u0', '{"a": "é"', '{"a": "x", "b": [[['],
    ),
    # Strings with a length range are compiled to counted repetitions.
    (
        {"type": "array", "items": {"type": "string", "minLength": 2, "maxLength": 9}},
        ["", "[", '["', '["a', '["éa', '["aaaaaaaa', '["aaaaaaaaa', '["aa", "'],
    ),
]


@pytest.mark.parametrize("max_threads, cache_enabled", [(1, False), (4, True)])
@pytest.mark.parametrize("schema, prefixes", schema__prefixes)
def test_json_schemas(
    schema: Dict[str, Any], prefixes: List[str], max_threads: int, cache_enabled: bool
):
    grammar = xgr.Grammar.from_json_schema(json.dumps(schema), any_whitespace=True)
    _check_prefixes(grammar, prefixes, max_threads, cache_enabled)


def test_compiled_grammar_does_not_depend_on_threads_or_cache():
    """A single-threaded compilation shares one configuration DFA between the states, and a
    multi-threaded one uses a DFA per state. Both must give the same compiled grammar."""
    vocab = _json_like_vocab()
    tokenizer_info = xgr.TokenizerInfo(vocab, stop_token_ids=[len(vocab) - 1])
    for ebnf, _ in ebnf__prefixes:
        serialized = {
            xgr.GrammarCompiler(
                tokenizer_info, max_threads=max_threads, cache_enabled=cache_enabled
            )
            .compile_grammar(ebnf)
            .serialize_json()
            for max_threads, cache_enabled in [(1, False), (4, False), (1, True), (8, True)]
        }
        assert len(serialized) == 1, ebnf


@pytest.mark.parametrize("max_threads", [1, 4])
def test_exclude_token_in_lookahead_assertion(max_threads: int):
    """A lookahead assertion with a token element leaves the tokens that reach it to the parent
    rule: "ca" and "caa" end the rule `a` and continue into the "aa" that follows it."""
    alphabet = [b"a", b"b", b"c", b"x", b"[", b"]"]
    vocab = [
        b"".join(symbols) for n in [1, 2, 3] for symbols in itertools.product(alphabet, repeat=n)
    ]
    vocab.append(b"</s>")
    tokenizer_info = xgr.TokenizerInfo(vocab, stop_token_ids=[len(vocab) - 1])
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=max_threads, cache_enabled=False)
    compiled = compiler.compile_grammar(
        'root ::= "[" a "aa" "]"\na ::= ("b" a | "c") (= ExcludeToken(1, 2, 3))'
    )
    matcher = xgr.GrammarMatcher(compiled)
    for prefix in ["", "[", "[b", "[bc", "[c", "[ca"]:
        matcher.reset()
        assert matcher.accept_string(prefix)
        _assert_mask_matches_matcher(matcher, vocab)


if __name__ == "__main__":
    pytest.main(sys.argv)
