"""Tests the Substring(...) macro of the grammar parser, printer and matcher."""

import itertools
import sys
from typing import List, Optional

import pytest

import xgrammar as xgr
from xgrammar.testing import (
    _ebnf_to_grammar_no_normalization,
    _get_masked_tokens_from_bitmask,
    _is_grammar_accept_string,
)


def test_substring_macro_parse_and_print():
    before = 'root ::= Substring("abc", "de", "fg")'
    expected = 'root ::= ((Substring("abc", "de", "fg")))\n'
    grammar = _ebnf_to_grammar_no_normalization(before)
    assert str(grammar) == expected
    # The printed form can be parsed again.
    assert str(xgr.Grammar.from_ebnf(str(grammar))) == 'root ::= Substring("abc", "de", "fg")\n'


def test_substring_macro_normalization():
    # A substring is kept as the direct body of a rule.
    grammar = xgr.Grammar.from_ebnf('root ::= Substring("abc", "de")')
    assert str(grammar) == 'root ::= Substring("abc", "de")\n'

    # A substring inside a sequence is extracted into a new rule.
    grammar = xgr.Grammar.from_ebnf('root ::= "x" Substring("abc") "y"')
    assert str(grammar) == 'root ::= (("x" root_1 "y"))\nroot_1 ::= Substring("abc")\n'


ebnf_str__input_str__accepted__test_substring_macro_accept_string = [
    # Contiguous chunk subsequences, including the empty one
    ('root ::= Substring("abc", "de", "fg")', "", True),
    ('root ::= Substring("abc", "de", "fg")', "abc", True),
    ('root ::= Substring("abc", "de", "fg")', "de", True),
    ('root ::= Substring("abc", "de", "fg")', "fg", True),
    ('root ::= Substring("abc", "de", "fg")', "abcde", True),
    ('root ::= Substring("abc", "de", "fg")', "defg", True),
    ('root ::= Substring("abc", "de", "fg")', "abcdefg", True),
    # Chunks are atomic and must stay contiguous and in order
    ('root ::= Substring("abc", "de", "fg")', "ab", False),
    ('root ::= Substring("abc", "de", "fg")', "cde", False),
    ('root ::= Substring("abc", "de", "fg")', "abcfg", False),
    ('root ::= Substring("abc", "de", "fg")', "deabc", False),
    # Repeated and empty chunks
    ('root ::= Substring("a", "", "b", "a", "b")', "", True),
    ('root ::= Substring("a", "", "b", "a", "b")', "ab", True),
    ('root ::= Substring("a", "", "b", "a", "b")', "ba", True),
    ('root ::= Substring("a", "", "b", "a", "b")', "abab", True),
    ('root ::= Substring("a", "", "b", "a", "b")', "aa", False),
    ('root ::= Substring("a", "", "b", "a", "b")', "abba", False),
    # Zero chunks match only the empty string
    ("root ::= Substring()", "", True),
    ("root ::= Substring()", "a", False),
    # Substring used in a sequence and in choices
    ('root ::= "x" Substring("ab", "cd") "y"', "xy", True),
    ('root ::= "x" Substring("ab", "cd") "y"', "xabcdy", True),
    ('root ::= "x" Substring("ab", "cd") "y"', "xbcy", False),
    ('root ::= Substring("ab") | "z"', "ab", True),
    ('root ::= Substring("ab") | "z"', "z", True),
    ('root ::= Substring("ab") | "z"', "az", False),
]


@pytest.mark.parametrize(
    "ebnf_str, input_str, accepted",
    ebnf_str__input_str__accepted__test_substring_macro_accept_string,
)
def test_substring_macro_accept_string(ebnf_str: str, input_str: str, accepted: bool):
    grammar = xgr.Grammar.from_ebnf(ebnf_str)
    assert _is_grammar_accept_string(grammar, input_str) == accepted


def test_substring_macro_nullable_rule():
    # The allow-empty analysis must detect that a substring rule accepts the empty string.
    grammar = xgr.Grammar.from_ebnf('root ::= r "z"\nr ::= Substring("ab")')
    assert _is_grammar_accept_string(grammar, "z")
    assert _is_grammar_accept_string(grammar, "abz")
    assert not _is_grammar_accept_string(grammar, "ab")


def test_substring_macro_nul_chunk_round_trip():
    # NUL bytes in chunks survive printing and re-parsing.
    grammar = xgr.Grammar.from_ebnf('root ::= Substring("a", "\\0", "b")')
    assert str(grammar) == 'root ::= Substring("a", "\\0", "b")\n'
    for candidate, accepted in [
        ("", True),
        ("a", True),
        ("\0", True),
        ("a\0b", True),
        ("ab", False),
    ]:
        assert _is_grammar_accept_string(grammar, candidate) == accepted
        assert _is_grammar_accept_string(xgr.Grammar.from_ebnf(str(grammar)), candidate) == accepted


def test_substring_macro_serialization_roundtrip():
    grammar = xgr.Grammar.from_ebnf('root ::= Substring("abc", "de")')
    roundtrip = xgr.Grammar.deserialize_json(grammar.serialize_json())
    assert str(roundtrip) == str(grammar)
    assert _is_grammar_accept_string(roundtrip, "abcde")
    assert not _is_grammar_accept_string(roundtrip, "bc")


chunks__test_substring_macro_brute_force_language: List[List[str]] = [
    # Triggers the clone branch of the suffix automaton construction
    ["a", "b", "a", "b", "b"],
    # Multi-byte chunks with overlapping content
    ["ab", "a", "b", "ba"],
    # One chunk is a prefix of another
    ["ab", "abc"],
    # Empty and repeated chunks
    ["a", "", "a", ""],
    ["aa", "aa", "aa"],
]


@pytest.mark.parametrize("chunks", chunks__test_substring_macro_brute_force_language)
def test_substring_macro_brute_force_language(chunks: List[str]):
    """The accepted language must be exactly the concatenations of contiguous chunk
    subsequences. Checked against a brute-force reference over all short strings."""
    expected_language = {
        "".join(chunks[i:j]) for i in range(len(chunks) + 1) for j in range(i, len(chunks) + 1)
    }
    alphabet = sorted(set("".join(chunks)))
    max_length = len("".join(chunks))

    ebnf_str = "root ::= Substring(" + ", ".join(f'"{chunk}"' for chunk in chunks) + ")"
    grammar = xgr.Grammar.from_ebnf(ebnf_str)

    for length in range(max_length + 1):
        for candidate_tuple in itertools.product(alphabet, repeat=length):
            candidate = "".join(candidate_tuple)
            assert _is_grammar_accept_string(grammar, candidate) == (
                candidate in expected_language
            ), f"Mismatch for {candidate!r} with chunks {chunks!r}"


def test_substring_macro_unicode_chunks():
    # Multi-byte UTF-8 chunks stay atomic: no partial chunk can be matched.
    grammar = xgr.Grammar.from_ebnf('root ::= Substring("你好", "世界")')
    for candidate, accepted in [
        ("", True),
        ("你好", True),
        ("世界", True),
        ("你好世界", True),
        ("你", False),
        ("好", False),
        ("你世界", False),
        ("世界你好", False),
    ]:
        assert _is_grammar_accept_string(grammar, candidate) == accepted


def test_substring_macro_lazy_rule():
    # A lazy rule completes at the first opportunity. Since a substring accepts the empty
    # string, a lazy substring rule always commits to the empty match.
    grammar = xgr.Grammar.from_ebnf('root ::= r "z"\nr[lazy] ::= Substring("ab")')
    assert _is_grammar_accept_string(grammar, "z")
    assert not _is_grammar_accept_string(grammar, "abz")


def test_substring_macro_in_lookahead_rejected():
    with pytest.raises(RuntimeError, match="Substring should not be in lookahead assertion"):
        xgr.Grammar.from_ebnf('root ::= "a" (=Substring("b"))')


def test_substring_macro_max_chars():
    # The per-rule max_chars budget applies to a substring rule.
    grammar = xgr.Grammar.from_ebnf('root ::= s\ns[max_chars=2] ::= Substring("a", "b", "c")')
    for candidate, accepted in [
        ("", True),
        ("a", True),
        ("ab", True),
        ("bc", True),
        ("abc", False),
    ]:
        assert _is_grammar_accept_string(grammar, candidate) == accepted


def test_substring_macro_union_and_concat():
    substring_grammar = xgr.Grammar.from_ebnf('root ::= Substring("ab", "cd")')
    z_grammar = xgr.Grammar.from_ebnf('root ::= "z"')

    union = xgr.Grammar.union(substring_grammar, z_grammar)
    for candidate, accepted in [
        ("", True),
        ("ab", True),
        ("abcd", True),
        ("z", True),
        ("b", False),
        ("abz", False),
    ]:
        assert _is_grammar_accept_string(union, candidate) == accepted

    concat = xgr.Grammar.concat(substring_grammar, z_grammar)
    for candidate, accepted in [
        ("z", True),
        ("abz", True),
        ("abcdz", True),
        ("ab", False),
        ("cdab", False),
    ]:
        assert _is_grammar_accept_string(concat, candidate) == accepted


def test_substring_macro_bitmask_matches_string_acceptance():
    """At every step of a generation, the token bitmask must allow exactly the vocabulary
    pieces the matcher would accept as the next characters."""
    grammar_str = 'root ::= "<" Substring("ab", "cd") ">"'
    vocab = ["a", "b", "ab", "c", "d", "cd", "abcd", "<", ">", "b>", "ba", "ac"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)
    compiled = compiler.compile_grammar(grammar_str)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    consumed = ""
    remaining = "<abcd>"
    while True:
        matcher.fill_next_token_bitmask(bitmask)
        rejected = set(_get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size))
        for token_id, piece in enumerate(vocab):
            fork = matcher.fork()
            assert fork.accept_string(piece) == (token_id not in rejected), (
                f"Bitmask disagrees with accept_string for piece {piece!r} "
                f"after consuming {consumed!r}"
            )
        if not remaining:
            break
        assert matcher.accept_string(remaining[0])
        consumed += remaining[0]
        remaining = remaining[1:]
    assert matcher.is_terminated()


ebnf_str__expected_error_regex__test_substring_macro_parser_errors = [
    ('root ::= Substring("a", foo=true)', "Substring\\(\\) does not accept named arguments"),
    ("root ::= Substring(abc)", "Substring\\(\\) arguments must be strings"),
    ('root ::= Substring("a", 1)', "Substring\\(\\) arguments must be strings"),
]


@pytest.mark.parametrize(
    "ebnf_str, expected_error_regex",
    ebnf_str__expected_error_regex__test_substring_macro_parser_errors,
)
def test_substring_macro_parser_errors(ebnf_str: str, expected_error_regex: Optional[str]):
    with pytest.raises(RuntimeError, match=expected_error_regex):
        _ebnf_to_grammar_no_normalization(ebnf_str)


if __name__ == "__main__":
    pytest.main(sys.argv)
