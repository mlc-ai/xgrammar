"""Tests the Substring(...) macro of the grammar parser, printer and matcher."""

import sys
from typing import Optional

import pytest

import xgrammar as xgr
from xgrammar.testing import _ebnf_to_grammar_no_normalization, _is_grammar_accept_string


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
