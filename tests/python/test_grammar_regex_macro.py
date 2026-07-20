"""Tests the Regex(...) macro of the grammar parser, printer and matcher."""

import json
import sys
from typing import Optional

import pytest

import xgrammar as xgr
from xgrammar.testing import _ebnf_to_grammar_no_normalization, _is_grammar_accept_string


def test_regex_macro_parse_and_print():
    before = 'root ::= Regex("[0-9]{5}")'
    expected = 'root ::= ((Regex("[0-9]{5}")))\n'
    grammar = _ebnf_to_grammar_no_normalization(before)
    assert str(grammar) == expected
    # The printed form can be parsed again.
    assert str(xgr.Grammar.from_ebnf(str(grammar))) == 'root ::= Regex("[0-9]{5}")\n'


def test_regex_macro_parse_and_print_json_string():
    before = r'root ::= Regex("\\S+", json_string=true)'
    expected = 'root ::= ((Regex("\\\\S+", json_string=true)))\n'
    grammar = _ebnf_to_grammar_no_normalization(before)
    assert str(grammar) == expected
    assert (
        str(xgr.Grammar.from_ebnf(str(grammar))) == 'root ::= Regex("\\\\S+", json_string=true)\n'
    )

    # json_string=false is the default and is not printed.
    before = r'root ::= Regex("\\S+", json_string=false)'
    expected = 'root ::= ((Regex("\\\\S+")))\n'
    grammar = _ebnf_to_grammar_no_normalization(before)
    assert str(grammar) == expected


def test_regex_macro_normalization():
    # A regex is kept as the direct body of a rule.
    grammar = xgr.Grammar.from_ebnf('root ::= Regex("[0-9]{5}")')
    assert str(grammar) == 'root ::= Regex("[0-9]{5}")\n'

    # A regex inside a sequence is extracted into a new rule.
    grammar = xgr.Grammar.from_ebnf('root ::= "x" Regex("[0-9]+") "y"')
    assert str(grammar) == 'root ::= (("x" root_1 "y"))\nroot_1 ::= Regex("[0-9]+")\n'


ebnf_str__input_str__accepted__test_regex_macro_accept_string = [
    # Literals and classes
    ('root ::= Regex("abc")', "abc", True),
    ('root ::= Regex("abc")', "ab", False),
    ('root ::= Regex("abc")', "abcd", False),
    ('root ::= Regex("[0-9]{5}")', "12345", True),
    ('root ::= Regex("[0-9]{5}")', "1234", False),
    ('root ::= Regex("[0-9]{5}")', "123456", False),
    ('root ::= Regex("[a-f]{2,3}")', "ab", True),
    ('root ::= Regex("[a-f]{2,3}")', "abc", True),
    ('root ::= Regex("[a-f]{2,3}")', "a", False),
    ('root ::= Regex("[a-f]{2,3}")', "abcd", False),
    ('root ::= Regex("[0-9]{2,}")', "123456", True),
    ('root ::= Regex("[0-9]{2,}")', "1", False),
    ('root ::= Regex("[^a-z]")', "A", True),
    ('root ::= Regex("[^a-z]")', "a", False),
    # Anchors are allowed and ignored
    ('root ::= Regex("^[0-9]+$")', "123", True),
    ('root ::= Regex("^[0-9]+$")', "a", False),
    # Union and grouping
    ('root ::= Regex("a|bc|def")', "a", True),
    ('root ::= Regex("a|bc|def")', "bc", True),
    ('root ::= Regex("a|bc|def")', "def", True),
    ('root ::= Regex("a|bc|def")', "b", False),
    ('root ::= Regex("(a|b)(c|d)")', "ac", True),
    ('root ::= Regex("(a|b)(c|d)")', "bd", True),
    ('root ::= Regex("(a|b)(c|d)")', "ab", False),
    # Repetition operators
    ('root ::= Regex("a+b*c?")', "a", True),
    ('root ::= Regex("a+b*c?")', "aabbc", True),
    ('root ::= Regex("a+b*c?")', "bc", False),
    ('root ::= Regex("a*")', "", True),
    ('root ::= Regex("a*")', "aaa", True),
    ('root ::= Regex("a*")', "b", False),
    # The accepting state of (ab)+ has a single epsilon edge back to the start state; the
    # simplification passes must not make the start state accepting.
    ('root ::= Regex("(ab)+")', "", False),
    ('root ::= Regex("(ab)+")', "a", False),
    ('root ::= Regex("(ab)+")', "ab", True),
    ('root ::= Regex("(ab)+")', "abab", True),
    # Escapes
    ('root ::= Regex("\\\\d+\\\\.\\\\d+")', "3.14", True),
    ('root ::= Regex("\\\\d+\\\\.\\\\d+")', "3.", False),
    ('root ::= Regex("\\\\w+")', "a_1", True),
    ('root ::= Regex("\\\\w+")', "a b", False),
    # . matches any byte, including multi-byte UTF-8 characters
    ('root ::= Regex("a.c")', "abc", True),
    ('root ::= Regex("a.c")', "a?c", True),
    ('root ::= Regex(".+")', "你好", True),
    # Regex used in a sequence and in choices
    ('root ::= "x" Regex("[0-9]+") "y"', "x123y", True),
    ('root ::= "x" Regex("[0-9]+") "y"', "xy", False),
    ('root ::= Regex("[0-9]+") | "abc"', "123", True),
    ('root ::= Regex("[0-9]+") | "abc"', "abc", True),
    ('root ::= Regex("[0-9]+") | "abc"', "abd", False),
]


@pytest.mark.parametrize(
    "ebnf_str, input_str, accepted", ebnf_str__input_str__accepted__test_regex_macro_accept_string
)
def test_regex_macro_accept_string(ebnf_str: str, input_str: str, accepted: bool):
    grammar = xgr.Grammar.from_ebnf(ebnf_str)
    assert _is_grammar_accept_string(grammar, input_str) == accepted


ebnf_str__input_str__accepted__test_regex_macro_json_string = [
    # json_string=true excludes '"', '\' and the control characters from every character
    # match, so the regex cannot produce an unescaped quote inside a JSON string literal.
    (r'root ::= Regex("\\S+", json_string=true)', "abc", True),
    (r'root ::= Regex("\\S+", json_string=true)', "a.b!c", True),
    (r'root ::= Regex("\\S+", json_string=true)', 'a"b', False),
    (r'root ::= Regex("\\S+", json_string=true)', "a\\b", False),
    (r'root ::= Regex("\\S+", json_string=true)', "a b", False),
    (r'root ::= Regex(".+", json_string=true)', "ab", True),
    (r'root ::= Regex(".+", json_string=true)', "a\tb", False),
    (r'root ::= Regex(".+", json_string=true)', "a\nb", False),
    (r'root ::= Regex(".+", json_string=true)', "你好", True),
    # Without the flag, the quote, backslash and control characters are accepted.
    (r'root ::= Regex("\\S+")', 'a"b', True),
    (r'root ::= Regex("\\S+")', "a\\b", True),
    (r'root ::= Regex(".+")', "a\tb", True),
]


@pytest.mark.parametrize(
    "ebnf_str, input_str, accepted", ebnf_str__input_str__accepted__test_regex_macro_json_string
)
def test_regex_macro_json_string(ebnf_str: str, input_str: str, accepted: bool):
    grammar = xgr.Grammar.from_ebnf(ebnf_str)
    assert _is_grammar_accept_string(grammar, input_str) == accepted


def test_regex_macro_nullable_rule():
    # The allow-empty analysis must detect that the regex rule accepts the empty string.
    grammar = xgr.Grammar.from_ebnf('root ::= r "z"\nr ::= Regex("a*")')
    assert _is_grammar_accept_string(grammar, "z")
    assert _is_grammar_accept_string(grammar, "aaz")
    assert not _is_grammar_accept_string(grammar, "a")

    grammar = xgr.Grammar.from_ebnf('root ::= r "z"\nr ::= Regex("a+")')
    assert not _is_grammar_accept_string(grammar, "z")
    assert _is_grammar_accept_string(grammar, "az")


def test_regex_macro_serialization_roundtrip():
    for ebnf_str in ['root ::= Regex("[0-9]{5}")', r'root ::= Regex("\\S+", json_string=true)']:
        grammar = xgr.Grammar.from_ebnf(ebnf_str)
        roundtrip = xgr.Grammar.deserialize_json(grammar.serialize_json())
        assert str(roundtrip) == str(grammar)

    # The json_string flag keeps its effect after the round trip.
    grammar = xgr.Grammar.from_ebnf(r'root ::= Regex("\\S+", json_string=true)')
    roundtrip = xgr.Grammar.deserialize_json(grammar.serialize_json())
    assert _is_grammar_accept_string(roundtrip, "abc")
    assert not _is_grammar_accept_string(roundtrip, 'a"b')


ebnf_str__expected_error_regex__test_regex_macro_parser_errors = [
    ("root ::= Regex()", "Regex expects exactly one string argument"),
    ('root ::= Regex("a", "b")', "Regex expects exactly one string argument"),
    ("root ::= Regex(abc)", "Regex pattern must be a string literal"),
    ('root ::= Regex("a", foo=true)', "Regex does not support the named argument foo"),
    ('root ::= Regex("a", json_string="yes")', "json_string must be a boolean"),
    ('root ::= Regex("a", json_string=1)', "json_string must be a boolean"),
]


@pytest.mark.parametrize(
    "ebnf_str, expected_error_regex", ebnf_str__expected_error_regex__test_regex_macro_parser_errors
)
def test_regex_macro_parser_errors(ebnf_str: str, expected_error_regex: Optional[str]):
    with pytest.raises(RuntimeError, match=expected_error_regex):
        _ebnf_to_grammar_no_normalization(ebnf_str)


def test_regex_macro_invalid_pattern():
    # The pattern is only compiled when the grammar automaton is built.
    grammar = xgr.Grammar.from_ebnf('root ::= Regex("+a")')
    with pytest.raises(RuntimeError, match="Failed to build the automaton for rule root"):
        _is_grammar_accept_string(grammar, "a")


def test_json_schema_pattern_uses_regex_macro():
    schema = json.dumps({"type": "string", "pattern": "^\\S+$"})
    grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=False)
    assert 'Regex("^\\\\S+$", json_string=true)' in str(grammar)
    assert _is_grammar_accept_string(grammar, '"abc"')
    assert _is_grammar_accept_string(grammar, '"a.b!c"')
    # An unescaped quote or backslash inside the string would be invalid JSON.
    assert not _is_grammar_accept_string(grammar, '"""')
    assert not _is_grammar_accept_string(grammar, '"a"b"')
    assert not _is_grammar_accept_string(grammar, '"a\\b"')
    assert not _is_grammar_accept_string(grammar, '"a b"')
    assert not _is_grammar_accept_string(grammar, '""')


def test_json_schema_pattern_repetition():
    # End-to-end check of the simplification passes on the compiled pattern automaton.
    schema = json.dumps({"type": "string", "pattern": "^(ab)+$"})
    grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=False)
    assert 'Regex("^(ab)+$", json_string=true)' in str(grammar)
    assert _is_grammar_accept_string(grammar, '"abab"')
    assert not _is_grammar_accept_string(grammar, '""')
    assert not _is_grammar_accept_string(grammar, '"a"')


def test_json_schema_pattern_fallback_to_cfg():
    # A pattern that requires a literal quote cannot be matched inside a JSON string body,
    # so the conversion falls back to the CFG expansion of the regex.
    schema = json.dumps({"type": "string", "pattern": '^a"b$'})
    grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=False)
    assert "Regex(" not in str(grammar)

    # A pattern with non-printable-ASCII characters also falls back to the CFG expansion.
    schema = json.dumps({"type": "string", "pattern": "^[一-龥]+$"})
    grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=False)
    assert "Regex(" not in str(grammar)
    assert _is_grammar_accept_string(grammar, '"你好"')
    assert not _is_grammar_accept_string(grammar, '"ab"')


def test_json_schema_pattern_properties():
    schema = json.dumps({"type": "object", "patternProperties": {"^[a-z]+$": {"type": "integer"}}})
    grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=False)
    assert "json_string=true" in str(grammar)
    assert _is_grammar_accept_string(grammar, '{"ab": 1}')
    assert not _is_grammar_accept_string(grammar, '{"AB": 1}')


if __name__ == "__main__":
    pytest.main(sys.argv)
