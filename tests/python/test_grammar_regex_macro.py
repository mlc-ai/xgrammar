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
    # json_string=true consumes the encoded body of a JSON string while the regex matches its
    # decoded characters. Valid short and Unicode escapes are therefore accepted.
    (r'root ::= Regex("\\S+", json_string=true)', "abc", True),
    (r'root ::= Regex("\\S+", json_string=true)', "a.b!c", True),
    (r'root ::= Regex("\\S+", json_string=true)', 'a"b', False),
    (r'root ::= Regex("\\S+", json_string=true)', "a\\b", True),
    (r'root ::= Regex("\\S+", json_string=true)', r"a\"b", True),
    (r'root ::= Regex("\\S+", json_string=true)', r"a\\b", True),
    (r'root ::= Regex("\\S+", json_string=true)', r"a\u0062", True),
    (r'root ::= Regex("\\S+", json_string=true)', r"a\qb", False),
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


ebnf_str__input_str__accepted__test_regex_macro_engine_features = [
    # A leading (?i) enables Unicode simple case folding.
    ('root ::= Regex("(?i)abc")', "aBc", True),
    ('root ::= Regex("(?i)abc")', "ABC", True),
    ('root ::= Regex("(?i)abc")', "abd", False),
    ('root ::= Regex("(?i)[a-d]+")', "AbCd", True),
    ('root ::= Regex("(?i)[a-d]+")', "e", False),
    ('root ::= Regex("(?i)[a-d]+")', "E", False),
    ('root ::= Regex("(?i)[^k]")', "a", True),
    ('root ::= Regex("(?i)[^k]")', "k", False),
    ('root ::= Regex("(?i)[^k]")', "K", False),
    # Simple Unicode case folding includes non-ASCII equivalence classes.
    ('root ::= Regex("(?i)Σ")', "Σ", True),
    ('root ::= Regex("(?i)Σ")', "σ", True),
    # \xHH, \uHHHH and \u{...} escapes, standalone and inside classes.
    (r'root ::= Regex("\\x41\\u0042\\u{43}")', "ABC", True),
    (r'root ::= Regex("\\x41\\u0042\\u{43}")', "ABD", False),
    (r'root ::= Regex("\\u{1F600}")', "😀", True),
    (r'root ::= Regex("\\u{1F600}")', "😁", False),
    (r'root ::= Regex("[\\u0041-\\u0043]+")', "ABC", True),
    (r'root ::= Regex("[\\u0041-\\u0043]+")', "D", False),
    # Escapes are folded under (?i) as well.
    (r'root ::= Regex("(?i)\\x41")', "a", True),
    (r'root ::= Regex("(?i)\\x41")', "A", True),
    (r'root ::= Regex("(?i)\\x41")', "b", False),
    # Unicode mode can be changed for a scope, including from an outer byte-mode pattern.
    (r'root ::= Regex("(?-u:\\w)+")', "Az_9", True),
    (r'root ::= Regex("(?-u:\\w)+")', "é", False),
    (r'root ::= Regex("(?u:\\w)", byte_mode=true)', "λ", True),
    (r'root ::= Regex("(?u:\\w)", byte_mode=true)', " ", False),
    # \cA is the control character U+0001.
    (r'root ::= Regex("x\\cAy")', "x\x01y", True),
    (r'root ::= Regex("x\\cAy")', "xy", False),
    # \s uses the Unicode White_Space property.
    (r'root ::= Regex("a\\sb")', "a b", True),
    (r'root ::= Regex("a\\sb")', "a\tb", True),
    (r'root ::= Regex("a\\sb")', "a\u00a0b", True),
    (r'root ::= Regex("a\\sb")', "a\x00b", False),
    (r'root ::= Regex("a\\sb")', "a\x01b", False),
    # \S is the codepoint-domain complement.
    (r'root ::= Regex("\\S")', "好", True),
    (r'root ::= Regex("\\S")', "\x00", True),
    (r'root ::= Regex("\\S")', " ", False),
    # A quantifier after a multi-byte character applies to the whole codepoint.
    ('root ::= Regex("好*")', "", True),
    ('root ::= Regex("好*")', "好好", True),
    ('root ::= Regex("好*")', "\xbd", False),
    # Lookahead assertions are ignored (treated as the empty string).
    ('root ::= Regex("a(?=b)c")', "ac", True),
    ('root ::= Regex("a(?=b)c")', "abc", False),
    ('root ::= Regex("a(?!b)c")', "ac", True),
    # Named groups compile like plain groups; the name is ignored.
    ('root ::= Regex("(?<name>ab)+")', "abab", True),
    ('root ::= Regex("(?<name>ab)+")', "a", False),
    ('root ::= Regex("(?P<name>ab)c")', "abc", True),
    ('root ::= Regex("(?<λ.名[1]>ab)c")', "abc", True),
    # Non-greedy quantifiers accept the same language as their greedy counterparts.
    ('root ::= Regex("a+?b")', "aab", True),
    ('root ::= Regex("a+?b")', "b", False),
    # Empty alternatives match the empty string.
    ('root ::= Regex("(a|)b")', "b", True),
    ('root ::= Regex("(a|)b")', "ab", True),
    ('root ::= Regex("a|")', "", True),
    ('root ::= Regex("a|")', "a", True),
]


@pytest.mark.parametrize(
    "ebnf_str, input_str, accepted", ebnf_str__input_str__accepted__test_regex_macro_engine_features
)
def test_regex_macro_engine_features(ebnf_str: str, input_str: str, accepted: bool):
    grammar = xgr.Grammar.from_ebnf(ebnf_str)
    assert _is_grammar_accept_string(grammar, input_str) == accepted


ebnf_str__input_str__accepted__test_regex_macro_flags = [
    # The i flag folds ASCII letters, like the Lark /.../i literal.
    ('root ::= Regex("abc", flags="i")', "aBc", True),
    ('root ::= Regex("abc", flags="i")', "ABC", True),
    ('root ::= Regex("abc", flags="i")', "abd", False),
    ('root ::= Regex("[a-d]+", flags="i")', "AbCd", True),
    ('root ::= Regex("[a-d]+", flags="i")', "E", False),
    # Simple Unicode case folding also applies outside ASCII.
    ('root ::= Regex("Σ", flags="i")', "Σ", True),
    ('root ::= Regex("Σ", flags="i")', "σ", True),
    # With the flags argument, '.' follows the standard semantics: no newline unless 's'.
    ('root ::= Regex("a.b", flags="")', "acb", True),
    ('root ::= Regex("a.b", flags="")', "a\nb", False),
    ('root ::= Regex("a.b", flags="s")', "a\nb", True),
    ('root ::= Regex("a.b", flags="is")', "A\nB", True),
    ('root ::= Regex("a.b", flags="i")', "A\nB", False),
    # 'u' is accepted as a no-op: patterns always use Unicode codepoint semantics.
    ('root ::= Regex("a.b", flags="u")', "a😀b", True),
    ('root ::= Regex("a.b", flags="u")', "a\nb", False),
    # Without the flags argument the pattern is used verbatim: '.' matches every codepoint.
    ('root ::= Regex("a.b")', "a\nb", True),
    # An escaped dot or a dot inside a character class is not rewritten.
    (r'root ::= Regex("a\\.b", flags="")', "a.b", True),
    (r'root ::= Regex("a\\.b", flags="")', "acb", False),
    ('root ::= Regex("a[.]b", flags="")', "a.b", True),
    ('root ::= Regex("a[.]b", flags="")', "acb", False),
    # An inline (?i) prefix combined with flags="i" is not applied twice.
    ('root ::= Regex("(?i)abc", flags="i")', "ABC", True),
    ('root ::= Regex("(?i)abc", flags="i")', "abd", False),
]


@pytest.mark.parametrize(
    "ebnf_str, input_str, accepted", ebnf_str__input_str__accepted__test_regex_macro_flags
)
def test_regex_macro_flags(ebnf_str: str, input_str: str, accepted: bool):
    grammar = xgr.Grammar.from_ebnf(ebnf_str)
    assert _is_grammar_accept_string(grammar, input_str) == accepted


def test_regex_macro_flags_json_string():
    # flags combines with json_string: the folded letters match, but the JSON-forbidden
    # characters stay excluded.
    grammar = xgr.Grammar.from_ebnf(r'root ::= Regex("a[b-d]+", flags="i", json_string=true)')
    assert _is_grammar_accept_string(grammar, "ABC")
    assert not _is_grammar_accept_string(grammar, "AE")

    grammar = xgr.Grammar.from_ebnf(r'root ::= Regex(".+", flags="i", json_string=true)')
    assert _is_grammar_accept_string(grammar, "AbC")
    assert not _is_grammar_accept_string(grammar, 'a"b')


def test_regex_macro_flags_print_round_trip():
    # The flags are folded into the stored pattern, which survives printing and re-parsing.
    grammar = xgr.Grammar.from_ebnf('root ::= Regex("a.b", flags="is")')
    restored = xgr.Grammar.from_ebnf(str(grammar))
    assert _is_grammar_accept_string(restored, "A\nB")
    assert not _is_grammar_accept_string(restored, "ab")

    grammar = xgr.Grammar.from_ebnf('root ::= Regex("a.b", flags="")')
    restored = xgr.Grammar.from_ebnf(str(grammar))
    assert _is_grammar_accept_string(restored, "acb")
    assert not _is_grammar_accept_string(restored, "a\nb")


def test_regex_macro_case_insensitive_print_round_trip():
    # The (?i) prefix survives printing (the '?' is escaped) and re-parsing.
    grammar = xgr.Grammar.from_ebnf('root ::= Regex("(?i)abc")')
    restored = xgr.Grammar.from_ebnf(str(grammar))
    assert _is_grammar_accept_string(restored, "ABC")
    assert not _is_grammar_accept_string(restored, "abd")


def test_regex_macro_case_insensitive_json_string():
    # (?i) combines with json_string: folded letters and valid JSON escapes match, while raw
    # JSON-forbidden characters stay excluded.
    grammar = xgr.Grammar.from_ebnf(r'root ::= Regex("(?i).+", json_string=true)')
    assert _is_grammar_accept_string(grammar, "AbC")
    assert not _is_grammar_accept_string(grammar, 'a"b')
    assert _is_grammar_accept_string(grammar, "a\\b")
    assert _is_grammar_accept_string(grammar, r"a\u0042")
    assert not _is_grammar_accept_string(grammar, r"a\q")

    grammar = xgr.Grammar.from_ebnf(r'root ::= Regex("(?i)a[b-d]+", json_string=true)')
    assert _is_grammar_accept_string(grammar, "ABC")
    assert not _is_grammar_accept_string(grammar, "AE")


def test_regex_macro_large_repetition_subrule():
    # Physically unrolling this repetition would exceed the FSM state limit, so this only
    # compiles if the repetition becomes a grammar-level repeat subrule.
    grammar = xgr.Grammar.from_ebnf('root ::= Regex("(ab){2,50000}c")')
    assert _is_grammar_accept_string(grammar, "ab" * 2 + "c")
    assert _is_grammar_accept_string(grammar, "ab" * 1000 + "c")
    assert not _is_grammar_accept_string(grammar, "abc")
    assert not _is_grammar_accept_string(grammar, "c")
    assert not _is_grammar_accept_string(grammar, "ab" * 2)

    # {n,} compiles the mandatory part as a repeat edge followed by a starred rule reference.
    grammar = xgr.Grammar.from_ebnf('root ::= Regex("(ab){200,}c")')
    assert _is_grammar_accept_string(grammar, "ab" * 200 + "c")
    assert _is_grammar_accept_string(grammar, "ab" * 321 + "c")
    assert not _is_grammar_accept_string(grammar, "ab" * 199 + "c")

    # A nullable repeated atom relaxes the lower bound to zero.
    grammar = xgr.Grammar.from_ebnf('root ::= Regex("(a?){2,300}b")')
    assert _is_grammar_accept_string(grammar, "b")
    assert _is_grammar_accept_string(grammar, "a" * 300 + "b")
    assert not _is_grammar_accept_string(grammar, "a" * 301 + "b")

    # Case-insensitive large repetition through the (?i) prefix.
    grammar = xgr.Grammar.from_ebnf('root ::= Regex("(?i)(ab){2,50000}c")')
    assert _is_grammar_accept_string(grammar, "aBAb" * 100 + "C")
    assert not _is_grammar_accept_string(grammar, "aBc")


def test_regex_macro_large_repetition_nullable_rule():
    # The allow-empty analysis must handle large repetitions without building their FSM.
    grammar = xgr.Grammar.from_ebnf('root ::= r "z"\nr ::= Regex("(ab){0,50000}")')
    assert _is_grammar_accept_string(grammar, "z")
    assert _is_grammar_accept_string(grammar, "ababz")
    assert not _is_grammar_accept_string(grammar, "ab")

    grammar = xgr.Grammar.from_ebnf('root ::= r "z"\nr ::= Regex("(ab){2,50000}")')
    assert not _is_grammar_accept_string(grammar, "z")
    assert _is_grammar_accept_string(grammar, "ababz")


def test_regex_macro_large_repetition_serialization_roundtrip():
    grammar = xgr.Grammar.from_ebnf('root ::= Regex("(?i)(ab){2,50000}c")')
    roundtrip = xgr.Grammar.deserialize_json(grammar.serialize_json())
    assert str(roundtrip) == str(grammar)
    assert _is_grammar_accept_string(roundtrip, "abABc")
    assert not _is_grammar_accept_string(roundtrip, "abc")


ebnf_str__expected_error__test_regex_macro_unsupported_features = [
    (r'root ::= Regex("a\\b")', "Word boundary assertion"),
    (r'root ::= Regex("a\\B")', "Word boundary assertion"),
    (r'root ::= Regex("\\pL")', r"Unicode property escapes \\p and \\P are not supported"),
    (r'root ::= Regex("(a)\\1")', "Backreference"),
    (r'root ::= Regex("(?<name>a)\\k<name>")', "Backreference"),
    ('root ::= Regex("(?<=a)b")', "Lookbehind assertion"),
    ('root ::= Regex("[]")', "Unclosed '\\['"),
    ('root ::= Regex("[^]")', "Unclosed '\\['"),
    (r'root ::= Regex("\\uZZ")', "must be followed by 4 hexadecimal digits"),
    (r'root ::= Regex("\\x4")', "must be followed by 2 hexadecimal digits"),
    (r'root ::= Regex("\\u{110000}")', "not a Unicode scalar value"),
]


@pytest.mark.parametrize(
    "ebnf_str, expected_error", ebnf_str__expected_error__test_regex_macro_unsupported_features
)
def test_regex_macro_unsupported_features(ebnf_str: str, expected_error: str):
    # The pattern is only compiled when the grammar automaton is built, so the error is
    # raised on first use.
    grammar = xgr.Grammar.from_ebnf(ebnf_str)
    with pytest.raises(RuntimeError, match=expected_error):
        _is_grammar_accept_string(grammar, "a")


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
    ('root ::= Regex("a", flags=true)', "flags must be a string"),
    ('root ::= Regex("a", flags="m")', "regular-expression flag 'm' is not supported"),
    ('root ::= Regex("a", flags="l")', "regular-expression flag 'l' is not supported"),
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
    assert "json_string=true" in str(grammar)
    assert _is_grammar_accept_string(grammar, '"abc"')
    assert _is_grammar_accept_string(grammar, '"a.b!c"')
    # An unescaped quote or invalid escape inside the string is invalid JSON. A valid short
    # escape is matched as its decoded non-whitespace character.
    assert not _is_grammar_accept_string(grammar, '"""')
    assert not _is_grammar_accept_string(grammar, '"a"b"')
    assert _is_grammar_accept_string(grammar, '"a\\b"')
    assert _is_grammar_accept_string(grammar, '"a\\u0062"')
    assert not _is_grammar_accept_string(grammar, '"a\\q"')
    assert not _is_grammar_accept_string(grammar, '"a b"')
    assert not _is_grammar_accept_string(grammar, '""')


def test_json_schema_pattern_repetition():
    # End-to-end check of the simplification passes on the compiled pattern automaton.
    schema = json.dumps({"type": "string", "pattern": "^(ab)+$"})
    grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=False)
    assert "json_string=true" in str(grammar)
    assert _is_grammar_accept_string(grammar, '"abab"')
    assert not _is_grammar_accept_string(grammar, '""')
    assert not _is_grammar_accept_string(grammar, '"a"')


def test_json_schema_pattern_encoded_spellings():
    # A decoded quote is matched through its escaped JSON source spelling.
    schema = json.dumps({"type": "string", "pattern": '^a"b$'})
    grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=False)
    assert "json_string=true" in str(grammar)
    assert _is_grammar_accept_string(grammar, '"a\\"b"')
    assert not _is_grammar_accept_string(grammar, '"ab"')

    # Unicode patterns stay in the regex FSM and accept raw and Unicode-escaped spellings.
    schema = json.dumps({"type": "string", "pattern": "^[一-龥]+$"})
    grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=False)
    assert "json_string=true" in str(grammar)
    assert _is_grammar_accept_string(grammar, '"你好"')
    assert _is_grammar_accept_string(grammar, '"\\u4F60\\u597D"')
    assert not _is_grammar_accept_string(grammar, '"ab"')


def test_json_schema_pattern_properties():
    schema = json.dumps({"type": "object", "patternProperties": {"^[a-z]+$": {"type": "integer"}}})
    grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=False)
    assert "json_string=true" in str(grammar)
    assert _is_grammar_accept_string(grammar, '{"ab": 1}')
    assert not _is_grammar_accept_string(grammar, '{"AB": 1}')


if __name__ == "__main__":
    pytest.main(sys.argv)
