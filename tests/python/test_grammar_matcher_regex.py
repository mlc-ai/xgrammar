"""This test is adopted from test_builtin_grammar_json.py, but the grammar is parsed from
a unoptimized, non-simplified EBNF string. This is to test the robustness of the grammar matcher.
"""

import sys

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


def test_simple():
    regex_str = "abc"
    grammar = xgr.Grammar.from_regex(regex_str)
    assert _is_grammar_accept_string(grammar, "abc")
    assert not _is_grammar_accept_string(grammar, "ab")
    assert not _is_grammar_accept_string(grammar, "abcd")


test_repetition_input_accepted_test_repetition = (
    ("aaa", True),
    ("abcbc", True),
    ("bcbcbcbcbc", True),
    ("bcbcbcbcbcbcbcb", True),
    ("d", False),
    ("aaaa", False),
)


@pytest.mark.parametrize("input, accepted", test_repetition_input_accepted_test_repetition)
def test_repetition(input: str, accepted: bool):
    regex_str = "(a|[bc]{4,}){2,3}"
    grammar = xgr.Grammar.from_regex(regex_str)
    assert _is_grammar_accept_string(grammar, input) == accepted


test_regex_accept_regex_input_accepted = [
    r"abc",
    r"[abc]+",
    r"[a-z0-9]+",
    r"[^abc]+",
    r"a*b+c?",
    r"(abc|def)+",
    r"a{2,4}",
    r"\d+",
    r"\w+",
    r"[A-Z][a-z]*",
    r"[0-9]{3}-[0-9]{3}-[0-9]{4}",
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
]


@pytest.mark.parametrize("regex_input_accepted", test_regex_accept_regex_input_accepted)
def test_regex_accept(regex_input_accepted: str):
    grammar = xgr.Grammar.from_regex(regex_input_accepted)
    assert grammar is not None


test_regex_refuse_regex_input_refused = (
    r"a{,3}",  # Invalid range
    r"a{3,2}",  # Invalid range (max < min)
    r"[z-a]",  # Invalid range (max < min)
    r"a??",  # Invalid repetition
    r"a++",  # Invalid repetition
    r"(?:a)",  # Non-capturing groups not supported
    r"(?=a)",  # Lookahead not supported
    r"(?!a)",  # Negative lookahead not supported
)


@pytest.mark.parametrize("regex_input_refused", test_regex_refuse_regex_input_refused)
def test_regex_refuse(regex_input_refused: str):
    with pytest.raises(RuntimeError):
        xgr.Grammar.from_regex(regex_input_refused)


test_advanced_regex_string_instance_is_accepted = [
    # Basic patterns
    (r"abc", "abc", True),
    (r"abc", "def", False),
    # Character classes
    (r"[abc]+", "aabbcc", True),
    (r"[abc]+", "abcd", False),
    (r"[a-z0-9]+", "abc123", True),
    (r"[a-z0-9]+", "ABC", False),
    (r"[^abc]+", "def", True),
    (r"[^abc]+", "aaa", False),
    # Lazy character class
    (r"[abc]+?abc", "aabc", True),
    # Quantifiers
    (r"a*b+c?", "b", True),
    (r"a*b+c?", "aaabbc", True),
    (r"a*b+c?", "c", False),
    # Alternation
    (r"(abc|def)+", "abcdef", True),
    (r"(abc|def)+", "abcabc", True),
    (r"(abc|def)+", "ab", False),
    # Repetition ranges
    (r"a{2,4}", "aa", True),
    (r"a{2,4}", "aaaa", True),
    (r"a{2,4}", "a", False),
    (r"a{2,4}", "aaaaa", False),
    # Common patterns
    (r"\d+", "123", True),
    (r"\d+", "abc", False),
    (r"\w+", "abc123", True),
    (r"\w+", "!@#", False),
    (r"[A-Z][a-z]*", "Hello", True),
    (r"[A-Z][a-z]*", "hello", False),
    # Complex patterns
    (r"[0-9]{3}-[0-9]{3}-[0-9]{4}", "123-456-7890", True),
    (r"[0-9]{3}-[0-9]{3}-[0-9]{4}", "12-34-567", False),
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}", "test@email.com", True),
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}", "invalid.email", False),
]


@pytest.mark.parametrize(
    "regex_string, instance, is_accepted", test_advanced_regex_string_instance_is_accepted
)
def test_advanced(regex_string: str, instance: str, is_accepted: bool):
    grammar = xgr.Grammar.from_regex(regex_string)
    assert _is_grammar_accept_string(grammar, instance) == is_accepted


if __name__ == "__main__":
    pytest.main(sys.argv)
