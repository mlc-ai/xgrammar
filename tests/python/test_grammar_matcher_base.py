"""This test tests character-based operations for the grammar matcher, mainly the matching
algorithm, which is defined in the C++ GrammarMatcherBase class."""

import sys

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

json_grammar = xgr.Grammar.builtin_json_grammar()


input_accepted = [
    '{"name": "John"}',
    '{ "name" : "John" }',
]


@pytest.mark.parametrize("input_accepted", input_accepted)
def test_json_accept(input_accepted: str):
    assert _is_grammar_accept_string(json_grammar, input_accepted)


input_refused = (
    '{ name: "John" }',
    '{ "name": "John" } ',
)


@pytest.mark.parametrize("input_refused", input_refused)
def test_json_refuse(input_refused: str):
    assert not _is_grammar_accept_string(json_grammar, input_refused)


input_accepted_test_repetition = (
    ("aaa", True),
    ("abcbc", True),
    ("bcbcbcbcbc", True),
    ("bcbcbcbcbcbcbcb", True),
    ("d", False),
    ("aaaa", False),
)


@pytest.mark.parametrize("input, accepted", input_accepted_test_repetition)
def test_repetition(input: str, accepted: bool):
    grammar_str = """
        root ::= rule {2, 3}
        rule ::= ("a" | [bc] {4,})
    """
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, input) == accepted


if __name__ == "__main__":
    pytest.main(sys.argv)
