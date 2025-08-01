import sys

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


def test_grammar_constructor_empty():
    expected_grammar = 'root ::= ("")\n'
    empty_grammar = xgr.Grammar.empty()
    assert empty_grammar is not None
    empty_grammar_str = str(empty_grammar)
    assert empty_grammar_str == expected_grammar


input_test_string_expected_grammar_test_grammar_constructor_string = (
    ("our beautiful Earth", "our beautiful Earth", 'root ::= (("our beautiful Earth"))\n'),
    ("wqepqw\\n", "wqepqw\n", 'root ::= (("wqepqw\\n"))\n'),
    ("世界", "世界", 'root ::= (("\\u4e16\\u754c"))\n'),
    ("ぉ", "ぉ", 'root ::= (("\\u3049"))\n'),
    (
        "123werlkここぉ1q12",
        "123werlkここぉ1q12",
        'root ::= (("123werlk\\u3053\\u3053\\u30491q12"))\n',
    ),
)


@pytest.mark.parametrize(
    "input, test_string, expected_grammar",
    input_test_string_expected_grammar_test_grammar_constructor_string,
)
def test_grammar_constructor_string(input: str, test_string: str, expected_grammar: str):
    grammar = xgr.Grammar.string(input)
    assert grammar is not None
    grammar_str = str(grammar)
    assert _is_grammar_accept_string(grammar, test_string)
    assert grammar_str == expected_grammar


if __name__ == "__main__":
    pytest.main(sys.argv)
