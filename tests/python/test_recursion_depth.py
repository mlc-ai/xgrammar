import sys

import pytest

import xgrammar as xgr
from xgrammar.testing import _get_matcher_from_grammar


def test_set_get_recursion_depth():
    """Test getting default recursion depth"""
    default_depth = xgr.get_max_recursion_depth()
    assert default_depth == 10000

    xgr.set_max_recursion_depth(1000)
    new_depth = xgr.get_max_recursion_depth()
    assert new_depth == 1000
    xgr.set_max_recursion_depth(default_depth)


def test_recursion_depth_context():
    """Test recursion depth context manager"""
    assert xgr.get_max_recursion_depth() == 10000
    with xgr.max_recursion_depth(1000):
        depth = xgr.get_max_recursion_depth()
        assert depth == 1000
    assert xgr.get_max_recursion_depth() == 10000


def test_error_set_recursion_depth():
    """Test setting recursion depth to an invalid value"""
    with pytest.raises(RuntimeError):
        xgr.set_max_recursion_depth(-1)

    with pytest.raises(RuntimeError):
        xgr.set_max_recursion_depth(100000000)


def test_recursion_exceed():
    # In Earley Parser, the recursion depth can't be exceeded.
    with xgr.max_recursion_depth(1000):
        grammar_ebnf = r"""
    root ::= "\"" basic_string "\""
    basic_string ::= "" | [^"\\\r\n] basic_string | "\\" escape basic_string
    escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
    """
        input_str = '"' + " " * 10000 + '"'

        matcher = _get_matcher_from_grammar(grammar_ebnf)

        matcher.accept_string(input_str)


if __name__ == "__main__":
    pytest.main(sys.argv)
