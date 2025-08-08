import sys

import pytest

import xgrammar as xgr


def test_empty_function_call():
    expect_grammar = """root ::= ("")
"""
    empty_call = xgr.Grammar.from_function_call(args_names=[], args_types=[])

    assert str(empty_call) == expect_grammar


if __name__ == "__main__":
    pytest.main(sys.argv)
