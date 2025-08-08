import sys

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


def test_empty_function_call():
    expect_grammar = """root ::= ("")
"""
    empty_call = xgr.Grammar.from_function_call(args_names=[], args_types=[])

    assert str(empty_call) == expect_grammar


def test_boolean_function_call():
    boolean_call = xgr.Grammar.from_function_call(
        args_names=["arg1", "arg2"], args_types=["bool", "binary"]
    )

    expect_grammar = """root ::= ((arg1 arg2))
boolean ::= (("true" [ \\n\\t]* "</parameter>") | ("false" [ \\n\\t]* "</parameter>"))
arg1 ::= (("<parameter=arg1>" [ \\n\\t]* boolean)) (=(arg2))
arg2 ::= (("<parameter=arg2>" [ \\n\\t]* boolean))
"""
    assert str(boolean_call) == expect_grammar
    assert _is_grammar_accept_string(
        boolean_call, "<parameter=arg1>true</parameter><parameter=arg2>false</parameter>"
    )
    assert _is_grammar_accept_string(
        boolean_call, "<parameter=arg1>\t\t\tfalse</parameter><parameter=arg2>\ntrue</parameter>"
    )
    assert not _is_grammar_accept_string(
        boolean_call, "<parameter=arg1>1</parameter><parameter=arg2>0</parameter>"
    )
    assert not _is_grammar_accept_string(
        boolean_call, "<parameter=arg1>True</parameter><parameter=arg2>False</parameter>"
    )


if __name__ == "__main__":
    pytest.main(sys.argv)
