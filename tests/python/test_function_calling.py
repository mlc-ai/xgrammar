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


def test_number_function_call():
    number_call = xgr.Grammar.from_function_call(
        args_names=["arg1", "arg2"], args_types=["int", "float"]
    )

    assert (
        str(number_call)
        == """root ::= ((arg1 arg2))
number ::= ((sign_1 "0" fraction_1 exponent_1 [ \\n\\t]* "</parameter>") | (sign_1 [1-9] [0-9]* fraction_1 exponent_1 [ \\n\\t]* "</parameter>"))
fraction_1 ::= ("" | ("." [0-9] [0-9]*))
exponent_1 ::= ("" | ("e" sign_1 [0-9] [0-9]*) | ("E" sign_1 [0-9] [0-9]*))
sign_1 ::= ("" | ("+") | ("-"))
arg1 ::= (("<parameter=arg1>" [ \\n\\t]* number)) (=(arg2))
arg2 ::= (("<parameter=arg2>" [ \\n\\t]* number))
"""
    )
    assert _is_grammar_accept_string(
        number_call,
        "<parameter=arg1>\t123\n</parameter><parameter=arg2>45.67\n\n\n\t  </parameter>",
    )
    assert _is_grammar_accept_string(
        number_call, "<parameter=arg1>0</parameter><parameter=arg2>-0.001</parameter>"
    )
    assert _is_grammar_accept_string(
        number_call, "<parameter=arg1>+123</parameter><parameter=arg2>-456.789e10</parameter>"
    )
    assert _is_grammar_accept_string(
        number_call, "<parameter=arg1>-123.33</parameter><parameter=arg2>+456.789E-10</parameter>"
    )
    assert not _is_grammar_accept_string(
        number_call, "<parameter=arg1>abc</parameter><parameter=arg2>123.45</parameter>"
    )
    assert not _is_grammar_accept_string(
        number_call, "<parameter=arg1>123.45</parameter><parameter=arg2>abc</parameter>"
    )
    assert not _is_grammar_accept_string(
        number_call, "<parameter=arg1>123.45e</parameter><parameter=arg2>--678.90</parameter>"
    )
    assert not _is_grammar_accept_string(
        number_call, "<parameter=arg1></parameter><parameter=arg2></parameter>"
    )


if __name__ == "__main__":
    pytest.main(sys.argv)
