import sys

import pytest

from xgrammar.testing import GrammarFunctor, _is_grammar_accept_string


def test_empty_function_call():
    expect_grammar = """root ::= ("")
"""
    empty_call = GrammarFunctor.from_function_call(args_names=[], args_types=[])

    assert str(empty_call) == expect_grammar


def test_boolean_function_call():
    boolean_call = GrammarFunctor.from_function_call(
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
    number_call = GrammarFunctor.from_function_call(
        args_names=["arg1", "arg2"], args_types=["int", "float"]
    )

    assert (
        str(number_call)
        == """root ::= ((arg1 arg2))
number ::= ((sign "0" fraction exponent [ \\n\\t]* "</parameter>") | (sign [1-9] [0-9]* fraction exponent [ \\n\\t]* "</parameter>"))
fraction ::= ("" | ("." [0-9] [0-9]*))
exponent ::= ("" | ("e" sign [0-9] [0-9]*) | ("E" sign [0-9] [0-9]*))
sign ::= ("" | ("+") | ("-"))
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
        number_call, "<parameter=arg1></parameter><parameter=arg2>123.45e-</parameter>"
    )


def test_string_function_call():
    string_call = GrammarFunctor.from_function_call(
        args_names=["arg1", "arg2"], args_types=["str", "str"]
    )

    expect_grammar = """root ::= ((arg1 arg2))
string ::= ((xml_content [ \\n\\t]* "</parameter>"))
xml_content ::= ((xml_content_1)) (=([ \\n\\t]* "</parameter>"))
escape ::= (([\\"\\\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
xml_content_1 ::= ("" | (xml_content_1_1 xml_content_1))
xml_content_1_1 ::= (([^<>&\\\\\\0-\\x1f]) | ("&lt;") | ("&gt;") | ("&amp;") | ("&quot;") | ("&apos;") | ("\\\\" escape)) (=(xml_content_1))
arg1 ::= (("<parameter=arg1>" [ \\n\\t]* string)) (=(arg2))
arg2 ::= (("<parameter=arg2>" [ \\n\\t]* string))
"""
    assert str(string_call) == expect_grammar
    assert _is_grammar_accept_string(
        string_call, "<parameter=arg1>hello world</parameter><parameter=arg2>123456</parameter>"
    )
    assert _is_grammar_accept_string(
        string_call, "<parameter=arg1>\t\n</parameter><parameter=arg2></parameter>"
    )
    assert _is_grammar_accept_string(
        string_call,
        "<parameter=arg1>&lt;hello&gt;</parameter><parameter=arg2>&lt;world&gt;</parameter>",
    )
    assert not _is_grammar_accept_string(
        string_call, "<parameter=arg1>&lt ;hello</parameter><parameter=arg2>abc</parameter>"
    )
    assert not _is_grammar_accept_string(
        string_call, "<parameter=arg1><hello</parameter><parameter=arg2>abc</parameter>"
    )
    assert not _is_grammar_accept_string(
        string_call, "<parameter=arg1>hello</parameter></parameter><parameter=arg2>abc</parameter>"
    )


def test_object_function_call():
    object_call = GrammarFunctor.from_function_call(args_names=["arg1"], args_types=["object"])
    assert (
        str(object_call)
        == """root ::= ((arg1))
root_1 ::= ((root_1_1 root_3))
root_1_1 ::= ((root_1_1_1) | ("null")) (=(root_3))
basic_escape ::= (([\\"\\\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(basic_string_sub))
basic_string_sub ::= (("\\"") | ([^<>&\\0-\\x1f\\"\\\\\\r\\n] basic_string_sub) | ("\\\\" basic_escape basic_string_sub) | ("&lt;" basic_string_sub) | ("&gt;" basic_string_sub) | ("&amp;" basic_string_sub) | ("&quot;" basic_string_sub) | ("&apos;" basic_string_sub)) (=([ \\n\\t]* [,}\\]:]))
basic_any ::= ((basic_number) | (basic_string) | ("true") | ("false") | ("null") | (basic_array) | (basic_object))
basic_number ::= ((basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\\"" basic_string_sub))
basic_array ::= (("[" [ \\n\\t]* basic_any basic_array_1 [ \\n\\t]* "]") | ("[" [ \\n\\t]* "]"))
basic_object ::= (("{" [ \\n\\t]* basic_string [ \\n\\t]* ":" [ \\n\\t]* basic_any basic_object_1 [ \\n\\t]* "}") | ("{" [ \\n\\t]* "}"))
root_1_1_1 ::= (("{" [ \\n\\t]* basic_string [ \\n\\t]* ":" [ \\n\\t]* basic_any root_1_1_1_1 [ \\n\\t]* "}") | ("{" [ \\n\\t]* "}"))
basic_number_1 ::= ("" | ("-")) (=([1-9] [0-9]*))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2)) (=(basic_number_6))
basic_number_4 ::= ("" | ([+\\-])) (=(basic_number_5))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \\n\\t]* "," [ \\n\\t]* basic_any basic_array_1)) (=([ \\n\\t]* "]"))
basic_object_1 ::= ("" | ([ \\n\\t]* "," [ \\n\\t]* basic_string [ \\n\\t]* ":" [ \\n\\t]* basic_any basic_object_1)) (=([ \\n\\t]* "}"))
root_1_1_1_1 ::= ("" | ([ \\n\\t]* "," [ \\n\\t]* basic_string [ \\n\\t]* ":" [ \\n\\t]* basic_any root_1_1_1_1)) (=([ \\n\\t]* "}"))
basic_number_7 ::= (("0") | (basic_number_1 [1-9] [0-9]*)) (=(basic_number_3 basic_number_6))
root_3 ::= (([ \\n\\t]* "</parameter>"))
arg1 ::= (("<parameter=arg1>" [ \\n\\t]* root_1))
"""
    )

    assert _is_grammar_accept_string(object_call, "<parameter=arg1>null</parameter>")
    assert _is_grammar_accept_string(
        object_call,
        '<parameter=arg1>{"key1": "value1", "key2": 123, "key3": {"sub_key1": "sub_value1"}}</parameter>',
    )
    assert _is_grammar_accept_string(object_call, "<parameter=arg1>{}</parameter>")
    assert not _is_grammar_accept_string(object_call, "<parameter=arg1>123</parameter>")
    assert not _is_grammar_accept_string(object_call, "<parameter=arg1></parameter>")
    assert not _is_grammar_accept_string(object_call, "<parameter=arg1>[123, 456]</parameter>")


arg_names_arg_types_input_accept = [
    (
        ["name", "age"],
        ["string", "int"],
        "<parameter=name>John</parameter><parameter=age>30</parameter>",
        True,
    ),
    (
        ["name", "scores", "awarded"],
        ["str", "dict", "bool"],
        """<parameter=name>
     John</parameter><parameter=scores>{\"math\": 90, \"science\": 85}</parameter><parameter=awarded>true</parameter>""",
        True,
    ),
    (
        ["height", "weight"],
        ["float", "float"],
        """<parameter=height>1.75</parameter><parameter=weight>70</parameter>""",
        True,
    ),
    (
        ["name", "scores"],
        ["str", "object"],
        "<parameter=name>John</parameter><parameter=scores>null</parameter>",
        True,
    ),
    (
        ["ID", "profit"],
        ["int", "float"],
        "<parameter=ID>abc</parameter><parameter=profit>1000.0</parameter>",
        False,
    ),
    (
        ["ID", "profit"],
        ["int", "float"],
        "<parameter=IC>1</parameter><parameter=profit>1000.0</parameter>",
        False,
    ),
    (["hobbies"], ["anything"], '<parameter=hobbies>reading and gaming"</parameter>', True),
]


@pytest.mark.parametrize(
    "input_args_names, input_args_types, input_string, accepted", arg_names_arg_types_input_accept
)
def test_complex_function_call(input_args_names, input_args_types, input_string, accepted):
    complex_function_call = GrammarFunctor.from_function_call(
        args_names=input_args_names, args_types=input_args_types
    )
    assert _is_grammar_accept_string(complex_function_call, input_string) == accepted


if __name__ == "__main__":
    pytest.main(sys.argv)
