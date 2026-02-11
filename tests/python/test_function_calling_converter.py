import sys

import pytest

from xgrammar import Grammar
from xgrammar.testing import (
    _deepseek_xml_tool_calling_to_ebnf,
    _is_grammar_accept_string,
    _minimax_xml_tool_calling_to_ebnf,
    _qwen_xml_tool_calling_to_ebnf,
)


def check_grammar_with_expected_grammar(grammar: Grammar, expected_grammar: str):
    assert (
        str(grammar).rstrip() == expected_grammar.rstrip()
    ), f"Expected grammar:\n{expected_grammar}\nActual grammar:\n{str(grammar)}"


def check_grammar_with_instance(grammar: Grammar, instance: str, accepted: bool):
    assert _is_grammar_accept_string(grammar, instance) == accepted


def _check_qwen_grammar(schema: dict, expected_grammar: str, instance: str, accepted: bool):
    ebnf_grammar = _qwen_xml_tool_calling_to_ebnf(schema)
    check_grammar_with_expected_grammar(ebnf_grammar, expected_grammar)
    check_grammar_with_instance(ebnf_grammar, instance, accepted)


def _check_minimax_grammar(schema: dict, expected_grammar: str, instance: str, accepted: bool):
    ebnf_grammar = _minimax_xml_tool_calling_to_ebnf(schema)
    check_grammar_with_expected_grammar(ebnf_grammar, expected_grammar)
    check_grammar_with_instance(ebnf_grammar, instance, accepted)


def _check_deepseek_grammar(schema: dict, expected_grammar: str, instance: str, accepted: bool):
    ebnf_grammar = _deepseek_xml_tool_calling_to_ebnf(schema)
    check_grammar_with_expected_grammar(ebnf_grammar, expected_grammar)
    check_grammar_with_instance(ebnf_grammar, instance, accepted)


test_string_schema_input_str_accepted = (
    ("<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter>", True),
    ("<parameter=name>Bob</parameter>\t\n<parameter=age>\t100\n</parameter>", False),
    ("<parameter=name>Bob</parameter><parameter=age>100</parameter>", True),
    (
        """<parameter=name><!DOCTYPE html>
<html lang="en">
  <body><h1>Hello</h1></body>
</html></parameter><parameter=age>100</parameter>""",
        True,
    ),
)


@pytest.mark.parametrize("input_str, accepted", test_string_schema_input_str_accepted)
def test_string_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_0 ::=  "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=   (("<parameter=name>" [ \n\t]* xml_string [ \n\t]* "</parameter>" root_part_0))
"""

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    _check_qwen_grammar(schema, expected_grammar, input_str, accepted)


test_additional_properties_schema_input_str_accepted = (
    (
        "<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter><parameter=location>New York</parameter>",
        True,
    ),
    (
        "<parameter=name>Bob</parameter><parameter=age>100</parameter><parameter=123invalid>A</parameter>",
        False,
    ),
)


@pytest.mark.parametrize(
    "input_str, accepted", test_additional_properties_schema_input_str_accepted
)
def test_additional_properties_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ( "<parameter=" xml_variable_name ">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root_part_0 ::=  "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::=   (("<parameter=name>" [ \n\t]* xml_string [ \n\t]* "</parameter>" root_part_0))
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": True,
    }
    _check_qwen_grammar(schema, expected_grammar, input_str, accepted)


test_not_required_properties_schema_input_str_accepted = (
    ("<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter>", True),
    ("<parameter=name>Bob</parameter>", True),
    ("<parameter=age>100</parameter>", True),
    ("", True),
    ("<parameter=anything>It's a string.</parameter>", True),
)


@pytest.mark.parametrize(
    "input_str, accepted", test_not_required_properties_schema_input_str_accepted
)
def test_not_required_properties_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ( "<parameter=" xml_variable_name ">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root_part_0 ::= root_part_1 |  "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::= (  (("<parameter=name>" [ \n\t]* xml_string [ \n\t]* "</parameter>" root_part_0) | ("<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1) | "<parameter=" xml_variable_name ">" [ \n\t]* root_addl [ \n\t]* "</parameter>" root_part_1) ) | [ \n\t]*
"""

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "additionalProperties": True,
    }
    _check_qwen_grammar(schema, expected_grammar, input_str, accepted)


test_part_required_properties_schema_input_str_accepted = (
    ("<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter>", True),
    ("<parameter=name>Bob</parameter>", True),
    ("<parameter=age>100</parameter>", False),
    (
        "<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter><parameter=anything>It's a string.</parameter>",
        True,
    ),
    ("<parameter=name>Bob</parameter><parameter=anything>It's a string.</parameter>", True),
    ("<parameter=anything>It's a string.</parameter>", False),
)


@pytest.mark.parametrize(
    "input_str, accepted", test_part_required_properties_schema_input_str_accepted
)
def test_part_required_properties_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ( "<parameter=" xml_variable_name ">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root_part_0 ::= root_part_1 |  "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::=   (("<parameter=name>" [ \n\t]* xml_string [ \n\t]* "</parameter>" root_part_0))
"""

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
        "additionalProperties": True,
    }
    _check_qwen_grammar(schema, expected_grammar, input_str, accepted)


def test_invalid_function_calling_schema():
    schema = {}

    with pytest.raises(RuntimeError):
        _qwen_xml_tool_calling_to_ebnf(schema)

    schema = {"type": "string"}

    with pytest.raises(RuntimeError):
        _qwen_xml_tool_calling_to_ebnf(schema)


test_inner_object_schema_input_str_accepted = (
    ('<parameter=address>{"street": "Main St", "city": "New York"}</parameter>', True),
    ('<parameter=address>{"street": "Main St", "city": "No more xml escape&<>"}</parameter>', True),
    ('<parameter=address>{"street": Main St, "city": New York}</parameter>', False),
    (
        "<parameter=address><parameter=street>Main St</parameter><parameter=city>New York</parameter></parameter>",
        False,
    ),
    ('<parameter=address>{"street": "Main St"}</parameter>', False),
    ('<parameter=address>{"city": "New York"}</parameter>', False),
)


@pytest.mark.parametrize("input_str, accepted", test_inner_object_schema_input_str_accepted)
def test_inner_object_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0_part_0 ::= [ \n\t]* "," [ \n\t]* "\"city\"" [ \n\t]* ":" [ \n\t]* basic_string ""
root_prop_0 ::= "{" [ \n\t]* (("\"street\"" [ \n\t]* ":" [ \n\t]* basic_string root_prop_0_part_0)) [ \n\t]* "}"
root ::=   (("<parameter=address>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" ""))
"""

    schema = {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {"street": {"type": "string"}, "city": {"type": "string"}},
                "required": ["street", "city"],
            }
        },
        "required": ["address"],
    }
    _check_qwen_grammar(schema, expected_grammar, input_str, accepted)


test_numbers_schema_input_str_accepted = (
    ("<parameter=age>25</parameter>", False),
    ("<parameter=name>Bob</parameter><parameter=age>25</parameter>", True),
    (
        "<parameter=name>Bob</parameter><parameter=ID>123456</parameter><parameter=is_student>true</parameter>",
        True,
    ),
    (
        "<parameter=name>John</parameter><parameter=age>1</parameter><parameter=ID>1</parameter><parameter=is_student>false</parameter>",
        False,
    ),
)


@pytest.mark.parametrize("input_str, accepted", test_numbers_schema_input_str_accepted)
def test_numbers_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_2 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_3 ::= "true" | "false"
root_part_2_1 ::=  "<parameter=is_student>" [ \n\t]* root_prop_3 [ \n\t]* "</parameter>" ""
root_part_2_2 ::= "" |  "<parameter=is_student>" [ \n\t]* root_prop_3 [ \n\t]* "</parameter>" ""
root_part_2_3 ::= ""
root_part_1_1 ::= root_part_2_1 |  "<parameter=ID>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_2
root_part_1_2 ::= root_part_2_2 |  "<parameter=ID>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_3
root_part_0_1 ::= root_part_1_1 |  "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1_2
root ::=   (("<parameter=name>" [ \n\t]* xml_string [ \n\t]* "</parameter>" root_part_0_1) | ("<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1_1) | ("<parameter=ID>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_1))
"""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "ID": {"type": "integer"},
            "is_student": {"type": "boolean"},
        },
        "maxProperties": 3,
        "minProperties": 2,
    }

    _check_qwen_grammar(schema, expected_grammar, input_str, accepted)


test_string_format_length_schema_input_str_accepted = {
    (
        '<parameter=name>ABC</parameter><parameter=contact_info>{"phone": "12345",   "email": "test@test.com"}</parameter>',
        True,
    ),
    (
        '<parameter=name>X</parameter><parameter=contact_info>{"phone": "67890", "email": "a@b.com"}</parameter>',
        True,
    ),
    (
        '<parameter=name></parameter><parameter=contact_info>{"phone": "12345", "email": "test@test.com"}</parameter>',
        False,
    ),
    (
        '<parameter=name>ABC</parameter><parameter=contact_info>{"phone": "1234", "email": "test@test.com"}</parameter>',
        False,
    ),
    (
        '<parameter=name>ABC</parameter><parameter=contact_info>{"phone": "12345", "email": "not-an-email"}</parameter>',
        False,
    ),
    (
        '<parameter=name>ABC</parameter><parameter=contact_info>{"phone": "12345"}</parameter>',
        False,
    ),
    (
        '<parameter=name>ABC</parameter><parameter=contact_info>{"email": "test@test.com"}</parameter>',
        False,
    ),
    ("<parameter=name>ABC</parameter>", False),
    ('<parameter=contact_info>{"phone": "12345", "email": "test@test.com"}</parameter>', False),
}


@pytest.mark.parametrize("input_str, accepted", test_string_format_length_schema_input_str_accepted)
def test_string_format_length_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0 ::= [^]{1,}
root_prop_1_prop_0 ::= "\"" [0-9]{5} "\""
root_prop_1_prop_1 ::= "\"" ( ( [a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+ ( "." [a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+ )* ) | "\\" "\"" ( "\\" [ -~] | [ !#-[\]-~] )* "\\" "\"" ) "@" ( [A-Za-z0-9] ( [\-A-Za-z0-9]* [A-Za-z0-9] )? ) ( ( "." [A-Za-z0-9] [\-A-Za-z0-9]* [A-Za-z0-9] )* ) "\""
root_prop_1_part_0 ::= [ \n\t]* "," [ \n\t]* "\"email\"" [ \n\t]* ":" [ \n\t]* root_prop_1_prop_1 ""
root_prop_1 ::= "{" [ \n\t]* (("\"phone\"" [ \n\t]* ":" [ \n\t]* root_prop_1_prop_0 root_prop_1_part_0)) [ \n\t]* "}"
root_part_0 ::=  "<parameter=contact_info>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=   (("<parameter=name>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" root_part_0))
"""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "contact_info": {
                "type": "object",
                "properties": {
                    "phone": {"type": "string", "pattern": "[0-9]{5}$"},
                    "email": {"type": "string", "format": "email"},
                },
                "required": ["phone", "email"],
            },
        },
        "required": ["name", "contact_info"],
    }

    _check_qwen_grammar(schema, expected_grammar, input_str, accepted)


test_array_schema_input_str_accepted = (
    ('<parameter=array>["foo", "bar"]</parameter>', True),
    ('<parameter=array>["foo", "bar", "baz"]</parameter>', True),
    ("<parameter=array>[]</parameter>", True),
    ("<parameter=array>[foo, bar, baz, qux, quux, corge]</parameter>", False),
)


@pytest.mark.parametrize("input_str, accepted", test_array_schema_input_str_accepted)
def test_array_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0 ::= (("[" [ \n\t]* basic_string ([ \n\t]* "," [ \n\t]* basic_string)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
root ::=   (("<parameter=array>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" ""))
"""
    schema = {
        "type": "object",
        "properties": {"array": {"type": "array", "items": {"type": "string"}}},
        "required": ["array"],
    }
    _check_qwen_grammar(schema, expected_grammar, input_str, accepted)


# ---------- MiniMax XML tool calling (_minimax_xml_tool_calling_to_ebnf) ----------
# Format: <parameter name="key">value</parameter> (not <parameter=key>)


minimax_test_string_schema_input_str_accepted = (
    ('<parameter name="name">Bob</parameter><parameter name="age">\t100\n</parameter>', True),
    ('<parameter name="name">Bob</parameter>\t\n<parameter name="age">\t100\n</parameter>', False),
    ('<parameter name="name">Bob</parameter><parameter name="age">100</parameter>', True),
    (
        """<parameter name="name"><!DOCTYPE html>
<html lang="en">
  <body><h1>Hello</h1></body>
</html></parameter><parameter name="age">100</parameter>""",
        True,
    ),
)


@pytest.mark.parametrize("input_str, accepted", minimax_test_string_schema_input_str_accepted)
def test_minimax_string_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_0 ::=  "<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=   (("<parameter name=\"name\">" [ \n\t]* xml_string [ \n\t]* "</parameter>" root_part_0))
"""

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    _check_minimax_grammar(schema, expected_grammar, input_str, accepted)


minimax_test_additional_properties_schema_input_str_accepted = (
    (
        '<parameter name="name">Bob</parameter><parameter name="age">\t100\n</parameter><parameter name="location">New York</parameter>',
        True,
    ),
    (
        '<parameter name="name">Bob</parameter><parameter name="age">100</parameter><parameter name="123invalid">A</parameter>',
        False,
    ),
)


@pytest.mark.parametrize(
    "input_str, accepted", minimax_test_additional_properties_schema_input_str_accepted
)
def test_minimax_additional_properties_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ( "<parameter name=\"" xml_variable_name "\">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root_part_0 ::=  "<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::=   (("<parameter name=\"name\">" [ \n\t]* xml_string [ \n\t]* "</parameter>" root_part_0))
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": True,
    }
    _check_minimax_grammar(schema, expected_grammar, input_str, accepted)


minimax_test_not_required_properties_schema_input_str_accepted = (
    ('<parameter name="name">Bob</parameter><parameter name="age">\t100\n</parameter>', True),
    ('<parameter name="name">Bob</parameter>', True),
    ('<parameter name="age">100</parameter>', True),
    ("", True),
    ('<parameter name="anything">It\'s a string.</parameter>', True),
)


@pytest.mark.parametrize(
    "input_str, accepted", minimax_test_not_required_properties_schema_input_str_accepted
)
def test_minimax_not_required_properties_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ( "<parameter name=\"" xml_variable_name "\">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root_part_0 ::= root_part_1 |  "<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::= (  (("<parameter name=\"name\">" [ \n\t]* xml_string [ \n\t]* "</parameter>" root_part_0) | ("<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1) | "<parameter name=\"" xml_variable_name "\">" [ \n\t]* root_addl [ \n\t]* "</parameter>" root_part_1) ) | [ \n\t]*
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "additionalProperties": True,
    }
    _check_minimax_grammar(schema, expected_grammar, input_str, accepted)


minimax_test_part_required_properties_schema_input_str_accepted = (
    ('<parameter name="name">Bob</parameter><parameter name="age">\t100\n</parameter>', True),
    ('<parameter name="name">Bob</parameter>', True),
    ('<parameter name="age">100</parameter>', False),
    (
        '<parameter name="name">Bob</parameter><parameter name="age">\t100\n</parameter><parameter name="anything">It\'s a string.</parameter>',
        True,
    ),
    (
        '<parameter name="name">Bob</parameter><parameter name="anything">It\'s a string.</parameter>',
        True,
    ),
    ('<parameter name="anything">It\'s a string.</parameter>', False),
)


@pytest.mark.parametrize(
    "input_str, accepted", minimax_test_part_required_properties_schema_input_str_accepted
)
def test_minimax_part_required_properties_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ( "<parameter name=\"" xml_variable_name "\">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root_part_0 ::= root_part_1 |  "<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::=   (("<parameter name=\"name\">" [ \n\t]* xml_string [ \n\t]* "</parameter>" root_part_0))
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
        "additionalProperties": True,
    }
    _check_minimax_grammar(schema, expected_grammar, input_str, accepted)


minimax_test_inner_object_schema_input_str_accepted = (
    ('<parameter name="address">{"street": "Main St", "city": "New York"}</parameter>', True),
    (
        '<parameter name="address">{"street": "Main St", "city": "No more xml escape&<>"}</parameter>',
        True,
    ),
    ('<parameter name="address">{"street": Main St, "city": New York}</parameter>', False),
    (
        '<parameter name="address"><parameter name="street">Main St</parameter><parameter name="city">New York</parameter></parameter>',
        False,
    ),
    ('<parameter name="address">{"street": "Main St"}</parameter>', False),
    ('<parameter name="address">{"city": "New York"}</parameter>', False),
    (
        '<parameter name="address">{"street": "Main St", "city": "New York", "additional_property": "value"}</parameter><parameter name="additional_property">value</parameter>',
        True,
    ),
    (
        '<parameter name="address">{"street": "Main St", "city": "New York", "additional_property": value}</parameter>',
        False,
    ),
)


@pytest.mark.parametrize("input_str, accepted", minimax_test_inner_object_schema_input_str_accepted)
def test_minimax_inner_object_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0_addl ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
root_prop_0_part_1 ::= ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* root_prop_0_addl)*
root_prop_0_part_0 ::= [ \n\t]* "," [ \n\t]* "\"city\"" [ \n\t]* ":" [ \n\t]* basic_string root_prop_0_part_1
root_prop_0 ::= "{" [ \n\t]* (("\"street\"" [ \n\t]* ":" [ \n\t]* basic_string root_prop_0_part_0)) [ \n\t]* "}"
root_addl ::= xml_string | basic_array | basic_object
root_part_0 ::= ( "<parameter name=\"" xml_variable_name "\">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root ::=   (("<parameter name=\"address\">" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" root_part_0))
"""
    schema = {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {"street": {"type": "string"}, "city": {"type": "string"}},
                "required": ["street", "city"],
                "additionalProperties": True,
            }
        },
        "additionalProperties": True,
        "required": ["address"],
    }
    _check_minimax_grammar(schema, expected_grammar, input_str, accepted)


minimax_test_numbers_schema_input_str_accepted = (
    ('<parameter name="age">25</parameter>', False),
    ('<parameter name="name">Bob</parameter><parameter name="age">25</parameter>', True),
    (
        '<parameter name="name">Bob</parameter><parameter name="ID">123456</parameter><parameter name="is_student">true</parameter>',
        True,
    ),
    (
        '<parameter name="name">John</parameter><parameter name="age">1</parameter><parameter name="ID">1</parameter><parameter name="is_student">false</parameter>',
        False,
    ),
)


@pytest.mark.parametrize("input_str, accepted", minimax_test_numbers_schema_input_str_accepted)
def test_minimax_numbers_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_2 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_3 ::= "true" | "false"
root_part_2_1 ::=  "<parameter name=\"is_student\">" [ \n\t]* root_prop_3 [ \n\t]* "</parameter>" ""
root_part_2_2 ::= "" |  "<parameter name=\"is_student\">" [ \n\t]* root_prop_3 [ \n\t]* "</parameter>" ""
root_part_2_3 ::= ""
root_part_1_1 ::= root_part_2_1 |  "<parameter name=\"ID\">" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_2
root_part_1_2 ::= root_part_2_2 |  "<parameter name=\"ID\">" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_3
root_part_0_1 ::= root_part_1_1 |  "<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1_2
root ::=   (("<parameter name=\"name\">" [ \n\t]* xml_string [ \n\t]* "</parameter>" root_part_0_1) | ("<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1_1) | ("<parameter name=\"ID\">" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_1))
"""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "ID": {"type": "integer"},
            "is_student": {"type": "boolean"},
        },
        "maxProperties": 3,
        "minProperties": 2,
    }
    _check_minimax_grammar(schema, expected_grammar, input_str, accepted)


minimax_test_string_format_length_schema_input_str_accepted = (
    (
        '<parameter name="name">ABC</parameter><parameter name="contact_info">{"phone": "12345",   "email": "test@test.com"}</parameter>',
        True,
    ),
    (
        '<parameter name="name">X</parameter><parameter name="contact_info">{"phone": "67890", "email": "a@b.com"}</parameter>',
        True,
    ),
    (
        '<parameter name="name"></parameter><parameter name="contact_info">{"phone": "12345", "email": "test@test.com"}</parameter>',
        False,
    ),
    (
        '<parameter name="name">ABC</parameter><parameter name="contact_info">{"phone": "1234", "email": "test@test.com"}</parameter>',
        False,
    ),
    (
        '<parameter name="name">ABC</parameter><parameter name="contact_info">{"phone": "12345", "email": "not-an-email"}</parameter>',
        False,
    ),
    (
        '<parameter name="name">ABC</parameter><parameter name="contact_info">{"phone": "12345"}</parameter>',
        False,
    ),
    (
        '<parameter name="name">ABC</parameter><parameter name="contact_info">{"email": "test@test.com"}</parameter>',
        False,
    ),
    ('<parameter name="name">ABC</parameter>', False),
    (
        '<parameter name="contact_info">{"phone": "12345", "email": "test@test.com"}</parameter>',
        False,
    ),
)


@pytest.mark.parametrize(
    "input_str, accepted", minimax_test_string_format_length_schema_input_str_accepted
)
def test_minimax_string_format_length_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0 ::= [^]{1,}
root_prop_1_prop_0 ::= "\"" [0-9]{5} "\""
root_prop_1_prop_1 ::= "\"" ( ( [a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+ ( "." [a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+ )* ) | "\\" "\"" ( "\\" [ -~] | [ !#-[\]-~] )* "\\" "\"" ) "@" ( [A-Za-z0-9] ( [\-A-Za-z0-9]* [A-Za-z0-9] )? ) ( ( "." [A-Za-z0-9] [\-A-Za-z0-9]* [A-Za-z0-9] )* ) "\""
root_prop_1_part_0 ::= [ \n\t]* "," [ \n\t]* "\"email\"" [ \n\t]* ":" [ \n\t]* root_prop_1_prop_1 ""
root_prop_1 ::= "{" [ \n\t]* (("\"phone\"" [ \n\t]* ":" [ \n\t]* root_prop_1_prop_0 root_prop_1_part_0)) [ \n\t]* "}"
root_part_0 ::=  "<parameter name=\"contact_info\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=   (("<parameter name=\"name\">" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" root_part_0))
"""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "contact_info": {
                "type": "object",
                "properties": {
                    "phone": {"type": "string", "pattern": "[0-9]{5}$"},
                    "email": {"type": "string", "format": "email"},
                },
                "required": ["phone", "email"],
            },
        },
        "required": ["name", "contact_info"],
    }
    _check_minimax_grammar(schema, expected_grammar, input_str, accepted)


# Minimax: reject Qwen format <parameter=key> and unquoted <parameter name=key>
minimax_reject_wrong_parameter_format_input_str_accepted = (
    ("<parameter=name>Bob</parameter><parameter=age>100</parameter>", False),  # Qwen format
    (
        "<parameter name=name>Bob</parameter><parameter name=age>100</parameter>",
        False,
    ),  # unquoted key
    (
        '<parameter name="name">Bob</parameter><parameter name="age">100</parameter>',
        True,
    ),  # correct
)


@pytest.mark.parametrize(
    "input_str, accepted", minimax_reject_wrong_parameter_format_input_str_accepted
)
def test_minimax_reject_wrong_parameter_format(input_str: str, accepted: bool):
    """MiniMax grammar must accept <parameter name=\"key\"> but reject <parameter=key> and <parameter name=key>."""
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ( "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_0 ::=  "<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=   (("<parameter name=\"name\">" [ \n\t]* xml_string [ \n\t]* "</parameter>" root_part_0))
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    _check_minimax_grammar(schema, expected_grammar, input_str, accepted)


# ---------- DeepSeek XML tool calling (_deepseek_xml_tool_calling_to_ebnf) ----------
# Format: <{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>


deepseek_test_string_schema_input_str_accepted = (
    (
        '<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter><{dsml_token}parameter name="age" string="false">\t100\n</{dsml_token}parameter>',
        True,
    ),
    (
        '<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter>\t\n<{dsml_token}parameter name="age" string="true">\t100\n</{dsml_token}parameter>',
        False,
    ),
    (
        '<{dsml_token}parameter name="name" string="false">Bob</{dsml_token}parameter><{dsml_token}parameter name="age" string="true">100</{dsml_token}parameter>',
        True,
    ),
    (
        """<{dsml_token}parameter name="name" string="true"><!DOCTYPE html>
<html lang="en">
  <body><h1>Hello</h1></body>
</html></{dsml_token}parameter><{dsml_token}parameter name="age" string="false">100</{dsml_token}parameter>""",
        True,
    ),
    ('<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter>', False),
    ('<{dsml_token}parameter name="age" string="false">100</{dsml_token}parameter>', False),
    (
        '<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter><{dsml_token}parameter name="age" string="false">100',
        False,
    ),
    (
        '<{dsml_token}parameter name="name">Bob</{dsml_token}parameter><{dsml_token}parameter name="age" string="false">100</{dsml_token}parameter>',
        False,
    ),
    (
        '<{dsml_token}parameter name="name" string="true">Bob</parameter><{dsml_token}parameter name="age" string="false">100</{dsml_token}parameter>',
        False,
    ),
)


@pytest.mark.parametrize("input_str, accepted", deepseek_test_string_schema_input_str_accepted)
def test_deepseek_string_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</{dsml_token}parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>" ( "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_0 ::=  "<{dsml_token}parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</{dsml_token}parameter>" ""
root ::=   (("<{dsml_token}parameter name=\"name\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_string [ \n\t]* "</{dsml_token}parameter>" root_part_0))
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    _check_deepseek_grammar(schema, expected_grammar, input_str, accepted)


deepseek_test_additional_properties_schema_input_str_accepted = (
    (
        '<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter><{dsml_token}parameter name="age" string="false">\t100\n</{dsml_token}parameter><{dsml_token}parameter name="location" string="true">New York</{dsml_token}parameter>',
        True,
    ),
    (
        '<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter><{dsml_token}parameter name="age" string="true">100</{dsml_token}parameter><{dsml_token}parameter name="123invalid" string="false">A</{dsml_token}parameter>',
        False,
    ),
    (
        '<{dsml_token}parameter name="location" string="true">New York</{dsml_token}parameter>',
        False,
    ),
    ('<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter>', False),
    (
        '<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter><{dsml_token}parameter name="age" string="false">100',
        False,
    ),
)


@pytest.mark.parametrize(
    "input_str, accepted", deepseek_test_additional_properties_schema_input_str_accepted
)
def test_deepseek_additional_properties_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</{dsml_token}parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>" ( "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ( "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* root_addl [ \n\t]* "</{dsml_token}parameter>")*
root_part_0 ::=  "<{dsml_token}parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</{dsml_token}parameter>" root_part_1
root ::=   (("<{dsml_token}parameter name=\"name\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_string [ \n\t]* "</{dsml_token}parameter>" root_part_0))
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": True,
    }
    _check_deepseek_grammar(schema, expected_grammar, input_str, accepted)


deepseek_test_not_required_properties_schema_input_str_accepted = (
    (
        '<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter><{dsml_token}parameter name="age" string="false">\t100\n</{dsml_token}parameter>',
        True,
    ),
    ('<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter>', True),
    ('<{dsml_token}parameter name="age" string="false">100</{dsml_token}parameter>', True),
    ("", True),
    (
        '<{dsml_token}parameter name="anything" string="true">It\'s a string.</{dsml_token}parameter>',
        True,
    ),
    ('<{dsml_token}parameter name="name" string="true">Bob', False),
    ('<{dsml_token}parameter name="name">Bob</{dsml_token}parameter>', False),
    ('<{dsml_token}parameter name="x" string="true">y</parameter>', False),
)


@pytest.mark.parametrize(
    "input_str, accepted", deepseek_test_not_required_properties_schema_input_str_accepted
)
def test_deepseek_not_required_properties_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</{dsml_token}parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>" ( "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ( "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* root_addl [ \n\t]* "</{dsml_token}parameter>")*
root_part_0 ::= root_part_1 |  "<{dsml_token}parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</{dsml_token}parameter>" root_part_1
root ::= (  (("<{dsml_token}parameter name=\"name\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_string [ \n\t]* "</{dsml_token}parameter>" root_part_0) | ("<{dsml_token}parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</{dsml_token}parameter>" root_part_1) | "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* root_addl [ \n\t]* "</{dsml_token}parameter>" root_part_1) ) | [ \n\t]*
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "additionalProperties": True,
    }
    _check_deepseek_grammar(schema, expected_grammar, input_str, accepted)


deepseek_test_part_required_properties_schema_input_str_accepted = (
    (
        '<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter><{dsml_token}parameter name="age" string="false">\t100\n</{dsml_token}parameter>',
        True,
    ),
    ('<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter>', True),
    ('<{dsml_token}parameter name="age" string="true">100</{dsml_token}parameter>', False),
    (
        '<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter><{dsml_token}parameter name="age" string="false">\t100\n</{dsml_token}parameter><{dsml_token}parameter name="anything" string="true">It\'s a string.</{dsml_token}parameter>',
        True,
    ),
    (
        '<{dsml_token}parameter name="name" string="false">Bob</{dsml_token}parameter><{dsml_token}parameter name="anything" string="true">It\'s a string.</{dsml_token}parameter>',
        True,
    ),
    (
        '<{dsml_token}parameter name="anything" string="true">It\'s a string.</{dsml_token}parameter>',
        False,
    ),
)


@pytest.mark.parametrize(
    "input_str, accepted", deepseek_test_part_required_properties_schema_input_str_accepted
)
def test_deepseek_part_required_properties_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</{dsml_token}parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>" ( "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ( "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* root_addl [ \n\t]* "</{dsml_token}parameter>")*
root_part_0 ::= root_part_1 |  "<{dsml_token}parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</{dsml_token}parameter>" root_part_1
root ::=   (("<{dsml_token}parameter name=\"name\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_string [ \n\t]* "</{dsml_token}parameter>" root_part_0))
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
        "additionalProperties": True,
    }
    _check_deepseek_grammar(schema, expected_grammar, input_str, accepted)


deepseek_test_inner_object_schema_input_str_accepted = (
    (
        '<{dsml_token}parameter name="address" string="true">{"street": "Main St", "city": "New York"}</{dsml_token}parameter>',
        True,
    ),
    (
        '<{dsml_token}parameter name="address" string="false">{"street": "Main St", "city": "No more xml escape&<>"}</{dsml_token}parameter>',
        True,
    ),
    (
        '<{dsml_token}parameter name="address" string="true">{"street": Main St, "city": New York}</{dsml_token}parameter>',
        False,
    ),
    (
        '<{dsml_token}parameter name="address" string="true"><{dsml_token}parameter name="street" string="true">Main St</{dsml_token}parameter><{dsml_token}parameter name="city" string="true">New York</{dsml_token}parameter></{dsml_token}parameter>',
        False,
    ),
    (
        '<{dsml_token}parameter name="address" string="true">{"street": "Main St"}</{dsml_token}parameter>',
        False,
    ),
    (
        '<{dsml_token}parameter name="address" string="false">{"city": "New York"}</{dsml_token}parameter>',
        False,
    ),
    (
        '<{dsml_token}parameter name="address" string="true">{"street": "Main St", "city": "New York", "additional_property": "value"}</{dsml_token}parameter><{dsml_token}parameter name="additional_property" string="true">value</{dsml_token}parameter>',
        True,
    ),
    (
        '<{dsml_token}parameter name="address" string="true">{"street": "Main St", "city": "New York", "additional_property": value}</{dsml_token}parameter>',
        False,
    ),
)


@pytest.mark.parametrize(
    "input_str, accepted", deepseek_test_inner_object_schema_input_str_accepted
)
def test_deepseek_inner_object_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</{dsml_token}parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>" ( "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0_addl ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
root_prop_0_part_1 ::= ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* root_prop_0_addl)*
root_prop_0_part_0 ::= [ \n\t]* "," [ \n\t]* "\"city\"" [ \n\t]* ":" [ \n\t]* basic_string root_prop_0_part_1
root_prop_0 ::= "{" [ \n\t]* (("\"street\"" [ \n\t]* ":" [ \n\t]* basic_string root_prop_0_part_0)) [ \n\t]* "}"
root_addl ::= xml_string | basic_array | basic_object
root_part_0 ::= ( "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* root_addl [ \n\t]* "</{dsml_token}parameter>")*
root ::=   (("<{dsml_token}parameter name=\"address\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_0 [ \n\t]* "</{dsml_token}parameter>" root_part_0))
"""
    schema = {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {"street": {"type": "string"}, "city": {"type": "string"}},
                "required": ["street", "city"],
                "additionalProperties": True,
            }
        },
        "additionalProperties": True,
        "required": ["address"],
    }
    _check_deepseek_grammar(schema, expected_grammar, input_str, accepted)


deepseek_test_numbers_schema_input_str_accepted = (
    ('<{dsml_token}parameter name="age" string="false">25</{dsml_token}parameter>', False),
    (
        '<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter><{dsml_token}parameter name="age" string="false">25</{dsml_token}parameter>',
        True,
    ),
    (
        '<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter><{dsml_token}parameter name="ID" string="false">123456</{dsml_token}parameter><{dsml_token}parameter name="is_student" string="true">true</{dsml_token}parameter>',
        True,
    ),
    (
        '<{dsml_token}parameter name="name" string="true">John</{dsml_token}parameter><{dsml_token}parameter name="age" string="false">1</{dsml_token}parameter><{dsml_token}parameter name="ID" string="false">1</{dsml_token}parameter><{dsml_token}parameter name="is_student" string="false">false</{dsml_token}parameter>',
        False,
    ),
)


@pytest.mark.parametrize("input_str, accepted", deepseek_test_numbers_schema_input_str_accepted)
def test_deepseek_numbers_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</{dsml_token}parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>" ( "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_2 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_3 ::= "true" | "false"
root_part_2_1 ::=  "<{dsml_token}parameter name=\"is_student\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_3 [ \n\t]* "</{dsml_token}parameter>" ""
root_part_2_2 ::= "" |  "<{dsml_token}parameter name=\"is_student\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_3 [ \n\t]* "</{dsml_token}parameter>" ""
root_part_2_3 ::= ""
root_part_1_1 ::= root_part_2_1 |  "<{dsml_token}parameter name=\"ID\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_2 [ \n\t]* "</{dsml_token}parameter>" root_part_2_2
root_part_1_2 ::= root_part_2_2 |  "<{dsml_token}parameter name=\"ID\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_2 [ \n\t]* "</{dsml_token}parameter>" root_part_2_3
root_part_0_1 ::= root_part_1_1 |  "<{dsml_token}parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</{dsml_token}parameter>" root_part_1_2
root ::=   (("<{dsml_token}parameter name=\"name\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_string [ \n\t]* "</{dsml_token}parameter>" root_part_0_1) | ("<{dsml_token}parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</{dsml_token}parameter>" root_part_1_1) | ("<{dsml_token}parameter name=\"ID\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_2 [ \n\t]* "</{dsml_token}parameter>" root_part_2_1))
"""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "ID": {"type": "integer"},
            "is_student": {"type": "boolean"},
        },
        "maxProperties": 3,
        "minProperties": 2,
    }
    _check_deepseek_grammar(schema, expected_grammar, input_str, accepted)


# DeepSeek: reject Qwen format <parameter=key>, Minimax format <parameter name="key"> (no string=), accept <{dsml_token}parameter name="key" string="true|false">
deepseek_reject_wrong_parameter_format_input_str_accepted = (
    ("<parameter=name>Bob</parameter><parameter=age>100</parameter>", False),  # Qwen format
    (
        '<parameter name="name">Bob</parameter><parameter name="age">100</parameter>',
        False,
    ),  # Minimax format (no string=)
    (
        '<{dsml_token}parameter name="name" string="true">Bob</{dsml_token}parameter><{dsml_token}parameter name="age" string="false">100</{dsml_token}parameter>',
        True,
    ),  # correct
)


@pytest.mark.parametrize(
    "input_str, accepted", deepseek_reject_wrong_parameter_format_input_str_accepted
)
def test_deepseek_reject_wrong_parameter_format(input_str: str, accepted: bool):
    """DeepSeek grammar must accept <{dsml_token}parameter name=\"key\" string=\"true|false\">, reject Qwen and Minimax formats."""
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
xml_string ::= TagDispatch(stop_eos=true,stop_str=(),loop_after_dispatch=false,excludes=("</{dsml_token}parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= (  "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>" ( "<{dsml_token}parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</{dsml_token}parameter>")* ) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_0 ::=  "<{dsml_token}parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</{dsml_token}parameter>" ""
root ::=   (("<{dsml_token}parameter name=\"name\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_string [ \n\t]* "</{dsml_token}parameter>" root_part_0))
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    _check_deepseek_grammar(schema, expected_grammar, input_str, accepted)


if __name__ == "__main__":
    pytest.main(sys.argv)
