import sys

import pytest

from xgrammar import Grammar
from xgrammar.testing import (
    _get_matcher_from_grammar,
    _is_grammar_accept_string,
    _json_schema_to_ebnf,
)


def check_grammar_with_expected_grammar(grammar: Grammar, expected_grammar: str):
    assert (
        str(grammar).rstrip() == expected_grammar.rstrip()
    ), f"Expected grammar:\n{expected_grammar}\nActual grammar:\n{str(grammar)}"


def check_grammar_with_instance(grammar: Grammar, instance: str, accepted: bool):
    assert _is_grammar_accept_string(grammar, instance) == accepted


def _check_qwen_grammar(schema: dict, expected_grammar: str, instance: str, accepted: bool):
    ebnf_grammar = _json_schema_to_ebnf(schema, json_format="qwen_xml")
    check_grammar_with_expected_grammar(ebnf_grammar, expected_grammar)
    check_grammar_with_instance(ebnf_grammar, instance, accepted)


def _check_minimax_grammar(schema: dict, expected_grammar: str, instance: str, accepted: bool):
    ebnf_grammar = _json_schema_to_ebnf(schema, json_format="minimax_xml")
    check_grammar_with_expected_grammar(ebnf_grammar, expected_grammar)
    check_grammar_with_instance(ebnf_grammar, instance, accepted)


def _check_deepseek_grammar(schema: dict, expected_grammar: str, instance: str, accepted: bool):
    ebnf_grammar = _json_schema_to_ebnf(schema, json_format="deepseek_xml")
    check_grammar_with_expected_grammar(ebnf_grammar, expected_grammar)
    check_grammar_with_instance(ebnf_grammar, instance, accepted)


def _check_glm_grammar(schema: dict, instance: str, accepted: bool):
    ebnf_grammar = _json_schema_to_ebnf(schema, json_format="glm_xml")
    check_grammar_with_instance(ebnf_grammar, instance, accepted)


test_string_schema_input_str_accepted = (
    ("<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter>", True),
    ("<parameter=name>Bob</parameter>\t\n<parameter=age>\t100\n</parameter>", True),
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_0 ::= [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=  [ \n\t]* (("<parameter=name>" xml_string "</parameter>" root_part_0)) [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root_part_0 ::= [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::=  [ \n\t]* (("<parameter=name>" xml_string "</parameter>" root_part_0)) [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root_part_0 ::= root_part_1 | [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::= ( [ \n\t]* (("<parameter=name>" xml_string "</parameter>" root_part_0) | ("<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1) | "<parameter=" xml_variable_name ">" [ \n\t]* root_addl [ \n\t]* "</parameter>" root_part_1) [ \n\t]*) | [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root_part_0 ::= root_part_1 | [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::=  [ \n\t]* (("<parameter=name>" xml_string "</parameter>" root_part_0)) [ \n\t]*
"""

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
        "additionalProperties": True,
    }
    _check_qwen_grammar(schema, expected_grammar, input_str, accepted)


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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0_part_0 ::= [ \n\t]* "," [ \n\t]* "\"city\"" [ \n\t]* ":" [ \n\t]* basic_string ""
root_prop_0 ::= "{" [ \n\t]* (("\"street\"" [ \n\t]* ":" [ \n\t]* basic_string root_prop_0_part_0)) [ \n\t]* "}"
root ::=  [ \n\t]* (("<parameter=address>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" "")) [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_2 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_3 ::= "true" | "false"
root_part_2_1 ::= [ \n\t]* "<parameter=is_student>" [ \n\t]* root_prop_3 [ \n\t]* "</parameter>" ""
root_part_2_2 ::= "" | [ \n\t]* "<parameter=is_student>" [ \n\t]* root_prop_3 [ \n\t]* "</parameter>" ""
root_part_2_3 ::= ""
root_part_1_1 ::= root_part_2_1 | [ \n\t]* "<parameter=ID>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_2
root_part_1_2 ::= root_part_2_2 | [ \n\t]* "<parameter=ID>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_3
root_part_0_1 ::= root_part_1_1 | [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1_2
root ::=  [ \n\t]* (("<parameter=name>" xml_string "</parameter>" root_part_0_1) | ("<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1_1) | ("<parameter=ID>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_1)) [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0 ::= [^]{1,}
root_prop_1_prop_0 ::= "\"" [0-9]{5} "\""
root_prop_1_prop_1 ::= "\"" ( ( [a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+ ( "." [a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+ )* ) | "\\" "\"" ( "\\" [ -~] | [ !#-[\]-~] )* "\\" "\"" ) "@" ( [A-Za-z0-9] ( [\-A-Za-z0-9]* [A-Za-z0-9] )? ) ( ( "." [A-Za-z0-9] [\-A-Za-z0-9]* [A-Za-z0-9] )* ) "\""
root_prop_1_part_0 ::= [ \n\t]* "," [ \n\t]* "\"email\"" [ \n\t]* ":" [ \n\t]* root_prop_1_prop_1 ""
root_prop_1 ::= "{" [ \n\t]* (("\"phone\"" [ \n\t]* ":" [ \n\t]* root_prop_1_prop_0 root_prop_1_part_0)) [ \n\t]* "}"
root_part_0 ::= [ \n\t]* "<parameter=contact_info>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=  [ \n\t]* (("<parameter=name>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" root_part_0)) [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0 ::= (("[" [ \n\t]* basic_string ([ \n\t]* "," [ \n\t]* basic_string)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
root ::=  [ \n\t]* (("<parameter=array>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" "")) [ \n\t]*
"""
    schema = {
        "type": "object",
        "properties": {"array": {"type": "array", "items": {"type": "string"}}},
        "required": ["array"],
    }
    _check_qwen_grammar(schema, expected_grammar, input_str, accepted)


# ---------- MiniMax XML tool calling (json_format="minimax_xml") ----------
# Format: <parameter name="key">value</parameter> (not <parameter=key>)


minimax_test_string_schema_input_str_accepted = (
    ('<parameter name="name">Bob</parameter><parameter name="age">\t100\n</parameter>', True),
    ('<parameter name="name">Bob</parameter>\t\n<parameter name="age">\t100\n</parameter>', True),
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_0 ::= [ \n\t]* "<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=  [ \n\t]* (("<parameter name=\"name\">" xml_string "</parameter>" root_part_0)) [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root_part_0 ::= [ \n\t]* "<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::=  [ \n\t]* (("<parameter name=\"name\">" xml_string "</parameter>" root_part_0)) [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root_part_0 ::= root_part_1 | [ \n\t]* "<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::= ( [ \n\t]* (("<parameter name=\"name\">" xml_string "</parameter>" root_part_0) | ("<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1) | "<parameter name=\"" xml_variable_name "\">" [ \n\t]* root_addl [ \n\t]* "</parameter>" root_part_1) [ \n\t]*) | [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root_part_0 ::= root_part_1 | [ \n\t]* "<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::=  [ \n\t]* (("<parameter name=\"name\">" xml_string "</parameter>" root_part_0)) [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0_addl ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
root_prop_0_addl_key ::= ["] (("\"" | [^cs\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "c" ("\"" | [^i\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "i" ("\"" | [^t\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "t" ("\"" | [^y\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "y" ([^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub)))) | "s" ("\"" | [^t\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "t" ("\"" | [^r\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "r" ("\"" | [^e\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "e" ("\"" | [^e\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "e" ("\"" | [^t\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "t" ([^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub)))))))) (= [ \n\t]* [,}\]:])
root_prop_0_part_1 ::= ([ \n\t]* "," [ \n\t]* root_prop_0_addl_key [ \n\t]* ":" [ \n\t]* root_prop_0_addl)*
root_prop_0_part_0 ::= [ \n\t]* "," [ \n\t]* "\"city\"" [ \n\t]* ":" [ \n\t]* basic_string root_prop_0_part_1
root_prop_0 ::= "{" [ \n\t]* (("\"street\"" [ \n\t]* ":" [ \n\t]* basic_string root_prop_0_part_0)) [ \n\t]* "}"
root_addl ::= xml_string | basic_array | basic_object
root_part_0 ::= ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* root_addl [ \n\t]* "</parameter>")*
root ::=  [ \n\t]* (("<parameter name=\"address\">" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" root_part_0)) [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_2 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_3 ::= "true" | "false"
root_part_2_1 ::= [ \n\t]* "<parameter name=\"is_student\">" [ \n\t]* root_prop_3 [ \n\t]* "</parameter>" ""
root_part_2_2 ::= "" | [ \n\t]* "<parameter name=\"is_student\">" [ \n\t]* root_prop_3 [ \n\t]* "</parameter>" ""
root_part_2_3 ::= ""
root_part_1_1 ::= root_part_2_1 | [ \n\t]* "<parameter name=\"ID\">" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_2
root_part_1_2 ::= root_part_2_2 | [ \n\t]* "<parameter name=\"ID\">" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_3
root_part_0_1 ::= root_part_1_1 | [ \n\t]* "<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1_2
root ::=  [ \n\t]* (("<parameter name=\"name\">" xml_string "</parameter>" root_part_0_1) | ("<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1_1) | ("<parameter name=\"ID\">" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_1)) [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0 ::= [^]{1,}
root_prop_1_prop_0 ::= "\"" [0-9]{5} "\""
root_prop_1_prop_1 ::= "\"" ( ( [a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+ ( "." [a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+ )* ) | "\\" "\"" ( "\\" [ -~] | [ !#-[\]-~] )* "\\" "\"" ) "@" ( [A-Za-z0-9] ( [\-A-Za-z0-9]* [A-Za-z0-9] )? ) ( ( "." [A-Za-z0-9] [\-A-Za-z0-9]* [A-Za-z0-9] )* ) "\""
root_prop_1_part_0 ::= [ \n\t]* "," [ \n\t]* "\"email\"" [ \n\t]* ":" [ \n\t]* root_prop_1_prop_1 ""
root_prop_1 ::= "{" [ \n\t]* (("\"phone\"" [ \n\t]* ":" [ \n\t]* root_prop_1_prop_0 root_prop_1_part_0)) [ \n\t]* "}"
root_part_0 ::= [ \n\t]* "<parameter name=\"contact_info\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=  [ \n\t]* (("<parameter name=\"name\">" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" root_part_0)) [ \n\t]*
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_0 ::= [ \n\t]* "<parameter name=\"age\">" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=  [ \n\t]* (("<parameter name=\"name\">" xml_string "</parameter>" root_part_0)) [ \n\t]*
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    _check_minimax_grammar(schema, expected_grammar, input_str, accepted)


# ---------- DeepSeek XML tool calling (json_format="deepseek_xml") ----------
# Format: <｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>


deepseek_test_string_schema_input_str_accepted = (
    (
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="false">\t100\n</｜DSML｜parameter>',
        True,
    ),
    (
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter>\t\n<｜DSML｜parameter name="age" string="true">\t100\n</｜DSML｜parameter>',
        True,
    ),
    (
        '<｜DSML｜parameter name="name" string="false">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="true">100</｜DSML｜parameter>',
        True,
    ),
    (
        """<｜DSML｜parameter name="name" string="true"><!DOCTYPE html>
<html lang="en">
  <body><h1>Hello</h1></body>
</html></｜DSML｜parameter><｜DSML｜parameter name="age" string="false">100</｜DSML｜parameter>""",
        True,
    ),
    ('<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter>', False),
    ('<｜DSML｜parameter name="age" string="false">100</｜DSML｜parameter>', False),
    (
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="false">100',
        False,
    ),
    (
        '<｜DSML｜parameter name="name">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="false">100</｜DSML｜parameter>',
        False,
    ),
    (
        '<｜DSML｜parameter name="name" string="true">Bob</parameter><｜DSML｜parameter name="age" string="false">100</｜DSML｜parameter>',
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</｜DSML｜parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>" ([ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_0 ::= [ \n\t]* "<｜DSML｜parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</｜DSML｜parameter>" ""
root ::=  [ \n\t]* (("<｜DSML｜parameter name=\"name\" string=\"" ("true" | "false") "\">" xml_string "</｜DSML｜parameter>" root_part_0)) [ \n\t]*
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    _check_deepseek_grammar(schema, expected_grammar, input_str, accepted)


deepseek_pattern_empty_leading_alternative_input_str_accepted = (
    ('<｜DSML｜parameter name="url" string="true">https://x.com/</｜DSML｜parameter>', True),
    # The "^$" branch allows an empty value.
    ('<｜DSML｜parameter name="url" string="true"></｜DSML｜parameter>', True),
    ('<｜DSML｜parameter name="url" string="true">http://x.com/</｜DSML｜parameter>', False),
)


@pytest.mark.parametrize(
    "input_str, accepted", deepseek_pattern_empty_leading_alternative_input_str_accepted
)
def test_deepseek_pattern_empty_leading_alternative(input_str: str, accepted: bool):
    # Regression: a pattern whose first alternative is empty ("^$|...") used to emit a bare
    # leading '|' (root_prop_0 ::= | ...) and crash the grammar parser on the deepseek_xml path.
    # It must now be emitted as root_prop_0 ::= "" | ...
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</｜DSML｜parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>" ([ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0 ::= "" | "h" "t" "t" "p" "s" ":" "/" "/" "x" "." "c" "o" "m" "/"
root ::=  [ \n\t]* (("<｜DSML｜parameter name=\"url\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_0 [ \n\t]* "</｜DSML｜parameter>" "")) [ \n\t]*
"""
    schema = {
        "type": "object",
        "properties": {"url": {"type": "string", "pattern": "^$|^https://x\\.com/"}},
        "required": ["url"],
    }
    _check_deepseek_grammar(schema, expected_grammar, input_str, accepted)


deepseek_test_additional_properties_schema_input_str_accepted = (
    (
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="false">\t100\n</｜DSML｜parameter><｜DSML｜parameter name="location" string="true">New York</｜DSML｜parameter>',
        True,
    ),
    (
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="true">100</｜DSML｜parameter><｜DSML｜parameter name="123invalid" string="false">A</｜DSML｜parameter>',
        False,
    ),
    ('<｜DSML｜parameter name="location" string="true">New York</｜DSML｜parameter>', False),
    ('<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter>', False),
    (
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="false">100',
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</｜DSML｜parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>" ([ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ([ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* root_addl [ \n\t]* "</｜DSML｜parameter>")*
root_part_0 ::= [ \n\t]* "<｜DSML｜parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</｜DSML｜parameter>" root_part_1
root ::=  [ \n\t]* (("<｜DSML｜parameter name=\"name\" string=\"" ("true" | "false") "\">" xml_string "</｜DSML｜parameter>" root_part_0)) [ \n\t]*
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
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="false">\t100\n</｜DSML｜parameter>',
        True,
    ),
    ('<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter>', True),
    ('<｜DSML｜parameter name="age" string="false">100</｜DSML｜parameter>', True),
    ("", True),
    ('<｜DSML｜parameter name="anything" string="true">It\'s a string.</｜DSML｜parameter>', True),
    ('<｜DSML｜parameter name="name" string="true">Bob', False),
    ('<｜DSML｜parameter name="name">Bob</｜DSML｜parameter>', False),
    ('<｜DSML｜parameter name="x" string="true">y</parameter>', False),
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</｜DSML｜parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>" ([ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ([ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* root_addl [ \n\t]* "</｜DSML｜parameter>")*
root_part_0 ::= root_part_1 | [ \n\t]* "<｜DSML｜parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</｜DSML｜parameter>" root_part_1
root ::= ( [ \n\t]* (("<｜DSML｜parameter name=\"name\" string=\"" ("true" | "false") "\">" xml_string "</｜DSML｜parameter>" root_part_0) | ("<｜DSML｜parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</｜DSML｜parameter>" root_part_1) | "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* root_addl [ \n\t]* "</｜DSML｜parameter>" root_part_1) [ \n\t]*) | [ \n\t]*
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "additionalProperties": True,
    }
    _check_deepseek_grammar(schema, expected_grammar, input_str, accepted)


deepseek_test_part_required_properties_schema_input_str_accepted = (
    (
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="false">\t100\n</｜DSML｜parameter>',
        True,
    ),
    ('<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter>', True),
    ('<｜DSML｜parameter name="age" string="true">100</｜DSML｜parameter>', False),
    (
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="false">\t100\n</｜DSML｜parameter><｜DSML｜parameter name="anything" string="true">It\'s a string.</｜DSML｜parameter>',
        True,
    ),
    (
        '<｜DSML｜parameter name="name" string="false">Bob</｜DSML｜parameter><｜DSML｜parameter name="anything" string="true">It\'s a string.</｜DSML｜parameter>',
        True,
    ),
    ('<｜DSML｜parameter name="anything" string="true">It\'s a string.</｜DSML｜parameter>', False),
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</｜DSML｜parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>" ([ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_addl ::= xml_string | basic_array | basic_object
root_part_1 ::= ([ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* root_addl [ \n\t]* "</｜DSML｜parameter>")*
root_part_0 ::= root_part_1 | [ \n\t]* "<｜DSML｜parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</｜DSML｜parameter>" root_part_1
root ::=  [ \n\t]* (("<｜DSML｜parameter name=\"name\" string=\"" ("true" | "false") "\">" xml_string "</｜DSML｜parameter>" root_part_0)) [ \n\t]*
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
        '<｜DSML｜parameter name="address" string="true">{"street": "Main St", "city": "New York"}</｜DSML｜parameter>',
        True,
    ),
    (
        '<｜DSML｜parameter name="address" string="false">{"street": "Main St", "city": "No more xml escape&<>"}</｜DSML｜parameter>',
        True,
    ),
    (
        '<｜DSML｜parameter name="address" string="true">{"street": Main St, "city": New York}</｜DSML｜parameter>',
        False,
    ),
    (
        '<｜DSML｜parameter name="address" string="true"><｜DSML｜parameter name="street" string="true">Main St</｜DSML｜parameter><｜DSML｜parameter name="city" string="true">New York</｜DSML｜parameter></｜DSML｜parameter>',
        False,
    ),
    (
        '<｜DSML｜parameter name="address" string="true">{"street": "Main St"}</｜DSML｜parameter>',
        False,
    ),
    (
        '<｜DSML｜parameter name="address" string="false">{"city": "New York"}</｜DSML｜parameter>',
        False,
    ),
    (
        '<｜DSML｜parameter name="address" string="true">{"street": "Main St", "city": "New York", "additional_property": "value"}</｜DSML｜parameter><｜DSML｜parameter name="additional_property" string="true">value</｜DSML｜parameter>',
        True,
    ),
    (
        '<｜DSML｜parameter name="address" string="true">{"street": "Main St", "city": "New York", "additional_property": value}</｜DSML｜parameter>',
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</｜DSML｜parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>" ([ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_0_addl ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
root_prop_0_addl_key ::= ["] (("\"" | [^cs\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "c" ("\"" | [^i\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "i" ("\"" | [^t\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "t" ("\"" | [^y\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "y" ([^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub)))) | "s" ("\"" | [^t\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "t" ("\"" | [^r\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "r" ("\"" | [^e\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "e" ("\"" | [^e\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "e" ("\"" | [^t\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub | "t" ([^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub)))))))) (= [ \n\t]* [,}\]:])
root_prop_0_part_1 ::= ([ \n\t]* "," [ \n\t]* root_prop_0_addl_key [ \n\t]* ":" [ \n\t]* root_prop_0_addl)*
root_prop_0_part_0 ::= [ \n\t]* "," [ \n\t]* "\"city\"" [ \n\t]* ":" [ \n\t]* basic_string root_prop_0_part_1
root_prop_0 ::= "{" [ \n\t]* (("\"street\"" [ \n\t]* ":" [ \n\t]* basic_string root_prop_0_part_0)) [ \n\t]* "}"
root_addl ::= xml_string | basic_array | basic_object
root_part_0 ::= ([ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* root_addl [ \n\t]* "</｜DSML｜parameter>")*
root ::=  [ \n\t]* (("<｜DSML｜parameter name=\"address\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_0 [ \n\t]* "</｜DSML｜parameter>" root_part_0)) [ \n\t]*
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
    ('<｜DSML｜parameter name="age" string="false">25</｜DSML｜parameter>', False),
    (
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="false">25</｜DSML｜parameter>',
        True,
    ),
    (
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter><｜DSML｜parameter name="ID" string="false">123456</｜DSML｜parameter><｜DSML｜parameter name="is_student" string="true">true</｜DSML｜parameter>',
        True,
    ),
    (
        '<｜DSML｜parameter name="name" string="true">John</｜DSML｜parameter><｜DSML｜parameter name="age" string="false">1</｜DSML｜parameter><｜DSML｜parameter name="ID" string="false">1</｜DSML｜parameter><｜DSML｜parameter name="is_student" string="false">false</｜DSML｜parameter>',
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</｜DSML｜parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>" ([ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_2 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_3 ::= "true" | "false"
root_part_2_1 ::= [ \n\t]* "<｜DSML｜parameter name=\"is_student\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_3 [ \n\t]* "</｜DSML｜parameter>" ""
root_part_2_2 ::= "" | [ \n\t]* "<｜DSML｜parameter name=\"is_student\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_3 [ \n\t]* "</｜DSML｜parameter>" ""
root_part_2_3 ::= ""
root_part_1_1 ::= root_part_2_1 | [ \n\t]* "<｜DSML｜parameter name=\"ID\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_2 [ \n\t]* "</｜DSML｜parameter>" root_part_2_2
root_part_1_2 ::= root_part_2_2 | [ \n\t]* "<｜DSML｜parameter name=\"ID\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_2 [ \n\t]* "</｜DSML｜parameter>" root_part_2_3
root_part_0_1 ::= root_part_1_1 | [ \n\t]* "<｜DSML｜parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</｜DSML｜parameter>" root_part_1_2
root ::=  [ \n\t]* (("<｜DSML｜parameter name=\"name\" string=\"" ("true" | "false") "\">" xml_string "</｜DSML｜parameter>" root_part_0_1) | ("<｜DSML｜parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</｜DSML｜parameter>" root_part_1_1) | ("<｜DSML｜parameter name=\"ID\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_2 [ \n\t]* "</｜DSML｜parameter>" root_part_2_1)) [ \n\t]*
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


# DeepSeek: reject Qwen format <parameter=key>, Minimax format <parameter name="key"> (no string=), accept <｜DSML｜parameter name="key" string="true|false">
deepseek_reject_wrong_parameter_format_input_str_accepted = (
    ("<parameter=name>Bob</parameter><parameter=age>100</parameter>", False),  # Qwen format
    (
        '<parameter name="name">Bob</parameter><parameter name="age">100</parameter>',
        False,
    ),  # Minimax format (no string=)
    (
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="false">100</｜DSML｜parameter>',
        True,
    ),  # correct
)


@pytest.mark.parametrize(
    "input_str, accepted", deepseek_reject_wrong_parameter_format_input_str_accepted
)
def test_deepseek_reject_wrong_parameter_format(input_str: str, accepted: bool):
    """DeepSeek grammar must accept <｜DSML｜parameter name=\"key\" string=\"true|false\">, reject Qwen and Minimax formats."""
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
xml_string ::= TagDispatch(loop_after_dispatch=false,excludes=("</｜DSML｜parameter>"))
xml_any ::= xml_string | basic_array | basic_object
xml_object ::= ( [ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>" ([ \n\t]* "<｜DSML｜parameter name=\"" xml_variable_name "\" string=\"" ("true" | "false") "\">" [ \n\t]* xml_any [ \n\t]* "</｜DSML｜parameter>")* [ \n\t]*) | [ \n\t]*
xml_variable_name ::= [a-zA-Z_][a-zA-Z0-9_]*
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_0 ::= [ \n\t]* "<｜DSML｜parameter name=\"age\" string=\"" ("true" | "false") "\">" [ \n\t]* root_prop_1 [ \n\t]* "</｜DSML｜parameter>" ""
root ::=  [ \n\t]* (("<｜DSML｜parameter name=\"name\" string=\"" ("true" | "false") "\">" xml_string "</｜DSML｜parameter>" root_part_0)) [ \n\t]*
"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    _check_deepseek_grammar(schema, expected_grammar, input_str, accepted)


# ---------- GLM XML tool calling (json_format="glm_xml") ----------
# Format: <arg_key>$PARAMETER_NAME</arg_key><arg_value>$PARAMETER_VALUE</arg_value>


glm_reject_wrong_parameter_format_input_str_accepted = (
    ("<parameter=name>Bob</parameter><parameter=age>100</parameter>", False),
    ('<parameter name="name">Bob</parameter><parameter name="age">100</parameter>', False),
    (
        '<｜DSML｜parameter name="name" string="true">Bob</｜DSML｜parameter><｜DSML｜parameter name="age" string="false">100</｜DSML｜parameter>',
        False,
    ),
    (
        "<arg_key>name</arg_key><arg_value>Bob</arg_value>"
        "<arg_key>age</arg_key><arg_value>100</arg_value>",
        True,
    ),
)


@pytest.mark.parametrize(
    "input_str, accepted", glm_reject_wrong_parameter_format_input_str_accepted
)
def test_glm_reject_wrong_parameter_format(input_str: str, accepted: bool):
    """GLM grammar must use arg_key/arg_value wrappers and reject other XML styles."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    ebnf_grammar = _json_schema_to_ebnf(schema, json_format="glm_xml")
    grammar_str = str(ebnf_grammar)
    assert "<arg_key>" in grammar_str
    assert "<arg_value>" in grammar_str

    _check_glm_grammar(schema, input_str, accepted)


def test_glm_unconstrained_string_whitespace_has_bounded_parser_states():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    grammar = _json_schema_to_ebnf(schema, json_format="glm_xml")
    matcher = _get_matcher_from_grammar(grammar)

    assert matcher.accept_string("<arg_key>value</arg_key><arg_value>")
    states_before = matcher._debug_print_internal_state().count("ParserState(")
    assert matcher.accept_string(" " * 1024)
    states_after = matcher._debug_print_internal_state().count("ParserState(")

    assert states_after <= states_before + 1
    assert matcher.accept_string("</arg_value>")
    assert matcher.is_terminated()


def test_nested_true_schema():
    schema = {"type": "object", "properties": {"name": True}, "required": ["name"]}
    ebnf_grammar = _json_schema_to_ebnf(schema, json_format="qwen_xml")
    assert _is_grammar_accept_string(ebnf_grammar, "<parameter=name>\nvalue\n</parameter>")
    assert _is_grammar_accept_string(ebnf_grammar, "<parameter=name>\n[1, 2, 3]\n</parameter>")
    assert _is_grammar_accept_string(
        ebnf_grammar, '<parameter=name>\n{"name": "Tom"}\n</parameter>'
    )
    assert not _is_grammar_accept_string(ebnf_grammar, "anything")


def test_true_schema():
    schema = "true"
    ebnf_grammar = _json_schema_to_ebnf(schema, json_format="qwen_xml")
    assert _is_grammar_accept_string(ebnf_grammar, "<parameter=name>\nvalue\n</parameter>")
    assert _is_grammar_accept_string(ebnf_grammar, "<parameter=abc>\n[1, 2, 3]\n</parameter>")
    assert _is_grammar_accept_string(
        ebnf_grammar, '<parameter=cdef>\n{"name": "Tom"}\n</parameter>'
    )
    assert not _is_grammar_accept_string(ebnf_grammar, "anything")


# ---------- MiniMax M3 recursive XML (json_format="minimax_m3_xml") ----------

M3_NS = "]<]minimax[>["


def _m3_element(name: str, value: str) -> str:
    return f"{M3_NS}<{name}>{value}{M3_NS}</{name}>"


M3_RECURSIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
        "active": {"type": "boolean"},
        "details": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "scores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 1,
                    "maxItems": 2,
                },
            },
            "required": ["city", "scores"],
            "additionalProperties": False,
        },
        "stops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 2,
        },
        "nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["name", "count", "active", "details", "stops", "nullable"],
    "additionalProperties": False,
}

M3_DETAILS = _m3_element(
    "details",
    _m3_element("city", "Hangzhou")
    + _m3_element("scores", _m3_element("item", "1.5") + _m3_element("item", "2")),
)
M3_STOPS = _m3_element(
    "stops",
    _m3_element("item", _m3_element("label", "West Lake"))
    + _m3_element("item", _m3_element("label", "Lingyin")),
)
M3_VALID_RECURSIVE = (
    _m3_element("name", "Alice")
    + _m3_element("count", "2")
    + _m3_element("active", "true")
    + M3_DETAILS
    + M3_STOPS
    + _m3_element("nullable", "null")
)


@pytest.mark.parametrize(
    "instance, accepted",
    [
        (M3_VALID_RECURSIVE, True),
        (M3_VALID_RECURSIVE.replace(f"{M3_NS}<count>", f"\n{M3_NS}<count>"), True),
        (M3_VALID_RECURSIVE.replace(f"{M3_NS}</name>", f"{M3_NS}</wrong>", 1), False),
        (M3_VALID_RECURSIVE.replace(f"{M3_NS}<count>", f",{M3_NS}<count>", 1), False),
        (
            M3_VALID_RECURSIVE.replace(M3_DETAILS, _m3_element("details", '{"city":"Hangzhou"}')),
            False,
        ),
        (
            M3_VALID_RECURSIVE.replace(
                M3_STOPS, _m3_element("stops", _m3_element("item", '{"label":"West Lake"}'))
            ),
            False,
        ),
        (
            M3_VALID_RECURSIVE.replace(
                M3_STOPS,
                _m3_element(
                    "stops",
                    _m3_element("item", _m3_element("label", "A"))
                    + _m3_element("item", _m3_element("label", "B"))
                    + _m3_element("item", _m3_element("label", "C")),
                ),
            ),
            False,
        ),
        (M3_VALID_RECURSIVE.replace(_m3_element("active", "true"), ""), False),
        (
            M3_VALID_RECURSIVE.replace(_m3_element("active", "true"), _m3_element("active", "yes")),
            False,
        ),
        (
            M3_VALID_RECURSIVE.replace(
                _m3_element("name", "Alice"),
                _m3_element("name", f"A{M3_NS}<unexpected>x{M3_NS}</unexpected>"),
            ),
            False,
        ),
        (
            M3_VALID_RECURSIVE.replace(
                _m3_element("count", "2"), _m3_element("unknown", "2") + _m3_element("count", "2")
            ),
            False,
        ),
    ],
)
def test_minimax_m3_recursive_object_and_array(instance: str, accepted: bool):
    grammar = _json_schema_to_ebnf(M3_RECURSIVE_SCHEMA, json_format="minimax_m3_xml")
    assert _is_grammar_accept_string(grammar, instance) == accepted


def test_minimax_m3_empty_object_and_array_are_empty_element_bodies():
    schema = {
        "type": "object",
        "properties": {
            "empty_object": {"type": "object", "properties": {}, "additionalProperties": False},
            "empty_array": {"type": "array", "items": False, "maxItems": 0},
        },
        "required": ["empty_object", "empty_array"],
        "additionalProperties": False,
    }
    grammar = _json_schema_to_ebnf(schema, json_format="minimax_m3_xml")
    assert _is_grammar_accept_string(
        grammar, _m3_element("empty_object", "") + _m3_element("empty_array", "")
    )
    assert not _is_grammar_accept_string(
        grammar, _m3_element("empty_object", " ") + _m3_element("empty_array", "")
    )
    assert not _is_grammar_accept_string(
        grammar, _m3_element("empty_object", "") + _m3_element("empty_array", "\n")
    )


@pytest.mark.parametrize(
    "schema, accepted, rejected",
    [
        (
            {
                "type": "array",
                "prefixItems": [{"type": "string"}, {"type": "integer"}],
                "items": False,
            },
            [_m3_element("item", "alpha") + _m3_element("item", "2")],
            [
                "",
                _m3_element("item", "alpha"),
                _m3_element("item", "alpha") + _m3_element("item", "not-an-integer"),
                _m3_element("item", "alpha")
                + _m3_element("item", "2")
                + _m3_element("item", "extra"),
            ],
        ),
        (
            {
                "type": "array",
                "prefixItems": [{"type": "string"}],
                "items": {"type": "boolean"},
                "minItems": 2,
                "maxItems": 3,
            },
            [
                _m3_element("item", "alpha") + _m3_element("item", "true"),
                _m3_element("item", "alpha")
                + _m3_element("item", "true")
                + _m3_element("item", "false"),
            ],
            [
                _m3_element("item", "alpha"),
                _m3_element("item", "alpha") + _m3_element("item", "1"),
                _m3_element("item", "alpha")
                + _m3_element("item", "true")
                + _m3_element("item", "false")
                + _m3_element("item", "true"),
            ],
        ),
        (
            {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 2},
            [
                "",
                _m3_element("item", "alpha"),
                _m3_element("item", "alpha") + _m3_element("item", "beta"),
            ],
            [
                _m3_element("item", _m3_element("nested", "value")),
                _m3_element("item", "alpha")
                + _m3_element("item", "beta")
                + _m3_element("item", "gamma"),
            ],
        ),
    ],
)
def test_minimax_m3_repeated_item_array_semantics(schema, accepted, rejected):
    """M3 preserves XGrammar's existing fixed-prefix ``prefixItems`` semantics."""

    grammar = _json_schema_to_ebnf(schema, json_format="minimax_m3_xml")
    for instance in accepted:
        assert _is_grammar_accept_string(grammar, instance)
    for instance in rejected:
        assert not _is_grammar_accept_string(grammar, instance)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "plain text",
        "42",
        "true",
        "null",
        _m3_element("nested", "value"),
        _m3_element("first", "1") + _m3_element("second", "2"),
    ],
)
def test_minimax_m3_any_uses_unambiguous_scalar_or_nested_elements(value: str):
    grammar = _json_schema_to_ebnf(
        {"type": "object", "additionalProperties": True}, json_format="minimax_m3_xml"
    )
    assert _is_grammar_accept_string(grammar, _m3_element("runtime", value))


def test_minimax_m3_max_whitespace_cnt_applies_between_elements():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    grammar = _json_schema_to_ebnf(schema, json_format="minimax_m3_xml", max_whitespace_cnt=1)
    first = _m3_element("a", "x")
    second = _m3_element("b", "y")
    assert _is_grammar_accept_string(grammar, first + "\n" + second)
    assert not _is_grammar_accept_string(grammar, first + "\n\n" + second)


def test_minimax_m3_const_enum_and_null_literals():
    schema = {
        "type": "object",
        "properties": {
            "fixed": {"const": {"x": [1, True]}},
            "choice": {"enum": ["a", "b"]},
            "as_string": {"type": "string"},
            "as_null": {"type": "null"},
        },
        "required": ["fixed", "choice", "as_string", "as_null"],
        "additionalProperties": False,
    }
    grammar = _json_schema_to_ebnf(schema, json_format="minimax_m3_xml")
    fixed = _m3_element(
        "fixed", _m3_element("x", _m3_element("item", "1") + _m3_element("item", "true"))
    )
    assert _is_grammar_accept_string(
        grammar,
        fixed
        + _m3_element("choice", "b")
        + _m3_element("as_string", "null")
        + _m3_element("as_null", "null"),
    )
    assert not _is_grammar_accept_string(
        grammar,
        _m3_element("fixed", '{"x":[1,true]}')
        + _m3_element("choice", "b")
        + _m3_element("as_string", "null")
        + _m3_element("as_null", "null"),
    )


def test_minimax_m3_ref_and_any_order_apply_recursively():
    schema = {
        "$defs": {
            "point": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            }
        },
        "type": "object",
        "properties": {"name": {"type": "string"}, "point": {"$ref": "#/$defs/point"}},
        "required": ["name", "point"],
        "additionalProperties": False,
    }
    reordered = _m3_element("point", _m3_element("y", "2") + _m3_element("x", "1")) + _m3_element(
        "name", "p"
    )
    ordered_grammar = _json_schema_to_ebnf(schema, json_format="minimax_m3_xml")
    any_order_grammar = _json_schema_to_ebnf(schema, json_format="minimax_m3_xml", any_order=True)
    assert not _is_grammar_accept_string(ordered_grammar, reordered)
    assert _is_grammar_accept_string(any_order_grammar, reordered)


@pytest.mark.parametrize("any_order", [False, True])
def test_minimax_m3_optional_property_has_valid_empty_alternative(any_order: bool):
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "additionalProperties": False,
    }
    grammar = _json_schema_to_ebnf(schema, json_format="minimax_m3_xml", any_order=any_order)
    assert _is_grammar_accept_string(grammar, "")
    assert _is_grammar_accept_string(grammar, _m3_element("q", "value"))
    assert not _is_grammar_accept_string(grammar, _m3_element("other", "value"))


@pytest.mark.parametrize(
    "string_schema",
    [
        {"type": "string", "pattern": ".*"},
        {"type": "string", "format": "email"},
        {"type": "string", "minLength": 1},
        {"type": "string", "maxLength": 10},
    ],
)
def test_minimax_m3_rejects_string_constraints_that_bypass_namespace_exclusion(string_schema: dict):
    schema = {
        "type": "object",
        "properties": {"q": string_schema},
        "required": ["q"],
        "additionalProperties": False,
    }
    with pytest.raises(RuntimeError, match="namespace-marker exclusion"):
        _json_schema_to_ebnf(schema, json_format="minimax_m3_xml")


def test_minimax_m3_unknown_string_format_falls_back_to_namespace_safe_text():
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string", "format": "vendor-custom"}},
        "required": ["q"],
        "additionalProperties": False,
    }
    grammar = _json_schema_to_ebnf(schema, json_format="minimax_m3_xml")
    assert _is_grammar_accept_string(grammar, _m3_element("q", "plain text"))
    assert not _is_grammar_accept_string(
        grammar, _m3_element("q", f"text{M3_NS}<nested>x{M3_NS}</nested>")
    )


@pytest.mark.parametrize(
    "schema, instance",
    [
        ({"type": "object"}, _m3_element("runtime", _m3_element("nested", "value"))),
        (
            {"type": "array"},
            _m3_element("item", "value") + _m3_element("item", _m3_element("nested", "value")),
        ),
    ],
)
def test_minimax_m3_strict_false_supports_dynamic_containers(schema: dict, instance: str):
    grammar = _json_schema_to_ebnf(schema, json_format="minimax_m3_xml", strict_mode=False)
    assert _is_grammar_accept_string(grammar, instance)


def test_minimax_m3_element_names_are_escaped_as_ebnf_literals():
    key = 'line\n"quoted"\\key'
    schema = {
        "type": "object",
        "properties": {key: {"type": "string"}},
        "required": [key],
        "additionalProperties": False,
    }
    grammar = _json_schema_to_ebnf(schema, json_format="minimax_m3_xml")
    assert _is_grammar_accept_string(grammar, _m3_element(key, "value"))


@pytest.mark.parametrize("key", ["", "/closing", "has>delimiter", " \t\n"])
def test_minimax_m3_rejects_unparseable_element_names(key: str):
    schema = {
        "type": "object",
        "properties": {key: {"type": "string"}},
        "required": [key],
        "additionalProperties": False,
    }
    with pytest.raises(RuntimeError, match="element name|cannot be blank"):
        _json_schema_to_ebnf(schema, json_format="minimax_m3_xml")


ROOT_SELF_REF_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "child": {"anyOf": [{"type": "null"}, {"$ref": "#"}]},
    },
    "required": ["name", "child"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "json_format, nested_json, nested_xml",
    [
        (
            "qwen_xml",
            '<parameter=name>root</parameter><parameter=child>{"name": "leaf", "child": null}</parameter>',
            "<parameter=name>root</parameter><parameter=child>"
            "<parameter=name>leaf</parameter><parameter=child>null</parameter></parameter>",
        ),
        (
            "minimax_xml",
            '<parameter name="name">root</parameter><parameter name="child">'
            '{"name": "leaf", "child": null}</parameter>',
            '<parameter name="name">root</parameter><parameter name="child">'
            '<parameter name="name">leaf</parameter><parameter name="child">null</parameter>'
            "</parameter>",
        ),
        (
            "deepseek_xml",
            '<｜DSML｜parameter name="name" string="true">root</｜DSML｜parameter>'
            '<｜DSML｜parameter name="child" string="false">'
            '{"name": "leaf", "child": null}</｜DSML｜parameter>',
            '<｜DSML｜parameter name="name" string="true">root</｜DSML｜parameter>'
            '<｜DSML｜parameter name="child" string="false">'
            '<｜DSML｜parameter name="name" string="true">leaf</｜DSML｜parameter>'
            '<｜DSML｜parameter name="child" string="false">null</｜DSML｜parameter>'
            "</｜DSML｜parameter>",
        ),
        (
            "glm_xml",
            "<arg_key>name</arg_key><arg_value>root</arg_value>"
            '<arg_key>child</arg_key><arg_value>{"name": "leaf", "child": null}</arg_value>',
            "<arg_key>name</arg_key><arg_value>root</arg_value>"
            "<arg_key>child</arg_key><arg_value>"
            "<arg_key>name</arg_key><arg_value>leaf</arg_value>"
            "<arg_key>child</arg_key><arg_value>null</arg_value></arg_value>",
        ),
    ],
)
def test_root_only_xml_self_ref_uses_nested_json_domain(
    json_format: str, nested_json: str, nested_xml: str
):
    grammar = _json_schema_to_ebnf(ROOT_SELF_REF_SCHEMA, json_format=json_format)
    assert _is_grammar_accept_string(grammar, nested_json)
    assert not _is_grammar_accept_string(grammar, nested_xml)


def test_minimax_m3_self_ref_stays_recursive_xml():
    grammar = _json_schema_to_ebnf(ROOT_SELF_REF_SCHEMA, json_format="minimax_m3_xml")
    recursive_xml = _m3_element("name", "root") + _m3_element(
        "child", _m3_element("name", "leaf") + _m3_element("child", "null")
    )
    nested_json = _m3_element("name", "root") + _m3_element(
        "child", '{"name": "leaf", "child": null}'
    )
    assert _is_grammar_accept_string(grammar, recursive_xml)
    assert not _is_grammar_accept_string(grammar, nested_json)


@pytest.mark.parametrize(
    "schema, accepted, rejected",
    [
        ({}, _m3_element("runtime", _m3_element("nested", "value")), None),
        (
            {"type": "object", "additionalProperties": {"type": "string"}},
            _m3_element("runtime key/城市", "value"),
            None,
        ),
        (
            {
                "type": "object",
                "patternProperties": {"^x.*": {"type": "string"}},
                "additionalProperties": False,
            },
            _m3_element("x_runtime", "value"),
            _m3_element("y_runtime", "value"),
        ),
        (
            {
                "type": "object",
                "propertyNames": {"pattern": "^[a-z]+$"},
                "additionalProperties": {"type": "string"},
            },
            _m3_element("runtime", "value"),
            _m3_element("runtime_1", "value"),
        ),
    ],
)
def test_minimax_m3_dynamic_property_schemas(schema: dict, accepted: str, rejected):
    grammar = _json_schema_to_ebnf(schema, json_format="minimax_m3_xml")
    assert _is_grammar_accept_string(grammar, accepted)
    if rejected is not None:
        assert not _is_grammar_accept_string(grammar, rejected)


if __name__ == "__main__":
    pytest.main(sys.argv)
