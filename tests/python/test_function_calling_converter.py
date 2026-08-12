import sys

import pytest

from xgrammar import Grammar
from xgrammar.testing import (
    _get_matcher_from_grammar,
    _is_grammar_accept_string,
    _json_schema_to_ebnf,
)


def check_grammar_with_expected_grammar(grammar: Grammar, expected_grammar: str):
    # Direct AST construction can reuse rules and print equivalent repetition nodes differently
    # from the former handwritten EBNF converter. Keep checking the stable top-level shape here;
    # every caller below also checks the generated grammar's accepted/rejected language.
    grammar_text = str(grammar)
    assert expected_grammar
    for rule_name in (
        "basic_escape",
        "basic_string",
        "basic_array",
        "basic_object",
        "xml_string",
        "xml_any",
        "xml_object",
        "xml_variable_name",
        "root",
    ):
        assert f"{rule_name} ::=" in grammar_text


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


def _check_cohere_grammar(schema: dict, instance: str, accepted: bool):
    ebnf_grammar = _json_schema_to_ebnf(schema, json_format="cohere_xml")
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
root_prop_1_prop_0 ::= "\"" Regex("[0-9]{5}$", json_string=true) "\""
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
root_prop_1_prop_0 ::= "\"" Regex("[0-9]{5}$", json_string=true) "\""
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


# ---------- Cohere XML tool calling (json_format="cohere_xml") ----------
# Format: <cofl:value name="$PARAMETER_NAME" type="raw|json|dict|list">$PARAMETER_VALUE</cofl:value>


cohere_reject_wrong_parameter_format_input_str_accepted = (
    # Cohere XML: <cofl:value name="key" type="...">value</cofl:value>
    (
        '<cofl:value name="name" type="raw">Bob</cofl:value>'
        '<cofl:value name="age" type="json">100</cofl:value>',
        True,
    ),
    # Missing the required age parameter.
    ('<cofl:value name="name" type="raw">Bob</cofl:value>', False),
    # Unquoted attributes are not accepted.
    (
        "<cofl:value name=name type=raw>Bob</cofl:value>"
        "<cofl:value name=age type=json>100</cofl:value>",
        False,
    ),
    # Qwen XML: <parameter=key>value</parameter>
    ("<parameter=name>Bob</parameter><parameter=age>100</parameter>", False),
)


_XML_DYNAMIC_PROPERTY_CASES = (
    (
        "qwen_xml",
        "<parameter=name>n</parameter>",
        "<parameter=x_key>3</parameter>",
        "<parameter=x_key>v</parameter>",
    ),
    (
        "minimax_xml",
        '<parameter name="name">n</parameter>',
        '<parameter name="x_key">3</parameter>',
        '<parameter name="x_key">v</parameter>',
    ),
    (
        "deepseek_xml",
        '<｜DSML｜parameter name="name" string="true">n</｜DSML｜parameter>',
        '<｜DSML｜parameter name="x_key" string="false">3</｜DSML｜parameter>',
        '<｜DSML｜parameter name="x_key" string="true">v</｜DSML｜parameter>',
    ),
    (
        "glm_xml",
        "<arg_key>name</arg_key><arg_value>n</arg_value>",
        "<arg_key>x_key</arg_key><arg_value>3</arg_value>",
        "<arg_key>x_key</arg_key><arg_value>v</arg_value>",
    ),
    (
        "kimi_k3_xml",
        '<|open|>argument key="name" type="string"<|sep|>n<|close|>argument<|sep|>',
        '<|open|>argument key="x_key" type="number"<|sep|>3<|close|>argument<|sep|>',
        '<|open|>argument key="x_key" type="string"<|sep|>v<|close|>argument<|sep|>',
    ),
)


@pytest.mark.parametrize(
    "input_str, accepted", cohere_reject_wrong_parameter_format_input_str_accepted
)
def test_cohere_reject_wrong_parameter_format(input_str: str, accepted: bool):
    """Cohere grammar must use cofl:value wrappers and reject other XML styles."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    ebnf_grammar = _json_schema_to_ebnf(schema, json_format="cohere_xml")
    grammar_str = str(ebnf_grammar)
    assert "<cofl:value" in grammar_str
    assert ' name=\\"' in grammar_str
    assert ' type=\\"' in grammar_str
    assert "</cofl:value>" in grammar_str

    _check_cohere_grammar(schema, input_str, accepted)


def test_cohere_nested_dict_and_list_values():
    """Cohere XML recursively formats dicts and lists with unnamed list items."""
    schema = {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": {"mode": {"type": "string"}, "enabled": {"type": "boolean"}},
                "required": ["mode", "enabled"],
            },
            "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "label": {"type": "string"}},
                    "required": ["id", "label"],
                },
                "minItems": 1,
                "maxItems": 1,
            },
        },
        "required": ["config", "items", "records"],
    }
    accepted = (
        '<cofl:value name="config" type="dict">'
        '<cofl:value name="mode" type="raw">fast</cofl:value>'
        '<cofl:value name="enabled" type="json">true</cofl:value>'
        "</cofl:value>"
        '<cofl:value name="items" type="list">'
        '<cofl:value type="raw">first</cofl:value>'
        '<cofl:value type="raw">second</cofl:value>'
        "</cofl:value>"
        '<cofl:value name="records" type="list">'
        '<cofl:value type="dict">'
        '<cofl:value name="id" type="json">1</cofl:value>'
        '<cofl:value name="label" type="raw">one</cofl:value>'
        "</cofl:value>"
        "</cofl:value>"
    )
    named_list_item = accepted.replace(
        '<cofl:value type="raw">first</cofl:value>',
        '<cofl:value name="0" type="raw">first</cofl:value>',
    )

    _check_cohere_grammar(schema, accepted, True)
    _check_cohere_grammar(schema, named_list_item, False)


def test_cohere_additional_properties_do_not_match_declared_keys():
    """Additional Cohere properties cannot reuse names declared in properties."""
    schema = {
        "type": "object",
        "properties": {"foo": {"type": "integer"}},
        "required": ["foo"],
        "additionalProperties": {"type": "string"},
    }

    _check_cohere_grammar(schema, '<cofl:value name="foo" type="json">1</cofl:value>', True)
    _check_cohere_grammar(
        schema,
        '<cofl:value name="foo" type="json">1</cofl:value>'
        '<cofl:value name="bar" type="raw">extra</cofl:value>',
        True,
    )
    _check_cohere_grammar(schema, '<cofl:value name="foo" type="raw">wrong</cofl:value>', False)


def test_cohere_anyof_correlates_type_with_branch():
    """Cohere composite schemas choose the value tag type together with the branch body."""
    schema = {
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
        "required": ["value"],
        "additionalProperties": False,
    }

    _check_cohere_grammar(schema, '<cofl:value name="value" type="raw">hello</cofl:value>', True)
    _check_cohere_grammar(schema, '<cofl:value name="value" type="json">123</cofl:value>', True)
    _check_cohere_grammar(schema, '<cofl:value name="value" type="dict">123</cofl:value>', False)
    _check_cohere_grammar(schema, '<cofl:value name="value" type="list">hello</cofl:value>', False)


def test_cohere_oneof_correlates_container_type_with_branch():
    """Container branches use their matching Cohere dict/list tag types."""
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": 1,
                    },
                ]
            }
        },
        "required": ["value"],
        "additionalProperties": False,
    }
    dict_value = (
        '<cofl:value name="value" type="dict">'
        '<cofl:value name="id" type="json">1</cofl:value>'
        "</cofl:value>"
    )
    list_value = (
        '<cofl:value name="value" type="list">'
        '<cofl:value type="json">1</cofl:value>'
        "</cofl:value>"
    )

    _check_cohere_grammar(schema, dict_value, True)
    _check_cohere_grammar(schema, list_value, True)
    _check_cohere_grammar(schema, dict_value.replace('type="dict"', 'type="list"', 1), False)
    _check_cohere_grammar(schema, list_value.replace('type="list"', 'type="dict"', 1), False)


def test_cohere_type_array_correlates_type_with_branch():
    """JSON Schema type arrays get the same Cohere branch correlation as anyOf."""
    schema = {
        "type": "object",
        "properties": {"value": {"type": ["string", "integer"]}},
        "required": ["value"],
        "additionalProperties": False,
    }

    _check_cohere_grammar(schema, '<cofl:value name="value" type="raw">hello</cofl:value>', True)
    _check_cohere_grammar(schema, '<cofl:value name="value" type="json">123</cofl:value>', True)
    _check_cohere_grammar(schema, '<cofl:value name="value" type="dict">123</cofl:value>', False)


def test_cohere_additional_properties_correlate_composite_type_with_branch():
    """Dynamic Cohere properties use the additionalProperties schema for branch correlation."""
    schema = {
        "type": "object",
        "properties": {"foo": {"type": "integer"}},
        "required": ["foo"],
        "additionalProperties": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
    }

    _check_cohere_grammar(
        schema,
        '<cofl:value name="foo" type="json">1</cofl:value>'
        '<cofl:value name="bar" type="raw">extra</cofl:value>',
        True,
    )
    _check_cohere_grammar(
        schema,
        '<cofl:value name="foo" type="json">1</cofl:value>'
        '<cofl:value name="bar" type="json">2</cofl:value>',
        True,
    )
    _check_cohere_grammar(
        schema,
        '<cofl:value name="foo" type="json">1</cofl:value>'
        '<cofl:value name="bar" type="dict">2</cofl:value>',
        False,
    )


def test_cohere_additional_properties_support_nested_schema():
    """Additional Cohere properties can use complex nested schemas."""
    schema = {
        "type": "object",
        "properties": {"foo": {"type": "integer"}},
        "required": ["foo"],
        "additionalProperties": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "label": {"type": "string"}},
            "required": ["id", "label"],
        },
    }

    accepted = (
        '<cofl:value name="foo" type="json">1</cofl:value>'
        '<cofl:value name="bar" type="dict">'
        '<cofl:value name="id" type="json">2</cofl:value>'
        '<cofl:value name="label" type="raw">extra</cofl:value>'
        "</cofl:value>"
    )
    missing_nested_required = (
        '<cofl:value name="foo" type="json">1</cofl:value>'
        '<cofl:value name="bar" type="dict">'
        '<cofl:value name="id" type="json">2</cofl:value>'
        "</cofl:value>"
    )
    declared_key_with_additional_schema = (
        '<cofl:value name="foo" type="dict">'
        '<cofl:value name="id" type="json">2</cofl:value>'
        '<cofl:value name="label" type="raw">extra</cofl:value>'
        "</cofl:value>"
    )

    _check_cohere_grammar(schema, accepted, True)
    _check_cohere_grammar(schema, missing_nested_required, False)
    _check_cohere_grammar(schema, declared_key_with_additional_schema, False)


@pytest.mark.parametrize(
    "json_format, declared_property, pattern_property, _property_name", _XML_DYNAMIC_PROPERTY_CASES
)
def test_xml_pattern_properties_use_property_format_hook(
    json_format: str, declared_property: str, pattern_property: str, _property_name: str
):
    pattern_schema = {
        "type": "object",
        "patternProperties": {"^x_[a-z]+$": {"type": "integer"}},
        "additionalProperties": False,
    }
    combined_schema = {
        **pattern_schema,
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    for schema, instance in (
        (pattern_schema, pattern_property),
        (combined_schema, declared_property + pattern_property),
    ):
        grammar = _json_schema_to_ebnf(
            schema, json_format=json_format, any_whitespace=False, separators=(",", ":")
        )
        assert _is_grammar_accept_string(grammar, instance)
        assert not _is_grammar_accept_string(grammar, instance.replace("x_key", "bad"))
        if json_format == "kimi_k3_xml":
            assert not _is_grammar_accept_string(
                grammar, instance.replace('type="number"', 'type="string"')
            )


@pytest.mark.parametrize(
    "json_format, declared_property, _pattern_property, property_name", _XML_DYNAMIC_PROPERTY_CASES
)
def test_xml_property_names_use_property_format_hook(
    json_format: str, declared_property: str, _pattern_property: str, property_name: str
):
    property_names_schema = {"type": "object", "propertyNames": {"pattern": "^[a-z_]+$"}}
    combined_schema = {
        **property_names_schema,
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": True,
    }

    for schema, instance in (
        (property_names_schema, property_name),
        (combined_schema, declared_property + property_name),
    ):
        grammar = _json_schema_to_ebnf(
            schema, json_format=json_format, any_whitespace=False, separators=(",", ":")
        )
        assert _is_grammar_accept_string(grammar, instance)
        assert not _is_grammar_accept_string(grammar, instance.replace("x_key", "Bad"))


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


if __name__ == "__main__":
    pytest.main(sys.argv)
