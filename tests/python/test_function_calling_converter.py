import sys

import pytest

from xgrammar.testing import _is_grammar_accept_string, _qwen_xml_tool_calling_to_ebnf

test_string_schema_input_str_accepted = (
    ("<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter>", True),
    ("<parameter=name>Bob</parameter>\t\n<parameter=age>\t100\n</parameter>", True),
    ("<parameter=name>Bob</parameter><parameter=age>100</parameter>", True),
    ("\n\t<parameter=name>Bob</parameter><parameter=age>100</parameter>", True),
    ('<parameter=name>"Bob&lt;"</parameter><parameter=age>100</parameter>', True),
    ("<parameter=name><>Bob</parameter><parameter=age>100</parameter>", False),
    ("<parameter=name>Bob</parameter><parameter=age>100</parameter>\t\t", False),
)


@pytest.mark.parametrize("input_str, accepted", test_string_schema_input_str_accepted)
def test_string_schema(input_str: str, accepted: bool):
    expected_grammar = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
xml_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
xml_entity ::=  "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
xml_string ::= ("" | [^<>&\0-\x1f\\\r\n] xml_string | "\\" xml_escape xml_string | xml_entity xml_string) (= [ \n\t]*)
xml_variable_name ::= [a-zA-Z_] [a-zA-Z0-9_]*
xml_string_0 ::= xml_string
xml_any ::= basic_number | xml_string | basic_boolean | basic_null | basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_0 ::= [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=  [ \n\t]* (("<parameter=name>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>" root_part_0))"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    ebnf_grammar = _qwen_xml_tool_calling_to_ebnf(schema)
    assert str(ebnf_grammar[:-2]) == expected_grammar
    assert _is_grammar_accept_string(ebnf_grammar, input_str) == accepted


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
xml_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
xml_entity ::=  "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
xml_string ::= ("" | [^<>&\0-\x1f\\\r\n] xml_string | "\\" xml_escape xml_string | xml_entity xml_string) (= [ \n\t]*)
xml_variable_name ::= [a-zA-Z_] [a-zA-Z0-9_]*
xml_string_0 ::= xml_string
xml_any ::= basic_number | xml_string | basic_boolean | basic_null | basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_1 ::= ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")*
root_part_0 ::= [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::=  [ \n\t]* (("<parameter=name>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>" root_part_0))"""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": True,
    }
    ebnf_grammar = _qwen_xml_tool_calling_to_ebnf(schema)
    assert str(ebnf_grammar[:-2]) == expected_grammar
    assert _is_grammar_accept_string(ebnf_grammar, input_str) == accepted


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
xml_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
xml_entity ::=  "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
xml_string ::= ("" | [^<>&\0-\x1f\\\r\n] xml_string | "\\" xml_escape xml_string | xml_entity xml_string) (= [ \n\t]*)
xml_variable_name ::= [a-zA-Z_] [a-zA-Z0-9_]*
xml_string_0 ::= xml_string
xml_any ::= basic_number | xml_string | basic_boolean | basic_null | basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_1 ::= ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")*
root_part_0 ::= root_part_1 | [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::= "" |  [ \n\t]* (("<parameter=name>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>" root_part_0) | ("<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1) | "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" root_part_1)"""

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "additionalProperties": True,
    }
    ebnf_grammar = _qwen_xml_tool_calling_to_ebnf(schema)
    assert str(ebnf_grammar[:-2]) == expected_grammar
    assert _is_grammar_accept_string(ebnf_grammar, input_str) == accepted


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
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
        "additionalProperties": True,
    }
    ebnf_grammar = _qwen_xml_tool_calling_to_ebnf(schema)
    assert _is_grammar_accept_string(ebnf_grammar, input_str) == accepted


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
xml_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
xml_entity ::=  "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
xml_string ::= ("" | [^<>&\0-\x1f\\\r\n] xml_string | "\\" xml_escape xml_string | xml_entity xml_string) (= [ \n\t]*)
xml_variable_name ::= [a-zA-Z_] [a-zA-Z0-9_]*
xml_string_0 ::= xml_string
xml_any ::= basic_number | xml_string | basic_boolean | basic_null | basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_prop_0_part_0 ::= [ \n\t]* "," [ \n\t]* "\"city\"" [ \n\t]* ":" [ \n\t]* basic_string ""
root_prop_0 ::= "{" [ \n\t]* (("\"street\"" [ \n\t]* ":" [ \n\t]* basic_string root_prop_0_part_0)) [ \n\t]* "}"
root ::=  [ \n\t]* (("<parameter=address>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" ""))"""

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
    ebnf_grammar = _qwen_xml_tool_calling_to_ebnf(schema)
    assert str(ebnf_grammar[:-2]) == expected_grammar
    assert _is_grammar_accept_string(ebnf_grammar, input_str) == accepted


test_numbers_schema_input_str_accepted = (
    ("<parameter=age>25</parameter>", False),
    ("<parameter=name>Bob</parameter>\n<parameter=age>25</parameter>", True),
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
xml_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
xml_entity ::=  "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
xml_string ::= ("" | [^<>&\0-\x1f\\\r\n] xml_string | "\\" xml_escape xml_string | xml_entity xml_string) (= [ \n\t]*)
xml_variable_name ::= [a-zA-Z_] [a-zA-Z0-9_]*
xml_string_0 ::= xml_string
xml_any ::= basic_number | xml_string | basic_boolean | basic_null | basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_2 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_3 ::= "true" | "false"
root_part_2_1 ::= [ \n\t]* "<parameter=is_student>" [ \n\t]* root_prop_3 [ \n\t]* "</parameter>" ""
root_part_2_2 ::= "" | [ \n\t]* "<parameter=is_student>" [ \n\t]* root_prop_3 [ \n\t]* "</parameter>" ""
root_part_2_3 ::= ""
root_part_1_1 ::= root_part_2_1 | [ \n\t]* "<parameter=ID>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_2
root_part_1_2 ::= root_part_2_2 | [ \n\t]* "<parameter=ID>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_3
root_part_0_1 ::= root_part_1_1 | [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1_2
root ::=  [ \n\t]* (("<parameter=name>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>" root_part_0_1) | ("<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1_1) | ("<parameter=ID>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_1))"""
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

    ebnf_grammar = _qwen_xml_tool_calling_to_ebnf(schema)
    assert str(ebnf_grammar[:-2]) == expected_grammar
    assert _is_grammar_accept_string(ebnf_grammar, input_str) == accepted


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
xml_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
xml_entity ::=  "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
xml_string ::= ("" | [^<>&\0-\x1f\\\r\n] xml_string | "\\" xml_escape xml_string | xml_entity xml_string) (= [ \n\t]*)
xml_variable_name ::= [a-zA-Z_] [a-zA-Z0-9_]*
xml_string_0 ::= xml_string
xml_any ::= basic_number | xml_string | basic_boolean | basic_null | basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
string ::= [^<>&\r\n]{1,}
root_prop_0 ::= string
string_0 ::= "\"" [0-9]{5} "\""
root_prop_1_prop_0 ::= string_0
string_1 ::= "\"" ( ( [a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+ ( "." [a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+ )* ) | "\\" "\"" ( "\\" [ -~] | [ !#-[\]-~] )* "\\" "\"" ) "@" ( [A-Za-z0-9] ( [\-A-Za-z0-9]* [A-Za-z0-9] )? ) ( ( "." [A-Za-z0-9] [\-A-Za-z0-9]* [A-Za-z0-9] )* ) "\""
root_prop_1_prop_1 ::= string_1
root_prop_1_part_0 ::= [ \n\t]* "," [ \n\t]* "\"email\"" [ \n\t]* ":" [ \n\t]* root_prop_1_prop_1 ""
root_prop_1 ::= "{" [ \n\t]* (("\"phone\"" [ \n\t]* ":" [ \n\t]* root_prop_1_prop_0 root_prop_1_part_0)) [ \n\t]* "}"
root_part_0 ::= [ \n\t]* "<parameter=contact_info>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=  [ \n\t]* (("<parameter=name>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" root_part_0))"""
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

    ebnf_grammar = _qwen_xml_tool_calling_to_ebnf(schema)
    assert str(ebnf_grammar[:-2]) == expected_grammar
    assert _is_grammar_accept_string(ebnf_grammar, input_str) == accepted


if __name__ == "__main__":
    pytest.main(sys.argv)
