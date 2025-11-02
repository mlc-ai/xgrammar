import json
import sys
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import pytest
from pydantic import BaseModel, Field, TypeAdapter, create_model

import xgrammar as xgr
from xgrammar.testing import (
    GrammarFunctor,
    _generate_float_regex,
    _generate_range_regex,
    _is_grammar_accept_string,
    _json_schema_to_ebnf,
)

basic_json_rules_ebnf = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
"""

basic_json_rules_ebnf_no_space = r"""basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" "" basic_any (", " basic_any)* "" "]") | ("[" "" "]"))
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{" "}"
"""


def check_schema_with_grammar(
    schema: Dict[str, Any],
    expected_grammar_ebnf: str,
    any_whitespace: bool = True,
    indent: Optional[int] = None,
    separators: Optional[Tuple[str, str]] = None,
    strict_mode: bool = True,
):
    json_schema_ebnf = _json_schema_to_ebnf(
        schema,
        any_whitespace=any_whitespace,
        indent=indent,
        separators=separators,
        strict_mode=strict_mode,
    )
    assert json_schema_ebnf == expected_grammar_ebnf


def check_schema_with_instance(
    schema: Dict[str, Any],
    instance: Union[str, BaseModel, Any],
    is_accepted: bool = True,
    any_whitespace: bool = True,
    indent: Optional[int] = None,
    separators: Optional[Tuple[str, str]] = None,
    strict_mode: bool = True,
    debug_print: bool = False,
):
    json_schema_grammar = xgr.Grammar.from_json_schema(
        json.dumps(schema),
        any_whitespace=any_whitespace,
        indent=indent,
        separators=separators,
        strict_mode=strict_mode,
    )

    # instance: pydantic model, json string, or any other object (dumped to json string)
    if isinstance(instance, BaseModel):
        instance = json.dumps(
            instance.model_dump(mode="json", round_trip=True), indent=indent, separators=separators
        )
    elif not isinstance(instance, str):
        instance = json.dumps(instance, indent=indent, separators=separators)

    accepted = _is_grammar_accept_string(json_schema_grammar, instance, debug_print=debug_print)
    assert accepted == is_accepted


def test_basic():
    class MainModel(BaseModel):
        integer_field: int
        number_field: float
        boolean_field: bool
        any_array_field: List
        array_field: List[str]
        tuple_field: Tuple[str, int, List[str]]
        object_field: Dict[str, int]
        nested_object_field: Dict[str, Dict[str, int]]

    ebnf_grammar = basic_json_rules_ebnf_no_space + (
        r"""root_prop_3 ::= (("[" "" basic_any (", " basic_any)* "" "]") | ("[" "" "]"))
root_prop_4 ::= (("[" "" basic_string (", " basic_string)* "" "]") | ("[" "" "]"))
root_prop_5_item_2 ::= (("[" "" basic_string (", " basic_string)* "" "]") | ("[" "" "]"))
root_prop_5 ::= ("[" "" (basic_string ", " basic_integer ", " root_prop_5_item_2) "" "]")
root_prop_6 ::= ("{" "" basic_string ": " basic_integer (", " basic_string ": " basic_integer)* "" "}") | "{" "}"
root_prop_7_addl ::= ("{" "" basic_string ": " basic_integer (", " basic_string ": " basic_integer)* "" "}") | "{" "}"
root_prop_7 ::= ("{" "" basic_string ": " root_prop_7_addl (", " basic_string ": " root_prop_7_addl)* "" "}") | "{" "}"
root_part_6 ::= ", " "\"nested_object_field\"" ": " root_prop_7 ""
root_part_5 ::= ", " "\"object_field\"" ": " root_prop_6 root_part_6
root_part_4 ::= ", " "\"tuple_field\"" ": " root_prop_5 root_part_5
root_part_3 ::= ", " "\"array_field\"" ": " root_prop_4 root_part_4
root_part_2 ::= ", " "\"any_array_field\"" ": " root_prop_3 root_part_3
root_part_1 ::= ", " "\"boolean_field\"" ": " basic_boolean root_part_2
root_part_0 ::= ", " "\"number_field\"" ": " basic_number root_part_1
root ::= "{" "" (("\"integer_field\"" ": " basic_integer root_part_0)) "" "}"
"""
    )

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=False)

    instance = MainModel(
        integer_field=42,
        number_field=3.14e5,
        boolean_field=True,
        any_array_field=[3.14, "foo", None, True],
        array_field=["foo", "bar"],
        tuple_field=("foo", 42, ["bar", "baz"]),
        object_field={"foo": 42, "bar": 43},
        nested_object_field={"foo": {"bar": 42}},
    )
    check_schema_with_instance(schema, instance, any_whitespace=False)


def test_indent():
    class MainModel(BaseModel):
        array_field: List[str]
        tuple_field: Tuple[str, int, List[str]]
        object_field: Dict[str, int]

    ebnf_grammar = basic_json_rules_ebnf_no_space + (
        r"""root_prop_0 ::= (("[" "\n    " basic_string (",\n    " basic_string)* "\n  " "]") | ("[" "" "]"))
root_prop_1_item_2 ::= (("[" "\n      " basic_string (",\n      " basic_string)* "\n    " "]") | ("[" "" "]"))
root_prop_1 ::= ("[" "\n    " (basic_string ",\n    " basic_integer ",\n    " root_prop_1_item_2) "\n  " "]")
root_prop_2 ::= ("{" "\n    " basic_string ": " basic_integer (",\n    " basic_string ": " basic_integer)* "\n  " "}") | "{" "}"
root_part_1 ::= ",\n  " "\"object_field\"" ": " root_prop_2 ""
root_part_0 ::= ",\n  " "\"tuple_field\"" ": " root_prop_1 root_part_1
root ::= "{" "\n  " (("\"array_field\"" ": " root_prop_0 root_part_0)) "\n" "}"
"""
    )

    instance = MainModel(
        array_field=["foo", "bar"],
        tuple_field=("foo", 42, ["bar", "baz"]),
        object_field={"foo": 42, "bar": 43},
    )

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=False, indent=2)
    check_schema_with_instance(schema, instance, any_whitespace=False, indent=2)
    check_schema_with_instance(
        schema, instance, any_whitespace=False, indent=None, separators=(",", ":")
    )


schema__grammar__accepted_instances__rejected_instances__test_non_strict = [
    (
        {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "integer"}]},
        basic_json_rules_ebnf
        + r"""root ::= ("[" [ \n\t]* (basic_integer [ \n\t]* "," [ \n\t]* basic_integer) ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]")
""",
        [[1, 2], [1, 2, 3], [1, 2, 3, "123"]],
        [[1]],
    ),
    (
        {
            "type": "object",
            "properties": {"foo": {"type": "integer"}, "bar": {"type": "integer"}},
            "required": ["foo", "bar"],
        },
        basic_json_rules_ebnf
        + r"""root_part_1 ::= ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)*
root_part_0 ::= [ \n\t]* "," [ \n\t]* "\"bar\"" [ \n\t]* ":" [ \n\t]* basic_integer root_part_1
root ::= "{" [ \n\t]* (("\"foo\"" [ \n\t]* ":" [ \n\t]* basic_integer root_part_0)) [ \n\t]* "}"
""",
        [{"foo": 1, "bar": 2}, {"foo": 1, "bar": 2, "baz": 3}],
        [{"foo": 1}],
    ),
]


@pytest.mark.parametrize(
    "schema, expected_grammar, accepted_instances, rejected_instances",
    schema__grammar__accepted_instances__rejected_instances__test_non_strict,
)
def test_non_strict(
    schema: Dict[str, Any],
    expected_grammar: str,
    accepted_instances: List[Any],
    rejected_instances: List[Any],
):
    check_schema_with_grammar(schema, expected_grammar, strict_mode=False)
    for instance in accepted_instances:
        check_schema_with_instance(schema, instance, is_accepted=True, strict_mode=False)
    for instance in rejected_instances:
        check_schema_with_instance(schema, instance, is_accepted=False, strict_mode=False)


def test_enum_const():
    class Field(Enum):
        FOO = "foo"
        BAR = "bar"

    class MainModel(BaseModel):
        bars: Literal["a"]
        str_values: Literal['a\n\r"']
        foo: Literal["a", "b", "c"]
        values: Literal[1, "a", True]
        field: Field

    ebnf_grammar = basic_json_rules_ebnf_no_space + (
        r"""root_prop_0 ::= "\"a\""
root_prop_1 ::= "\"a\\n\\r\\\"\""
root_prop_2 ::= ("\"a\"") | ("\"b\"") | ("\"c\"")
root_prop_3 ::= ("1") | ("\"a\"") | ("true")
defs_Field ::= ("\"foo\"") | ("\"bar\"")
root_prop_4 ::= defs_Field
root_part_3 ::= ", " "\"field\"" ": " root_prop_4 ""
root_part_2 ::= ", " "\"values\"" ": " root_prop_3 root_part_3
root_part_1 ::= ", " "\"foo\"" ": " root_prop_2 root_part_2
root_part_0 ::= ", " "\"str_values\"" ": " root_prop_1 root_part_1
root ::= "{" "" (("\"bars\"" ": " root_prop_0 root_part_0)) "" "}"
"""
    )

    schema = MainModel.model_json_schema()
    instance = MainModel(foo="a", values=1, bars="a", str_values='a\n\r"', field=Field.FOO)
    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=False)
    check_schema_with_instance(schema, instance, any_whitespace=False)


def test_optional():
    class MainModel(BaseModel):
        num: int = 0
        opt_bool: Optional[bool] = None
        size: Optional[float]
        name: str = ""

    ebnf_grammar = basic_json_rules_ebnf_no_space + (
        r"""root_prop_1 ::= basic_boolean | basic_null
root_prop_2 ::= basic_number | basic_null
root_part_2 ::= "" | ", " "\"name\"" ": " basic_string ""
root_part_1 ::= ", " "\"size\"" ": " root_prop_2 root_part_2
root_part_0 ::= root_part_1 | ", " "\"opt_bool\"" ": " root_prop_1 root_part_1
root ::= "{" "" (("\"num\"" ": " basic_integer root_part_0) | ("\"opt_bool\"" ": " root_prop_1 root_part_1) | ("\"size\"" ": " root_prop_2 root_part_2)) "" "}"
"""
    )

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=False)

    instance = MainModel(num=42, opt_bool=True, size=3.14, name="foo")
    check_schema_with_instance(schema, instance, any_whitespace=False)

    instance = MainModel(size=None)
    check_schema_with_instance(schema, instance, any_whitespace=False)

    check_schema_with_instance(schema, '{"size": null}', any_whitespace=False)
    check_schema_with_instance(schema, '{"size": null, "name": "foo"}', any_whitespace=False)
    check_schema_with_instance(
        schema, '{"num": 1, "size": null, "name": "foo"}', any_whitespace=False
    )


def test_all_optional():
    class MainModel(BaseModel):
        size: int = 0
        state: bool = False
        num: float = 0

    ebnf_grammar = basic_json_rules_ebnf_no_space + (
        r"""root_part_1 ::= "" | ", " "\"num\"" ": " basic_number ""
root_part_0 ::= root_part_1 | ", " "\"state\"" ": " basic_boolean root_part_1
root ::= ("{" "" (("\"size\"" ": " basic_integer root_part_0) | ("\"state\"" ": " basic_boolean root_part_1) | ("\"num\"" ": " basic_number "")) "" "}") | "{" "}"
"""
    )

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=False)

    instance = MainModel(size=42, state=True, num=3.14)
    check_schema_with_instance(schema, instance, any_whitespace=False)

    check_schema_with_instance(schema, '{"state": false}', any_whitespace=False)
    check_schema_with_instance(schema, '{"size": 1, "num": 1.5}', any_whitespace=False)


def test_all_optional_non_strict():
    class MainModel(BaseModel):
        size: int = 0
        state: bool = False
        num: float = 0

    ebnf_grammar_non_strict = basic_json_rules_ebnf_no_space + (
        r"""root_part_2 ::= (", " basic_string ": " basic_any)*
root_part_1 ::= root_part_2 | ", " "\"num\"" ": " basic_number root_part_2
root_part_0 ::= root_part_1 | ", " "\"state\"" ": " basic_boolean root_part_1
root ::= ("{" "" (("\"size\"" ": " basic_integer root_part_0) | ("\"state\"" ": " basic_boolean root_part_1) | ("\"num\"" ": " basic_number root_part_2) | basic_string ": " basic_any root_part_2) "" "}") | "{" "}"
"""
    )

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(
        schema, ebnf_grammar_non_strict, any_whitespace=False, strict_mode=False
    )

    check_schema_with_instance(
        schema, '{"size": 1, "num": 1.5, "other": false}', any_whitespace=False, strict_mode=False
    )
    check_schema_with_instance(schema, '{"other": false}', any_whitespace=False, strict_mode=False)


def test_empty():
    class MainModel(BaseModel):
        pass

    ebnf_grammar = basic_json_rules_ebnf_no_space + (
        r"""root ::= "{" "}"
"""
    )

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=False)

    instance = MainModel()
    check_schema_with_instance(schema, instance, any_whitespace=False)

    check_schema_with_instance(schema, '{"tmp": 123}', any_whitespace=False, strict_mode=False)


def test_reference():
    class Foo(BaseModel):
        count: int
        size: Optional[float] = None

    class Bar(BaseModel):
        apple: str = "x"
        banana: str = "y"

    class MainModel(BaseModel):
        foo: Foo
        bars: List[Bar]

    instance = MainModel(
        foo=Foo(count=42, size=3.14), bars=[Bar(apple="a", banana="b"), Bar(apple="c", banana="d")]
    )

    ebnf_grammar = basic_json_rules_ebnf_no_space + (
        r"""defs_Foo_prop_1 ::= basic_number | basic_null
defs_Foo_part_0 ::= "" | ", " "\"size\"" ": " defs_Foo_prop_1 ""
defs_Foo ::= "{" "" (("\"count\"" ": " basic_integer defs_Foo_part_0)) "" "}"
root_prop_0 ::= defs_Foo
defs_Bar_part_0 ::= "" | ", " "\"banana\"" ": " basic_string ""
defs_Bar ::= ("{" "" (("\"apple\"" ": " basic_string defs_Bar_part_0) | ("\"banana\"" ": " basic_string "")) "" "}") | "{" "}"
root_prop_1_additional ::= defs_Bar
root_prop_1 ::= (("[" "" root_prop_1_additional (", " root_prop_1_additional)* "" "]") | ("[" "" "]"))
root_part_0 ::= ", " "\"bars\"" ": " root_prop_1 ""
root ::= "{" "" (("\"foo\"" ": " root_prop_0 root_part_0)) "" "}"
"""
    )

    schema = MainModel.model_json_schema()
    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=False)
    check_schema_with_instance(schema, instance, any_whitespace=False)


def test_reference_schema():
    # Test simple reference with $defs
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/nested"}},
        "required": ["value"],
        "$defs": {
            "nested": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            }
        },
    }

    instance = {"value": {"name": "John", "age": 30}}
    instance_rejected = {"value": {"name": "John"}}

    check_schema_with_instance(schema, instance, any_whitespace=False)
    check_schema_with_instance(schema, instance_rejected, is_accepted=False, any_whitespace=False)

    # Test simple reference with definitions
    schema_def = {
        "type": "object",
        "properties": {"value": {"$ref": "#/definitions/nested"}},
        "required": ["value"],
        "definitions": {
            "nested": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            }
        },
    }

    check_schema_with_instance(schema_def, instance, any_whitespace=False)
    check_schema_with_instance(
        schema_def, instance_rejected, is_accepted=False, any_whitespace=False
    )

    # Test multi-level reference path
    schema_multi = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/level1/level2/nested"}},
        "required": ["value"],
        "$defs": {
            "level1": {
                "level2": {
                    "nested": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                        "required": ["name", "age"],
                    }
                }
            }
        },
    }

    check_schema_with_instance(schema_multi, instance, any_whitespace=False)
    check_schema_with_instance(
        schema_multi, instance_rejected, is_accepted=False, any_whitespace=False
    )

    # Test nested reference
    schema_nested = {
        "type": "object",
        "properties": {"value": {"$ref": "#/definitions/node_a"}},
        "required": ["value"],
        "definitions": {
            "node_a": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "child": {"$ref": "#/definitions/node_b"},
                },
                "required": ["name"],
            },
            "node_b": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
        },
    }

    instance_nested = {"value": {"name": "first", "child": {"id": 1}}}
    instance_nested_rejected = {"value": {"name": "first", "child": {}}}

    check_schema_with_instance(schema_nested, instance_nested, any_whitespace=False)
    check_schema_with_instance(
        schema_nested, instance_nested_rejected, is_accepted=False, any_whitespace=False
    )

    # Test schema with self-recursion through $defs
    schema_self_recursive = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/node"}},
        "required": ["value"],
        "$defs": {
            "node": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "next": {"$ref": "#/$defs/node"}},
                "required": ["id"],
            }
        },
    }

    instance_self_recursive = {"value": {"id": 1, "next": {"id": 2, "next": {"id": 3}}}}
    instance_self_recursive_1 = {"value": {"id": 1}}
    instance_self_recursive_rejected = {"value": {"id": 1, "next": {"next": {"id": 3}}}}

    check_schema_with_instance(schema_self_recursive, instance_self_recursive, any_whitespace=False)
    check_schema_with_instance(
        schema_self_recursive, instance_self_recursive_1, any_whitespace=False
    )
    check_schema_with_instance(
        schema_self_recursive,
        instance_self_recursive_rejected,
        is_accepted=False,
        any_whitespace=False,
    )

    # Test schema with circular references between multiple schemas
    schema_circular = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/schema_a"}},
        "required": ["value"],
        "$defs": {
            "schema_a": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "next": {"$ref": "#/$defs/schema_b"}},
                "required": ["name", "next"],
            },
            "schema_b": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "child": {"$ref": "#/$defs/schema_a"}},
                "required": ["id"],
            },
        },
    }

    instance_circular = {
        "value": {
            "name": "first",
            "next": {"id": 1, "child": {"name": "second", "next": {"id": 2}}},
        }
    }
    instance_circular_complex = {
        # fmt: off
        "value": {"name": "root", "next": {
            "id": 1, "child": {"name": "level1", "next": {
                "id": 2, "child": {"name": "level2", "next": {
                    "id": 3, "child": {"name": "level3", "next": {
                        "id": 4, "child": {"name": "level4", "next": {"id": 5}}
                    }}
                }}
            }}
        }}
        # fmt: on
    }
    instance_circular_rejected = {
        "value": {"name": "first", "next": {"child": {"name": "second", "next": {"id": 2}}}}
    }

    check_schema_with_instance(schema_circular, instance_circular, any_whitespace=False)
    check_schema_with_instance(schema_circular, instance_circular_complex, any_whitespace=False)
    check_schema_with_instance(
        schema_circular, instance_circular_rejected, is_accepted=False, any_whitespace=False
    )

    # Test self-referential schema
    schema_recursive = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "children": {"type": "array", "items": {"$ref": "#"}},
        },
        "required": ["name"],
    }

    instance_recursive = {
        "name": "root",
        "children": [{"name": "child1", "children": [{"name": "grandchild1"}]}, {"name": "child2"}],
    }
    instance_recursive_rejected = {"children": [{"name": "child1"}]}

    check_schema_with_instance(schema_recursive, instance_recursive, any_whitespace=False)
    check_schema_with_instance(
        schema_recursive, instance_recursive_rejected, is_accepted=False, any_whitespace=False
    )


def test_union():
    class Cat(BaseModel):
        name: str
        color: str

    class Dog(BaseModel):
        name: str
        breed: str

    ta = TypeAdapter(Union[Cat, Dog])

    model_schema = ta.json_schema()

    ebnf_grammar = basic_json_rules_ebnf_no_space + (
        r"""defs_Cat_part_0 ::= ", " "\"color\"" ": " basic_string ""
defs_Cat ::= "{" "" (("\"name\"" ": " basic_string defs_Cat_part_0)) "" "}"
root_case_0 ::= defs_Cat
defs_Dog_part_0 ::= ", " "\"breed\"" ": " basic_string ""
defs_Dog ::= "{" "" (("\"name\"" ": " basic_string defs_Dog_part_0)) "" "}"
root_case_1 ::= defs_Dog
root ::= root_case_0 | root_case_1
"""
    )

    check_schema_with_grammar(model_schema, ebnf_grammar, any_whitespace=False)

    check_schema_with_instance(model_schema, Cat(name="kitty", color="black"), any_whitespace=False)
    check_schema_with_instance(
        model_schema, Dog(name="doggy", breed="bulldog"), any_whitespace=False
    )
    check_schema_with_instance(
        model_schema, '{"name": "kitty", "test": "black"}', False, any_whitespace=False
    )


def test_anyof_oneof():
    schema = {
        "type": "object",
        "properties": {"name": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }
    schema_accepted_1 = '{"name": "John"}'
    schema_accepted_2 = '{"name": 123}'
    schema_rejected = '{"name": {"a": 1}}'
    check_schema_with_instance(schema, schema_accepted_1, any_whitespace=False)
    check_schema_with_instance(schema, schema_accepted_2, any_whitespace=False)
    check_schema_with_instance(schema, schema_rejected, is_accepted=False, any_whitespace=False)

    schema = {
        "type": "object",
        "properties": {"name": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
    }

    schema_accepted_1 = '{"name": "John"}'
    schema_accepted_2 = '{"name": 123}'
    schema_rejected = '{"name": {"a": 1}}'
    check_schema_with_instance(schema, schema_accepted_1, any_whitespace=False)
    check_schema_with_instance(schema, schema_accepted_2, any_whitespace=False)
    check_schema_with_instance(schema, schema_rejected, is_accepted=False, any_whitespace=False)


def test_alias():
    class MainModel(BaseModel):
        test: str = Field(..., alias="name")

    ebnf_grammar = basic_json_rules_ebnf_no_space + (
        r"""root ::= "{" "" (("\"name\"" ": " basic_string "")) "" "}"
"""
    )

    check_schema_with_grammar(MainModel.model_json_schema(), ebnf_grammar, any_whitespace=False)

    instance = MainModel(name="kitty")
    instance_str = json.dumps(instance.model_dump(mode="json", round_trip=True, by_alias=False))
    check_schema_with_instance(
        MainModel.model_json_schema(by_alias=False), instance_str, any_whitespace=False
    )

    instance_str = json.dumps(instance.model_dump(mode="json", round_trip=True, by_alias=True))
    check_schema_with_instance(
        MainModel.model_json_schema(by_alias=True), instance_str, any_whitespace=False
    )

    # property name contains space
    class MainModelSpace(BaseModel):
        test: Literal["abc"] = Field(..., alias="name 1")

    ebnf_grammar_space = basic_json_rules_ebnf_no_space + (
        r"""root_prop_0 ::= "\"abc\""
root ::= "{" "" (("\"name 1\"" ": " root_prop_0 "")) "" "}"
"""
    )

    check_schema_with_grammar(
        MainModelSpace.model_json_schema(), ebnf_grammar_space, any_whitespace=False
    )

    instance_space = MainModelSpace(**{"name 1": "abc"})
    instance_space_str = json.dumps(
        instance_space.model_dump(mode="json", round_trip=True, by_alias=True)
    )
    check_schema_with_instance(
        MainModelSpace.model_json_schema(by_alias=True), instance_space_str, any_whitespace=False
    )


def test_restricted_string():
    class MainModel(BaseModel):
        restricted_string: str = Field(..., pattern=r"[a-f]")

    instance = MainModel(restricted_string="a")
    instance_str = json.dumps(instance.model_dump(mode="json"))
    check_schema_with_instance(MainModel.model_json_schema(), instance_str, any_whitespace=False)

    check_schema_with_instance(
        MainModel.model_json_schema(),
        '{"restricted_string": "j"}',
        is_accepted=False,
        any_whitespace=False,
    )


def test_complex_restrictions():
    class RestrictedModel(BaseModel):
        restricted_string: str = Field(..., pattern=r"[^\"]*")
        restricted_value: int = Field(..., strict=True, ge=0, lt=44)

    # working instance
    instance = RestrictedModel(restricted_string="abd", restricted_value=42)
    instance_str = json.dumps(instance.model_dump(mode="json"))
    check_schema_with_instance(
        RestrictedModel.model_json_schema(), instance_str, any_whitespace=False
    )

    instance_err = RestrictedModel(restricted_string='"', restricted_value=42)
    instance_str = json.dumps(instance_err.model_dump(mode="json"))
    check_schema_with_instance(
        RestrictedModel.model_json_schema(), instance_str, is_accepted=False, any_whitespace=False
    )

    check_schema_with_instance(
        RestrictedModel.model_json_schema(),
        '{"restricted_string": "j", "restricted_value": 45}',
        is_accepted=False,
        any_whitespace=False,
    )


def test_dynamic_model():
    class MainModel(BaseModel):
        restricted_string: str = Field(..., pattern=r"[a-f]")

    additional_fields = {"restricted_string_dynamic": (str, Field(..., pattern=r"[a-x]"))}

    CompleteModel: Type[BaseModel] = create_model(
        "CompleteModel", __base__=MainModel, **additional_fields
    )
    instance = CompleteModel(restricted_string="a", restricted_string_dynamic="j")
    instance_str = json.dumps(instance.model_dump(mode="json"))
    check_schema_with_instance(
        CompleteModel.model_json_schema(), instance_str, any_whitespace=False
    )


def test_any_whitespace():
    class SimpleModel(BaseModel):
        value: str
        arr: List[int]
        obj: Dict[str, int]

    schema = SimpleModel.model_json_schema()

    ebnf_grammar = basic_json_rules_ebnf + (
        r"""root_prop_1 ::= (("[" [ \n\t]* basic_integer ([ \n\t]* "," [ \n\t]* basic_integer)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
root_prop_2 ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_integer ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_integer)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_part_1 ::= [ \n\t]* "," [ \n\t]* "\"obj\"" [ \n\t]* ":" [ \n\t]* root_prop_2 ""
root_part_0 ::= [ \n\t]* "," [ \n\t]* "\"arr\"" [ \n\t]* ":" [ \n\t]* root_prop_1 root_part_1
root ::= "{" [ \n\t]* (("\"value\"" [ \n\t]* ":" [ \n\t]* basic_string root_part_0)) [ \n\t]* "}"
"""
    )

    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=True, strict_mode=True)

    ebnf_grammar = basic_json_rules_ebnf + (
        r"""root_prop_1 ::= (("[" [ \n\t]* basic_integer ([ \n\t]* "," [ \n\t]* basic_integer)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
root_prop_2 ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_integer ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_integer)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_part_2 ::= ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)*
root_part_1 ::= [ \n\t]* "," [ \n\t]* "\"obj\"" [ \n\t]* ":" [ \n\t]* root_prop_2 root_part_2
root_part_0 ::= [ \n\t]* "," [ \n\t]* "\"arr\"" [ \n\t]* ":" [ \n\t]* root_prop_1 root_part_1
root ::= "{" [ \n\t]* (("\"value\"" [ \n\t]* ":" [ \n\t]* basic_string root_part_0)) [ \n\t]* "}"
"""
    )

    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=True, strict_mode=False)

    # Test that different whitespace variations are accepted when any_whitespace=True
    instances = [
        '{"value": "test", "arr": [1, 2], "obj": {"a": 1}}',
        '{ "value" : "test", "arr": [1, 2], "obj": {"a": 1} }',
        '{\n  "value"  :  "test",\n  "arr"  :  [1, 2],\n  "obj"  :  {"a": 1}\n}',
        '{\t"value"\t:\t"test",\t"arr":\t[1,\t2],\t"obj":\t{"a":\t1}\t}',
    ]
    for instance in instances:
        check_schema_with_instance(schema, instance, any_whitespace=True)


schema__err_message__test_array_schema_error_cases = [
    ({"type": "array", "prefixItems": {"type": "string"}}, "prefixItems must be an array"),
    (
        {"type": "array", "prefixItems": ["not an object"]},
        "prefixItems must be an array of objects or booleans",
    ),
    ({"type": "array", "prefixItems": [False]}, "prefixItems contains false"),
    ({"type": "array", "items": "not an object"}, "items must be a boolean or an object"),
    (
        {"type": "array", "unevaluatedItems": "not an object"},
        "unevaluatedItems must be a boolean or an object",
    ),
    ({"type": "array", "minItems": "not an integer"}, "minItems must be an integer"),
    ({"type": "array", "maxItems": -1}, "maxItems must be a non-negative integer"),
    ({"type": "array", "minItems": 5, "maxItems": 3}, "minItems is greater than maxItems: 5 > 3"),
    (
        {"type": "array", "prefixItems": [{}, {}, {}], "maxItems": 2},
        "maxItems is less than the number of prefixItems: 2 < 3",
    ),
    (
        {"type": "array", "prefixItems": [{}, {}], "minItems": 3, "items": False},
        "minItems is greater than the number of prefixItems, but additional items are not "
        "allowed: 3 > 2",
    ),
]


@pytest.mark.parametrize("schema, err_message", schema__err_message__test_array_schema_error_cases)
def test_array_schema_error_cases(schema: Dict[str, Any], err_message: str):
    with pytest.raises(Exception) as e:
        _json_schema_to_ebnf(schema)
    assert err_message in str(e.value)


schema__expected_grammar__instances__test_array_schema = [
    (
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
        },
        (
            basic_json_rules_ebnf
            + r"""root_additional_part_0 ::= [ \n\t]* "," [ \n\t]* "\"age\"" [ \n\t]* ":" [ \n\t]* basic_integer ""
root_additional ::= "{" [ \n\t]* (("\"name\"" [ \n\t]* ":" [ \n\t]* basic_string root_additional_part_0)) [ \n\t]* "}"
root ::= (("[" [ \n\t]* root_additional ([ \n\t]* "," [ \n\t]* root_additional)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
"""
        ),
        [
            ([{"name": "John", "age": 30}, {"name": "Jane", "age": 25}], True),
            ([{"name": "John"}], False),
        ],
    ),
    (
        {
            "type": "array",
            "prefixItems": [
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    "required": ["name", "age"],
                },
                {"type": "integer"},
                {"type": "string"},
            ],
            "additionalItems": False,
        },
        (
            basic_json_rules_ebnf
            + r"""root_item_0_part_0 ::= [ \n\t]* "," [ \n\t]* "\"age\"" [ \n\t]* ":" [ \n\t]* basic_integer ""
root_item_0 ::= "{" [ \n\t]* (("\"name\"" [ \n\t]* ":" [ \n\t]* basic_string root_item_0_part_0)) [ \n\t]* "}"
root ::= ("[" [ \n\t]* (root_item_0 [ \n\t]* "," [ \n\t]* basic_integer [ \n\t]* "," [ \n\t]* basic_string) [ \n\t]* "]")
"""
        ),
        [
            ([{"name": "John", "age": 30}, 42, "test"], True),
            ([{"name": "John", "age": 30}, 42], False),
            ([{"name": "John", "age": 30}, "test", 42], False),
            ([{"name": "John"}, 42, "test"], False),
        ],
    ),
    (
        {
            "type": "array",
            "prefixItems": [
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    "required": ["name", "age"],
                },
                {"type": "integer"},
            ],
            "unevaluatedItems": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
        (
            basic_json_rules_ebnf
            + r"""root_item_0_part_0 ::= [ \n\t]* "," [ \n\t]* "\"age\"" [ \n\t]* ":" [ \n\t]* basic_integer ""
root_item_0 ::= "{" [ \n\t]* (("\"name\"" [ \n\t]* ":" [ \n\t]* basic_string root_item_0_part_0)) [ \n\t]* "}"
root_additional ::= "{" [ \n\t]* (("\"name\"" [ \n\t]* ":" [ \n\t]* basic_string "")) [ \n\t]* "}"
root ::= ("[" [ \n\t]* (root_item_0 [ \n\t]* "," [ \n\t]* basic_integer) ([ \n\t]* "," [ \n\t]* root_additional)* [ \n\t]* "]")
"""
        ),
        [
            ([{"name": "John", "age": 30}, 42, {"name": "Jane"}], True),
            ([{"name": "John", "age": 30}, 42], True),
            ([{"name": "John", "age": 30}, 42, 123], False),
            ([{"name": "John", "age": 30}, {"name": "Jane"}], False),
        ],
    ),
]


@pytest.mark.parametrize(
    "schema, expected_grammar, instances", schema__expected_grammar__instances__test_array_schema
)
def test_array_schema(
    schema: Dict[str, Any], expected_grammar: str, instances: List[Tuple[Any, bool]]
):
    check_schema_with_grammar(schema, expected_grammar)

    for instance, is_accepted in instances:
        check_schema_with_instance(schema, instance, is_accepted=is_accepted)


schema__expected_grammar__instances__test_array_schema_min_max = [
    # prefix empty, additional items not allowed
    (
        {"type": "array", "items": False, "prefixItems": []},
        (
            basic_json_rules_ebnf
            + r"""root ::= ("[" [ \n\t]* "]")
"""
        ),
        [([], True), ([1], False), ([1, 2], False)],
    ),
    # prefix empty, additional items allowed, min=0 max=0
    (
        {"type": "array", "items": {"type": "integer"}, "minItems": 0, "maxItems": 0},
        (
            basic_json_rules_ebnf
            + r"""root ::= ("[" [ \n\t]* "]")
"""
        ),
        [([], True), ([1], False), ([1, 2], False)],
    ),
    # prefix empty, additional items allowed, min=0 max>0
    (
        {"type": "array", "items": {"type": "integer"}, "minItems": 0, "maxItems": 2},
        (
            basic_json_rules_ebnf
            + r"""root ::= (("[" [ \n\t]* basic_integer ([ \n\t]* "," [ \n\t]* basic_integer)? [ \n\t]* "]") | ("[" [ \n\t]* "]"))
"""
        ),
        [([], True), ([1], True), ([1, 2], True), ([1, 2, 3], False)],
    ),
    # prefix empty, additional items allowed, min>0
    (
        {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 3},
        (
            basic_json_rules_ebnf
            + r"""root ::= ("[" [ \n\t]* basic_integer ([ \n\t]* "," [ \n\t]* basic_integer){1,2} [ \n\t]* "]")
"""
        ),
        [([], False), ([1], False), ([1, 2], True), ([1, 2, 3], True), ([1, 2, 3, 4], False)],
    ),
    # prefix non-empty, additional items not allowed
    (
        {"type": "array", "items": False, "prefixItems": [{"type": "string"}, {"type": "integer"}]},
        (
            basic_json_rules_ebnf
            + r"""root ::= ("[" [ \n\t]* (basic_string [ \n\t]* "," [ \n\t]* basic_integer) [ \n\t]* "]")
"""
        ),
        [(["foo", 42], True), (["foo", 42, "bar"], False), (["foo"], False), ([42, "foo"], False)],
    ),
    # prefix non-empty, additional items allowed
    (
        {
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "integer"}],
            "items": {"type": "boolean"},
            "minItems": 3,
            "maxItems": 4,
        },
        (
            basic_json_rules_ebnf
            + r"""root ::= ("[" [ \n\t]* (basic_string [ \n\t]* "," [ \n\t]* basic_integer) ([ \n\t]* "," [ \n\t]* basic_boolean){1,2} [ \n\t]* "]")
"""
        ),
        [
            (["foo", 42, True], True),
            (["foo", 42, True, False], True),
            (["foo", 42], False),
            (["foo", 42, True, False, True], False),
            (["foo", 42, "bar"], False),
        ],
    ),
    # prefix non-empty, additional items allowed, maxItems not set
    (
        {
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "integer"}],
            "items": {"type": "boolean"},
            "minItems": 3,
        },
        (
            basic_json_rules_ebnf
            + r"""root ::= ("[" [ \n\t]* (basic_string [ \n\t]* "," [ \n\t]* basic_integer) ([ \n\t]* "," [ \n\t]* basic_boolean)+ [ \n\t]* "]")
"""
        ),
        [
            (["foo", 42, True], True),
            (["foo", 42, True, False], True),
            (["foo", 42, True, False, True], True),
            (["foo", 42], False),
            (["foo", 42, "bar"], False),
        ],
    ),
]


@pytest.mark.parametrize(
    "schema, expected_grammar, instances",
    schema__expected_grammar__instances__test_array_schema_min_max,
)
def test_array_schema_min_max(
    schema: Dict[str, Any], expected_grammar: str, instances: List[Tuple[Any, bool]]
):
    grammar_ebnf = _json_schema_to_ebnf(schema)
    assert grammar_ebnf == expected_grammar
    for instance, is_accepted in instances:
        check_schema_with_instance(schema, instance, is_accepted=is_accepted)


def test_array_with_only_items_keyword():
    schema = {
        "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
    }
    instance_accepted = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
    instance_rejected = [{"name": "John"}]
    check_schema_with_instance(schema, instance_accepted, any_whitespace=False)
    check_schema_with_instance(schema, instance_rejected, is_accepted=False, any_whitespace=False)

    schema_prefix_items = {
        "prefixItems": [
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
        ]
    }

    check_schema_with_instance(schema_prefix_items, instance_accepted, any_whitespace=False)
    check_schema_with_instance(
        schema_prefix_items, instance_rejected, is_accepted=False, any_whitespace=False
    )

    schema_unevaluated_items = {
        "unevaluatedItems": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
    }

    check_schema_with_instance(schema_unevaluated_items, instance_accepted, any_whitespace=False)
    check_schema_with_instance(
        schema_unevaluated_items, instance_rejected, is_accepted=False, any_whitespace=False
    )


def test_object_with_only_properties_keyword():
    schema = {
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    instance_accepted = {"name": "John", "age": 30}
    instance_rejected = {"name": "John"}
    check_schema_with_instance(schema, instance_accepted, any_whitespace=False)
    check_schema_with_instance(schema, instance_rejected, is_accepted=False, any_whitespace=False)

    schema_additional_properties = {"additionalProperties": {"type": "string"}}
    instance_accepted = {"name": "John"}
    instance_rejected = {"name": "John", "age": 30}

    check_schema_with_instance(
        schema_additional_properties, instance_accepted, any_whitespace=False
    )
    check_schema_with_instance(
        schema_additional_properties, instance_rejected, is_accepted=False, any_whitespace=False
    )

    schema_unevaluated_properties = {"unevaluatedProperties": {"type": "string"}}

    check_schema_with_instance(
        schema_unevaluated_properties, instance_accepted, any_whitespace=False
    )
    check_schema_with_instance(
        schema_unevaluated_properties, instance_rejected, is_accepted=False, any_whitespace=False
    )


def test_object_with_pattern_properties_and_property_names():
    schema = {
        "type": "object",
        "patternProperties": {
            "^[a-zA-Z]+$": {"type": "string"},
            "^[0-9]+$": {"type": "integer"},
            "^[a-zA-Z]*_[0-9]*$": {"type": "object"},
        },
    }

    instance_accepted = [
        {"aBcDe": "aaa"},
        {"12345": 12345},
        {"abc_123": {"key": "value"}},
        {"_": {"key": "value"}},
        {"a": "value", "b": "another_value", "000": 12345, "abc_123": {"key": "value"}},
        {"000": 12345, "a": "value", "abc_123": {"key": "value"}, "b": "another_value"},
    ]

    instance_rejected = [
        {"233A": "adfa"},
        {"aBcDe": 12345},
        {"12345": "aaa"},
        {"abc_123": 12345},
        {"a": "value", "b": "another_value", "000": 12345, "abc_123": "aaa"},
        {
            "a": "value",
            "b": "another_value",
            "000": 12345,
            "???": {"key": "value"},
            "abc_123": {"key": "value"},
        },
        {"000": 12345, "a": "value", "abc_123": {"key": "value"}, "b": 12345},
    ]

    for instance in instance_accepted:
        check_schema_with_instance(schema, instance, any_whitespace=False)
    for instance in instance_rejected:
        check_schema_with_instance(schema, instance, is_accepted=False, any_whitespace=False)

    schema = {"type": "object", "propertyNames": {"pattern": "^[a-zA-Z0-9_]+$"}}

    instance_accepted = [
        {"aBcDe": "aaa"},
        {"12345": 12345},
        {"abc_123": {"key": "value"}},
        {"_": {"key": "value"}},
        {"a": "value", "b": "another_value", "000": 12345, "abc_123": {"key": "value"}},
        {"000": 12345, "a": "value", "abc_123": {"key": "value"}, "b": "another_value"},
    ]

    instance_rejected = [
        {"aBc?De": "aaa"},
        {"1234|5": 12345},
        {"abc_1.23": {"key": "value"}},
        {"_/": {"key": "value"}},
        {"a&": "value", "b": "another_value", "000": 12345, "abc_123": {"key": "value"}},
        {"00(0": 12345, "a": "value", "abc_123": {"key": "value"}, "b": "another_value"},
    ]

    for instance in instance_accepted:
        check_schema_with_instance(schema, instance, any_whitespace=False)
    for instance in instance_rejected:
        check_schema_with_instance(schema, instance, is_accepted=False, any_whitespace=False)


def test_object_with_property_numbers():
    base_schema = {
        "properties": {
            "key1": {"type": "string"},
            "key2": {"type": "string"},
            "key3": {"type": "string"},
            "key4": {"type": "string"},
        }
    }
    # fmt: off
    instances = [
        ({}, 0 , False),
        ({"key1": "value1"}, 1, False),
        ({"key4": "value4"}, 1, False),
        ({"additional_key": "value"}, 1, True),
        ({"key1": "value1", "additional_key": "value"}, 2, True),
        ({"key1": "value1", "key2": "value2"}, 2, False),
        ({"key1": "value1", "key4": "value4"}, 2, False),
        ({"key2": "value2", "key3": "value3"}, 2, False),
        ({"key1": "value1", "key2": "value2", "additional_key": "value"}, 3, True),
        ({"key1": "value1", "key4": "value4", "additional_key": "value"}, 3, True),
        ({"key3": "value3", "key4": "value4", "additional_key": "value"}, 3, True),
        ({"key1": "value1", "key2": "value2", "key3": "value3"}, 3, False),
        ({"key2": "value2", "key3": "value3", "key4": "value4"}, 3, False),
        ({"key1": "value1", "key2": "value2", "key3": "value3", "additional_key": "value"}, 4, True),
        ({"key2": "value2", "key3": "value3", "key4": "value4", "additional_key": "value"}, 4, True),
        ({"key1": "value1", "key2": "value2", "key3": "value3", "key4": "value4"}, 4, False),
        ({"key1": "value1", "key2": "value2", "key3": "value3", "key4": "value4", "additional_key": "value"}, 5, True),
        ({"additional_key1": "value1", "additional_key2": "value2"}, 2, True),
        ({"key1": "value1", "key2": "value2", "additional_key1": "value", "additional_key2": "value2"}, 4 , True),
        ({"key1": "value1", "key2": "value2", "key3": "value3", "additional_key1": "value", "additional_key2": "value2"}, 5, True),
        ({"key1": "value1", "key2": "value2", "key3": "value3", "key4": "value4", "additional_key1": "value", "additional_key2": "value2"}, 6, True),
        ({"additional_key1": "value1", "additional_key2": "value2", "additional_key3": "value3"}, 3, True),
        ({"additional_key1": "value1", "additional_key2": "value2", "additional_key3": "value3", "additional_key4": "value4", "additional_key5": "value5", "additional_key6": "value6"}, 6, True),
    ]
    # fmt: on

    # Case 1. all properties are optional
    # Case 1.1. no additional properties
    schema = {**base_schema}
    for instance, num_properties, have_additional in instances:
        check_schema_with_instance(
            schema, instance, is_accepted=not have_additional, any_whitespace=False
        )

    for lower_bound in range(8):
        schema = {**base_schema, "minProperties": lower_bound}
        for instance, num_properties, have_additional in instances:
            if lower_bound > len(base_schema["properties"]):
                continue
            if num_properties < lower_bound:
                is_accepted = False
            else:
                is_accepted = not have_additional
            check_schema_with_instance(
                schema, instance, is_accepted=is_accepted, any_whitespace=False
            )

        for upper_bound in range(lower_bound, 8):
            schema = {**base_schema, "minProperties": lower_bound, "maxProperties": upper_bound}
            for instance, num_properties, have_additional in instances:
                if lower_bound > len(base_schema["properties"]):
                    continue
                if num_properties < lower_bound or num_properties > upper_bound:
                    is_accepted = False
                else:
                    is_accepted = not have_additional
                check_schema_with_instance(
                    schema, instance, is_accepted=is_accepted, any_whitespace=False
                )

    # Case 1.2. additional properties allowed
    schema = {**base_schema, "additionalProperties": {"type": "string"}}
    for instance, num_properties, have_additional in instances:
        check_schema_with_instance(schema, instance, any_whitespace=False)

    for lower_bound in range(8):
        schema = {
            **base_schema,
            "minProperties": lower_bound,
            "additionalProperties": {"type": "string"},
        }
        for instance, num_properties, have_additional in instances:
            if num_properties < lower_bound:
                is_accepted = False
            else:
                is_accepted = True
            check_schema_with_instance(
                schema, instance, is_accepted=is_accepted, any_whitespace=False
            )

        for upper_bound in range(lower_bound, 8):
            schema = {
                **base_schema,
                "minProperties": lower_bound,
                "maxProperties": upper_bound,
                "additionalProperties": {"type": "string"},
            }
            for instance, num_properties, have_additional in instances:
                if lower_bound > len(base_schema["properties"]):
                    continue
                if num_properties < lower_bound or num_properties > upper_bound:
                    is_accepted = False
                else:
                    is_accepted = True
                check_schema_with_instance(
                    schema, instance, is_accepted=is_accepted, any_whitespace=False
                )

    # Case 2. required properties are defined
    # Case 2.1. no additional properties
    required_properties_instance = [
        [],
        ["key1"],
        ["key3"],
        ["key4"],
        ["key1", "key2"],
        ["key1", "key4"],
        ["key3", "key4"],
        ["key1", "key2", "key3"],
        ["key2", "key3", "key4"],
        ["key1", "key2", "key3", "key4"],
    ]

    for required_properties in required_properties_instance:
        schema = {**base_schema, "required": required_properties}
        for instance, num_properties, have_additional in instances:
            is_accepted = (
                all(prop in instance for prop in required_properties) and not have_additional
            )
            check_schema_with_instance(
                schema, instance, is_accepted=is_accepted, any_whitespace=False
            )

        for lower_bound in range(8):
            schema = {**base_schema, "minProperties": lower_bound, "required": required_properties}
            for instance, num_properties, have_additional in instances:
                if lower_bound > len(base_schema["properties"]):
                    continue
                if num_properties < lower_bound:
                    is_accepted = False
                else:
                    is_accepted = (
                        all(prop in instance for prop in required_properties)
                        and not have_additional
                    )
                check_schema_with_instance(
                    schema, instance, is_accepted=is_accepted, any_whitespace=False
                )

            for upper_bound in range(lower_bound, 8):
                schema = {
                    **base_schema,
                    "minProperties": lower_bound,
                    "maxProperties": upper_bound,
                    "required": required_properties,
                }
                for instance, num_properties, have_additional in instances:
                    if lower_bound > len(base_schema["properties"]) or upper_bound < len(
                        required_properties
                    ):
                        continue
                    if num_properties < lower_bound or num_properties > upper_bound:
                        is_accepted = False
                    else:
                        is_accepted = (
                            all(prop in instance for prop in required_properties)
                            and not have_additional
                        )
                    check_schema_with_instance(
                        schema, instance, is_accepted=is_accepted, any_whitespace=False
                    )

    # Case 2.2. additional properties allowed
    for required_properties in required_properties_instance:
        schema = {
            **base_schema,
            "required": required_properties,
            "additionalProperties": {"type": "string"},
        }
        for instance, num_properties, have_additional in instances:
            is_accepted = all(prop in instance for prop in required_properties)
            check_schema_with_instance(
                schema, instance, is_accepted=is_accepted, any_whitespace=False
            )

        for lower_bound in range(8):
            schema = {
                **base_schema,
                "minProperties": lower_bound,
                "required": required_properties,
                "additionalProperties": {"type": "string"},
            }
            for instance, num_properties, have_additional in instances:
                if lower_bound > len(base_schema["properties"]):
                    continue
                if num_properties < lower_bound:
                    is_accepted = False
                else:
                    is_accepted = all(prop in instance for prop in required_properties)
                check_schema_with_instance(
                    schema, instance, is_accepted=is_accepted, any_whitespace=False
                )

            for upper_bound in range(lower_bound, 8):
                schema = {
                    **base_schema,
                    "minProperties": lower_bound,
                    "maxProperties": upper_bound,
                    "required": required_properties,
                    "additionalProperties": {"type": "string"},
                }
                for instance, num_properties, have_additional in instances:
                    if lower_bound > len(base_schema["properties"]) or upper_bound < len(
                        required_properties
                    ):
                        continue
                    if num_properties < lower_bound or num_properties > upper_bound:
                        is_accepted = False
                    else:
                        is_accepted = all(prop in instance for prop in required_properties)
                    check_schema_with_instance(
                        schema, instance, is_accepted=is_accepted, any_whitespace=False
                    )

    # Case 3. No properties defined
    for lower_bound in range(8):
        schema = {
            "type": "object",
            "minProperties": lower_bound,
            "additionalProperties": {"type": "string"},
        }
        for instance, num_properties, have_additional in instances:
            if num_properties < lower_bound:
                is_accepted = False
            else:
                is_accepted = True
            check_schema_with_instance(
                schema, instance, is_accepted=is_accepted, any_whitespace=False
            )

        for upper_bound in range(lower_bound, 8):
            schema = {
                "type": "object",
                "minProperties": lower_bound,
                "maxProperties": upper_bound,
                "additionalProperties": {"type": "string"},
            }
            for instance, num_properties, have_additional in instances:
                if num_properties < lower_bound or num_properties > upper_bound:
                    is_accepted = False
                else:
                    is_accepted = True
                check_schema_with_instance(
                    schema, instance, is_accepted=is_accepted, any_whitespace=False
                )


def test_object_error_handle():
    # Test error handling for invalid object schemas

    def compile_from_schema(schema):
        xgr.Grammar.from_json_schema(
            json.dumps(schema), any_whitespace=True, indent=None, separators=None, strict_mode=True
        )

    schema = {"type": "object", "properties": "not an object"}
    with pytest.raises(Exception) as e:
        compile_from_schema(schema)
    assert "properties must be an object" in str(e.value)

    schema = {"type": "object", "required": {"key": "not an array"}}
    with pytest.raises(Exception) as e:
        compile_from_schema(schema)
    assert "required must be an array" in str(e.value)

    with pytest.raises(Exception) as e:
        compile_from_schema({"type": "object", "patternProperties": ["not an object"]})
    assert "patternProperties must be an object" in str(e.value)

    with pytest.raises(Exception) as e:
        compile_from_schema({"type": "object", "propertyNames": "not an object"})
    assert "propertyNames must be an object" in str(e.value)

    with pytest.raises(Exception) as e:
        compile_from_schema({"type": "object", "propertyNames": {"type": "object"}})
    assert "propertyNames must be an object that validates string" in str(e.value)

    with pytest.raises(Exception) as e:
        compile_from_schema({"type": "object", "minProperties": "not an integer"})
    assert "minProperties must be an integer" in str(e.value)

    with pytest.raises(Exception) as e:
        compile_from_schema({"type": "object", "maxProperties": "not an integer"})
    assert "maxProperties must be an integer" in str(e.value)

    with pytest.raises(Exception) as e:
        compile_from_schema({"type": "object", "minProperties": -1})
    assert "minProperties must be a non-negative integer" in str(e.value)

    with pytest.raises(Exception) as e:
        compile_from_schema({"type": "object", "maxProperties": -1})
    assert "maxProperties must be a non-negative integer" in str(e.value)

    with pytest.raises(Exception) as e:
        compile_from_schema({"type": "object", "minProperties": 5, "maxProperties": 3})
    assert "minxPropertiesmax is greater than maxProperties" in str(e.value)

    with pytest.raises(Exception) as e:
        compile_from_schema({"type": "object", "maxProperties": 1, "required": ["key1", "key2"]})
    assert "maxProperties is less than the number of required properties" in str(e.value)

    with pytest.raises(Exception) as e:
        compile_from_schema(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"key": {"type": "string"}},
                "minProperties": 2,
            }
        )
    assert (
        "minProperties is greater than the number of properties, but additional properties aren't allowed"
        in str(e.value)
    )


def test_generate_range_regex():
    # Basic range tests
    assert _generate_range_regex(12, 16) == r"^((1[2-6]))$"
    assert _generate_range_regex(1, 10) == r"^(([1-9]|10))$"
    assert (
        _generate_range_regex(2134, 3459)
        == r"^((2[2-9]\d{2}|2[2-9]\d{2}|21[4-9]\d{1}|213[5-9]|2134|3[0-3]\d{2}|3[0-3]\d{2}|34[0-4]\d{1}|345[0-8]|3459))$"
    )

    # Negative to positive range
    assert _generate_range_regex(-5, 10) == r"^(-([1-5])|0|([1-9]|10))$"

    # Pure negative range
    assert _generate_range_regex(-15, -10) == r"^(-(1[0-5]))$"

    # Large ranges
    assert (
        _generate_range_regex(-1999, -100)
        == r"^(-([1-9]\d{2}|1[0-8]\d{2}|19[0-8]\d{1}|199[0-8]|1999))$"
    )
    assert _generate_range_regex(1, 9999) == r"^(([1-9]|[1-9]\d{1}|[1-9]\d{2}|[1-9]\d{3}))$"

    # Unbounded ranges (None cases)
    assert _generate_range_regex(None, None) == r"^-?\d+$"
    assert _generate_range_regex(5, None) == r"^([5-9]|[1-9]\d*)$"
    assert _generate_range_regex(None, 0) == r"^(-[1-9]\d*|0)$"

    # Medium range
    assert (
        _generate_range_regex(78, 1278)
        == r"^(([8-9]\d{1}|79|78|[1-9]\d{2}|1[0-1]\d{2}|12[0-6]\d{1}|127[0-7]|1278))$"
    )

    # Symmetric range around zero
    assert (
        _generate_range_regex(-100, 100) == r"^(-([1-9]|[1-9]\d{1}|100)|0|([1-9]|[1-9]\d{1}|100))$"
    )

    # Upper bound negative
    assert _generate_range_regex(None, -123) == r"^(-123|-1[0-1]\d{1}|-12[0-2]|-[1-9]\d{3,})$"

    # Additional edge cases
    # Single number
    assert _generate_range_regex(5, 5) == r"^((5))$"

    # Zero-inclusive ranges
    assert _generate_range_regex(-10, 0) == r"^(-([1-9]|10)|0)$"
    assert _generate_range_regex(0, 10) == r"^(0|([1-9]|10))$"


instance__accepted__test_email_format = [
    (r"simple@example.com", True),
    (r"very.common@example.com", True),
    (r"FirstName.LastName@EasierReading.org", True),
    (r"x@example.com", True),
    (r"long.email-address-with-hyphens@and.subdomains.example.com", True),
    (r"user.name+tag+sorting@example.com", True),
    (r"name/surname@example.com", True),
    (r"admin@example", True),
    (r"example@s.example", True),
    (r"\" \"@example.org", True),  #
    (r"\"john..doe\"@example.org", True),  #
    (r"mailhost!username@example.org", True),
    (r"\"very.(),:;<>[]\\\".VERY.\\\"very@\\\\ \\\"very\\\".unusual\"@strange.example.com", True),
    (r"user%example.com@example.org", True),
    (r"user-@example.org", True),
    (r"abc.example.com", False),
    (r"a@b@c@example.com", False),
    (r'a"b(c)d,e:f;g<h>i[j\k]l@example.com', False),
    (r'just"not"right@example.com', False),
    (r'this is"not\allowed@example.com', False),
    (r"this\ still\"not\\allowed@example.com", False),
    (r"i.like.underscores@but_they_are_not_allowed_in_this_part", False),
]


@pytest.mark.parametrize("instance, accepted", instance__accepted__test_email_format)
def test_email_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "email"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" ( ( [a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+ ( "." [a-zA-Z0-9_!#$%&'*+/=?^`{|}~-]+ )* ) | "\\" "\"" ( "\\" [ -~] | [ !#-[\]-~] )* "\\" "\"" ) "@" ( [A-Za-z0-9] ( [\-A-Za-z0-9]* [A-Za-z0-9] )? ) ( ( "." [A-Za-z0-9] [\-A-Za-z0-9]* [A-Za-z0-9] )* ) "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


instance__accepted__test_date_format = [
    (r"0000-01-01", True),
    (r"9999-12-31", True),
    (r"10-01-01", False),
    (r"2025-00-01", False),
    (r"2025-13-01", False),
    (r"2025-01-00", False),
    (r"2025-01-32", False),
]


@pytest.mark.parametrize("instance, accepted", instance__accepted__test_date_format)
def test_date_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "date"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" ( [0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [01] ) ) "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


instance__accepted__test_time_format = [
    (r"00:00:00Z", True),
    (r"23:59:60Z", True),
    (r"12:34:56Z", True),
    (r"12:34:56+07:08", True),
    (r"12:34:56-07:08", True),
    (r"12:34:56.7Z", True),
    (r"12:34:56.7+08:09", True),
    (r"12:34:56.7-08:09", True),
    (r"00:00:00", False),
    (r"23:59:60", False),
    (r"12:34:56.7", False),
    (r"12:34:56.7890", False),
    (r"24:00:00", False),
    (r"00:60:00", False),
    (r"00:00:61", False),
    (r"00:00:00.", False),
    (r"12:34:56+07:", False),
    (r"12:34:56-07:", False),
    (r"12:34:56.7+-08:09", False),
]


@pytest.mark.parametrize("instance, accepted", instance__accepted__test_time_format)
def test_time_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "time"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] ":" ( [0-5] [0-9] | "6" "0" ) ( "." [0-9]+ )? ( "Z" | [+-] ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] ) "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


instance__accepted__test_duration_format = [
    (r"P0Y", True),
    (r"P12M", True),
    (r"P345D", True),
    (r"P6789W", True),
    (r"P01234D", True),
    (r"PT9H", True),
    (r"PT87M", True),
    (r"PT654S", True),
    (r"P1Y23M456D", True),
    (r"P23M456D", True),
    (r"P1Y0M456D", True),
    (r"P1Y23M", True),
    (r"PT9H87M654S", True),
    (r"PT87M654S", True),
    (r"PT9H0M654S", True),
    (r"PT9H87M", True),
    (r"P1Y23M456DT9H87M654S", True),
    (r"P", False),
    (r"PD", False),
    (r"P1", False),
    (r"PT", False),
    (r"P1Y456D", False),
    (r"PT9H654S", False),
]


@pytest.mark.parametrize("instance, accepted", instance__accepted__test_duration_format)
def test_duration_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "duration"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" "P" ( ( [0-9]+ "D" | [0-9]+ "M" ( [0-9]+ "D" )? | [0-9]+ "Y" ( [0-9]+ "M" ( [0-9]+ "D" )? )? ) ( "T" ( [0-9]+ "S" | [0-9]+ "M" ( [0-9]+ "S" )? | [0-9]+ "H" ( [0-9]+ "M" ( [0-9]+ "S" )? )? ) )? | "T" ( [0-9]+ "S" | [0-9]+ "M" ( [0-9]+ "S" )? | [0-9]+ "H" ( [0-9]+ "M" ( [0-9]+ "S" )? )? ) | [0-9]+ "W" ) "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


instance__accepted__test_ipv6_format = [
    (r"0123:4567:890a:bced:fABC:DEF0:1234:5678", True),
    (r"::6666:6666:6666:6666:6666:6666", True),
    (r"::6666:6666:6666:6666:6666", True),
    (r"::6666:6666:6666:6666", True),
    (r"::6666:6666:6666", True),
    (r"::6666:6666", True),
    (r"::6666", True),
    (r"::", True),
    (r"8888:8888:8888:8888:8888:8888::", True),
    (r"8888:8888:8888:8888:8888::", True),
    (r"8888:8888:8888:8888::", True),
    (r"8888:8888:8888::", True),
    (r"8888:8888::", True),
    (r"8888::", True),
    (r"1111::2222", True),
    (r"1111:1111::2222", True),
    (r"1111::2222:2222", True),
    (r"1111:1111:1111::2222", True),
    (r"1111:1111::2222:2222", True),
    (r"1111::2222:2222:2222", True),
    (r"1111:1111:1111:1111::2222", True),
    (r"1111:1111:1111::2222:2222", True),
    (r"1111:1111::2222:2222:2222", True),
    (r"1111::2222:2222:2222:2222", True),
    (r"1111:1111:1111:1111:1111::2222", True),
    (r"1111:1111:1111:1111::2222:2222", True),
    (r"1111:1111:1111::2222:2222:2222", True),
    (r"1111:1111::2222:2222:2222:2222", True),
    (r"1111::2222:2222:2222:2222:2222", True),
    (r"1111:1111:1111:1111:1111:1111::2222", True),
    (r"1111:1111:1111:1111:1111::2222:2222", True),
    (r"1111:1111:1111:1111::2222:2222:2222", True),
    (r"1111:1111:1111::2222:2222:2222:2222", True),
    (r"1111:1111::2222:2222:2222:2222:2222", True),
    (r"1111::2222:2222:2222:2222:2222:2222", True),
    (r"2001:db8:3:4::192.0.2.33", True),
    (r"64:ff9b::192.0.2.33", True),
    (r"::ffff:0:255.255.255.255", True),
    (r"::111.111.222.222", True),
    (r":", False),
    (r":::", False),
    (r"::5555:5555:5555:5555:5555:5555:5555:5555", False),
    (r"5555::5555:5555:5555:5555:5555:5555:5555", False),
    (r"5555:5555::5555:5555:5555:5555:5555:5555", False),
    (r"5555:5555:5555::5555:5555:5555:5555:5555", False),
    (r"5555:5555:5555:5555::5555:5555:5555:5555", False),
    (r"5555:5555:5555:5555:5555::5555:5555:5555", False),
    (r"5555:5555:5555:5555:5555:5555::5555:5555", False),
    (r"5555:5555:5555:5555:5555:5555:5555::5555", False),
    (r"5555:5555:5555:5555:5555:5555:5555:5555::", False),
]


@pytest.mark.parametrize("instance, accepted", instance__accepted__test_ipv6_format)
def test_ipv6_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "ipv6"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" ( ( [0-9a-fA-F]{1,4} ":" ){7,7} [0-9a-fA-F]{1,4} | ( [0-9a-fA-F]{1,4} ":" ){1,7} ":" | ( [0-9a-fA-F]{1,4} ":" ){1,6} ":" [0-9a-fA-F]{1,4} | ( [0-9a-fA-F]{1,4} ":" ){1,5} ( ":" [0-9a-fA-F]{1,4} ){1,2} | ( [0-9a-fA-F]{1,4} ":" ){1,4} ( ":" [0-9a-fA-F]{1,4} ){1,3} | ( [0-9a-fA-F]{1,4} ":" ){1,3} ( ":" [0-9a-fA-F]{1,4} ){1,4} | ( [0-9a-fA-F]{1,4} ":" ){1,2} ( ":" [0-9a-fA-F]{1,4} ){1,5} | [0-9a-fA-F]{1,4} ":" ( ( ":" [0-9a-fA-F]{1,4} ){1,6} ) | ":" ( ( ":" [0-9a-fA-F]{1,4} ){1,7} | ":" ) | ":" ":" ( "f" "f" "f" "f" ( ":" "0"{1,4} ){0,1} ":" ){0,1} ( ( "2" "5" [0-5] | ( "2" [0-4] | "1"{0,1} [0-9] ){0,1} [0-9] ) "." ){3,3} ( "2" "5" [0-5] | ( "2" [0-4] | "1"{0,1} [0-9] ){0,1} [0-9] ) | ( [0-9a-fA-F]{1,4} ":" ){1,4} ":" ( ( "2" "5" [0-5] | ( "2" [0-4] | "1"{0,1} [0-9] ){0,1} [0-9] ) "." ){3,3} ( "2" "5" [0-5] | ( "2" [0-4] | "1"{0,1} [0-9] ){0,1} [0-9] ) ) "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


instance__accepted__test_ipv4_format = [
    # (r"0.0.0.0", True),
    (r"00.00.00.00", True),
    (r"000.000.000.000", True),
    (r"255.255.255.255", True),
    (r"1", False),
    (r"1.", False),
    (r"1.1", False),
    (r"1.1.", False),
    (r"1.1.1", False),
    (r"1.1.1.", False),
    (r"0001.0001.0001.0001", False),
    (r"256.256.256.256", False),
]


@pytest.mark.parametrize("instance, accepted", instance__accepted__test_ipv4_format)
def test_ipv4_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "ipv4"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" ( ( "2" "5" [0-5] | "2" [0-4] [0-9] | [0-1]? [0-9]? [0-9] ) "." ){3} ( "2" "5" [0-5] | "2" [0-4] [0-9] | [0-1]? [0-9]? [0-9] ) "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


instance__accepted__test_hostname_format = [
    (r"0", True),
    (r"9", True),
    (r"a", True),
    (r"z", True),
    (r"www.github.com", True),
    (r"w-w-w.g-i-t-h-u-b.c-o-m", True),
    (r"ww-w.gi-th-ub.co-m", True),
    (r"w--ww.git---hub.co----m", True),
    (r".", False),
    (r"-", False),
    (r"-.", False),
    (r".-", False),
    (r"_", False),
    (r"a.", False),
    (r"-b", False),
    (r"c-", False),
    (r"d.-", False),
    (r"e-.", False),
    (r"-f.", False),
    (r"g-.h", False),
    (r"-i.j", False),
]


@pytest.mark.parametrize("instance, accepted", instance__accepted__test_hostname_format)
def test_hostname_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "hostname"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" ( [a-z0-9] ( [a-z0-9-]* [a-z0-9] )? ) ( "." [a-z0-9] ( [a-z0-9-]* [a-z0-9] )? )* "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


instance__accepted__test_uuid_format = [
    (r"00000000-0000-0000-0000-000000000000", True),
    (r"FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF", True),
    (r"01234567-89AB-CDEF-abcd-ef0123456789", True),
    (r"-", False),
    (r"----", False),
    (r"AAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA", False),
    (r"BBBBBBBB-BBB-BBBB-BBBB-BBBBBBBBBBBB", False),
    (r"CCCCCCCC-CCCC-CCC-CCCC-CCCCCCCCCCCC", False),
    (r"DDDDDDDD-DDDD-DDDD-DDD-DDDDDDDDDDDD", False),
    (r"EEEEEEEE-EEEE-EEEE-EEEE-EEEEEEEEEEE", False),
    (r"AAAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA", False),
    (r"BBBBBBBB-BBBBB-BBBB-BBBB-BBBBBBBBBBBB", False),
    (r"CCCCCCCC-CCCC-CCCCC-CCCC-CCCCCCCCCCCC", False),
    (r"DDDDDDDD-DDDD-DDDD-DDDDD-DDDDDDDDDDDD", False),
    (r"EEEEEEEE-EEEE-EEEE-EEEE-EEEEEEEEEEEEE", False),
]


@pytest.mark.parametrize("instance, accepted", instance__accepted__test_uuid_format)
def test_uuid_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "uuid"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" [0-9A-Fa-f]{8} "-" [0-9A-Fa-f]{4} "-" [0-9A-Fa-f]{4} "-" [0-9A-Fa-f]{4} "-" [0-9A-Fa-f]{12} "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


instance__accepted__test_uri_format = [
    (r"aaa:?azAZ09-._~%Ff!$&'()*+,;=:@#azAZ09-._~%Aa!$&'()*+,;=:@", True),
    (r"z+.-:", True),
    (r"abc:", True),
    (r"abc:a", True),
    (r"abc:/", True),
    (r"abc:/a", True),
    (r"abc://", True),
    (r"abc://///////", True),
    (r"abc://azAZ09-._~%Ff!$&'()*+,;=:@", True),
    (r"abc://:", True),
    (r"abc://:0123", True),
    (r"abc://azAZ09-._~%Ff!$&'()*+,;=", True),
    (r"xyz:/a", True),
    (r"xyz:/azAZ09-._~%Ff!$&'()*+,;=:@", True),
    (r"aaa:?[#]", False),
    (r"abc://@@", False),
    (r"abc://::", False),
    (r"abc:/[]", False),
]


@pytest.mark.parametrize("instance, accepted", instance__accepted__test_uri_format)
def test_uri_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "uri"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" [a-zA-Z] [a-zA-Z+.-]* ":" ( "/" "/" ( ( [a-zA-Z0-9_.~!$&'()*+,;=:-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* "@" )? ( [a-zA-Z0-9_.~!$&'()*+,;=-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* ( ":" [0-9]* )? ( "/" ( [a-zA-Z0-9_.~!$&'()*+,;=:@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )* | "/"? ( ( [a-zA-Z0-9_.~!$&'()*+,;=:@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )+ ( "/" ( [a-zA-Z0-9_.~!$&'()*+,;=:@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )* )? ) ( "\?" ( [a-zA-Z0-9_.~!$&'()*+,;=:@/\?-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )? ( "#" ( [a-zA-Z0-9_.~!$&'()*+,;=:@/\?-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )? "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


instance__accepted__test_uri_reference_format = [
    (r"?azAZ09-._~%Ff!$&'()*+,;=:@#azAZ09-._~%Aa!$&'()*+,;=:@", True),
    (r"", True),
    (r"a", True),
    (r"/", True),
    (r"/a", True),
    (r"//", True),
    (r"/////////", True),
    (r"//azAZ09-._~%Ff!$&'()*+,;=:@", True),
    (r"//:", True),
    (r"//:0123", True),
    (r"//azAZ09-._~%Ff!$&'()*+,;=", True),
    (r"/a", True),
    (r"/azAZ09-._~%Ff!$&'()*+,;=:@", True),
    (r"?[#]", False),
    (r"//@@", False),
    (r"//::", False),
    (r"/[]", False),
    (r":", False),
]


@pytest.mark.parametrize("instance, accepted", instance__accepted__test_uri_reference_format)
def test_uri_reference_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "uri-reference"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" ( "/" "/" ( ( [a-zA-Z0-9_.~!$&'()*+,;=:-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* "@" )? ( [a-zA-Z0-9_.~!$&'()*+,;=-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* ( ":" [0-9]* )? ( "/" ( [a-zA-Z0-9_.~!$&'()*+,;=:@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )* | "/" ( ( [a-zA-Z0-9_.~!$&'()*+,;=:@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )+ ( "/" ( [a-zA-Z0-9_.~!$&'()*+,;=:@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )* )? | ( [a-zA-Z0-9_.~!$&'()*+,;=@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )+ ( "/" ( [a-zA-Z0-9_.~!$&'()*+,;=:@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )* )? ( "\?" ( [a-zA-Z0-9_.~!$&'()*+,;=:@/\?-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )? ( "#" ( [a-zA-Z0-9_.~!$&'()*+,;=:@/\?-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )? "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


instance__accepted__test_uri_template_format = [
    (r"", True),
    (r"!#$&()*+,-./09:;=?@AZ[]_az~%Ff", True),
    (r"{+a}{#a}{.a}{/a}{;a}{?a}{&a}{=a}{,a}{!a}{@a}{|a}", True),
    (r"{%Ff}", True),
    (r"{i.j.k}", True),
    (r"{a_b_c:1234}", True),
    (r"{x_y_z*}", True),
    (r'"', False),
    (r"'", False),
    (r"%", False),
    (r"<", False),
    (r">", False),
    (r"\\\\", False),
    (r"^", False),
    (r"`", False),
    (r"{", False),
    (r"|", False),
    (r"}", False),
    (r"{n.}", False),
    (r"{m:100001}", False),
    (r"%1", False),
    (r"%Gg", False),
]


@pytest.mark.parametrize("instance, accepted", instance__accepted__test_uri_template_format)
def test_uri_template_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "uri-template"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" ( ( [!#-$&(-;=\?-[\]_a-z~] | "%" [0-9A-Fa-f] [0-9A-Fa-f] ) | "{" ( [+#./;\?&=,!@|] )? ( [a-zA-Z0-9_] | "%" [0-9A-Fa-f] [0-9A-Fa-f] ) ( "."? ( [a-zA-Z0-9_] | "%" [0-9A-Fa-f] [0-9A-Fa-f] ) )* ( ":" [1-9] [0-9]? [0-9]? [0-9]? | "*" )? ( "," ( [a-zA-Z0-9_] | "%" [0-9A-Fa-f] [0-9A-Fa-f] ) ( "."? ( [a-zA-Z0-9_] | "%" [0-9A-Fa-f] [0-9A-Fa-f] ) )* ( ":" [1-9] [0-9]? [0-9]? [0-9]? | "*" )? )* "}" )* "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


instance__accepted__test_json_pointer_format = [
    (r"/", True),
    (r"//", True),
    (r"/a/bc/def/ghij", True),
    (r"/~0/~1/", True),
    (r"abc", False),
    (r"/~", False),
    (r"/~2", False),
]


@pytest.mark.parametrize("instance, accepted", instance__accepted__test_json_pointer_format)
def test_json_pointer_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "json-pointer"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" ( "/" ( [\0-.] | [0-}] | [\x7f-\U0010ffff] | "~" [01] )* )* "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


instance__accepted__test_relative_json_pointer_format = [
    (r"0/", True),
    (r"123/a/bc/def/ghij", True),
    (r"45/~0/~1/", True),
    (r"6789#", True),
    (r"#", False),
    (r"abc", False),
    (r"/", False),
    (r"9/~2", False),
]


@pytest.mark.parametrize(
    "instance, accepted", instance__accepted__test_relative_json_pointer_format
)
def test_relative_json_pointer_format(instance: str, accepted: bool):
    schema = {"type": "string", "format": "relative-json-pointer"}

    expected_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" ( "0" | [1-9] [0-9]* ) ( "#" | ( "/" ( [\0-.] | [0-}] | [\x7f-\U0010ffff] | "~" [01] )* )* ) "\""
root ::= string
"""
    )
    check_schema_with_grammar(schema, expected_grammar)

    check_schema_with_instance(schema, '"' + instance + '"', is_accepted=accepted)


def test_min_max_length():
    schema = {"type": "string", "minLength": 1, "maxLength": 10}

    ebnf_grammar = basic_json_rules_ebnf + (
        r"""string ::= "\"" [^"\\\r\n]{1,10} "\""
root ::= string
"""
    )

    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=True)

    instance_accepted = '"abcdefghij"'
    instance_rejected = '"abcdefghijk"'

    check_schema_with_instance(schema, instance_accepted, any_whitespace=True)
    check_schema_with_instance(schema, instance_rejected, is_accepted=False, any_whitespace=True)


def test_type_array():
    schema = {
        "type": ["integer", "string"],
        "minLength": 1,
        "maxLength": 10,
        "minimum": 1,
        "maximum": 10,
    }

    ebnf_grammar = basic_json_rules_ebnf + (
        r"""root_integer ::= ( ( [1-9] | "1" "0" ) )
string ::= "\"" [^"\\\r\n]{1,10} "\""
root_string ::= string
root ::= root_integer | root_string
"""
    )

    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=True)

    instance_accepted = "1"
    instance_accepted_2 = '"1234567890"'
    instance_rejected = "11"
    instance_rejected_2 = '"12345678901"'

    check_schema_with_instance(schema, instance_accepted, any_whitespace=True)
    check_schema_with_instance(schema, instance_accepted_2, any_whitespace=True)
    check_schema_with_instance(schema, instance_rejected, is_accepted=False, any_whitespace=True)
    check_schema_with_instance(schema, instance_rejected_2, is_accepted=False, any_whitespace=True)


def test_type_array_empty():
    schema = {"type": []}

    ebnf_grammar = basic_json_rules_ebnf + (
        r"""root ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
"""
    )

    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=True)


def test_empty_array():
    schema = {"items": {"type": "string"}, "type": "array"}

    ebnf_grammar = basic_json_rules_ebnf + (
        r"""root ::= (("[" [ \n\t]* basic_string ([ \n\t]* "," [ \n\t]* basic_string)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
"""
    )

    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=True)

    instance_accepted = "[]"
    instance_accepted_2 = '["a"]'

    check_schema_with_instance(schema, instance_accepted, any_whitespace=True)
    check_schema_with_instance(schema, instance_accepted_2, any_whitespace=True)


def test_empty_object():
    schema = {"properties": {"name": {"type": "string"}}, "type": "object"}

    ebnf_grammar = basic_json_rules_ebnf + (
        r"""root ::= ("{" [ \n\t]* (("\"name\"" [ \n\t]* ":" [ \n\t]* basic_string "")) [ \n\t]* "}") | "{" [ \n\t]* "}"
"""
    )

    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=True)

    instance_accepted = "{}"
    instance_accepted_2 = '{"name": "test"}'

    check_schema_with_instance(schema, instance_accepted, any_whitespace=True)
    check_schema_with_instance(schema, instance_accepted_2, any_whitespace=True)


def test_primitive_type_string():
    schema = {"type": "string"}
    ebnf_grammar = basic_json_rules_ebnf + (
        r"""root ::= basic_string
"""
    )

    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=True)

    instance_accepted = '"test"'
    instance_rejected = "123"

    check_schema_with_instance(schema, instance_accepted, any_whitespace=True)
    check_schema_with_instance(schema, instance_rejected, is_accepted=False, any_whitespace=True)


def test_primitive_type_object():
    schema = {"type": "object"}
    ebnf_grammar = basic_json_rules_ebnf + (
        r"""root ::= basic_object
"""
    )

    check_schema_with_grammar(schema, ebnf_grammar, any_whitespace=True)

    instance_accepted = '{"name": "test"}'
    instance_rejected = '"test"'

    check_schema_with_instance(schema, instance_accepted, any_whitespace=True)
    check_schema_with_instance(schema, instance_rejected, is_accepted=False, any_whitespace=True)


def test_generate_float_regex():
    assert _generate_float_regex(1.0, 5.0) == r"^(1|5|(([2-4]))(\.\d{1,6})?|1\.\d{1,6}|5\.\d{1,6})$"

    assert (
        _generate_float_regex(1.5, 5.75)
        == r"^(1.5|5.75|(([2-4]))(\.\d{1,6})?|1\.6\d{0,5}|1\.7\d{0,5}|1\.8\d{0,5}|1\.9\d{0,5}|5\.0\d{0,5}|5\.1\d{0,5}|5\.2\d{0,5}|5\.3\d{0,5}|5\.4\d{0,5}|5\.5\d{0,5}|5\.6\d{0,5}|5\.70\d{0,4}|5\.71\d{0,4}|5\.72\d{0,4}|5\.73\d{0,4}|5\.74\d{0,4})$"
    )

    assert (
        _generate_float_regex(-3.14, 2.71828)
        == r"^(-3.14|2.71828|(-([1-3])|0|(1))(\.\d{1,6})?|-3\.0\d{0,5}|-3\.10\d{0,4}|-3\.11\d{0,4}|-3\.12\d{0,4}|-3\.13\d{0,4}|2\.0\d{0,5}|2\.1\d{0,5}|2\.2\d{0,5}|2\.3\d{0,5}|2\.4\d{0,5}|2\.5\d{0,5}|2\.6\d{0,5}|2\.70\d{0,4}|2\.710\d{0,3}|2\.711\d{0,3}|2\.712\d{0,3}|2\.713\d{0,3}|2\.714\d{0,3}|2\.715\d{0,3}|2\.716\d{0,3}|2\.717\d{0,3}|2\.7180\d{0,2}|2\.7181\d{0,2}|2\.71820\d{0,1}|2\.71821\d{0,1}|2\.71822\d{0,1}|2\.71823\d{0,1}|2\.71824\d{0,1}|2\.71825\d{0,1}|2\.71826\d{0,1}|2\.71827\d{0,1})$"
    )

    assert (
        _generate_float_regex(0.5, None)
        == r"^(0.5|0\.6\d{0,5}|0\.7\d{0,5}|0\.8\d{0,5}|0\.9\d{0,5}|([1-9]|[1-9]\d*)(\.\d{1,6})?)$"
    )

    assert (
        _generate_float_regex(None, -1.5)
        == r"^(-1.5|-1\.6\d{0,5}|-1\.7\d{0,5}|-1\.8\d{0,5}|-1\.9\d{0,5}|(-[3-9]|-[1-9]\d*)(\.\d{1,6})?)$"
    )

    assert _generate_float_regex(None, None) == r"^-?\d+(\.\d{1,6})?$"

    assert _generate_float_regex(3.14159, 3.14159) == r"^(3.14159)$"

    assert _generate_float_regex(10.5, 2.5) == r"^()$"

    assert _generate_float_regex(5.123456, 5.123457) == r"^(5.123456|5.123457)$"

    assert (
        _generate_float_regex(-0.000001, 0.000001)
        == r"^(-0.000001|0.000001|-0\.000000\d{0,0}|0\.000000\d{0,0})$"
    )


def test_limited_whitespace_cnt():
    expected_grammar = r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(basic_string_sub))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=(basic_string_sub_1 [,}\]:]))
basic_string ::= (("\"" basic_string_sub)) (=(root_4 "}"))
root ::= (("{" root_1 "\"key\"" root_2 ":" root_3 basic_string root_4 "}"))
basic_string_sub_1 ::= ("" | ([ \n\t]) | ([ \n\t] [ \n\t]))
root_1 ::= ("" | ([ \n\t]) | ([ \n\t] [ \n\t])) (=("\"key\"" root_2 ":" root_3 basic_string root_4 "}"))
root_2 ::= ("" | ([ \n\t]) | ([ \n\t] [ \n\t])) (=(":" root_3 basic_string root_4 "}"))
root_3 ::= ("" | ([ \n\t]) | ([ \n\t] [ \n\t])) (=(basic_string root_4 "}"))
root_4 ::= ("" | ([ \n\t]) | ([ \n\t] [ \n\t])) (=("}"))
"""
    schema = {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}
    grammar = xgr.Grammar.from_json_schema(schema, any_whitespace=True, max_whitespace_cnt=2)
    grammar = GrammarFunctor.grammar_optimizer(grammar)
    assert grammar is not None
    assert str(grammar) == expected_grammar
    assert _is_grammar_accept_string(grammar, '{  "key"  :  "value"  }')
    assert _is_grammar_accept_string(grammar, '{"key":"value"}')
    assert not _is_grammar_accept_string(grammar, '{   "key"  :  "value"   }')
    assert not _is_grammar_accept_string(grammar, '{    "key"  :  "value"    }')


def test_limited_whitespace_compile():
    expected_grammar = r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(basic_string_sub))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=(basic_string_sub_1 [,}\]:]))
basic_string ::= (("\"" basic_string_sub)) (=(root_4 "}"))
root ::= (("{" root_1 "\"key\"" root_2 ":" root_3 basic_string root_4 "}"))
basic_string_sub_1 ::= ("" | ([ \n\t]) | ([ \n\t] [ \n\t]))
root_1 ::= ("" | ([ \n\t]) | ([ \n\t] [ \n\t])) (=("\"key\"" root_2 ":" root_3 basic_string root_4 "}"))
root_2 ::= ("" | ([ \n\t]) | ([ \n\t] [ \n\t])) (=(":" root_3 basic_string root_4 "}"))
root_3 ::= ("" | ([ \n\t]) | ([ \n\t] [ \n\t])) (=(basic_string root_4 "}"))
root_4 ::= ("" | ([ \n\t]) | ([ \n\t] [ \n\t])) (=("}"))
"""
    schema = {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}
    tokenizer_info = xgr.TokenizerInfo([])
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_json_schema(
        schema, any_whitespace=True, max_whitespace_cnt=2
    )
    assert compiled_grammar is not None
    grammar = compiled_grammar.grammar
    assert str(grammar) == expected_grammar
    assert grammar is not None
    assert _is_grammar_accept_string(grammar, '{  "key"  :  "value"  }')
    assert _is_grammar_accept_string(grammar, '{"key":"value"}')
    assert not _is_grammar_accept_string(grammar, '{   "key"  :  "value"   }')
    assert not _is_grammar_accept_string(grammar, '{    "key"  :  "value"    }')


def test_utf8_in_enum():
    schema = {"type": "string", "enum": ["こんにちは", "😊", "你好", "hello", "\n"]}
    grammar = xgr.Grammar.from_json_schema(schema)
    assert _is_grammar_accept_string(grammar, '"こんにちは"')
    assert _is_grammar_accept_string(grammar, '"😊"')
    assert _is_grammar_accept_string(grammar, '"你好"')
    assert _is_grammar_accept_string(grammar, '"hello"')
    assert _is_grammar_accept_string(grammar, '"\\n"')


def test_utf8_string_in_const():
    schema = {"const": "常数constじょうすう\n\r\t"}
    grammar = xgr.Grammar.from_json_schema(schema)
    assert _is_grammar_accept_string(grammar, '"常数constじょうすう\\n\\r\\t"')


def test_utf8_object_array_in_enum():
    schema = {
        "type": "object",
        "enum": [
            {"key": "こんにちは"},
            {"key": "😊"},
            {"key": "你好"},
            {"key": "hello"},
            {"key": "\n"},
            [123, "こんにちは", "😊", "你好", "hello", "\n"],
        ],
    }
    grammar = xgr.Grammar.from_json_schema(schema)
    assert _is_grammar_accept_string(grammar, '{"key":"こんにちは"}')
    assert _is_grammar_accept_string(grammar, '{"key":"😊"}')
    assert _is_grammar_accept_string(grammar, '{"key":"你好"}')
    assert _is_grammar_accept_string(grammar, '{"key":"hello"}')
    assert _is_grammar_accept_string(grammar, '{"key":"\\n"}')
    assert _is_grammar_accept_string(grammar, '[123,"こんにちは","😊","你好","hello","\\n"]')


def test_utf8_object_const():
    schema = {"type": "object", "const": {"key": "こんにちは常数constじょうすう\n\r\t"}}
    grammar = xgr.Grammar.from_json_schema(schema)
    assert _is_grammar_accept_string(grammar, '{"key":"こんにちは常数constじょうすう\\n\\r\\t"}')


def test_utf8_array_const():
    schema = {"type": "array", "const": ["こんにちは", "😊", "你好", "hello", "\n"]}
    grammar = xgr.Grammar.from_json_schema(schema)
    assert _is_grammar_accept_string(grammar, '["こんにちは","😊","你好","hello","\\n"]')


if __name__ == "__main__":
    pytest.main(sys.argv)
