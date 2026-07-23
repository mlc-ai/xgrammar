import json
from typing import Any

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


def _accepts(schema: dict[str, Any], text: str, **kwargs: Any) -> bool:
    grammar = xgr.Grammar.from_json_schema(json.dumps(schema), **kwargs)
    return _is_grammar_accept_string(grammar, text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"name":"ab","count":0}', True),
        ('{"name": "abcd", "count": -2, "choice": "a", "nullable": null}', True),
        ('{"name":"a","count":0}', False),
        ('{"count":0,"name":"ab"}', False),
        ('{"name":"ab","count":9}', False),
        ('{"name":"ab","count":0,"extra":true}', False),
    ],
)
def test_direct_converter_basic_types_and_compositions(text: str, expected: bool):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 2, "maxLength": 4},
            "count": {"type": "integer", "minimum": -2, "maximum": 8},
            "choice": {"enum": ["a", "b", 3]},
            "nullable": {"type": ["string", "null"]},
        },
        "required": ["name", "count"],
        "additionalProperties": False,
    }
    assert _accepts(schema, text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ('["header",{"id":3,"code":"AB12"}]', True),
        ('["header", {"id": 6, "code": "XY9"}, {"id": -3, "code": "AA0"}]', True),
        ('["header"]', False),
        ('["header",{"id":4,"code":"AB12"}]', False),
        ('["header",{"id":3,"code":"ab12"}]', False),
    ],
)
def test_direct_converter_arrays_references_and_patterns(text: str, expected: bool):
    schema = {
        "$defs": {
            "entry": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "multipleOf": 3},
                    "code": {"type": "string", "pattern": "^[A-Z]{2}[0-9]+$"},
                },
                "required": ["id", "code"],
                "additionalProperties": False,
            }
        },
        "type": "array",
        "prefixItems": [{"const": "header"}],
        "items": {"$ref": "#/$defs/entry"},
        "minItems": 2,
        "maxItems": 3,
    }
    assert _accepts(schema, text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"fixed":true}', True),
        ('{"x_a":2}', True),
        ('{"other":"value"}', True),
        ('{"fixed":false,"x_test":3,"other":"value"}', True),
        ("{}", False),
        ('{"fixed":true,"x_a":1,"one":"1","two":"2"}', False),
    ],
)
def test_direct_converter_property_constraints_and_additional_keys(text: str, expected: bool):
    schema = {
        "type": "object",
        "properties": {"fixed": {"type": "boolean"}, "optional": {"type": "number"}},
        "patternProperties": {"^x_[a-z]+$": {"type": "integer"}},
        "additionalProperties": {"type": "string"},
        "minProperties": 1,
        "maxProperties": 3,
    }
    assert _accepts(schema, text) is expected


def test_direct_converter_formatting_and_any_order():
    schema = {
        "type": "object",
        "properties": {"first": {"type": "string"}, "second": {"type": "integer"}},
        "required": ["first", "second"],
        "additionalProperties": False,
    }
    options = {"any_whitespace": False, "indent": 2, "strict_mode": True, "any_order": True}
    assert _accepts(schema, '{\n  "first": "x",\n  "second": 1\n}', **options)
    assert _accepts(schema, '{\n  "second": 1,\n  "first": "x"\n}', **options)
    assert not _accepts(schema, '{"first":"x","second":1}', **options)

    bounded = {"any_whitespace": True, "max_whitespace_cnt": 3}
    assert _accepts(schema, '{"first":"x","second":1}', **bounded)
    assert _accepts(schema, '{"first": "x", "second": 1}', **bounded)
    assert not _accepts(schema, '{    "first":"x","second":1}', **bounded)


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"value":1,"next":null}', True),
        ('{"value":1,"next":{"value":2,"next":null}}', True),
        ('{"value":1}', False),
        ('{"value":1,"next":{"value":"bad","next":null}}', False),
    ],
)
def test_direct_converter_recursive_reference(text: str, expected: bool):
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "value": {"type": "integer"},
            "next": {"anyOf": [{"$ref": "#"}, {"type": "null"}]},
        },
        "required": ["value", "next"],
        "additionalProperties": False,
    }
    assert _accepts(schema, text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ('"user@example.com"', True),
        ('"first.last+tag@example.co.uk"', True),
        ('"not-an-email"', False),
        ('"@example.com"', False),
    ],
)
def test_direct_converter_builtin_string_format(text: str, expected: bool):
    assert _accepts({"type": "string", "format": "email"}, text) is expected


def test_direct_converter_reuses_identical_schema_rules():
    repeated_schema = {"type": "string", "minLength": 2, "maxLength": 8}
    schema = {
        "type": "object",
        "properties": {"first": repeated_schema, "second": repeated_schema},
        "required": ["first", "second"],
        "additionalProperties": False,
    }

    grammar_text = str(xgr.Grammar.from_json_schema(json.dumps(schema)))
    assert "root_prop_0 ::=" in grammar_text
    assert "root_prop_1 ::=" not in grammar_text


def test_direct_converter_reuses_cached_reference_targets():
    repeated_schema = {"type": "string", "minLength": 2, "maxLength": 8}
    schema = {
        "$defs": {"shared": repeated_schema},
        "type": "object",
        "properties": {"inline": repeated_schema, "referenced": {"$ref": "#/$defs/shared"}},
        "required": ["inline", "referenced"],
        "additionalProperties": False,
    }

    grammar = xgr.Grammar.from_json_schema(json.dumps(schema))
    assert _is_grammar_accept_string(grammar, '{"inline":"ab","referenced":"cd"}')
    assert "defs_shared ::=" not in str(grammar)


def test_direct_converter_keeps_indented_rules_at_different_depths_separate():
    repeated_schema = {
        "type": "object",
        "properties": {"value": {"type": "string", "minLength": 2}},
        "required": ["value"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "first": repeated_schema,
            "wrapper": {
                "type": "object",
                "properties": {"second": repeated_schema},
                "required": ["second"],
                "additionalProperties": False,
            },
        },
        "required": ["first", "wrapper"],
        "additionalProperties": False,
    }
    text = (
        "{\n"
        '  "first": {\n'
        '    "value": "ab"\n'
        "  },\n"
        '  "wrapper": {\n'
        '    "second": {\n'
        '      "value": "cd"\n'
        "    }\n"
        "  }\n"
        "}"
    )

    grammar = xgr.Grammar.from_json_schema(json.dumps(schema), any_whitespace=False, indent=2)
    assert _is_grammar_accept_string(grammar, text)
