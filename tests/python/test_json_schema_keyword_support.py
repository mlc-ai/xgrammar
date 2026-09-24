"""Per-keyword JSON Schema support, checked against the reference implementation.

Answers the question raised in #104 (which JSON Schema keywords are actually
supported) in a form that cannot silently go stale: the table in
`docs/defining_structures/json_generation.md` is asserted here, so a change in
converter behaviour fails a test rather than leaving the docs wrong.

A keyword is `ENFORCED` when the compiled grammar accepts the schema-valid
instance and rejects the schema-invalid one. It is `NOT_ENFORCED` when the
grammar accepts the schema-invalid instance -- in that direction the grammar
is more permissive than the schema, and generation can produce output that
fails schema validation.

Every fixture below was checked against the `jsonschema` reference
implementation (4.26.0) when written: each `accepted` value validates against
its schema and each `rejected` value does not. The expectations are inlined so
this test needs no extra dependency at runtime.
"""

import json
from typing import Any, Dict

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

ENFORCED = "enforced"
NOT_ENFORCED = "not_enforced"


# (keyword, schema, accepted instance, rejected instance, expected status)
KEYWORD_CASES = [
    ("type:string", {"type": "string"}, "a", 1, ENFORCED),
    ("type:integer", {"type": "integer"}, 1, 1.5, ENFORCED),
    ("type:number", {"type": "number"}, 1.5, "x", ENFORCED),
    ("type:boolean", {"type": "boolean"}, True, "true", ENFORCED),
    ("type:null", {"type": "null"}, None, 0, ENFORCED),
    ("enum", {"type": "string", "enum": ["a", "b"]}, "a", "z", ENFORCED),
    ("const", {"const": "k"}, "k", "j", ENFORCED),
    ("minLength", {"type": "string", "minLength": 3}, "abcd", "a", ENFORCED),
    ("maxLength", {"type": "string", "maxLength": 3}, "ab", "abcdef", ENFORCED),
    ("pattern", {"type": "string", "pattern": "^a+$"}, "aaa", "bbb", ENFORCED),
    (
        "format:date",
        {"type": "string", "format": "date"},
        "2020-01-01",
        "nope",
        ENFORCED,
    ),
    ("minimum:integer", {"type": "integer", "minimum": 10}, 12, 3, ENFORCED),
    ("maximum:integer", {"type": "integer", "maximum": 10}, 5, 99, ENFORCED),
    ("minimum:number", {"type": "number", "minimum": 10}, 12.0, 3.5, ENFORCED),
    ("maximum:number", {"type": "number", "maximum": 10}, 5.0, 99.5, ENFORCED),
    ("exclusiveMinimum", {"type": "integer", "exclusiveMinimum": 10}, 11, 10, ENFORCED),
    ("exclusiveMaximum", {"type": "integer", "exclusiveMaximum": 10}, 9, 10, ENFORCED),
    ("multipleOf:integer", {"type": "integer", "multipleOf": 5}, 10, 7, ENFORCED),
    (
        "minItems",
        {"type": "array", "items": {"type": "integer"}, "minItems": 2},
        [1, 2],
        [1],
        ENFORCED,
    ),
    (
        "maxItems",
        {"type": "array", "items": {"type": "integer"}, "maxItems": 2},
        [1, 2],
        [1, 2, 3],
        ENFORCED,
    ),
    (
        "required",
        {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
        {"a": 1},
        {},
        ENFORCED,
    ),
    (
        "additionalProperties:false",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": False,
        },
        {"a": 1},
        {"a": 1, "b": 2},
        ENFORCED,
    ),
    ("anyOf", {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}, 1, "s", ENFORCED),
    ("oneOf", {"oneOf": [{"type": "integer"}, {"type": "boolean"}]}, 1, "s", ENFORCED),
    # Keywords the converter does not enforce. Listed explicitly so the
    # documented table and the code cannot drift apart.
    ("multipleOf:number", {"type": "number", "multipleOf": 5}, 10.0, 0.5, NOT_ENFORCED),
    (
        "uniqueItems",
        {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
        [1, 2],
        [1, 1],
        NOT_ENFORCED,
    ),
    ("not", {"not": {"type": "string"}}, 1, "s", NOT_ENFORCED),
]


@pytest.mark.parametrize(
    "keyword, schema, accepted, rejected, expected",
    KEYWORD_CASES,
    ids=[c[0] for c in KEYWORD_CASES],
)
def test_keyword_support(
    keyword: str, schema: Dict[str, Any], accepted: Any, rejected: Any, expected: str
):
    grammar = xgr.Grammar.from_json_schema(json.dumps(schema))
    assert _is_grammar_accept_string(grammar, json.dumps(accepted)), (
        f"{keyword}: grammar rejects a schema-valid instance"
    )

    grammar_accepts_invalid = _is_grammar_accept_string(grammar, json.dumps(rejected))
    if expected == ENFORCED:
        assert not grammar_accepts_invalid, (
            f"{keyword}: grammar accepts a schema-invalid instance"
        )
    else:
        assert grammar_accepts_invalid, (
            f"{keyword}: now enforced -- update the support table in "
            "docs/defining_structures/json_generation.md"
        )


def test_additional_properties_default_is_strict():
    """`strict_mode` (default True) is stricter than the specification.

    The specification permits additional properties when
    `additionalProperties` is omitted; the default grammar does not.
    Documented in docs/defining_structures/json_generation.md.
    """
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    # The specification accepts this instance: additionalProperties defaults
    # to true when omitted.
    instance = json.dumps({"a": 1, "b": 2})

    strict = xgr.Grammar.from_json_schema(json.dumps(schema))
    assert not _is_grammar_accept_string(strict, instance)

    lenient = xgr.Grammar.from_json_schema(json.dumps(schema), strict_mode=False)
    assert _is_grammar_accept_string(lenient, instance)


def test_allof_is_not_enforced():
    """`allOf` branches are not intersected; a contradiction still compiles.

    Tracked in #858. Listed here so the documented table stays accurate.
    """
    # Contradictory: nothing can be both an integer and a string, so the
    # specification accepts no instance at all.
    schema = {"allOf": [{"type": "integer"}, {"type": "string"}]}
    grammar = xgr.Grammar.from_json_schema(json.dumps(schema))
    assert _is_grammar_accept_string(grammar, "1")
