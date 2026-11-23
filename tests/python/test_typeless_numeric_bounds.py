import json

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


@pytest.mark.parametrize("strict_mode", [True, False])
@pytest.mark.parametrize(
    "schema,allowed,rejected",
    [
        ({"minimum": 2}, [2, 3, 2.5], [1, 1.5, -2]),
        ({"maximum": 3}, [3, 2, 2.5], [4, 3.5]),
        ({"exclusiveMinimum": 2}, [3, 2.5], [1, 2]),
        ({"exclusiveMaximum": 3}, [2, 2.5], [3, 4]),
        ({"minimum": 2, "maximum": 2}, [2], [1, 3]),
        ({"minimum": 5, "maximum": 2}, [], [1, 3, 6]),
        ({"minimum": 2, "exclusiveMaximum": 2}, [], [1, 2, 3]),
    ],
)
def test_typeless_numeric_bounds(strict_mode, schema, allowed, rejected):
    grammar = xgr.Grammar.from_json_schema(schema, strict_mode=strict_mode)
    for value in allowed + ["x", True, False, None, [], [1, {"a": 2}], {}, {"a": [1, 2]}]:
        assert _is_grammar_accept_string(grammar, json.dumps(value)), (schema, value)
    for value in rejected:
        assert not _is_grammar_accept_string(grammar, json.dumps(value)), (schema, value)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"value": {"minimum": 2}}, "required": ["value"]},
        {
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/bounded"}},
            "required": ["value"],
            "$defs": {"bounded": {"minimum": 2}},
        },
    ],
)
def test_typeless_numeric_bounds_nested(schema):
    grammar = xgr.Grammar.from_json_schema(schema)
    for value in [2, 3, "x", None, {"a": 1}, [1]]:
        assert _is_grammar_accept_string(grammar, json.dumps({"value": value}))
    assert not _is_grammar_accept_string(grammar, '{"value":1}')


@pytest.mark.parametrize("keyword", ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"])
def test_typeless_numeric_bounds_invalid(keyword):
    with pytest.raises(RuntimeError, match="number"):
        xgr.Grammar.from_json_schema({keyword: "bad"})
