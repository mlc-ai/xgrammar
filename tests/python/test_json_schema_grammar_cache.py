"""Behavior of JSON schema conversion when schemas repeat across conversions.

Converted JSON schema grammars are cached per process and reused by later conversions. These tests
only check observable behavior: repeated conversions give the same grammars and the grammars stay
correct. The cache internals (hits, keys, eviction, budget) are tested in
tests/cpp/test_json_schema_grammar_cache.cc.
"""

import json
import sys
import threading
from typing import Any, Dict, List

import pytest

import xgrammar as xgr
from xgrammar.structural_tag import JSONSchemaFormat, StructuralTag, TagFormat, TriggeredTagsFormat
from xgrammar.testing import GrammarFunctor, _is_grammar_accept_string

ALL_STYLES = [
    "json",
    "qwen_xml",
    "minimax_xml",
    "minimax_m3_xml",
    "deepseek_xml",
    "deepseek_v4_1_xml",
    "glm_xml",
    "cohere_xml",
    "kimi_k3_xml",
    "gemma",
    "muse_xml",
]

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city"],
}

# A recursive schema: a tree whose children are trees, through $defs and $ref.
TREE_SCHEMA = {
    "type": "object",
    "properties": {"root": {"$ref": "#/$defs/node"}},
    "required": ["root"],
    "$defs": {
        "node": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
            },
            "required": ["value"],
        }
    },
}

# Mutual recursion between two definitions, plus a definition used by both.
MUTUAL_SCHEMA = {
    "type": "object",
    "properties": {"expr": {"$ref": "#/$defs/expr"}},
    "required": ["expr"],
    "$defs": {
        "expr": {
            "anyOf": [
                {"$ref": "#/$defs/leaf"},
                {
                    "type": "object",
                    "properties": {"op": {"type": "string"}, "args": {"$ref": "#/$defs/args"}},
                    "required": ["op", "args"],
                },
            ]
        },
        "args": {"type": "array", "items": {"$ref": "#/$defs/expr"}, "minItems": 1},
        "leaf": {"type": "object", "properties": {"num": {"type": "number"}}, "required": ["num"]},
    },
}

ALL_SCHEMAS = [SIMPLE_SCHEMA, TREE_SCHEMA, MUTUAL_SCHEMA]


def tools_stag(tools: List[Dict[str, Any]], **format_kwargs) -> StructuralTag:
    """A structural tag that lets the model call any of the tools, like the builtin templates do."""
    return StructuralTag(
        format=TriggeredTagsFormat(
            triggers=["<tool="],
            tags=[
                TagFormat(
                    begin=f"<tool={t['name']}>",
                    content=JSONSchemaFormat(json_schema=t["schema"], **format_kwargs),
                    end="</tool>",
                )
                for t in tools
            ],
        )
    )


def stag_to_str(stag: StructuralTag) -> str:
    return str(xgr.Grammar.from_structural_tag(stag))


@pytest.mark.parametrize("style", ALL_STYLES)
@pytest.mark.parametrize("schema_index", range(len(ALL_SCHEMAS)))
def test_repeated_conversion_is_identical(style: str, schema_index: int):
    stag = tools_stag([{"name": "f", "schema": ALL_SCHEMAS[schema_index]}], style=style)
    assert stag_to_str(stag) == stag_to_str(stag)


def test_growing_tool_list_keeps_earlier_tools_identical():
    """Deferred tool loading: each request repeats the previous tools and adds one."""
    tools = [{"name": f"tool_{i}", "schema": s} for i, s in enumerate(ALL_SCHEMAS)]
    for n in range(1, len(tools) + 1):
        grammar = stag_to_str(tools_stag(tools[:n]))
        assert grammar == stag_to_str(tools_stag(tools[:n]))


def test_recursive_schemas_stay_correct_on_repeat():
    stag = tools_stag(
        [{"name": "tree", "schema": TREE_SCHEMA}, {"name": "m", "schema": MUTUAL_SCHEMA}]
    )
    tree = {"root": {"value": "a", "children": [{"value": "b", "children": [{"value": "c"}]}]}}
    expr = {"expr": {"op": "add", "args": [{"num": 1}, {"op": "neg", "args": [{"num": 2}]}]}}
    accepted = [
        "<tool=tree>" + json.dumps(tree) + "</tool>",
        "<tool=m>" + json.dumps(expr) + "</tool>",
    ]
    rejected = [
        "<tool=tree>" + json.dumps({"root": {"children": []}}) + "</tool>",
        "<tool=m>" + json.dumps({"expr": {"op": "add", "args": []}}) + "</tool>",
    ]
    for _ in range(2):
        grammar = xgr.Grammar.from_structural_tag(stag)
        for text in accepted:
            assert _is_grammar_accept_string(grammar, text)
        for text in rejected:
            assert not _is_grammar_accept_string(grammar, text)


def test_same_schema_twice_in_one_tag():
    stag = tools_stag([{"name": "a", "schema": TREE_SCHEMA}, {"name": "b", "schema": TREE_SCHEMA}])
    grammar = xgr.Grammar.from_structural_tag(stag)
    text = '<tool=b>{"root": {"value": "x", "children": [{"value": "y"}]}}</tool>'
    assert _is_grammar_accept_string(grammar, text)


def test_options_change_the_grammar():
    """The same schema under different options keeps converting to its own grammar."""
    variants = [
        {},
        {"style": "qwen_xml"},
        {"style": "muse_xml"},
        {"any_order": True},
        {"max_whitespace_cnt": 2},
        {"excludes": ["</tool>"]},
    ]
    stags = [tools_stag([{"name": "f", "schema": SIMPLE_SCHEMA}], **v) for v in variants]
    first = [stag_to_str(stag) for stag in stags]
    assert len(set(first)) == len(variants)
    assert [stag_to_str(stag) for stag in stags] == first


def test_property_order_changes_the_grammar():
    a = {"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "integer"}}}
    b = {"type": "object", "properties": {"y": {"type": "integer"}, "x": {"type": "string"}}}
    ga = xgr.Grammar.from_json_schema(json.dumps(a))
    gb = xgr.Grammar.from_json_schema(json.dumps(b))
    assert str(ga) != str(gb)
    assert _is_grammar_accept_string(gb, '{"y": 1, "x": "s"}')
    assert not _is_grammar_accept_string(ga, '{"y": 1, "x": "s"}')


def test_in_place_passes_do_not_change_later_conversions():
    schema = json.dumps(TREE_SCHEMA)
    expected = str(xgr.Grammar.from_json_schema(schema))
    returned = xgr.Grammar.from_json_schema(schema)
    GrammarFunctor.repetition_normalizer(returned)
    GrammarFunctor.fsm_builder(returned)
    assert str(xgr.Grammar.from_json_schema(schema)) == expected


def test_repeated_compile_is_identical():
    tokenizer_info = xgr.TokenizerInfo(
        [f"<{i}>" for i in range(64)] + list('{}[]":,0123456789abcdefgh <>=/') + ["tool"]
    )
    stag = tools_stag([{"name": f"t{i}", "schema": s} for i, s in enumerate(ALL_SCHEMAS)])
    first = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_structural_tag(stag)
    second = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_structural_tag(stag)
    assert first.serialize_json() == second.serialize_json()


def test_concurrent_conversions():
    """Threads converting overlapping tool sets get the same grammars as one thread does."""
    extra = [{"type": "object", "properties": {f"field_{i}": {"type": "string"}}} for i in range(8)]
    tools = [{"name": f"tool_{i}", "schema": s} for i, s in enumerate(ALL_SCHEMAS + extra)]
    requests = [
        tools_stag(tools[i : i + 5], style=style)
        for i in range(0, 6, 2)
        for style in ("json", "muse_xml")
    ]
    expected = [stag_to_str(stag) for stag in requests]
    errors: List[str] = []

    def worker(offset: int):
        try:
            for r in range(5):
                for j in range(len(requests)):
                    k = (j + offset + r) % len(requests)
                    if stag_to_str(requests[k]) != expected[k]:
                        errors.append(f"mismatch on request {k}")
        except Exception as e:  # pragma: no cover - reported below
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


if __name__ == "__main__":
    pytest.main(sys.argv)
