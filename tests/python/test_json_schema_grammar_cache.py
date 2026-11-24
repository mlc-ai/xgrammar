"""Tests for the process-wide cache of grammars converted from JSON schemas."""

import json
import sys
import threading
from contextlib import contextmanager
from typing import Any, Dict, List

import pytest

import xgrammar as xgr
from xgrammar.structural_tag import JSONSchemaFormat, StructuralTag, TagFormat, TriggeredTagsFormat
from xgrammar.testing import GrammarFunctor, _is_grammar_accept_string

# Every test reads or changes the process-wide cache, so none may run concurrently with another.
pytestmark = pytest.mark.thread_unsafe

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
# Styles that accept a non-empty excludes list.
STYLES_WITH_EXCLUDES = [s for s in ALL_STYLES if s not in ("cohere_xml", "minimax_m3_xml")]

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

# Mutual recursion between two definitions, plus a definition shared by both.
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

VARIED_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 20},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "mode": {"enum": ["fast", "slow"]},
        "ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "when": {"type": "string", "format": "date"},
        "extra": {"type": "object", "additionalProperties": {"type": "integer"}},
        "flag": {"type": "boolean"},
        "nothing": {"type": "null"},
    },
    "required": ["name", "mode"],
}

ALL_SCHEMAS = [SIMPLE_SCHEMA, TREE_SCHEMA, MUTUAL_SCHEMA, VARIED_SCHEMA]


@contextmanager
def cache_budget(max_memory_bytes: int):
    """Run with the given cache budget on an empty cache, then restore the previous budget."""
    previous = xgr.Grammar.get_json_schema_cache_max_memory()
    xgr.Grammar.set_json_schema_cache_max_memory(max_memory_bytes)
    xgr.Grammar.clear_json_schema_cache()
    try:
        yield
    finally:
        xgr.Grammar.set_json_schema_cache_max_memory(previous)
        xgr.Grammar.clear_json_schema_cache()


def uncached(fn, *args, **kwargs):
    """Run fn with the cache disabled, for a reference result."""
    previous = xgr.Grammar.get_json_schema_cache_max_memory()
    xgr.Grammar.set_json_schema_cache_max_memory(0)
    try:
        return fn(*args, **kwargs)
    finally:
        xgr.Grammar.set_json_schema_cache_max_memory(previous)


def tool_tag(name: str, schema: Dict[str, Any], **format_kwargs) -> TagFormat:
    return TagFormat(
        begin=f"<tool={name}>",
        content=JSONSchemaFormat(json_schema=schema, **format_kwargs),
        end="</tool>",
    )


def tools_stag(tools: List[Dict[str, Any]], **format_kwargs) -> StructuralTag:
    """A structural tag that lets the model call any of the tools, like the builtin templates do."""
    return StructuralTag(
        format=TriggeredTagsFormat(
            triggers=["<tool="],
            tags=[tool_tag(t["name"], t["schema"], **format_kwargs) for t in tools],
        )
    )


def stag_to_str(stag: StructuralTag) -> str:
    return str(xgr.Grammar.from_structural_tag(stag))


def test_default_budget_and_stats_shape():
    stats = xgr.Grammar.get_json_schema_cache_stats()
    assert set(stats) == {
        "hits",
        "misses",
        "evictions",
        "num_entries",
        "memory_bytes",
        "max_memory_bytes",
    }
    assert xgr.Grammar.get_json_schema_cache_max_memory() == 64 * 1024 * 1024


def test_hit_across_json_schema_conversions():
    schema = json.dumps(SIMPLE_SCHEMA)
    with cache_budget(-1):
        first = str(xgr.Grammar.from_json_schema(schema))
        stats = xgr.Grammar.get_json_schema_cache_stats()
        assert (stats["hits"], stats["misses"], stats["num_entries"]) == (0, 1, 1)
        assert stats["memory_bytes"] > 0

        second = str(xgr.Grammar.from_json_schema(schema))
        stats = xgr.Grammar.get_json_schema_cache_stats()
        assert (stats["hits"], stats["misses"], stats["num_entries"]) == (1, 1, 1)
        assert first == second == uncached(lambda: str(xgr.Grammar.from_json_schema(schema)))


def test_hit_across_structural_tags():
    """Two requests with overlapping tool sets: the shared tools hit the cache."""
    tools = [{"name": f"tool_{i}", "schema": s} for i, s in enumerate(ALL_SCHEMAS)]
    with cache_budget(-1):
        stag_to_str(tools_stag(tools[:3]))
        stats = xgr.Grammar.get_json_schema_cache_stats()
        assert (stats["hits"], stats["misses"]) == (0, 3)

        # The next request adds a tool (the deferred-tool-loading pattern).
        result = stag_to_str(tools_stag(tools))
        stats = xgr.Grammar.get_json_schema_cache_stats()
        assert (stats["hits"], stats["misses"], stats["num_entries"]) == (3, 4, 4)
        assert result == uncached(stag_to_str, tools_stag(tools))


def test_structural_tag_and_json_schema_share_entries():
    """Grammar.from_json_schema and a "json" style tag with default options use the same key when
    the schema text is the same (the key is the exact text; a structural tag stores its schemas
    compactly serialized)."""
    with cache_budget(-1):
        xgr.Grammar.from_json_schema(json.dumps(SIMPLE_SCHEMA, separators=(",", ":")))
        stag_to_str(tools_stag([{"name": "a", "schema": SIMPLE_SCHEMA}]))
        stats = xgr.Grammar.get_json_schema_cache_stats()
        assert (stats["hits"], stats["misses"], stats["num_entries"]) == (1, 1, 1)


@pytest.mark.parametrize("style", ALL_STYLES)
@pytest.mark.parametrize("schema_index", range(len(ALL_SCHEMAS)))
def test_cached_grammar_identical_per_style(style: str, schema_index: int):
    schema = ALL_SCHEMAS[schema_index]
    if style == "minimax_m3_xml" and schema is VARIED_SCHEMA:
        # minimax_m3_xml rejects string format/length constraints and free-form object keys;
        # test the rest of the schema.
        schema = json.loads(json.dumps(schema))
        schema["properties"]["name"] = {"type": "string"}
        del schema["properties"]["when"]
        del schema["properties"]["extra"]
    stag = tools_stag([{"name": "f", "schema": schema}], style=style)
    expected = uncached(stag_to_str, stag)
    with cache_budget(-1):
        cold = stag_to_str(stag)
        warm = stag_to_str(stag)
        assert xgr.Grammar.get_json_schema_cache_stats()["hits"] == 1
    assert cold == expected
    assert warm == expected


def test_recursive_schema_accepts_after_hit():
    """The recursive rules are spliced whole: a hit still accepts nested and rejects bad input."""
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
    with cache_budget(-1):
        stag_to_str(stag)  # fill the cache
        grammar = xgr.Grammar.from_structural_tag(stag)
        assert xgr.Grammar.get_json_schema_cache_stats()["hits"] == 2
        for text in accepted:
            assert _is_grammar_accept_string(grammar, text)
        for text in rejected:
            assert not _is_grammar_accept_string(grammar, text)


def test_same_tool_twice_in_one_tag():
    """Two tools with the same schema splice two copies of one cached grammar."""
    tools = [{"name": "a", "schema": TREE_SCHEMA}, {"name": "b", "schema": TREE_SCHEMA}]
    stag = tools_stag(tools)
    expected = uncached(stag_to_str, stag)
    with cache_budget(-1):
        assert stag_to_str(stag) == expected
        grammar = xgr.Grammar.from_structural_tag(stag)
    text = '<tool=b>{"root": {"value": "x", "children": [{"value": "y"}]}}</tool>'
    assert _is_grammar_accept_string(grammar, text)


STAG_OPTION_VARIANTS = [
    {},
    {"style": "qwen_xml"},
    {"style": "muse_xml"},
    {"style": "kimi_k3_xml"},
    {"any_order": True},
    {"max_whitespace_cnt": 2},
    {"max_whitespace_cnt": 3},
    {"excludes": ["</tool>"]},
    {"excludes": ["</tool>", "<tool="]},
    {"excludes": ["<tool=", "</tool>"]},
    {"style": "muse_xml", "excludes": ["</tool>"]},
    {"style": "muse_xml", "any_order": True, "max_whitespace_cnt": 1},
]


def test_structural_tag_options_never_share_entries():
    """The same schema under different options must convert separately (no false hits)."""
    stags = [
        tools_stag([{"name": "f", "schema": VARIED_SCHEMA}], **v) for v in STAG_OPTION_VARIANTS
    ]
    expected = [uncached(stag_to_str, stag) for stag in stags]
    # Most variants change the grammar (reordered excludes do not). Each must reproduce its own
    # uncached grammar.
    assert len(set(expected)) >= len(expected) - 2
    with cache_budget(-1):
        for _ in range(2):
            assert [stag_to_str(stag) for stag in stags] == expected
        stats = xgr.Grammar.get_json_schema_cache_stats()
        # Variants whose converter inputs end up equal (the tag normalizes the order of excludes)
        # may share an entry; variants with different grammars never do.
        assert len(set(expected)) <= stats["num_entries"] <= len(stags)
        assert stats["hits"] + stats["misses"] == 2 * len(stags)
        assert stats["misses"] == stats["num_entries"]


JSON_SCHEMA_OPTION_VARIANTS = [
    {},
    {"any_whitespace": False},
    {"any_whitespace": False, "indent": 2},
    {"any_whitespace": False, "indent": 4},
    {"any_whitespace": False, "separators": (",", ":")},
    {"any_whitespace": False, "separators": (", ", ": ")},
    {"any_whitespace": False, "indent": 2, "separators": (",", ": ")},
    {"strict_mode": False},
    {"max_whitespace_cnt": 2},
    {"any_order": True},
]


def test_json_schema_options_never_share_entries():
    schema = json.dumps(VARIED_SCHEMA)
    convert = lambda kwargs: str(xgr.Grammar.from_json_schema(schema, **kwargs))
    expected = [uncached(convert, kwargs) for kwargs in JSON_SCHEMA_OPTION_VARIANTS]
    assert len(set(expected)) >= len(expected) - 2
    with cache_budget(-1):
        for _ in range(2):
            assert [convert(kwargs) for kwargs in JSON_SCHEMA_OPTION_VARIANTS] == expected
        stats = xgr.Grammar.get_json_schema_cache_stats()
        assert stats["num_entries"] == len(JSON_SCHEMA_OPTION_VARIANTS)
        assert stats["hits"] == len(JSON_SCHEMA_OPTION_VARIANTS)


def test_schema_text_is_part_of_the_key():
    """Property order changes the fixed-order grammar, so reordered schemas are different keys."""
    a = {"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "integer"}}}
    b = {"type": "object", "properties": {"y": {"type": "integer"}, "x": {"type": "string"}}}
    with cache_budget(-1):
        ga = str(xgr.Grammar.from_json_schema(json.dumps(a)))
        gb = str(xgr.Grammar.from_json_schema(json.dumps(b)))
        assert xgr.Grammar.get_json_schema_cache_stats()["num_entries"] == 2
    assert ga != gb
    assert _is_grammar_accept_string(
        xgr.Grammar.from_json_schema(json.dumps(b)), '{"y": 1, "x": "s"}'
    )


def test_recursion_limit_is_part_of_the_key():
    """A schema converted under the default limit must still fail under a smaller one."""
    depth = 60
    schema: Dict[str, Any] = {"type": "string"}
    for _ in range(depth):
        schema = {"type": "object", "properties": {"a": schema}, "required": ["a"]}
    text = json.dumps(schema)
    with cache_budget(-1):
        xgr.Grammar.from_json_schema(text)
        with xgr.max_recursion_depth(10):
            with pytest.raises(RuntimeError):
                xgr.Grammar.from_json_schema(text)
        xgr.Grammar.from_json_schema(text)
        assert xgr.Grammar.get_json_schema_cache_stats()["hits"] == 1


def test_failed_conversion_is_not_cached():
    bad = json.dumps({"type": "object", "properties": {"a": {"$ref": "#/$defs/missing"}}})
    with cache_budget(-1):
        for _ in range(2):
            with pytest.raises(Exception):
                xgr.Grammar.from_json_schema(bad)
        assert xgr.Grammar.get_json_schema_cache_stats()["num_entries"] == 0


def test_in_place_passes_do_not_touch_cached_grammar():
    """Grammar.from_json_schema returns a private copy, so in-place passes on it are harmless."""
    schema = json.dumps(TREE_SCHEMA)
    with cache_budget(-1):
        expected = str(xgr.Grammar.from_json_schema(schema))
        returned = xgr.Grammar.from_json_schema(schema)
        GrammarFunctor.repetition_normalizer(returned)
        GrammarFunctor.fsm_builder(returned)
        assert str(xgr.Grammar.from_json_schema(schema)) == expected
        assert xgr.Grammar.get_json_schema_cache_stats()["hits"] == 2


def test_disabled_cache_keeps_nothing():
    with cache_budget(0):
        for _ in range(2):
            xgr.Grammar.from_json_schema(json.dumps(SIMPLE_SCHEMA))
            stag_to_str(tools_stag([{"name": "a", "schema": TREE_SCHEMA}]))
        stats = xgr.Grammar.get_json_schema_cache_stats()
        assert stats["num_entries"] == 0
        assert stats["memory_bytes"] == 0
        assert stats["hits"] == 0


def _many_schemas(n: int) -> List[Dict[str, Any]]:
    return [
        {
            "type": "object",
            "properties": {f"field_{i}": {"type": "string"}},
            "required": [f"field_{i}"],
        }
        for i in range(n)
    ]


def test_lru_eviction_respects_budget():
    schemas = [json.dumps(s) for s in _many_schemas(20)]
    with cache_budget(-1):
        xgr.Grammar.from_json_schema(schemas[0])
        entry_size = xgr.Grammar.get_json_schema_cache_stats()["memory_bytes"]
    budget = entry_size * 5 + entry_size // 2  # room for five entries
    with cache_budget(budget):
        for s in schemas:
            xgr.Grammar.from_json_schema(s)
        stats = xgr.Grammar.get_json_schema_cache_stats()
        assert stats["memory_bytes"] <= budget
        assert stats["num_entries"] == 5
        assert stats["evictions"] == 15

        # The most recent entries are kept; touching one protects it from the next eviction.
        xgr.Grammar.from_json_schema(schemas[15])
        assert xgr.Grammar.get_json_schema_cache_stats()["hits"] == 1
        xgr.Grammar.from_json_schema(schemas[0])  # evicts the oldest, schemas[16]
        xgr.Grammar.from_json_schema(schemas[15])
        assert xgr.Grammar.get_json_schema_cache_stats()["hits"] == 2
        xgr.Grammar.from_json_schema(schemas[16])
        assert xgr.Grammar.get_json_schema_cache_stats()["hits"] == 2

        # Shrinking the budget evicts at once.
        xgr.Grammar.set_json_schema_cache_max_memory(entry_size * 2 + entry_size // 2)
        assert xgr.Grammar.get_json_schema_cache_stats()["num_entries"] == 2


def test_entry_larger_than_budget_is_skipped():
    with cache_budget(64):
        xgr.Grammar.from_json_schema(json.dumps(VARIED_SCHEMA))
        stats = xgr.Grammar.get_json_schema_cache_stats()
        assert stats["num_entries"] == 0
        assert stats["memory_bytes"] == 0


def test_clear_resets_entries_and_counters():
    with cache_budget(-1):
        for _ in range(2):
            xgr.Grammar.from_json_schema(json.dumps(SIMPLE_SCHEMA))
        xgr.Grammar.clear_json_schema_cache()
        stats = xgr.Grammar.get_json_schema_cache_stats()
        assert (
            stats["num_entries"] == stats["hits"] == stats["misses"] == stats["memory_bytes"] == 0
        )


def test_compiled_structural_tag_unchanged():
    """The compiled grammar of a cached conversion matches the uncached one."""
    tokenizer_info = xgr.TokenizerInfo(
        [f"<{i}>" for i in range(64)] + list('{}[]":,0123456789abcdefgh <>=/') + ["tool"]
    )
    stag = tools_stag([{"name": f"t{i}", "schema": s} for i, s in enumerate(ALL_SCHEMAS)])
    reference = uncached(
        lambda: xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_structural_tag(
            stag
        )
    )
    with cache_budget(-1):
        stag_to_str(stag)
        compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_structural_tag(
            stag
        )
        assert xgr.Grammar.get_json_schema_cache_stats()["hits"] == len(ALL_SCHEMAS)
    assert str(compiled.grammar) == str(reference.grammar)
    assert compiled.serialize_json() == reference.serialize_json()


def test_concurrent_conversions():
    """Threads converting overlapping tool sets, while the budget changes, get correct grammars."""
    schemas = ALL_SCHEMAS + _many_schemas(12)
    tools = [{"name": f"tool_{i}", "schema": s} for i, s in enumerate(schemas)]
    requests = [
        tools_stag(tools[i : i + 6], style=style)
        for i in range(0, 10, 2)
        for style in ("json", "muse_xml")
    ]
    expected = [uncached(stag_to_str, stag) for stag in requests]

    errors: List[str] = []
    rounds = 6

    def worker(offset: int):
        try:
            for r in range(rounds):
                for j in range(len(requests)):
                    k = (j + offset + r) % len(requests)
                    if stag_to_str(requests[k]) != expected[k]:
                        errors.append(f"mismatch on request {k}")
        except Exception as e:  # pragma: no cover - reported below
            errors.append(repr(e))

    def churn():
        try:
            for r in range(rounds * 4):
                xgr.Grammar.set_json_schema_cache_max_memory([-1, 200_000, 50_000][r % 3])
                if r % 5 == 0:
                    xgr.Grammar.clear_json_schema_cache()
                xgr.Grammar.get_json_schema_cache_stats()
        except Exception as e:  # pragma: no cover
            errors.append(repr(e))

    with cache_budget(-1):
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        threads.append(threading.Thread(target=churn))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        xgr.Grammar.set_json_schema_cache_max_memory(-1)
        # After the churn, a warm pass still reproduces every grammar.
        for stag, exp in zip(requests, expected):
            assert stag_to_str(stag) == exp
    assert errors == []


def test_concurrent_compiles_with_multithreaded_compiler():
    """The compiler's own worker threads convert tags concurrently through the shared cache."""
    tokenizer_info = xgr.TokenizerInfo(
        [f"<{i}>" for i in range(64)] + list('{}[]":,0123456789abcdefgh <>=/')
    )
    tools = [
        {"name": f"tool_{i}", "schema": s} for i, s in enumerate(ALL_SCHEMAS + _many_schemas(8))
    ]
    requests = [tools_stag(tools[i : i + 5]) for i in range(0, 8)]
    expected = [
        uncached(
            lambda s=s: str(
                xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)
                .compile_structural_tag(s)
                .grammar
            )
        )
        for s in requests
    ]
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8, cache_enabled=False)
    results: Dict[int, str] = {}

    def compile_one(i: int):
        results[i] = str(compiler.compile_structural_tag(requests[i]).grammar)

    with cache_budget(-1):
        threads = [threading.Thread(target=compile_one, args=(i,)) for i in range(len(requests))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert [results[i] for i in range(len(requests))] == expected


if __name__ == "__main__":
    pytest.main(sys.argv)
