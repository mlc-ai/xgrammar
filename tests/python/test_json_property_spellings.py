"""Regression coverage for decoded JSON property identity (#1074)."""

import itertools
import json

import pytest

import xgrammar as xgr
from xgrammar.structural_tag import JSONSchemaFormat, StructuralTag
from xgrammar.testing import _is_grammar_accept_string


def grammar_for(schema, excludes=(), **options):
    return xgr.Grammar.from_structural_tag(
        StructuralTag(
            format=JSONSchemaFormat(json_schema=schema, excludes=list(excludes), **options)
        )
    )


@pytest.mark.parametrize("excludes", [[], ["forbidden"]])
@pytest.mark.parametrize("keyword", ["additionalProperties", "unevaluatedProperties"])
def test_property_spelling_partition(keyword, excludes):
    names = ["", "a", "ab", "é", "😀", "\0", "a\0b", "\0a", "a\0", '"', "\\", "/", *"\b\f\n\r\t"]
    schema = {
        "type": "object",
        "properties": {k: {"type": "integer"} for k in names},
        keyword: {"type": "string"},
    }
    grammar = grammar_for(schema, excludes)
    for key in [
        *[json.dumps(k, ensure_ascii=False)[1:-1] for k in names],
        r"\u0061",
        r"\u0061b",
        r"a\u0062",
        r"\u0061\u0062",
        r"\u00E9",
        r"\uD83d\udE00",
        r"\u0022",
        r"\u005c",
        r"\/",
        r"\u002F",
        r"\u0008",
        r"\u000c",
        r"\u000a",
        r"\u000d",
        r"\u0009",
        r"\\u0061",
        r"\u0062",
        r"\ud83d\ude01",
        r"\ud83d",
        r"\ude00",
        r"e\u0301",
        "abc",
    ]:
        declared = json.loads('"' + key + '"') in names
        for value, expected in [("1", declared), ('"s"', not declared)]:
            assert _is_grammar_accept_string(grammar, '{"' + key + '":' + value + "}") == expected
    for bad in [r"\u01G0", r"\uD83D\uDE0X", r"\U0061", r"\v"]:
        assert not _is_grammar_accept_string(grammar, '{"' + bad + '":"s"}')


def test_spelling_excludes_cross_character_boundaries():
    schema = {
        "type": "object",
        "properties": {"abc": {"type": "integer"}},
        "additionalProperties": True,
    }
    grammar = grammar_for(schema, ["61b", "abc"])
    for parts in itertools.product(("a", r"\u0061"), ("b", r"\u0062"), ("c", r"\u0063")):
        body = "".join(parts)
        assert _is_grammar_accept_string(grammar, '{"' + body + '":1}') == (
            "61b" not in body and "abc" not in body
        )
        assert not _is_grammar_accept_string(grammar, '{"' + body + '":"s"}')
    assert not _is_grammar_accept_string(grammar, r'{"\u0061bx":"s"}')


def test_split_and_crossing_tokens():
    schema = {
        "type": "object",
        "properties": {"😀": {"type": "integer"}},
        "required": ["😀"],
        "additionalProperties": False,
    }
    grammar = xgr.Grammar.from_json_schema(json.dumps(schema))
    vocab = [r'{"\uD83D', r'\uDE00":1}', r'\uDE00":"s"}', "[EOS]"]
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(vocab, stop_token_ids=[3]))
    for candidate in [grammar, xgr.Grammar.from_ebnf(str(grammar))]:
        compiled = compiler.compile_grammar(candidate)
        text = "".join(vocab[:2])
        for cut in range(1, len(text)):
            matcher = xgr.GrammarMatcher(compiled)
            assert matcher.accept_string(text[:cut]) and matcher.accept_string(text[cut:])
            assert matcher.is_completed()
        matcher = xgr.GrammarMatcher(compiled)
        assert matcher.accept_token(0)
        mask = xgr.allocate_token_bitmask(1, len(vocab))
        matcher.fill_next_token_bitmask(mask)
        assert int(mask[0, 0]) & 2 and not int(mask[0, 0]) & 4
        assert matcher.accept_token(1) and matcher.is_completed()


@pytest.mark.parametrize("any_order", [False, True])
@pytest.mark.parametrize("excludes", [[], ["unused"]])
def test_optional_open_object(any_order, excludes):
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "additionalProperties": True,
    }
    grammar = grammar_for(schema, excludes, any_order=any_order)
    for key in ["a", r"\u0061"]:
        assert _is_grammar_accept_string(grammar, '{"' + key + '":1}')
        assert not _is_grammar_accept_string(grammar, '{"' + key + '":"wrong"}')
    for key in [r"\u0062", r"\\u0061"]:
        assert _is_grammar_accept_string(grammar, '{"' + key + '":"additional"}')


def test_nested_json_key_in_xml():
    child = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "additionalProperties": True,
    }
    grammar = grammar_for(
        {"type": "object", "properties": {"value": child}, "required": ["value"]},
        ["unused"],
        style="kimi_k3_xml",
    )
    prefix = '<|open|>argument key="value" type="object"<|sep|>'
    suffix = "<|close|>argument<|sep|>"
    assert _is_grammar_accept_string(grammar, prefix + r'{"\u0061":1}' + suffix)
    assert not _is_grammar_accept_string(grammar, prefix + r'{"\u0061":"s"}' + suffix)
    assert not _is_grammar_accept_string(
        grammar, prefix.replace("value", r"\u0076alue") + "{}" + suffix
    )


@pytest.mark.xfail(
    strict=True, reason="Pattern and propertyNames overrides skip named-key exclusion"
)
@pytest.mark.parametrize(
    "override", [{"patternProperties": {"^z$": {}}}, {"propertyNames": {"type": "string"}}]
)
def test_named_key_with_override(override):
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "additionalProperties": True,
        **override,
    }
    assert not _is_grammar_accept_string(grammar_for(schema), '{"a":"s"}')


@pytest.mark.parametrize("key", [r"\ud800", r"\udfff"])
def test_unpaired_surrogate_in_schema_name(key):
    with pytest.raises(RuntimeError, match="Failed to parse JSON"):
        xgr.Grammar.from_json_schema('{"type":"object","properties":{"' + key + '":{}}}')
