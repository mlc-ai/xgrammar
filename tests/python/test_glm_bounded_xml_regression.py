# Regression coverage for length-bounded GLM XML tool arguments.
import json

import torch

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string, _json_schema_to_ebnf


def test_length_boundaries_and_unicode():
    checks = 0
    for lo, hi in [(0, 0), (0, 1), (1, 1), (1, 4), (2, 4), (1, 16), (3, -1)]:
        prop = {"type": "string", "minLength": lo}
        if hi >= 0:
            prop["maxLength"] = hi
        schema = {
            "type": "object",
            "properties": {"query": prop},
            "required": ["query"],
            "additionalProperties": False,
        }
        grammar = _json_schema_to_ebnf(schema, json_format="glm_xml")
        for value in [
            "",
            "a",
            "ab",
            "abcd",
            "abcde",
            " ",
            "  ",
            " a ",
            "\n",
            "é",
            "你",
            "😀",
            "a😀",
            "<",
            "a < b",
            "<div>",
            "</arg_valu",
            "x" * 16,
            "x" * 17,
        ]:
            want = len(value) >= lo and (hi < 0 or len(value) <= hi)
            xml = "<arg_key>query</arg_key><arg_value>" + value + "</arg_value>"
            actual = _is_grammar_accept_string(grammar, xml)
            assert actual == want, (lo, hi, repr(value), want, actual)
            checks += 1
    print(json.dumps({"boundary_and_unicode_checks": checks, "result": "PASS"}))


def test_refs_and_composed_schemas():
    base = {"type": "string", "minLength": 2, "maxLength": 4}
    variants = {
        "direct": base,
        "ref": {"$ref": "#/$defs/value"},
        "allOf": {"allOf": [base]},
        "anyOf": {"anyOf": [base, {"type": "integer"}]},
    }
    failures = []
    count = 0
    for name, prop in variants.items():
        schema = {
            "type": "object",
            "properties": {"query": prop},
            "required": ["query"],
            "additionalProperties": False,
            "$defs": {"value": base},
        }
        grammar = _json_schema_to_ebnf(schema, json_format="glm_xml")
        for value in ["a", "ab", "abcd", "abcde", " a ", "     ", "é😀", "a < b", "  10 ", "12"]:
            want = 2 <= len(value) <= 4 or (name == "anyOf" and value.strip().isdigit())
            text = "<arg_key>query</arg_key><arg_value>" + value + "</arg_value>"
            actual = _is_grammar_accept_string(grammar, text)
            count += 1
            if actual != want:
                failures.append((name, repr(value), want, actual))
    print(json.dumps({"checks": count, "failures": failures}))
    assert not failures


def test_masks_and_speculative_rollback():
    vocab = [
        "a",
        "b",
        "<",
        ">",
        "</arg_value>",
        "<arg_key>query</arg_key><arg_value>",
        "é",
        "😀",
        " ",
        "</tool_call>",
        "<tool_call>probe",
        "\n",
        "ab</arg_value>",
        "</arg_value></tool_call>",
        "<eos>",
    ]
    info = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, stop_token_ids=[len(vocab) - 1])
    compiler = xgr.GrammarCompiler(info, max_threads=1)
    spec = {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<tool_call>"],
            "tags": [
                {
                    "begin": "<tool_call>probe",
                    "content": {
                        "type": "json_schema",
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "minLength": 1, "maxLength": 4}
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                        "style": "glm_xml",
                    },
                    "end": "</tool_call>",
                }
            ],
            "at_least_one": False,
            "stop_after_first": False,
        },
    }
    ctx = compiler.compile_structural_tag(json.dumps(spec))
    mask = xgr.allocate_token_bitmask(1, len(vocab))
    history = []
    m = xgr.GrammarMatcher(ctx, max_rollback_tokens=4)
    checks = 0
    call = [10, 11, 5, 7, 0, 1, 13]
    for step, token in enumerate(call * 4):
        m.fill_next_token_bitmask(mask, 0)
        for candidate in range(len(vocab)):
            fresh = xgr.GrammarMatcher(ctx, max_rollback_tokens=4)
            for t in history:
                assert fresh.accept_token(t)
            want = fresh.accept_token(candidate)
            allowed = bool((int(mask[0, candidate // 32]) & 0xFFFFFFFF) & (1 << (candidate % 32)))
            assert allowed == want, (step, candidate, allowed, want)
            checks += 1
        assert m.accept_token(token), (step, token)
        history.append(token)
        m.fill_next_token_bitmask(mask, 0)
        expected = mask.clone()
        n = min(4, len(history))
        m.rollback(n)
        for t in history[-n:]:
            assert m.accept_token(t)
        m.fill_next_token_bitmask(mask, 0)
        assert torch.equal(mask, expected), ("rollback", step)
    print(
        json.dumps(
            {"mask_accept_checks": checks, "rollback_checks": len(history), "result": "PASS"}
        )
    )


def test_recursive_schema():
    schema = {
        "type": "object",
        "properties": {"query": {"$ref": "#/$defs/value"}},
        "required": ["query"],
        "additionalProperties": False,
        "$defs": {
            "value": {
                "anyOf": [
                    {"type": "string", "minLength": 1, "maxLength": 4},
                    {"$ref": "#/$defs/value"},
                ]
            }
        },
    }
    grammar = _json_schema_to_ebnf(schema, json_format="glm_xml")
    for v, want in [("a", True), ("abcd", True), ("abcde", False), ("     ", False)]:
        actual = _is_grammar_accept_string(
            grammar, "<arg_key>query</arg_key><arg_value>" + v + "</arg_value>"
        )
        assert actual == want, (repr(v), actual, want)
    print(json.dumps({"recursive_alias_checks": 4, "result": "PASS"}))


def test_repeated_calls_do_not_accumulate_parser_states():
    vocab = [chr(i) for i in range(32, 127)] + ["\n", "<eos>"]
    info = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, stop_token_ids=[len(vocab) - 1])
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 4000}},
        "required": ["query"],
        "additionalProperties": False,
    }
    tag = {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<tool_call>"],
            "tags": [
                {
                    "begin": "<tool_call>probe",
                    "content": {"type": "json_schema", "json_schema": schema, "style": "glm_xml"},
                    "end": "</tool_call>",
                }
            ],
            "stop_after_first": False,
        },
    }
    ctx = xgr.GrammarCompiler(info, max_threads=1).compile_structural_tag(json.dumps(tag))
    matcher = xgr.GrammarMatcher(ctx)
    call = "<tool_call>probe\n<arg_key>query</arg_key><arg_value>hello</arg_value></tool_call>"
    for _ in range(100):
        assert matcher.accept_string(call)
    states = matcher._debug_print_internal_state().count("ParserState(")
    assert states < 64, states
