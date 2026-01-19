"""This test uses the optimized JSON grammar provided by the grammar library."""

import sys
import threading
import time
from typing import Dict, List, Tuple

import pytest
from pydantic import BaseModel
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.testing import _get_allow_empty_rule_ids


@pytest.mark.hf_token_required
def test_compiled_grammar():
    grammar = xgr.Grammar.builtin_json_grammar()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    time_start = time.monotonic_ns()
    context = compiler.compile_grammar(grammar)
    time_end = time.monotonic_ns()
    print(f"Time to get compiled grammar: {(time_end - time_start) / 1e3} us")

    def check_matcher(matcher: xgr.GrammarMatcher):
        assert not matcher.is_terminated()
        assert not matcher.accept_string('{ name: "John" }')
        assert matcher.accept_string('{"name": "John"}')
        assert matcher.is_terminated()

    time_start = time.monotonic_ns()
    matcher_1 = xgr.GrammarMatcher(context, terminate_without_stop_token=True)
    time_end = time.monotonic_ns()
    print(f"Time to init matcher 1: {(time_end - time_start) / 1e3} us")
    check_matcher(matcher_1)
    time_start = time.monotonic_ns()
    matcher_2 = xgr.GrammarMatcher(context, terminate_without_stop_token=True)
    time_end = time.monotonic_ns()
    print(f"Time to init matcher 2: {(time_end - time_start) / 1e3} us")
    check_matcher(matcher_2)


# Test max_threads=1 since we have a special logic to avoid using ThreadPool and mutex
@pytest.mark.hf_token_required
@pytest.mark.parametrize("max_threads", (8, 1))
def test_grammar_compiler_json(max_threads):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    time_start = time.monotonic_ns()
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=max_threads)
    time_end = time.monotonic_ns()
    print(f"Time to init cached grammar compiler: {(time_end - time_start) / 1e3} us")

    def check_matcher(matcher: xgr.GrammarMatcher):
        assert not matcher.is_terminated()
        assert not matcher.accept_string('{ name: "John" }')
        assert matcher.accept_string('{"name": "John"}')
        assert matcher.is_terminated()

    time_start = time.monotonic_ns()
    compiled_grammar = grammar_compiler.compile_builtin_json_grammar()
    time_end = time.monotonic_ns()
    print(f"Time to get compiled grammar: {(time_end - time_start) / 1e3} us")
    matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    check_matcher(matcher)

    time_start = time.monotonic_ns()
    compiled_grammar = grammar_compiler.compile_builtin_json_grammar()
    time_end = time.monotonic_ns()
    print(f"Time to get compiled grammar again: {(time_end - time_start) / 1e3} us")
    matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    check_matcher(matcher)

    grammar_compiler.clear_cache()

    time_start = time.monotonic_ns()
    compiled_grammar = grammar_compiler.compile_builtin_json_grammar()
    time_end = time.monotonic_ns()
    print(f"Time to get compiled grammar after clear: {(time_end - time_start) / 1e3} us")
    matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    check_matcher(matcher)


@pytest.mark.hf_token_required
def test_grammar_compiler_json_schema():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

    class MainModel(BaseModel):
        integer_field: int
        number_field: float
        boolean_field: bool
        any_array_field: List
        array_field: List[str]
        tuple_field: Tuple[str, int, List[str]]
        object_field: Dict[str, int]
        nested_object_field: Dict[str, Dict[str, int]]

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

    def check_with_fmt(any_whitespace, indent, separators, test_id):
        instance_str = instance.model_dump_json(indent=indent, round_trip=True)

        time_start = time.monotonic_ns()
        compiled_grammar = grammar_compiler.compile_json_schema(
            MainModel, any_whitespace=any_whitespace, indent=indent, separators=separators
        )
        time_end = time.monotonic_ns()
        print(f"Time to get compiled grammar {test_id}: {(time_end - time_start) / 1e3} us")
        matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)

        assert not matcher.is_terminated()
        assert matcher.accept_string(instance_str)
        assert matcher.is_terminated()

    check_with_fmt(False, None, (",", ":"), "1")
    check_with_fmt(False, None, (",", ":"), "2")
    check_with_fmt(False, 2, None, "3")
    check_with_fmt(False, 2, (",", ": "), "4")

    check_with_fmt(True, None, (",", ":"), "5")
    check_with_fmt(True, None, (",", ":"), "6")
    check_with_fmt(True, 2, None, "7")
    check_with_fmt(True, 2, (",", ": "), "8")

    grammar_compiler.clear_cache()

    check_with_fmt(False, None, (",", ":"), "9")


grammar_expected_test_get_allow_empty_rule_ids = [
    (
        r"""root ::= rule1 rule2 | "abc"
    rule1 ::= "abc" | ""
    rule2 ::= "def" rule3 | ""
    rule3 ::= "ghi"
    """,
        [0, 1, 2],
    ),
    (
        r"""root ::= rule1 rule2 [a-z]*
    rule1 ::= "abc" | ""
    rule2 ::= "def" | ""
    """,
        [0, 1, 2],
    ),
    (
        r"""root ::= rule1 rule3
    rule1 ::= "abc" | ""
    rule2 ::= "def" | ""
    rule3 ::= rule1 rule2
    """,
        [0, 1, 2, 3],
    ),
    (
        r"""root ::= [a]* [b]* rule1
rule1 ::= [abc]* [def]*
""",
        [0, 1],
    ),
]


@pytest.mark.parametrize("grammar, expected", grammar_expected_test_get_allow_empty_rule_ids)
def test_get_allow_empty_rule_ids(grammar: str, expected: List[int]):
    grammar_compiler = xgr.GrammarCompiler(xgr.TokenizerInfo([]))
    compiled_grammar = grammar_compiler.compile_grammar(grammar)
    allow_empty_rule_ids = _get_allow_empty_rule_ids(compiled_grammar)
    assert allow_empty_rule_ids == expected


schema_instances = [
    (
        '{"type": "object","properties":{"username":{"type": "string"}},"required":["username"]}',
        '{"username":"Alice"}',
    ),
    (
        '{"type": "object","properties":{"age":{"type": "integer"}},"required":["age"]}',
        '{"age":30}',
    ),
    (
        '{"type": "object","properties":{"city":{"type": "string"}},"required":["city"]}',
        '{"city":"Paris"}',
    ),
    (
        '{"type": "object","properties":{"isActive":{"type": "boolean"}},"required":["isActive"]}',
        '{"isActive":true}',
    ),
    (
        '{"type": "object","properties":{"rating":{"type": "number"}},"required":["rating"]}',
        '{"rating":4.5}',
    ),
    (
        '{"type": "object","properties":{"name":{"type": "string"}},"required":["name"]}',
        '{"name":"Bob"}',
    ),
    (
        '{"type": "object","properties":{"quantity":{"type": "integer"}},"required":["quantity"]}',
        '{"quantity":10}',
    ),
    (
        '{"type": "object","properties":{"color":{"type": "string"}},"required":["color"]}',
        '{"color":"blue"}',
    ),
    (
        '{"type": "object","properties":{"temperature":{"type": "number"}},"required":["temperature"]}',
        '{"temperature":22.5}',
    ),
    (
        '{"type": "object","properties":{"isCompleted":{"type": "boolean"}},"required":["isCompleted"]}',
        '{"isCompleted":false}',
    ),
]


@pytest.mark.hf_token_required
def test_grammar_compiler_json_schema_concurrent():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

    def check_matcher(matcher: xgr.GrammarMatcher, instance_str: str):
        assert not matcher.is_terminated()
        assert matcher.accept_string(instance_str)
        assert matcher.is_terminated()

    num_schemas = len(schema_instances)
    thread_cnt = 100
    threads = []

    def compile_grammar(id: int, schema: str, instance_str: str):
        schema_id = id % num_schemas
        time_mid = time.monotonic_ns()
        print(f"Thread {id} start compile grammar {schema_id}: {(time_mid - time_start) / 1e3} us")
        compiled_grammar = grammar_compiler.compile_json_schema(
            schema, indent=None, separators=(",", ":"), strict_mode=True
        )
        time_end = time.monotonic_ns()
        print(f"Thread {id} end compile grammar {schema_id}: {(time_end - time_start) / 1e3} us")
        matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        check_matcher(matcher, instance_str)

    time_start = time.monotonic_ns()
    for i in range(thread_cnt):
        t = threading.Thread(target=compile_grammar, args=(i, *schema_instances[i % num_schemas]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


@pytest.mark.hf_token_required
def test_grammar_compiler_cache_unlimited():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)

    def make_schema(name_str: str):
        return {
            "properties": {name_str: {"type": "string"}},
            "required": [name_str],
            "type": "object",
        }

    MB = 1024 * 1024

    # Default no limit
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    assert grammar_compiler.cache_limit_bytes == -1  # No limit (default, -1)
    assert grammar_compiler.get_cache_size_bytes() == 0  # No memory usage
    sum_single = 0
    for i in range(10):
        schema = make_schema(f"name_{i}")
        compiled_grammar = grammar_compiler.compile_json_schema(schema, strict_mode=True)
        sum_single += compiled_grammar.memory_size_bytes
        memory_usage = grammar_compiler.get_cache_size_bytes()
        assert memory_usage == sum_single
        print(f"Cache memory usage after {i + 1} schemas: {memory_usage / MB:.3f} MB / unlimited")

    old_size = grammar_compiler.get_cache_size_bytes()
    grammar_compiler.compile_json_schema(make_schema("name_0"), strict_mode=True)
    assert grammar_compiler.get_cache_size_bytes() == old_size


@pytest.mark.hf_token_required
def test_grammar_compiler_cache_limited():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)

    def make_schema(name_str: str):
        return {
            "properties": {name_str: {"type": "string"}},
            "required": [name_str],
            "type": "object",
        }

    MB = 1024 * 1024

    # with a 2MB limit
    limit = int(2 * MB)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info, cache_limit_bytes=limit)
    assert grammar_compiler.cache_limit_bytes == limit
    assert grammar_compiler.get_cache_size_bytes() == 0
    sum_single = 0
    for i in range(10):
        schema = make_schema(f"name_{i}")
        compiled_grammar = grammar_compiler.compile_json_schema(schema, strict_mode=True)
        sum_single += compiled_grammar.memory_size_bytes
        memory_usage = grammar_compiler.get_cache_size_bytes()
        assert 0 <= memory_usage <= min(sum_single, limit * 2)  # this is a rough estimate
        print(
            f"Cache memory usage after {i + 1} schemas: {memory_usage / MB:.3f} MB / {limit / MB:.3f} MB"
        )

    # Test clear_cache
    grammar_compiler.clear_cache()
    assert grammar_compiler.get_cache_size_bytes() == 0


if __name__ == "__main__":
    pytest.main(sys.argv)
