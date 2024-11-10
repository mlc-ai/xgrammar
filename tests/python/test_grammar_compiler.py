"""This test uses the optimized JSON grammar provided by the grammar library."""

import sys
import threading
import time
from typing import Dict, List, Tuple

import pytest
from pydantic import BaseModel
from transformers import AutoTokenizer

from xgrammar import BuiltinGrammar, CompiledGrammar, GrammarMatcher, TokenizerInfo
from xgrammar.xgrammar import CachedGrammarCompiler


def test_compiled_grammar():
    grammar = BuiltinGrammar.json()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    time_start = time.monotonic_ns()
    context = CompiledGrammar(grammar, tokenizer_info)
    time_end = time.monotonic_ns()
    print(f"Time to get compiled grammar: {(time_end - time_start) / 1e3} us")

    def check_matcher(matcher: GrammarMatcher):
        assert matcher.mask_vocab_size == 32000
        assert not matcher.is_terminated()
        assert not matcher.accept_string('{ name: "John" }')
        assert matcher.accept_string('{"name": "John"}')
        assert matcher.is_terminated()

    time_start = time.monotonic_ns()
    matcher_1 = GrammarMatcher(context, terminate_without_stop_token=True)
    time_end = time.monotonic_ns()
    print(f"Time to init matcher 1: {(time_end - time_start) / 1e3} us")
    check_matcher(matcher_1)
    time_start = time.monotonic_ns()
    matcher_2 = GrammarMatcher(context, terminate_without_stop_token=True)
    time_end = time.monotonic_ns()
    print(f"Time to init matcher 2: {(time_end - time_start) / 1e3} us")
    check_matcher(matcher_2)


def test_cached_grammar_compiler_json():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    time_start = time.monotonic_ns()
    cached_grammar_compiler = CachedGrammarCompiler(tokenizer_info)
    time_end = time.monotonic_ns()
    print(f"Time to init cached grammar compiler: {(time_end - time_start) / 1e3} us")

    def check_matcher(matcher: GrammarMatcher):
        assert matcher.mask_vocab_size == 32000
        assert not matcher.is_terminated()
        assert not matcher.accept_string('{ name: "John" }')
        assert matcher.accept_string('{"name": "John"}')
        assert matcher.is_terminated()

    time_start = time.monotonic_ns()
    compiled_grammar = cached_grammar_compiler.get_compiled_grammar_for_json()
    time_end = time.monotonic_ns()
    print(f"Time to get compiled grammar: {(time_end - time_start) / 1e3} us")
    matcher = GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    check_matcher(matcher)

    time_start = time.monotonic_ns()
    compiled_grammar = cached_grammar_compiler.get_compiled_grammar_for_json()
    time_end = time.monotonic_ns()
    print(f"Time to get compiled grammar again: {(time_end - time_start) / 1e3} us")
    matcher = GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    check_matcher(matcher)

    cached_grammar_compiler.clear()

    time_start = time.monotonic_ns()
    compiled_grammar = cached_grammar_compiler.get_compiled_grammar_for_json()
    time_end = time.monotonic_ns()
    print(f"Time to get compiled grammar after clear: {(time_end - time_start) / 1e3} us")
    matcher = GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    check_matcher(matcher)


def test_cached_grammar_compiler_json_schema():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    cached_grammar_compiler = CachedGrammarCompiler(tokenizer_info)

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

    def check_with_fmt(indent, separators, test_id):
        instance_str = instance.model_dump_json(indent=indent, round_trip=True)

        time_start = time.monotonic_ns()
        compiled_grammar = cached_grammar_compiler.get_compiled_grammar_for_json_schema(
            MainModel, indent=indent, separators=separators
        )
        time_end = time.monotonic_ns()
        print(f"Time to get compiled grammar {test_id}: {(time_end - time_start) / 1e3} us")
        matcher = GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)

        assert matcher.mask_vocab_size == 32000
        assert not matcher.is_terminated()
        assert matcher.accept_string(instance_str)
        assert matcher.is_terminated()

    check_with_fmt(None, (",", ":"), "1")
    check_with_fmt(None, (",", ":"), "2")
    check_with_fmt(2, None, "3")
    check_with_fmt(2, (",", ": "), "4")

    cached_grammar_compiler.clear()

    check_with_fmt(None, (",", ":"), "5")


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


def test_cached_grammar_compiler_json_schema_concurrent():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    cached_grammar_compiler = CachedGrammarCompiler(tokenizer_info)

    def check_matcher(matcher: GrammarMatcher, instance_str: str):
        assert matcher.mask_vocab_size == 32000
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
        compiled_grammar = cached_grammar_compiler.get_compiled_grammar_for_json_schema(
            schema, indent=None, separators=(",", ":"), strict_mode=True
        )
        time_end = time.monotonic_ns()
        print(f"Thread {id} end compile grammar {schema_id}: {(time_end - time_start) / 1e3} us")
        matcher = GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        check_matcher(matcher, instance_str)

    time_start = time.monotonic_ns()
    for i in range(thread_cnt):
        t = threading.Thread(target=compile_grammar, args=(i, *schema_instances[i % num_schemas]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    pytest.main(sys.argv)
