"""This test uses the optimized JSON grammar provided by the grammar library."""

import time
from typing import Dict, List, Tuple

import pytest
from pydantic import BaseModel
from transformers import AutoTokenizer

from xgrammar import (
    BuiltinGrammar,
    GrammarMatcher,
    GrammarMatcherInitContext,
    TokenizerInfo,
)
from xgrammar.xgrammar import GrammarMatcherInitContextCache


def test_init_context():
    grammar = BuiltinGrammar.json()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    time_start = time.monotonic_ns()
    context = GrammarMatcherInitContext(grammar, tokenizer_info)
    time_end = time.monotonic_ns()
    print(f"Time to init context: {(time_end - time_start) / 1e3} us")

    def check_matcher(matcher: GrammarMatcher):
        assert matcher.vocab_size == 32000
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


def test_init_context_cache_json():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    time_start = time.monotonic_ns()
    init_context_cache = GrammarMatcherInitContextCache(tokenizer_info)
    time_end = time.monotonic_ns()
    print(f"Time to init context cache: {(time_end - time_start) / 1e3} us")

    def check_matcher(matcher: GrammarMatcher):
        assert matcher.vocab_size == 32000
        assert not matcher.is_terminated()
        assert not matcher.accept_string('{ name: "John" }')
        assert matcher.accept_string('{"name": "John"}')
        assert matcher.is_terminated()

    time_start = time.monotonic_ns()
    init_context = init_context_cache.get_init_context_for_json()
    time_end = time.monotonic_ns()
    print(f"Time to get init context 1: {(time_end - time_start) / 1e3} us")
    matcher = GrammarMatcher(init_context, terminate_without_stop_token=True)
    check_matcher(matcher)

    time_start = time.monotonic_ns()
    init_context = init_context_cache.get_init_context_for_json()
    time_end = time.monotonic_ns()
    print(f"Time to get init context 2: {(time_end - time_start) / 1e3} us")
    matcher = GrammarMatcher(init_context, terminate_without_stop_token=True)
    check_matcher(matcher)


def test_init_context_cache_json_schema():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    init_context_cache = GrammarMatcherInitContextCache(tokenizer_info)

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
        init_context = init_context_cache.get_init_context_for_json_schema(
            MainModel, indent=indent, separators=separators
        )
        time_end = time.monotonic_ns()
        print(f"Time to get init context {test_id}: {(time_end - time_start) / 1e3} us")
        matcher = GrammarMatcher(init_context, terminate_without_stop_token=True)

        assert matcher.vocab_size == 32000
        assert not matcher.is_terminated()
        assert matcher.accept_string(instance_str)
        assert matcher.is_terminated()

    check_with_fmt(None, (",", ":"), "1")
    check_with_fmt(None, (",", ":"), "2")
    check_with_fmt(2, None, "3")
    check_with_fmt(2, (",", ": "), "4")


if __name__ == "__main__":
    pytest.main([__file__])
