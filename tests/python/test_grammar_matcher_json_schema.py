import sys
import time
from typing import Dict, List, Tuple

import pytest
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.testing import (
    _get_masked_tokens_from_bitmask,
    _get_matcher_from_grammar_and_tokenizer_info,
)


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
instance_str = instance.model_dump_json(indent=2, round_trip=True)


@pytest.mark.hf_token_required
def test_json_schema_debug_accept_string():
    grammar = xgr.Grammar.from_json_schema(MainModel, indent=2)

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
    instance_str = instance.model_dump_json(indent=2, round_trip=True)

    tokenizer_path = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, tokenizer_info)

    for c in instance_str:
        assert matcher._debug_accept_string(c)
    assert matcher.accept_token(2)
    assert matcher.is_terminated()


def test_json_schema_find_jump_forward_string():
    grammar = xgr.Grammar.from_json_schema(MainModel, indent=2)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, xgr.TokenizerInfo([]))

    for i, c in enumerate(instance_str):
        jump_forward_str = matcher.find_jump_forward_string()
        assert instance_str[i : i + len(jump_forward_str)] == jump_forward_str
        assert matcher._debug_accept_string(c)
    assert matcher.find_jump_forward_string() == ""


tokenizer_path = ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct"]


@pytest.mark.hf_token_required
@pytest.mark.parametrize("tokenizer_path", tokenizer_path)
def test_fill_next_token_bitmask(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)

    time_start = time.monotonic_ns()
    compiled_grammar = compiler.compile_json_schema(MainModel, indent=2)
    matcher = xgr.GrammarMatcher(compiled_grammar)
    time_end = time.monotonic_ns()
    print(f"Time to init GrammarMatcher: {(time_end - time_start) / 1e3} us")

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    input_bytes = instance_str.encode("utf-8")

    for _, c in enumerate(input_bytes):
        # 1. fill_next_token_bitmask
        time_start = time.monotonic_ns()
        matcher.fill_next_token_bitmask(token_bitmask)
        time_end = time.monotonic_ns()
        print(f"Time to fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")

        # 2. accept_string
        print("Accepting char:", bytes([c]))
        time_start = time.monotonic_ns()
        assert matcher._debug_accept_string(bytes([c]))
        time_end = time.monotonic_ns()
        print(f"Time to accept_token: {(time_end - time_start) / 1e3} us")

    # 3. Final correctness verification
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert tokenizer.eos_token_id not in rejected_token_ids


if __name__ == "__main__":
    pytest.main(sys.argv)
