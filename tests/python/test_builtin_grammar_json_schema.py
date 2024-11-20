import sys
import time
from typing import Dict, List, Tuple

import pytest
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

import xgrammar as xgr


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


def test_json_schema__debug_accept_string():
    grammar = xgr.BNFGrammar.builtin_json_grammar_schema(MainModel, indent=2)

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
    matcher = xgr.GrammarMatcher(grammar, tokenizer_info)

    for c in instance_str:
        assert matcher._debug_accept_string(c)
    assert matcher.accept_token(2)
    assert matcher.is_terminated()


def test_json_schema_find_jump_forward_string():
    grammar = xgr.BNFGrammar.builtin_json_grammar_schema(MainModel, indent=2)
    matcher = xgr.GrammarMatcher(grammar)

    for i, c in enumerate(instance_str):
        jump_forward_str = matcher.find_jump_forward_string()
        assert instance_str[i : i + len(jump_forward_str)] == jump_forward_str
        assert matcher._debug_accept_string(c)
    assert matcher.find_jump_forward_string() == ""


tokenizer_path = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


@pytest.mark.parametrize("tokenizer_path", tokenizer_path)
def test_fill_next_token_bitmask(tokenizer_path: str):
    grammar = xgr.BNFGrammar.builtin_json_grammar_schema(MainModel, indent=2)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)

    time_start = time.monotonic_ns()
    matcher = xgr.GrammarMatcher(grammar, tokenizer_info)
    time_end = time.monotonic_ns()
    print(f"Time to init GrammarMatcher: {(time_end - time_start) / 1e3} us")

    token_bitmask = _allocate_token_bitmask(tokenizer_info.vocab_size)
    logits_gpu = torch.zeros(matcher.vocab_size, dtype=torch.float32, device="cuda")

    input_bytes = instance_str.encode("utf-8")

    for _, c in enumerate(input_bytes):
        # 1. fill_next_token_bitmask
        time_start = time.monotonic_ns()
        matcher.fill_next_token_bitmask(token_bitmask)
        time_end = time.monotonic_ns()
        print(f"Time to fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")

        # 2. Test run of debug_get_masked_tokens_from_bitmask
        rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask)

        # 3. apply_token_bitmask_inplace
        torch.cuda.synchronize()
        time_start = time.monotonic_ns()
        xgr.GrammarMatcher.apply_token_bitmask_inplace(logits_gpu, token_bitmask)
        torch.cuda.synchronize()
        time_end = time.monotonic_ns()
        print(f"Time to apply_token_bitmask_inplace: {(time_end - time_start) / 1e3} us")

        # 4. accept_string
        print("Accepting char:", bytes([c]))
        time_start = time.monotonic_ns()
        assert matcher._debug_accept_string(bytes([c]))
        time_end = time.monotonic_ns()
        print(f"Time to accept_token: {(time_end - time_start) / 1e3} us")

    # 5. Final correctness verification
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask)
    assert tokenizer.eos_token_id not in rejected_token_ids


if __name__ == "__main__":
    pytest.main(sys.argv)
