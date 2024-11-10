import sys
from typing import Dict, List, Tuple

import pytest
from pydantic import BaseModel
from transformers import AutoTokenizer

from xgrammar import GrammarMatcher
from xgrammar.xgrammar import BuiltinGrammar


def test_json_schema_accept_find_token():
    class MainModel(BaseModel):
        integer_field: int
        number_field: float
        boolean_field: bool
        any_array_field: List
        array_field: List[str]
        tuple_field: Tuple[str, int, List[str]]
        object_field: Dict[str, int]
        nested_object_field: Dict[str, Dict[str, int]]

    grammar = BuiltinGrammar.json_schema(MainModel, indent=2)

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
    matcher = GrammarMatcher(grammar, tokenizer)

    for c in instance_str:
        matcher.get_next_token_bitmask()
        assert matcher.accept_string(c)
    final_bitmask = matcher.get_next_token_bitmask()
    final_rejected_tokens = GrammarMatcher.get_rejected_tokens_from_bitmask(
        final_bitmask, matcher.mask_vocab_size
    )
    assert 2 not in final_rejected_tokens
    assert matcher.accept_token(2)
    assert matcher.is_terminated()


def test_json_schema_find_jump_forward_string():
    class MainModel(BaseModel):
        integer_field: int
        number_field: float
        boolean_field: bool
        any_array_field: List
        array_field: List[str]
        tuple_field: Tuple[str, int, List[str]]
        object_field: Dict[str, int]
        nested_object_field: Dict[str, Dict[str, int]]

    grammar = BuiltinGrammar.json_schema(MainModel, indent=2)

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
    matcher = GrammarMatcher(grammar, tokenizer)

    for i, c in enumerate(instance_str):
        jump_forward_str = matcher.find_jump_forward_string()
        assert instance_str[i : i + len(jump_forward_str)] == jump_forward_str
        assert matcher.accept_string(c)
    assert matcher.find_jump_forward_string() == ""


if __name__ == "__main__":
    pytest.main(sys.argv)
