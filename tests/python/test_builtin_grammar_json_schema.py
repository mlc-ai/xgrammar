# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=redefined-outer-name,unbalanced-tuple-unpacking
"""This test is adopted from test_grammar_state_matcher_json.py, but the grammar is parsed from
a unoptimized, non-simplified EBNF string. This is to test the robustness of the grammar state
matcher."""
import json
from typing import Dict, List, Tuple

import pytest
from pydantic import BaseModel
from transformers import AutoTokenizer

from xgrammar import BNFGrammar, GrammarStateMatcher
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

    schema = MainModel.model_json_schema()
    schema_str = json.dumps(schema)
    grammar = BuiltinGrammar.json_schema(schema_str, indent=2)

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
    matcher = GrammarStateMatcher(grammar, tokenizer)

    for c in instance_str:
        matcher.find_next_token_bitmask()
        assert matcher._accept_string(c)
    final_bitmask = matcher.find_next_token_bitmask()
    final_rejected_tokens = GrammarStateMatcher.get_rejected_tokens_from_bitmask(
        final_bitmask, matcher.vocab_size
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

    schema = MainModel.model_json_schema()
    schema_str = json.dumps(schema)
    grammar = BuiltinGrammar.json_schema(schema_str, indent=2)

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
    matcher = GrammarStateMatcher(grammar, tokenizer)

    for i, c in enumerate(instance_str):
        jump_forward_str = matcher.find_jump_forward_string()
        assert instance_str[i : i + len(jump_forward_str)] == jump_forward_str
        assert matcher._accept_string(c)
    assert matcher.find_jump_forward_string() == ""


if __name__ == "__main__":
    pytest.main([__file__])
