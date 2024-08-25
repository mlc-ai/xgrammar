# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=redefined-outer-name,unbalanced-tuple-unpacking
"""This test is adopted from test_grammar_state_matcher_json.py, but the grammar is parsed from
a unoptimized, non-simplified EBNF string. This is to test the robustness of the grammar state
matcher."""
import json
import sys
from typing import Dict, List, Optional, Tuple

import pytest
import tvm
import tvm.testing
from pydantic import BaseModel

from xgrammar import BNFGrammar, GrammarStateMatcher

def test_find_next_rejected_tokens_schema() -> None:
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
    ebnf_grammar = BNFGrammar.from_schema(schema_str, indent=2)

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

    tokenizer_path = "dist/Llama-2-7b-chat-hf-q4f16_1-MLC"
    tokenizer = Tokenizer(tokenizer_path)
    matcher = GrammarStateMatcher(ebnf_grammar, tokenizer)

    for c in instance_str:
        matcher.find_next_rejected_tokens(True)
        print("Accepting char:", c, file=sys.stderr)
        assert matcher.debug_accept_char(ord(c))
    assert 2 not in matcher.find_next_rejected_tokens(True)


def test_find_jump_forward_string_schema():
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
    ebnf_grammar = BNFGrammar.from_schema(schema_str, indent=2)

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

    tokenizer_path = "dist/Llama-2-7b-chat-hf-q4f16_1-MLC"
    tokenizer = Tokenizer(tokenizer_path)
    matcher = GrammarStateMatcher(ebnf_grammar, tokenizer)

    for i, c in enumerate(instance_str):
        jump_forward_str = matcher.find_jump_forward_string()
        print(f"Jump forward string at {i}: {jump_forward_str}")
        assert instance_str[i : i + len(jump_forward_str)] == jump_forward_str
        print("Accepting char:", c, file=sys.stderr)
        assert matcher.debug_accept_char(ord(c))
    assert matcher.find_jump_forward_string() == ""


if __name__ == "__main__":
    tvm.testing.main()
