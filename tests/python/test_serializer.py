import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pytest
from pydantic import BaseModel
from transformers import AutoTokenizer  # type: ignore

import xgrammar as xgr

TOKENIZER_PATH = "meta-llama/Llama-3.1-8B-Instruct"


def make_trivial_schema(num: int):
    name_int = f"schema_int_{num}"
    name_str = f"schema_str_{num}"
    return {
        "properties": {name_int: {"type": "integer"}, name_str: {"type": "string"}},
        "required": [name_int, name_str],
        "type": "object",
    }


JSON_GRAMMAR = r"""
basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_prop_3 ::= (("[" "" basic_any (", " basic_any)* "" "]") | ("[" "" "]"))
root_prop_4 ::= (("[" "" basic_string (", " basic_string)* "" "]") | ("[" "" "]"))
root_prop_5_item_2 ::= (("[" "" basic_string (", " basic_string)* "" "]") | ("[" "" "]"))
root_prop_5 ::= ("[" "" (basic_string ", " basic_integer ", " root_prop_5_item_2) "" "]")
root_prop_6 ::= ("{" "" basic_string ": " basic_integer (", " basic_string ": " basic_integer)* "" "}") | "{" "}"
root_prop_7_addl ::= ("{" "" basic_string ": " basic_integer (", " basic_string ": " basic_integer)* "" "}") | "{" "}"
root_prop_7 ::= ("{" "" basic_string ": " root_prop_7_addl (", " basic_string ": " root_prop_7_addl)* "" "}") | "{" "}"
root ::= "{" "" "\"integer_field\"" ": " basic_integer ", " "\"number_field\"" ": " basic_number ", " "\"boolean_field\"" ": " basic_boolean ", " "\"any_array_field\"" ": " root_prop_3 ", " "\"array_field\"" ": " root_prop_4 ", " "\"tuple_field\"" ": " root_prop_5 ", " "\"object_field\"" ": " root_prop_6 ", " "\"nested_object_field\"" ": " root_prop_7 "" "}"
"""

EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}"


class ExampleModel(BaseModel):
    id: int
    name: str


EXAMPLE_TAGS = [
    xgr.StructuralTagItem(begin="<function=f>", schema=ExampleModel, end="</function>"),
    xgr.StructuralTagItem(begin="<function=g>", schema=ExampleModel, end="</function>"),
]

EXAMPLE_TRIGGERS = ["<function=f", "<function=g"]


@pytest.mark.hf_token_required
def test_serializer_roundtrip():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    serializer = xgr.JSONSerializer()

    def _serialize_test(grammar: xgr.CompiledGrammar, name: str):
        tic = time.monotonic_ns()
        serialized_json = serializer.serialize_compiled_grammar(grammar)
        toc = time.monotonic_ns()
        print(f"Serialization {name}: {(toc - tic) / 1_000_000:.2f} ms")
        tic = time.monotonic_ns()
        deserialized_obj = serializer.deserialize_compiled_grammar(serialized_json, tokenizer_info)
        toc = time.monotonic_ns()
        print(f"Deserialization {name}: {(toc - tic) / 1_000_000:.2f} ms")
        serialized_json_new = serializer.serialize_compiled_grammar(deserialized_obj)
        return serialized_json == serialized_json_new

    # skip the first serialization and deserialization to warm up
    _ = serializer.deserialize_compiled_grammar(
        serializer.serialize_compiled_grammar(grammar_compiler.compile_builtin_json_grammar()),
        tokenizer_info,
    )

    grammar = grammar_compiler.compile_json_schema(make_trivial_schema(0))
    _serialize_test(grammar, f"JSON schema")

    grammar = grammar_compiler.compile_structural_tag(EXAMPLE_TAGS, EXAMPLE_TRIGGERS)
    _serialize_test(grammar, "structural tag")

    grammar = grammar_compiler.compile_grammar(JSON_GRAMMAR)
    _serialize_test(grammar, "simple JSON EBNF")

    grammar = grammar_compiler.compile_builtin_json_grammar()
    _serialize_test(grammar, "builtin JSON EBNF")

    grammar = grammar_compiler.compile_regex(EMAIL_REGEX)
    _serialize_test(grammar, "email regex")

    # test the overall serialization time
    schemas = [grammar_compiler.compile_json_schema(make_trivial_schema(i)) for i in range(100)]
    tic = time.monotonic_ns()
    results = [serializer.serialize_compiled_grammar(schema) for schema in schemas]
    toc = time.monotonic_ns()
    print(f"Serialization of 100 simple schemas: {(toc - tic) / 1_000_000:.2f} ms")
    tic = time.monotonic_ns()
    schemas_new = [
        serializer.deserialize_compiled_grammar(result, tokenizer_info) for result in results
    ]
    toc = time.monotonic_ns()
    print(f"Deserialization of 100 simple schemas: {(toc - tic) / 1_000_000:.2f} ms")
    results_new = [serializer.serialize_compiled_grammar(schema) for schema in schemas_new]
    assert results == results_new, "Serialized and deserialized schemas do not match"


tokenizer_path__input_str__expected_rejected_sizes = [
    (
        "meta-llama/Llama-2-7b-chat-hf",
        '{"id": 1,"name": "Example"}',
        [
            # fmt: off
            31989, 31912, 270, 270, 270, 31973, 31846, 31846, 31948, 31915, 270, 270, 270, 270,
            270, 31973, 31846, 31846, 263, 263, 263, 263, 263, 263, 263, 263, 31974, 31999,
            # fmt: on
        ],
    ),
    (
        # test for llama 3
        "meta-llama/Meta-Llama-3-8B-Instruct",
        '{"id": 1,"name": "Example哈哈"}',
        [
            # fmt: off
            128235, 127497, 4744, 4744, 4744, 127849, 126399, 126399, 126760, 127499, 4744, 4744,
            4744, 4744, 4744, 127849, 126399, 126399, 4694, 4694, 4694, 4694, 4694, 4694, 4694,
            4694, 128066, 128111, 4694, 128066, 128111, 4694, 127873, 128255,
            # fmt: on
        ],
    ),
]


@pytest.mark.hf_token_required
@pytest.mark.parametrize(
    "tokenizer_path, input_str, expected_rejected_sizes",
    tokenizer_path__input_str__expected_rejected_sizes,
)
def test_serializer_correctness_functional(
    tokenizer_path: str, input_str: str, expected_rejected_sizes: Optional[List[int]]
):
    # test serialization and deserialization in practice
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    serializer = xgr.JSONSerializer()

    # copied from test_grammar_matcher_basic.py
    from xgrammar.testing import _get_masked_tokens_from_bitmask

    json_grammar = xgr.Grammar.builtin_json_grammar()
    grammar = grammar_compiler.compile_grammar(json_grammar)
    serialized = serializer.serialize_compiled_grammar(grammar)
    deserialized = serializer.deserialize_compiled_grammar(serialized, tokenizer_info)
    matcher = xgr.GrammarMatcher(deserialized)
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    input_bytes = input_str.encode("utf-8")
    rejected_sizes = []
    for i, c in enumerate(input_bytes):
        matcher.fill_next_token_bitmask(token_bitmask)
        rejected_token_ids = _get_masked_tokens_from_bitmask(
            token_bitmask, tokenizer_info.vocab_size
        )
        rejected_sizes.append(len(rejected_token_ids))
        if expected_rejected_sizes is not None:
            assert rejected_sizes[-1] == expected_rejected_sizes[i], (
                rejected_sizes[-1],
                expected_rejected_sizes[i],
            )
        assert matcher.accept_string(bytes([c]))

    matcher.fill_next_token_bitmask(token_bitmask)
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    rejected_sizes.append(len(rejected_token_ids))
    if expected_rejected_sizes is not None:
        assert rejected_sizes[-1] == expected_rejected_sizes[-1]
