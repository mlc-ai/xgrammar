from typing import Any, List, Literal, Optional, Tuple

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

serialize_name__args: List[Tuple[str, Tuple]] = [
    ("json_schema", (make_trivial_schema(0),)),
    ("structural_tag", (EXAMPLE_TAGS, EXAMPLE_TRIGGERS)),
    ("grammar", (JSON_GRAMMAR,)),
    ("builtin_json_grammar", ()),
    ("regex", (EMAIL_REGEX,)),
]

tokenizer_paths = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
]

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
def test_serialize_compiled_grammar_string():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    date_regex = r"\d{4}-\d{2}-\d{2}"

    class GrammarModel(BaseModel):
        rules_: List[Any]
        rule_expr_data_: List[int]
        rule_expr_indptr_: List[int]
        root_rule_id_: int
        root_tag_dispatch_fsm: Any
        tag_dispatch_end_node_to_rule_id: List[int]
        allow_empty_rule_ids: List[int]

    class TokenizerInfoMetaData(BaseModel):
        vocab_type: int
        vocab_size: int
        add_prefix_space: bool
        stop_token_ids: List[int]
        special_token_ids: List[int]

    class Output(BaseModel):
        grammar: GrammarModel
        tokenizer_metadata: TokenizerInfoMetaData
        adaptive_token_mask_cache: List
        __VERSION__: Literal["v1"]

    # test serialization and deserialization in practice
    grammar = grammar_compiler.compile_regex(date_regex)
    json_str = grammar.serialize_json()
    Output.model_validate_json(json_str)


@pytest.mark.hf_token_required
@pytest.mark.parametrize("name, args", serialize_name__args)
def test_serialize_compiled_grammar_roundtrip(name: str, args: Tuple):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

    method_name = f"compile_{name}"
    grammar: xgr.CompiledGrammar = getattr(grammar_compiler, method_name)(*args)

    serialized_json = grammar.serialize_json()
    deserialized_obj = xgr.CompiledGrammar.deserialize_json(serialized_json, tokenizer_info)
    serialized_json_new = deserialized_obj.serialize_json()
    assert serialized_json == serialized_json_new, f"Serialized JSON mismatch for {name}."


@pytest.mark.hf_token_required
@pytest.mark.parametrize("tokenizer_path", tokenizer_paths)
def test_serialize_tokenizer_info_roundtrip(tokenizer_path: str):
    # test serialization and deserialization in practice
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)

    # tokenizer roundtrip
    serialized_tokenizer = tokenizer_info.serialize_json()
    deserialized_tokenizer_info = xgr.TokenizerInfo.deserialize_json(serialized_tokenizer, [])
    assert (
        tokenizer_info.vocab_type == deserialized_tokenizer_info.vocab_type
        and tokenizer_info.vocab_size == deserialized_tokenizer_info.vocab_size
        and tokenizer_info.add_prefix_space == deserialized_tokenizer_info.add_prefix_space
        and tokenizer_info.stop_token_ids == deserialized_tokenizer_info.stop_token_ids
        and tokenizer_info.special_token_ids == deserialized_tokenizer_info.special_token_ids
    ), "Tokenizer info mismatch after serialization and deserialization."


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

    # copied from test_grammar_matcher_basic.py
    from xgrammar.testing import _get_masked_tokens_from_bitmask

    # go through a roundtrip for the grammar object
    serialized = grammar_compiler.compile_grammar(
        xgr.Grammar.builtin_json_grammar()
    ).serialize_json()
    deserialized = xgr.CompiledGrammar.deserialize_json(serialized, tokenizer_info)

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
