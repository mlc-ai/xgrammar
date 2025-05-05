import json
import sys
import time
from typing import Dict, List, Tuple

import pytest
from pydantic import BaseModel, Field
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


class RangeSchema(BaseModel):
    value: int = Field(ge=1, le=100)


class ExtendedRangeSchema(BaseModel):
    value: int = Field(ge=-128, le=256)


class NegativeRangeSchema(BaseModel):
    value: int = Field(ge=-1000, le=-1)


class LargeRangeSchema(BaseModel):
    value: int = Field(ge=-99999, le=99999)


class FloatRangeSchema(BaseModel):
    value: float = Field(ge=0.0, le=1.0)


class NegativeFloatRangeSchema(BaseModel):
    value: float = Field(ge=-10.0, le=-0.1)


class ComplexFloatRangeSchema(BaseModel):
    value: float = Field(ge=-12345.12345, le=56789.56789)


class LargeFloatRangeSchema(BaseModel):
    value: float = Field(ge=-1000.0, le=1000.0)


class MultipleBoundariesSchema(BaseModel):
    small_value: int = Field(ge=-10, le=10)
    medium_value: int = Field(ge=-100, le=100)
    large_value: int = Field(ge=-1000, le=1000)


class MixedTypeRangeSchema(BaseModel):
    int_value: int = Field(ge=-100, le=100)
    float_value: float = Field(ge=-10.0, le=10.0)


@pytest.mark.parametrize("tokenizer_path", tokenizer_path)
@pytest.mark.parametrize(
    "schema_class,test_value",
    [
        # Integer test cases
        (RangeSchema, 42),
        (ExtendedRangeSchema, -128),
        (ExtendedRangeSchema, 0),
        (ExtendedRangeSchema, 256),
        (ExtendedRangeSchema, 14),
        (NegativeRangeSchema, -1000),
        (NegativeRangeSchema, -500),
        (NegativeRangeSchema, -1),
        (LargeRangeSchema, -99999),
        (LargeRangeSchema, -5678),
        (LargeRangeSchema, 0),
        (LargeRangeSchema, 5678),
        (LargeRangeSchema, 99999),
        # Float test cases
        (FloatRangeSchema, 0.0),
        (FloatRangeSchema, 0.5),
        (FloatRangeSchema, 1.0),
        (NegativeFloatRangeSchema, -10.0),
        (NegativeFloatRangeSchema, -5.5),
        (NegativeFloatRangeSchema, -0.1),
        (LargeFloatRangeSchema, -1000.0),
        (LargeFloatRangeSchema, -500.5),
        (LargeFloatRangeSchema, 0.0),
        (LargeFloatRangeSchema, 500.5),
        (LargeFloatRangeSchema, 1000.0),
        (ComplexFloatRangeSchema, (-1234.1234)),
        (ComplexFloatRangeSchema, (0)),
        (ComplexFloatRangeSchema, (5671.123456)),
    ],
)
@pytest.mark.hf_token_required
def test_fill_next_token_bitmask_intfloat_range(tokenizer_path: str, schema_class, test_value):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)

    instance = schema_class(value=test_value)
    instance_str = instance.model_dump_json()

    print(f"Testing {schema_class.__name__} with value {test_value}")

    time_start = time.monotonic_ns()
    compiled_grammar = compiler.compile_json_schema(schema_class)
    matcher = xgr.GrammarMatcher(compiled_grammar)
    time_end = time.monotonic_ns()
    print(f"Time to init GrammarMatcher: {(time_end - time_start) / 1e3} us")

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    input_bytes = instance_str.encode("utf-8")
    for c in input_bytes:
        time_start = time.monotonic_ns()
        matcher.fill_next_token_bitmask(token_bitmask)
        time_end = time.monotonic_ns()
        print(f"Time to fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")

        assert matcher._debug_accept_string(bytes([c]))

    matcher.fill_next_token_bitmask(token_bitmask)
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert tokenizer.eos_token_id not in rejected_token_ids


@pytest.mark.hf_token_required
@pytest.mark.parametrize("tokenizer_path", tokenizer_path)
def test_mixed_type_range_schema(tokenizer_path: str):
    """Test the MixedTypeRangeSchema with both integer and float fields"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)

    test_instances = [
        MixedTypeRangeSchema(int_value=-100, float_value=-10.0),
        MixedTypeRangeSchema(int_value=100, float_value=10.0),
        MixedTypeRangeSchema(int_value=0, float_value=0.0),
        MixedTypeRangeSchema(int_value=-50, float_value=5.5),
    ]

    for instance in test_instances:
        instance_str = instance.model_dump_json()

        print(f"Testing MixedTypeRangeSchema with values: {instance}")

        time_start = time.monotonic_ns()
        compiled_grammar = compiler.compile_json_schema(MixedTypeRangeSchema)
        matcher = xgr.GrammarMatcher(compiled_grammar)
        time_end = time.monotonic_ns()
        print(f"Time to init GrammarMatcher: {(time_end - time_start) / 1e3} us")

        token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

        input_bytes = instance_str.encode("utf-8")
        for c in input_bytes:
            time_start = time.monotonic_ns()
            matcher.fill_next_token_bitmask(token_bitmask)
            time_end = time.monotonic_ns()
            print(f"Time to fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")

            assert matcher._debug_accept_string(bytes([c]))

        matcher.fill_next_token_bitmask(token_bitmask)
        rejected_token_ids = _get_masked_tokens_from_bitmask(
            token_bitmask, tokenizer_info.vocab_size
        )
        assert tokenizer.eos_token_id not in rejected_token_ids


@pytest.mark.hf_token_required
@pytest.mark.parametrize("tokenizer_path", tokenizer_path)
def test_multiple_boundaries_schema(tokenizer_path: str):
    """Test the complex MultipleBoundariesSchema with multiple integer fields"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)

    test_instances = [
        MultipleBoundariesSchema(
            small_value=-10, medium_value=-100, large_value=-1000
        ),  # All lower bounds
        MultipleBoundariesSchema(
            small_value=10, medium_value=100, large_value=1000
        ),  # All upper bounds
        MultipleBoundariesSchema(small_value=0, medium_value=0, large_value=0),
        MultipleBoundariesSchema(small_value=-5, medium_value=50, large_value=-500),
    ]

    for instance in test_instances:
        instance_str = instance.model_dump_json()

        print(f"Testing MultipleBoundariesSchema with values: {instance}")

        time_start = time.monotonic_ns()
        compiled_grammar = compiler.compile_json_schema(MultipleBoundariesSchema)
        matcher = xgr.GrammarMatcher(compiled_grammar)
        time_end = time.monotonic_ns()
        print(f"Time to init GrammarMatcher: {(time_end - time_start) / 1e3} us")

        token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

        input_bytes = instance_str.encode("utf-8")
        for c in input_bytes:
            time_start = time.monotonic_ns()
            matcher.fill_next_token_bitmask(token_bitmask)
            time_end = time.monotonic_ns()
            print(f"Time to fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")

            assert matcher._debug_accept_string(bytes([c]))

        matcher.fill_next_token_bitmask(token_bitmask)
        rejected_token_ids = _get_masked_tokens_from_bitmask(
            token_bitmask, tokenizer_info.vocab_size
        )
        assert tokenizer.eos_token_id not in rejected_token_ids


string_format_instances = [
    (r"long.email-address-with-hyphens@and.subdomains.example.com", "email"),
    (r'"very.(),:;<>[]\".VERY.\"very@\\ \"very\".unusual"@strange.example.com', "email"),
    (r"128.255.000.222", "ipv4"),
    (r"2001:db8:3:4::192.0.2.33", "ipv6"),
    (r"P1Y23M456DT9H87M654S", "duration"),
    (r"2025-01-01T12:34:56.7+08:09", "date-time"),
    (r"123--abc.efgh---789-xyz.rst-uvw", "hostname"),
    (r"01234567-89AB-CDEF-abcd-ef0123456789", "uuid"),
    (
        r"http://azAZ09-._~%Ff!$&'()*+,;=:@xyz:987/-/./+/*?aA0-._~%Ff!$&'()@#zZ9-._~%Aa!$&,;=:",
        "uri",
    ),
]

# not frequently used
string_format_instances_skipped = [
    (
        r"//azAZ09-._~%Ff!$&'()*+,;=:@xyz:987/-/./+/*?aA0-._~%Ff!$&'()@#zZ9-._~%Aa!$&,;=:",
        "uri-reference",
    ),
    (r"!#$&()*+,-./{+abc}{#def}{.ghi}{/jkl}{;mno:2468}", "uri-template"),
    (r"/a/bc/def/ghij/~0~1//", "json-pointer"),
    (r"1234/a/bc/def/ghij/~0~1//", "relative-json-pointer"),
]


@pytest.mark.hf_token_required
@pytest.mark.parametrize("value, format", string_format_instances)
def test_mask_generation_format(value: str, format: str):
    class MainModel(BaseModel):
        name: str = Field(json_schema_extra={"format": format})

    instance = json.dumps(MainModel(name=value).model_dump(mode="json"))

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)

    time_start = time.monotonic_ns()
    compiled_grammar = grammar_compiler.compile_json_schema(MainModel)
    time_end = time.monotonic_ns()
    print(f"Time for preprocessing: {(time_end - time_start) / 1e3} us")

    matcher = xgr.GrammarMatcher(compiled_grammar)
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    for c in instance.encode("utf-8"):
        time_start = time.monotonic_ns()
        matcher.fill_next_token_bitmask(token_bitmask)
        time_end = time.monotonic_ns()
        delta_us = (time_end - time_start) / 1e3
        print(f"Time for fill_next_token_bitmask: {delta_us} us before accepting char {bytes([c])}")
        accepted = matcher._debug_accept_string(bytes([c]))
        assert accepted

    time_start = time.monotonic_ns()
    matcher.fill_next_token_bitmask(token_bitmask)
    time_end = time.monotonic_ns()
    print(f"Time for fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")

    assert matcher.accept_token(tokenizer.eos_token_id)
    assert matcher.is_terminated()


if __name__ == "__main__":
    pytest.main(sys.argv)
