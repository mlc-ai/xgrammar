import json
import sys
import time
from typing import Dict, List, Tuple

import pytest
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoTokenizer

import xgrammar as xgr
from xgrammar.testing import (
    _get_masked_tokens_from_bitmask,
    _get_matcher_from_grammar_and_tokenizer_info,
    _is_grammar_accept_string,
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
        assert matcher.accept_string(c)
    assert matcher.accept_token(2)
    assert matcher.is_terminated()


def test_json_schema_find_jump_forward_string():
    grammar = xgr.Grammar.from_json_schema(MainModel, indent=2)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, xgr.TokenizerInfo([]))

    for i, c in enumerate(instance_str):
        jump_forward_str = matcher.find_jump_forward_string()
        assert instance_str[i : i + len(jump_forward_str)] == jump_forward_str
        assert matcher.accept_string(c)
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
        assert matcher.accept_string(bytes([c]))
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


class LargeRangeSchemaStartZero(BaseModel):
    value: int = Field(ge=0, le=20_000_000_000)


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


class VeryLargeFloatRangeSchema(BaseModel):
    value: float = Field(ge=-20_000_000_000.123123, le=20_000_000_000.456789)


class ExceedsInt64MaxSchema(BaseModel):
    value: int = Field(ge=0, le=18446744073709551615)


class ExceedsInt64MinSchema(BaseModel):
    value: int = Field(ge=-9223372036854775809, le=100)


class ExceedsInt64RangeSchema(BaseModel):
    value: int = Field(ge=-18446744073709551616, le=18446744073709551616)


class ValidInt64MaxSchema(BaseModel):
    value: int = Field(ge=0, le=9223372036854775807)


class ValidInt64MinSchema(BaseModel):
    value: int = Field(ge=-9223372036854775808, le=0)


class ValidLargeIntSchema(BaseModel):
    value: int = Field(ge=0, le=1000000000000000000)


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
        (LargeRangeSchemaStartZero, 20000000000),
        (LargeRangeSchemaStartZero, 0),
        (LargeRangeSchemaStartZero, 10000000000),
        (LargeRangeSchemaStartZero, 19999999999),
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
        (VeryLargeFloatRangeSchema, (20_000_000_000.456788)),
        (VeryLargeFloatRangeSchema, (-19_999_999_999.456789)),
        # Signed 64-bit boundary test cases (should succeed)
        (ValidInt64MaxSchema, 9223372036854775807),
        (ValidInt64MaxSchema, 1000),
        (ValidInt64MinSchema, -9223372036854775808),
        (ValidInt64MinSchema, -1000),
        (ValidLargeIntSchema, 1000000000000000000),
        (ValidLargeIntSchema, 1000),
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

        assert matcher.accept_string(bytes([c]))

    matcher.fill_next_token_bitmask(token_bitmask)
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert tokenizer.eos_token_id not in rejected_token_ids


@pytest.mark.parametrize("tokenizer_path", tokenizer_path)
@pytest.mark.parametrize(
    "schema_class,should_fail,error_pattern",
    [
        (ExceedsInt64MaxSchema, True, "exceeds"),
        (ExceedsInt64MinSchema, True, "exceeds"),
        (ExceedsInt64RangeSchema, True, "exceeds"),
    ],
)
@pytest.mark.hf_token_required
def test_64bit_limit_validation(
    tokenizer_path: str, schema_class, should_fail: bool, error_pattern: str
):
    """Test that schemas exceeding signed 64-bit integer limits are properly rejected"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)

    if should_fail:
        with pytest.raises((ValueError, OverflowError, RuntimeError)) as exc_info:
            compiler.compile_json_schema(schema_class)

        assert error_pattern.lower() in str(exc_info.value).lower()


@pytest.mark.parametrize("tokenizer_path", tokenizer_path)
@pytest.mark.parametrize(
    "boundary_value,schema_class",
    [
        (9223372036854775807, ValidInt64MaxSchema),
        (-9223372036854775808, ValidInt64MinSchema),
        (1000000000000000000, ValidLargeIntSchema),
    ],
)
@pytest.mark.hf_token_required
def test_signed_64bit_boundary_values_work(tokenizer_path: str, boundary_value: int, schema_class):
    """Test that signed 64-bit boundary values work correctly"""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)

    try:
        compiled_grammar = compiler.compile_json_schema(schema_class)
        matcher = xgr.GrammarMatcher(compiled_grammar)

        test_value = min(abs(boundary_value), 1000) if boundary_value != 0 else 1000
        if boundary_value < 0:
            test_value = -test_value
        test_instance = schema_class(value=test_value)
        instance_str = test_instance.model_dump_json()

        token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        for c in instance_str.encode("utf-8"):
            matcher.fill_next_token_bitmask(token_bitmask)
            assert matcher.accept_string(bytes([c]))

    except Exception as e:
        pytest.fail(f"Signed 64-bit boundary value {boundary_value} unexpectedly failed: {e}")


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

            assert matcher.accept_string(bytes([c]))

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

            assert matcher.accept_string(bytes([c]))

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
        accepted = matcher.accept_string(bytes([c]))
        assert accepted

    time_start = time.monotonic_ns()
    matcher.fill_next_token_bitmask(token_bitmask)
    time_end = time.monotonic_ns()
    print(f"Time for fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")

    assert matcher.accept_token(tokenizer.eos_token_id)
    assert matcher.is_terminated()


@pytest.mark.hf_token_required
def test_implicit_left_recursion_schema():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    json_schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "pattern": "^(https?://)?([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([/\\w \\.-]*)*/?",
            }
        },
    }
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    _ = grammar_compiler.compile_json_schema(schema=json.dumps(json_schema))


@pytest.mark.hf_token_required
def test_regression_accept_invalid_token():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
    vocab_size = 151936
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size, stop_token_ids=[tokenizer.eos_token_id]
    )
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info=tokenizer_info)
    ctx = grammar_compiler.compile_json_schema(
        schema="""{"type": "object", "properties": {"value": {"type": ["string", "null"], "maxLength": 10}, "nested": {"type": "object", "properties": {"value": {"type": ["string", "null"]}, "nested_nested": {"type": "array", "items": {"type": ["string", "null"]}}}, "required": ["value", "nested_nested"], "maxItems": 10, "minItems": 1}}, "required": ["value", "nested"], "additionalProperties": false}"""
    )
    matcher = xgr.GrammarMatcher(ctx, max_rollback_tokens=200, override_stop_tokens=None)
    token_bitmask = xgr.allocate_token_bitmask(vocab_size=vocab_size, batch_size=7)
    token_bitmask.fill_(0)
    for i, token in enumerate([4913, 957, 788, 330, 1072, 67212, 788]):
        if i == 0:
            accepted = True
        else:
            parent_pos = i - 1
            curr_token_id = token
            parent_bitmask = token_bitmask[parent_pos]
            # 32 boolean bitmask values are packed into 32-bit integers
            accepted = (parent_bitmask[curr_token_id // 32] & (1 << (curr_token_id % 32))) != 0
        assert matcher.accept_token(token) == accepted
        matcher.fill_next_token_bitmask(token_bitmask, i)


def test_regression_empty_property_key_regex():
    schema = {
        "type": "object",
        "properties": {
            "_links": {
                "type": "object",
                "patternProperties": {
                    "": {"type": "object", "properties": {"href": {"type": "string"}}}
                },
            }
        },
    }
    _ = xgr.Grammar.from_json_schema(schema)
    assert _ is not None


def test_json_schema_number_without_constraint():
    schema = {"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]}
    grammar = xgr.Grammar.from_json_schema(schema)
    assert _is_grammar_accept_string(grammar, '{"value": -0.5}')
    assert _is_grammar_accept_string(grammar, '{"value": -1.5}')
    assert _is_grammar_accept_string(grammar, '{"value": 0}')
    assert _is_grammar_accept_string(grammar, '{"value": 1234567890}')
    assert _is_grammar_accept_string(grammar, '{"value": 3.14159}')
    assert _is_grammar_accept_string(grammar, '{"value": 1e10}')
    assert _is_grammar_accept_string(grammar, '{"value": -2.5E-3}')
    assert _is_grammar_accept_string(grammar, '{"value": 0.0}')
    assert _is_grammar_accept_string(grammar, '{"value": -0.0}')
    assert not _is_grammar_accept_string(grammar, '{"value": "abc"}')


if __name__ == "__main__":
    pytest.main(sys.argv)
