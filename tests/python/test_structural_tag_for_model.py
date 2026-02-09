"""Tests for get_structural_tag_for_model and generated structural tags.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.structural_tag import StructuralTag
from xgrammar.structural_tag_for_model import (
    get_structural_tag_for_model,
    get_structural_tag_supported_models,
)
from xgrammar.testing import _is_grammar_accept_string


def _input_dict_to_get_stag_kwargs(format_type: str, input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert input_dict (used by old template function API) to kwargs for get_structural_tag_for_model."""
    return {
        "model": format_type,
        "tools": input_dict.get("tools", []),
        "builtin_tools": input_dict.get("builtin_tools", []),
        "reasoning": input_dict.get("reasoning", input_dict.get("reasoning", True)),
        "force_empty_reasoning": input_dict.get("force_empty_reasoning", False),
    }


# ---------- Fixtures / Helpers ----------


class Profiler:
    def __init__(self, tokenizer_id: str):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, use_fast=True, trust_remote_code=True
        )
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
        self.compiler = xgr.GrammarCompiler(
            self.tokenizer_info, max_threads=16, cache_enabled=False
        )

    def profile_stag(self, structural_tag: StructuralTag, instance: str):
        time_begin = time.monotonic_ns()
        compiled_grammar = self.compiler.compile_structural_tag(structural_tag)
        time_end = time.monotonic_ns()
        compiler_duration = time_end - time_begin
        print(f"Compiling structural tag {structural_tag.format}")
        print(f"Compile time: {compiler_duration / 1000 / 1000} ms")
        matcher = xgr.GrammarMatcher(compiled_grammar)
        token_bitmask = xgr.allocate_token_bitmask(1, self.tokenizer_info.vocab_size)
        print(f"Matching instance: {instance}")
        for char in instance:
            matcher.accept_string(char)
            time_begin = time.monotonic_ns()
            matcher.fill_next_token_bitmask(token_bitmask)
            time_end = time.monotonic_ns()
            duration = time_end - time_begin
            print(f"Time to generate mask: {duration / 1000} us, Character: '{char}'")


profiler: Optional[Profiler] = None
PROFILER_ON = True
tokenizer_id = "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture(autouse=True, scope="module")
def disable_profiler(request):
    global PROFILER_ON
    global profiler
    markexpr = getattr(request.config.option, "markexpr", "") or request.config.getoption(
        "markexpr", ""
    )
    hf_token_not_provided = "not hf_token_required" in (markexpr or "")
    if hf_token_not_provided:
        PROFILER_ON = False
    else:
        profiler = Profiler(tokenizer_id)


def check_stag_with_grammar(structural_tag: StructuralTag, expected_grammar_ebnf: str):
    """Assert structural tag compiles to expected EBNF."""
    if expected_grammar_ebnf == "":
        return
    stag_ebnf = xgr.Grammar.from_structural_tag(structural_tag)
    assert (
        str(stag_ebnf) == expected_grammar_ebnf
    ), f"Expected:\n{expected_grammar_ebnf}\ngot:\n{str(stag_ebnf)}"


def check_stag_with_instance(
    structural_tag: StructuralTag,
    instance: str,
    is_accepted: bool = True,
    debug_print: bool = False,
):
    stag_grammar = xgr.Grammar.from_structural_tag(structural_tag)
    accepted = _is_grammar_accept_string(stag_grammar, instance, debug_print=debug_print)
    assert accepted == is_accepted
    if PROFILER_ON:
        profiler.profile_stag(structural_tag, instance)


# ---------- Shared tool definitions ----------

SIMPLE_SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}}


def make_tools(names: List[str], schema: Dict[str, Any] = SIMPLE_SCHEMA) -> List[Dict[str, Any]]:
    return [{"function": {"name": n, "parameters": schema}} for n in names]


# Tool lists used by instance tests (all in one place)
_tools_llama = make_tools(["t1"])
_tools_kimi = make_tools(["get_weather"])
_tools_deepseek = make_tools(["search"])
_tools_qwen_coder = make_tools(["run_sql"])
_tools_qwen = make_tools(["t1"])
_tools_harmony = make_tools(["comment_tool"])
_builtin_harmony = make_tools(["analysis_tool"])


# ---------- Test: unknown format type ----------


def test_get_structural_tag_for_model_unknown_format():
    """get_structural_tag_for_model raises ValueError for unknown format type."""
    with pytest.raises(ValueError) as exc_info:
        get_structural_tag_for_model("unknown_format")
    assert "Unknown format type" in str(exc_info.value)
    assert "unknown_format" in str(exc_info.value)


# ---------- Test: get_structural_tag_supported_models ----------


def test_get_structural_tag_supported_models_all():
    """get_structural_tag_supported_models() returns dict of all styles to model lists."""
    result = get_structural_tag_supported_models()
    assert isinstance(result, dict)
    expected_styles = {"llama", "qwen", "qwen_coder", "kimi", "deepseek_r1", "harmony"}
    assert set(result.keys()) == expected_styles
    for style, models in result.items():
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)


@pytest.mark.parametrize(
    "style, expected_models",
    [
        ("llama", ["llama3.1", "llama4"]),
        ("kimi", ["kimi-k2", "kimi-k2.5"]),
        ("deepseek_r1", ["deepseek-v3.1", "deepseek-r1", "deepseek-v3.2-exp"]),
        ("qwen_coder", ["qwen3-coder", "qwen3-coder-next"]),
        ("qwen", ["qwen3"]),
        ("harmony", ["gpt-oss"]),
    ],
)
def test_get_structural_tag_supported_models_by_style(style: str, expected_models: List[str]):
    """get_structural_tag_supported_models(style) returns list of supported models for that style."""
    result = get_structural_tag_supported_models(style)
    assert result == expected_models


def test_get_structural_tag_supported_models_unknown_style():
    """get_structural_tag_supported_models(unknown_style) raises KeyError."""
    with pytest.raises(KeyError):
        get_structural_tag_supported_models("unknown_style")


# ---------- Test: input validation errors ----------

# (format_type, input_dict, substring that must appear in the error message)
input_validation_error_cases: List[Tuple[str, Dict[str, Any], str]] = [
    # tools must be a list
    ("llama", {"tools": "not_a_list"}, "must be a list"),
    ("llama", {"tools": 123}, "must be a list"),
    ("harmony", {"tools": None}, "must be a list"),
    # tool[function] must have "name" and "parameters"
    ("llama", {"tools": [{"function": {}}]}, "must be a dictionary with 'name' and 'parameters'"),
    (
        "llama",
        {"tools": [{"function": {"name": "t1"}}]},
        "must be a dictionary with 'name' and 'parameters'",
    ),
    (
        "llama",
        {"tools": [{"function": {"parameters": {}}}]},
        "must be a dictionary with 'name' and 'parameters'",
    ),
    # name must be string
    (
        "llama",
        {"tools": [{"function": {"name": 123, "parameters": {}}}]},
        "'name' key in each tool must be a string",
    ),
    # parameters must be dict
    (
        "llama",
        {"tools": [{"function": {"name": "t1", "parameters": "not_a_dict"}}]},
        "'parameters' key in each tool must be a dict",
    ),
    (
        "llama",
        {"tools": [{"function": {"name": "t1", "parameters": []}}]},
        "'parameters' key in each tool must be a dict",
    ),
    # harmony: builtin_tools must be list
    ("harmony", {"tools": [], "builtin_tools": "not_list"}, "must be a list"),
    # harmony: builtin_tool[function] must have name and parameters
    ("harmony", {"tools": [], "builtin_tools": [{"function": {}}]}, "'name' and 'parameters'"),
    (
        "harmony",
        {"tools": [], "builtin_tools": [{"function": {"name": "b1", "parameters": 1}}]},
        "must be a dict",
    ),
    ("qwen", {"tools": [], "reasoning": "not_bool"}, "must be a boolean"),
]


@pytest.mark.parametrize("format_type, input_dict, error_substring", input_validation_error_cases)
def test_generate_structural_tag_input_validation_errors(
    format_type: str, input_dict: Dict[str, Any], error_substring: str
):
    """get_structural_tag_for_model raises ValueError for invalid input."""
    with pytest.raises(ValueError) as exc_info:
        get_structural_tag_for_model(**_input_dict_to_get_stag_kwargs(format_type, input_dict))
    msg = str(exc_info.value)
    if ".*" in error_substring:
        assert re.search(
            error_substring, msg, re.DOTALL
        ), f"Expected match for {error_substring!r} in {msg!r}"
    else:
        assert error_substring in msg, f"Expected {error_substring!r} in {msg!r}"


# ---------- Test: instance positive / negative ----------

# (format_type, input_dict, instances, grammar and is_accepted cases(not reasoning, reasoning, empty reasoning))
instance_cases: List[Tuple[str, Dict[str, Any], List[str], List[Tuple[str, List[bool]]]]] = [
    # ----- llama
    (
        "llama",
        {"tools": _tools_llama},
        [
            '{"name": "t1", "parameters": {"q": "v"}}',
            'text{"name": "t1", "parameters": {}}',
            '<think>123</think>text{"name": "t1", "parameters": {"q": ""}}',
            "<think>\n\n</think></think>",
            '<think>\n\n</think>text{"name": "t1", "parameters": {"q": "v"}}',
        ],
        [
            ("", [True, True, False, False, False]),
            ("", [False, False, True, False, True]),
            ("", [False, False, False, False, True]),
        ],
    ),
    # ----- kimi
    (
        "kimi",
        {"tools": _tools_kimi},
        [
            '123<|tool_call_begin|>get_weather<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|>',
            "123<|tool_call_begin|>123<|tool_call_argument_begin|>{}<|tool_call_end|>",
            "<think>123</think>",
            "<think>\n\n</think></think>",
            '<think>\n\n</think>123<|tool_call_begin|>get_weather<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|>',
        ],
        [
            ("", [True, False, False, False, False]),
            ("", [False, False, True, False, True]),
            ("", [False, False, False, False, True]),
        ],
    ),
    # ----- deepseek
    (
        "deepseek_r1",
        {"tools": _tools_deepseek},
        [
            'text<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
            '123</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
            'thinking</think>text<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
            "</think>text<think>123</think>",
            '</think>text<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
        ],
        [
            ("", [True, False, False, False, False]),
            ("", [False, True, True, False, True]),
            ("", [False, False, False, False, True]),
        ],
    ),
    # ----- qwen_coder
    (
        "qwen_coder",
        {"tools": _tools_qwen_coder},
        [
            "<tool_call>\n<function=run_sql>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=other>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
            "<think>123</think><tool_call>\n<function=run_sql>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
            "<think>\n\n</think><think></think>",
            "<think>\n\n</think>text<tool_call>\n<function=run_sql>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
        ],
        [
            ("", [True, False, False, False, False]),
            ("", [False, False, True, False, True]),
            ("", [False, False, False, False, True]),
        ],
    ),
    # ----- qwen
    (
        "qwen",
        {"tools": _tools_qwen},
        [
            'text<tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>',
            '<think>123</think><tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>',
            "<think>\n\n<think></think>"
            '<think>\n\n</think><tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>',
        ],
        [
            ("", [True, False, False, False]),
            ("", [False, True, False, True]),
            ("", [False, False, False, True]),
        ],
    ),
    # ----- harmony
    (
        "harmony",
        {"tools": _tools_harmony, "builtin_tools": _builtin_harmony},
        [
            "<|channel|>analysis<|message|><|end|>",
            '<|channel|>commentary to=comment_tool<|constrain|>json<|message|>{"q": "v"}<|call|>',
            '<|channel|>analysis to=analysis_tool<|message|>{"q": "v"}<|call|>',
            "<|channel|>commentary to=wrong_tool<|constrain|>json<|message|>{}<|call|>",
            "<|channel|>analysis<|message|>think<|end|>",
            '<|channel|>commentary to=comment_tool<|constrain|>json<|message|>{"q": "v"}<|call|>',
        ],
        [
            ("", [False, True, True, False, False, True]),
            ("", [True, True, True, False, True, True]),
            ("", [True, True, True, False, False, True]),
        ],
    ),
]


@pytest.mark.parametrize(
    "format_type, input_dict, instances, expected_grammar_and_results", instance_cases
)
def test_generate_structural_tag_instance(
    format_type: str,
    input_dict: Dict[str, Any],
    instances: List[str],
    expected_grammar_and_results: List[Tuple[str, List[bool]]],
):
    """Generated structural tag accepts/rejects instance as expected."""
    assert (
        len(expected_grammar_and_results) == 3
    ), "3 modes: not reasoning, reasoning, empty reasoning"
    for i in range(3):
        current_grammar = expected_grammar_and_results[i][0]
        current_results = expected_grammar_and_results[i][1]
        tools = input_dict.get("tools", [])
        builtin_tools = input_dict.get("builtin_tools", [])

        if i == 0:
            reasoning = False
        else:
            reasoning = True

        if i == 2:
            force_empty_reasoning = True
        else:
            force_empty_reasoning = False

        stag = get_structural_tag_for_model(
            format_type,
            reasoning=reasoning,
            force_empty_reasoning=force_empty_reasoning,
            tools=tools,
            builtin_tools=builtin_tools,
        )
        check_stag_with_grammar(stag, current_grammar)
        for j in range(len(instances)):
            instance = instances[j]
            is_accepted = current_results[j]
            check_stag_with_instance(stag, instance, is_accepted)
