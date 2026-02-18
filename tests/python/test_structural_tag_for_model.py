"""Tests for get_structural_tag_for_model and generated structural tags.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.builtin_structural_tag import (
    get_builtin_structural_tag,
    get_builtin_structural_tag_supported_models,
)
from xgrammar.structural_tag import StructuralTag
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
    stag_ebnf = xgr.Grammar.from_structural_tag(structural_tag)
    assert (
        str(stag_ebnf) == expected_grammar_ebnf
    ), f"Expected:\n{expected_grammar_ebnf}\nGot:\n{str(stag_ebnf)}"


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
_tools_deepseek_v3_2 = make_tools(["search"])
_tools_minimax = make_tools(["search"])


# ---------- Test: unknown format type ----------


def test_get_structural_tag_for_model_unknown_format():
    """get_structural_tag_for_model raises ValueError for unknown format type."""
    with pytest.raises(ValueError) as exc_info:
        get_builtin_structural_tag("unknown_format")
    assert "Unknown format type" in str(exc_info.value)
    assert "unknown_format" in str(exc_info.value)


# ---------- Test: get_builtin_structural_tag_supported_models ----------


def test_get_builtin_structural_tag_supported_models_all():
    """get_structural_tag_supported_models() returns dict of all styles to model lists."""
    result = get_builtin_structural_tag_supported_models()
    assert isinstance(result, dict)
    expected_styles = {
        "llama",
        "qwen",
        "qwen_coder",
        "kimi",
        "deepseek_r1",
        "harmony",
        "deepseek_v3_2",
        "minimax",
    }
    assert set(result.keys()) == expected_styles
    for style, models in result.items():
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)


@pytest.mark.parametrize(
    "style, expected_models",
    [
        ("llama", ["Meta-Llama-3", "Llama-3.1", "Llama-3.2", "Llama-4"]),
        ("kimi", ["Kimi-K2", "Kimi-K2.5"]),
        ("deepseek_r1", ["DeepSeek-V3.1", "DeepSeek-R1", "DeepSeek-V3.2-exp"]),
        ("qwen_coder", ["Qwen3-Coder", "Qwen3-Coder-Next"]),
        ("qwen", ["Qwen3"]),
        ("harmony", ["gpt-oss"]),
        ("deepseek_v3_2", ["DeepSeek-V3.2"]),
        ("minimax", ["MiniMax-M2.5"]),
    ],
)
def test_get_structural_tag_supported_models_by_style(style: str, expected_models: List[str]):
    """get_structural_tag_supported_models(style) returns list of supported models for that style."""
    result = get_builtin_structural_tag_supported_models(style)
    assert result == expected_models


def test_get_structural_tag_supported_models_unknown_style():
    """get_structural_tag_supported_models(unknown_style) raises KeyError."""
    with pytest.raises(KeyError):
        get_builtin_structural_tag_supported_models("unknown_style")


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
def test_get_builtin_structural_tag_input_validation_errors(
    format_type: str, input_dict: Dict[str, Any], error_substring: str
):
    """get_builtin_structural_tag raises ValueError for invalid input."""
    with pytest.raises(ValueError) as exc_info:
        get_builtin_structural_tag(**_input_dict_to_get_stag_kwargs(format_type, input_dict))
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
            (
                r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("\"t1\", \"parameters\": " root_0 "}"))
triggered_tags ::= TagDispatch(
  ("{\"name\": ", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
                [True, True, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("\"t1\", \"parameters\": " root_0 "}"))
triggered_tags ::= TagDispatch(
  ("{\"name\": ", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
                [False, False, True, False, True],
            ),
            (
                r"""const_string ::= (("<think>\n\n</think>"))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("\"t1\", \"parameters\": " root_0 "}"))
triggered_tags ::= TagDispatch(
  ("{\"name\": ", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
        ],
    ),
    (
        "llama",
        {},
        [
            "",
            "text",
            "<think>123</think>text",
            "<think>\n\n</think></think>",
            "<think>\n\n</think>text",
        ],
        [
            (
                r"""any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
                [True, True, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
any_text_1 ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
                [False, False, True, False, True],
            ),
            (
                r"""const_string ::= (("<think>\n\n</think>"))
any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
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
            (
                r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("get_weather<|tool_call_argument_begin|>" root_0 "<|tool_call_end|>"))
triggered_tags ::= TagDispatch(
  ("<|tool_call_begin|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
                [True, False, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("get_weather<|tool_call_argument_begin|>" root_0 "<|tool_call_end|>"))
triggered_tags ::= TagDispatch(
  ("<|tool_call_begin|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
                [False, False, True, False, True],
            ),
            (
                r"""const_string ::= (("<think>\n\n</think>"))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("get_weather<|tool_call_argument_begin|>" root_0 "<|tool_call_end|>"))
triggered_tags ::= TagDispatch(
  ("<|tool_call_begin|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
        ],
    ),
    (
        "kimi",
        {},
        [
            "",
            "text",
            "<think>123</think>",
            "<think>\n\n</think></think>",
            "<think>\n\n</think>text",
        ],
        [
            (
                r"""any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
                [True, True, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
any_text_1 ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
                [False, False, True, False, True],
            ),
            (
                r"""const_string ::= (("<think>\n\n</think>"))
any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
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
            (
                r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("search<\uff5ctool\u2581sep\uff5c>" root_0 "<\uff5ctool\u2581call\u2581end\uff5c>"))
triggered_tags ::= TagDispatch(
  ("<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
                [True, False, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("" any_text))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("search<\uff5ctool\u2581sep\uff5c>" root_0 "<\uff5ctool\u2581call\u2581end\uff5c>"))
triggered_tags ::= TagDispatch(
  ("<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
                [False, True, True, False, True],
            ),
            (
                r"""const_string ::= (("</think>"))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("search<\uff5ctool\u2581sep\uff5c>" root_0 "<\uff5ctool\u2581call\u2581end\uff5c>"))
triggered_tags ::= TagDispatch(
  ("<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
        ],
    ),
    (
        "deepseek_r1",
        {},
        ["", "text", "123</think>123", "</think></think>", "</think>text"],
        [
            (
                r"""any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
                [True, True, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("" any_text))
any_text_1 ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
                [False, False, True, False, True],
            ),
            (
                r"""const_string ::= (("</think>"))
any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
        ],
    ),
    # ----- deepseek_v3_2
    (
        "deepseek_v3_2",
        {"tools": _tools_deepseek_v3_2},
        [
            'text<｜DSML｜function_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
            '<think>123</think><｜DSML｜function_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
            '<think>123</think>text<｜DSML｜function_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
            "<think>\n\n</think>text<think>123</think>",
            '<think>\n\n</think>text<｜DSML｜function_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
        ],
        [
            (
                r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
xml_string ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (("<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (("<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>") | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ("<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= (("\n" tags_with_separator_tags tags_with_separator_sub) | ("</\uff5cDSML\uff5cfunction_calls>\n"))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator))
triggered_tags ::= TagDispatch(
  ("<\uff5cDSML\uff5cfunction_calls>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
                [True, False, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
xml_string ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (("<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (("<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>") | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ("<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag_1 ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tags_with_separator_tags ::= ((tag_1))
tags_with_separator_sub ::= (("\n" tags_with_separator_tags tags_with_separator_sub) | ("</\uff5cDSML\uff5cfunction_calls>\n"))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator))
triggered_tags ::= TagDispatch(
  ("<\uff5cDSML\uff5cfunction_calls>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
                [False, True, True, False, True],
            ),
            (
                r"""const_string ::= (("<think>\n\n</think>"))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
xml_string ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (("<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (("<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>") | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ("<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= (("\n" tags_with_separator_tags tags_with_separator_sub) | ("</\uff5cDSML\uff5cfunction_calls>\n"))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator))
triggered_tags ::= TagDispatch(
  ("<\uff5cDSML\uff5cfunction_calls>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
        ],
    ),
    (
        "deepseek_v3_2",
        {},
        ["", "text", "<think>123</think>123", "<think></think></think>", "<think>\n\n</think>text"],
        [
            (
                r"""any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
                [True, True, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
any_text_1 ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
                [False, False, True, False, True],
            ),
            (
                r"""const_string ::= (("<think>\n\n</think>"))
any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
        ],
    ),
    # ----- minimax
    (
        "minimax",
        {"tools": _tools_minimax},
        [
            'text<minimax:tool_call>\n<invoke name="search">\n<parameter name="q">v</parameter></invoke>\n</minimax:tool_call>\n',
            '<think>123</think><minimax:tool_call>\n<invoke name="search">\n<parameter name="q">v</parameter></invoke>\n</minimax:tool_call>\n',
            '<think>123</think>text<minimax:tool_call>\n<invoke name="search">\n<parameter name="q">v</parameter></invoke>\n</minimax:tool_call>\n',
            "<think>\n\n</think>text<think>123</think>",
            '<think>\n\n</think>text<minimax:tool_call>\n<invoke name="search">\n<parameter name="q">v</parameter></invoke>\n</minimax:tool_call>\n',
        ],
        [
            (
                r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
xml_string ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (("<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (("<parameter name=\"q\">" [ \n\t]* xml_string [ \n\t]* "</parameter>") | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ("<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag ::= (("<invoke name=\"search\">\n" root_0 "</invoke>\n"))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= (("\n" tags_with_separator_tags tags_with_separator_sub) | ("</minimax:tool_call>\n"))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator))
triggered_tags ::= TagDispatch(
  ("<minimax:tool_call>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
                [True, False, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
xml_string ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (("<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (("<parameter name=\"q\">" [ \n\t]* xml_string [ \n\t]* "</parameter>") | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ("<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag_1 ::= (("<invoke name=\"search\">\n" root_0 "</invoke>\n"))
tags_with_separator_tags ::= ((tag_1))
tags_with_separator_sub ::= (("\n" tags_with_separator_tags tags_with_separator_sub) | ("</minimax:tool_call>\n"))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator))
triggered_tags ::= TagDispatch(
  ("<minimax:tool_call>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
                [False, True, True, False, True],
            ),
            (
                r"""const_string ::= (("<think>\n\n</think>"))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
xml_string ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (("<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (("<parameter name=\"q\">" [ \n\t]* xml_string [ \n\t]* "</parameter>") | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ("<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag ::= (("<invoke name=\"search\">\n" root_0 "</invoke>\n"))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= (("\n" tags_with_separator_tags tags_with_separator_sub) | ("</minimax:tool_call>\n"))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator))
triggered_tags ::= TagDispatch(
  ("<minimax:tool_call>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
        ],
    ),
    (
        "minimax",
        {},
        ["", "text", "<think>123</think>123", "<think></think></think>", "<think>\n\n</think>text"],
        [
            (
                r"""any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
                [True, True, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
any_text_1 ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
                [False, False, True, False, True],
            ),
            (
                r"""const_string ::= (("<think>\n\n</think>"))
any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
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
            (
                r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
xml_string ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (("<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (("<parameter=q>" [ \n\t]* xml_string [ \n\t]* "</parameter>") | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ("<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("run_sql>\n" root_0 "\n</function>\n</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>\n<function=", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
                [True, False, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
xml_string ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (("<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (("<parameter=q>" [ \n\t]* xml_string [ \n\t]* "</parameter>") | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ("<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("run_sql>\n" root_0 "\n</function>\n</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>\n<function=", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
                [False, False, True, False, True],
            ),
            (
                r"""const_string ::= (("<think>\n\n</think>"))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
xml_string ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (("<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (("<parameter=q>" [ \n\t]* xml_string [ \n\t]* "</parameter>") | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ("<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("run_sql>\n" root_0 "\n</function>\n</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>\n<function=", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
        ],
    ),
    (
        "qwen_coder",
        {},
        [
            "",
            "text",
            "<think>123</think>123",
            "<think>\n\n</think></think>",
            "<think>\n\n</think>text",
        ],
        [
            (
                r"""any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
                [True, True, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
any_text_1 ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
                [False, False, True, False, True],
            ),
            (
                r"""const_string ::= (("<think>\n\n</think>"))
any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
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
            (
                r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("\n{\"name\": \"t1\", \"arguments\": " root_0 "}\n</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
                [True, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("\n{\"name\": \"t1\", \"arguments\": " root_0 "}\n</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
                [False, True, False, True],
            ),
            (
                r"""const_string ::= (("<think>\n\n</think>"))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("\n{\"name\": \"t1\", \"arguments\": " root_0 "}\n</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
                [False, False, False, True],
            ),
        ],
    ),
    (
        "qwen",
        {},
        [
            "",
            "text",
            "<think>123</think>123",
            "<think>\n\n</think></think>",
            "<think>\n\n</think>text",
        ],
        [
            (
                r"""any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
                [True, True, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
any_text_1 ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
                [False, False, True, False, True],
            ),
            (
                r"""const_string ::= (("<think>\n\n</think>"))
any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
                [False, False, False, False, True],
            ),
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
            "<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>123<|end|>",
            '<|channel|>commentary to=comment_tool<|constrain|>json<|message|>{"q": "v"}<|call|>',
        ],
        [
            (
                r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag ::= (("<|channel|>commentary to=comment_tool<|constrain|>json<|message|>" root_0 "<|call|>"))
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
basic_any_1 ::= ((basic_number_8) | (basic_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_number_8 ::= ((basic_number_1_1 basic_number_7_1 basic_number_3_1 basic_number_6_1))
basic_string_1 ::= (("\"" basic_string_sub_1))
basic_boolean_1 ::= (("true") | ("false"))
basic_null_1 ::= (("null"))
basic_array_2 ::= (("[" [ \n\t]* basic_any_1 basic_array_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_2 ::= (("{" [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1_1 ::= ("" | ("-"))
basic_number_1_1 ::= ("" | ("-"))
basic_number_2_1 ::= (([0-9] basic_number_2_1) | ([0-9]))
basic_number_3_1 ::= ("" | ("." basic_number_2_1))
basic_number_4_1 ::= ("" | ([+\-]))
basic_number_5_1 ::= (([0-9] basic_number_5_1) | ([0-9]))
basic_number_6_1 ::= ("" | ([eE] basic_number_4_1 basic_number_5_1))
basic_array_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_1 basic_array_1_1))
basic_object_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1))
basic_number_7_1 ::= (("0") | ([1-9] [0-9]*))
tag_1 ::= (("<|channel|>analysis to=analysis_tool<|message|>" root_1 "<|call|>"))
any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
tag_2 ::= (("<|channel|>final<|message|>" any_text))
tags_with_separator_tags ::= ((tag) | (tag_1) | (tag_2))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
                [False, True, True, False, False, True],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<|channel|>analysis<|message|>" any_text))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag_1 ::= (("<|channel|>commentary to=comment_tool<|constrain|>json<|message|>" root_0 "<|call|>"))
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
basic_any_1 ::= ((basic_number_8) | (basic_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_number_8 ::= ((basic_number_1_1 basic_number_7_1 basic_number_3_1 basic_number_6_1))
basic_string_1 ::= (("\"" basic_string_sub_1))
basic_boolean_1 ::= (("true") | ("false"))
basic_null_1 ::= (("null"))
basic_array_2 ::= (("[" [ \n\t]* basic_any_1 basic_array_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_2 ::= (("{" [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1_1 ::= ("" | ("-"))
basic_number_1_1 ::= ("" | ("-"))
basic_number_2_1 ::= (([0-9] basic_number_2_1) | ([0-9]))
basic_number_3_1 ::= ("" | ("." basic_number_2_1))
basic_number_4_1 ::= ("" | ([+\-]))
basic_number_5_1 ::= (([0-9] basic_number_5_1) | ([0-9]))
basic_number_6_1 ::= ("" | ([eE] basic_number_4_1 basic_number_5_1))
basic_array_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_1 basic_array_1_1))
basic_object_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1))
basic_number_7_1 ::= (("0") | ([1-9] [0-9]*))
tag_2 ::= (("<|channel|>analysis to=analysis_tool<|message|>" root_1 "<|call|>"))
any_text_1 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
tag_3 ::= (("<|channel|>final<|message|>" any_text_1))
tags_with_separator_tags ::= ((tag) | (tag_1) | (tag_2) | (tag_3))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
                [True, True, True, False, True, True],
            ),
            (
                r"""const_string ::= (("<|end|>"))
tag ::= (("<|channel|>analysis<|message|>" const_string))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag_1 ::= (("<|channel|>commentary to=comment_tool<|constrain|>json<|message|>" root_0 "<|call|>"))
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
basic_any_1 ::= ((basic_number_8) | (basic_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_number_8 ::= ((basic_number_1_1 basic_number_7_1 basic_number_3_1 basic_number_6_1))
basic_string_1 ::= (("\"" basic_string_sub_1))
basic_boolean_1 ::= (("true") | ("false"))
basic_null_1 ::= (("null"))
basic_array_2 ::= (("[" [ \n\t]* basic_any_1 basic_array_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_2 ::= (("{" [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1_1 ::= ("" | ("-"))
basic_number_1_1 ::= ("" | ("-"))
basic_number_2_1 ::= (([0-9] basic_number_2_1) | ([0-9]))
basic_number_3_1 ::= ("" | ("." basic_number_2_1))
basic_number_4_1 ::= ("" | ([+\-]))
basic_number_5_1 ::= (([0-9] basic_number_5_1) | ([0-9]))
basic_number_6_1 ::= ("" | ([eE] basic_number_4_1 basic_number_5_1))
basic_array_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_1 basic_array_1_1))
basic_object_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1))
basic_number_7_1 ::= (("0") | ([1-9] [0-9]*))
tag_2 ::= (("<|channel|>analysis to=analysis_tool<|message|>" root_1 "<|call|>"))
any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
tag_3 ::= (("<|channel|>final<|message|>" any_text))
tags_with_separator_tags ::= ((tag) | (tag_1) | (tag_2) | (tag_3))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
                [True, True, True, False, False, True],
            ),
        ],
    ),
    (
        "harmony",
        {},
        [
            "",
            "<|channel|>final<|message|>123<|end|>",
            "<|channel|>analysis<|message|>123<|end|><|start|>assistant<|channel|>final<|message|>123<|end|>",
            "<|channel|>analysis<|message|><|end|>",
            "<think>\n\n</think>text",
        ],
        [
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<|channel|>final<|message|>" any_text))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
                [True, True, False, False, False],
            ),
            (
                r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<|channel|>analysis<|message|>" any_text))
any_text_1 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
tag_1 ::= (("<|channel|>final<|message|>" any_text_1))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
                [True, True, True, True, False],
            ),
            (
                r"""const_string ::= (("<|end|>"))
tag ::= (("<|channel|>analysis<|message|>" const_string))
any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
tag_1 ::= (("<|channel|>final<|message|>" any_text))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
                [True, True, False, True, False],
            ),
        ],
    ),
]


@pytest.mark.parametrize(
    "format_type, input_dict, instances, expected_grammar_and_results", instance_cases
)
def test_get_builtin_structural_tag_instance(
    format_type: str,
    input_dict: Dict[str, Any],
    instances: List[str],
    expected_grammar_and_results: List[Tuple[str, List[bool]]],
):
    """get_builtin_structural_tag accepts/rejects instance as expected."""
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

        stag = get_builtin_structural_tag(
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
