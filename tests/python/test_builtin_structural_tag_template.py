"""Tests for get_builtin_structural_tag_template_function and generated structural tags.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar import get_builtin_structural_tag_template_function
from xgrammar.structural_tag import StructuralTag
from xgrammar.testing import _is_grammar_accept_string

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
    """Assert structural tag compiles to expected EBNF. Skip assertion if expected is empty."""
    if not expected_grammar_ebnf:
        return
    stag_ebnf = xgr.Grammar.from_structural_tag(structural_tag)
    assert str(stag_ebnf) == expected_grammar_ebnf


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
    return [{"name": n, "parameters": schema} for n in names]


# ---------- Test: unknown format type ----------


def test_get_builtin_structural_tag_template_function_unknown_format():
    """get_builtin_structural_tag_template_function raises ValueError for unknown format type."""
    with pytest.raises(ValueError) as exc_info:
        get_builtin_structural_tag_template_function("unknown_format")
    assert "Unknown format type" in str(exc_info.value)
    assert "unknown_format" in str(exc_info.value)


# ---------- Test: input validation errors ----------

# (format_type, input_dict, substring that must appear in the error message)
input_validation_error_cases: List[Tuple[str, Dict[str, Any], str]] = [
    # tools must be a list
    ("llama", {"tools": "not_a_list"}, "must be a list"),
    ("llama", {"tools": 123}, "must be a list"),
    ("harmony", {"tools": None}, "must be a list"),
    # tool must have "name" and "parameters"
    ("llama", {"tools": [{}]}, "must be a dictionary with 'name' and 'parameters'"),
    ("llama", {"tools": [{"name": "t1"}]}, "must be a dictionary with 'name' and 'parameters'"),
    ("llama", {"tools": [{"parameters": {}}]}, "must be a dictionary with 'name' and 'parameters'"),
    # name must be string
    (
        "llama",
        {"tools": [{"name": 123, "parameters": {}}]},
        "'name' key in each tool must be a string",
    ),
    # parameters must be dict
    (
        "llama",
        {"tools": [{"name": "t1", "parameters": "not_a_dict"}]},
        "'parameters' key in each tool must be a dict",
    ),
    (
        "llama",
        {"tools": [{"name": "t1", "parameters": []}]},
        "'parameters' key in each tool must be a dict",
    ),
    # harmony: builtin_tools must be list
    ("harmony", {"tools": [], "builtin_tools": "not_list"}, "builtin_tools.*must be a list"),
    # harmony: builtin_tool must have name and parameters
    ("harmony", {"tools": [], "builtin_tools": [{}]}, "builtin tool.*'name' and 'parameters'"),
    (
        "harmony",
        {"tools": [], "builtin_tools": [{"name": "b1", "parameters": 1}]},
        "builtin tool.*must be a dict",
    ),
]


@pytest.mark.parametrize("format_type, input_dict, error_substring", input_validation_error_cases)
def test_generate_structural_tag_input_validation_errors(
    format_type: str, input_dict: Dict[str, Any], error_substring: str
):
    """Generated template function raises ValueError for invalid input_dict."""
    fn = get_builtin_structural_tag_template_function(format_type)
    with pytest.raises(ValueError) as exc_info:
        fn(input_dict)
    msg = str(exc_info.value)
    if ".*" in error_substring:
        assert re.search(
            error_substring, msg, re.DOTALL
        ), f"Expected match for {error_substring!r} in {msg!r}"
    else:
        assert error_substring in msg, f"Expected {error_substring!r} in {msg!r}"


# ---------- Test: grammar (expected left empty for you to fill) ----------

# (format_type, input_dict, expected_grammar_ebnf). Use "" to skip grammar check.
grammar_cases: List[Tuple[str, Dict[str, Any], str]] = [
    # llama
    ("llama", {"tools": []}, ""),
    ("llama", {"tools": make_tools(["t1"])}, ""),
    ("llama", {"tools": make_tools(["t1", "t2"])}, ""),
    # kimi
    ("kimi", {"tools": []}, ""),
    ("kimi", {"tools": make_tools(["tool_a"])}, ""),
    ("kimi", {"tools": make_tools(["a", "b"])}, ""),
    # deepseek
    ("deepseek", {"tools": []}, ""),
    ("deepseek", {"tools": make_tools(["get_weather"])}, ""),
    ("deepseek", {"tools": make_tools(["f1", "f2", "f3"])}, ""),
    # qwen_coder
    ("qwen_coder", {"tools": []}, ""),
    ("qwen_coder", {"tools": make_tools(["run_sql"])}, ""),
    ("qwen_coder", {"tools": make_tools(["a", "b"])}, ""),
    # qwen (with reasoning True / False)
    ("qwen", {"tools": [], "thinking": True}, ""),
    ("qwen", {"tools": [], "thinking": False}, ""),
    ("qwen", {"tools": make_tools(["t1"]), "thinking": True}, ""),
    ("qwen", {"tools": make_tools(["t1"]), "thinking": False}, ""),
    # harmony (tools only, builtin_tools only, both)
    ("harmony", {"tools": [], "builtin_tools": []}, ""),
    ("harmony", {"tools": make_tools(["t1"]), "builtin_tools": []}, ""),
    ("harmony", {"tools": [], "builtin_tools": make_tools(["b1"])}, ""),
    ("harmony", {"tools": make_tools(["t1", "t2"]), "builtin_tools": make_tools(["b1"])}, ""),
]


@pytest.mark.parametrize("format_type, input_dict, expected_grammar_ebnf", grammar_cases)
def test_generate_structural_tag_grammar(
    format_type: str, input_dict: Dict[str, Any], expected_grammar_ebnf: str
):
    """Generated structural tag compiles; EBNF left empty for you to fill."""
    fn = get_builtin_structural_tag_template_function(format_type)
    stag = fn(input_dict)
    assert isinstance(stag, StructuralTag)
    check_stag_with_grammar(stag, expected_grammar_ebnf)


# ---------- Test: instance positive / negative ----------

# (format_type, input_dict, instance, is_accepted)
instance_cases: List[Tuple[str, Dict[str, Any], str, bool]] = []

# ----- llama
_tools_llama = make_tools(["t1"])
instance_cases += [
    ("llama", {"tools": _tools_llama}, '{"name": "t1", "parameters": {"q": "v"}}', True),
    ("llama", {"tools": _tools_llama}, '{"name": "t1", "parameters": {}}', True),
    ("llama", {"tools": _tools_llama}, '{"name": "t1", "parameters": {"q": ""}}', True),
    ("llama", {"tools": _tools_llama}, '{"name": "wrong", "parameters": {"q": "v"}}', False),
    ("llama", {"tools": _tools_llama}, '{"name": "t1", "parameters": {"q": 1}}', False),
    ("llama", {"tools": _tools_llama}, '{"name": "t1", "parameters": invalid}', False),
]
instance_cases += [("llama", {"tools": []}, "", True)]

# ----- kimi
_tools_kimi = make_tools(["get_weather"])
instance_cases += [
    (
        "kimi",
        {"tools": _tools_kimi, "thinking": True},
        '<think>12opj</think><|tool_call_begin|>get_weather<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|>',
        True,
    ),
    (
        "kimi",
        {"tools": _tools_kimi, "thinking": True},
        "<think>213</think><|tool_call_begin|>get_weather<|tool_call_argument_begin|>{}<|tool_call_end|>",
        True,
    ),
    (
        "kimi",
        {"tools": _tools_kimi, "thinking": True},
        '<think>123</think><|tool_call_begin|>other<|tool_call_argument_begin|>{"q":"v"}<|tool_call_end|>',
        False,
    ),
    (
        "kimi",
        {"tools": _tools_kimi, "thinking": True},
        '<think>123</think><|tool_call_begin|>get_weather<|tool_call_argument_begin|>{"q":1}<|tool_call_end|>',
        False,
    ),
    (
        "kimi",
        {"tools": _tools_kimi, "thinking": False},
        '<think></think><|tool_call_begin|>get_weather<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|>',
        True,
    ),
    (
        "kimi",
        {"tools": _tools_kimi, "thinking": False},
        "<think>123</think><|tool_call_begin|>get_weather<|tool_call_argument_begin|>{}<|tool_call_end|>",
        False,
    ),
]
instance_cases += [
    ("kimi", {"tools": [], "thinking": True}, "<think>123</think>123", True),
    ("kimi", {"tools": [], "thinking": False}, "<think>123</think>123", False),
]

# ----- deepseek (format: <tool_call>name</tool_call> + JSON + </tool_call>)
_tools_deepseek = make_tools(["search"])
instance_cases += [
    (
        "deepseek",
        {"tools": _tools_deepseek},
        '<think></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
        True,
    ),
    (
        "deepseek",
        {"tools": _tools_deepseek},
        "<think></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{}<｜tool▁call▁end｜>",
        True,
    ),
    (
        "deepseek",
        {"tools": _tools_deepseek},
        '<think></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>wrong<｜tool▁sep｜>{"q":"v"}<｜tool▁call▁end｜>',
        False,
    ),
]
instance_cases += [("deepseek", {"tools": [], "thinking": True}, "<think>123</think>123", True)]
instance_cases += [("deepseek", {"tools": [], "thinking": False}, "<think></think>123", True)]

# ----- qwen_coder
_tools_qwen_coder = make_tools(["run_sql"])
instance_cases += [
    (
        "qwen_coder",
        {"tools": _tools_qwen_coder},
        "<function=run_sql><parameter=q>v</parameter></function>",
        True,
    ),
    ("qwen_coder", {"tools": _tools_qwen_coder}, "<function=run_sql></function>", True),
    (
        "qwen_coder",
        {"tools": _tools_qwen_coder},
        "<function=other><parameter=q>v</parameter></function>",
        False,
    ),
]
instance_cases += [("qwen_coder", {"tools": []}, "", True)]

# ----- qwen (with <think> prefix when thinking=True)
_tools_qwen = make_tools(["t1"])
instance_cases += [
    (
        "qwen",
        {"tools": _tools_qwen, "thinking": True},
        '<think>123</think><tool_call>{"name": "t1", "arguments": {"q": "v"}}</tool_call>',
        True,
    ),
    (
        "qwen",
        {"tools": _tools_qwen, "thinking": False},
        '<think></think><tool_call>{"name": "t1", "arguments": {"q": "v"}}</tool_call>',
        True,
    ),
    (
        "qwen",
        {"tools": _tools_qwen, "thinking": True},
        '<think>123</think><tool_call>{"name": "t1", "arguments": {"q": "v"}}</tool_call>',
        True,
    ),
    (
        "qwen",
        {"tools": _tools_qwen, "thinking": False},
        '<think>123</think><tool_call>{"name": "t1", "arguments": {"q": "v"}}</tool_call>',
        False,
    ),
]
instance_cases += [
    ("qwen", {"tools": [], "thinking": True}, "<think>123</think>", True),
    ("qwen", {"tools": [], "thinking": False}, "<think></think>", True),
]

# ----- harmony (fixed tags + tool / builtin_tool tags)
_tools_harmony = make_tools(["comment_tool"])
_builtin_harmony = make_tools(["analysis_tool"])
instance_cases += [
    (
        "harmony",
        {"tools": _tools_harmony, "builtin_tools": _builtin_harmony},
        "<|start|>assistant<|channel|>analysis<|message|>some text<|end|>",
        True,
    ),
    (
        "harmony",
        {"tools": _tools_harmony, "builtin_tools": _builtin_harmony},
        '<|start|>assistant<|channel|>commentary to=comment_tool<|constrain|>json<|message|>{"q": "v"}<|end|>',
        True,
    ),
    (
        "harmony",
        {"tools": _tools_harmony, "builtin_tools": _builtin_harmony},
        '<|start|>assistant<|channel|>analysis to=analysis_tool<|message|>{"q": "v"}<|end|>',
        True,
    ),
    (
        "harmony",
        {"tools": _tools_harmony, "builtin_tools": _builtin_harmony},
        "<|start|>assistant<|channel|>commentary to=wrong_tool<|constrain|>json<|message|>{}<|end|>",
        False,
    ),
]
instance_cases += [("harmony", {"tools": [], "builtin_tools": []}, "", True)]


@pytest.mark.parametrize("format_type, input_dict, instance, is_accepted", instance_cases)
def test_generate_structural_tag_instance(
    format_type: str, input_dict: Dict[str, Any], instance: str, is_accepted: bool
):
    """Generated structural tag accepts/rejects instance as expected."""
    fn = get_builtin_structural_tag_template_function(format_type)
    stag = fn(input_dict)
    check_stag_with_instance(stag, instance, is_accepted)
