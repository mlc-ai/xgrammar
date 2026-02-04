"""Tests for get_builtin_structural_tag_template_function and generated structural tags.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.structural_tag import StructuralTag
from xgrammar.structural_tag_template_function import get_builtin_structural_tag_template_function
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
    """Assert structural tag compiles to expected EBNF."""
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
    return [{"function": {"name": n, "parameters": schema}} for n in names]


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


# ---------- Test: grammar ----------

# (format_type, input_dict, expected_grammar_ebnf)
grammar_cases: List[Tuple[str, Dict[str, Any], str]] = [
    # llama
    (
        "llama",
        {"tools": []},
        r"""any_text ::= (([\0-\U0010ffff]*))
root ::= ((any_text))
""",
    ),
    (
        "llama",
        {"tools": make_tools(["t1", "t2"])},
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
triggered_tags_group ::= (("\"t1\", \"parameters\": " root_0 "}") | ("\"t2\", \"parameters\": " root_1 "}"))
triggered_tags ::= TagDispatch(
  ("{\"name\": ", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
    ),
    # kimi, thinking True / False
    (
        "kimi",
        {"tools": [], "thinking": True},
        r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
any_text_1 ::= (([\0-\U0010ffff]*))
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
    ),
    (
        "kimi",
        {"tools": [], "thinking": False},
        r"""const_string ::= (("<think></think>"))
any_text ::= (([\0-\U0010ffff]*))
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
    ),
    (
        "kimi",
        {"tools": make_tools(["tool_a", "tool_b"]), "thinking": True},
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
triggered_tags_group ::= (("tool_a<|tool_call_argument_begin|>" root_0 "<|tool_call_end|>") | ("tool_b<|tool_call_argument_begin|>" root_1 "<|tool_call_end|>"))
triggered_tags ::= TagDispatch(
  ("<|tool_call_begin|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
    ),
    (
        "kimi",
        {"tools": make_tools(["tool_a", "tool_b"]), "thinking": False},
        r"""const_string ::= (("<think></think>"))
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
triggered_tags_group ::= (("tool_a<|tool_call_argument_begin|>" root_0 "<|tool_call_end|>") | ("tool_b<|tool_call_argument_begin|>" root_1 "<|tool_call_end|>"))
triggered_tags ::= TagDispatch(
  ("<|tool_call_begin|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
    ),
    # deepseek, thinking True / False
    (
        "deepseek",
        {"tools": [], "thinking": True},
        r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
any_text_1 ::= (([\0-\U0010ffff]*))
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
    ),
    (
        "deepseek",
        {"tools": [], "thinking": False},
        r"""const_string ::= (("<think></think>"))
any_text ::= (([\0-\U0010ffff]*))
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
    ),
    (
        "deepseek",
        {"tools": make_tools(["tool_a", "tool_b"]), "thinking": True},
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
triggered_tags_group ::= (("tool_a<\uff5ctool\u2581sep\uff5c>" root_0 "<\uff5ctool\u2581call\u2581end\uff5c>") | ("tool_b<\uff5ctool\u2581sep\uff5c>" root_1 "<\uff5ctool\u2581call\u2581end\uff5c>"))
triggered_tags ::= TagDispatch(
  ("<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
    ),
    (
        "deepseek",
        {"tools": make_tools(["tool_a", "tool_b"]), "thinking": False},
        r"""const_string ::= (("<think></think>"))
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
triggered_tags_group ::= (("tool_a<\uff5ctool\u2581sep\uff5c>" root_0 "<\uff5ctool\u2581call\u2581end\uff5c>") | ("tool_b<\uff5ctool\u2581sep\uff5c>" root_1 "<\uff5ctool\u2581call\u2581end\uff5c>"))
triggered_tags ::= TagDispatch(
  ("<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
    ),
    # qwen_coder
    (
        "qwen_coder",
        {"tools": []},
        r"""any_text ::= (([\0-\U0010ffff]*))
root ::= ((any_text))
""",
    ),
    (
        "qwen_coder",
        {"tools": make_tools(["tool_a", "tool_b"])},
        r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
xml_string ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
xml_string_0 ::= ((xml_string))
xml_any ::= ((basic_number) | (xml_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= ("" | ([ \n\t]* "<parameter=q>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>"))
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
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
xml_string_1 ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_variable_name_1 ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
xml_string_0_1 ::= ((xml_string_1))
xml_any_1 ::= ((basic_number_8) | (xml_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_any_1 ::= ((basic_number_8) | (basic_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_number_8 ::= ((basic_number_1_1 basic_number_7_1 basic_number_3_1 basic_number_6_1))
basic_string_1 ::= (("\"" basic_string_sub_1))
basic_boolean_1 ::= (("true") | ("false"))
basic_null_1 ::= (("null"))
basic_array_2 ::= (("[" [ \n\t]* basic_any_1 basic_array_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_2 ::= (("{" [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_1 ::= ("" | ([ \n\t]* "<parameter=q>" [ \n\t]* xml_string_0_1 [ \n\t]* "</parameter>"))
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
triggered_tags_group ::= (("tool_a>" root_0 "</function>") | ("tool_b>" root_1 "</function>"))
triggered_tags ::= TagDispatch(
  ("<function=", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
    ),
    # qwen (with reasoning True / False)
    (
        "qwen",
        {"tools": [], "thinking": True},
        r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("</think>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<think>" any_text))
any_text_1 ::= (([\0-\U0010ffff]*))
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
    ),
    (
        "qwen",
        {"tools": [], "thinking": False},
        r"""const_string ::= (("<think></think>"))
any_text ::= (([\0-\U0010ffff]*))
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
    ),
    (
        "qwen",
        {"tools": make_tools(["tool_a", "tool_b"]), "thinking": True},
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
triggered_tags_group ::= (("{\"name\": \"tool_a\", \"arguments\": " root_0 "}</tool_call>") | ("{\"name\": \"tool_b\", \"arguments\": " root_1 "}</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
    ),
    (
        "qwen",
        {"tools": make_tools(["tool_a", "tool_b"]), "thinking": False},
        r"""const_string ::= (("<think></think>"))
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
triggered_tags_group ::= (("{\"name\": \"tool_a\", \"arguments\": " root_0 "}</tool_call>") | ("{\"name\": \"tool_b\", \"arguments\": " root_1 "}</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
    ),
    # harmony (tools only, builtin_tools only, both)
    (
        "harmony",
        {"tools": [], "builtin_tools": []},
        r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
any_text_1 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|return|>"),
  loop_after_dispatch=false,
  excludes=()
)
any_text_2 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|call|>"),
  loop_after_dispatch=false,
  excludes=()
)
triggered_tags_group ::= (("assistant<|channel|>analysis<|message|>" any_text) | ("assistant<|channel|>final<|message|>" any_text_1) | ("assistant<|channel|>final<|message|>" any_text_2))
triggered_tags ::= TagDispatch(
  ("<|start|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
    ),
    (
        "harmony",
        {"tools": make_tools(["tool_a", "tool_b"]), "builtin_tools": []},
        r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
any_text_1 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|return|>"),
  loop_after_dispatch=false,
  excludes=()
)
any_text_2 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|call|>"),
  loop_after_dispatch=false,
  excludes=()
)
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
triggered_tags_group ::= (("assistant<|channel|>analysis<|message|>" any_text) | ("assistant<|channel|>final<|message|>" any_text_1) | ("assistant<|channel|>final<|message|>" any_text_2) | ("assistant<|channel|>commentary to=tool_a<|constrain|>json<|message|>" root_0 "<|end|>") | ("assistant<|channel|>commentary to=tool_b<|constrain|>json<|message|>" root_1 "<|end|>"))
triggered_tags ::= TagDispatch(
  ("<|start|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
    ),
    (
        "harmony",
        {"tools": [], "builtin_tools": make_tools(["builtin_tool_a", "builtin_tool_b"])},
        r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
any_text_1 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|return|>"),
  loop_after_dispatch=false,
  excludes=()
)
any_text_2 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|call|>"),
  loop_after_dispatch=false,
  excludes=()
)
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
triggered_tags_group ::= (("assistant<|channel|>analysis<|message|>" any_text) | ("assistant<|channel|>final<|message|>" any_text_1) | ("assistant<|channel|>final<|message|>" any_text_2) | ("assistant<|channel|>analysis to=builtin_tool_a<|message|>" root_0 "<|end|>") | ("assistant<|channel|>analysis to=builtin_tool_b<|message|>" root_1 "<|end|>"))
triggered_tags ::= TagDispatch(
  ("<|start|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
    ),
    (
        "harmony",
        {
            "tools": make_tools(["tool_a", "tool_b"]),
            "builtin_tools": make_tools(["tool_a", "tool_b"]),
        },
        r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
any_text_1 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|return|>"),
  loop_after_dispatch=false,
  excludes=()
)
any_text_2 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|call|>"),
  loop_after_dispatch=false,
  excludes=()
)
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
basic_escape_2 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_2 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_2) | ("\\" basic_escape_2 basic_string_sub_2)) (=([ \n\t]* [,}\]:]))
basic_any_2 ::= ((basic_number_9) | (basic_string_2) | (basic_boolean_2) | (basic_null_2) | (basic_array_3) | (basic_object_3))
basic_integer_3 ::= (("0") | (basic_integer_1_2 [1-9] [0-9]*))
basic_number_9 ::= ((basic_number_1_2 basic_number_7_2 basic_number_3_2 basic_number_6_2))
basic_string_2 ::= (("\"" basic_string_sub_2))
basic_boolean_2 ::= (("true") | ("false"))
basic_null_2 ::= (("null"))
basic_array_3 ::= (("[" [ \n\t]* basic_any_2 basic_array_1_2 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_3 ::= (("{" [ \n\t]* basic_string_2 [ \n\t]* ":" [ \n\t]* basic_any_2 basic_object_1_2 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_2 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string_2 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1_2 ::= ("" | ("-"))
basic_number_1_2 ::= ("" | ("-"))
basic_number_2_2 ::= (([0-9] basic_number_2_2) | ([0-9]))
basic_number_3_2 ::= ("" | ("." basic_number_2_2))
basic_number_4_2 ::= ("" | ([+\-]))
basic_number_5_2 ::= (([0-9] basic_number_5_2) | ([0-9]))
basic_number_6_2 ::= ("" | ([eE] basic_number_4_2 basic_number_5_2))
basic_array_1_2 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_2 basic_array_1_2))
basic_object_1_2 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_2 [ \n\t]* ":" [ \n\t]* basic_any_2 basic_object_1_2))
basic_number_7_2 ::= (("0") | ([1-9] [0-9]*))
basic_escape_3 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_3 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_3) | ("\\" basic_escape_3 basic_string_sub_3)) (=([ \n\t]* [,}\]:]))
basic_any_3 ::= ((basic_number_10) | (basic_string_3) | (basic_boolean_3) | (basic_null_3) | (basic_array_4) | (basic_object_4))
basic_integer_4 ::= (("0") | (basic_integer_1_3 [1-9] [0-9]*))
basic_number_10 ::= ((basic_number_1_3 basic_number_7_3 basic_number_3_3 basic_number_6_3))
basic_string_3 ::= (("\"" basic_string_sub_3))
basic_boolean_3 ::= (("true") | ("false"))
basic_null_3 ::= (("null"))
basic_array_4 ::= (("[" [ \n\t]* basic_any_3 basic_array_1_3 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_4 ::= (("{" [ \n\t]* basic_string_3 [ \n\t]* ":" [ \n\t]* basic_any_3 basic_object_1_3 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_3 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string_3 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1_3 ::= ("" | ("-"))
basic_number_1_3 ::= ("" | ("-"))
basic_number_2_3 ::= (([0-9] basic_number_2_3) | ([0-9]))
basic_number_3_3 ::= ("" | ("." basic_number_2_3))
basic_number_4_3 ::= ("" | ([+\-]))
basic_number_5_3 ::= (([0-9] basic_number_5_3) | ([0-9]))
basic_number_6_3 ::= ("" | ([eE] basic_number_4_3 basic_number_5_3))
basic_array_1_3 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_3 basic_array_1_3))
basic_object_1_3 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_3 [ \n\t]* ":" [ \n\t]* basic_any_3 basic_object_1_3))
basic_number_7_3 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("assistant<|channel|>analysis<|message|>" any_text) | ("assistant<|channel|>final<|message|>" any_text_1) | ("assistant<|channel|>final<|message|>" any_text_2) | ("assistant<|channel|>commentary to=tool_a<|constrain|>json<|message|>" root_0 "<|end|>") | ("assistant<|channel|>commentary to=tool_b<|constrain|>json<|message|>" root_1 "<|end|>") | ("assistant<|channel|>analysis to=tool_a<|message|>" root_2 "<|end|>") | ("assistant<|channel|>analysis to=tool_b<|message|>" root_3 "<|end|>"))
triggered_tags ::= TagDispatch(
  ("<|start|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
    ),
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
