"""Tests for get_structural_tag_for_model and generated structural tags."""

import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.builtin_structural_tag import (
    get_deepseek_structural_tag,
    get_deepseek_v3_2_structural_tag,
    get_deepseek_v4_structural_tag,
    get_gemma4_structural_tag,
    get_glm47_structural_tag,
    get_harmony_structural_tag,
    get_kimi_structural_tag,
    get_llama_structural_tag,
    get_minimax_structural_tag,
    get_model_structural_tag,
    get_qwen_coder_structural_tag,
    get_qwen_structural_tag,
)
from xgrammar.openai_tool_call_schema import BuiltinToolParam, FunctionToolParam
from xgrammar.structural_tag import JSONSchemaFormat, StructuralTag, TagFormat
from xgrammar.testing import _is_grammar_accept_string


def _input_dict_to_get_stag_kwargs(format_type: str, input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert input_dict (used by old template function API) to kwargs for get_structural_tag_for_model."""
    tools = input_dict.get("tools", [])
    if isinstance(tools, list):
        tools = list(tools)
    builtin_tools = input_dict.get("builtin_tools", [])
    if not isinstance(builtin_tools, list):
        tools = builtin_tools
        builtin_tools = []
    for builtin_tool in builtin_tools:
        function = builtin_tool.get("function", {})
        tools.append(
            {
                "type": function.get("name"),
                "name": function.get("name"),
                "parameters": function.get("parameters"),
            }
        )
    tool_choice = input_dict.get("tool_choice", "auto")
    if tool_choice == "forced":
        tool_choice = {
            "type": "function",
            "function": {"name": input_dict.get("forced_function_name")},
        }
    return {
        "model": format_type,
        "tools": tools,
        "reasoning": input_dict.get("reasoning", input_dict.get("reasoning", True)),
        "force_empty_reasoning": input_dict.get("force_empty_reasoning", False),
        "tool_choice": tool_choice,
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
    # Import shared token check from conftest (handles env vars + cached login)
    from conftest import _hf_token_available, _hf_token_explicitly_disabled

    if not _hf_token_available() or _hf_token_explicitly_disabled(request.config):
        PROFILER_ON = False
    else:
        profiler = Profiler(tokenizer_id)


def check_stag_with_grammar(structural_tag: StructuralTag, expected_grammar_ebnf: str):
    """Assert structural tag compiles to expected EBNF.

    If ``expected_grammar_ebnf`` is empty or whitespace-only, skip the comparison
    (useful while expected EBNF is not yet filled in).
    """

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


def _walk_structural_format(format_obj):
    """Yield every nested structural format object."""

    yield format_obj
    for attr_name in ("content", "format"):
        child = getattr(format_obj, attr_name, None)
        if child is not None:
            yield from _walk_structural_format(child)
    for attr_name in ("elements", "tags"):
        for child in getattr(format_obj, attr_name, []) or []:
            yield from _walk_structural_format(child)


def _collect_tag_begins(structural_tag: StructuralTag) -> List[str]:
    """Collect TagFormat begin strings from a structural tag."""

    return [
        format_obj.begin
        for format_obj in _walk_structural_format(structural_tag.format)
        if isinstance(format_obj, TagFormat) and isinstance(format_obj.begin, str)
    ]


def _collect_json_schema_values(structural_tag: StructuralTag) -> List[Any]:
    """Collect JSON schema values from nested JSONSchemaFormat nodes."""

    return [
        format_obj.json_schema
        for format_obj in _walk_structural_format(structural_tag.format)
        if isinstance(format_obj, JSONSchemaFormat)
    ]


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
_tools_deepseek_v4 = make_tools(["search"])
_tools_minimax = make_tools(["search"])
_tools_glm47 = make_tools(["search"])
_tools_gemma4 = make_tools(["get_weather"])

# Two distinct tools for tool_choice=required / forced instance tests.
_tools_llama_pair = make_tools(["t1", "t2"])
_tools_kimi_pair = make_tools(["t1", "t2"])
_tools_deepseek_pair = make_tools(["search", "alt"])
_tools_deepseek_v3_2_pair = make_tools(["search", "alt"])
_tools_deepseek_v4_pair = make_tools(["search", "alt"])
_tools_minimax_pair = make_tools(["search", "alt"])
_tools_qwen_coder_pair = make_tools(["run_sql", "run_py"])
_tools_qwen_pair = make_tools(["t1", "t2"])
_tools_harmony_pair = make_tools(["comment_tool", "other_tool"])
_tools_glm47_pair = make_tools(["search", "alt"])
_tools_gemma4_pair = make_tools(["search", "alt"])


# ---------- Test: unknown format type ----------


def test_unknown_format():
    """get_structural_tag_for_model raises ValueError for unknown format type."""
    with pytest.raises(ValueError) as exc_info:
        get_model_structural_tag("unknown_format")
    assert "Unknown format type" in str(exc_info.value)
    assert "unknown_format" in str(exc_info.value)


# ---------- Test: input validation errors ----------

# (format_type, input_dict, substring that must appear in the error message)
input_validation_error_cases: List[Tuple[str, Dict[str, Any], str]] = [
    # tools must be a list
    ("llama", {"tools": "not_a_list"}, "must be a list"),
    ("llama", {"tools": 123}, "must be a list"),
    # tool[function] must have "name" and "parameters"
    ("llama", {"tools": [{"function": {}}]}, "function.name"),
    ("llama", {"tools": [{"function": {"parameters": {}}}]}, "function.name"),
    # name must be string
    ("llama", {"tools": [{"function": {"name": 123, "parameters": {}}}]}, "function.name"),
    # parameters must be dict
    (
        "llama",
        {"tools": [{"function": {"name": "t1", "parameters": "not_a_dict"}}]},
        "function.parameters",
    ),
    ("llama", {"tools": [{"function": {"name": "t1", "parameters": []}}]}, "function.parameters"),
    # Legacy builtin_tools test data is converted into public tools before calling the API.
    ("harmony", {"tools": [], "builtin_tools": "not_list"}, "must be a list"),
    (
        "harmony",
        {"tools": [], "builtin_tools": [{"function": {"name": "b1", "parameters": 1}}]},
        "parameters",
    ),
]


@pytest.mark.parametrize("format_type, input_dict, error_substring", input_validation_error_cases)
def test_input_validation_errors(
    format_type: str, input_dict: Dict[str, Any], error_substring: str
):
    """get_model_structural_tag raises ValueError for invalid input."""
    with pytest.raises(ValueError) as exc_info:
        get_model_structural_tag(**_input_dict_to_get_stag_kwargs(format_type, input_dict))
    msg = str(exc_info.value)
    if ".*" in error_substring:
        assert re.search(
            error_substring, msg, re.DOTALL
        ), f"Expected match for {error_substring!r} in {msg!r}"
    else:
        assert error_substring in msg, f"Expected {error_substring!r} in {msg!r}"


@pytest.mark.parametrize(
    "kwargs, error_substring",
    [
        # Required mode needs at least one function or builtin tool after filtering.
        ({"tools": [], "tool_choice": "required"}, "required"),
        # Named function choices must reference an existing function tool.
        (
            {
                "tools": make_tools(["get_weather"]),
                "tool_choice": {"type": "function", "function": {"name": "missing"}},
            },
            "missing",
        ),
        # Builtin choices must reference an existing builtin tool type.
        (
            {
                "tools": [
                    {
                        "type": "web_search_preview",
                        "name": "browser.search",
                        "parameters": SIMPLE_SCHEMA,
                    }
                ],
                "tool_choice": {"type": "code_interpreter"},
            },
            "exactly one",
        ),
        # Builtin choices by type are ambiguous when multiple builtin tools share a type.
        (
            {
                "tools": [
                    {
                        "type": "web_search_preview",
                        "name": "browser.search",
                        "parameters": SIMPLE_SCHEMA,
                    },
                    {
                        "type": "web_search_preview",
                        "name": "browser.open",
                        "parameters": SIMPLE_SCHEMA,
                    },
                ],
                "tool_choice": {"type": "web_search_preview"},
            },
            "exactly one",
        ),
        # Allowed tool refs must reference available builtin tools.
        (
            {
                "tools": make_tools(["get_weather"]),
                "tool_choice": {
                    "type": "allowed_tools",
                    "allowed_tools": {"mode": "auto", "tools": [{"type": "web_search_preview"}]},
                },
            },
            "Allowed builtin",
        ),
    ],
)
def test_public_api_validation_errors(kwargs: Dict[str, Any], error_substring: str):
    """Public API rejects invalid tool and tool_choice combinations."""

    with pytest.raises(ValueError) as exc_info:
        get_model_structural_tag("harmony", **kwargs)
    assert error_substring in str(exc_info.value)


def test_public_tool_shapes():
    """Public tools accepts dict, FunctionToolParam, and BuiltinToolParam values."""

    structural_tag = get_model_structural_tag(
        "harmony",
        tools=[
            {"type": "function", "function": {"name": "get_weather", "parameters": SIMPLE_SCHEMA}},
            FunctionToolParam(
                function={"name": "get_time", "parameters": {"type": "object", "properties": {}}}
            ),
            BuiltinToolParam(
                type="web_search_preview",
                name="browser.search",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ],
        reasoning=False,
    )

    begins = _collect_tag_begins(structural_tag)
    assert any("get_weather" in begin for begin in begins)
    assert any("get_time" in begin for begin in begins)
    assert any("browser.search" in begin for begin in begins)
    assert xgr.Grammar.from_structural_tag(structural_tag) is not None


def test_named_choice_forces_function():
    """Named function tool_choice is normalized to one forced function tool."""

    structural_tag = get_model_structural_tag(
        "llama",
        tools=make_tools(["get_weather", "get_time"]),
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        reasoning=False,
    )

    begins = _collect_tag_begins(structural_tag)
    assert any("get_weather" in begin for begin in begins)
    assert not any("get_time" in begin for begin in begins)


def test_builtin_choice_forces_builtin():
    """Builtin tool_choice is normalized to one forced builtin tool."""

    structural_tag = get_model_structural_tag(
        "harmony",
        tools=[
            *make_tools(["get_weather"]),
            {
                "type": "web_search_preview",
                "name": "browser.search",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            {
                "type": "code_interpreter",
                "name": "browser.open",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
            },
        ],
        tool_choice={"type": "web_search_preview"},
        reasoning=False,
    )

    begins = _collect_tag_begins(structural_tag)
    assert any("browser.search" in begin for begin in begins)
    assert not any("browser.open" in begin for begin in begins)
    assert not any("get_weather" in begin for begin in begins)


def test_allowed_choice_filters_tools():
    """Allowed tools filters both function tools and builtin tools."""

    structural_tag = get_model_structural_tag(
        "harmony",
        tools=[
            *make_tools(["get_weather", "get_time"]),
            {
                "type": "web_search_preview",
                "name": "browser.search",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            {
                "type": "code_interpreter",
                "name": "browser.open",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
            },
        ],
        tool_choice={
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "auto",
                "tools": [
                    {"type": "function", "function": {"name": "get_weather"}},
                    {"type": "web_search_preview"},
                ],
            },
        },
        reasoning=False,
    )

    begins = _collect_tag_begins(structural_tag)
    assert any("get_weather" in begin for begin in begins)
    assert any("browser.search" in begin for begin in begins)
    assert not any("get_time" in begin for begin in begins)
    assert not any("browser.open" in begin for begin in begins)


def test_none_parameters_unconstrained():
    """None parameters are converted to the True JSON schema."""

    structural_tag = get_model_structural_tag(
        "llama",
        tools=[{"type": "function", "function": {"name": "ping", "parameters": None}}],
        reasoning=False,
    )

    json_schema_values = _collect_json_schema_values(structural_tag)
    assert True in json_schema_values
    assert None not in json_schema_values


@pytest.mark.parametrize(
    "structural_tag_fn",
    [
        get_llama_structural_tag,
        get_kimi_structural_tag,
        get_deepseek_structural_tag,
        get_qwen_coder_structural_tag,
        get_qwen_structural_tag,
        get_harmony_structural_tag,
        get_deepseek_v3_2_structural_tag,
        get_deepseek_v4_structural_tag,
        get_minimax_structural_tag,
        get_glm47_structural_tag,
        get_gemma4_structural_tag,
    ],
)
@pytest.mark.parametrize(
    "case",
    [
        # Normal case: one function tool and one builtin tool are both available.
        {
            "tools": [
                FunctionToolParam(
                    function={
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    }
                )
            ],
            "builtin_tools": [
                BuiltinToolParam(
                    type="web_search_preview",
                    name="browser.search",
                    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                )
            ],
            "tool_choice": "auto",
        },
        # Empty auto case: public "none" is normalized to no tools plus "auto".
        {"tools": [], "builtin_tools": [], "tool_choice": "auto"},
        # None parameters case: missing schema must become unconstrained JSON.
        {
            "tools": [FunctionToolParam(function={"name": "ping", "parameters": None})],
            "builtin_tools": [],
            "tool_choice": "auto",
        },
        # Non-strict case: strict=False ignores the provided schema.
        {
            "tools": [
                FunctionToolParam(
                    function={
                        "name": "ping",
                        "parameters": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                        },
                        "strict": False,
                    }
                )
            ],
            "builtin_tools": [],
            "tool_choice": "auto",
        },
        # Forced case: the top-level API has already filtered to one tool.
        {
            "tools": [FunctionToolParam(function={"name": "ping", "parameters": None})],
            "builtin_tools": [],
            "tool_choice": "forced",
        },
    ],
)
def test_specific_functions_cases(structural_tag_fn, case: Dict[str, Any]):
    """Specific functions accept normalized internal inputs from the public API."""

    structural_tag = structural_tag_fn(
        tools=case["tools"],
        builtin_tools=case["builtin_tools"],
        tool_choice=case["tool_choice"],
        reasoning=True,
        force_empty_reasoning=False,
    )
    assert isinstance(structural_tag, StructuralTag)
    xgr.Grammar.from_structural_tag(structural_tag)
    assert None not in _collect_json_schema_values(structural_tag)


@pytest.mark.parametrize(
    "format_type, instance, is_accepted",
    [
        ("llama", '{"name": "t1", "parameters": {"q": "v"}}', True),
        (
            "kimi",
            '123<|tool_call_begin|>functions.t1:0<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|>',
            True,
        ),
        (
            "kimi",
            '123<|tool_call_begin|>functions.t2:0<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|>',
            False,
        ),
        (
            "deepseek_r1",
            'text<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>t1<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
            True,
        ),
        (
            "deepseek_r1",
            'text<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>t2{"q": "v"}<｜tool▁call▁end｜>',
            False,
        ),
        (
            "deepseek_v3_2",
            '<｜DSML｜function_calls>\n<｜DSML｜invoke name="t1">\n<q>{"type": "string"}</q>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
            True,
        ),
        (
            "deepseek_v3_2",
            '<｜DSML｜function_calls>\n<｜DSML｜invoke name="t2">\n<q>{"type": "string"}</q>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
            False,
        ),
        (
            "deepseek_v4",
            '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="t1">\n<q>{"type": "string"}</q>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n',
            True,
        ),
        (
            "deepseek_v4",
            '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="t2">\n<q>{"type": "string"}</q>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n',
            False,
        ),
        (
            "minimax",
            '<minimax:tool_call>\n<invoke name="t1">\n<q>{"type": "string"}</q>\n</invoke>\n</minimax:tool_call>\n',
            True,
        ),
        (
            "minimax",
            '<minimax:tool_call>\n<invoke name="t2">\n<q>{"type": "string"}</q>\n</invoke>\n</minimax:tool_call>\n',
            False,
        ),
        (
            "qwen_coder",
            '<tool_call>\n<function=t1>\n<q>{"type": "string"}</q>\n</function>\n</tool_call>',
            True,
        ),
        ("qwen", 'text<tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>', True),
        ("qwen", 'text<tool_call>\n{"name": "t2", "arguments": {"q": "v"}}\n</tool_call>', False),
        ("qwen", 'text<tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>', True),
        ("qwen", 'text<tool_call>\n{"name": "t2", "arguments": {"q": "v"}}\n</tool_call>', False),
        (
            "harmony",
            '<|channel|>commentary to=t1<|constrain|>json<|message|>{"q": "v"}<|call|>',
            True,
        ),
        (
            "harmony",
            '<|channel|>commentary to=t2<|constrain|>json<|message|>{"q": "v"}<|call|>',
            False,
        ),
    ],
)
@pytest.mark.parametrize(
    "tool",
    [
        {
            "function": {
                "name": "t1",
                "strict": False,
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        },
        # strict=False without parameters
        {"function": {"name": "t1", "strict": False}},
        # no strict, no parameters
        {"function": {"name": "t1"}},
    ],
)
def test_strict_or_missing_parameters(
    format_type: str, instance: str, is_accepted: bool, tool: Dict[str, Any]
):
    """strict=False or missing 'parameters' should still accept/reject instances correctly."""
    if format_type == "harmony":
        tools = [tool]
        stag = get_model_structural_tag(format_type, tools=tools, reasoning=False)
    else:
        tools = [tool]
        stag = get_model_structural_tag(format_type, tools=tools, reasoning=False)

    check_stag_with_instance(stag, instance, is_accepted)


# ---------- Test: instance positive / negative ----------

# Case: (input_dict, instances, reasoning, force_empty_reasoning, expected_grammar_ebnf, expected_accept_per_instance)
# input_dict may include tool_choice ("auto" | "forced" | "required") and forced_function_name (str | None).
# When expected_grammar_ebnf is empty or whitespace-only, grammar equality is skipped (fill in EBNF when ready).
InstanceCase = Tuple[Dict[str, Any], List[str], bool, bool, str, List[bool]]


def run_instance_case(format_type: str, case: InstanceCase):
    """Run one instance test case (grammar + accept/reject per instance string)."""
    (
        input_dict,
        instances,
        reasoning,
        force_empty_reasoning,
        expected_grammar_ebnf,
        expected_accept_per_instance,
    ) = case
    kwargs = _input_dict_to_get_stag_kwargs(format_type, input_dict)
    kwargs["reasoning"] = reasoning
    kwargs["force_empty_reasoning"] = force_empty_reasoning
    stag = get_model_structural_tag(**kwargs)
    check_stag_with_grammar(stag, expected_grammar_ebnf)
    for j, instance in enumerate(instances):
        check_stag_with_instance(stag, instance, expected_accept_per_instance[j])


def _run_instance_cases_for_style(
    format_type: str, cases: List[Tuple[Dict[str, Any], List[str], List[Tuple[str, List[bool]]]]]
):
    """Run instance accept/reject tests for a style. Each case is (input_dict, instances, expected_grammar_and_results)."""
    for input_dict, instances, expected_grammar_and_results in cases:
        assert (
            len(expected_grammar_and_results) == 3
        ), "3 modes: not reasoning, reasoning, empty reasoning"
        for i in range(3):
            current_grammar = expected_grammar_and_results[i][0]
            current_results = expected_grammar_and_results[i][1]
            if i == 0:
                reasoning = False
            else:
                reasoning = True

            if i == 2:
                force_empty_reasoning = True
            else:
                force_empty_reasoning = False

            kwargs = _input_dict_to_get_stag_kwargs(format_type, input_dict)
            kwargs["reasoning"] = reasoning
            kwargs["force_empty_reasoning"] = force_empty_reasoning
            stag = get_model_structural_tag(**kwargs)
            check_stag_with_grammar(stag, current_grammar)
            for j in range(len(instances)):
                instance = instances[j]
                is_accepted = current_results[j]
                check_stag_with_instance(stag, instance, is_accepted)


# ----- llama

_llama_instances_with_tools = [
    '{"name": "t1", "parameters": {"q": "v"}}',
    'text{"name": "t1", "parameters": {}}',
    '<think>123</think>text{"name": "t1", "parameters": {"q": ""}}',
    "<think>\n\n</think></think>",
    '<think>\n\n</think>text{"name": "t1", "parameters": {"q": "v"}}',
]
_llama_instances_no_tools = [
    "",
    "text",
    "<think>123</think>text",
    "<think>\n\n</think></think>",
    "<think>\n\n</think>text",
]

llama_instance_cases: List[InstanceCase] = [
    # with tools, reasoning=False, force_empty_reasoning=False
    (
        {"tools": _tools_llama},
        _llama_instances_with_tools,
        False,
        False,
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
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
        [True, True, False, False, False],
    ),
    # with tools, reasoning=True, force_empty_reasoning=False
    (
        {"tools": _tools_llama},
        _llama_instances_with_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
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
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
        [False, False, True, False, True],
    ),
    # with tools, reasoning=True, force_empty_reasoning=True
    (
        {"tools": _tools_llama},
        _llama_instances_with_tools,
        True,
        True,
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
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
    # no tools, reasoning=False, force_empty_reasoning=False
    (
        {},
        _llama_instances_no_tools,
        False,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
        [True, True, False, False, False],
    ),
    # no tools, reasoning=True, force_empty_reasoning=False
    (
        {},
        _llama_instances_no_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
any_text_1 ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
        [False, False, True, False, True],
    ),
    # no tools, reasoning=True, force_empty_reasoning=True
    (
        {},
        _llama_instances_no_tools,
        True,
        True,
        r"""const_string ::= (("<think>\n\n</think>"))
any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
]


@pytest.mark.parametrize("case", llama_instance_cases)
def test_llama_instances(case: InstanceCase):
    """get_model_structural_tag(llama) accepts/rejects instance as expected."""
    run_instance_case("llama", case)


# ----- kimi

_kimi_instances_with_tools = [
    '123<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|>',
    "123<|tool_call_begin|>123<|tool_call_argument_begin|>{}<|tool_call_end|>",
    "<think>123</think>",
    "<think></think></think>",
    "<think></think>123<|tool_calls_section_begin|>\n"
    + '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|>'
    + "\n<|tool_calls_section_end|>",
    "<think></think>123<|tool_calls_section_begin|>"
    + '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"q": "v0"}<|tool_call_end|>'
    + '<|tool_call_begin|>functions.get_weather:1<|tool_call_argument_begin|>{"q": "v1"}<|tool_call_end|>'
    + "<|tool_calls_section_end|>",
]
_kimi_instances_no_tools = [
    "",
    "text",
    "<think>123</think>",
    "<think>\n\n</think></think>",
    "<think></think>text",
    "</think>123",
]

kimi_instance_cases: List[InstanceCase] = [
    # with tools, reasoning=False, force_empty_reasoning=False
    (
        {"tools": _tools_kimi},
        _kimi_instances_with_tools,
        False,
        False,
        r"""root_0 ::= ((root_1))
root_1 ::= (([0-9] root_1) | ([0-9]))
const_string ::= (("<|tool_call_argument_begin|>"))
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
root_2 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
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
sequence ::= ((root_0 const_string root_2))
triggered_tags_group ::= (("functions.get_weather:" sequence "<|tool_call_end|>"))
triggered_tags ::= TagDispatch(
  ("<|tool_call_begin|>", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
        [True, False, False, False, False, False],
    ),
    # with tools, reasoning=True, force_empty_reasoning=False
    (
        {"tools": _tools_kimi},
        _kimi_instances_with_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
root_0 ::= ((root_1))
root_1 ::= (([0-9] root_1) | ([0-9]))
const_string ::= (("<|tool_call_argument_begin|>"))
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
root_2 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
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
sequence ::= ((root_0 const_string root_2))
triggered_tags_group ::= (("functions.get_weather:" sequence "<|tool_call_end|>"))
triggered_tags ::= TagDispatch(
  ("<|tool_call_begin|>", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence_1 ::= ((tag triggered_tags))
root ::= ((sequence_1))
""",
        [False, False, True, False, True, True],
    ),
    # with tools, reasoning=True, force_empty_reasoning=True
    (
        {"tools": _tools_kimi},
        _kimi_instances_with_tools,
        True,
        True,
        r"""const_string ::= (("<think></think>"))
root_0 ::= ((root_1))
root_1 ::= (([0-9] root_1) | ([0-9]))
const_string_1 ::= (("<|tool_call_argument_begin|>"))
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
root_2 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
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
sequence ::= ((root_0 const_string_1 root_2))
triggered_tags_group ::= (("functions.get_weather:" sequence "<|tool_call_end|>"))
triggered_tags ::= TagDispatch(
  ("<|tool_call_begin|>", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence_1 ::= ((const_string triggered_tags))
root ::= ((sequence_1))
""",
        [False, False, False, False, True, True],
    ),
    # no tools, reasoning=False, force_empty_reasoning=False
    (
        {},
        _kimi_instances_no_tools,
        False,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
        [True, True, False, False, False, False],
    ),
    # no tools, reasoning=True, force_empty_reasoning=False
    (
        {},
        _kimi_instances_no_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
any_text_1 ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
        [False, False, True, False, True, False],
    ),
    # no tools, reasoning=True, force_empty_reasoning=True
    (
        {},
        _kimi_instances_no_tools,
        True,
        True,
        r"""const_string ::= (("<think></think>"))
any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
        [False, False, False, False, True, False],
    ),
]


@pytest.mark.parametrize("case", kimi_instance_cases)
def test_kimi_instances(case: InstanceCase):
    """get_model_structural_tag(kimi) accepts/rejects instance as expected."""
    run_instance_case("kimi", case)


# ----- deepseek_r1

_deepseek_r1_instances_with_tools = [
    'text<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
    '123</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
    'thinking</think>text<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
    "</think>text<think>123</think>",
    '</think>text<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
]
_deepseek_r1_instances_no_tools = ["", "text", "123</think>123", "</think></think>", "</think>text"]

deepseek_r1_instance_cases: List[InstanceCase] = [
    # with tools, reasoning=False, force_empty_reasoning=False
    (
        {"tools": _tools_deepseek},
        _deepseek_r1_instances_with_tools,
        False,
        False,
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
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
        [True, False, False, False, False],
    ),
    # with tools, reasoning=True, force_empty_reasoning=False
    (
        {"tools": _tools_deepseek},
        _deepseek_r1_instances_with_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("" any_text "</think>"))
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
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
        [False, True, True, False, True],
    ),
    # with tools, reasoning=True, force_empty_reasoning=True
    (
        {"tools": _tools_deepseek},
        _deepseek_r1_instances_with_tools,
        True,
        True,
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
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
    # no tools, reasoning=False, force_empty_reasoning=False
    (
        {},
        _deepseek_r1_instances_no_tools,
        False,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
        [True, True, False, False, False],
    ),
    # no tools, reasoning=True, force_empty_reasoning=False
    (
        {},
        _deepseek_r1_instances_no_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("" any_text "</think>"))
any_text_1 ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
        [False, False, True, False, True],
    ),
    # no tools, reasoning=True, force_empty_reasoning=True
    (
        {},
        _deepseek_r1_instances_no_tools,
        True,
        True,
        r"""const_string ::= (("</think>"))
any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
]


@pytest.mark.parametrize("case", deepseek_r1_instance_cases)
def test_deepseek_r1_instances(case: InstanceCase):
    """get_model_structural_tag(deepseek_r1) accepts/rejects instance as expected."""
    run_instance_case("deepseek_r1", case)


# ----- deepseek_v3_2

_deepseek_v3_2_instances_with_tools = [
    'text<｜DSML｜function_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
    '<think>123</think><｜DSML｜function_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
    '<think>123</think>text<｜DSML｜function_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
    "<think>\n\n</think>text<think>123</think>",
    '<think>\n\n</think>text<｜DSML｜function_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
]
_deepseek_v3_2_instances_no_tools = [
    "",
    "text",
    "<think>123</think>123",
    "<think></think></think>",
    "<think>\n\n</think>text",
]

deepseek_v3_2_instance_cases: List[InstanceCase] = [
    # with tools, reasoning=False, force_empty_reasoning=False
    (
        {"tools": _tools_deepseek_v3_2},
        _deepseek_v3_2_instances_with_tools,
        False,
        False,
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
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator "</\uff5cDSML\uff5cfunction_calls>\n"))
triggered_tags ::= TagDispatch(
  ("<\uff5cDSML\uff5cfunction_calls>", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
        [True, False, False, False, False],
    ),
    # with tools, reasoning=True, force_empty_reasoning=False
    (
        {"tools": _tools_deepseek_v3_2},
        _deepseek_v3_2_instances_with_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
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
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag_1 ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tags_with_separator_tags ::= ((tag_1))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator "</\uff5cDSML\uff5cfunction_calls>\n"))
triggered_tags ::= TagDispatch(
  ("<\uff5cDSML\uff5cfunction_calls>", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
        [False, True, True, False, True],
    ),
    # with tools, reasoning=True, force_empty_reasoning=True
    (
        {"tools": _tools_deepseek_v3_2},
        _deepseek_v3_2_instances_with_tools,
        True,
        True,
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
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator "</\uff5cDSML\uff5cfunction_calls>\n"))
triggered_tags ::= TagDispatch(
  ("<\uff5cDSML\uff5cfunction_calls>", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
    # no tools, reasoning=False, force_empty_reasoning=False
    (
        {},
        _deepseek_v3_2_instances_no_tools,
        False,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
        [True, True, False, False, False],
    ),
    # no tools, reasoning=True, force_empty_reasoning=False
    (
        {},
        _deepseek_v3_2_instances_no_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
any_text_1 ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
        [False, False, True, False, True],
    ),
    # no tools, reasoning=True, force_empty_reasoning=True
    (
        {},
        _deepseek_v3_2_instances_no_tools,
        True,
        True,
        r"""const_string ::= (("<think>\n\n</think>"))
any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
]


@pytest.mark.parametrize("case", deepseek_v3_2_instance_cases)
def test_deepseek_v3_2_instances(case: InstanceCase):
    """get_model_structural_tag(deepseek_v3_2) accepts/rejects instance as expected."""
    run_instance_case("deepseek_v3_2", case)


# ----- deepseek_v4

_deepseek_v4_instances_with_tools = [
    'text<｜DSML｜tool_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>\n',
    '<think>123</think><｜DSML｜tool_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>\n',
    '<think>123</think>text<｜DSML｜tool_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>\n',
    "<think>\n\n</think>text<think>123</think>",
    '<think>\n\n</think>text<｜DSML｜tool_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>\n',
]
_deepseek_v4_instances_no_tools = [
    "",
    "text",
    "<think>123</think>123",
    "<think></think></think>",
    "<think>\n\n</think>text",
]

deepseek_v4_instance_cases: List[InstanceCase] = [
    # with tools, reasoning=False, force_empty_reasoning=False
    (
        {"tools": _tools_deepseek_v4},
        _deepseek_v4_instances_with_tools,
        False,
        False,
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
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator "</\uff5cDSML\uff5ctool_calls>\n"))
triggered_tags ::= TagDispatch(
  ("<\uff5cDSML\uff5ctool_calls>", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
        [True, False, False, False, False],
    ),
    # with tools, reasoning=True, force_empty_reasoning=False
    (
        {"tools": _tools_deepseek_v4},
        _deepseek_v4_instances_with_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
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
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag_1 ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tags_with_separator_tags ::= ((tag_1))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator "</\uff5cDSML\uff5ctool_calls>\n"))
triggered_tags ::= TagDispatch(
  ("<\uff5cDSML\uff5ctool_calls>", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
        [False, True, True, False, True],
    ),
    # with tools, reasoning=True, force_empty_reasoning=True
    (
        {"tools": _tools_deepseek_v4},
        _deepseek_v4_instances_with_tools,
        True,
        True,
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
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator "</\uff5cDSML\uff5ctool_calls>\n"))
triggered_tags ::= TagDispatch(
  ("<\uff5cDSML\uff5ctool_calls>", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
    # no tools, reasoning=False, force_empty_reasoning=False
    (
        {},
        _deepseek_v4_instances_no_tools,
        False,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
        [True, True, False, False, False],
    ),
    # no tools, reasoning=True, force_empty_reasoning=False
    (
        {},
        _deepseek_v4_instances_no_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
any_text_1 ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
        [False, False, True, False, True],
    ),
    # no tools, reasoning=True, force_empty_reasoning=True
    (
        {},
        _deepseek_v4_instances_no_tools,
        True,
        True,
        r"""const_string ::= (("<think>\n\n</think>"))
any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
]


@pytest.mark.parametrize("case", deepseek_v4_instance_cases)
def test_deepseek_v4_instances(case: InstanceCase):
    """get_model_structural_tag(deepseek_v4) accepts/rejects instance as expected."""
    run_instance_case("deepseek_v4", case)


# ----- minimax

_minimax_instances_with_tools = [
    'text<minimax:tool_call>\n<invoke name="search">\n<parameter name="q">v</parameter></invoke>\n</minimax:tool_call>\n',
    '<think>123</think><minimax:tool_call>\n<invoke name="search">\n<parameter name="q">v</parameter></invoke>\n</minimax:tool_call>\n',
    '<think>123</think>text<minimax:tool_call>\n<invoke name="search">\n<parameter name="q">v</parameter></invoke>\n</minimax:tool_call>\n',
    "<think>\n\n</think>text<think>123</think>",
    '<think>\n\n</think>text<minimax:tool_call>\n<invoke name="search">\n<parameter name="q">v</parameter></invoke>\n</minimax:tool_call>\n',
]
_minimax_instances_no_tools = [
    "",
    "text",
    "<think>123</think>123",
    "<think></think></think>",
    "<think>\n\n</think>text",
]

minimax_instance_cases: List[InstanceCase] = [
    # with tools, reasoning=False, force_empty_reasoning=False
    (
        {"tools": _tools_minimax},
        _minimax_instances_with_tools,
        False,
        False,
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
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<parameter name=\"q\">" [ \n\t]* xml_string [ \n\t]* "</parameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag ::= (("<invoke name=\"search\">\n" root_0 "</invoke>\n"))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator "</minimax:tool_call>\n"))
triggered_tags ::= TagDispatch(
  ("<minimax:tool_call>", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
        [True, False, False, False, False],
    ),
    # with tools, reasoning=True, force_empty_reasoning=False
    (
        {"tools": _tools_minimax},
        _minimax_instances_with_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
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
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<parameter name=\"q\">" [ \n\t]* xml_string [ \n\t]* "</parameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag_1 ::= (("<invoke name=\"search\">\n" root_0 "</invoke>\n"))
tags_with_separator_tags ::= ((tag_1))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator "</minimax:tool_call>\n"))
triggered_tags ::= TagDispatch(
  ("<minimax:tool_call>", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
        [False, True, True, False, True],
    ),
    # with tools, reasoning=True, force_empty_reasoning=True
    (
        {"tools": _tools_minimax},
        _minimax_instances_with_tools,
        True,
        True,
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
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<parameter name=\"q\">" [ \n\t]* xml_string [ \n\t]* "</parameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag ::= (("<invoke name=\"search\">\n" root_0 "</invoke>\n"))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
triggered_tags_group ::= (("\n" tags_with_separator "</minimax:tool_call>\n"))
triggered_tags ::= TagDispatch(
  ("<minimax:tool_call>", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
    # no tools, reasoning=False, force_empty_reasoning=False
    (
        {},
        _minimax_instances_no_tools,
        False,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
        [True, True, False, False, False],
    ),
    # no tools, reasoning=True, force_empty_reasoning=False
    (
        {},
        _minimax_instances_no_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
any_text_1 ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
        [False, False, True, False, True],
    ),
    # no tools, reasoning=True, force_empty_reasoning=True
    (
        {},
        _minimax_instances_no_tools,
        True,
        True,
        r"""const_string ::= (("<think>\n\n</think>"))
any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
]


@pytest.mark.parametrize("case", minimax_instance_cases)
def test_minimax_instances(case: InstanceCase):
    """get_model_structural_tag(minimax) accepts/rejects instance as expected."""
    run_instance_case("minimax", case)


def test_glm47_instances():
    """get_model_structural_tag(glm47) accepts/rejects instance as expected."""
    stag = get_model_structural_tag("glm47", tools=_tools_glm47, reasoning=False)
    grammar_str = str(xgr.Grammar.from_structural_tag(stag))
    assert "<tool_call>" in grammar_str
    assert "<arg_key>" in grammar_str
    assert "<arg_value>" in grammar_str

    check_stag_with_instance(
        stag, "<tool_call>search<arg_key>q</arg_key><arg_value>v</arg_value></tool_call>", True
    )
    check_stag_with_instance(stag, "<tool_call>search<parameter=q>v</parameter></tool_call>", False)
    check_stag_with_instance(
        stag, '<tool_call>search<parameter name="q">v</parameter></tool_call>', False
    )


# ----- qwen_coder

_qwen_coder_instances_with_tools = [
    "<tool_call>\n<function=run_sql>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=other>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
    "<think>123</think><tool_call>\n<function=run_sql>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
    "<think>\n\n</think><think></think>",
    "<think>\n\n</think>text<tool_call>\n<function=run_sql>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
]
_qwen_coder_instances_no_tools = [
    "",
    "text",
    "<think>123</think>123",
    "<think>\n\n</think></think>",
    "<think>\n\n</think>text",
]

qwen_coder_instance_cases: List[InstanceCase] = [
    # with tools, reasoning=False, force_empty_reasoning=False
    (
        {"tools": _tools_qwen_coder},
        _qwen_coder_instances_with_tools,
        False,
        False,
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
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<parameter=q>" [ \n\t]* xml_string [ \n\t]* "</parameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("run_sql>\n" root_0 "\n</function>\n</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>\n<function=", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
        [True, False, False, False, False],
    ),
    # with tools, reasoning=True, force_empty_reasoning=False
    (
        {"tools": _tools_qwen_coder},
        _qwen_coder_instances_with_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
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
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<parameter=q>" [ \n\t]* xml_string [ \n\t]* "</parameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("run_sql>\n" root_0 "\n</function>\n</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>\n<function=", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
        [False, False, True, False, True],
    ),
    # with tools, reasoning=True, force_empty_reasoning=True
    (
        {"tools": _tools_qwen_coder},
        _qwen_coder_instances_with_tools,
        True,
        True,
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
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<parameter=q>" [ \n\t]* xml_string [ \n\t]* "</parameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("run_sql>\n" root_0 "\n</function>\n</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>\n<function=", triggered_tags_group),
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
    # no tools, reasoning=False, force_empty_reasoning=False
    (
        {},
        _qwen_coder_instances_no_tools,
        False,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
        [True, True, False, False, False],
    ),
    # no tools, reasoning=True, force_empty_reasoning=False
    (
        {},
        _qwen_coder_instances_no_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
any_text_1 ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
        [False, False, True, False, True],
    ),
    # no tools, reasoning=True, force_empty_reasoning=True
    (
        {},
        _qwen_coder_instances_no_tools,
        True,
        True,
        r"""const_string ::= (("<think>\n\n</think>"))
any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
]


@pytest.mark.parametrize("case", qwen_coder_instance_cases)
def test_qwen_coder_instances(case: InstanceCase):
    """get_model_structural_tag(qwen_coder) accepts/rejects instance as expected."""
    run_instance_case("qwen_coder", case)


# ----- qwen

_qwen_instances_with_tools = [
    'text<tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>',
    '<think>123</think><tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>',
    "<think>\n\n</think></think>",
    '<think>\n\n</think><tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>',
]
_qwen_instances_no_tools = [
    "",
    "text",
    "<think>123</think>123",
    "<think>\n\n</think></think>",
    "<think>\n\n</think>text",
]

qwen_instance_cases: List[InstanceCase] = [
    # with tools, reasoning=False, force_empty_reasoning=False
    (
        {"tools": _tools_qwen},
        _qwen_instances_with_tools,
        False,
        False,
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
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
root ::= ((triggered_tags))
""",
        [True, False, False, False],
    ),
    # with tools, reasoning=True, force_empty_reasoning=False
    (
        {"tools": _tools_qwen},
        _qwen_instances_with_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
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
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag triggered_tags))
root ::= ((sequence))
""",
        [False, True, False, True],
    ),
    # with tools, reasoning=True, force_empty_reasoning=True
    (
        {"tools": _tools_qwen},
        _qwen_instances_with_tools,
        True,
        True,
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
  loop_after_dispatch=true,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string triggered_tags))
root ::= ((sequence))
""",
        [False, False, False, True],
    ),
    # no tools, reasoning=False, force_empty_reasoning=False
    (
        {},
        _qwen_instances_no_tools,
        False,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
root ::= ((any_text))
""",
        [True, True, False, False, False],
    ),
    # no tools, reasoning=True, force_empty_reasoning=False
    (
        {},
        _qwen_instances_no_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("</think>")
)
tag ::= (("<think>" any_text "</think>"))
any_text_1 ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((tag any_text_1))
root ::= ((sequence))
""",
        [False, False, True, False, True],
    ),
    # no tools, reasoning=True, force_empty_reasoning=True
    (
        {},
        _qwen_instances_no_tools,
        True,
        True,
        r"""const_string ::= (("<think>\n\n</think>"))
any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<think>", "</think>")
)
sequence ::= ((const_string any_text))
root ::= ((sequence))
""",
        [False, False, False, False, True],
    ),
]


@pytest.mark.parametrize("case", qwen_instance_cases)
def test_qwen_instances(case: InstanceCase):
    """get_model_structural_tag(qwen) accepts/rejects instance as expected."""
    run_instance_case("qwen", case)


# ----- harmony

_harmony_instances_with_tools = [
    "<|channel|>analysis<|message|><|end|>",
    '<|channel|>commentary to=comment_tool<|constrain|>json<|message|>{"q": "v"}<|call|>',
    '<|channel|>analysis to=analysis_tool<|message|>{"q": "v"}<|call|>',
    "<|channel|>commentary to=wrong_tool<|constrain|>json<|message|>{}<|call|>",
    "<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>123<|end|>",
    '<|channel|>commentary to=comment_tool<|constrain|>json<|message|>{"q": "v"}<|call|>',
]
_harmony_instances_no_tools = [
    "",
    "<|channel|>final<|message|>123<|end|>",
    "<|channel|>analysis<|message|>123<|end|><|start|>assistant<|channel|>final<|message|>123<|end|>",
    "<|channel|>analysis<|message|><|end|>",
    "<think>\n\n</think>text",
]

harmony_instance_cases: List[InstanceCase] = [
    # with tools, reasoning=False, force_empty_reasoning=False
    (
        {"tools": _tools_harmony, "builtin_tools": _builtin_harmony},
        _harmony_instances_with_tools,
        False,
        False,
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
tag_1 ::= (("<|channel|>analysis to=analysis_tool<|message|>" root_0 "<|call|>"))
any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<|end|>")
)
tag_2 ::= (("<|channel|>final<|message|>" any_text "<|end|>"))
tags_with_separator_tags ::= ((tag) | (tag_1) | (tag_2))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
        [False, True, True, False, False, True],
    ),
    # with tools, reasoning=True, force_empty_reasoning=False
    (
        {"tools": _tools_harmony, "builtin_tools": _builtin_harmony},
        _harmony_instances_with_tools,
        True,
        False,
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
tag_1 ::= (("<|channel|>analysis to=analysis_tool<|message|>" root_0 "<|call|>"))
any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<|end|>")
)
tag_2 ::= (("<|channel|>final<|message|>" any_text "<|end|>"))
tag_3 ::= (("<|channel|>analysis<|message|>" any_text "<|end|>"))
tags_with_separator_tags ::= ((tag) | (tag_1) | (tag_2) | (tag_3))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
        [True, True, True, False, True, True],
    ),
    # with tools, reasoning=True, force_empty_reasoning=True
    (
        {"tools": _tools_harmony, "builtin_tools": _builtin_harmony},
        _harmony_instances_with_tools,
        True,
        True,
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
tag_1 ::= (("<|channel|>analysis to=analysis_tool<|message|>" root_0 "<|call|>"))
any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<|end|>")
)
tag_2 ::= (("<|channel|>final<|message|>" any_text "<|end|>"))
const_string ::= (("<|end|>"))
tag_3 ::= (("<|channel|>analysis<|message|>" const_string))
tags_with_separator_tags ::= ((tag) | (tag_1) | (tag_2) | (tag_3))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
        [True, True, True, False, False, True],
    ),
    # no tools, reasoning=False, force_empty_reasoning=False
    (
        {},
        _harmony_instances_no_tools,
        False,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<|end|>")
)
tag ::= (("<|channel|>final<|message|>" any_text "<|end|>"))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
        [True, True, False, False, False],
    ),
    # no tools, reasoning=True, force_empty_reasoning=False
    (
        {},
        _harmony_instances_no_tools,
        True,
        False,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<|end|>")
)
tag ::= (("<|channel|>final<|message|>" any_text "<|end|>"))
tag_1 ::= (("<|channel|>analysis<|message|>" any_text "<|end|>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
        [True, True, True, True, False],
    ),
    # no tools, reasoning=True, force_empty_reasoning=True
    (
        {},
        _harmony_instances_no_tools,
        True,
        True,
        r"""any_text ::= TagDispatch(
  loop_after_dispatch=false,
  excludes=("<|end|>")
)
tag ::= (("<|channel|>final<|message|>" any_text "<|end|>"))
const_string ::= (("<|end|>"))
tag_1 ::= (("<|channel|>analysis<|message|>" const_string))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
        [True, True, False, True, False],
    ),
]


@pytest.mark.parametrize("case", harmony_instance_cases)
def test_harmony_instances(case: InstanceCase):
    """get_model_structural_tag(harmony) accepts/rejects instance as expected."""
    run_instance_case("harmony", case)


# tool_choice=required / forced: expected_grammar_ebnf left "" for manual completion.
_tool_choice_instance_cases = [
    pytest.param(
        "llama",
        (
            {"tools": _tools_llama_pair, "tool_choice": "required"},
            ["", '{"name": "t1", "parameters": {"q": "v"}}'],
            False,
            False,
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
tag ::= (("{\"name\": \"t1\", \"parameters\": " root_0 "}"))
tag_1 ::= (("{\"name\": \"t2\", \"parameters\": " root_0 "}"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
            [False, True],
        ),
        id="llama-required",
    ),
    pytest.param(
        "llama",
        (
            {"tools": _tools_llama_pair, "tool_choice": "forced", "forced_function_name": "t1"},
            [
                '{"name": "t1", "parameters": {"q": "v"}}',
                '{"name": "t2", "parameters": {"q": "v"}}',
            ],
            False,
            False,
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
tag ::= (("{\"name\": \"t1\", \"parameters\": " root_0 "}"))
root ::= ((tag))
""",
            [True, False],
        ),
        id="llama-forced",
    ),
    pytest.param(
        "kimi",
        (
            {"tools": _tools_kimi_pair, "tool_choice": "required"},
            [
                "",
                '<|tool_call_begin|>functions.t1:0<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|>',
            ],
            False,
            False,
            r"""root_0 ::= ((root_1))
root_1 ::= (([0-9] root_1) | ([0-9]))
const_string ::= (("<|tool_call_argument_begin|>"))
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
root_2 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
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
sequence ::= ((root_0 const_string root_2))
tag ::= (("<|tool_call_begin|>functions.t1:" sequence "<|tool_call_end|>"))
tag_1 ::= (("<|tool_call_begin|>functions.t2:" sequence "<|tool_call_end|>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
            [False, True],
        ),
        id="kimi-required",
    ),
    pytest.param(
        "kimi",
        (
            {"tools": _tools_kimi_pair, "tool_choice": "forced", "forced_function_name": "t1"},
            [
                '<|tool_call_begin|>functions.t1:0<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|>',
                '<|tool_call_begin|>functions.t2:0<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|>',
            ],
            False,
            False,
            r"""root_0 ::= ((root_1))
root_1 ::= (([0-9] root_1) | ([0-9]))
const_string ::= (("<|tool_call_argument_begin|>"))
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
root_2 ::= (("{" [ \n\t]* "\"q\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
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
sequence ::= ((root_0 const_string root_2))
tag ::= (("<|tool_call_begin|>functions.t1:" sequence "<|tool_call_end|>"))
root ::= ((tag))
""",
            [True, False],
        ),
        id="kimi-forced",
    ),
    pytest.param(
        "deepseek_r1",
        (
            {"tools": _tools_deepseek_pair, "tool_choice": "required"},
            [
                "",
                '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
            ],
            False,
            False,
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
tag ::= (("<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>search<\uff5ctool\u2581sep\uff5c>" root_0 "<\uff5ctool\u2581call\u2581end\uff5c>"))
tag_1 ::= (("<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>alt<\uff5ctool\u2581sep\uff5c>" root_0 "<\uff5ctool\u2581call\u2581end\uff5c>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
            [False, True],
        ),
        id="deepseek_r1-required",
    ),
    pytest.param(
        "deepseek_r1",
        (
            {
                "tools": _tools_deepseek_pair,
                "tool_choice": "forced",
                "forced_function_name": "search",
            },
            [
                '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
                '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>alt<｜tool▁sep｜>{"q": "v"}<｜tool▁call▁end｜>',
            ],
            False,
            False,
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
tag ::= (("<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>search<\uff5ctool\u2581sep\uff5c>" root_0 "<\uff5ctool\u2581call\u2581end\uff5c>"))
root ::= ((tag))
""",
            [True, False],
        ),
        id="deepseek_r1-forced",
    ),
    pytest.param(
        "deepseek_v3_2",
        (
            {"tools": _tools_deepseek_v3_2_pair, "tool_choice": "required"},
            [
                "",
                '<｜DSML｜function_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
            ],
            False,
            False,
            r"""const_string ::= (("<\uff5cDSML\uff5cfunction_calls>\n"))
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
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tag_1 ::= (("<\uff5cDSML\uff5cinvoke name=\"alt\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
const_string_1 ::= (("</\uff5cDSML\uff5cfunction_calls>\n"))
sequence ::= ((const_string tags_with_separator const_string_1))
root ::= ((sequence))
""",
            [False, True],
        ),
        id="deepseek_v3_2-required",
    ),
    pytest.param(
        "deepseek_v4",
        (
            {"tools": _tools_deepseek_v4_pair, "tool_choice": "required"},
            [
                "",
                '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>\n',
            ],
            False,
            False,
            r"""const_string ::= (("<\uff5cDSML\uff5ctool_calls>\n"))
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
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tag_1 ::= (("<\uff5cDSML\uff5cinvoke name=\"alt\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
const_string_1 ::= (("</\uff5cDSML\uff5ctool_calls>\n"))
sequence ::= ((const_string tags_with_separator const_string_1))
root ::= ((sequence))
""",
            [False, True],
        ),
        id="deepseek_v4-required",
    ),
    pytest.param(
        "deepseek_v3_2",
        (
            {
                "tools": _tools_deepseek_v3_2_pair,
                "tool_choice": "forced",
                "forced_function_name": "search",
            },
            [
                '<｜DSML｜function_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
                '<｜DSML｜function_calls>\n<｜DSML｜invoke name="alt">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>\n',
            ],
            False,
            False,
            r"""const_string ::= (("<\uff5cDSML\uff5cfunction_calls>\n"))
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
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
const_string_1 ::= (("</\uff5cDSML\uff5cfunction_calls>\n"))
sequence ::= ((const_string tag const_string_1))
root ::= ((sequence))
""",
            [True, False],
        ),
        id="deepseek_v3_2-forced",
    ),
    pytest.param(
        "deepseek_v4",
        (
            {
                "tools": _tools_deepseek_v4_pair,
                "tool_choice": "forced",
                "forced_function_name": "search",
            },
            [
                '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>\n',
                '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="alt">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>\n',
            ],
            False,
            False,
            r"""const_string ::= (("<\uff5cDSML\uff5ctool_calls>\n"))
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
  loop_after_dispatch=false,
  excludes=("</\uff5cDSML\uff5cparameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_2 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"q\" string=\"" root_1 "\">" [ \n\t]* xml_string [ \n\t]* "</\uff5cDSML\uff5cparameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<\uff5cDSML\uff5cparameter name=\"" xml_variable_name "\" string=\"" xml_object_1_1 "\">" [ \n\t]* xml_any [ \n\t]* "</\uff5cDSML\uff5cparameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
xml_object_2 ::= (("true") | ("false"))
root_1 ::= (("true") | ("false"))
xml_object_1_1 ::= (("true") | ("false"))
tag ::= (("<\uff5cDSML\uff5cinvoke name=\"search\">\n" root_0 "</\uff5cDSML\uff5cinvoke>\n"))
const_string_1 ::= (("</\uff5cDSML\uff5ctool_calls>\n"))
sequence ::= ((const_string tag const_string_1))
root ::= ((sequence))
""",
            [True, False],
        ),
        id="deepseek_v4-forced",
    ),
    pytest.param(
        "minimax",
        (
            {"tools": _tools_minimax_pair, "tool_choice": "required"},
            [
                "",
                '<minimax:tool_call>\n<invoke name="search">\n<parameter name="q">v</parameter></invoke>\n</minimax:tool_call>\n',
            ],
            False,
            False,
            r"""const_string ::= (("<minimax:tool_call>\n"))
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
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<parameter name=\"q\">" [ \n\t]* xml_string [ \n\t]* "</parameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag ::= (("<invoke name=\"search\">\n" root_0 "</invoke>\n"))
tag_1 ::= (("<invoke name=\"alt\">\n" root_0 "</invoke>\n"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
const_string_1 ::= (("</minimax:tool_call>\n"))
sequence ::= ((const_string tags_with_separator const_string_1))
root ::= ((sequence))
""",
            [False, True],
        ),
        id="minimax-required",
    ),
    pytest.param(
        "minimax",
        (
            {
                "tools": _tools_minimax_pair,
                "tool_choice": "forced",
                "forced_function_name": "search",
            },
            [
                '<minimax:tool_call>\n<invoke name="search">\n<parameter name="q">v</parameter></invoke>\n</minimax:tool_call>\n',
                '<minimax:tool_call>\n<invoke name="alt">\n<parameter name="q">v</parameter></invoke>\n</minimax:tool_call>\n',
            ],
            False,
            False,
            r"""const_string ::= (("<minimax:tool_call>\n"))
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
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<parameter name=\"q\">" [ \n\t]* xml_string [ \n\t]* "</parameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<parameter name=\"" xml_variable_name "\">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag ::= (("<invoke name=\"search\">\n" root_0 "</invoke>\n"))
const_string_1 ::= (("</minimax:tool_call>\n"))
sequence ::= ((const_string tag const_string_1))
root ::= ((sequence))
""",
            [True, False],
        ),
        id="minimax-forced",
    ),
    pytest.param(
        "qwen_coder",
        (
            {"tools": _tools_qwen_coder_pair, "tool_choice": "required"},
            [
                "",
                "<tool_call>\n<function=run_sql>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
            ],
            False,
            False,
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
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<parameter=q>" [ \n\t]* xml_string [ \n\t]* "</parameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag ::= (("<tool_call>\n<function=run_sql>\n" root_0 "\n</function>\n</tool_call>"))
tag_1 ::= (("<tool_call>\n<function=run_py>\n" root_0 "\n</function>\n</tool_call>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
            [False, True],
        ),
        id="qwen_coder-required",
    ),
    pytest.param(
        "qwen_coder",
        (
            {
                "tools": _tools_qwen_coder_pair,
                "tool_choice": "forced",
                "forced_function_name": "run_sql",
            },
            [
                "<tool_call>\n<function=run_sql>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
                "<tool_call>\n<function=run_py>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
            ],
            False,
            False,
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
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<parameter=q>" [ \n\t]* xml_string [ \n\t]* "</parameter>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag ::= (("<tool_call>\n<function=run_sql>\n" root_0 "\n</function>\n</tool_call>"))
root ::= ((tag))
""",
            [True, False],
        ),
        id="qwen_coder-forced",
    ),
    pytest.param(
        "qwen",
        (
            {"tools": _tools_qwen_pair, "tool_choice": "required"},
            ["", '<tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>'],
            False,
            False,
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
tag ::= (("<tool_call>\n{\"name\": \"t1\", \"arguments\": " root_0 "}\n</tool_call>"))
tag_1 ::= (("<tool_call>\n{\"name\": \"t2\", \"arguments\": " root_0 "}\n</tool_call>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
            [False, True],
        ),
        id="qwen-required",
    ),
    pytest.param(
        "qwen",
        (
            {"tools": _tools_qwen_pair, "tool_choice": "forced", "forced_function_name": "t1"},
            [
                '<tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>',
                '<tool_call>\n{"name": "t2", "arguments": {"q": "v"}}\n</tool_call>',
            ],
            False,
            False,
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
tag ::= (("<tool_call>\n{\"name\": \"t1\", \"arguments\": " root_0 "}\n</tool_call>"))
root ::= ((tag))
""",
            [True, False],
        ),
        id="qwen-forced",
    ),
    pytest.param(
        "harmony",
        (
            {"tools": _tools_harmony_pair, "tool_choice": "required"},
            [
                "plain text without channels",
                '<|channel|>commentary to=comment_tool<|constrain|>json<|message|>{"q": "v"}<|call|>',
            ],
            False,
            False,
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
tag_1 ::= (("<|channel|>commentary to=other_tool<|constrain|>json<|message|>" root_0 "<|call|>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
            [False, True],
        ),
        id="harmony-required",
    ),
    pytest.param(
        "harmony",
        (
            {
                "tools": _tools_harmony_pair,
                "tool_choice": "forced",
                "forced_function_name": "comment_tool",
            },
            [
                '<|channel|>commentary to=comment_tool<|constrain|>json<|message|>{"q": "v"}<|call|>',
                '<|channel|>commentary to=other_tool<|constrain|>json<|message|>{"q": "v"}<|call|>',
            ],
            False,
            False,
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
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= ("" | ("<|start|>assistant" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
            [True, False],
        ),
        id="harmony-forced",
    ),
    pytest.param(
        "glm47",
        (
            {"tools": _tools_glm47_pair, "tool_choice": "required"},
            ["", "<tool_call>search<arg_key>q</arg_key><arg_value>v</arg_value></tool_call>"],
            False,
            False,
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
  loop_after_dispatch=false,
  excludes=("</arg_value>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<arg_key>" xml_variable_name "</arg_key>" [ \n\t]* "<arg_value>" [ \n\t]* xml_any [ \n\t]* "</arg_value>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<arg_key>q</arg_key>" [ \n\t]* "<arg_value>" [ \n\t]* xml_string [ \n\t]* "</arg_value>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<arg_key>" xml_variable_name "</arg_key>" [ \n\t]* "<arg_value>" [ \n\t]* xml_any [ \n\t]* "</arg_value>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag ::= (("<tool_call>search" root_0 "</tool_call>"))
tag_1 ::= (("<tool_call>alt" root_0 "</tool_call>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
            [False, True],
        ),
        id="glm47-required",
    ),
    pytest.param(
        "gemma4",
        (
            {"tools": _tools_gemma4_pair, "tool_choice": "required"},
            ["", '<|tool_call>call:search{"q": "v"}<tool_call|>'],
            False,
            False,
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
tag ::= (("<|tool_call>call:search" root_0 "<tool_call|>"))
tag_1 ::= (("<|tool_call>call:alt" root_0 "<tool_call|>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
            [False, True],
        ),
        id="gemma4-required",
    ),
    pytest.param(
        "glm47",
        (
            {"tools": _tools_glm47_pair, "tool_choice": "forced", "forced_function_name": "search"},
            [
                "<tool_call>search<arg_key>q</arg_key><arg_value>v</arg_value></tool_call>",
                "<tool_call>alt<arg_key>q</arg_key><arg_value>v</arg_value></tool_call>",
            ],
            False,
            False,
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
  loop_after_dispatch=false,
  excludes=("</arg_value>")
)
xml_any ::= ((xml_string) | (basic_array) | (basic_object))
xml_object ::= (([ \n\t]* "<arg_key>" xml_variable_name "</arg_key>" [ \n\t]* "<arg_value>" [ \n\t]* xml_any [ \n\t]* "</arg_value>" xml_object_1 [ \n\t]*) | ([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
root_0 ::= (([ \n\t]* "<arg_key>q</arg_key>" [ \n\t]* "<arg_value>" [ \n\t]* xml_string [ \n\t]* "</arg_value>" [ \n\t]*) | ([ \n\t]*))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
xml_object_1 ::= ("" | ([ \n\t]* "<arg_key>" xml_variable_name "</arg_key>" [ \n\t]* "<arg_value>" [ \n\t]* xml_any [ \n\t]* "</arg_value>" xml_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
tag ::= (("<tool_call>search" root_0 "</tool_call>"))
root ::= ((tag))
""",
            [True, False],
        ),
        id="glm47-forced",
    ),
]


@pytest.mark.parametrize("format_type, case", _tool_choice_instance_cases)
def test_tool_choice_instances(format_type: str, case: InstanceCase):
    """tool_choice required/forced: instance checks; fill expected EBNF in cases when ready."""
    run_instance_case(format_type, case)


_TOOLS: List[Dict[str, Any]] = [
    {"function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
]


@pytest.mark.parametrize(
    "format_type, kwargs",
    [
        ("llama", {"tools": _TOOLS}),
        ("kimi", {"tools": _TOOLS}),
        ("deepseek_r1", {"tools": _TOOLS}),
        ("qwen_coder", {"tools": _TOOLS}),
        ("qwen", {"tools": _TOOLS}),
        ("deepseek_v3_2", {"tools": _TOOLS}),
        ("deepseek_v4", {"tools": _TOOLS}),
        ("minimax", {"tools": _TOOLS}),
        ("gemma4", {"tools": _TOOLS}),
        (
            "harmony",
            {
                "tools": [
                    *_TOOLS,
                    {
                        "type": "builtin_get_time",
                        "name": "builtin_get_time",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ]
            },
        ),
    ],
)
def test_no_parameter_tools_build_grammar(format_type: str, kwargs: Dict[str, Any]):
    """Smoke test: each built-in format can generate StructuralTag and build Grammar."""
    structural_tag = get_model_structural_tag(format_type, **kwargs)
    grammar = xgr.Grammar.from_structural_tag(structural_tag)
    assert grammar is not None


# ----- gemma4


def test_gemma4_instances():
    """get_model_structural_tag(gemma4) accepts/rejects instances as expected."""

    # --- No tools, no reasoning: plain text only ---
    stag = get_model_structural_tag("gemma4", reasoning=False)
    check_stag_with_instance(stag, "Hello world", True)
    check_stag_with_instance(stag, "", True)

    # --- No tools, reasoning enabled ---
    stag = get_model_structural_tag("gemma4", reasoning=True)
    check_stag_with_instance(
        stag, "<|channel>thought\nLet me think...<channel|>The answer is 42.", True
    )
    check_stag_with_instance(stag, "<|channel>thought\n<channel|>Quick answer.", True)
    # Missing thinking prefix should be rejected
    check_stag_with_instance(stag, "No thinking here.", False)

    # --- No tools, reasoning with force_empty_reasoning ---
    stag = get_model_structural_tag("gemma4", reasoning=True, force_empty_reasoning=True)
    check_stag_with_instance(stag, "<|channel>thought\n<channel|>Direct answer.", True)
    # Non-empty thinking should be rejected when force_empty
    check_stag_with_instance(stag, "<|channel>thought\nSome reasoning<channel|>Answer.", False)

    # --- With tools, no reasoning ---
    stag = get_model_structural_tag("gemma4", tools=_tools_gemma4, reasoning=False)
    check_stag_with_instance(stag, '<|tool_call>call:get_weather{"q": "Paris"}<tool_call|>', True)
    check_stag_with_instance(stag, "Plain text without tool call.", True)
    # Wrong function name should be rejected
    check_stag_with_instance(stag, '<|tool_call>call:unknown_func{"q": "x"}<tool_call|>', False)

    # --- Multiple tool calls ---
    check_stag_with_instance(
        stag,
        '<|tool_call>call:get_weather{"q": "Paris"}<tool_call|>'
        '<|tool_call>call:get_weather{"q": "London"}<tool_call|>',
        True,
    )

    # --- With tools, reasoning enabled ---
    stag = get_model_structural_tag("gemma4", tools=_tools_gemma4, reasoning=True)
    check_stag_with_instance(
        stag,
        "<|channel>thought\nI should check the weather.<channel|>"
        '<|tool_call>call:get_weather{"q": "Paris"}<tool_call|>',
        True,
    )
    check_stag_with_instance(
        stag, "<|channel>thought\nThinking...<channel|>Just text, no tool call.", True
    )


if __name__ == "__main__":
    pytest.main(sys.argv)
