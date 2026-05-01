"""Tests for get_structural_tag_for_model and generated structural tags."""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.builtin_structural_tag import (
    get_deepseek_r1_structural_tag,
    get_deepseek_v3_1_structural_tag,
    get_deepseek_v3_2_structural_tag,
    get_deepseek_v4_structural_tag,
    get_glm_4_7_structural_tag,
    get_harmony_structural_tag,
    get_kimi_structural_tag,
    get_llama_structural_tag,
    get_minimax_structural_tag,
    get_model_structural_tag,
    get_qwen_3_5_structural_tag,
    get_qwen_3_coder_structural_tag,
    get_qwen_3_structural_tag,
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
_tools_qwen_3_coder = make_tools(["run_sql"])
_tools_qwen_3 = make_tools(["t1"])
_tools_qwen_3_5 = make_tools(["run_sql"])
_tools_harmony = make_tools(["comment_tool"])
_builtin_harmony = make_tools(["analysis_tool"])
_tools_deepseek_v3_2 = make_tools(["search"])
_tools_deepseek_v4 = make_tools(["search"])
_tools_minimax = make_tools(["search"])
_tools_glm_4_7 = make_tools(["search"])

# Two distinct tools for tool_choice=required / forced instance tests.
_tools_llama_pair = make_tools(["t1", "t2"])
_tools_kimi_pair = make_tools(["t1", "t2"])
_tools_deepseek_pair = make_tools(["search", "alt"])
_tools_deepseek_v3_2_pair = make_tools(["search", "alt"])
_tools_deepseek_v4_pair = make_tools(["search", "alt"])
_tools_minimax_pair = make_tools(["search", "alt"])
_tools_qwen_3_coder_pair = make_tools(["run_sql", "run_py"])
_tools_qwen_3_pair = make_tools(["t1", "t2"])
_tools_qwen_3_5_pair = make_tools(["run_sql", "run_py"])
_tools_harmony_pair = make_tools(["comment_tool", "other_tool"])
_tools_glm_4_7_pair = make_tools(["search", "alt"])


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


def test_harmony_builtin_tool_instance():
    """Harmony builtin tools follow the browser sample shape from harmony examples."""

    structural_tag = get_model_structural_tag(
        "harmony",
        tools=[
            {
                "type": "web_search_preview",
                "name": "browser.search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
        reasoning=False,
    )

    check_stag_with_instance(
        structural_tag,
        '<|channel|>commentary to=browser.search code<|message|>{"query": "weather"}<|call|>',
        True,
    )
    check_stag_with_instance(
        structural_tag,
        '<|channel|>analysis to=browser.search<|message|>{"query": "weather"}<|call|>',
        False,
    )


@pytest.mark.parametrize(
    "structural_tag_fn",
    [
        get_llama_structural_tag,
        get_kimi_structural_tag,
        get_deepseek_r1_structural_tag,
        get_deepseek_v3_1_structural_tag,
        get_qwen_3_5_structural_tag,
        get_qwen_3_coder_structural_tag,
        get_qwen_3_structural_tag,
        get_harmony_structural_tag,
        get_deepseek_v3_2_structural_tag,
        get_deepseek_v4_structural_tag,
        get_minimax_structural_tag,
        get_glm_4_7_structural_tag,
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
            'text<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>t1\n```json\n{"q": "v"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
            True,
        ),
        (
            "deepseek_r1",
            'text<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>t2\n```json\n{"q": "v"}\n```<｜tool▁call▁end｜>',
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
            '\n</think>\n\n\n\n<minimax:tool_call>\n<invoke name="t1">\n<q>{"type": "string"}</q>\n</invoke>\n</minimax:tool_call>\n',
            True,
        ),
        (
            "minimax",
            '<minimax:tool_call>\n<invoke name="t2">\n<q>{"type": "string"}</q>\n</invoke>\n</minimax:tool_call>\n',
            False,
        ),
        (
            "qwen_3_coder",
            '<tool_call>\n<function=t1>\n<q>{"type": "string"}</q>\n</function>\n</tool_call>',
            True,
        ),
        (
            "qwen_3_5",
            '<tool_call>\n<function=t1>\n<q>{"type": "string"}</q>\n</function>\n</tool_call>',
            True,
        ),
        ("qwen_3", 'text<tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>', True),
        ("qwen_3", 'text<tool_call>\n{"name": "t2", "arguments": {"q": "v"}}\n</tool_call>', False),
        ("qwen_3", 'text<tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>', True),
        ("qwen_3", 'text<tool_call>\n{"name": "t2", "arguments": {"q": "v"}}\n</tool_call>', False),
        (
            "harmony",
            '<|channel|>commentary to=functions.t1<|constrain|>json<|message|>{"q": "v"}<|call|>',
            True,
        ),
        (
            "harmony",
            '<|channel|>commentary to=functions.t2<|constrain|>json<|message|>{"q": "v"}<|call|>',
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

# Case: (input_dict, instances, reasoning, expected_grammar_ebnf, expected_accept_per_instance)
# input_dict may include tool_choice ("auto" | "forced" | "required") and forced_function_name (str | None).
# When expected_grammar_ebnf is empty or whitespace-only, grammar equality is skipped (fill in EBNF when ready).
InstanceCase = Tuple[Dict[str, Any], List[str], bool, List[bool]]


def run_instance_case(format_type: str, case: InstanceCase):
    """Run one instance test case (accept/reject per instance string)."""
    (input_dict, instances, reasoning, expected_accept_per_instance) = case
    kwargs = _input_dict_to_get_stag_kwargs(format_type, input_dict)
    kwargs["reasoning"] = reasoning
    stag = get_model_structural_tag(**kwargs)
    for j, instance in enumerate(instances):
        check_stag_with_instance(stag, instance, expected_accept_per_instance[j])


# tool_choice=required / forced: expected_grammar_ebnf left "" for manual completion.
_tool_choice_instance_cases = [
    pytest.param(
        "llama",
        (
            {"tools": _tools_llama_pair, "tool_choice": "required"},
            ["", '{"name": "t1", "parameters": {"q": "v"}}'],
            False,
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
                '<|tool_calls_section_begin|><|tool_call_begin|>functions.t1:0<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|><|tool_calls_section_end|>',
            ],
            False,
            [False, True],
        ),
        id="kimi-required",
    ),
    pytest.param(
        "kimi",
        (
            {"tools": _tools_kimi_pair, "tool_choice": "forced", "forced_function_name": "t1"},
            [
                '<|tool_calls_section_begin|><|tool_call_begin|>functions.t1:0<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|><|tool_calls_section_end|>',
                '<|tool_calls_section_begin|><|tool_call_begin|>functions.t2:0<|tool_call_argument_begin|>{"q": "v"}<|tool_call_end|><|tool_calls_section_end|>',
            ],
            False,
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
                '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n```json\n{"q": "v"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
            ],
            False,
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
                '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n```json\n{"q": "v"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
                '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>alt\n```json\n{"q": "v"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
            ],
            False,
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
                '\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>',
            ],
            False,
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
                '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>',
            ],
            False,
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
                '\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>',
                '\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name="alt">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜function_calls>',
            ],
            False,
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
                '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="search">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>',
                '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="alt">\n<｜DSML｜parameter name="q" string="true">v</｜DSML｜parameter></｜DSML｜invoke>\n</｜DSML｜tool_calls>',
            ],
            False,
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
            [False, False],
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
            [False, False],
        ),
        id="minimax-forced",
    ),
    pytest.param(
        "qwen_3_coder",
        (
            {"tools": _tools_qwen_3_coder_pair, "tool_choice": "required"},
            [
                "",
                "<tool_call>\n<function=run_sql>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
            ],
            False,
            [False, True],
        ),
        id="qwen_coder-required",
    ),
    pytest.param(
        "qwen_3_coder",
        (
            {
                "tools": _tools_qwen_3_coder_pair,
                "tool_choice": "forced",
                "forced_function_name": "run_sql",
            },
            [
                "<tool_call>\n<function=run_sql>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
                "<tool_call>\n<function=run_py>\n<parameter=q>v</parameter>\n</function>\n</tool_call>",
            ],
            False,
            [True, False],
        ),
        id="qwen_coder-forced",
    ),
    pytest.param(
        "qwen_3",
        (
            {"tools": _tools_qwen_3_pair, "tool_choice": "required"},
            ["", '<tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>'],
            False,
            [False, True],
        ),
        id="qwen_3-required",
    ),
    pytest.param(
        "qwen_3",
        (
            {"tools": _tools_qwen_3_pair, "tool_choice": "forced", "forced_function_name": "t1"},
            [
                '<tool_call>\n{"name": "t1", "arguments": {"q": "v"}}\n</tool_call>',
                '<tool_call>\n{"name": "t2", "arguments": {"q": "v"}}\n</tool_call>',
            ],
            False,
            [True, False],
        ),
        id="qwen_3-forced",
    ),
    pytest.param(
        "harmony",
        (
            {"tools": _tools_harmony_pair, "tool_choice": "required"},
            [
                "plain text without channels",
                '<|channel|>commentary to=functions.comment_tool<|constrain|>json<|message|>{"q": "v"}<|call|>',
            ],
            False,
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
                '<|channel|>commentary to=functions.comment_tool<|constrain|>json<|message|>{"q": "v"}<|call|>',
                '<|channel|>commentary to=functions.other_tool<|constrain|>json<|message|>{"q": "v"}<|call|>',
            ],
            False,
            [True, False],
        ),
        id="harmony-forced",
    ),
    pytest.param(
        "glm_4_7",
        (
            {"tools": _tools_glm_4_7_pair, "tool_choice": "required"},
            ["", "<tool_call>search<arg_key>q</arg_key><arg_value>v</arg_value></tool_call>"],
            False,
            [False, True],
        ),
        id="glm_4_7-required",
    ),
    pytest.param(
        "glm_4_7",
        (
            {
                "tools": _tools_glm_4_7_pair,
                "tool_choice": "forced",
                "forced_function_name": "search",
            },
            [
                "<tool_call>search<arg_key>q</arg_key><arg_value>v</arg_value></tool_call>",
                "<tool_call>alt<arg_key>q</arg_key><arg_value>v</arg_value></tool_call>",
            ],
            False,
            [True, False],
        ),
        id="glm_4_7-forced",
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
        ("qwen_3_coder", {"tools": _TOOLS}),
        ("qwen_3", {"tools": _TOOLS}),
        ("deepseek_v3_2", {"tools": _TOOLS}),
        ("deepseek_v4", {"tools": _TOOLS}),
        ("minimax", {"tools": _TOOLS}),
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
