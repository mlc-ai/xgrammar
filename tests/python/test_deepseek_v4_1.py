"""DeepSeek-V4.1 output constraints, including the spaces in its DSML tag names."""

import copy
import json

import pytest

import xgrammar as xgr
from xgrammar.builtin_structural_tag import get_model_structural_tag
from xgrammar.structural_tag import JSONSchemaFormat, StructuralTag
from xgrammar.testing import _is_grammar_accept_string, _json_schema_to_ebnf

SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}},
    "required": ["query", "limit"],
    "additionalProperties": False,
}
TOOLS = [
    {"type": "function", "function": {"name": name, "parameters": SCHEMA}}
    for name in ("search", "other")
]
CALL = (
    '<｜DSML｜ invoke name="search">\n'
    '<｜DSML｜ parameter name="query" string="true">北京\n<code>"hi"</code></｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="limit" string="false">2</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n"
)
CALLS = "\n\n<｜DSML｜ calls>\n" + CALL + "</｜DSML｜ calls>"


def make_grammar(reasoning=False, choice="required", **kwargs):
    return xgr.Grammar.from_structural_tag(
        get_model_structural_tag(
            "deepseek_v4_1", tools=TOOLS, reasoning=reasoning, tool_choice=choice, **kwargs
        )
    )


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("policy", ["auto", "required", "forced", "none", "allowed"])
def test_tool_choice(reasoning, policy):
    choice = policy
    if policy == "forced":
        choice = {"type": "function", "function": {"name": "search"}}
    elif policy == "allowed":
        choice = {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "required",
                "tools": [{"type": "function", "function": {"name": "search"}}],
            },
        }
    grammar = make_grammar(reasoning, choice)
    prefix = "Plan the search.</think>" if reasoning else ""
    assert _is_grammar_accept_string(grammar, prefix + CALLS) == (policy != "none")
    assert _is_grammar_accept_string(grammar, prefix + "Hello") == (policy in ["auto", "none"])
    parallel = CALLS.replace("</｜DSML｜ calls>", CALL + "</｜DSML｜ calls>")
    assert _is_grammar_accept_string(grammar, prefix + parallel) == (
        policy in ["auto", "required", "allowed"]
    )
    other = CALLS.replace('name="search"', 'name="other"')
    assert _is_grammar_accept_string(grammar, prefix + other) == (policy in ["auto", "required"])


@pytest.mark.parametrize("policy", ["auto", "required"])
@pytest.mark.parametrize(
    "output",
    [
        CALLS.replace('name="search"', 'name="unknown"'),
        CALLS.replace('name="query"', 'name="unknown"'),
        CALLS.replace('string="false">2', 'string="false">0'),
        CALLS.replace('string="false">2', 'string="false">"two"'),
        CALLS.replace(
            '<｜DSML｜ parameter name="limit" string="false">2</｜DSML｜ parameter>\n', ""
        ),
        CALLS.replace("</｜DSML｜ invoke>\n", "</｜DSML｜ invoke>\n\n"),
        CALLS.replace("｜DSML｜ parameter", "｜DSML｜parameter"),
        CALLS.replace("｜DSML｜ invoke", "｜DSML｜invoke"),
        "\n\n<｜DSML｜ calls>\n</｜DSML｜ calls>",
        CALLS[: -len("</｜DSML｜ calls>")],
    ],
)
def test_invalid_calls(policy, output):
    assert not _is_grammar_accept_string(make_grammar(choice=policy), output)


def test_reasoning_and_prompt_boundary():
    grammar = make_grammar(reasoning=True)
    assert _is_grammar_accept_string(grammar, "</think>" + CALLS)
    assert not _is_grammar_accept_string(grammar, CALLS)
    assert not _is_grammar_accept_string(grammar, "thinking</think>")
    assert not _is_grammar_accept_string(make_grammar(), "</think>" + CALLS)


def test_v4_is_a_different_wire_format():
    legacy = xgr.Grammar.from_structural_tag(
        get_model_structural_tag(
            "deepseek_v4", tools=TOOLS, reasoning=False, tool_choice="required"
        )
    )
    legacy_output = CALLS.replace("｜DSML｜ calls", "｜DSML｜tool_calls").replace(
        "｜DSML｜ ", "｜DSML｜"
    )
    assert _is_grammar_accept_string(legacy, legacy_output)
    assert not _is_grammar_accept_string(legacy, CALLS)
    assert not _is_grammar_accept_string(make_grammar(), legacy_output)


@pytest.mark.parametrize("exclude_special_tokens", [False, True])
def test_no_tools(exclude_special_tokens):
    grammar = xgr.Grammar.from_structural_tag(
        get_model_structural_tag(
            "deepseek_v4_1", reasoning=False, exclude_special_tokens=exclude_special_tokens
        )
    )
    assert _is_grammar_accept_string(grammar, "Hello")
    assert not _is_grammar_accept_string(grammar, CALLS)


def test_namespaced_tool():
    tool = copy.deepcopy(TOOLS[0])
    tool["function"]["name"] = "web::search"
    grammar = xgr.Grammar.from_structural_tag(
        get_model_structural_tag(
            "deepseek_v4_1",
            tools=[tool],
            reasoning=False,
            tool_choice={"type": "function", "function": {"name": "web::search"}},
        )
    )
    assert _is_grammar_accept_string(grammar, CALLS.replace('name="search"', 'name="web::search"'))
    assert not _is_grammar_accept_string(grammar, CALLS)


@pytest.mark.parametrize("any_order", [False, True])
def test_parameter_order(any_order):
    lines = CALLS.splitlines(keepends=True)
    # query's raw string spans two lines; reverse the two complete parameters.
    output = "".join(lines[:4] + lines[6:7] + lines[4:6] + lines[7:])
    assert output != CALLS
    assert _is_grammar_accept_string(make_grammar(any_order=any_order), output) == any_order


@pytest.mark.parametrize(
    "schema,value,string_attr",
    [
        ({"type": "string"}, 'raw "quotes" & <tag>\n你好', "true"),
        ({"type": "integer"}, "42", "false"),
        ({"type": "number"}, "-1.25", "false"),
        ({"type": "boolean"}, "true", "false"),
        ({"type": "null"}, "null", "false"),
        ({"type": "array", "items": {"type": "integer"}}, "[1, 2]", "false"),
        (
            {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            '{"x": "hi"}',
            "false",
        ),
        ({"const": "fixed"}, "fixed", "true"),
        ({"enum": [1, 2]}, "2", "false"),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "null", "false"),
    ],
)
def test_parameter_style(schema, value, string_attr):
    schema = {
        "type": "object",
        "properties": {"value": schema},
        "required": ["value"],
        "additionalProperties": False,
    }
    output = f'<｜DSML｜ parameter name="value" string="{string_attr}">{value}</｜DSML｜ parameter>'
    stag = StructuralTag(format=JSONSchemaFormat(json_schema=schema, style="deepseek_v4_1_xml"))
    # Exercise both the production structural-tag converter and the EBNF conversion path.
    for grammar in [
        xgr.Grammar.from_structural_tag(stag),
        xgr.Grammar.from_ebnf(_json_schema_to_ebnf(schema, json_format="deepseek_v4_1_xml")),
    ]:
        assert _is_grammar_accept_string(grammar, output)
        assert not _is_grammar_accept_string(
            grammar, output.replace("｜DSML｜ parameter", "｜DSML｜parameter")
        )
        assert not _is_grammar_accept_string(grammar, output + output)


@pytest.mark.parametrize("parameters", [{}, {"type": "object", "properties": {}}, None])
def test_empty_arguments(parameters):
    tool = {"type": "function", "function": {"name": "ping", "parameters": parameters}}
    grammar = xgr.Grammar.from_structural_tag(
        get_model_structural_tag(
            "deepseek_v4_1", tools=[tool], reasoning=False, tool_choice="required"
        )
    )
    output = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="ping">\n\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    assert _is_grammar_accept_string(grammar, output)


@pytest.mark.parametrize(
    "function",
    [
        {"name": "search"},
        {"name": "search", "parameters": None},
        {"name": "search", "parameters": {}},
        {"name": "search", "parameters": SCHEMA, "strict": False},
    ],
)
def test_unconstrained_tool_parameters(function):
    grammar = xgr.Grammar.from_structural_tag(
        get_model_structural_tag(
            "deepseek_v4_1",
            tools=[{"type": "function", "function": function}],
            reasoning=False,
            tool_choice="required",
        )
    )
    assert _is_grammar_accept_string(grammar, CALLS)


def test_whitespace_limit_and_serialization():
    stag = get_model_structural_tag(
        "deepseek_v4_1", tools=TOOLS, tool_choice="required", reasoning=False, max_whitespace_cnt=2
    )
    grammar = xgr.Grammar.from_structural_tag(json.loads(stag.model_dump_json()))
    assert _is_grammar_accept_string(grammar, CALLS)
    assert not _is_grammar_accept_string(
        grammar, CALLS.replace('string="false">2', 'string="false">   2')
    )


@pytest.mark.hf_token_required
@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("policy", ["auto", "required", "forced"])
def test_official_tokenizer_masks(reasoning, policy):
    from test_builtin_structural_tag_alignment import extract_output_encoder
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V4.1-Flash", revision="dba1be0a40aa45a94ad051997016db3960a90277"
    )
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=129280)
    assert info.vocab_type == xgr.VocabType.BYTE_LEVEL
    assert info.stop_token_ids == [1]
    assert tokenizer.encode("｜DSML｜", add_special_tokens=False) == [128825]
    tool_choice = (
        {"type": "function", "function": {"name": "search"}} if policy == "forced" else policy
    )
    tools = copy.deepcopy(TOOLS)
    stag = get_model_structural_tag(
        "deepseek_v4_1", tools=tools, reasoning=reasoning, tool_choice=tool_choice
    )
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(stag)
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = xgr.allocate_token_bitmask(1, info.vocab_size)
    message = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Plan." if reasoning else "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": name, "arguments": {"query": "北京\n<code>", "limit": 2}},
            }
            for name in (["search"] if policy == "forced" else ["search", "other"])
        ],
    }
    output = extract_output_encoder(
        "dsv41",
        "deepseek_v4_1",
        message,
        tools,
        {"thinking_mode": "thinking" if reasoning else "chat"},
    )
    token_ids = tokenizer.encode(output, add_special_tokens=False)
    assert tokenizer.decode(token_ids) == output
    for index, token_id in enumerate(token_ids + [tokenizer.eos_token_id]):
        matcher.fill_next_token_bitmask(bitmask)
        if policy != "auto" and index < len(token_ids):
            assert not (int(bitmask[0, 0]) >> tokenizer.eos_token_id) & 1
        assert (int(bitmask[0, token_id // 32]) >> (token_id % 32)) & 1, token_id
        assert matcher.accept_token(token_id), token_id
    assert matcher.is_terminated()
