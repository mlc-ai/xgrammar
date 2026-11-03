"""Preserve Unicode and punctuation in DeepSeek dynamic parameter names."""

import json

import pytest

import xgrammar as xgr
from xgrammar.structural_tag import JSONSchemaFormat, StructuralTag
from xgrammar.testing import _is_grammar_accept_string, _json_schema_to_ebnf

STYLES = [("deepseek_xml", ""), ("deepseek_v4_1_xml", " ")]
SAFE_NAMES = [
    "Temp °C",
    "café0",
    "温度",
    "😀",
    "x-y",
    "9input",
    " ",
    "a&b",
    "name\\path",
    "a<inside>b",
    "a｜DSML｜b",
    "e\u0301",
    "a\u2028b",
]
UNSAFE_NAMES = [
    "",
    'a"b',
    "a\x00b",
    "a\nb",
    "a\tb",
    "a\x1fb",
    "a\x7fb",
    "a<｜DSML｜ parameter>b",
    "a</｜DSML｜ parameter>b",
    "a</｜DSML｜ invoke>b",
    "a</｜DSML｜ calls>b",
    "a</｜DSML｜tool_calls>b",
    "a<｜end▁of▁sentence｜>b",
    "a<think>b",
    "a</think>b",
]


def parameter(name, spacing, value="1", string_attr="false"):
    tag = f"｜DSML｜{spacing}parameter"
    return f'<{tag} name="{name}" string="{string_attr}">{value}</{tag}>'


def schema_grammar(schema, style, **kwargs):
    return xgr.Grammar.from_structural_tag(
        StructuralTag(format=JSONSchemaFormat(json_schema=schema, style=style, **kwargs))
    )


@pytest.mark.parametrize("style,spacing", STYLES)
@pytest.mark.parametrize("name", SAFE_NAMES)
@pytest.mark.parametrize("schema", [True, {}, {"type": "object", "additionalProperties": True}])
def test_deepseek_dynamic_names_preserve_unicode_and_punctuation(style, spacing, name, schema):
    output = parameter(name, spacing)
    assert _is_grammar_accept_string(schema_grammar(schema, style), output)
    # The public EBNF round trip must preserve Unicode in the filtered automaton.
    ebnf = _json_schema_to_ebnf(json.dumps(schema), json_format=style)
    assert _is_grammar_accept_string(xgr.Grammar.from_ebnf(ebnf), output)


@pytest.mark.parametrize("style,spacing", STYLES)
@pytest.mark.parametrize("name", UNSAFE_NAMES)
def test_deepseek_dynamic_names_reject_attribute_and_protocol_delimiters(style, spacing, name):
    assert not _is_grammar_accept_string(schema_grammar(True, style), parameter(name, spacing))


@pytest.mark.parametrize("style,spacing", STYLES)
@pytest.mark.parametrize(
    "choice", ["required", "auto", {"type": "function", "function": {"name": "record"}}]
)
def test_deepseek_explicit_loose_tool_choice_preserves_original_names(style, spacing, choice):
    tool = {
        "type": "function",
        "function": {
            "name": "record",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {"Temp °C": {"type": "number"}},
                "required": ["Temp °C"],
                "additionalProperties": False,
            },
        },
    }
    calls = "calls" if spacing else "tool_calls"
    model = "deepseek_v4_1" if spacing else "deepseek_v4"
    tag = xgr.get_model_structural_tag(
        model, tools=[tool], tool_choice=choice, reasoning=False, parallel_tool_calls=False
    )
    grammar = xgr.Grammar.from_structural_tag(tag)

    def call(body):
        return (
            f'\n\n<｜DSML｜{spacing}{calls}>\n<｜DSML｜{spacing}invoke name="record">\n'
            + body
            + f"\n</｜DSML｜{spacing}invoke>\n</｜DSML｜{spacing}{calls}>"
        )

    assert _is_grammar_accept_string(grammar, call(parameter("Temp °C", spacing)))
    # Strict=false must continue to allow undeclared keys and different value types.
    assert _is_grammar_accept_string(
        grammar, call(parameter("未声明 extra", spacing, "free text", "true"))
    )


@pytest.mark.parametrize("style,spacing", STYLES)
def test_deepseek_additional_names_keep_declared_and_nested_constraints(style, spacing):
    schema = {
        "type": "object",
        "properties": {"Temp °C": {"type": "integer"}},
        "required": ["Temp °C"],
        "additionalProperties": {"type": "object"},
    }
    grammar = schema_grammar(schema, style)
    declared = parameter("Temp °C", spacing)
    nested = '{"温度": 2, "quote\\"key": "ok", "nested": {"😀": true}}'
    assert _is_grammar_accept_string(grammar, declared + parameter("其他 extra", spacing, nested))
    assert not _is_grammar_accept_string(grammar, declared + parameter("其他 extra", spacing, "3"))
    fixed = schema_grammar({**schema, "additionalProperties": False}, style)
    assert _is_grammar_accept_string(fixed, declared)
    assert not _is_grammar_accept_string(fixed, declared + parameter("extra", spacing))


@pytest.mark.parametrize("style,spacing", STYLES)
def test_deepseek_name_exclusions_do_not_spread_to_values(style, spacing):
    grammar = schema_grammar(True, style, excludes=["禁用", "blocked"])
    assert _is_grammar_accept_string(grammar, parameter("允许 field", spacing))
    for name in ["a禁用b", "ablockedb"]:
        assert not _is_grammar_accept_string(grammar, parameter(name, spacing))
    for value in ["a禁用b", "ablockedb"]:
        assert not _is_grammar_accept_string(grammar, parameter("allowed", spacing, value, "true"))
    for value in ["<think>", "</think>", "<｜end▁of▁sentence｜>", "</｜DSML｜ invoke>"]:
        assert _is_grammar_accept_string(grammar, parameter("allowed", spacing, value, "true"))


@pytest.mark.parametrize(
    "style,output,unicode_allowed",
    [
        ("qwen_xml", "<parameter={}>1</parameter>", False),
        ("minimax_xml", '<parameter name="{}">1</parameter>', False),
        ("glm_xml", "<arg_key>{}</arg_key><arg_value>1</arg_value>", False),
        ("cohere_xml", '<cofl:value name="{}" type="json">1</cofl:value>', True),
        (
            "kimi_k3_xml",
            '<|open|>argument key="{}" type="number"<|sep|>1<|close|>argument<|sep|>',
            False,
        ),
    ],
)
def test_other_xml_dialects_keep_dynamic_name_contract(style, output, unicode_allowed):
    grammar = schema_grammar(True, style)
    assert _is_grammar_accept_string(grammar, output.format("ascii_key"))
    assert _is_grammar_accept_string(grammar, output.format("Temp °C")) == unicode_allowed


@pytest.mark.parametrize("style,spacing", STYLES)
def test_deepseek_unicode_token_masks_and_rollback(style, spacing):
    prefix = f'<｜DSML｜{spacing}parameter name="'
    suffix = f'" string="false">1</｜DSML｜{spacing}parameter>'
    vocab = [
        "<eos>",
        prefix,
        "温",
        "度",
        "😀",
        "Temp °C",
        "<",
        "</",
        "｜",
        "inside",
        ">",
        suffix,
        "<think>",
        "\n",
    ]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=0)
    compiled = xgr.GrammarCompiler(info, max_threads=1).compile_grammar(schema_grammar(True, style))
    matcher = xgr.GrammarMatcher(compiled)
    mask = xgr.allocate_token_bitmask(1, len(vocab))

    def allows(token):
        matcher.fill_next_token_bitmask(mask)
        token_id = vocab.index(token)
        return bool((int(mask[0, token_id // 32]) >> (token_id % 32)) & 1)

    def accept(token):
        assert allows(token), token
        assert matcher.accept_token(vocab.index(token)), token

    accept(prefix)
    assert not allows(suffix)  # Empty names cannot close.
    assert not allows("<think>")
    assert not allows("\n")
    accept("温")
    forked = matcher.fork()
    accept("度")
    accept("<")
    assert not allows("｜")  # A marker split across tokens is still forbidden.
    matcher.rollback(1)
    assert allows("｜")
    accept("😀")
    accept(suffix)
    assert matcher.is_completed()
    assert forked.accept_token(vocab.index("Temp °C"))
    assert forked.accept_token(vocab.index(suffix))
    assert forked.is_completed()
