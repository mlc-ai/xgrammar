"""Tests for XML tool-calling objects that cannot contain any property.

The XML tool-calling formats compile such an object into an empty string rather
than ``whitespace*``. A whitespace-only body is a self-loop with no
non-whitespace successor, so a model whose prior favours indentation can keep
emitting whitespace until its token budget is exhausted instead of closing the
tool call.
"""

import sys

import pytest

import xgrammar as xgr
from xgrammar.structural_tag import JSONSchemaFormat, StructuralTag, TagFormat, TriggeredTagsFormat
from xgrammar.testing import _is_grammar_accept_string, _json_schema_to_ebnf

XML_FORMATS = ["qwen_xml", "minimax_xml", "deepseek_xml", "glm_xml"]

WHITESPACE_INSTANCES = [" ", "\t", "\n", "\t\t\t", " \n\t"]

# The schema OpenAI-style clients emit for a tool that takes no argument. Note
# the absence of ``additionalProperties``, which is the shape that surfaced the
# whitespace loop in practice.
NO_ARGUMENT_SCHEMA = {"type": "object", "properties": {}, "required": []}

CLOSED_NO_ARGUMENT_SCHEMA = {**NO_ARGUMENT_SCHEMA, "additionalProperties": False}

QWEN_TOOL_CALL_BEGIN = "<tool_call>\n<function=no_argument_tool>\n"
QWEN_TOOL_CALL_END = "\n</function>\n</tool_call>"


@pytest.mark.parametrize("json_format", XML_FORMATS)
@pytest.mark.parametrize(
    "schema",
    [NO_ARGUMENT_SCHEMA, CLOSED_NO_ARGUMENT_SCHEMA],
    ids=["implicit_additional_properties", "explicit_additional_properties"],
)
def test_no_argument_object_rejects_whitespace(json_format: str, schema: dict):
    grammar = _json_schema_to_ebnf(schema, json_format=json_format)

    assert _is_grammar_accept_string(grammar, "")
    for instance in WHITESPACE_INSTANCES:
        assert not _is_grammar_accept_string(grammar, instance)


@pytest.mark.parametrize(
    "extra",
    [
        {"title": "NoArgument", "description": "Takes nothing."},
        {"minProperties": 0},
        {"maxProperties": 0},
        {"maxProperties": 5},
        {"unevaluatedProperties": False},
    ],
    ids=["annotations", "min_properties", "max_properties_zero", "max_properties", "unevaluated"],
)
def test_no_argument_object_ignores_irrelevant_keywords(extra: dict):
    grammar = _json_schema_to_ebnf({**NO_ARGUMENT_SCHEMA, **extra}, json_format="qwen_xml")

    assert _is_grammar_accept_string(grammar, "")
    assert not _is_grammar_accept_string(grammar, "\t")


def test_no_argument_object_behind_ref():
    grammar = _json_schema_to_ebnf(
        {"$defs": {"NoArgument": NO_ARGUMENT_SCHEMA}, "$ref": "#/$defs/NoArgument"},
        json_format="qwen_xml",
    )

    assert _is_grammar_accept_string(grammar, "")
    assert not _is_grammar_accept_string(grammar, "\t")


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object"},
        {**NO_ARGUMENT_SCHEMA, "additionalProperties": True},
        {**NO_ARGUMENT_SCHEMA, "additionalProperties": {"type": "string"}},
        {**NO_ARGUMENT_SCHEMA, "unevaluatedProperties": True},
    ],
    ids=["bare_object", "additional_true", "additional_schema", "unevaluated_true"],
)
def test_object_allowing_properties_still_emits_parameters(schema: dict):
    """Objects that may still carry properties must keep their parameter branch.

    ``True`` / open schemas mean "arguments are unconstrained", not "there are
    no arguments", so they must not be collapsed to the empty string.
    """
    grammar = _json_schema_to_ebnf(schema, json_format="qwen_xml")

    assert _is_grammar_accept_string(grammar, "<parameter=city>Paris</parameter>")


@pytest.mark.parametrize("schema", [NO_ARGUMENT_SCHEMA, CLOSED_NO_ARGUMENT_SCHEMA])
def test_json_format_keeps_empty_braces(schema: dict):
    """The plain JSON format is unaffected; it still requires ``{}``."""
    grammar = _json_schema_to_ebnf(schema, json_format="json")

    assert _is_grammar_accept_string(grammar, "{}")
    assert _is_grammar_accept_string(grammar, "{ \n\t}")
    assert not _is_grammar_accept_string(grammar, "")


def test_nested_empty_object_keeps_json_braces():
    grammar = _json_schema_to_ebnf(
        {
            "type": "object",
            "properties": {"config": CLOSED_NO_ARGUMENT_SCHEMA},
            "required": ["config"],
            "additionalProperties": False,
        },
        json_format="qwen_xml",
    )

    assert _is_grammar_accept_string(grammar, "<parameter=config>{}</parameter>")
    assert _is_grammar_accept_string(grammar, "<parameter=config>{ \n\t}</parameter>")
    assert not _is_grammar_accept_string(grammar, "<parameter=config></parameter>")


def _qwen_tool_call_tag(schema: dict, name: str = "no_argument_tool") -> TagFormat:
    return TagFormat(
        begin=f"<tool_call>\n<function={name}>\n",
        content=JSONSchemaFormat(json_schema=schema, style="qwen_xml"),
        end=QWEN_TOOL_CALL_END,
    )


@pytest.mark.parametrize("schema", [NO_ARGUMENT_SCHEMA, CLOSED_NO_ARGUMENT_SCHEMA])
def test_structural_tag_closes_no_argument_tool_call(schema: dict):
    """End to end: the parameter zone must hand over to the tag's ``end``."""
    grammar = xgr.Grammar.from_structural_tag(StructuralTag(format=_qwen_tool_call_tag(schema)))

    assert _is_grammar_accept_string(grammar, QWEN_TOOL_CALL_BEGIN + QWEN_TOOL_CALL_END)
    for instance in WHITESPACE_INSTANCES:
        assert not _is_grammar_accept_string(
            grammar, QWEN_TOOL_CALL_BEGIN + instance + QWEN_TOOL_CALL_END
        )


def test_structural_tag_mixed_tools():
    """A no-argument tool must not disturb a sibling tool that takes arguments."""
    grammar = xgr.Grammar.from_structural_tag(
        StructuralTag(
            format=TriggeredTagsFormat(
                triggers=["<tool_call>\n<function="],
                tags=[
                    _qwen_tool_call_tag(NO_ARGUMENT_SCHEMA),
                    _qwen_tool_call_tag(
                        {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                        name="with_argument_tool",
                    ),
                ],
            )
        )
    )

    assert _is_grammar_accept_string(grammar, QWEN_TOOL_CALL_BEGIN + QWEN_TOOL_CALL_END)
    assert not _is_grammar_accept_string(grammar, QWEN_TOOL_CALL_BEGIN + "\t" + QWEN_TOOL_CALL_END)
    assert _is_grammar_accept_string(
        grammar,
        "<tool_call>\n<function=with_argument_tool>\n"
        "<parameter=city>Paris</parameter>" + QWEN_TOOL_CALL_END,
    )


def test_unsatisfiable_no_argument_object():
    """``minProperties >= 1`` with no allowed property stays a hard error."""
    with pytest.raises(RuntimeError):
        _json_schema_to_ebnf({**NO_ARGUMENT_SCHEMA, "minProperties": 1}, json_format="qwen_xml")


if __name__ == "__main__":
    pytest.main(sys.argv)
