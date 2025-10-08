"""Tests for parsing structural tag outputs with parser tags."""

import importlib.util
from pathlib import Path

STRUCTURAL_TAG_PATH = Path(__file__).resolve().parents[2] / "python" / "xgrammar" / "structural_tag.py"
spec = importlib.util.spec_from_file_location("xgrammar.structural_tag", STRUCTURAL_TAG_PATH)
structural_tag = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(structural_tag)


def test_parse_triggered_tags_tool_calls():
    tool_call_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "object"},
        },
        "required": ["name", "arguments"],
    }

    tool_calls_format = structural_tag.TriggeredTagsFormat(
        triggers=["<tool_call>"],
        tags=[
            structural_tag.TagFormat(
                begin="<tool_call>",
                content=structural_tag.JSONSchemaFormat(
                    json_schema=tool_call_schema,
                    parser_tag={"capture_id": "item"},
                ),
                end="</tool_call>",
            )
        ],
        parser_tag={"capture_id": "tool_calls", "combine": "append"},
    )

    output = (
        "<tool_call>{\"name\": \"weather\", \"arguments\": {\"location\": \"sf\"}}</tool_call>"
        "<tool_call>{\"name\": \"time\", \"arguments\": {}}</tool_call>"
    )

    parsed = structural_tag.parse_structural_tag_output(tool_calls_format, output)

    assert parsed == {
        "tool_calls": [
            {"name": "weather", "arguments": {"location": "sf"}},
            {"name": "time", "arguments": {}},
        ]
    }


def test_parse_tags_with_separator_messages():
    message_schema = {
        "type": "object",
        "properties": {
            "role": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["role", "content"],
    }

    messages_format = structural_tag.TagsWithSeparatorFormat(
        tags=[
            structural_tag.TagFormat(
                begin="<message>",
                content=structural_tag.JSONSchemaFormat(
                    json_schema=message_schema,
                    parser_tag={"capture_id": "item"},
                ),
                end="</message>",
            )
        ],
        separator="\n",
        parser_tag={"capture_id": "messages", "combine": "append"},
    )

    output = (
        "<message>{\"role\": \"assistant\", \"content\": \"Hi there\"}</message>\n"
        "<message>{\"role\": \"tool\", \"content\": \"Done\"}</message>"
    )

    parsed = structural_tag.parse_structural_tag_output(messages_format, output)

    assert parsed == {
        "messages": [
            {"role": "assistant", "content": "Hi there"},
            {"role": "tool", "content": "Done"},
        ]
    }


def test_parse_concat_from_triggered_segments():
    concat_format = structural_tag.TriggeredTagsFormat(
        triggers=["<part>"],
        tags=[
            structural_tag.TagFormat(
                begin="<part>",
                content=structural_tag.AnyTextFormat(
                    parser_tag={"capture_id": "item"}
                ),
                end="</part>",
            )
        ],
        parser_tag={"capture_id": "content", "combine": "concat"},
    )

    output = "<part>Hello</part>\n<part>World</part>"

    parsed = structural_tag.parse_structural_tag_output(concat_format, output)

    assert parsed == {"content": "HelloWorld"}


def test_parse_nested_paths_from_tag():
    message_format = structural_tag.TagFormat(
        begin="<message>",
        content=structural_tag.AnyTextFormat(
            parser_tag={"capture_id": "content"}
        ),
        end="</message>",
        parser_tag={"capture_id": "message"},
    )

    parsed = structural_tag.parse_structural_tag_output(message_format, "<message>Hi</message>")

    assert parsed == {"message": {"content": "Hi"}}
