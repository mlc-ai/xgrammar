"""Defines all structural tag formats."""

import json
import re
from dataclasses import dataclass
from json import JSONDecoder
from typing import Any, Dict, List, Literal, Optional, Type, Union

try:
    # Python 3.9+
    from typing import Annotated
except ImportError:
    # Python 3.8
    from typing_extensions import Annotated

from pydantic import BaseModel, Field

class ParserTag(BaseModel):
    """Metadata for structure tags to allow output parsing."""

    capture_id: Optional[str] = Field(
        default=None,
        description=(
            "Identifier that the parser can use to collect this node's content or serve as a prefix for"
            " descendants."
        ),
    )
    combine: Optional[str] = Field(
        default=None,
        description=(
            "Instruction for the parser on how to combine multiple nodes with the same capture_id. Options are"
            " 'append' or 'concat'. 'append' will create a list of values, while 'concat' will concatenate"
            " them into a single string. When 'append' is used the output will always be a list, even if there is"
            " only one item. 'concat' will always return a string, and can only be used for string values."
        ),
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Catch-all to allow us to sneak in extra data later if we need it."
        ),
    )

# ---------- Basic Formats ----------


class ConstStringFormat(BaseModel):
    """A format that matches a constant string."""

    type: Literal["const_string"] = "const_string"
    """The type of the format."""
    value: str
    """The constant string."""


class JSONSchemaFormat(BaseModel):
    """A format that matches a JSON schema."""

    type: Literal["json_schema"] = "json_schema"
    """The type of the format."""
    json_schema: Union[bool, Dict[str, Any]]
    """The JSON schema."""
    parser_tag: Optional[ParserTag] = None
    """Optional information for output parsing."""


class QwenXMLParameterFormat(BaseModel):
    """A format that matches Qwen XML function calls.

    Examples
    --------
    .. code-block:: python

        structural_tag = QwenXMLParameterFormat(
            json_schema={
                "type": "qwen_xml_parameter",
                "json_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    "required": ["name", "age"],
                },
            }
        )

    The above structural tag can accept the following outputs::

        <parameter=name>Bob</parameter><parameter=age>100</parameter>
        <parameter=name>"Bob&lt;"</parameter><parameter=age>100</parameter>

    """

    type: Literal["qwen_xml_parameter"] = "qwen_xml_parameter"
    """The type of the format."""

    json_schema: Union[bool, Dict[str, Any]]
    """The JSON schema for the parameters of the function calling."""

    parser_tag: Optional[ParserTag] = None
    """Optional information for output parsing."""


class AnyTextFormat(BaseModel):
    """A format that matches any text."""

    type: Literal["any_text"] = "any_text"
    """The type of the format."""

    parser_tag: Optional[ParserTag] = None
    """Optional information for output parsing."""


class GrammarFormat(BaseModel):
    """A format that matches an ebnf grammar."""

    type: Literal["grammar"] = "grammar"
    """The type of the format."""

    grammar: str
    """The ebnf grammar."""


class RegexFormat(BaseModel):
    """A format that matches a regex pattern."""

    type: Literal["regex"] = "regex"
    """The type of the format."""

    pattern: str
    """The regex pattern."""

    parser_tag: Optional[ParserTag] = None
    """Optional information for output parsing."""


# ---------- Combinatorial Formats ----------


class SequenceFormat(BaseModel):
    """A format that matches a sequence of formats."""

    type: Literal["sequence"] = "sequence"
    """The type of the format."""
    elements: List["Format"]
    """The elements of the sequence."""


class OrFormat(BaseModel):
    """A format that matches one of the formats."""

    type: Literal["or"] = "or"
    """The type of the format."""
    elements: List["Format"]
    """The elements of the or."""


class TagFormat(BaseModel):
    """A format that matches a tag: ``begin content end``."""

    type: Literal["tag"] = "tag"
    """The type of the format."""
    begin: str
    """The begin tag."""
    content: "Format"
    """The content of the tag. It can be any of the formats."""
    end: str
    """The end tag."""
    parser_tag: Optional[ParserTag] = None
    """Optional information for output parsing."""


class TriggeredTagsFormat(BaseModel):
    """A format that matches triggered tags. It can allow any output until a trigger is
    encountered, then dispatch to the corresponding tag; when the end tag is encountered, the
    grammar will allow any following output, until the next trigger is encountered.

    Each tag should be matched by exactly one trigger. "matching" means the trigger should be a
    prefix of the begin tag.

    Examples
    --------

    .. code-block:: python

        structural_tag = TriggeredTagsFormat(
            triggers=["<function="],
            tags=[
                TagFormat(
                    begin="<function=func1>",
                    content=JSONSchemaFormat(json_schema=...),
                    end="</function>",
                ),
                TagFormat(
                    begin="<function=func2>",
                    content=JSONSchemaFormat(json_schema=...),
                    end="</function>",
                ),
            ],
            at_least_one=False,
            stop_after_first=False,
        )

    The above structural tag can accept the following outputs::

        <function=func1>{"name": "John", "age": 30}</function>
        <function=func2>{"name": "Jane", "age": 25}</function>
        any_text<function=func1>{"name": "John", "age": 30}</function>any_text1<function=func2>{"name": "Jane", "age": 25}</function>any_text2

    """

    type: Literal["triggered_tags"] = "triggered_tags"
    """The type of the format."""
    triggers: List[str]
    """The triggers of the triggered tags."""
    tags: List[TagFormat]
    """The tags of the triggered tags."""
    at_least_one: bool = False
    """Whether at least one of the tags must be generated."""
    stop_after_first: bool = False
    """Whether to stop after the first tag is generated."""
    parser_tag: Optional[ParserTag] = None
    """Optional information for output parsing."""


class TagsWithSeparatorFormat(BaseModel):
    """A format that matches a tags with separator. It can match zero, one, or more tags, separated
    by the separator, with no other text allowed.

    Examples
    --------

    .. code-block:: python

        structural_tag = TagsWithSeparatorFormat(
            tags=[
                TagFormat(begin="<function=func1>", content=JSONSchemaFormat(json_schema=...), end="</function>"),
                TagFormat(begin="<function=func2>", content=JSONSchemaFormat(json_schema=...), end="</function>"),
            ],
            separator=",",
            at_least_one=False,
            stop_after_first=False,
        )

    The above structural tag can accept an empty string, or the following outputs::

        <function=func1>{"name": "John", "age": 30}</function>
        <function=func1>{"name": "John", "age": 30}</function>,<function=func2>{"name": "Jane", "age": 25}</function>
        <function=func1>{"name": "John", "age": 30}</function>,<function=func2>{"name": "Jane", "age": 25}</function>,<function=func1>{"name": "John", "age": 30}</function>
    """

    type: Literal["tags_with_separator"] = "tags_with_separator"
    """The type of the format."""
    tags: List[TagFormat]
    """The tags of the tags with separator."""
    separator: str
    """The separator of the tags with separator."""
    at_least_one: bool = False
    """Whether at least one of the tags must be matched."""
    stop_after_first: bool = False
    """Whether to stop after the first tag is matched."""
    parser_tag: Optional[ParserTag] = None
    """Optional information for output parsing."""


# ---------- Discriminated Union ----------


Format = Annotated[
    Union[
        AnyTextFormat,
        ConstStringFormat,
        JSONSchemaFormat,
        GrammarFormat,
        RegexFormat,
        QwenXMLParameterFormat,
        OrFormat,
        SequenceFormat,
        TagFormat,
        TriggeredTagsFormat,
        TagsWithSeparatorFormat,
    ],
    Field(discriminator="type"),
]
"""Union of all structural tag formats."""


# Solve forward references
if hasattr(BaseModel, "model_rebuild"):
    SequenceFormat.model_rebuild()
    TagFormat.model_rebuild()
    TriggeredTagsFormat.model_rebuild()
    TagsWithSeparatorFormat.model_rebuild()
elif hasattr(BaseModel, "update_forward_refs"):
    # This is for backward compatibility with pydantic v1
    SequenceFormat.update_forward_refs()
    TagFormat.update_forward_refs()
    TriggeredTagsFormat.update_forward_refs()
    TagsWithSeparatorFormat.update_forward_refs()
else:
    raise RuntimeError("Unsupported pydantic version")


# ---------- Top Level ----------


class StructuralTagItem(BaseModel):
    """Deprecated. Definition of a structural tag item.

    See :meth:`xgrammar.Grammar.from_structural_tag` for more details.
    """

    begin: str
    """The begin tag."""
    schema_: Union[str, Type[BaseModel], Dict[str, Any]] = Field(alias="schema")
    """The schema."""
    end: str
    """The end tag."""


class StructuralTag(BaseModel):
    """
    Describes a complete structural tag structure. It corresponds to
    ``"response_format": {"type": "structural_tag", "format": {...}}`` in API.
    """

    type: Literal["structural_tag"] = "structural_tag"
    """The type must be "structural_tag"."""
    format: Format
    """The format of the structural tag. Could be any of the structural tag formats."""

    @staticmethod
    def from_legacy_structural_tag(
        tags: List[StructuralTagItem], triggers: List[str]
    ) -> "StructuralTag":
        """Convert a legacy structural tag item to a structural tag."""
        return StructuralTag(
            type="structural_tag",
            format=TriggeredTagsFormat(
                type="triggered_tags",
                triggers=triggers,
                tags=[
                    TagFormat(
                        begin=tag.begin,
                        content=JSONSchemaFormat(
                            json_schema=(
                                json.loads(tag.schema_)
                                if isinstance(tag.schema_, str)
                                else (
                                    tag.schema_.model_json_schema()
                                    if isinstance(tag.schema_, type)
                                    and issubclass(tag.schema_, BaseModel)
                                    else tag.schema_
                                )
                            )
                        ),
                        end=tag.end,
                    )
                    for tag in tags
                ],
            ),
        )

    @staticmethod
    def from_json(json_str: Union[str, Dict[str, Any]]) -> "StructuralTag":
        """Convert a JSON string to a structural tag."""
        if isinstance(json_str, str):
            return StructuralTag.model_validate_json(json_str)
        elif isinstance(json_str, dict):
            return StructuralTag.model_validate(json_str)
        else:
            raise ValueError("Invalid JSON string or dictionary")


_JSON_DECODER = JSONDecoder()


@dataclass
class _ParseNode:
    format: Format
    payload: Any = None
    children: List["_ParseNode"] = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []


def _parse_format(format_obj: Format, text: str) -> _ParseNode:
    if isinstance(format_obj, ConstStringFormat):
        if text != format_obj.value:
            raise ValueError("Const string does not match expected value")
        return _ParseNode(format=format_obj, payload=format_obj.value)
    if isinstance(format_obj, JSONSchemaFormat):
        try:
            value, end_index = _JSON_DECODER.raw_decode(text)
        except ValueError as exc:  # pragma: no cover - passthrough for context
            raise ValueError("Failed to parse JSON content") from exc
        remainder = text[end_index:]
        if remainder.strip():
            raise ValueError("Unexpected trailing content after JSON value")
        return _ParseNode(format=format_obj, payload=value)
    if isinstance(format_obj, QwenXMLParameterFormat):
        raise NotImplementedError("Parsing Qwen XML parameter format is not supported yet")
    if isinstance(format_obj, AnyTextFormat):
        return _ParseNode(format=format_obj, payload=text)
    if isinstance(format_obj, GrammarFormat):
        raise NotImplementedError("Parsing generic grammar formats is not supported")
    if isinstance(format_obj, RegexFormat):
        if not re.fullmatch(format_obj.pattern, text, flags=re.DOTALL):
            raise ValueError("Text does not match required regular expression")
        return _ParseNode(format=format_obj, payload=text)
    if isinstance(format_obj, SequenceFormat):
        raise NotImplementedError("Parsing sequence formats is not implemented")
    if isinstance(format_obj, OrFormat):
        raise NotImplementedError("Parsing or formats is not implemented")
    if isinstance(format_obj, TagFormat):
        return _parse_tag(format_obj, text)
    if isinstance(format_obj, TriggeredTagsFormat):
        return _parse_triggered_tags(format_obj, text)
    if isinstance(format_obj, TagsWithSeparatorFormat):
        return _parse_tags_with_separator(format_obj, text)
    raise TypeError(f"Unsupported structural tag format: {type(format_obj)!r}")


def _parse_tag(format_obj: TagFormat, text: str) -> _ParseNode:
    if not text.startswith(format_obj.begin):
        raise ValueError("Tag begin delimiter not found where expected")
    if format_obj.end == "":
        raise ValueError("Parsing tags without explicit end delimiters is not supported")
    if not text.endswith(format_obj.end):
        raise ValueError("Tag end delimiter not found where expected")
    inner_text = text[len(format_obj.begin) : len(text) - len(format_obj.end)]
    child_node = _parse_format(format_obj.content, inner_text)
    return _ParseNode(format=format_obj, children=[child_node])


def _parse_triggered_tags(format_obj: TriggeredTagsFormat, text: str) -> _ParseNode:
    idx = 0
    length = len(text)
    children: List[_ParseNode] = []
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        matched_tag = None
        for tag in format_obj.tags:
            if text.startswith(tag.begin, idx):
                matched_tag = tag
                break
        if matched_tag is None:
            if text[idx:].strip():
                raise ValueError("Unexpected content encountered inside triggered tags")
            break
        end_idx = text.find(matched_tag.end, idx + len(matched_tag.begin))
        if end_idx == -1:
            raise ValueError("Failed to locate end delimiter for triggered tag")
        tag_text = text[idx : end_idx + len(matched_tag.end)]
        child_node = _parse_tag(matched_tag, tag_text)
        children.append(child_node)
        idx = end_idx + len(matched_tag.end)
        if format_obj.stop_after_first:
            break
    if format_obj.at_least_one and not children:
        raise ValueError("Expected at least one triggered tag but found none")
    if text[idx:].strip():
        raise ValueError("Unexpected trailing content after triggered tags")
    return _ParseNode(format=format_obj, children=children)


def _parse_tags_with_separator(format_obj: TagsWithSeparatorFormat, text: str) -> _ParseNode:
    if text == "":
        if format_obj.at_least_one:
            raise ValueError("Expected at least one tag before separator")
        return _ParseNode(format=format_obj)
    idx = 0
    length = len(text)
    children: List[_ParseNode] = []
    while idx < length:
        matched_tag = None
        for tag in format_obj.tags:
            if text.startswith(tag.begin, idx):
                matched_tag = tag
                break
        if matched_tag is None:
            raise ValueError("Unable to match tag at current position when parsing separated tags")
        end_idx = text.find(matched_tag.end, idx + len(matched_tag.begin))
        if end_idx == -1:
            raise ValueError("Failed to locate end delimiter for separated tag")
        tag_text = text[idx : end_idx + len(matched_tag.end)]
        child_node = _parse_tag(matched_tag, tag_text)
        children.append(child_node)
        idx = end_idx + len(matched_tag.end)
        if idx == length:
            break
        if not text.startswith(format_obj.separator, idx):
            raise ValueError("Expected separator between tags")
        idx += len(format_obj.separator)
    return _ParseNode(format=format_obj, children=children)


def _extract_item(value: Dict[str, Any]) -> Any:
    if isinstance(value, dict) and set(value.keys()) == {"item"}:
        return value["item"]
    return value


def _merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    if not update:
        return base
    result = dict(base)
    for key, value in update.items():
        if key not in result:
            result[key] = value
            continue
        existing = result[key]
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _merge_dicts(existing, value)
        elif isinstance(existing, list) and isinstance(value, list):
            result[key] = existing + value
        elif existing == value:
            continue
        else:
            raise ValueError(f"Conflicting captures for field '{key}'")
    return result


def _wrap_capture(parser_tag: ParserTag, value: Any) -> Dict[str, Any]:
    capture_id = parser_tag.capture_id
    if capture_id is None:
        return {}
    combine = parser_tag.combine
    if combine is None:
        prepared_value = value
    elif combine == "append":
        if value is None:
            prepared_value = []
        elif isinstance(value, list):
            prepared_value = value
        else:
            prepared_value = [value]
    elif combine == "concat":
        if value is None:
            prepared_value = ""
        elif isinstance(value, list):
            prepared_value = "".join(str(item) for item in value)
        else:
            prepared_value = str(value)
    else:
        raise ValueError(f"Unsupported combine strategy: {combine}")

    if capture_id == "":
        if isinstance(prepared_value, dict):
            return prepared_value
        raise ValueError("Empty capture_id requires a dictionary value")

    segments = capture_id.split(".")
    nested: Any = prepared_value
    for segment in reversed(segments):
        nested = {segment: nested}
    return nested


def _aggregate_parse_tree(node: _ParseNode) -> Dict[str, Any]:
    fmt = node.format
    child_results = [_aggregate_parse_tree(child) for child in node.children]

    if isinstance(fmt, (TriggeredTagsFormat, TagsWithSeparatorFormat)):
        values = [_extract_item(result) for result in child_results if result]
        node_value: Any = values
        aggregated_children: Dict[str, Any] = {}
    elif isinstance(fmt, TagFormat):
        aggregated_children = {}
        for result in child_results:
            aggregated_children = _merge_dicts(aggregated_children, result)
        node_value = aggregated_children
    elif isinstance(fmt, (JSONSchemaFormat, AnyTextFormat, RegexFormat, ConstStringFormat)):
        aggregated_children = {}
        node_value = node.payload
    else:
        aggregated_children = {}
        for result in child_results:
            aggregated_children = _merge_dicts(aggregated_children, result)
        node_value = aggregated_children if aggregated_children else node.payload

    parser_tag = getattr(fmt, "parser_tag", None)
    if parser_tag:
        captured = _wrap_capture(parser_tag, node_value)
        if parser_tag.capture_id:
            return captured
    return aggregated_children


def parse_structural_tag_output(
    structural_tag: Union[StructuralTag, BaseModel, Dict[str, Any], str], text: str
) -> Dict[str, Any]:
    """Parse ``text`` according to ``structural_tag`` using parser-tag annotations."""

    if isinstance(structural_tag, (str, dict)):
        structural_tag = StructuralTag.from_json(structural_tag)
        format_obj: Format = structural_tag.format
    elif isinstance(structural_tag, StructuralTag):
        format_obj = structural_tag.format
    elif isinstance(structural_tag, BaseModel):
        format_obj = structural_tag  # type: ignore[assignment]
    else:
        raise TypeError(
            "structural_tag must be a StructuralTag, Format instance, dict, or JSON string"
        )

    parse_tree = _parse_format(format_obj, text)
    return _aggregate_parse_tree(parse_tree)


__all__ = [
    "ParserTag",
    "ConstStringFormat",
    "JSONSchemaFormat",
    "QwenXMLParameterFormat",
    "AnyTextFormat",
    "GrammarFormat",
    "RegexFormat",
    "SequenceFormat",
    "OrFormat",
    "TagFormat",
    "TriggeredTagsFormat",
    "TagsWithSeparatorFormat",
    "Format",
    "StructuralTagItem",
    "StructuralTag",
    "parse_structural_tag_output",
]
