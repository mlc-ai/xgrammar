import json
from typing import Annotated, Any, Dict, List, Literal, Type, Union

from pydantic import BaseModel, Field

# ---------- Basic Formats ----------


class ConstStringFormat(BaseModel):
    type: Literal["const_string"] = "const_string"
    text: str


class JSONSchemaFormat(BaseModel):
    type: Literal["json_schema"] = "json_schema"
    json_schema: Union[bool, Dict[str, Any]]


class AnyTextFormat(BaseModel):
    type: Literal["any_text"] = "any_text"


# ---------- Combinatorial Formats ----------


class SequenceFormat(BaseModel):
    type: Literal["sequence"] = "sequence"
    elements: List["Format"]


class OrFormat(BaseModel):
    type: Literal["or"] = "or"
    elements: List["Format"]


class TagFormat(BaseModel):
    type: Literal["tag"] = "tag"
    begin: str
    content: "Format"
    end: str


class TriggeredTagsFormat(BaseModel):
    type: Literal["triggered_tags"] = "triggered_tags"
    triggers: List[str]
    tags: List[TagFormat]
    at_least_one: bool = False
    stop_after_first: bool = False


class TagsWithSeparatorFormat(BaseModel):
    type: Literal["tags_with_separator"] = "tags_with_separator"
    tags: List[TagFormat]
    separator: str
    at_least_one: bool = False
    stop_after_first: bool = False


# ---------- Discriminated Union ----------


Format = Annotated[
    Union[
        AnyTextFormat,
        ConstStringFormat,
        JSONSchemaFormat,
        OrFormat,
        SequenceFormat,
        TagFormat,
        TriggeredTagsFormat,
        TagsWithSeparatorFormat,
    ],
    Field(discriminator="type"),
]


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
    Top level object, corresponding to `"response_format": {"type":"structural_tag", "format":{...}}` in API
    """

    type: Literal["structural_tag"] = "structural_tag"
    format: Format

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
