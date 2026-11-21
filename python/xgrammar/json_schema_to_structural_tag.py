from typing import Any, Dict, Optional

from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
)

_THINK_TAG_END = "</think>"
_THINK_SUFFIX = "\n\n"

# Models that emit optional thinking before plain JSON output.
_QWEN_STYLE_MODELS = frozenset({"qwen_3_5", "qwen_3_coder", "qwen_3"})


def json_schema_to_structural_tag(
    model: str,
    schema: Dict[str, Any],
    reasoning: bool = True,
    *,
    any_order: bool = False,
    max_whitespace_cnt: Optional[int] = None,
) -> StructuralTag:
    """Build a structural tag that constrains output to a JSON schema.

    For models with optional thinking (e.g. Qwen3.5), ``reasoning=True`` allows
    free-form thinking before the JSON payload, matching the prefix used by
    :func:`get_model_structural_tag`.
    """
    suffix_tag = JSONSchemaFormat(
        json_schema=schema, style="json", any_order=any_order, max_whitespace_cnt=max_whitespace_cnt
    )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if model not in _QWEN_STYLE_MODELS:
        supported = sorted(_QWEN_STYLE_MODELS)
        raise ValueError(
            f"Unknown model for json_schema_to_structural_tag: {model}. "
            f"Supported models: {supported}"
        )

    prefix_tag = SequenceFormat(
        elements=[
            TagFormat(begin="", content=AnyTextFormat(), end=_THINK_TAG_END),
            ConstStringFormat(value=_THINK_SUFFIX),
        ]
    )
    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))
