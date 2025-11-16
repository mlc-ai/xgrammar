from .structural_tag import (
    AnyTextFormat,
    JSONSchemaFormat,
    QwenXMLParameterFormat,
    StructuralTag,
    TagFormat,
    TriggeredTagsFormat,
)

# ---------- Template Structural Tag ----------

_structural_tag_registry = {}


def _register_template_structural_tag_format(name: str):
    """Register a structural tag format."""

    def decorator(func):
        _structural_tag_registry[name] = func
        return func

    return decorator


def get_builtin_template_structural_tag(format_type: str) -> StructuralTag:
    """Get builtin template structural tag format by format type.
    In all the template structural tag formats, users should provide
    a list of tools, each tool should have a "name" and "parameters" field.
    to use the template structural tag format. Besides, for the OpenAI Harmony Response Format,
    users should also provide a list of builtin tools, each builtin tool should have a "name"
    and "parameters" field.

    Examples
    --------

    .. code-block:: python

        from xgrammar import structural_tag, Grammar
        tools = [
            {"name": "tool1", "parameters": {"param1": {"type": "string"}}},
            {"name": "tool2", "parameters": {"param2": {"type": "integer"}}},
        ]
        builtin_tools = [
            {"name": "builtin_tool1", "parameters": {"param1": {"type": "string"}}},
            {"name": "builtin_tool2", "parameters": {"param2": {"type": "integer"}}},
        ]
        template_structural_tag = structural_tag.get_builtin_template_structural_tag("Harmony")
        grammar = Grammar.apply_template_structural_tag(template_structural_tag, tools=tools, builtin_tools=builtin_tools)

    The above grammar can be used to construct a grammar that matches the function calling
    format of the specified model.



    Parameters
    ----------
    format_type : str
        The format type, must be one of the registered format types:
        "Llama", "Kimi", "Deepseek", "Qwen_Coder", "Qwen", "Harmony".

    Returns
    -------
    StructuralTag
        A structural tag template.

    Raises
    ------
    ValueError
        If the format type is unknown.

    """
    func = _structural_tag_registry.get(format_type)
    if func is None:
        support_types = list(_structural_tag_registry.keys())
        raise ValueError(f"Unknown format type: {format_type}, support types: {support_types}")
    return func()


@_register_template_structural_tag_format("llama")
def _get_llama_structural_tag_template() -> StructuralTag:
    """Get Llama style structural tag format.

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is used by Llama 3 and other models that follow the same style.

    """
    return StructuralTag(
        format=TriggeredTagsFormat(
            triggers=['{"name": '],
            tags=[
                TagFormat(
                    begin='{"name": "{{tools[].name}}", "parameters": ',
                    content=JSONSchemaFormat(json_schema="{{tools[].parameters}}"),
                    end="}",
                )
            ],
        )
    )


@_register_template_structural_tag_format("kimi")
def _get_kimi_structral_tag_template() -> StructuralTag:
    """Get Kimi style structural tag format.

    Returns
    -------
    StructuralTag
        A template structural tag format dictionary.
        This format is used by Kimi-v2 and other models that follow the same style.

    """
    return StructuralTag(
        format=TriggeredTagsFormat(
            triggers=["<|tool_call_begin|>"],
            tags=[
                TagFormat(
                    begin="<|tool_call_begin|>{{tools[].name}}<|tool_call_argument_begin|>",
                    content=JSONSchemaFormat(json_schema="{{tools[].parameters}}"),
                    end="<|tool_call_end|>",
                )
            ],
        )
    )


@_register_template_structural_tag_format("deepseek")
def _get_deepseek_structural_tag_template() -> StructuralTag:
    """Get Deepseek style structural tag format.

    Returns
    -------
    StructuralTag
        A template structural tag format dictionary.
        This format is used by Deepseek-v3.1 and other models that follow the same style.

    """
    return StructuralTag(
        format=TriggeredTagsFormat(
            triggers=["<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"],
            tags=[
                TagFormat(
                    begin="<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{{tools[].name}}<｜tool▁sep｜>",
                    content=JSONSchemaFormat(json_schema="{{tools[].parameters}}"),
                    end="<｜tool▁call▁end｜>",
                )
            ],
        )
    )


@_register_template_structural_tag_format("qwen_coder")
def _get_qwen_coder_structural_tag_template() -> StructuralTag:
    """Get Qwen Coder style structural tag format.

    Returns
    -------
    StructuralTag
        A template structural tag format dictionary.
        This format is used by Qwen3 Coder and other models that follow the same style.
    """
    return StructuralTag(
        format=TriggeredTagsFormat(
            triggers=["<function="],
            tags=[
                TagFormat(
                    begin="<function={{tools[].name}}>",
                    content=QwenXMLParameterFormat(json_schema="{{tools[].parameters}}"),
                    end="</function>",
                )
            ],
        )
    )


@_register_template_structural_tag_format("qwen")
def _get_qwen_structural_tag_template() -> StructuralTag:
    """Get Qwen style structural tag format.

    Returns
    -------
    StructuralTag
        A template structural tag format dictionary.
        This format is used by Qwen3 and other models that follow the same style.

    """
    return StructuralTag(
        format=TriggeredTagsFormat(
            triggers=["<tool_call>"],
            tags=[
                TagFormat(
                    begin='<tool_call>{"name": "{{tools[].name}}", "arguments": ',
                    content=JSONSchemaFormat(json_schema="{{tools[].parameters}}"),
                    end="}</tool_call>",
                )
            ],
        )
    )


@_register_template_structural_tag_format("harmony")
def _get_harmony_structural_tag_template() -> StructuralTag:
    """Get harmony style structural tag format.

    Returns
    -------
    StructuralTag
        A template structural tag format dictionary.
        This format is in OpenAI Harmony Response Format, which is used by GPT-oss
        and other models that follow the same style.

    """
    return StructuralTag(
        format=TriggeredTagsFormat(
            triggers=["<|start|>"],
            tags=[
                TagFormat(
                    begin="<|start|>assistant<|channel|>analysis<|message|>",
                    content=AnyTextFormat(),
                    end="<|end|>",
                ),
                TagFormat(
                    begin="<|start|>assistant<|channel|>final<|message|>",
                    content=AnyTextFormat(),
                    end="<|return|>",
                ),
                TagFormat(
                    begin="<|start|>assistant<|channel|>final<|message|>",
                    content=AnyTextFormat(),
                    end="<|call|>",
                ),
                TagFormat(
                    begin="<|start|>assistant<|channel|>commentary to={{tools[].name}}<|constrain|>json<|message|>",
                    content=JSONSchemaFormat(json_schema="{{tools[].parameters}}"),
                    end="<|end|>",
                ),
                TagFormat(
                    begin="<|start|>assistant<|channel|>analysis to={{builtin_tools[].name}}<|message|>",
                    content=JSONSchemaFormat(json_schema="{{builtin_tools[].parameters}}"),
                    end="<|end|>",
                ),
            ],
        )
    )
