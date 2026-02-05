from typing import Any, Callable, Dict, Literal

from .structural_tag import (
    AnyTextFormat,
    JSONSchemaFormat,
    QwenXMLParameterFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
    TriggeredTagsFormat,
)

# ---------- Structural Tag Template ----------


def _validate_tool_function(tools: Any) -> None:
    if not isinstance(tools, list):
        raise ValueError("The 'tools' key in the input_dict must be a list.")
    for tool in tools:
        if "function" not in tool:
            continue
        function = tool["function"]
        if not isinstance(function, dict) or "name" not in function or "parameters" not in function:
            raise ValueError(
                "Each function in the 'tools' list must be a dictionary with 'name' and 'parameters' keys."
            )
        if not isinstance(function["name"], str):
            raise ValueError("The 'name' key in each tool must be a string.")
        parameters = function["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("The 'parameters' key in each tool must be a dict.")


_structural_tag_registry = {}


def _register_structural_tag_template(name: str):
    """Register a structural tag template."""

    def decorator(func):
        _structural_tag_registry[name] = func
        return func

    return decorator


SupportedTemplateNames = Literal["llama", "qwen", "qwen_coder", "kimi", "deepseek", "harmony"]


def get_builtin_structural_tag_template_function(
    format_type: SupportedTemplateNames,
) -> Callable[[Dict[str, Any]], StructuralTag]:
    """Get builtin structural tag template function by format type.
    In all the structural tag template formats, users should provide
    a list of tools, each tool should have a "function" key, which is a dictionary
    containing "name" and "parameters" fields. Besides, for the OpenAI Harmony Response Format,
    users should also provide a list of builtin tools, each builtin tool should have a "function"
    key, which is a dictionary containing "name" and "parameters" fields.

    Examples
    --------

    .. code-block:: python

        from xgrammar import get_builtin_structural_tag_template_function, Grammar
        tools = [
            {"function": {"name": "tool1", "parameters": {"param1": {"type": "string"}}}},
            {"function": {"name": "tool2", "parameters": {"param2": {"type": "integer"}}}},
        ]
        builtin_tools = [
            {"function": {"name": "builtin_tool1", "parameters": {"param1": {"type": "string"}}}},
            {"function": {"name": "builtin_tool2", "parameters": {"param2": {"type": "integer"}}}},
        ]
        template_structural_tag = get_builtin_structural_tag_template_function("harmony")
        structural_tag = template_structural_tag({"tools": tools, "builtin_tools": builtin_tools})
        grammar = Grammar.from_structural_tag(structural_tag)

    The above grammar can be used to construct a grammar that matches the function calling
    format of the specified model.



    Parameters
    ----------
    format_type : SupportedTemplateNames
        The format type of the structural tag template.
        Currently supported format types are:
        - "llama": Llama style structural tag format.
        - "qwen": Qwen style structural tag format.
        - "qwen_coder": Qwen Coder style structural tag format.
        - "kimi": Kimi style structural tag format.
        - "deepseek": Deepseek style structural tag format.
        - "harmony": OpenAI Harmony Response Format.

    Returns
    -------
    Callable[[Dict[str, Any]], StructuralTag]
        The corresponding structural tag template function for the given format type.

    Raises
    ------
    ValueError
        If the format type is unknown.

    """
    func = _structural_tag_registry.get(format_type)
    if func is None:
        support_types = list(_structural_tag_registry.keys())
        raise ValueError(f"Unknown format type: {format_type}, support types: {support_types}")
    return func


@_register_structural_tag_template("llama")
def _generate_llama_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Llama 3.1 style structural tag format.
    Reference: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.
        This format is used by Llama 3 and other models that follow the same style.

    """

    tools = input_dict.get("tools", [])
    _validate_tool_function(tools)

    tags = []

    for tool in tools:
        if "function" not in tool:
            continue

        function = tool["function"]
        parameters = function["parameters"]
        name = function["name"]
        tags.append(
            TagFormat(
                begin=('{"name": "' + name + '", "parameters": '),
                content=JSONSchemaFormat(json_schema=parameters),
                end="}",
            )
        )

    if len(tags) > 0:
        return StructuralTag(format=TriggeredTagsFormat(triggers=['{"name": '], tags=tags))
    else:
        return StructuralTag(format=AnyTextFormat())


@_register_structural_tag_template("kimi")
def _generate_kimi_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Kimi-k2 style structural tag format.
    Reference: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "thinking": a boolean indicating whether to enable thinking mode.

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is used by Kimi-k2 and other models that follow the same style.
    """
    tools = input_dict.get("tools", [])
    thinking = input_dict.get("thinking", True)
    if not isinstance(thinking, bool):
        raise ValueError("The 'thinking' key in the input_dict must be a boolean.")
    _validate_tool_function(tools)

    tags = []
    for tool in tools:
        if "function" not in tool:
            continue

        function = tool["function"]
        parameters = function["parameters"]
        name = function["name"]
        tags.append(
            TagFormat(
                begin=f"<|tool_call_begin|>{name}<|tool_call_argument_begin|>",
                content=JSONSchemaFormat(json_schema=parameters),
                end="<|tool_call_end|>",
            )
        )

    think_excludes = ["<think>", "</think>"]

    if thinking:
        prefix_tag = TagFormat(begin="<think>", content=AnyTextFormat(), end="</think>")
        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=["<|tool_call_begin|>"], tags=tags, excludes=think_excludes
            )
        else:
            suffix_tag = AnyTextFormat(excludes=think_excludes)
        sequence_format = SequenceFormat(elements=[prefix_tag, suffix_tag])
        return StructuralTag(format=sequence_format)
    else:
        if len(tags) > 0:
            return StructuralTag(
                format=TriggeredTagsFormat(
                    triggers=["<|tool_call_begin|>"], tags=tags, excludes=think_excludes
                )
            )
        else:
            return StructuralTag(format=AnyTextFormat(excludes=think_excludes))


@_register_structural_tag_template("deepseek")
def _generate_deepseek_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Deepseek v3.1 style structural tag format.
    Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/tokenizer_config.json
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "thinking": a boolean indicating whether to enable thinking mode.

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.
        This format is used by Deepseek-v3.1 and other models that follow the same style.

    """
    tools = input_dict.get("tools", [])
    thinking = input_dict.get("thinking", True)
    if not isinstance(thinking, bool):
        raise ValueError("The 'thinking' key in the input_dict must be a boolean.")
    _validate_tool_function(tools)

    tags = []
    for tool in tools:
        if "function" not in tool:
            continue

        function = tool["function"]
        parameters = function["parameters"]
        name = function["name"]
        tags.append(
            TagFormat(
                begin=f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{name}<｜tool▁sep｜>",
                content=JSONSchemaFormat(json_schema=parameters),
                end="<｜tool▁call▁end｜>",
            )
        )
    think_excludes = ["<think>", "</think>"]

    if thinking:
        prefix_tag = TagFormat(begin="<think>", content=AnyTextFormat(), end="</think>")
        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=["<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"],
                tags=tags,
                excludes=think_excludes,
            )
        else:
            suffix_tag = AnyTextFormat(excludes=think_excludes)
        sequence_format = SequenceFormat(elements=[prefix_tag, suffix_tag])
        return StructuralTag(format=sequence_format)
    else:
        if len(tags) > 0:
            return StructuralTag(
                format=TriggeredTagsFormat(
                    triggers=["<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"],
                    tags=tags,
                    excludes=think_excludes,
                )
            )
        else:
            return StructuralTag(format=AnyTextFormat(excludes=think_excludes))


@_register_structural_tag_template("qwen_coder")
def _generate_qwen_coder_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Qwen3-Coder style structural tag format.
    Reference: https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8/blob/main/chat_template.jinja
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.
        This format is used by Qwen3-Coder and other models that follow the same style.
    """
    tools = input_dict.get("tools", [])
    _validate_tool_function(tools)

    tags = []
    for tool in tools:
        if "function" not in tool:
            continue

        function = tool["function"]
        parameters = function["parameters"]
        name = function["name"]
        tags.append(
            TagFormat(
                begin=f"<tool_call>\n<function={name}>\n",
                content=QwenXMLParameterFormat(json_schema=parameters),
                end="\n</function>\n</tool_call>",
            )
        )

    if len(tags) > 0:
        return StructuralTag(
            format=TriggeredTagsFormat(triggers=["<tool_call>\n<function="], tags=tags)
        )
    else:
        return StructuralTag(format=AnyTextFormat())


@_register_structural_tag_template("qwen")
def _generate_qwen_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Qwen3 style structural tag format.
    Reference: https://qwen.readthedocs.io/en/latest/framework/function_call.html
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "thinking": a boolean indicating whether to enable thinking mode.

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is used by Qwen3 and other models that follow the same style.

    """
    tools = input_dict.get("tools", [])
    thinking = input_dict.get("thinking", True)
    if not isinstance(thinking, bool):
        raise ValueError("The 'thinking' key in the input_dict must be a boolean.")
    _validate_tool_function(tools)

    tags = []
    for tool in tools:
        if "function" not in tool:
            continue

        function = tool["function"]
        parameters = function["parameters"]
        name = function["name"]
        tags.append(
            TagFormat(
                begin=('<tool_call>\n{"name": "' + name + '", "arguments": '),
                content=JSONSchemaFormat(json_schema=parameters),
                end="}\n</tool_call>",
            )
        )
    think_excludes = ["<think>", "</think>"]

    if thinking:
        prefix_tag = TagFormat(begin="<think>", content=AnyTextFormat(), end="</think>")
        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=["<tool_call>"], tags=tags, excludes=think_excludes
            )
        else:
            suffix_tag = AnyTextFormat(excludes=think_excludes)
        sequence_format = SequenceFormat(elements=[prefix_tag, suffix_tag])
        return StructuralTag(format=sequence_format)
    else:
        if len(tags) > 0:
            return StructuralTag(
                format=TriggeredTagsFormat(
                    triggers=["<tool_call>"], tags=tags, excludes=think_excludes
                )
            )
        else:
            return StructuralTag(format=AnyTextFormat(excludes=think_excludes))


@_register_structural_tag_template("harmony")
def _generate_harmony_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get harmony style structural tag format.
    Reference: https://developers.openai.com/cookbook/articles/openai-harmony
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "builtin_tools": a list of builtin tools, each builtin tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is in OpenAI Harmony Response Format, which is used by GPT-oss
        and other models that follow the same style.

    """
    tools = input_dict.get("tools", [])
    builtin_tools = input_dict.get("builtin_tools", [])
    _validate_tool_function(tools)
    _validate_tool_function(builtin_tools)

    tags = [
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
    ]

    for tool in tools:
        if "function" not in tool:
            continue

        function = tool["function"]
        parameters = function["parameters"]
        name = function["name"]
        tags.append(
            TagFormat(
                begin=f"<|start|>assistant<|channel|>commentary to={name}<|constrain|>json<|message|>",
                content=JSONSchemaFormat(json_schema=parameters),
                end="<|end|>",
            )
        )

    for tool in builtin_tools:
        if "function" not in tool:
            continue

        function = tool["function"]
        parameters = function["parameters"]
        name = function["name"]
        tags.append(
            TagFormat(
                begin=f"<|start|>assistant<|channel|>analysis to={name}<|message|>",
                content=JSONSchemaFormat(json_schema=parameters),
                end="<|end|>",
            )
        )

    return StructuralTag(format=TriggeredTagsFormat(triggers=["<|start|>"], tags=tags))
