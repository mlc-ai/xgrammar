from typing import Any, Callable, Dict, List, Literal, Optional, Union

from .structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    QwenXMLParameterFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

# ---------- Structural Tag Template ----------

SupportedModelStyles = Literal["llama", "qwen", "qwen_coder", "kimi", "deepseek_r1", "harmony"]
_structural_tag_registry: Dict[SupportedModelStyles, Callable[[Dict[str, Any]], StructuralTag]] = {}
_structural_tag_supported_models: Dict[SupportedModelStyles, List[str]] = {}
_KTHINKEXCLUDES = ["<think>", "</think>"]


def _validate_tool_function(tools: Any) -> None:
    if not isinstance(tools, list):
        raise ValueError("The 'tools' key in the input_dict must be a list.")
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("Each item in the 'tools' list must be a dictionary.")
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


def _register_structural_tag_template(name: str, supported_models: List[str]):
    """Register a structural tag template."""

    def decorator(func):
        _structural_tag_registry[name] = func
        _structural_tag_supported_models[name] = supported_models
        return func

    return decorator


def _get_builtin_structural_tag_template_function(
    format_type: SupportedModelStyles,
) -> Callable[[Dict[str, Any]], StructuralTag]:
    """Get builtin structural tag template function by format type.
    In all the structural tag template formats, users should provide
    a list of tools, each tool should have a "function" key, which is a dictionary
    containing "name" and "parameters" fields. Besides, for the OpenAI Harmony Response Format,
    users should also provide a list of builtin tools, each builtin tool should have a "function"
    key, which is a dictionary containing "name" and "parameters" fields. In addition, for the "qwen",
    "deepseek_r1" and "harmony" formats, "reasoning" key can be provided to enable/disable reasoning mode.
    By default, reasoning mode is enabled.

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
    format_type : SupportedModelStyles
        The format type of the structural tag template.
        Currently supported format types are:
        - "llama": Llama3.1 style structural tag format.
          Supported Models: Llama 3, Llama 4 and other models that follow the same style.
        - "qwen": Qwen3 style structural tag format.
          Supported Models: Qwen3 and other models that follow the same style.
        - "qwen_coder": Qwen-Coder style structural tag format.
          Supported Models: Qwen3-Coder, Qwen3-Coder-Next and other models that follow the same style.
        - "kimi": Kimi-k2 style structural tag format.
          Supported Models: Kimi-k2, Kimi-k2.5 and other models that follow the same style.
        - "deepseek_r1": Deepseek-v3.1 style structural tag format.
          Supported Models: Deepseek-v3.1, Deepseek-R1, Deepseek-v3.2-exp and other models that follow the same style.
        - "harmony": OpenAI Harmony Response Format (gpt-oss).
          Supported Models: GPT-oss and other models that follow the same style.

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


def get_structural_tag_for_model(
    model: SupportedModelStyles,
    reasoning: bool = True,
    tools: List[Dict[str, Any]] = [],
    builtin_tools: List[Dict[str, Any]] = [],
    force_empty_reasoning: bool = False,
) -> StructuralTag:
    r"""Get structural tag for model. This function can generate structural tag for the given model
    with the given tools, builtin tools and reasoning mode.

    Parameters
    ----------
    model : SupportedModelStyles
        The model type of the structural tag template.
    reasoning : bool
        Whether to enable reasoning mode. i.e. whether to enable the <think>
        and </think> tags.
    tools : List[Dict[str, Any]]
        A list of tools, each tool should have a "function" key, which is a
        dictionary containing "name" and "parameters" fields.
    builtin_tools : List[Dict[str, Any]]
        A list of builtin tools, each builtin tool should have a "function" key,
        which is a dictionary containing "name" and "parameters" fields. This
        is only used for Harmony style.
    force_empty_reasoning : bool
        Whether to force empty reasoning mode. i.e. The model will output
        the empty thinking content at the beginning of the response.

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.
    """
    if not isinstance(reasoning, bool):
        raise ValueError("The 'reasoning' key in the input_dict must be a boolean.")
    if not isinstance(force_empty_reasoning, bool):
        raise ValueError("The 'force_empty_reasoning' key in the input_dict must be a boolean.")
    _validate_tool_function(tools)
    _validate_tool_function(builtin_tools)

    func = _get_builtin_structural_tag_template_function(model)
    input_dict = {
        "tools": tools,
        "builtin_tools": builtin_tools,
        "reasoning": reasoning,
        "force_empty_reasoning": force_empty_reasoning,
    }
    return func(input_dict)


def get_structural_tag_supported_models(
    strucutural_tag_style: Optional[SupportedModelStyles] = None,
) -> Union[Dict[str, List[str]], List[str]]:
    """Get supported models for a given structural tag style.
    If strucutural_tag_style is not provided, return all supported models.

    Parameters
    ----------
    strucutural_tag_style : Optional[SupportedModelStyles]
        The structural tag style.
    Returns
    -------
    Union[Dict[str, List[str]], List[str]]
        A dictionary of supported models for each structural tag style, or a list of supported models.
    """
    if strucutural_tag_style is None:
        return _structural_tag_supported_models
    else:
        return _structural_tag_supported_models[strucutural_tag_style]


@_register_structural_tag_template("llama", ["llama3.1", "llama4"])
def _generate_llama_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Llama 3.1 style structural tag format.
    Reference: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.
    - "force_empty_reasoning": a boolean; when reasoning is on, if True use empty-thinking, if False use thinking.

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.
        This format is used by Llama 3 and other models that follow the same style.

    """
    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)

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
        suffix_tag = TriggeredTagsFormat(
            triggers=['{"name": '], tags=tags, excludes=_KTHINKEXCLUDES
        )
    else:
        suffix_tag = AnyTextFormat(excludes=_KTHINKEXCLUDES)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value="<think>\n\n</think>")
    else:
        prefix_tag = TagFormat(begin="<think>", content=AnyTextFormat(), end="</think>")

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@_register_structural_tag_template("kimi", ["kimi-k2", "kimi-k2.5"])
def _generate_kimi_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Kimi-k2 style structural tag format.
    Reference: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.
    - "force_empty_reasoning": a boolean; when reasoning is on, if True use empty-thinking, if False use thinking.

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is used by Kimi-k2 and other models that follow the same style.
    """
    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)

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

    if len(tags) > 0:
        suffix_tag = TriggeredTagsFormat(
            triggers=["<|tool_call_begin|>"], tags=tags, excludes=_KTHINKEXCLUDES
        )
    else:
        suffix_tag = AnyTextFormat(excludes=_KTHINKEXCLUDES)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value="<think>\n\n</think>")
    else:
        prefix_tag = TagFormat(begin="<think>", content=AnyTextFormat(), end="</think>")

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@_register_structural_tag_template(
    "deepseek_r1", ["deepseek-v3.1", "deepseek-r1", "deepseek-v3.2-exp"]
)
def _generate_deepseek_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Deepseek v3.1 style structural tag format.
    Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/tokenizer_config.json
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.
    - "force_empty_reasoning": a boolean; when reasoning is on, if True use empty-thinking, if False use thinking.

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.
        This format is used by Deepseek-v3.1 and other models that follow the same style.

    """
    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)

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

    if len(tags) > 0:
        suffix_tag = TriggeredTagsFormat(
            triggers=["<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"],
            tags=tags,
            excludes=_KTHINKEXCLUDES,
        )
    else:
        suffix_tag = AnyTextFormat(excludes=_KTHINKEXCLUDES)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value="</think>")
    else:
        prefix_tag = TagFormat(begin="", content=AnyTextFormat(), end="</think>")

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@_register_structural_tag_template("qwen_coder", ["qwen3-coder", "qwen3-coder-next"])
def _generate_qwen_coder_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Qwen3-Coder style structural tag format.
    Reference: https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8/blob/main/chat_template.jinja
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.
    - "force_empty_reasoning": a boolean; when reasoning is on, if True use empty-thinking, if False use thinking.

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.
        This format is used by Qwen3-Coder and other models that follow the same style.
    """
    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)

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
        suffix_tag = TriggeredTagsFormat(
            triggers=["<tool_call>\n<function="], tags=tags, excludes=_KTHINKEXCLUDES
        )
    else:
        suffix_tag = AnyTextFormat(excludes=_KTHINKEXCLUDES)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value="<think>\n\n</think>")
    else:
        prefix_tag = TagFormat(begin="<think>", content=AnyTextFormat(), end="</think>")

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@_register_structural_tag_template("qwen", ["qwen3"])
def _generate_qwen_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Qwen3 style structural tag format.
    Reference: https://qwen.readthedocs.io/en/latest/framework/function_call.html
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is used by Qwen3 and other models that follow the same style.

    """
    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)

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
    if len(tags) > 0:
        suffix_tag = TriggeredTagsFormat(
            triggers=["<tool_call>"], tags=tags, excludes=_KTHINKEXCLUDES
        )
    else:
        suffix_tag = AnyTextFormat(excludes=_KTHINKEXCLUDES)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value="<think>\n\n</think>")
    else:
        prefix_tag = TagFormat(begin="<think>", content=AnyTextFormat(), end="</think>")

    sequence_format = SequenceFormat(elements=[prefix_tag, suffix_tag])
    return StructuralTag(format=sequence_format)


@_register_structural_tag_template("harmony", ["gpt-oss"])
def _generate_harmony_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get harmony(gpt-oss) style structural tag format.
    Reference: https://developers.openai.com/cookbook/articles/openai-harmony
    Reference: https://huggingface.co/openai/gpt-oss-120b/blob/main/chat_template.jinja
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "builtin_tools": a list of builtin tools, each builtin tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.
    - "force_empty_reasoning": a boolean; when reasoning is on, if True use empty-thinking, if False use thinking.

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is in OpenAI Harmony Response Format, which is used by GPT-oss
        and other models that follow the same style.

    """
    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)
    builtin_tools = input_dict.get("builtin_tools", [])

    tags = []

    if reasoning:
        if force_empty_reasoning:
            analysis_tag = TagFormat(
                begin="<|channel|>analysis<|message|>",
                content=ConstStringFormat(value="<|end|>"),
                end="",
            )
        else:
            analysis_tag = TagFormat(
                begin="<|channel|>analysis<|message|>", content=AnyTextFormat(), end="<|end|>"
            )
        tags.append(analysis_tag)

    for tool in tools:
        if "function" not in tool:
            continue

        function = tool["function"]
        parameters = function["parameters"]
        name = function["name"]
        tags.append(
            TagFormat(
                begin=f"<|channel|>commentary to={name}<|constrain|>json<|message|>",
                content=JSONSchemaFormat(json_schema=parameters),
                end="<|call|>",
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
                begin=f"<|channel|>analysis to={name}<|message|>",
                content=JSONSchemaFormat(json_schema=parameters),
                end="<|call|>",
            )
        )

    final_tag = TagFormat(
        begin="<|channel|>final<|message|>", content=AnyTextFormat(), end="<|end|>"
    )

    tags.append(final_tag)
    tags_with_separator = TagsWithSeparatorFormat(tags=tags, separator="<|start|>assistant")
    return StructuralTag(format=tags_with_separator)
