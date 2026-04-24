from typing import Any, Callable, Dict, List, Literal, Union

from .structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    QwenXMLParameterFormat,
    RegexFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

# ---------- API Functions ----------


def get_builtin_structural_tag(
    model: str,
    reasoning: bool = True,
    tools: List[Dict[str, Any]] = [],
    builtin_tools: List[Dict[str, Any]] = [],
    force_empty_reasoning: bool = False,
    tool_choice: Literal["auto", "required"] = "auto",
) -> StructuralTag:
    r"""Get structural tag for model. This function can generate structural tag for the given model
    with the given tools, builtin tools and reasoning mode.

    Parameters
    ----------
    model : str
        The model type of the structural tag template. It should be one of the values:
        - "llama"
        - "qwen"
        - "qwen_coder"
        - "kimi"
        - "deepseek_r1"
        - "harmony"
        - "deepseek_v3_2"
        - "minimax"
        - "glm47"
        - "gemma4"
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
        Some models like Qwen3, DeepSeek-R1 and etc. prefer empty-thinking mode to disable
        reasoning mode instead of non-thinking mode.
    tool_choice : Literal["auto", "required"]
        How tool calls are constrained relative to the provided tools list.
        "auto" means whether to call a tool(s) and which tool(s) to call is determined by the model,
        and these calling will appear in any-form text.
        "required" means that the model must call a tool(s) and other form outputs are not allowed.

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.
    """
    if not isinstance(reasoning, bool):
        raise ValueError("The 'reasoning' key in the input_dict must be a boolean.")
    if not isinstance(force_empty_reasoning, bool):
        raise ValueError("The 'force_empty_reasoning' key in the input_dict must be a boolean.")
    _validate_tool_choice_params(tool_choice)
    _validate_tool_function(tools)
    _validate_tool_function(builtin_tools)

    func = _get_builtin_structural_tag_function(model)
    input_dict = {
        "tools": tools,
        "builtin_tools": builtin_tools,
        "reasoning": reasoning,
        "force_empty_reasoning": force_empty_reasoning,
        "tool_choice": tool_choice,
    }
    return func(input_dict)


# ---------- Helper Functions And Constants ----------


_structural_tag_registry: Dict[str, Callable[[Dict[str, Any]], StructuralTag]] = {}
_THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]
_GEMMA4_EXCLUDE_TOKENS = ["<|channel>", "<channel|>"]
_REQUIRED_TOOLS_ERROR = (
    "The 'tools' list is empty, which is not allowed when 'tool_choice' is 'required'."
)


def _validate_tool_choice_params(tool_choice: Literal["auto", "required"]) -> None:
    if tool_choice not in ("auto", "required"):
        raise ValueError(
            "The 'tool_choice' must be one of " f"['auto', 'required'], got {tool_choice!r}."
        )


def _validate_tool_function(tools: Any) -> None:
    if not isinstance(tools, list):
        raise ValueError("The 'tools' key in the input_dict must be a list.")
    for tool in tools:
        if "function" not in tool:
            continue
        function = tool["function"]
        if "name" not in function:
            raise ValueError("Each function in the 'tools' list must have 'name' key.")
        if not isinstance(function["name"], str):
            raise ValueError("The 'name' key in each tool must be a string.")

        if ("strict" in function and function["strict"] is False) or ("parameters" not in function):
            continue
        else:
            parameters = function["parameters"]
            if not (isinstance(parameters, dict) or isinstance(parameters, bool)):
                raise ValueError("The 'parameters' key in each tool must be a dict or a boolean.")


def _get_function_parameters(function: Dict[str, Any]) -> Union[Dict[str, Any], bool]:
    if ("strict" in function and function["strict"] is False) or ("parameters" not in function):
        return True
    return function["parameters"]


def register_builtin_structural_tag(name: str):
    """Register a structural tag template."""

    def decorator(func):
        _structural_tag_registry[name] = func
        return func

    return decorator


def _get_builtin_structural_tag_function(
    format_type: str,
) -> Callable[[Dict[str, Any]], StructuralTag]:
    """Get builtin structural tag template function by format type.
    In all the structural tag template formats, users should provide
    a list of tools, each tool should have a "function" key, which is a dictionary
    containing "name" and "parameters" fields. Besides, for the OpenAI Harmony Response Format,
    users should also provide a list of builtin tools, each builtin tool should have a "function"
    key, which is a dictionary containing "name" and "parameters" fields. In addition, for the "qwen",
    "deepseek_r1" and "harmony" formats, "reasoning" key can be provided to enable/disable reasoning mode.
    By default, reasoning mode is enabled.
    The ``input_dict`` may also include ``tool_choice`` (``"auto"`` or ``"required"``),
    as set by :func:`get_builtin_structural_tag`.

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
    format_type : str
        The format type of the structural tag template.
        Currently supported format types are:
        - "llama": Llama3.1 style structural tag format.
          Supported Models: Llama 3, Llama 4 and other models that follow the same style.
        - "qwen": Qwen3 style structural tag format.
          Supported Models: Qwen3 and other models that follow the same style.
        - "qwen_coder": Qwen-Coder style structural tag format.
          Supported Models: Qwen3-Coder, Qwen3-Coder-Next and other models that follow the same style.
        - "kimi": Kimi-K2 style structural tag format.
          Supported Models: Kimi-K2, Kimi-K2.5 and other models that follow the same style.
        - "deepseek_r1": Deepseek-R1 style structural tag format.
          Supported Models: Deepseek-V3.1, Deepseek-R1, Deepseek-V3.2-exp and other models that follow the same style.
        - "harmony": OpenAI Harmony Response Format (gpt-oss).
          Supported Models: GPT-oss and other models that follow the same style.
        - "gemma4": Gemma 4 style structural tag format.
          Supported Models: Gemma-4 and other models that follow the same style.

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


# ---------- Each Built-in Structural Tag Function ----------


@register_builtin_structural_tag("llama")
def get_llama_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Llama style structural tag format.
    Reference: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.
    - "force_empty_reasoning": a boolean; when reasoning is on, if True use empty-thinking, if False use thinking.

    Supported models
    ----------------
    - Meta-Llama-3
    - Llama-3.1
    - Llama-3.2
    - Llama-4

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.
        This format is used by Llama 3 and other models that follow the same style.

    """
    TOOL_OBJECT_BEGIN_PREFIX = '{"name": "'
    TOOL_OBJECT_PARAMETERS_PREFIX = '", "parameters": '
    TOOLS_TRIGGER = '{"name": '
    THINK_TAG_BEGIN = "<think>"
    THINK_TAG_END = "</think>"
    EMPTY_THINK_CONTENT = "<think>\n\n</think>"

    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)
    tool_choice = input_dict.get("tool_choice", "auto")

    if tool_choice == "auto":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=(TOOL_OBJECT_BEGIN_PREFIX + name + TOOL_OBJECT_PARAMETERS_PREFIX),
                    content=JSONSchemaFormat(json_schema=parameters),
                    end="}",
                )
            )

        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOLS_TRIGGER], tags=tags, excludes=_THINK_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=_THINK_EXCLUDE_TOKENS)

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=(TOOL_OBJECT_BEGIN_PREFIX + name + TOOL_OBJECT_PARAMETERS_PREFIX),
                    content=JSONSchemaFormat(json_schema=parameters),
                    end="}",
                )
            )
        if len(tags) > 0:
            suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
        else:
            raise ValueError(_REQUIRED_TOOLS_ERROR)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value=EMPTY_THINK_CONTENT)
    else:
        prefix_tag = TagFormat(begin=THINK_TAG_BEGIN, content=AnyTextFormat(), end=THINK_TAG_END)

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@register_builtin_structural_tag("kimi")
def get_kimi_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Kimi-K2 style structural tag format.
    Reference: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.
    - "force_empty_reasoning": a boolean; when reasoning is on, if True use empty-thinking, if False use thinking.

    Supported models
    ----------------
    - Kimi-K2
    - Kimi-K2.5

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is used by Kimi-K2 and other models that follow the same style.
    """
    TOOL_CALL_BEGIN_PREFIX = "<|tool_call_begin|>functions."
    TOOL_CALL_SUFFIX = ":"
    TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"
    TOOL_CALL_END = "<|tool_call_end|>"
    TOOL_CALL_TRIGGER = "<|tool_call_begin|>"
    THINK_TAG_BEGIN = "<think>"
    THINK_TAG_END = "</think>"
    EMPTY_THINK_CONTENT = "<think></think>"

    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)
    tool_choice = input_dict.get("tool_choice", "auto")

    if tool_choice == "auto":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{name}{TOOL_CALL_SUFFIX}",
                    content=SequenceFormat(
                        elements=[
                            RegexFormat(pattern=r"\d+"),
                            ConstStringFormat(value=TOOL_CALL_ARGUMENT_BEGIN),
                            JSONSchemaFormat(json_schema=parameters),
                        ]
                    ),
                    end=TOOL_CALL_END,
                )
            )

        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALL_TRIGGER], tags=tags, excludes=_THINK_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=_THINK_EXCLUDE_TOKENS)

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue
            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{name}{TOOL_CALL_SUFFIX}",
                    content=SequenceFormat(
                        elements=[
                            RegexFormat(pattern=r"\d+"),
                            ConstStringFormat(value=TOOL_CALL_ARGUMENT_BEGIN),
                            JSONSchemaFormat(json_schema=parameters),
                        ]
                    ),
                    end=TOOL_CALL_END,
                )
            )
        if len(tags) > 0:
            suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
        else:
            raise ValueError(_REQUIRED_TOOLS_ERROR)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value=EMPTY_THINK_CONTENT)
    else:
        prefix_tag = TagFormat(begin=THINK_TAG_BEGIN, content=AnyTextFormat(), end=THINK_TAG_END)

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@register_builtin_structural_tag("deepseek_r1")
def get_deepseek_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get DeepSeek-R1 style structural tag format.
    Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/tokenizer_config.json
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.
    - "force_empty_reasoning": a boolean; when reasoning is on, if True use empty-thinking, if False use thinking.

    Supported models
    ----------------
    - DeepSeek-V3.1
    - DeepSeek-R1
    - DeepSeek-V3.2-exp

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.
        This format is used by DeepSeek-R1 and other models that follow the same style.

    """
    TOOL_CALLS_PREFIX = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
    TOOL_SEP = "<｜tool▁sep｜>"
    TOOL_CALL_END = "<｜tool▁call▁end｜>"
    TOOL_CALL_TRIGGER = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
    THINK_TAG_END = "</think>"
    EMPTY_THINK_CONTENT = "</think>"

    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)
    tool_choice = input_dict.get("tool_choice", "auto")

    if tool_choice == "auto":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALLS_PREFIX}{name}{TOOL_SEP}",
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=TOOL_CALL_END,
                )
            )

        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALL_TRIGGER], tags=tags, excludes=_THINK_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=_THINK_EXCLUDE_TOKENS)

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue
            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALLS_PREFIX}{name}{TOOL_SEP}",
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=TOOL_CALL_END,
                )
            )

        if len(tags) > 0:
            suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
        else:
            raise ValueError(_REQUIRED_TOOLS_ERROR)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value=EMPTY_THINK_CONTENT)
    else:
        prefix_tag = TagFormat(begin="", content=AnyTextFormat(), end=THINK_TAG_END)

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@register_builtin_structural_tag("qwen_coder")
def get_qwen_coder_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Qwen3-Coder style structural tag format.
    Reference: https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8/blob/main/chat_template.jinja
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.
    - "force_empty_reasoning": a boolean; when reasoning is on, if True use empty-thinking, if False use thinking.

    Supported models
    ----------------
    - Qwen3-Coder
    - Qwen3-Coder-Next

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.
        This format is used by Qwen3-Coder and other models that follow the same style.
    """
    TOOL_CALL_BEGIN_PREFIX = "<tool_call>\n<function="
    TOOL_CALL_BEGIN_SUFFIX = ">\n"
    TOOL_CALL_END = "\n</function>\n</tool_call>"
    TOOL_CALL_TRIGGER = "<tool_call>\n<function="
    THINK_TAG_BEGIN = "<think>"
    THINK_TAG_END = "</think>"
    EMPTY_THINK_CONTENT = "<think>\n\n</think>"

    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)
    tool_choice = input_dict.get("tool_choice", "auto")

    if tool_choice == "auto":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{name}{TOOL_CALL_BEGIN_SUFFIX}",
                    content=QwenXMLParameterFormat(json_schema=parameters),
                    end=TOOL_CALL_END,
                )
            )

        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALL_TRIGGER], tags=tags, excludes=_THINK_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=_THINK_EXCLUDE_TOKENS)

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue
            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{name}{TOOL_CALL_BEGIN_SUFFIX}",
                    content=QwenXMLParameterFormat(json_schema=parameters),
                    end=TOOL_CALL_END,
                )
            )

        if len(tags) > 0:
            suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
        else:
            raise ValueError(_REQUIRED_TOOLS_ERROR)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value=EMPTY_THINK_CONTENT)
    else:
        prefix_tag = TagFormat(begin=THINK_TAG_BEGIN, content=AnyTextFormat(), end=THINK_TAG_END)

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@register_builtin_structural_tag("qwen")
def get_qwen_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Qwen3 style structural tag format.
    Reference: https://qwen.readthedocs.io/en/latest/framework/function_call.html
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.

    Supported models
    ----------------
    - Qwen3

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is used by Qwen3 and other models that follow the same style.

    """
    TOOL_CALL_BEGIN_PREFIX = '<tool_call>\n{"name": "'
    ARGUMENTS_FIELD_PREFIX = '", "arguments": '
    TOOL_CALL_END = "}\n</tool_call>"
    TOOL_CALL_TRIGGER = "<tool_call>"
    THINK_TAG_BEGIN = "<think>"
    THINK_TAG_END = "</think>"
    EMPTY_THINK_CONTENT = "<think>\n\n</think>"

    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)
    tool_choice = input_dict.get("tool_choice", "auto")

    if tool_choice == "auto":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=(TOOL_CALL_BEGIN_PREFIX + name + ARGUMENTS_FIELD_PREFIX),
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=TOOL_CALL_END,
                )
            )
        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALL_TRIGGER], tags=tags, excludes=_THINK_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=_THINK_EXCLUDE_TOKENS)

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue
            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=(TOOL_CALL_BEGIN_PREFIX + name + ARGUMENTS_FIELD_PREFIX),
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=TOOL_CALL_END,
                )
            )

        if len(tags) > 0:
            suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
        else:
            raise ValueError(_REQUIRED_TOOLS_ERROR)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value=EMPTY_THINK_CONTENT)
    else:
        prefix_tag = TagFormat(begin=THINK_TAG_BEGIN, content=AnyTextFormat(), end=THINK_TAG_END)

    sequence_format = SequenceFormat(elements=[prefix_tag, suffix_tag])
    return StructuralTag(format=sequence_format)


@register_builtin_structural_tag("harmony")
def get_harmony_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get harmony(gpt-oss) style structural tag format.
    Reference: https://developers.openai.com/cookbook/articles/openai-harmony
    Reference: https://huggingface.co/openai/gpt-oss-120b/blob/main/chat_template.jinja
    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "builtin_tools": a list of builtin tools, each builtin tool should have a "function" key, which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.
    - "force_empty_reasoning": a boolean; when reasoning is on, if True use empty-thinking, if False use thinking.

    Supported models
    ----------------
    - gpt-oss

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is in OpenAI Harmony Response Format, which is used by GPT-oss
        and other models that follow the same style.

    """
    COMMENTARY_CHANNEL_PREFIX = "<|channel|>commentary to="
    ANALYSIS_CHANNEL_PREFIX = "<|channel|>analysis to="
    JSON_CONSTRAIN_SUFFIX = "<|constrain|>json<|message|>"
    ANALYSIS_MESSAGE_SUFFIX = "<|message|>"
    CALL_END = "<|call|>"
    FINAL_BEGIN = "<|channel|>final<|message|>"
    FINAL_END = "<|end|>"
    ANALYSIS_BEGIN = "<|channel|>analysis<|message|>"
    TAG_SEPARATOR = "<|start|>assistant"

    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)
    builtin_tools = input_dict.get("builtin_tools", [])
    tool_choice = input_dict.get("tool_choice", "auto")
    tags = []

    if tool_choice == "auto":

        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=f"{COMMENTARY_CHANNEL_PREFIX}{name}{JSON_CONSTRAIN_SUFFIX}",
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=CALL_END,
                )
            )

        for tool in builtin_tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=f"{ANALYSIS_CHANNEL_PREFIX}{name}{ANALYSIS_MESSAGE_SUFFIX}",
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=CALL_END,
                )
            )
        final_tag = TagFormat(begin=FINAL_BEGIN, content=AnyTextFormat(), end=FINAL_END)
        tags.append(final_tag)

    elif tool_choice == "required":
        for tool in builtin_tools:
            if "function" not in tool:
                continue
            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=f"{ANALYSIS_CHANNEL_PREFIX}{name}{ANALYSIS_MESSAGE_SUFFIX}",
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=CALL_END,
                )
            )
        for tool in tools:
            if "function" not in tool:
                continue
            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=f"{COMMENTARY_CHANNEL_PREFIX}{name}{JSON_CONSTRAIN_SUFFIX}",
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=CALL_END,
                )
            )
        if len(tags) <= 0:
            raise ValueError(_REQUIRED_TOOLS_ERROR)

    if reasoning:
        if force_empty_reasoning:
            analysis_tag = TagFormat(
                begin=ANALYSIS_BEGIN, content=ConstStringFormat(value=FINAL_END), end=""
            )
        else:
            analysis_tag = TagFormat(begin=ANALYSIS_BEGIN, content=AnyTextFormat(), end=FINAL_END)
        tags.append(analysis_tag)

    tags_with_separator = TagsWithSeparatorFormat(tags=tags, separator=TAG_SEPARATOR)
    return StructuralTag(format=tags_with_separator)


@register_builtin_structural_tag("deepseek_v3_2")
def get_deepseek_v3_2_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get DeepSeek-V3.2 style structural tag format.

    Supported models
    ----------------
    - DeepSeek-V3.2
    """
    INVOKE_BEGIN_PREFIX = '<｜DSML｜invoke name="'
    INVOKE_BEGIN_SUFFIX = '">\n'
    INVOKE_END = "</｜DSML｜invoke>\n"
    FUNCTION_CALLS_BEGIN = "<｜DSML｜function_calls>\n"
    FUNCTION_CALLS_END = "</｜DSML｜function_calls>\n"
    FUNCTION_CALLS_TRIGGER = "<｜DSML｜function_calls>"
    THINK_TAG_BEGIN = "<think>"
    THINK_TAG_END = "</think>"
    EMPTY_THINK_CONTENT = "<think>\n\n</think>"
    XML_STYLE = "deepseek_xml"

    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)
    tool_choice = input_dict.get("tool_choice", "auto")

    if tool_choice == "auto":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(json_schema=parameters, style=XML_STYLE),
                    end=INVOKE_END,
                )
            )

        # generate function calling triggered tag
        if len(tags) > 0:
            function_calling_tags = TagsWithSeparatorFormat(
                tags=tags, separator="\n", at_least_one=True
            )

            suffix_tag = TriggeredTagsFormat(
                triggers=[FUNCTION_CALLS_TRIGGER],
                tags=[
                    TagFormat(
                        begin=FUNCTION_CALLS_BEGIN,
                        content=function_calling_tags,
                        end=FUNCTION_CALLS_END,
                    )
                ],
                excludes=_THINK_EXCLUDE_TOKENS,
            )
        else:
            suffix_tag = AnyTextFormat(excludes=_THINK_EXCLUDE_TOKENS)

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue
            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(json_schema=parameters, style=XML_STYLE),
                    end=INVOKE_END,
                )
            )
        if len(tags) > 0:
            suffix_tag = SequenceFormat(
                elements=[
                    ConstStringFormat(value=FUNCTION_CALLS_BEGIN),
                    TagsWithSeparatorFormat(tags=tags, separator="\n", at_least_one=True),
                    ConstStringFormat(value=FUNCTION_CALLS_END),
                ]
            )
        else:
            raise ValueError(_REQUIRED_TOOLS_ERROR)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value=EMPTY_THINK_CONTENT)
    else:
        prefix_tag = TagFormat(begin=THINK_TAG_BEGIN, content=AnyTextFormat(), end=THINK_TAG_END)

    sequence_format = SequenceFormat(elements=[prefix_tag, suffix_tag])
    return StructuralTag(format=sequence_format)


@register_builtin_structural_tag("minimax")
def get_minimax_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get MiniMax-M2.5 style structural tag format.

    Supported models
    ----------------
    - MiniMax-M2.5
    """
    INVOKE_BEGIN_PREFIX = '<invoke name="'
    INVOKE_BEGIN_SUFFIX = '">\n'
    INVOKE_END = "</invoke>\n"
    TOOL_CALL_BEGIN = "<minimax:tool_call>\n"
    TOOL_CALL_END = "</minimax:tool_call>\n"
    TOOL_CALL_TRIGGER = "<minimax:tool_call>"
    THINK_TAG_BEGIN = "<think>"
    THINK_TAG_END = "</think>"
    EMPTY_THINK_CONTENT = "<think>\n\n</think>"
    XML_STYLE = "minimax_xml"

    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)
    tool_choice = input_dict.get("tool_choice", "auto")

    if tool_choice == "auto":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(json_schema=parameters, style=XML_STYLE),
                    end=INVOKE_END,
                )
            )

        # generate function calling triggered tag
        if len(tags) > 0:
            function_calling_tags = TagsWithSeparatorFormat(
                tags=tags, separator="\n", at_least_one=True
            )

            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALL_TRIGGER],
                tags=[
                    TagFormat(
                        begin=TOOL_CALL_BEGIN, content=function_calling_tags, end=TOOL_CALL_END
                    )
                ],
                excludes=_THINK_EXCLUDE_TOKENS,
            )
        else:
            suffix_tag = AnyTextFormat(excludes=_THINK_EXCLUDE_TOKENS)

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue
            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(json_schema=parameters, style=XML_STYLE),
                    end=INVOKE_END,
                )
            )
        if len(tags) > 0:
            suffix_tag = SequenceFormat(
                elements=[
                    ConstStringFormat(value=TOOL_CALL_BEGIN),
                    TagsWithSeparatorFormat(tags=tags, separator="\n", at_least_one=True),
                    ConstStringFormat(value=TOOL_CALL_END),
                ]
            )
        else:
            raise ValueError(_REQUIRED_TOOLS_ERROR)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value=EMPTY_THINK_CONTENT)
    else:
        prefix_tag = TagFormat(begin=THINK_TAG_BEGIN, content=AnyTextFormat(), end=THINK_TAG_END)

    sequence_format = SequenceFormat(elements=[prefix_tag, suffix_tag])
    return StructuralTag(format=sequence_format)


@register_builtin_structural_tag("glm47")
def get_glm47_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get GLM-4.7/GLM-5 style structural tag format.

    The GLM tool calling format uses XML-like tags:
    <tool_call>function_name
    <arg_key>key</arg_key><arg_value>value</arg_value>
    </tool_call>

    The input_dict should be a dictionary with the following keys:
    - "tools": a list of tools, each tool should have a "function" key, which is a dictionary
      containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.
    - "force_empty_reasoning": a boolean; when reasoning is on, if True use empty-thinking,
      if False use thinking.

    Supported models
    ----------------
    - GLM-5
    - GLM-4.7

    Returns
    -------
    StructuralTag
        A structural tag for GLM function calling format.
    """
    TOOL_CALL_BEGIN_PREFIX = "<tool_call>"
    TOOL_CALL_END = "</tool_call>"
    TOOL_CALL_TRIGGER = "<tool_call>"
    THINK_TAG_BEGIN = "<think>"
    THINK_TAG_END = "</think>"
    EMPTY_THINK_CONTENT = "<think>\n\n</think>"
    XML_STYLE = "glm_xml"

    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)
    tool_choice = input_dict.get("tool_choice", "auto")

    if tool_choice == "auto":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = function["parameters"]
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{name}",
                    content=JSONSchemaFormat(json_schema=parameters, style=XML_STYLE),
                    end=TOOL_CALL_END,
                )
            )

        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALL_TRIGGER], tags=tags, excludes=_THINK_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=_THINK_EXCLUDE_TOKENS)

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue
            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{name}",
                    content=JSONSchemaFormat(json_schema=parameters, style=XML_STYLE),
                    end=TOOL_CALL_END,
                )
            )
        if len(tags) > 0:
            suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
        else:
            raise ValueError(_REQUIRED_TOOLS_ERROR)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value=EMPTY_THINK_CONTENT)
    else:
        prefix_tag = TagFormat(begin=THINK_TAG_BEGIN, content=AnyTextFormat(), end=THINK_TAG_END)

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@register_builtin_structural_tag("gemma4")
def get_gemma4_structural_tag(input_dict: Dict[str, Any]) -> StructuralTag:
    """Get Gemma 4 style structural tag format.

    Gemma 4 uses channel markers for reasoning and tool calls instead of
    ``<think>``/``</think>``:

    - Thinking: ``<|channel>thought\\n...thinking...<channel|>``
    - Tool calls: ``<|tool_call>call:func_name{...}<tool_call|>``
    - Turn end: ``<turn|>``

    Reference: https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4

    The input_dict should be a dictionary with the following keys:

    - "tools": a list of tools, each tool should have a "function" key,
      which is a dictionary containing "name" and "parameters" fields.
    - "reasoning": a boolean indicating whether to enable reasoning mode.
    - "force_empty_reasoning": a boolean; when reasoning is on, if True
      use empty-thinking (pre-closed channel), if False use thinking.
    - "tool_choice": ``"auto"`` or ``"required"``; ``"required"`` forces at least one tool call.

    Supported models
    ----------------
    - Gemma-4
    - gemma-4-12b-it
    - gemma-4-26b-a4b-it
    - gemma-4-31b-it
    - gemma-4-e2b-it

    Returns
    -------
    StructuralTag
        A structural tag for Gemma 4 function calling format.
    """
    TOOL_CALL_BEGIN_PREFIX = "<|tool_call>call:"
    TOOL_CALL_END = "<tool_call|>"
    TOOL_CALL_TRIGGER = "<|tool_call>"
    THINK_TAG_BEGIN = "<|channel>thought\n"
    THINK_TAG_END = "<channel|>"
    EMPTY_THINK_CONTENT = THINK_TAG_BEGIN + THINK_TAG_END

    tools = input_dict.get("tools", [])
    reasoning = input_dict.get("reasoning", True)
    force_empty_reasoning = input_dict.get("force_empty_reasoning", False)
    tool_choice = input_dict.get("tool_choice", "auto")

    if tool_choice == "auto":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=TOOL_CALL_BEGIN_PREFIX + name,
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=TOOL_CALL_END,
                )
            )

        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALL_TRIGGER], tags=tags, excludes=_GEMMA4_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=_GEMMA4_EXCLUDE_TOKENS)

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            if "function" not in tool:
                continue

            function = tool["function"]
            parameters = _get_function_parameters(function)
            name = function["name"]
            tags.append(
                TagFormat(
                    begin=TOOL_CALL_BEGIN_PREFIX + name,
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=TOOL_CALL_END,
                )
            )
        if len(tags) > 0:
            suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
        else:
            raise ValueError(_REQUIRED_TOOLS_ERROR)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    if force_empty_reasoning:
        prefix_tag = ConstStringFormat(value=EMPTY_THINK_CONTENT)
    else:
        prefix_tag = TagFormat(begin=THINK_TAG_BEGIN, content=AnyTextFormat(), end=THINK_TAG_END)

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))
