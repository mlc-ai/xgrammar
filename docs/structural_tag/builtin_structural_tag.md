# Builtin Structural Tag

## Introduction

XGrammar can be used to force different LLMs to output in correct format. In general, there are two common scenarios:

- LLMs can decide whether to call a tool and which tool to call, and output these tool-calling formats within any-form text.
- LLMs are forced to output one or more tools, and any other text are not allowed.

Moreover, different LLMs usually have different tool-calling tags. To support these scenarios naturally, we provide **Builtin Structural tag** to generate the `StructuralTag` for different scenarios, and different models. The generated `StructuralTag` can be used to constrain the output of LLMs.

## Basic API: `get_builtin_structural_tag`

`get_builtin_structural_tag` generates a `StructuralTag` for the given model type with the specified tools and options. The returned `StructuralTag` can be used with `Grammar.from_structural_tag` or `GrammarCompiler.compile_structural_tag` to obtain the corresponding grammar.

Use it when you need to constrain the model to output in a fixed pattern such as "tool name + parameter JSON", e.g. for Llama, Qwen, Kimi, DeepSeek, or OpenAI Harmony, etc.

### Parameters

- **model** (`str`): The structural-tag style. Valid values are `"llama"`, `"qwen"`, `"qwen_coder"`, `"kimi"`, `"deepseek_r1"`, `"harmony"`, `"deepseek_v3_2"`, `"minimax"`, `"glm47"`, `"gemma4"`. The corresponding supported model names are listed below.
- **reasoning** (`bool`, optional): Whether to enable reasoning mode (`<think>`/`</think>` tags). Default `True`.
- **tools** (`List[Dict[str, Any]]`, optional): List of tools; each item is a dict with a `"function"` key. The `"function"` dict **must** contain a `"name"` (string), and **may** contain:
  - `"parameters"`: JSON Schema, which can be:
    - a **dict** (regular JSON Schema object), for example `{"type": "object", "properties": {...}}`
    - a **bool**: `True` means "any JSON value is accepted", `False` means "no value is accepted".
    If `"parameters"` is missing, the no constraint will be applied.
  - `"strict"`: `bool`. Controls whether the parameter constraints are applied. When `False`, only the function name will be enforced, but the parameters is unconstrained. Default: `True`.
  Default value is `[]`.
- **builtin_tools** (`List[Dict[str, Any]]`, optional): List of built-in tools (used only for `"harmony"`); each element has the same structure as items in `tools`. Default `[]`.
- **force_empty_reasoning** (`bool`, optional): When reasoning is on, whether to force empty thinking content at the beginning. Default `False`.
- **tool_choice** (`Literal["auto", "required"]`, optional): How tool calling is constrained relative to the `tools` list (e.g. optional tools vs. requiring a tool call). `"auto"` means whether to call a tool(s) and which tool(s) to call is determined by the model,and these calling will appear in any-form text. `"required"` means that the model must call a tool(s) and other form outputs are not allowed.

Passing an unsupported `model`, an invalid `tool_choice` will raise `ValueError`.

### Returns

`StructuralTag`: The structural tag for the given model's function-calling format.

## Example

```python
from xgrammar import Grammar, get_builtin_structural_tag

tools = [
    {"function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
    {"function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
]

# Get the Llama-style structural tag and build a grammar
structural_tag = get_builtin_structural_tag("llama", tools=tools)
grammar = Grammar.from_structural_tag(structural_tag)
```

For the Harmony format you must provide both `tools` and `builtin_tools`:

```python
structural_tag = get_builtin_structural_tag(
    "harmony",
    tools=[
        # User tool in strict mode, with a full JSON Schema
        {
            "function": {
                "name": "user_tool",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        },
        # User tool in non-strict mode: name only, arguments unconstrained (equivalent to parameters=True)
        {"function": {"name": "user_tool_untyped", "strict": False}},
    ],
    builtin_tools=[
        # Built-in tools support the same strict / parameters combinations
        {
            "function": {
                "name": "builtin_tool",
                "parameters": {"type": "object", "properties": {}},
            }
        },
    ],
)
grammar = Grammar.from_structural_tag(structural_tag)
```

For formats that support reasoning (like Qwen3, Deepseek-R1, Kimi-k2-thinking ), pass `reasoning` to enable/disable the reasoning mode:

```python
structural_tag = get_builtin_structural_tag("qwen", tools=tools, reasoning=True)
grammar = Grammar.from_structural_tag(structural_tag)
```

If `reasoning` is not passed, reasoning mode is enabled by default. Besides, when `reasoning` is `True`, you can also set `force_empty_reasoning` to constrain the reasoning content.

You can pass **tool choice** options when building the structural_tag; they are included in the internal input passed to each built-in style handler:

```python
structural_tag = get_builtin_structural_tag(
    "llama",
    tools=tools,
    tool_choice="forced",
    forced_function_name="get_weather",
)
```

---

## Supported models


The `model` argument of `get_builtin_structural_tag` accepts the style names below:

| `model` (style) | Supported models |
|-----------------|-------------------|
| `"llama"` | Meta-Llama-3, Llama-3.1, Llama-3.2, Llama-4 |
| `"qwen"` | Qwen3 |
| `"qwen_coder"` | Qwen3-Coder, Qwen3-Coder-Next |
| `"kimi"` | Kimi-K2, Kimi-K2.5 |
| `"deepseek_r1"` | DeepSeek-V3.1, DeepSeek-R1, DeepSeek-V3.2-exp |
| `"harmony"` | gpt-oss |
| `"deepseek_v3_2"` | DeepSeek-V3.2 |
| `"minimax"` | MiniMax-M2.5 |
| `"glm47"` | GLM-5, GLM-4.7 |
| `"gemma4"` | Gemma-4, gemma-4-12b-it, gemma-4-26b-a4b-it, gemma-4-31b-it, gemma-4-e2b-it |

## Mapping to OpenAI Tool Calling Options

Builtin Structural tags also support the strict-format parts of the OpenAI tool-calling API.

- `tool_choice = "auto"`: use `tool_choice = "auto"` in `get_builtin_structural_tag`.
- `tool_choice = "required"`: use `tool_choice = "required"` in `get_builtin_structural_tag`.
- `tool_choice = {"type": "function", "function": {"name": ...}}`: pass the only function as the tools, and use `tool_choice = "required"`.

## Next Steps

* For API reference, see [Structural Tag API Reference](../api/python/builtin_structural_tag).
* For advanced usage, see [Advanced Topics of the Structural Tag](advanced_usage).
