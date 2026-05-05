# Tool Calling and Reasoning

## Introduction

An LLM response can mix up to three parts: a **reasoning** section (e.g. `<think>...</think>`), **plain text** output, and one or more **tool calls**. Different models use different tags and orderings for these parts. XGrammar provides **Builtin Structural Tag** to generate a `StructuralTag` that describes the full output structure for a given model, so all three parts can be constrained during decoding.

The API accepts `tools` and `tool_choice` in the [OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat) convention. Serving engines already using that format can adopt XGrammar with minimal changes. For builtin / hosted tools (e.g. `web_search_preview`), the API extends the convention with XGrammar-specific fields; see [OpenAI Tool Call Schema](../api/python/openai_tool_call_schema) for type definitions.

The `tool_choice` parameter controls how tool calls and text are mixed:

- **Auto** (`"auto"`): the model may output plain text, tool calls, or both.
- **Required** (`"required"`): at least one tool call is required; plain-text-only output is not allowed.
- **Forced** (named choice): exactly one specified tool must be called.
- **None** (`"none"`): tool calls are disabled; only text and reasoning are allowed.

The `reasoning` parameter controls whether the model-specific reasoning section is enabled.

## Basic API: `get_model_structural_tag`

`get_model_structural_tag` generates a `StructuralTag` for the given model type with the specified tools and options. The returned `StructuralTag` can be used with `Grammar.from_structural_tag` or `GrammarCompiler.compile_structural_tag` to obtain the corresponding grammar.

Use it when you need to constrain the model to output in a fixed pattern such as "tool name + parameter JSON", e.g. for Llama, Qwen, Kimi, DeepSeek, OpenAI Harmony, etc.

### Parameters

- **model** (`str`): The structural-tag style. Valid values are `"llama"`, `"qwen_3"`, `"qwen_3_5"`, `"qwen_3_coder"`, `"kimi"`, `"deepseek_r1"`, `"deepseek_v3_1"`, `"harmony"`, `"deepseek_v3_2"`, `"minimax"`, `"glm_4_7"`, `"deepseek_v4"`.
- **tools** (`List[ToolParam | dict]`, optional): Function and builtin tools available to the model. The list can contain two kinds of tools:
  - **Function tools** use the OpenAI Chat Completions shape:
    ```json
    {"type": "function", "function": {"name": "...", "parameters": {...}}}
    ```
    The `"parameters"` field accepts a JSON Schema dict, `True` (any JSON), or can be omitted (unconstrained). When `"strict"` is `False`, the parameters constraint is skipped.
  - **Builtin tools** use a compact shape with XGrammar-specific fields:
    ```json
    {"type": "web_search_preview", "name": "browser.search", "parameters": {...}}
    ```
    - `type`: the provider-level builtin tool type.
    - `name`: the model-output tool name (defaults to `type` if omitted).
    - `parameters`: the JSON schema for constrained decoding of the builtin tool arguments.

  Default `None` (treated as empty list).
- **tool_choice** (`ToolChoiceOptionParam | dict | None`, optional): Controls whether the model may or must call tools. Default `"auto"`.
  - `"auto"`: the model chooses between text output and tool calls.
  - `None`: treated the same as `"auto"`.
  - `"none"`: disables all tools.
  - `"required"`: requires at least one tool call.
  - `{"type": "function", "function": {"name": ...}}`: forces one function tool.
  - `{"type": <builtin_type>}`: forces one builtin tool (matched by `type`).
  - `{"type": "allowed_tools", "allowed_tools": {"mode": ..., "tools": [...]}}`: limits available tools before applying its `mode`. The `tools` list may contain both function refs and builtin refs (matched by `type`).
- **reasoning** (`bool`, optional): Whether to enable reasoning mode (`<think>`/`</think>` tags or model-specific equivalents). Default `True`.

Passing an unsupported `model` or an invalid `tool_choice` will raise `ValueError`.

### Returns

`StructuralTag`: The structural tag for the given model's function-calling format.

## Examples

### Function tools

```python
from xgrammar import Grammar, get_model_structural_tag

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

structural_tag = get_model_structural_tag("llama", tools=tools)
grammar = Grammar.from_structural_tag(structural_tag)
```

### Builtin tools

For models that support builtin tools (e.g. Harmony / gpt-oss), include builtin tools in the same `tools` list:

```python
structural_tag = get_model_structural_tag(
    "harmony",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "user_tool",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        },
        {
            "type": "web_search_preview",
            "name": "browser.search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    ],
)
grammar = Grammar.from_structural_tag(structural_tag)
```

### Reasoning mode

For formats that support reasoning (like Qwen3, DeepSeek-R1, Kimi-K2), pass `reasoning` to enable/disable:

```python
structural_tag = get_model_structural_tag("qwen_3", tools=tools, reasoning=True)
grammar = Grammar.from_structural_tag(structural_tag)
```

If `reasoning` is not passed, reasoning mode is enabled by default.

### Tool choice

Force a specific function tool:

```python
structural_tag = get_model_structural_tag(
    "llama",
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}},
)
```

Force a builtin tool by type:

```python
structural_tag = get_model_structural_tag(
    "harmony",
    tools=[...],
    tool_choice={"type": "web_search_preview"},
)
```

Allow only a subset of tools:

```python
structural_tag = get_model_structural_tag(
    "harmony",
    tools=[...],
    tool_choice={
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "auto",
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}},
                {"type": "web_search_preview"},
            ],
        },
    },
)
```

---

## Supported models

The `model` argument of `get_model_structural_tag` accepts the style names below:

| `model` (style) | Supported models |
|-----------------|-------------------|
| `"llama"` | Meta-Llama-3, Llama-3.1, Llama-3.2 |
| `"qwen_3"` | Qwen3, Qwen3-Next |
| `"qwen_3_5"` | Qwen3.5, Qwen3.6 |
| `"qwen_3_coder"` | Qwen3-Coder, Qwen3-Coder-Next |
| `"kimi"` | Kimi-K2, Kimi-K2.5 |
| `"deepseek_r1"` | DeepSeek-R1, DeepSeek-R1-0528 |
| `"deepseek_v3_1"` | DeepSeek-V3.1, DeepSeek-V3.2-exp |
| `"harmony"` | gpt-oss |
| `"deepseek_v3_2"` | DeepSeek-V3.2 |
| `"minimax"` | MiniMax-M2.5 |
| `"glm_4_7"` | GLM-5, GLM-4.7 |
| `"deepseek_v4"` | DeepSeek-V4 |

## Extending with custom models

Use `register_model_structural_tag` to add support for a new model format. See the [Builtin Structural Tag API Reference](../api/python/builtin_structural_tag) for details.

## Next Steps

* For function and tool choice schema definitions, see [OpenAI Tool Call Schema API Reference](../api/python/openai_tool_call_schema).
* For builtin structural tag API reference, see [Builtin Structural Tag API Reference](../api/python/builtin_structural_tag).
* For advanced usage, see [Advanced Topics of the Structural Tag](advanced_usage).
