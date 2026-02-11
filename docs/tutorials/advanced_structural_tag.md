# Advanced Topics of the Structural Tag

## Built-in Structural Tag: `get_builtin_structural_tag`

`get_builtin_structural_tag` generates a `StructuralTag` for the given model type with the specified tools and options. The returned structural tag can be used with `Grammar.from_structural_tag` or `GrammarCompiler.compile_structural_tag` to obtain a grammar that matches the function-calling format in the corresponding model's style within any-form text.

Use it when you need to constrain the model to output in a fixed pattern such as "tool name + parameter JSON", e.g. for Llama, Qwen, Kimi, DeepSeek, or OpenAI Harmony.

### Parameters

- **model** (`SupportedModelStyles`): The model type. Supported values:
  - `"llama"`: Llama-style (e.g. Llama 3, Llama 4)
  - `"qwen"`: Qwen-style (e.g. Qwen3)
  - `"qwen_coder"`: Qwen Coder-style (e.g. Qwen3-Coder, Qwen3-Coder-Next)
  - `"kimi"`: Kimi-style (e.g. Kimi-K2, Kimi-K2.5)
  - `"deepseek_r1"`: DeepSeek-style (e.g. DeepSeek-V3.1, DeepSeek-R1, DeepSeek-V3.2-exp)
  - `"harmony"`: OpenAI Harmony Response Format (e.g. gpt-oss)
- **reasoning** (`bool`, optional): Whether to enable reasoning mode (`<think>`/`</think>` tags). Default `True`.
- **tools** (`List[Dict[str, Any]]`, optional): List of tools; each item is a dict with a `"function"` key, which is a dict with `"name"` and `"parameters"` (`parameters` is a JSON Schema dict). Default `[]`.
- **builtin_tools** (`List[Dict[str, Any]]`, optional): List of built-in tools (harmony only); same structure as `tools`. Default `[]`.
- **force_empty_reasoning** (`bool`, optional): When reasoning is on, whether to force empty thinking content at the beginning. Default `False`.

Passing an unsupported `model` raises `ValueError`.

### Returns

`StructuralTag`: The structural tag for the given model's function-calling format.

### Example

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
        {"function": {"name": "user_tool", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
    ],
    builtin_tools=[
        {"function": {"name": "builtin_tool", "parameters": {"type": "object", "properties": {}}}},
    ],
)
grammar = Grammar.from_structural_tag(structural_tag)
```

For formats that support reasoning (like Qwen3, Deepseek-R1, Kimi-k2-thinking ), pass `reasoning` to enable/disable the reasoning mode:

```python
structural_tag = get_builtin_structural_tag("qwen", tools=tools, reasoning=True)
grammar = Grammar.from_structural_tag(structural_tag)
```

If `reasoning` is not passed, reasoning mode is enabled by default. Besides, whe `reasoning` is `True`, You can also set the `force_empty_reasoning` to constrain the reasoning content.

---

## Supported models: `get_builtin_structural_tag_supported_models`

`get_builtin_structural_tag_supported_models` returns the supported model list for each built-in structural tag function. Call it with no args to get `Dict[str, List[str]]` (style → models), or pass a style name (e.g. `"llama"`, `"qwen"`) to get `List[str]` for that style. Use it to confirm which style a model uses before calling `get_builtin_structural_tag`.

---

## Deprecated API: `Grammar.from_structural_tag(tags, triggers)`

**The deprecated API is still available for backward compatibility. However, it is recommended to use the new API instead.**

Create a grammar from structural tags. The structural tag handles the dispatching of different grammars based on the tags and triggers: it initially allows any output, until a trigger is encountered, then dispatch to the corresponding tag; when the end tag is encountered, the grammar will allow any following output, until the next trigger is encountered.

The tags parameter is used to specify the output pattern. It is especially useful for LLM function calling, where the pattern is:
`<function=func_name>{"arg1": ..., "arg2": ...}</function>`.
This pattern consists of three parts: a begin tag (`<function=func_name>`), a parameter list according to some schema (`{"arg1": ..., "arg2": ...}`), and an end tag (`</function>`). This pattern can be described in a StructuralTagItem with a begin tag, a schema, and an end tag. The structural tag is able to handle multiple such patterns by passing them into multiple tags.

The triggers parameter is used to trigger the dispatching of different grammars. The trigger should be a prefix of a provided begin tag. When the trigger is encountered, the corresponding tag should be used to constrain the following output. There can be multiple tags matching the same trigger. Then if the trigger is encountered, the following output should match one of the tags. For example, in function calling, the triggers can be `["<function="]`. Then if `"<function="` is encountered, the following output must match one of the tags (e.g. `<function=get_weather>{"city": "Beijing"}</function>`).

The correspondence of tags and triggers is automatically determined: all tags with the same trigger will be grouped together. User should make sure any trigger is not a prefix of another trigger: then the correspondence of tags and triggers will be ambiguous.

To use this grammar in grammar-guided generation, the GrammarMatcher constructed from structural tag will generate a mask for each token. When the trigger is not encountered, the mask will likely be all-1 and not have to be used (fill_next_token_bitmask returns False, meaning no token is masked). When a trigger is encountered, the mask should be enforced (fill_next_token_bitmask will return True, meaning some token is masked) to the output logits.

The benefit of this method is the token boundary between tags and triggers is automatically handled. The user does not need to worry about the token boundary.

### Parameters(deprecated)

- **tags** (`List[StructuralTagItem]`): The structural tags.
- **triggers** (`List[str]`): The triggers.

### Returns(deprecated)

- **grammar** (`Grammar`): The constructed grammar.

### Example(deprecated)

```python
from pydantic import BaseModel
from typing import List
from xgrammar import Grammar, StructuralTagItem

class Schema1(BaseModel):
    arg1: str
    arg2: int

class Schema2(BaseModel):
    arg3: float
    arg4: List[str]

tags = [
    StructuralTagItem(begin="<function=f>", schema=Schema1, end="</function>"),
    StructuralTagItem(begin="<function=g>", schema=Schema2, end="</function>"),
]
triggers = ["<function="]
grammar = Grammar.from_structural_tag(tags, triggers)
```
