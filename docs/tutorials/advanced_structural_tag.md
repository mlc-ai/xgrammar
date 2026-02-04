# Advanced Topics of the Structural Tag

## Built-in structural tag template: `get_builtin_structural_tag_template_function`

`get_builtin_structural_tag_template_function` returns a template function for the given model/format type. That template function takes tool definitions (and any format-specific options), returns a `StructuralTag`, and can be used with `Grammar.from_structural_tag` or `GrammarCompiler.compile_structural_tag` to obtain a grammar that matches each vendor’s function-calling convention without hand-writing begin/schema/end structures.

Use it when you need to constrain the model to output in a fixed pattern such as “tool name + parameter JSON”, e.g. for Llama, Qwen, Kimi, DeepSeek, or OpenAI Harmony.

### Parameters

- **format_type** (`SupportedTemplateNames`): The template type. Supported values:
  - `"llama"`: Llama-style (e.g. Llama 3)
  - `"qwen"`: Qwen-style (e.g. Qwen3; optional thinking)
  - `"qwen_coder"`: Qwen Coder-style (e.g. Qwen3 Coder)
  - `"kimi"`: Kimi-style (e.g. Kimi-v2.5; optional thinking)
  - `"deepseek"`: DeepSeek-style (e.g. DeepSeek-v3.1; optional thinking)
  - `"harmony"`: OpenAI Harmony Response Format

### Returns

A callable: `Callable[[Dict[str, Any]], StructuralTag]`. The dict you pass typically contains:

- **tools** (all formats): List of tools; each item is a dict with `"name"` and `"parameters"` (`parameters` is a JSON Schema dict).
- **thinking** (optional; for qwen / kimi / deepseek): Whether to enable the thinking block; default `True`.
- **builtin_tools** (harmony only): List of built-in tools; same structure as `tools`.

Passing an unsupported `format_type` raises `ValueError`.

### Example

```python
from xgrammar import Grammar, get_builtin_structural_tag_template_function

tools = [
    {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
    {"name": "get_time", "parameters": {"type": "object", "properties": {}}},
]

# Get the Llama-style template function and build a structural tag
fn = get_builtin_structural_tag_template_function("llama")
structural_tag = fn({"tools": tools})

# Build a grammar from the structural tag for constrained generation
grammar = Grammar.from_structural_tag(structural_tag)
```

For the Harmony format you must provide both `tools` and `builtin_tools`:

```python
fn = get_builtin_structural_tag_template_function("harmony")
structural_tag = fn({
    "tools": [
        {"name": "user_tool", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
    ],
    "builtin_tools": [
        {"name": "builtin_tool", "parameters": {"type": "object", "properties": {}}},
    ],
})
grammar = Grammar.from_structural_tag(structural_tag)
```

For formats that support thinking (qwen, kimi, deepseek), pass `thinking` in the dict:

```python
fn = get_builtin_structural_tag_template_function("qwen")
structural_tag = fn({"tools": tools, "thinking": True})
grammar = Grammar.from_structural_tag(structural_tag)
```

If `thinking` is not passed, then the thinking mode will be enabled by default.

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
