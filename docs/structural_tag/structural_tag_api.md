# Structural Tag Usage

Structural tags describe LLM outputs as a tree of composable format objects. Use them when a
plain JSON schema is not enough, such as:

- tool calling with model-specific wrappers
- reasoning tags like `<think>...</think>`
- responses that mix free-form text with structured fragments
- token-level delimiters that are not represented as normal text

Structural tags are also compatible with the OpenAI-style `response_format` request shape.

## Request Shape

Pass a structural tag as the `response_format`:

```json
{
    "model": "...",
    "messages": [
        ...
    ],
    "response_format": {
        "type": "structural_tag",
        "format": {
            "type": "...",
            ...
        }
    }
}
```

The `format` field is required. It contains one format object, and that object can recursively nest other format objects. Each format object represents a "chunk" of text.

## How Structural Tags Work

Think of a structural tag as a tree of chunks:

- **Primitive formats** match one chunk directly, such as an exact string or a JSON payload.
- **Composition formats** combine other formats, such as "A then B" or "A or B".
- **Tagging and dispatch formats** wrap content in delimiters or switch into constrained generation
  only after a trigger is seen.
- **Token-level formats** do the same kinds of matching, but at the tokenizer level instead of the
  string level.

### Format Categories

| Category | Formats | What they do |
| --- | --- | --- |
| Primitive | `const_string`, `json_schema`, `grammar`, `regex`, `any_text` | Match one text fragment |
| Composition | `sequence`, `or`, `optional`, `plus`, `star`, `repeat` | Build larger structures from smaller ones |
| Tagging / dispatch | `tag`, `triggered_tags`, `tags_with_separator`, `dispatch` | Wrap content or switch between free text and structured regions |
| Token-level | `token`, `exclude_token`, `any_tokens`, `token_triggered_tags`, `token_dispatch` | Constrain output at token boundaries |

## Quick Start

### Minimal `tag`

Use `tag` when the output must look like `begin + content + end`.

```json
{
    "type": "tag",
    "begin": "<think>",
    "content": {
        "type": "any_text"
    },
    "end": "</think>"
}
```

This matches `<think>...</think>`.

### Minimal `triggered_tags`

Use `triggered_tags` when the model should be free to emit normal text, but switch into a
structured region after seeing a trigger.

```json
{
    "type": "triggered_tags",
    "triggers": ["<function="],
    "tags": [
        {
            "type": "tag",
            "begin": "<function=get_weather>",
            "content": {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            },
            "end": "</function>"
        }
    ]
}
```

This accepts output like:

```text
I will call a tool now. <function=get_weather>{"city": "San Francisco"}</function>
```

### Unlimited Formats and End Detection

Some formats can consume an unbounded amount of output:

- `any_text`
- `any_tokens`
- `triggered_tags`
- `token_triggered_tags`

When one of these formats appears inside a `tag`, the compiler automatically uses the enclosing
`end` marker as part of the stop condition when the levels match:

- string end -> string-level unlimited format
- token end -> token-level unlimited format

This is why a format like `tag("<think>", any_text, "</think>")` can stop cleanly at
`</think>`. For more details, see [Advanced Topics of the Structural Tag](advanced_usage).

## Format Reference

### Primitive Formats

#### `const_string`

Matches one exact string.

| Field | Type | Default |
| --- | --- | --- |
| `value` | `string` | (required) |

- **Use it when**: the output must contain a fixed literal

```json
{
    "type": "const_string",
    "value": "Let's think step by step."
}
```

#### `json_schema`

Matches content that conforms to a JSON Schema.

| Field | Type | Default |
| --- | --- | --- |
| `json_schema` | `object` | (required) |
| `style` | `"json"` \| `"qwen_xml"` \| `"minimax_xml"` \| `"deepseek_xml"` | `"json"` |

- **Use it when**: the structured part is naturally expressed as schema-constrained data

`style` values:

- `"json"`: standard JSON
- `"qwen_xml"`: Qwen-style XML parameters, such as `<parameter=name>value</parameter>`
- `"minimax_xml"`: MiniMax-style XML parameters, such as `<parameter name="name">value</parameter>`
- `"deepseek_xml"`: DeepSeek-v3.2 XML parameter format

```json
{
    "type": "json_schema",
    "json_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
}
```

Use a non-JSON style only when the surrounding model format expects it:

```json
{
    "type": "json_schema",
    "json_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    },
    "style": "qwen_xml"
}
```

#### `grammar`

Matches text with an EBNF grammar.

| Field | Type | Default |
| --- | --- | --- |
| `grammar` | `string` | (required) |

- **Use it when**: JSON Schema is too restrictive or the structure is better described directly as a grammar

```json
{
    "type": "grammar",
    "grammar": "root ::= (\"yes\" | \"no\")"
}
```

If the grammar is too broad, it can make later constraints ineffective. Avoid patterns that can
consume almost anything when you still need precise structure afterward.

#### `regex`

Matches text with a regular expression.

| Field | Type | Default |
| --- | --- | --- |
| `pattern` | `string` | (required) |

- **Use it when**: a small local text fragment is easiest to describe with regex

```json
{
    "type": "regex",
    "pattern": "[A-Z]{3}-[0-9]{4}"
}
```

As with `grammar`, avoid overly broad patterns when later structure still needs to be enforced.

#### `any_text`

Matches arbitrary text.

| Field | Type | Default |
| --- | --- | --- |
| `excludes` | `string[]` | `[]` |

- **Use it when**: the content should remain free-form until some enclosing boundary is reached

```json
{
    "type": "any_text",
    "excludes": ["<function="]
}
```

When `any_text` appears inside a string-level `tag`, the enclosing end string is automatically
treated as a stop condition.

### Composition Formats

#### `sequence`

Matches several formats in order.

| Field | Type | Default |
| --- | --- | --- |
| `elements` | `format[]` | (required) |

- **Use it when**: the output is "first this, then that"

```json
{
    "type": "sequence",
    "elements": [
        {
            "type": "const_string",
            "value": "Answer: "
        },
        {
            "type": "json_schema",
            "json_schema": {
                "type": "object"
            }
        }
    ]
}
```

#### `or`

Matches any one of several alternatives.

| Field | Type | Default |
| --- | --- | --- |
| `elements` | `format[]` | (required) |

- **Use it when**: multiple shapes are valid at the same position

```json
{
    "type": "or",
    "elements": [
        {
            "type": "const_string",
            "value": "yes"
        },
        {
            "type": "const_string",
            "value": "no"
        }
    ]
}
```

#### `optional`

Matches the inner format zero or one time.

| Field | Type | Default |
| --- | --- | --- |
| `content` | `format` | (required) |

- **Use it when**: a fragment may be present or omitted

```json
{
    "type": "optional",
    "content": {
        "type": "const_string",
        "value": "Optional prefix: "
    }
}
```

#### `plus`

Matches the inner format one or more times.

| Field | Type | Default |
| --- | --- | --- |
| `content` | `format` | (required) |

- **Use it when**: at least one repetition is required

```json
{
    "type": "plus",
    "content": {
        "type": "const_string",
        "value": "item"
    }
}
```

#### `star`

Matches the inner format zero or more times.

| Field | Type | Default |
| --- | --- | --- |
| `content` | `format` | (required) |

- **Use it when**: repetition is allowed but not required

```json
{
    "type": "star",
    "content": {
        "type": "const_string",
        "value": "x"
    }
}
```

#### `repeat`

Matches the inner format between `min` and `max` times, inclusive. Use `max: -1` for an unbounded
upper limit.

| Field | Type | Default |
| --- | --- | --- |
| `min` | `int` | (required) |
| `max` | `int` | (required, `-1` = unbounded) |
| `content` | `format` | (required) |

- **Use it when**: repetition needs an explicit range

```json
{
    "type": "repeat",
    "min": 1,
    "max": 3,
    "content": {
        "type": "const_string",
        "value": "item"
    }
}
```

### Tagging and Dispatch Formats

#### `tag`

Matches `begin + content + end`.

| Field | Type | Default |
| --- | --- | --- |
| `begin` | `string` \| `token` | (required) |
| `content` | `format` | (required) |
| `end` | `string` \| `string[]` \| `token` | (required) |

- **Use it when**: one region is wrapped by known delimiters

```json
{
    "type": "tag",
    "begin": "<response>",
    "content": {
        "type": "json_schema",
        "json_schema": {
            "type": "object"
        }
    },
    "end": ["</response>", "</answer>"]
}
```

#### `triggered_tags`

Allows arbitrary text until a trigger is encountered, then dispatches to one of several tags.
After the tag ends, arbitrary text is allowed again until the next trigger.

| Field | Type | Default |
| --- | --- | --- |
| `triggers` | `string[]` | (required) |
| `tags` | `tag[]` | (required) |
| `at_least_one` | `bool` | `false` |
| `stop_after_first` | `bool` | `false` |
| `excludes` | `string[]` | `[]` |

- **Use it when**: tool calls or other structured regions can appear inside otherwise free-form text

Important rules:

- tags inside `triggered_tags` must use string `begin` fields
- each tag should match exactly one trigger
- each trigger should be an unambiguous prefix of the tag(s) it dispatches to

```json
{
    "type": "triggered_tags",
    "triggers": ["<function="],
    "tags": [
        {
            "type": "tag",
            "begin": "<function=get_weather>",
            "content": {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            },
            "end": "</function>"
        },
        {
            "type": "tag",
            "begin": "<function=get_time>",
            "content": {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string"}
                    },
                    "required": ["timezone"]
                }
            },
            "end": "</function>"
        }
    ],
    "at_least_one": false,
    "stop_after_first": false
}
```

Semantics of the two control flags:

- `at_least_one: true` requires at least one dispatched tag and therefore disallows leading free text
- `stop_after_first: true` finishes the `triggered_tags` structure after the first dispatched tag

#### `tags_with_separator`

Matches zero, one, or more tags separated by a fixed separator, with no extra text outside the tag
sequence.

| Field | Type | Default |
| --- | --- | --- |
| `tags` | `tag[]` | (required) |
| `separator` | `string` | (required) |
| `at_least_one` | `bool` | `false` |
| `stop_after_first` | `bool` | `false` |

- **Use it when**: the output is a pure list of tagged fragments

```json
{
    "type": "tags_with_separator",
    "tags": [
        {
            "type": "tag",
            "begin": "<item>",
            "content": {
                "type": "json_schema",
                "json_schema": {"type": "object"}
            },
            "end": "</item>"
        }
    ],
    "separator": ",",
    "at_least_one": false,
    "stop_after_first": false
}
```

#### `dispatch`

Allows free-form text, but when a specified pattern string appears, the following output must
conform to the corresponding format. Uses Aho-Corasick matching internally for efficient
multi-pattern detection.

| Field | Type | Default |
| --- | --- | --- |
| `rules` | `[string, format][]` | (required) |
| `loop` | `bool` | `true` |
| `excludes` | `string[]` | `[]` |

- **Use it when**: you need pattern-triggered structured regions inside free-form text, with
  string-level pattern matching

```json
{
    "type": "dispatch",
    "rules": [
        ["<function=func1>", {"type": "json_schema", "json_schema": ...}],
        ["<function=func2>", {"type": "json_schema", "json_schema": ...}]
    ],
    "loop": true,
    "excludes": ["</end>"]
}
```

To end matching at a specific string, put it in `excludes` and follow the `dispatch` with a
`const_string` in a `sequence`, or use the `end` of an enclosing `tag`:

```json
{
    "type": "tag",
    "begin": "<response>",
    "content": {
        "type": "dispatch",
        "rules": [
            ["<function=func1>", {"type": "json_schema", "json_schema": ...}],
            ["<function=func2>", {"type": "json_schema", "json_schema": ...}]
        ],
        "loop": true,
        "excludes": ["</response>"]
    },
    "end": "</response>"
}
```

Here `dispatch` allows free text and tool calls inside `<response>...</response>`, but
stops when `</response>` appears because it is in `excludes`. The enclosing `tag` then
consumes that end delimiter.

### Token-level Formats

#### `token`

Matches one token by token ID or token string.

| Field | Type | Default |
| --- | --- | --- |
| `token` | `int` \| `string` | (required) |

- **Use it when**: the delimiter is a tokenizer-level symbol rather than normal text

```json
{"type": "token", "token": 42}
{"type": "token", "token": "<|tool_call|>"}
```

When `token` is a string, it is resolved with `tokenizer_info`.

#### `exclude_token`

Matches any single token except those in `exclude_tokens`.

| Field | Type | Default |
| --- | --- | --- |
| `exclude_tokens` | `(int \| string)[]` | `[]` |

- **Use it when**: one token is needed, but some token values must be blocked

```json
{"type": "exclude_token", "exclude_tokens": [42, "</s>"]}
```

When used inside a token-level `tag`, the enclosing end token is automatically excluded.

#### `any_tokens`

Matches zero or more tokens, excluding those in `exclude_tokens`.

| Field | Type | Default |
| --- | --- | --- |
| `exclude_tokens` | `(int \| string)[]` | `[]` |

- **Use it when**: content should remain unconstrained until a token-level boundary is reached

```json
{"type": "any_tokens", "exclude_tokens": ["<|eos|>"]}
```

Semantically, `any_tokens` is equivalent to `star(exclude_token(...))`.

#### `token_triggered_tags`

Token-level version of `triggered_tags`.

| Field | Type | Default |
| --- | --- | --- |
| `trigger_tokens` | `(int \| string)[]` | (required) |
| `tags` | `tag[]` | (required) |
| `exclude_tokens` | `(int \| string)[]` | `[]` |
| `at_least_one` | `bool` | `false` |
| `stop_after_first` | `bool` | `false` |

- **Use it when**: dispatch must happen on special tokens rather than text prefixes

Important rules:

- tags inside `token_triggered_tags` must use `token` format objects as their `begin`
- elements of `trigger_tokens` and `exclude_tokens` can be token IDs or token strings

```json
{
    "type": "token_triggered_tags",
    "trigger_tokens": ["<|tool_call_start|>"],
    "tags": [
        {
            "type": "tag",
            "begin": {"type": "token", "token": "<|tool_call_start|>"},
            "content": {
                "type": "json_schema",
                "json_schema": {
                    "type": "object"
                }
            },
            "end": {"type": "token", "token": "<|tool_call_end|>"}
        }
    ],
    "exclude_tokens": ["<|eos|>"],
    "at_least_one": false,
    "stop_after_first": false
}
```

#### `token_dispatch`

Token-level version of `dispatch`. Patterns are token IDs or token strings instead of text
strings.

| Field | Type | Default |
| --- | --- | --- |
| `rules` | `[int \| string, format][]` | (required) |
| `loop` | `bool` | `true` |
| `exclude_tokens` | `(int \| string)[]` | `[]` |

- **Use it when**: dispatch must happen on special tokens rather than text patterns

```json
{
    "type": "token_dispatch",
    "rules": [
        [100, {"type": "const_string", "value": "x"}],
        ["<|tool|>", {"type": "json_schema", "json_schema": ...}]
    ],
    "loop": true,
    "exclude_tokens": ["</s>"]
}
```

## Common Recipes

### Force Exactly One Tool Call

To require exactly one tool call from a tool set, use `triggered_tags` with both control flags:

```json
{
    "type": "triggered_tags",
    "triggers": ["<function="],
    "tags": [
        {
            "type": "tag",
            "begin": "<function=func1>",
            "content": {"type": "json_schema", "json_schema": ...},
            "end": "</function>"
        },
        {
            "type": "tag",
            "begin": "<function=func2>",
            "content": {"type": "json_schema", "json_schema": ...},
            "end": "</function>"
        }
    ],
    "at_least_one": true,
    "stop_after_first": true
}
```

If the function is fixed in advance, a single `tag` is simpler.

### Mix Reasoning, Free Text, and Tool Calls

Use `sequence` to force an initial reasoning region, then allow later tool calls inside normal text:

```json
{
    "type": "sequence",
    "elements": [
        {
            "type": "tag",
            "begin": "<think>",
            "content": {"type": "any_text"},
            "end": "</think>"
        },
        {
            "type": "triggered_tags",
            "triggers": ["<function="],
            "tags": [
                {
                    "type": "tag",
                    "begin": "<function=func1>",
                    "content": {"type": "json_schema", "json_schema": ...},
                    "end": "</function>"
                },
                {
                    "type": "tag",
                    "begin": "<function=func2>",
                    "content": {"type": "json_schema", "json_schema": ...},
                    "end": "</function>"
                }
            ]
        }
    ]
}
```

### Token-level Delimiters

Some models use special tokens instead of literal strings to start or end a structured region.
Combine `tag`, `any_tokens`, and `token_triggered_tags` for that case:

```json
{
    "type": "sequence",
    "elements": [
        {
            "type": "tag",
            "begin": {"type": "token", "token": "<|think_start|>"},
            "content": {"type": "any_tokens"},
            "end": {"type": "token", "token": "<|think_end|>"}
        },
        {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<|tool_call_start|>"],
            "tags": [
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<|tool_call_start|>"},
                    "content": {"type": "json_schema", "json_schema": ...},
                    "end": {"type": "token", "token": "<|tool_call_end|>"}
                }
            ],
            "exclude_tokens": ["<|eos|>"],
            "at_least_one": true
        }
    ]
}
```

## Built-in Model Styles

If you only need a standard tool-calling layout for a supported model family, prefer the built-in
helper instead of hand-writing every wrapper.

Common examples include OpenAI Harmony response format, Llama, Qwen, Kimi, DeepSeek, and others.

See [Tool Calling and Reasoning](tool_calling_and_reasoning) for `get_model_structural_tag` and the list of supported models.

## Mapping to OpenAI Tool Calling Options

Structural tags are flexible enough to implement the strict-format parts of the OpenAI tool-calling
API. The exact wrapper still depends on the target model's syntax, but the control knobs map
cleanly:

- `tool_choice = "auto"`: use `triggered_tags` with `at_least_one: false`
- `tool_choice = "required"`: use `TagsWithSeparatorFormat` or `OrFormat`.
- `tool_choice = {"type": "function", "function": {"name": ...}}`: use a fixed `tag` format to describe the tool-calling format.
- `parallel_tool_calls = false`: set `stop_after_first: true`
- `parallel_tool_calls = true`: keep `stop_after_first: false`, or use `tags_with_separator` if
  the model expects a pure separated list of calls

See [Tool Calling and Reasoning](tool_calling_and_reasoning) for the mapping from `get_model_structural_tag` to OpenAI Tool Calling Options.

## Next Steps

- For API reference, see [Structural Tag API Reference](../api/python/structural_tag).
- For tool calling and reasoning, see [Tool Calling and Reasoning](tool_calling_and_reasoning).
- For automatic end detection details, and deprecated APIs, see
  [Advanced Topics of the Structural Tag](advanced_usage).
