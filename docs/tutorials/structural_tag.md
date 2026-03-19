# Structural Tag Usage

The structural tag API aims to provide a JSON-config-based way to precisely describe the output format of an LLM. It is more flexible and dynamic than the OpenAI API:

* **Flexible**: supports various structures, including tool calling, reasoning (\<think\>...\</think\>), etc.
* **Dynamic**: allows a mixture of free-form text and structures such as tool calls, entering constrained generation when a pre-set trigger is met.

It can also be used in the LLM engine to implement the OpenAI Tool Calling API with strict
format constraints, with these benefits:

* Support the advanced tool calling features, such as forced tool calling, parallel tool calling, etc.
* Support the tool calling format of most of the LLMs available in the market with minimal effort.

## Usage

The structural tag is a response format. It's compatible with the OpenAI API. With the
structural tag, the request should be like:

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

The format field requires a format object. We provide several basic format objects, and they can be composed to allow for more complex formats. Each format object represent a "chunk" of text.

## Format Types

1. `const_string`

    The LLM output must exactly match the given string.

    This is useful for like force reasoning, where the LLM output must start with "Let's think step by step".

    ```json
    {
        "type": "const_string",
        "value": "..."
    }
    ```

2. `json_schema`

    The output should be a valid JSON object that matches the JSON schema.

    **`style`** (optional, default: `"json"`): Controls how the content is parsed. Supported values:

    * `"json"`: Standard JSON format. The output must be valid JSON that conforms to the schema.
    * `"qwen_xml"`: Qwen-coder XML style format. The output uses XML-style tags such as `<parameter=name>value</parameter>` to represent schema properties, as used in Qwen tool-calling formats. Use this when you need the same behavior as the legacy `qwen_xml_parameter` format.
    * `"minimax_xml"`: MiniMax XML style format. The output uses XML-style tags such as `<parameter name="key">value</parameter>` (attribute form) to represent schema properties, as used in MiniMax tool-calling formats. Same semantics as `qwen_xml` except the tag format is `<parameter name="key">` instead of `<parameter=key>`.
    * `"deepseek_xml"`: DeepSeek-v3.2 XML style format. The output should be in the format of `<{dsml_token}parameter name=\"key\" string=\"true|false\">value</{dsml_token}parameter>`.

    When `style` is omitted, it defaults to `"json"`.

    ```json
    {
        "type": "json_schema",
        "json_schema": {
            ...
        },
        "style": "json"
    }
    ```

    For Qwen / MiniMax / Deepseek XML style output, set `style` to `qwen_xml` / `minimax_xml` / `deepseek_xml`:

    ```json
    {
        "type": "json_schema",
        "json_schema": {
            ...
        },
        "style": "qwen_xml" / "minimax_xml" / "deepseek_xml"
    }
    ```

3. `grammar`

    This format can be used to match a given ebnf grammar.

    ```json
    {
        "type": "grammar",
        "grammar": "..."
    }
    ```

    We can use it as the context of other structural tags as well. When using grammar constraints, you need to be especially careful. If the grammar is too general (for example .*), it will cause the subsequent constraints to become ineffective.

4. `regex`

    This format can be used to match a given ebnf grammar.

    ```json
    {
        "type": "regex",
        "pattern": "..."
    }
    ```

    We can use it as the context of other structural tags as well. As GrammarFormat, if the regex pattern is too general, it will cause the subsequent constraints to become inefficient as well.

5. `sequence`

    The output should match a sequence of elements.

    ```json
    {
        "type": "sequence",
        "elements": [
            {
                "type": "...",
            },
            {
                "type": "...",
            },
            ...
        ]
    }
    ```

6. `or`

    The output should follow any of the elements.

    ```json
    {
        "type": "or",
        "elements": [
            {
                "type": "...",
            },
            {
                "type": "...",
            },
            ...
        ]
    }
    ```

7. `tag`

    The output must follow `begin content end`. `begin` and `end` can be strings or `token` format objects, and `content` can be any format object. This is useful for LLM outputs such as `<think>...</think>` or `<function>...</function>`.

    ```json
    {
        "type": "tag",
        "begin": "...",
        "content": {
            "type": "...",
        },
        "end": "..."
    }
    ```

    `begin` and `end` can also be `token` format objects for token-level boundary matching:

    ```json
    {
        "type": "tag",
        "begin": {"type": "token", "token": "<|tool_call|>"},
        "content": {"type": "json_schema", "json_schema": ...},
        "end": {"type": "token", "token": "<|end|>"}
    }
    ```

8. `any_text`

    The any_text format allows any text.

    ```json
    {
        "type": "any_text",
        "excludes": ["...", ]
    }
    ```

    We will handle it as a special case when wrapped in a tag:

    ```json
    {
        "type": "tag",
        "begin": "...",
        "content": {
            "type": "any_text",
            "excludes": ["...", ]
        },
        "end": "...",
    }
    ```

    It first accepts the begin tag (can be empty), then any text **except the end tag** and the **excludes**, then the end tag.

9. `triggered_tags`

    The output will match triggered tags. It can allow any output until a trigger is
    encountered, then dispatch to the corresponding tag; when the end tag is encountered, the
    grammar will allow any following output, until the next trigger is encountered.

    Each tag should be matched by exactly one trigger. "matching" means the trigger should be a
    prefix of the begin tag. All the strings in `excludes` will not be accepted before the tag is triggerred.

    **Note:** Tags inside `triggered_tags` must use **string** `begin` fields (not `token` format objects). For token-level dispatch, use `token_triggered_tags` instead.

    ```json
    {
        "type": "triggered_tags",
        "triggers": ["<function="],
        "tags": [
            {
                "begin": "...",
                "content": {
                    ...
                },
                "end": "..."
            },
            {
                "begin": "...",
                "content": {
                    ...
                },
                "end": "..."
            },
        ],
        "at_least_one": bool,
        "stop_after_first": bool,
        "excludes": ["...", ]
    }
    ```

    For example,

    ```json
    {
        "type": "triggered_tags",
        "triggers": ["<function="],
        "tags": [
            {
                "begin": "<function=func1>",
                "content": {
                    "type": "json_schema",
                    "json_schema": ...
                },
                "end": "</function>",
            },
            {
                "begin": "<function=func2>",
                "content": {
                    "type": "json_schema",
                    "json_schema": ...
                },
                "end": "</function>",
            },
        ],
        "at_least_one": false,
        "stop_after_first": false,
    }
    ```

    The above structural tag can accept the following outputs:

    ```text
    <function=func1>{"name": "John", "age": 30}</function>
    <function=func2>{"name": "Jane", "age": 25}</function>
    any_text<function=func1>{"name": "John", "age": 30}</function>any_text1<function=func2>{"name": "Jane", "age": 25}</function>any_text2
    ```

    `at_least_one` makes sure at least one of the tags must be generated. The first tag will
    be generated at the beginning of the output.

    `stop_after_first` will reach the end of the `triggered_tags` structure after the first tag is generated. If there are following tags, they will still be generated; otherwise, the generation
    will stop.

10. `tags_with_separator`

    The output should match zero, one, or more tags, separated by the separator, with no other text allowed.

    ```json
    {
        "type": "tags_with_separator",
        "tags": [
            {
                "type": "tag",
                "begin": "...",
                "content": {
                    "type": "...",
                },
                "end": "...",
            },
        ],
        "separator": "...",
        "at_least_one": bool,
        "stop_after_first": bool,
    }
    ```

    For example,

    ```json
    {
        "type": "tags_with_separator",
        "tags": [
            {
                "type": "tag",
                "begin": "<function=func1>",
                "content": {
                    "type": "json_schema",
                    "json_schema": ...
                },
                "end": "</function>",
            },
        ],
        "separator": ",",
        "at_least_one": false,
        "stop_after_first": false,
    }
    ```

    The above structural tag can accept an empty string, or the following outputs:

    ```text
    <function=func1>{"name": "John", "age": 30}</function>
    <function=func1>{"name": "John", "age": 30}</function>,<function=func2>{"name": "Jane", "age": 25}</function>
    <function=func1>{"name": "John", "age": 30}</function>,<function=func2>{"name": "Jane", "age": 25}</function>,<function=func1>{"name": "John", "age": 30}</function>
    ```

    `at_least_one` makes sure at least one of the tags must be generated.

    `stop_after_first` will reach the end of the `tags_with_separator` structure after the first
    tag is generated. If there are following tags, they will still be generated; otherwise, the
    generation will stop.

11. `optional`

    The inner format may appear **0 or 1 time** (EBNF optional). The output can either match the content once or be empty.

    **`content`** (required): A format object. It can be any of the format types.

    ```json
    {
        "type": "optional",
        "content": {
            "type": "..."
        }
    }
    ```

    For example, to allow an optional prefix:

    ```json
    {
        "type": "optional",
        "content": {
            "type": "const_string",
            "value": "Optional prefix: "
        }
    }
    ```

    The above accepts either an empty string or exactly `Optional prefix: `.

12. `plus`

    The inner format must appear **1 or more times** (EBNF plus). The output matches one or more consecutive occurrences of the content.

    **`content`** (required): A format object. It can be any of the format types.

    ```json
    {
        "type": "plus",
        "content": {
            "type": "..."
        }
    }
    ```

    For example, to require one or more comma-separated items (simplified):

    ```json
    {
        "type": "plus",
        "content": {
            "type": "const_string",
            "value": "item"
        }
    }
    ```

    The above accepts `item`, `itemitem`, `itemitemitem`, and so on.

13. `star`

    The inner format may appear **0 or more times** (EBNF star). The output matches zero or more consecutive occurrences of the content.

    **`content`** (required): A format object. It can be any of the format types.

    ```json
    {
        "type": "star",
        "content": {
            "type": "..."
        }
    }
    ```

    For example, to allow zero or more occurrences of a fragment:

    ```json
    {
        "type": "star",
        "content": {
            "type": "const_string",
            "value": "x"
        }
    }
    ```

    The above accepts an empty string, or `x`, `xx`, `xxx`, and so on.

14. `repeat`

    The inner format may appear **between `min` and `max` times (inclusive)**. This generalizes `plus` (min=1, unbounded) and `star` (min=0, unbounded). Use `max: -1` for an unbounded upper limit (i.e. "at least `min` times").

    **`min`** (required): Minimum number of occurrences (inclusive). Must be non-negative.

    **`max`** (required): Maximum number of occurrences (inclusive). Use `-1` for unbounded.

    **`content`** (required): A format object. It can be any of the format types.

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

    The above accepts `item`, `itemitem`, or `itemitemitem` (exactly 1 to 3 times).

    For "at least N times" (no upper bound), set `max` to `-1`:

    ```json
    {
        "type": "repeat",
        "min": 2,
        "max": -1,
        "content": {
            "type": "const_string",
            "value": "x"
        }
    }
    ```

    The above accepts `xx`, `xxx`, `xxxx`, and so on (2 or more times).

15. `QwenXMLParameterFormat` *(not recommended)*

    **Deprecated.** This format is kept for backward compatibility only. Prefer using `json_schema` with `style: "qwen_xml"` instead (see the `json_schema` format above).

    This is designed for the parameter format of Qwen3-coder. The output should match the given JSON schema in XML style.

    ```json
    {
        "type": "qwen_xml_parameter",
        "json_schema": {
            ...
        }
    }
    ```

    For example,

    ```json
    {
        "type": "qwen_xml_parameter",
        "json_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        },
    }
    ```

    This can accept outputs such like:

    ```xml
    <parameter=name>Bob</parameter><parameter=age>\t100\n</parameter>
    <parameter=name>Bob</parameter>\t\n<parameter=age>\t100\n</parameter>
    <parameter=name>Bob</parameter><parameter=age>100</parameter>
    <parameter=name>"Bob&lt;"</parameter><parameter=age>100</parameter>
    ```

    Note that strings here are in XML style. Moreover, if the parameter's type is `object`, the inner `object` will still be in JSON style. For example:

    ```json
    {
        "type": "qwen_xml_parameter",
        "json_schema": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}, "city": {"type": "string"}},
                    "required": ["street", "city"],
                }
            },
            "required": ["address"],
        },
    }
    ```

    These are valid outputs:

    ```xml
    <parameter=address>{"street": "Main St", "city": "New York"}</parameter>
    <parameter=address>{"street": "Main St", "city": "No more xml escape&<>"}</parameter>
    ```

    And this is an invalid output:

    ```xml
    <parameter=address><parameter=street>Main St</parameter><parameter=city>New York</parameter></parameter>
    ```

15. `token`

    Matches a single token by token ID or token string. Can be used standalone or as the `begin`/`end` of a `tag`.

    ```json
    {"type": "token", "token": 42}
    {"type": "token", "token": "<|tool_call|>"}
    ```

    When `token` is a string, it is resolved to a token ID via `tokenizer_info`.

16. `exclude_token`

    Matches any single token except those in the given set. When wrapped in a `tag` whose `end` is a `token` format, the end token is automatically excluded.

    ```json
    {"type": "exclude_token"}
    {"type": "exclude_token", "exclude_tokens": [42, "</s>"]}
    ```

    Elements in `exclude_tokens` can be integers (token IDs) or strings (token strings resolved via `tokenizer_info`).

17. `any_tokens`

    Matches zero or more tokens, excluding those in the given set. Semantically equivalent to `star(exclude_token(...))`. When wrapped in a `tag` whose `end` is a `token` format, the end token is automatically excluded.

    ```json
    {"type": "any_tokens"}
    {"type": "any_tokens", "exclude_tokens": [42, "</s>"]}
    ```

    Elements in `exclude_tokens` can be integers (token IDs) or strings (token strings resolved via `tokenizer_info`).

18. `token_triggered_tags`

    Similar to `triggered_tags`, but triggers and excludes operate at the token level. When a trigger token is generated, the output dispatches to the corresponding tag.

    **Note:** Tags inside `token_triggered_tags` must use **`token` format objects** as their `begin` fields (not strings). For string-level dispatch, use `triggered_tags` instead.

    ```json
    {
        "type": "token_triggered_tags",
        "trigger_tokens": [100, "<|tool_call|>"],
        "tags": [
            {
                "begin": {"type": "token", "token": 100},
                "content": {"type": "json_schema", "json_schema": ...},
                "end": {"type": "token", "token": 101}
            }
        ],
        "exclude_tokens": ["</s>"],
        "at_least_one": false,
        "stop_after_first": false
    }
    ```

    `trigger_tokens` and `exclude_tokens` elements can be integers or strings. `at_least_one` and `stop_after_first` have the same semantics as in `triggered_tags`.

19. `dispatch`

    A format that maps directly to a **TagDispatch** grammar. It accepts a list of (trigger string, content format) pairs. When the output matches a trigger string, generation continues with the corresponding content format. This is a lower-level alternative to `triggered_tags` when you want exact trigger→rule mapping without tag begin/end structure.

    **`cases`** (required): Array of objects, each with:
    * **`trigger`** (required): Non-empty string. When this prefix is seen, dispatch to the pair’s content.
    * **`content`** (required): A format object for the content after the trigger.

    **`loop`** (optional, default `true`): If true, after handling one dispatch the grammar allows any text until the next trigger. If false, generation stops after the first dispatch.

    **`excludes`** (optional, default `[]`): List of strings that must not appear before a trigger is seen.

    ```json
    {
        "type": "dispatch",
        "cases": [
            {"trigger": "<a>", "content": {"type": "const_string", "value": "x"}},
            {"trigger": "<b>", "content": {"type": "json_schema", "json_schema": {...}}}
        ],
        "loop": true,
        "excludes": ["<bad>"]
    }
    ```

20. `token_dispatch`

    A format that maps directly to a **TokenTagDispatch** grammar. It accepts a list of (trigger token, content format) pairs. When the model generates the trigger token, generation continues with the corresponding content format. This is a lower-level alternative to `token_triggered_tags` when you want exact token→rule mapping without tag begin/end structure.

    **`cases`** (required): Array of objects, each with:
    * **`trigger`** (required): Token ID (integer) or token string (resolved via `tokenizer_info`).
    * **`content`** (required): A format object for the content after the trigger token.

    **`loop`** (optional, default `true`): If true, after one dispatch the grammar allows more tokens until the next trigger.

    **`exclude_tokens`** (optional, default `[]`): Token IDs or strings to exclude before a trigger is seen.

    ```json
    {
        "type": "token_dispatch",
        "cases": [
            {"trigger": 100, "content": {"type": "const_string", "value": "x"}},
            {"trigger": "<|tool|>", "content": {"type": "json_schema", "json_schema": {...}}}
        ],
        "loop": true,
        "exclude_tokens": ["</s>"]
    }
    ```

## Examples

### Example 1: Tool calling

The structural tag can support most common tool calling formats.

Llama JSON-based tool calling, Gemma:

```json
{"name": "function_name", "parameters": params}
```

Corresponding structural tag:

```json
{
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["{\"name\":"],
        "tags": [
            {
                "begin": "{\"name\": \"func1\", \"parameters\": ",
                "content": {"type": "json_schema", "json_schema": ...},
                "end": "}"
            },
            {
                "begin": "{\"name\": \"func2\", \"parameters\": ",
                "content": {"type": "json_schema", "json_schema": ...},
                "end": "}"
            },
        ],
    },
}
```

Llama user-defined custom tool calling:

```xml
<function=function_name>params</function>
```

Corresponding structural tag:

```json
{
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["<function="],
        "tags": [
            {
                "begin": "<function=func1>",
                "content": {"type": "json_schema", "json_schema": ...},
                "end": "</function>",
            },
            {
                "begin": "<function=func2>",
                "content": {"type": "json_schema", "json_schema": ...},
                "end": "</function>",
            },
        ],
    },
}
```

Qwen 2.5/3, Hermes:

```text
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}}
</tool_call>
```

Corresponding structural tag:

```json
{
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["<tool_call>"],
        "tags": [
            {
                "begin": "<tool_call>\n{\"name\": \"func1\", \"arguments\": ",
                "content": {"type": "json_schema", "json_schema": ...},
                "end": "}\n</tool_call>",
            },
            {
                "begin": "<tool_call>\n{\"name\": \"func2\", \"arguments\": ",
                "content": {"type": "json_schema", "json_schema": ...},
                "end": "}\n</tool_call>",
            },
        ],
    },
}
```

DeepSeek:

There is a special tag `<｜tool▁calls▁begin｜> ... <｜tool▁calls▁end｜>` quotes the whole tool calling part.

````text
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1
```jsonc
{params}
```<｜tool▁call▁end｜>

```jsonc
{params}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
````

Corresponding structural tag:

```json
{
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["<｜tool▁calls▁begin｜>"],
        "tags": [
            {
                "begin": "<｜tool▁calls▁begin｜>",
                "end": "<｜tool▁calls▁end｜>",
                "content": {
                    "type": "tags_with_separator",
                    "separator": "\n",
                    "tags": [
                        {
                            "begin": "<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n",
                            "content": {"type": "json_schema", "json_schema": ...},
                            "end": "\n```<｜tool▁call▁end｜>",
                        },
                        {
                            "begin": "<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2\n```jsonc\n",
                            "content": {"type": "json_schema", "json_schema": ...},
                            "end": "\n```<｜tool▁call▁end｜>",
                        }
                    ]
                }
            }
        ],
        "stop_after_first": true,

    },
}
```

Phi-4-mini:

Similar to DeepSeek-V3, but the tool calling part is wrapped in `<|tool_call|>...<|/tool_call|>` and organized in a list.

```text
<|tool_call|>[{"name": "function_name_1", "arguments": params}, {"name": "function_name_2", "arguments": params}]<|/tool_call|>
```

Corresponding structural tag:

```json
{
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["<|tool_call|>"],
        "tags": [
            {
                "begin": "<|tool_call|>[",
                "end": "]<|/tool_call|>",
                "content": {
                    "type": "tags_with_separator",
                    "separator": ", ",
                    "tags": [
                        {
                            "begin": "{\"name\": \"function_name_1\", \"arguments\": ",
                            "content": {"type": "json_schema", "json_schema": ...},
                            "end": "}",
                        },
                        {
                            "begin": "{\"name\": \"function_name_2\", \"arguments\": ",
                            "content": {"type": "json_schema", "json_schema": ...},
                            "end": "}",
                        }
                    ]
                }
            }
        ],
        "stop_after_first": true,
    },
}
```

### Example 2: Force think

The output should start with a reasoning part (`<think>...</think>`), then can generate a mix of text and tool calls.

Format:

```text
<think> any_text </think> any_text <function=func1> params </function> any_text
```

Corresponding structural tag:

```json
{
    "type": "structural_tag",
    "format": {
        "type": "sequence",
        "elements": [
            {
                "type": "tag",
                "begin": "<think>",
                "content": {"type": "any_text"},
                "end": "</think>",
            },
            {
                "type": "triggered_tags",
                "triggers": ["<function="],
                "tags": [
                    {
                        "begin": "<function=func1>",
                        "content": {"type": "json_schema", "json_schema": ...},
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=func2>",
                        "content": {"type": "json_schema", "json_schema": ...},
                        "end": "</function>",
                    },
                ],
            },
        ],
    },
}
```

### Example 3: Think & Force tool calling (Llama style)

The output should start with a reasoning part (`<think>...</think>`), then need to generate exactly one tool call in the tool set.

Format:

```text
<think> any_text </think> <function=func1> params </function>
```

Corresponding structural tag:

```json
{
    "type": "structural_tag",
    "format": {
        "type": "sequence",
        "elements": [
            {
                "type": "tag",
                "begin": "<think>",
                "content": {"type": "any_text"},
                "end": "</think>",
            },
            {
                "type": "triggered_tags",
                "triggers": ["<function="],
                "tags": [
                    {
                        "begin": "<function=func1>",
                        "content": {"type": "json_schema", "json_schema": ...},
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=func2>",
                        "content": {"type": "json_schema", "json_schema": ...},
                        "end": "</function>",
                    },
                ],
                "stop_after_first": true,
                "at_least_one": true,
            },
        ],
    },
}
```

### Example 4: Think & force tool calling (DeepSeek style)

The output should start with a reasoning part (`<think>...</think>`), then must generate a tool call following the DeepSeek style.

Config:

```json
{
    "type": "structural_tag",
    "format": {
        "type": "sequence",
        "elements": [
            {
                "type": "tag",
                "begin": "<think>",
                "content": {"type": "any_text"},
                "end": "</think>",
            },
            {
                "type": "tag_and_text",
                "triggers": ["<｜tool▁calls▁begin｜>"],
                "tags": [
                    {
                        "begin": "<｜tool▁calls▁begin｜>",
                        "end": "<｜tool▁calls▁end｜>",
                        "content": {
                            "type": "tags_with_separator",
                            "separator": "\n",
                            "tags": [
                                {
                                    "begin": "<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n",
                                    "content": {"type": "json_schema", "json_schema": ...},
                                    "end": "\n```<｜tool▁call▁end｜>",
                                },
                                {
                                    "begin": "<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2\n```jsonc\n",
                                    "content": {"type": "json_schema", "json_schema": ...},
                                    "end": "\n```<｜tool▁call▁end｜>",
                                }
                            ],
                            "at_least_one": true, // Note this line!
                            "stop_after_first": true, // Note this line!
                        }
                    }
                ],
                "stop_after_first": true,
            },
        ],
    },
},
```

### Example 5: Force non-thinking mode

Qwen-3 has a hybrid thinking mode that allows switching between thinking and non-thinking mode. Thinking mode is the same as above, while in non-thinking mode, the output would start with a empty thinking part `<think></think>`, and then can generate any text.

We now specify the non-thinking mode.

```json
{
    "type": "structural_tag",
    "format": {
        "type": "sequence",
        "elements": [
            {
                "type": "const_string",
                "text": "<think></think>"
            },
            {
                "type": "triggered_tags",
                "triggers": ["<tool_call>"],
                "tags": [
                    {
                        "begin": "<tool_call>\n{\"name\": \"func1\", \"arguments\": ",
                        "content": {"type": "json_schema", "json_schema": ...},
                        "end": "}\n</tool_call>",
                    },
                    {
                        "begin": "<tool_call>\n{\"name\": \"func2\", \"arguments\": ",
                        "content": {"type": "json_schema", "json_schema": ...},
                        "end": "}\n</tool_call>",
                    },
                ],
            },
        ],
    },
}
```

### Example 6: Token-level tool calling

Some models use special tokens (not byte strings) to delimit tool calls. The `token_triggered_tags` format handles this by dispatching on token IDs.

```json
{
    "type": "structural_tag",
    "format": {
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
}
```

## Compatibility with the OpenAI Tool Calling API

The structural tag can be used to implement the OpenAI Tool Calling API with strict format
constraints. In LLM serving engines, you can use the `xgrammar` Python package to construct the
structural tag and apply it to constrained decoding.

In the OpenAI Tool Calling API, a set of tools is provided using JSON schema. There are also several
features: tool choice (control at least one tool or exactly one tool is called),
parallel tool calling (allow only one tool or multiple tools can be called in one round), etc.

You can construct the structural tag according to the provided tools, and the LLM's specific tool
calling format. The structural tag can be used in XGrammar's constrained decoding workflow to
enable strict format constraints.

### Tool Choice

`tool_choice` is a parameter in the OpenAI API. It can be

* `auto`: Let the model decide which tool to use
* `required`: Call at least one tool in the tool set
* `{"type": "function", "function": {"name": "function_name"}}`: The forced mode, call exactly one specific function

The required mode can be implemented by

```json
{
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["<function="],
        "tags": [
            {
                "begin": "<function=func1>",
                "content": {"type": "json_schema", "json_schema": ...},
                "end": "</function>",
            },
            {
                "begin": "<function=func2>",
                "content": {"type": "json_schema", "json_schema": ...},
                "end": "</function>",
            },
        ],
        "at_least_one": true,
    },
}
```

The forced mode can be implemented by

```json
{
    "type": "structural_tag",
    "format": {
        "type": "tag",
        "begin": "<function=func1>",
        "content": {"type": "json_schema", "json_schema": ...},
        "end": "</function>",
    },
}
```

### Parallel Tool Calling

OAI's `parallel_tool_calls` parameter controls if the model can call multiple functions in one round.

* If `true`, the model can call multiple functions in one round. (This is default)
* If `false`, the model can call at most one function in one round.

`triggered_tags` and `tags_with_separator` has a parameter `stop_after_first` to control if the
generation should stop after the first tag is generated. So the `false` mode can be implemented by:

```json
{
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["<function="],
        "tags": [
            {
                "begin": "<function=func1>",
                "content": {"type": "json_schema", "json_schema": ...},
                "end": "</function>",
            },
            {
                "begin": "<function=func2>",
                "content": {"type": "json_schema", "json_schema": ...},
                "end": "</function>",
            },
        ],
        "stop_after_first": true,
    },
}
```

The `true` mode can be implemented by setting `stop_after_first` to `false`.

## Next Steps

* For API reference, see [Structural Tag API Reference](../api/python/structural_tag).
* For advanced usage, see [Advanced Topics of the Structural Tag](advanced_structural_tag).
