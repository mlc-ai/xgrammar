# Structural Tag Usage

The structural tag API aims to provide a JSON-config-based way to precisely describe the output format of an LLM. It is more flexible and dynamic than the OpenAI API:

* flexible: supports various structures, including function calling, reasoning (\<think\>...\</think\>), etc. More will be supported in the future.
* dynamic: allows a mixture of free-form text and structures such as function calls, entering constrained generation when a pre-set trigger is met.

## Usage

The structural tag is a response format. It's compatible with the OpenAI API.

```javascript
{
    ..., // other request parameters
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

1. const\_string

   The LLM output must exactly match the given string.

   This is useful for like force reasoning, where the LLM output must start with "Let's think step by step".

```javascript
{
    "type": "const_string",
    "text": "..."
}
```

2. JSON schema

   The output should be a valid JSON object that matches the JSON schema.

   This is similar to OpenAI's structured output API. It is a key component of the following elements.

```javascript
{
    "type": "json_schema",
    "json_schema": "..."
}
```

3. sequence

   Concatenate several elements into one. LLM's output must follow each element in order.

```
{
    "type": "sequence",
    "elements": [
        ...
    ]
}
```



4. Or
   The output should follow at least one of the elements.

```
{
    "type": "or",
    "elements": [
        ...
    ]
}
```

5. tag

   The tag represents a chunk of text starting and ending with certain strings. Say `<think>...</think>` or `<function>...</function>`. The content inside needs to follow a specific format

```
{
    "type": "tag",
    "begin": "...",
    "content": {
        ...
    },
    "end": "..."
}
```

   The output should start with the "begin" value, then follow the format in "content", and end with the "end" value.



6. wildcard tag

   The wildcard tag is a special tag whose content allows any text except the end tag.

```
{
    "type": "tag",
    "begin": "...",
    "content": {
        "type": "any_text",
    },
    "end": "..."
}
```

   The wildcard part should not contain an end tag. The end tag should not be empty.



   The any\_text format cannot be used outside a tag.



7. triggered\_tags

   The output should be a mixture of text and tags. Each tag is selected from the tags provided in the tags field. E.g.

```
any_text tag0 any_text tag1 any_text tag2 any_text ...
```

(TODO: consider renaming to triggered\_tags)
Config:

```
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
}
```

The text and tags are separated by triggers. When a trigger is found in the output, the mode will be switched from text to tag.

Each element in the tags field should be an object of the tag format.

By setting the stop\_after\_first to true, this part will stop after the first tag is found.

"at\_least\_one" and "stop\_after\_first" may not be implemented in the first version, but will be added in the future.

8. tags\_with\_separator

   The output should be an alternation of tags and string separators. Each tag is selected from the tags provided in the tags field. E.g.

```
tag0 separator tag1 separator tag2 separator ... tagx
```

   Note there is no separator before the first tag and after the last tag.



   Config:

```
{
    "type": "tags_with_separator",
    "tags": [
        {
            "type": "tag", // optional
            "begin": "..."
            "content": {
                ...
            }
            "end": "..."
        },
        {
            "type": "tag", // optional
            ...
        }
    ],
    "separator": "...",
    "at_least_one": bool,
    "stop_after_first": bool
}
```

   "at\_least\_one" and "stop\_after\_first" may not be implemented in the first version, but will be added in the future.

There will be more formats in the future to provide more control over the output.

## Examples

### Example 1: Tool calling

The structural tag can support most common tool calling formats.

Llama JSON-based tool calling, Gemma:

```
{"name": "function_name", "parameters": params}
```

Config:

```
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

```
<function=function_name>params</function>
```

Config:

```
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

```
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}}
</tool_call>
```

Config:

```
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

````
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1
```jsonc
{params}
```<｜tool▁call▁end｜>

```jsonc
{params}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
````

Config:

````
{
    "type": "structural_tag",
    "format": {
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
                    ]
                }
            }
        ],
        "stop_after_first": true,

    },
}
````

Phi-4-mini:

Similar to DeepSeek-V3, but the tool calling part is wrapped in `<|tool_call|>...<|/tool_call|>` and organized in a list.

```
<|tool_call|>[{"name": "function_name_1", "arguments": params}, {"name": "function_name_2", "arguments": params}]<|/tool_call|>
```

Config:

```
{
    "type": "structural_tag",
    "format": {
        "type": "tag_and_text",
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

The output should start with a reasoning part (`<think>...</think>`), then can generate a mix of text and function calls.

Format:

```
<think> any_text </think> any_text <function=func1> params </function> any_text
```

Config:

```
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

```
<think> any_text </think> <function=func1> params </function>
```

Config:

```
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

````
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
````

### Example 5: Force non-thinking mode

Qwen-3 has a hybrid thinking mode that allows switching between thinking and non-thinking mode. Thinking mode is the same as above, while in non-thinking mode, the output would start with a empty thinking part `<think></think>`, and then can generate any text.

We now specify the non-thinking mode.

```
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

## Compatibility with the OAI Function Calling API

In addition to the OpenAI-compatible API usage described above, the structural tag also provides an equivalent Python API, which can be used to implement the OpenAI Function Calling API with strict format constraints.

### Basic Function Calling

The OpenAI API requires user to specify the tools in the request. Each tool contains a name, description, and parameters.

```py
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            },
            "required": [
                "location"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]

completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "What is the weather like in Paris today?"}],
    tools=tools
)
```

To use the structural tag, we need to

1. Find the tool calling format of the current model (need to be defined in the LLM engine)
2. Construct a structural tag as in *Example 1: Tool calling*
3. Use XGrammar to do constraint decoding with the structural tag.
4. Use the XGrammar structural tag parser to parse the string response into a structured object.
5. Convert the parsed result to the OpenAI Function Calling API format.

### Tool Choice

`tool_choice` is a parameter in the OpenAI API. It can be

* `auto`: Let the model decide which tool to use
* `required`: Call at least one tool
* `{"type": "function", "function": {"name": "function_name"}}`: The forced mode, call exactly one specific function

The above basic section describes the support of the auto mode.

The required mode can be implemented by

```
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

```
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

### Parallel Function Calling

OAI's `parallel_tool_calls` parameter controls if the model can call multiple functions in one round.

* If `true`, the model can call multiple functions in one round. (This is default)
* If `false`, the model can call at most one function in one round.

The above basic section describes the support of the true mode.

The false mode can be implemented by

```
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
