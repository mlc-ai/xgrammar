# Structural Tag Templates

Based on the [structural tags](./structural_tag.md), structural tag templates provides a more convineient method for users to generate grammars to describe the grammar to constrain the LLMs' output.
In general, structural tag templates support placeholders in the structural tags, which are in the form of `\{\{(var_name([])?)(\.var_name([])?)\}\}`, such as `{{A.name}}`, `{{A[].name}}`, `{{A.name[].other}}`. `var_name[]` means that the value is an array, and it will be automatically expanded with the users' input. `var_name` means that the values is a terminal value, and it will be automatically replaced with the input. Users can use `xgrammar.Grammar.from_structural_tag_template()` to generate a `xgrammar.Grammar`. Here, `template_json_str` is the structural tag templates, and kwargs are a series of values. For example, the structural tag template `stag` contains `{{A[].name}}` and `{{A[].age}}`, then the users can use `xgrammar.Grammar.from_structural_tag_template(stag, A=A)` to generate the grammar, where `A=[{"name":..., "age":...},.{"name":...,"age":...}, ...]`. This function will replace the placeholders automatically.

## Template Placeholders in Formats

1. `const_string`

Each `const_string` format can contain multiple placeholders, but they must be from the same value mapping. for example, `A=[{"begin": "It is", "end":"."}, {"begin": "Is it", "end": "?"}]`, and the template is:

```json
{
    "type": "const_string",
    "value": "{{A[].begin}} a dog{{A[].end}}"
}
```

Is allowed. With the following codes:

```python
A = [{"begin": "It is", "end":"."}, {"begin": "Is it", "end": "?"}]
stag_template = {
    "type": "structural_tag",
    "format": {
        "type": "const_string",
        "value": "{{A[].begin}} a dog{{A[].end}}"
    }
}
stag = StructuralTag.from_template(stag_template, A=A)
```

 And the output is:

 ```json
 {
    "type":"structural_tag",
    "format": {
        "type": "or",
        "elements": [
            {
                "type": "const_string",
                "value": "It is a dog."
            },
            {
                "type": "const_string",
                "value": "Is is a dog?"
            }
        ]
    }
 }
 ```

However, if the provided values are `Begin=[{"word": "It is"}, {"word": "Is it"}], End=[{"word": "."}, {"word": "?"}]`, and the template is

```json
{
    "type": "const_string",
    "value": "{{Begin[].word}} a dog{{End[].word}}"
}
```

cannot be compiled. Because the meaning is ambiguous, and we call these templates **mingled**, we cannot compile them. This template format will be expanded into a `const_string` format or an `or` format, or return nothing.

2. `grammar`, `json_schema`, `qwen_xml_parameter`, `regex`

If the template placeholder is in these formats' value, **only if** the value is exactly the placeholder. For example,

```json
{
    "type": "json_schemas",
    "json_schema": "{{schemas[].schema}}"
}
```

can be automatically replaced with the given `schemas`. For example:

```python
schemas = [
    {
        "value": r"""{"type":"object", "properties": {"arg": {"type": "string"}}, "required": ["arg"]}"""
    },
    {
        "value": r"""{"type":"string"}"""
    }
]
stag_template = {
    "type": "structural_tag",
    "format": {"type": "json_schema", "json_schema": "{{schemas[].value}}"}
}
stag = StructuralTag.from_template(stag_template, schemas=schemas)
```

Then the result will be:

```json
 {
    "type":"structural_tag",
    "format": {
        "type": "or",
        "elements": [
            {
                "type": "json_schema",
                "json_schema": {
                    "type":"object",
                    "properties": {
                        "arg": {"type": "string"}
                    },
                    "required": ["arg"]
                }
            },
            {
                "type": "json_schema",
                "json_schema": {
                    "type":"string"
                }
            }
        ]
    }
 }

```


However, this format will not be replaced:

```json
{
    "type": "json_schemas",
    "json_schema": {
        "type": "{{schemas[].schema}}"
        }
}
```

The same rule holds for the four formats. This template format will be expanded into a `json_schemas` format, or an `or` format, or return nothing.

3. `tag`

For a tag, it is allowed to contain a placeholder in the `begin` and `end` fields. It is also okay if the `content` field is also a template format. However, as the same as `const_string`, `begin` and `end` can contain multiple placeholders, but they must be from the same value mapping. Otherwise, the template is mingled and cannot be compiled. For example, this is a valid tag template:

```json
{
    "type": "tag",
    "begin": "<function={{tools[].name}}",
    "content": {
        "type": "json_schema",
        "json_schema": "{{tools[].parameters}}"
        },
    "end": "</function>"
}
```

This format will be expanded into a `tag` format or an `or` format, or a series of `tag` formats in `triggered_tags`, `tags_with_separator`, or return nothing.

4. `triggered_tags`, `tags_with_separator`

Template placeholders are not allowed in `triggers` and `separators`. The expansion rules are shown in the `tag` format. For example:

```python
tools = [
    {
        "name": "Calculator",
        "description": "A calculator that can perform basic arithmetic operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
    {
        "name": "Weather",
        "description": "A tool to get the current weather in a specified location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The name of the city or location to get the weather for.",
                }
            },
            "required": ["location"],
        },
    },
]
stag_template = {
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "triggers": ["<function="],
        "tags": [{
            "type": "tag",
            "begin": "<function={{tools[].name}}",
            "content": {
                "type": "json_schema",
                "json_schema": "{{tools[].parameters}}"
                },
            "end": "</function>"
        }]
    }
}
stag = StructuralTag.from_template(stag_template, tools=tools)
```

Then the result will be:

```json
{
    "type": "structural_tag",
    "format": [{
        "type": "tag",
        "begin": "<function=Calculator",
        "content": "..."
        "end": "</function>"
    },
    {
        "type": "tag",
        "begin": "<function=Weather",
        "content": "..."
        "end": "</function>"
    }]
}
```

5. `sequence`, `or`, `any_text`

Template placeholders cannot be directly contained in these formats.

## Common Invalid Patterns

### Type 1: Mingled Scope

Not all the structural tag templates are valid. For example, the mingled formats mentioned above are not valid structural tag templates. Besides, there are some other situations where the structural tag template cannot be compiled. For example:

```json
{
    "type": "sequence",
    "elements": [
            {
                "type": "const_string",
                "value": "{{A[].value1}}"
            },
            {
                "type": "const_string",
                "value": "{{B[].value1}}"
            },
            {
                "type": "const_string",
                "value": "{{A[].value2}}"
            },
        ]
}
```

cannot be compiled because we cannot analyze the meaning of the sequence. Each placeholder should have a
distinguishable scope.

### Type 2: Ambiguous array

Stuctural tag templates allow users to create placeholders like `a[].b[].c`. However, for each value, the array-like elements should not be ambiguous, for example:

```json
{
    "type": "sequence",
    "elements": [
            {
                "type": "const_string",
                "value": "{{A[].value1[]}}"
            },
            {
                "type": "const_string",
                "value": "{{A[].value2[]}}"
            },
        ]
}
```

is not allowed. for each value, it should have at most one array in each layer. For example, the below structural tag templates is allowed:

```json
{
    "type": "sequence",
    "elements": [
            {
                "type": "const_string",
                "value": "{{A[].value1[]}}"
            },
            {
                "type": "const_string",
                "value": "{{A[].value1[].value2[]}}"
            },
        ]
}
```

Besides, strucutral tag templates like this is also allowed:

```json
{
    "type": "sequence",
    "elements": [
            {
                "type": "const_string",
                "value": "{{A[].value1[]}}"
            },
            {
                "type": "const_string",
                "value": "{{B[].value2[]}}"
            },
        ]
}
```

## Builtin Structural Tag Templates: Prepared for LLMs' tool-calling

There are several builtin structural tag templates, designed for different LLMs' tool-calling formats. Users can get the builtin structural tag templates with `xgrammar.builtin_structural_tag_template.get_builtin_structural_tag_template(format_type: SupportedTemplateName)`. Currently, these `format_type`s are supported: `Llama`, `kimi`, `deepseek`, `qwen_coder`, `qwen`, `harmony`.
For `llama`, `kimi`, `deepseek`, `qwen_coder`, `qwen`, users need to provide a value `tools=[{"name":..., "parameters":...}, ...]` for the structural tag templates. For `harmony` format, users must provide both `tools=[{"name":..., "parameters":...}]` and `builtin_tools=[{"name":..., "parameters":...}, ...]` for the template. These templates will force the LLMs to output the correct function-calling formats, with other natural language outputs.

For example, if we want to generate a grammar to constrain the output of a Llama model to output text or valid tool-calling, we can use the following code:

```python
from xgrammar import Grammar
from xgrammar.builtin_structural_tag_template import get_builtin_structural_tag_template

# Get the llama-style strucutural tag template.
template_stag_format = get_builtin_structural_tag_template("llama")

# Define the tools. 'name' and 'parameters' are required.
tools = [
            {
                "name": "Calculator",
                "description": "A calculator that can perform basic arithmetic operations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                        },
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            },
            {
                "name": "Weather",
                "description": "A tool to get the current weather in a specified location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The name of the city or location to get the weather for.",
                        }
                    },
                    "required": ["location"],
                },
            },
        ]

# Get the EBNF grammar.
grammar = Grammar.from_structural_tag_template(template_stag_format, tools=tools)
```

Then, we can further compile the grammar to constrain the output of the LLM. Here are some valid outputs:


```python
'OK, I will use the Calculator tool to perform the operation. {"name": "Calculator", "parameters": {"operation": "add", "a": 5, "b": 3}}'

'I need to know the weather in Paris. {"name": "Weather", "parameters": {"location": "Paris"}}'

'Some random text'
```
