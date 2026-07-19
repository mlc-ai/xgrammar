# Lark Grammar Frontend

XGrammar's Lark frontend is compatible with the Lark dialect used by LLGuidance. The result is a
normal `Grammar`, so it works with the existing compiler, matcher, serialization, and engine
integrations.

```python
import xgrammar as xgr

grammar = xgr.Grammar.from_lark(
    r'''
    %import common.INT
    %import common.WS
    %ignore WS

    start: item ("," item)*
    item: "id=" INT
    '''
)
```

The root rule must be named `start`.

## Supported Syntax

- lowercase rules and uppercase terminals
- string and regular-expression literals
- ASCII case-insensitive string literals such as `"yes"i`
- the regex `s` flag, such as `/.*/s`; without `s`, `.` does not match newline
- sequences, alternatives, groups, and optional groups
- `?`, `*`, `+`, `~M..N`, `{M}`, `{M,N}`, `{M,}`, and `{,N}` repetitions
- single-character ranges such as `"a".."z"`
- recursive rules and forward references
- `#` and `//` comments
- multiline alternatives beginning with `|`
- hyphens in identifiers
- `%import common.X`, import aliases, and multi-imports
- `%ignore`
- inline `%json { ... }`
- nested `%lark { ... }`
- numeric token sets such as `<[1,3-8]>`, exclusions such as `<[^1,3]>`, and `<[*]>`
- named special tokens when `tokenizer_info` is supplied
- named grammars referenced with `@name`
- `%grammar_options {"allow_initial_skip": true}`

Case-insensitive string literals containing non-ASCII characters are rejected. Regex flags other
than `s` are not currently supported.

## Inline JSON

`%json` behaves like a nonterminal and compiles through XGrammar's JSON Schema converter.

```python
grammar = xgr.Grammar.from_lark(
    r'''
    start: "<tool_call>" arguments "</tool_call>"
    arguments: %json {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"],
      "additionalProperties": false
    }
    '''
)
```

Whitespace outside the JSON value is controlled by the surrounding Lark grammar. Whitespace inside
the value follows the JSON Schema converter's normal behavior.

## Named Grammars

Pass existing `Grammar` objects or Lark grammar strings through `named_grammars` and reference them
with `@name`. Dictionary keys do not include the leading `@`.

```python
arguments = xgr.Grammar.from_json_schema(
    {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
)
grammar = xgr.Grammar.from_lark(
    'start: "<tool_call>" @arguments "</tool_call>" ":" @status',
    named_grammars={
        "arguments": arguments,
        "status": 'start: "ok" | "cancelled"',
    },
)
```

String values are parsed as Lark grammars and may reference other entries in the same mapping. The
same named grammar may be referenced more than once and from inside nested `%lark` blocks.

## Dynamic Tool Calls

A lazy text lexeme can allow arbitrary text until a tool-call trigger appears. The grammar remains
static; the trigger changes the active parser state rather than replacing the grammar at runtime.
XGrammar recognizes the structural-tag pattern below and lowers the complete `start` rule to
`TagDispatch`:

```python
grammar = xgr.Grammar.from_lark(
    r'''
    start: tool* tail
    tail: TEXT

    tool_head[lazy]: TEXT "<tool_call>"
    tool: tool_head %json {
      "type": "object",
      "properties": {"x": {"type": "integer"}},
      "required": ["x"],
      "additionalProperties": false
    } "</tool_call>"

    TEXT: /(\n|.)*/
    '''
)
```

This accepts plain text, one tool call, or multiple tool calls. Once the complete trigger is seen,
the payload and end tag are mandatory. After the end tag, free text is allowed again.

Multiple tool forms may share a trigger prefix:

```text
start: (foo | bar)* tail
tail: TEXT

foo_head[lazy]: TEXT "<function"
foo: foo_head "=foo>" /[a-z]+/ "</function>"

bar_head[lazy]: TEXT "<function"
bar: bar_head "=bar>" /[A-Z]+/ "</function>"

TEXT: /(\n|.)*/
```

The supported lazy form is deliberately narrow: an arbitrary-text terminal followed by one fixed
string or special-token trigger. Dynamic heads may also express the same fixed string trigger as a
regex suffix or a `suffix` attribute:

```text
regex_head[lazy]: /(\n|.)*<tool>/
suffix_head[suffix="<tool>"]: TEXT
```

For regex suffixes, the prefix must be one of the recognized arbitrary-text forms and the remainder
must be a fixed regex literal. These forms are only enabled when the entire `start: tool* tail`
pattern can be lowered to dispatch. General shortest-match regex and standalone suffix semantics
are not approximated.

## Special Tokens

Numeric token references do not require tokenizer metadata:

```python
grammar = xgr.Grammar.from_lark("start: <[128000-128010]>")
```

Named references are resolved by exact match against the decoded vocabulary:

```python
tokenizer_info = xgr.TokenizerInfo(["a", "<|tool|>", "b"])
grammar = xgr.Grammar.from_lark("start: <|tool|>", tokenizer_info=tokenizer_info)
```
