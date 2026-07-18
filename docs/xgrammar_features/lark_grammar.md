# Lark Grammar Frontend

XGrammar can compile a practical subset of the Lark grammar syntax used by LLGuidance. The result
is a normal `Grammar`, so it works with the existing compiler, matcher, serialization, and engine
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
- `%llguidance {"allow_initial_skip": true}`

String and regex flags are not currently supported.

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

## Dynamic Tool Calls

LLGuidance uses a lazy text lexeme to allow arbitrary text until a tool-call trigger appears. The
grammar remains static; the trigger changes the active parser state rather than replacing the
grammar at runtime.

XGrammar recognizes the LLGuidance structural-tag pattern below and lowers the complete `start`
rule to `TagDispatch`:

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
string or special-token trigger. General shortest-match regex semantics are not approximated.

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

## Unsupported LLGuidance Extensions

The following features require regular-language or generation-runtime state that is not represented
by XGrammar's current `Grammar` API. The frontend rejects them with a source-located error:

- captures
- general `lazy`, `stop`, `suffix`, and stop captures
- rule-level `max_tokens` and `temperature`
- terminal intersection and complement (`&` and `~`)
- structured `%regex` substring nodes
- multiple named grammars referenced with `@name`
- parametric grammar rules and `%if`
- templates, priorities, `%declare`, and `%override`
- `%llguidance` options other than `allow_initial_skip`

Rejecting these constructs is intentional. Ignoring them would silently accept a different language
or drop generation-time behavior.
