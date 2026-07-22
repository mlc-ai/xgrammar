# EBNF Grammar

EBNF (Extended Backus-Naur Form) is XGrammar's native grammar format. A grammar written in this
format describes exactly which strings the LLM may generate: fixed strings, character sets,
alternatives, repetition, and recursion. Grammars are constructed with
[`xgr.Grammar.from_ebnf`](xgrammar.Grammar.from_ebnf), or by passing the grammar string directly
to [`xgr.GrammarCompiler.compile_grammar`](xgrammar.GrammarCompiler.compile_grammar).

EBNF is also XGrammar's common intermediate representation: every other frontend (JSON Schema,
regular expressions, structural tags, Lark) is converted to it internally, and printing any
[`xgr.Grammar`](xgrammar.Grammar) with `str(grammar)` produces an equivalent grammar in this
format that can be parsed again.

The syntax is compatible with the GBNF (GGML BNF) format used by
[llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md), with many
extensions, such as repetition ranges and macros.

```python
import xgrammar as xgr

grammar = xgr.Grammar.from_ebnf(r"""
root ::= expr ("=" expr)*
expr ::= term ([-+*/] term)*
term ::= num | "(" expr ")"
num  ::= [0-9]+
""")
```

This grammar accepts strings such as `1+2`, `(3*4)-5=7`, or `2=2=2`.

## Usage

```python
xgr.Grammar.from_ebnf(
    ebnf_string: str,
    *,
    root_rule_name: str = "root",
) -> xgr.Grammar
```

- `ebnf_string` is the grammar source.
- `root_rule_name` is the name of the entry rule. By default the grammar must define a rule named
  `root`; pass a different name to use another rule as the entry point.

The returned grammar is compiled and matched like any other grammar:

```python
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
compiler = xgr.GrammarCompiler(tokenizer_info)
compiled = compiler.compile_grammar(grammar)
matcher = xgr.GrammarMatcher(compiled)
```

`compile_grammar` also accepts the EBNF string directly, in which case the string is used as the
cache key of the compiler's cache:

```python
compiled = compiler.compile_grammar('root ::= "a" | "b"')
```

Errors in the grammar raise `RuntimeError` with the line and the column of the offending
construct:

```text
EBNF parser error at line 1, column 10: Rule "a" is not defined
```

## Grammar Structure

A grammar is a sequence of rule definitions:

```text
name ::= expression
```

- A rule name must be the first token on its line. Apart from this requirement, whitespace and
  line breaks are insignificant, so a rule body may span multiple lines:

```text
root ::= "a" |
         "b" |
         "c"
```

- Names consist of letters, digits, underscores (`_`), hyphens (`-`), and dots (`.`), and cannot
  start with a digit. Unlike Lark, there is no case convention: EBNF has no separate terminal
  concept, and every definition is a rule.
- Comments start with `#` and run to the end of the line.
- Each rule name may be defined only once. Rules may reference each other in any order, including
  forward references. Both right recursion and left recursion are supported:

```text
root ::= root "a" | "a"     # left recursion: accepts "a", "aa", "aaa", ...
```

## Elements

### String Literals

String literals use double quotes: `"hello"`. Matching is case-sensitive and exact. The empty
literal `""` matches the empty string, which is useful for making an alternative optional:

```text
sign ::= "" | "+" | "-"
```

Supported escape sequences:

| Escape | Meaning |
| --- | --- |
| `\"` `\\` `\/` `\'` `\?` | the character itself |
| `\n` `\r` `\t` `\b` `\f` `\a` `\v` `\0` `\e` | control characters (`\e` is ESC, `\0` is NUL) |
| `\xHH...` | code point from hex digits (arbitrary length, e.g. `\x41`, `\x1F600`) |
| `\uXXXX` | code point from exactly 4 hex digits |
| `\UXXXXXXXX` | code point from exactly 8 hex digits |

Non-ASCII characters may also be written directly: `"中文"`, `"😀"`. A literal cannot contain a
raw newline; write `\n` instead.

### Character Classes

`[...]` matches exactly one character, like in regular expressions:

- `[abc]` matches `a`, `b`, or `c`.
- `[a-zA-Z0-9]` matches one character from any of the ranges.
- `[^0-9]` is negated: it matches any character except a digit.
- `[^]` matches any character (nothing is excluded), including newline.

Characters and range endpoints may be any Unicode character (`[\u4e00-\u9fff]` matches one common
CJK character) and may use the same escapes as string literals. Additionally, the characters
`^ $ \ . * + ? ( ) [ ] { } | / -` can be escaped with a backslash; escape `-` as `\-` when it
should be a literal dash rather than a range separator. A raw newline is not allowed inside a
character class; write `[\n]`.

Regex-style shorthand classes such as `\d`, `\w`, `\s` are **not** supported inside character
classes and are rejected. Write the ranges explicitly (`[0-9]`, `[a-zA-Z0-9_]`, `[ \t\n\r]`).

### Rule References

Using a rule name inside an expression matches that rule:

```text
root  ::= value ("," value)*
value ::= [0-9]+
```

Referencing an undefined rule is an error.

### Sequences, Alternatives, and Groups

| Form | Example | Meaning |
| --- | --- | --- |
| Sequence | `"a" "b"` | Match the elements in order. |
| Alternative | `"a" \| "b"` | Match any one branch. Each branch is a sequence. |
| Group | `("a" \| "b") "c"` | Group a sub-expression; may carry repetition. |
| Empty group | `( )` | Matches the empty string. |

## Repetition

A repetition operator follows an element — a string literal, a character class, a rule reference,
or a parenthesized group:

| Form | Meaning |
| --- | --- |
| `x?` | zero or one |
| `x*` | zero or more |
| `x+` | one or more |
| `x{3}` | exactly 3 |
| `x{2,4}` | 2 to 4, inclusive |
| `x{2,}` | 2 or more |

The lower bound is required: `x{,4}` is not accepted (use `x{0,4}`). `x{0}` is allowed and
matches the empty string. Two operators cannot be applied directly to the same element (`"a"++`
is an error); wrap the element in parentheses instead: `("a"+)+`.

```text
root       ::= identifier ("," identifier){0,4}
identifier ::= [a-zA-Z_] [a-zA-Z0-9_]*
```

## TagDispatch

`TagDispatch` is a macro for the common tool-calling pattern: the model produces free text until
a trigger tag appears, then the output must follow the grammar associated with that tag, and
afterwards the model returns to free text.

```text
root ::= TagDispatch(
    ("<tool_call>", tool_call_body),
    ("<code>", code_body),
    loop_after_dispatch=true,
    excludes=()
)
tool_call_body ::= "{" [^}]* "}"
code_body ::= "```" [^`]* "```"
```

Positional arguments are `("tag_string", rule_name)` pairs. The semantics:

- Any text that does not contain a tag is accepted, and the match may end at any point in this
  free-text state. An incomplete tag prefix at the end of the output (for example a final
  `<tool_ca`) also counts as free text.
- When a complete tag appears, matching dispatches to the corresponding rule, and the rule must
  be completed.
- After the rule completes, if `loop_after_dispatch` is `true` (the default), matching returns to
  the free-text state and more dispatches may follow. If `false`, the `TagDispatch` ends
  immediately after the first dispatched rule completes.

With the grammar above, all of the following are accepted:

```text
I will call a tool. <tool_call>{"name": "get_weather"}</tool_call> Done.
no tool call at all
<code>```print(1)```<tool_call>{}
```

`excludes` lists strings that must not appear in the free-text portion. This is typically used
together with an outer stop tag, so that the stop tag reliably terminates the dispatch loop
instead of being swallowed as free text:

```text
root ::= body "</answer>"
body ::= TagDispatch(
    ("<tool_call>", tool_call_body),
    excludes=("</answer>",)
)
tool_call_body ::= "{" [^}]* "}"
```

An excluded string must not be a prefix of any trigger tag.

## Token-Level Elements

The following macros match **tokens by their IDs in the model's vocabulary** rather than text.
They are useful when the pattern involves special tokens (such as `<|python_start|>`) that the
tokenizer encodes as single tokens.

### Token and ExcludeToken

`Token(id, ...)` matches exactly one token whose ID is in the given set. `ExcludeToken(id, ...)`
matches exactly one token whose ID is **not** in the given set. IDs must be non-negative
integers; duplicates are merged.

```text
root ::= Token(128010) content
content ::= [a-z]+
```

Token-level and character-level elements can be mixed freely in one rule.

### TokenTagDispatch

`TokenTagDispatch` is the token-level analog of `TagDispatch`: free tokens instead of free text,
and each trigger is a single token ID.

```text
root ::= TokenTagDispatch(
    (128010, tool_call_body),
    loop_after_dispatch=true,
    excludes=(128020,)
)
tool_call_body ::= "{" [^}]* "}"
```

- Before a trigger token appears, any token is accepted except those listed in `excludes`.
- When a trigger token is produced, matching dispatches to the corresponding rule; after the rule
  completes, `loop_after_dispatch` behaves as in `TagDispatch`.
- Trigger token IDs must not overlap with `excludes`.

## A Complete Example

A grammar for JSON-style quoted strings, showing literals, character classes, repetition, and
recursion working together:

```python
import xgrammar as xgr

grammar = xgr.Grammar.from_ebnf(r"""
root   ::= "\"" char* "\""
char   ::= [^"\\\x00-\x1f] | "\\" escape
escape ::= ["\\/bfnrt] | "u" [0-9A-Fa-f]{4}
""")
```

This accepts `"hello"`, `"line1\nline2"`, and `"\u00e9"`, and rejects unterminated or badly
escaped strings.

As a larger example, the grammar below describes free-form JSON:

```text
root ::= basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= (([\"] basic_string_1 [\"]))
basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
ws ::= [ \n\t]*
```

## Relation to Other Grammar Formats

- `str(grammar)` prints any grammar in this EBNF format. The printed form is normalized (for
  example, `*` and `+` are expanded into recursive rules) and can be parsed again by `from_ebnf`.
- [`xgr.Grammar.from_json_schema`](xgrammar.Grammar.from_json_schema) and
  [`xgr.Grammar.from_regex`](xgrammar.Grammar.from_regex) convert their inputs to EBNF
  internally; pass `print_converted_ebnf=True` to inspect the result.
- [`xgr.Grammar.union`](xgrammar.Grammar.union) and
  [`xgr.Grammar.concat`](xgrammar.Grammar.concat) combine grammars from any frontend, including
  EBNF.
- The [Lark frontend](lark_grammar.md) offers an alternative notation with terminals, `%ignore`,
  and inline JSON Schema, and converts to the same internal representation.
