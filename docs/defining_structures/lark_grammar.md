# Lark Grammar

XGrammar can build grammars from a dialect of the
[Lark grammar language](https://lark-parser.readthedocs.io/en/latest/grammar.html) through
[`xgr.Grammar.from_lark`](xgrammar.Grammar.from_lark). Lark is a compact, readable notation for
describing structured text: fixed strings, alternatives, repetition, recursion, and regular
expressions. The result is a normal [`xgr.Grammar`](xgrammar.Grammar), so it works with the
existing compiler, matcher, serialization, `Grammar.union` / `Grammar.concat`, and all engine
integrations.

```python
import xgrammar as xgr

grammar = xgr.Grammar.from_lark(
    r"""
    %import common.INT
    %import common.WS
    %ignore WS

    start: item ("," item)*
    item: "id=" INT
    """
)
```

This grammar accepts strings such as `id=1`, `id=1, id=42`, or `id=1 ,id=2 , id=3`.

XGrammar's Lark dialect is compatible with the dialect used by
[llguidance](https://github.com/guidance-ai/llguidance).

## Usage

```python
xgr.Grammar.from_lark(
    lark_string: str,
    *,
    tokenizer_info: Optional[xgr.TokenizerInfo] = None,
    named_grammars: Optional[Dict[str, Union[xgr.Grammar, str]]] = None,
) -> xgr.Grammar
```

- `lark_string` is the grammar source. It must define a rule named `start`, which is the entry
  point of the grammar.
- `tokenizer_info` is only required when the grammar uses named special tokens (such as
  `<|tool_call|>`) or the all-token wildcard `<[*]>`. See [Special Tokens](#special-tokens).
- `named_grammars` supplies external grammars referenced with `@name` in the Lark source. See
  [Named Grammars](#named-grammars).

The returned grammar is compiled and matched like any other grammar:

```python
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
compiler = xgr.GrammarCompiler(tokenizer_info)
compiled = compiler.compile_grammar(xgr.Grammar.from_lark('start: "a" | "b"'))
matcher = xgr.GrammarMatcher(compiled)
```

Errors in the grammar raise `RuntimeError` with the line, the column, and the offending source
line:

```text
Lark error at line 3, column 6: expected ':' after rule name
item missing
     ^
```

## Grammar Structure

A grammar is a sequence of items separated by newlines. Each item is one of:

- a rule definition: `name: expression`
- a terminal definition: `NAME: expression`
- a directive: `%import`, `%ignore`, or `%grammar_options`

Comments start with `#` or `//` and run to the end of the line. Blank lines are ignored. An
alternative may continue on the next line when that line starts with `|`:

```text
start: "a"
     | "b"
     | "c"
```

### Rules and Terminals

Names consist of letters, digits, underscores, and hyphens. A definition whose name starts with a
lowercase letter (ignoring a leading underscore) is a **rule**; one that starts with an uppercase
letter is a **terminal**.

```text
start: value (";" value)*     // rule
value: INT | NAME             // rule referencing terminals
INT: /[0-9]+/                 // terminal
NAME: /[a-z]+/                // terminal
```

Rules may reference each other freely, including forward references, direct recursion, and
indirect recursion:

```text
start: value
value: "x" | "(" value ")" | "[" (value ("," value)*)? "]"
```

Terminals are matched as one indivisible unit:

- A terminal may be composed of strings, character ranges, regular expressions, repetition, and
  other terminals, but it cannot reference rules and cannot be recursive.
- Content skipped by `%ignore` is never inserted inside a terminal. In the first example above,
  with `%ignore WS`, spaces may appear around an `INT` but not between its digits.
- `%json`, `%lark`, special tokens, and `@name` references cannot appear inside terminals.

For compatibility with grammars written for parse-tree-producing parsers, XGrammar accepts and
ignores the rule prefixes `?` and `!` (as in `?value: ...`) and alternative aliases
(`"a" -> first`). These constructs only affect parse-tree shaping and have no effect on which
strings the grammar accepts.

### String Literals

String literals use double quotes and JSON escape syntax: `\"`, `\\`, `\/`, `\b`, `\f`, `\n`,
`\r`, `\t`, and `\uXXXX`. Non-ASCII characters may be written directly (`"中文"`, `"😀"`) or with
Unicode escapes (`"\u03bb"` matches `λ`).

A trailing `i` makes the literal case-insensitive: `"yes"i` matches `yes`, `YES`, `Yes`, and so
on. Case-insensitive literals currently support ASCII characters only; a case-insensitive literal
containing non-ASCII characters is rejected.

### Character Ranges

`"a".."z"` matches one character between the two endpoints, inclusive. Both endpoints must be
exactly one character and may be any Unicode character: `"α".."γ"` matches `α`, `β`, or `γ`. The
`i` flag is not allowed on range endpoints.

### Regular Expressions

`/pattern/` matches text against a regular expression. The pattern is compiled through XGrammar's
regex converter (the same engine as [`xgr.Grammar.from_regex`](xgrammar.Grammar.from_regex)) and
supports character classes, alternation, groups, repetition (`*`, `+`, `?`, `{m,n}`), and the
usual escapes. A `/` inside the pattern is written `\/`.

`.` matches one Unicode character. By default it does not match newline; adding the `s` flag
(`/pattern/s`) makes `.` match newline as well. `s` is currently the only supported regex flag.

```text
start: /a.b/      // accepts "acb", "a😀b"; rejects "a\nb"
line: /a.b/s      // also accepts "a\nb"
```

### Sequences, Alternatives, and Groups

| Form | Example | Meaning |
| --- | --- | --- |
| Sequence | `"a" "b"` | Match the elements in order. |
| Alternative | `"a" \| "b"` | Match any one branch. |
| Group | `("a" \| "b") "c"` | Group a sub-expression; may carry repetition. |
| Optional group | `["a" "b"]` | The whole group appears zero or one time. |
| Empty | `start:` or `start: \| "a"` or `""` | Matches the empty string. |

### Repetition

Repetition operators follow an element (a literal, a name, or a group):

| Form | Meaning |
| --- | --- |
| `x?` | zero or one |
| `x*` | zero or more |
| `x+` | one or more |
| `x~3` | exactly 3 |
| `x~2..4` | 2 to 4, inclusive |
| `x{3}` | exactly 3 |
| `x{2,4}` | 2 to 4, inclusive |
| `x{2,}` | 2 or more |
| `x{,4}` | 0 to 4 |

Zero counts such as `x{0}` are allowed and match the empty string. Ranges with the upper bound
below the lower bound are rejected.

## Directives

### `%import common`

XGrammar provides a built-in library of common terminals. `%import` brings one of them into scope
as a terminal definition:

```text
%import common.INT                 // defines INT
%import common.INT -> NUMBER       // defines NUMBER with INT's pattern
%import common (INT, WS, CNAME)    // multiple imports in one line
```

Imports may appear anywhere in the grammar, including after the first use of the imported name.
Importing a name that is already defined is an error. The available names:

| Category | Names |
| --- | --- |
| Numbers | `DIGIT`, `HEXDIGIT`, `INT`, `SIGNED_INT`, `DECIMAL`, `_EXP`, `FLOAT`, `SIGNED_FLOAT`, `NUMBER`, `SIGNED_NUMBER` |
| Strings and names | `ESCAPED_STRING`, `LCASE_LETTER`, `UCASE_LETTER`, `LETTER`, `WORD`, `CNAME` |
| Whitespace | `WS_INLINE`, `WS`, `CR`, `LF`, `NEWLINE` |
| Comments | `SH_COMMENT`, `CPP_COMMENT`, `C_COMMENT`, `SQL_COMMENT` |

Only the `common` library can be imported.

### `%ignore`

`%ignore` declares content that may appear between terminals, typically whitespace:

```text
%import common.WS
%ignore WS
start: "a" DIGIT
DIGIT: "0".."9"
```

This accepts `a1`, `a 1`, and `a\n1  `. The ignored content:

- may appear between any two lexemes (terminals, string literals, character ranges, regexes) in a
  rule, and after the last one;
- may **not** appear before the first lexeme, unless `allow_initial_skip` is enabled (see below);
- is never inserted inside a terminal.

The `%ignore` expression may be a terminal name, a string, a regex, or a combination. Multiple
`%ignore` declarations are merged:

```text
%import common (WS, CPP_COMMENT)
%ignore WS
%ignore CPP_COMMENT
%ignore /;+/
```

### `%grammar_options`

`%grammar_options` takes a JSON object that configures the whole grammar:

```text
%grammar_options {"allow_initial_skip": true}
```

`allow_initial_skip` (boolean, default `false`) allows `%ignore` content to appear before the
first lexeme of the output. Multiple `%grammar_options` declarations are merged; unknown option
names are rejected.

## Inline JSON Schema

`%json { ... }` embeds a JSON Schema and behaves like a rule reference: the element matches any
JSON value conforming to the schema, converted through XGrammar's JSON Schema converter.

```python
grammar = xgr.Grammar.from_lark(
    r"""
    start: "<tool_call>" arguments "</tool_call>"
    arguments: %json {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"],
      "additionalProperties": false
    }
    """
)
```

`%json` may appear inside sequences, alternatives, and repetition. Whitespace outside the JSON
value is controlled by the surrounding Lark grammar; whitespace inside the value follows the JSON
Schema converter's normal behavior. `%json` cannot be used inside terminals.

## Nested Grammars

`%lark { ... }` embeds a complete Lark grammar as one element. The nested grammar has its own
independent namespace: it must define its own `start` rule, and it may declare its own imports,
`%ignore`, and `%grammar_options` without affecting the outer grammar. Rule names may be reused
across the boundary.

```python
grammar = xgr.Grammar.from_lark(
    r"""
    start: "[" %lark {
      %import common.WS
      %ignore WS
      start: item ":" %json {"type": "integer"}
      item: "x" | "(" item ")"
    } "]"
    """
)
```

Multiple `%lark` blocks may appear in the same rule. Nested grammars may use every feature of the
top-level grammar, including further nesting and `@name` references.

## Special Tokens

Special-token elements match exactly one token from the model's vocabulary, rather than text.
They may only be used in rules, not in terminals.

**Numeric token sets** reference tokens by ID and do not require tokenizer metadata (except for
the wildcard):

```text
start: <[128010]>            // exactly token 128010
start: <[128000-128255]>     // one token in the inclusive range
start: <[1,3-8,10]>          // union of IDs and ranges (duplicates are merged)
start: <[^1,3-8]>            // any token NOT in the set
start: <[*]>                 // any token (requires tokenizer_info)
```

A range whose endpoints differ by more than 1,000,000 is rejected, as is the negated wildcard
`<[^*]>`.

**Named special tokens** are resolved against the tokenizer's decoded vocabulary by exact string
match and therefore require `tokenizer_info`:

```python
tokenizer_info = xgr.TokenizerInfo(["a", "<|tool|>", "b"])
grammar = xgr.Grammar.from_lark("start: <|tool|>", tokenizer_info=tokenizer_info)
```

The reference matches every vocabulary entry whose decoded text equals the written form,
including the angle brackets. A name that matches no vocabulary entry is an error.

## Named Grammars

`named_grammars` passes external grammars into the Lark source, referenced with `@name`.
Dictionary keys do not include the leading `@`; values are either `Grammar` objects or Lark
source strings.

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

- Names may contain letters, digits, underscores, and hyphens, and must be unique.
- String values are complete Lark grammars with their own `start` rule, their own terminals, and
  their own `%ignore` declarations. They may reference other entries of the same mapping with
  `@name`; circular references are reported as errors with the reference chain.
- The same named grammar may be referenced multiple times and from inside nested `%lark` blocks.
  Each named grammar is compiled once and shared.
- `@name` references may only appear in rules, not in terminals.

## Dynamic Tool-Call Dispatch

A common pattern for tool calling lets the model produce free text until a trigger string (such
as `<tool_call>`) appears, then switches to a strict argument grammar, and returns to free text
after the call completes. XGrammar recognizes this pattern and compiles it into an efficient
token-level dispatch structure:

```python
grammar = xgr.Grammar.from_lark(
    r"""
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
    """
)
```

This accepts plain text with no tool call, one tool call, or several tool calls separated by free
text. A partial trigger (for example a final `<tool_cal`) still counts as free text. Once the
complete trigger has been produced, the payload and the end tag become mandatory.

### The Recognized Pattern

The `start` rule must have the shape

```text
start: tool* tail            // or: start: (tool_a | tool_b | ...)* tail
tail: TEXT
```

where `TEXT` is an **any-text terminal**: a terminal whose body is one of the regexes
`/(.|\n)*/`, `/(\n|.)*/`, `/(?:.|\n)*/`, `/(?:\n|.)*/`, `/[\s\S]*/`, `/(?s:.*)/`, or `/.*/s`
(possibly through another terminal name).

Each tool rule starts with a trigger and continues with an ordinary grammar for the payload. The
trigger is written in one of these equivalent head forms:

```text
head[lazy]: TEXT "<tool>"          // any text, then a fixed trigger string
head[lazy]: TEXT <|tool_token|>    // any text, then a special token (requires tokenizer_info)
head[lazy]: /(\n|.)*<tool>/        // the same trigger written as a regex suffix
head[suffix="<tool>"]: TEXT        // the same trigger written as a suffix attribute
tool: TEXT <|tool_token|> ...      // token trigger written inline, without a head rule
```

For the regex-suffix form, the pattern must be one of the any-text regexes followed by a fixed
literal (escapes such as `\n` or `\.` are allowed; alternation, repetition, and other variable
constructs are not).

### Multiple Tools

Different tools may use different triggers, or share one trigger and differentiate on the text
that follows:

```python
grammar = xgr.Grammar.from_lark(
    r"""
    start: (foo | bar)* tail
    tail: TEXT

    foo_head[lazy]: TEXT "<function"
    foo: foo_head "=foo>" /[a-z]+/ "</function>"

    bar_head[lazy]: TEXT "<function"
    bar: bar_head "=bar>" /[0-9]+/ "</function>"

    TEXT: /(\n|.)*/
    """
)
```

After `<function` is produced, the output must continue with `=foo>` or `=bar>` and the matching
payload. All triggers within one grammar must be at the same level: either all trigger strings or
all special tokens. Negated token sets cannot be used as triggers.

### Standalone Lazy Rules

Outside of the dispatch pattern, a lazy head of the form `head[lazy]: TEXT "<end>"` (or with a
special-token trigger) may also be used on its own:

```text
start: head
head[lazy]: TEXT "<end>"
TEXT: /(\n|.)*/
```

This matches arbitrary text and completes as soon as the trigger appears; nothing may follow the
trigger. Text that never produces the trigger is also accepted. The regex-suffix and
`suffix="..."` head forms are only accepted inside the full dispatch pattern.

### General Lazy Rules (Committed-Shortest Matching)

Any other rule may also carry `[lazy]`, which gives it **committed-shortest** matching: at the
first position where the rule's body can end, it must end — the derivations in which this
occurrence keeps consuming input are discarded.

```text
start: "<" name ">" rest
name[lazy]: /[a-z]+/     // stops at the first position where it can end
rest: /[a-z]+/
```

Notes:

- The body must compile to a single terminal-like automaton: sequences and alternations of
  strings and character classes, and the `+`/`*` quantifiers over single-character elements
  (character classes, single-character strings, and alternations of these — directly or through
  terminal references like `TEXT*`). Bodies that need rule references (recursion, `?` in the
  middle of a sequence, quantifiers over multi-character strings, and repetition ranges like
  `{2,5}`) are rejected at compile time.
- A lazy rule that can match the empty string always matches the empty string (for example
  `foo[lazy]: /.*/`); the compiler emits a warning for this.
- Lazy rules are compiled as lexemes: `%ignore` is not woven inside their bodies, and like
  terminals they take the ignored-token skip after them.
- Each occurrence of the rule commits independently, and the commit is exact for validation
  (`accept_string`) as well as mask generation. `rollback`/`fork`/`reset` restore the state
  across a commit exactly.
- The same attribute is available in the EBNF frontend: `name[lazy] ::= ...`, and it round-trips
  through `Grammar.__str__()` / `Grammar.from_ebnf()`.

## Token Budgets

A rule can be given a token budget with the `max_tokens` attribute:

```text
start: <think> reasoning </think> answer
reasoning[max_tokens=512]: TEXT
answer: /[0-9]+/
TEXT: /(\n|.)*/
```

Each occurrence of the rule may then consume at most `max_tokens` LLM tokens. Once the budget
is exhausted, the token mask only allows leaving the rule, which bounds the length of free-text
segments such as reasoning blocks while the rest of the output stays grammar-constrained.

The budget is enforced by the matcher at generation time. The body compiles normally and
every predicted occurrence of the rule carries a deadline: the index of the last token its
derivation may consume. Once the deadline passes, each mask forces the rule to end if ending
is possible at the current position; otherwise the budget is relaxed for one step and
enforcement is retried, so the rule ends at the earliest possible position and the output
always stays grammar-valid. Bodies that can end at any position — such as the arbitrary-text
form above — therefore never exceed their budget. For other bodies (e.g. `/(\S*\s)+/`) the
budget is best-effort and a compile-time warning marks the rule.

The budget applies **per occurrence**: in `(r ",")* r` every element gets its own budget, and
to bound a whole loop the budget goes on a wrapper rule (`list[max_tokens=N]: item+`). Nested
budgets combine by taking the minimum. Rules inside a budgeted rule may also be used outside of
it — the budget follows the derivation, not the rule.

The first time a budget is exceeded (a token is consumed by a derivation past its budget), a
warning is logged, once per matcher. The budget state lives in the parser state, so
`rollback()` restores it exactly and speculative decoding keeps working. `accept_string`
advances without token boundaries and is not counted (budgets constrain mask-driven
generation, not validation/prefill).

`max_tokens` must be positive and cannot be combined with `lazy` or `suffix`, used on
terminals, or on rules consumed by the dynamic dispatch pattern.

## Capture Groups

A rule can be marked with the `capture` attribute so that the matcher records the input span the
rule matched:

```text
start: tool* tail
tail: TEXT

tool_head[lazy]: TEXT "<tool_call>"
tool: tool_head arg "</tool_call>"
arg[capture]: /[0-9]+/

TEXT: /(\n|.)*/
```

`rule[capture]` uses the rule name as the capture name; `rule[capture="name"]` sets an explicit
name. Capture names may contain letters, digits, `_`, `-` and `.`. The recorded captures are
retrieved from the matcher:

```python
matcher = xgr.GrammarMatcher(compiled)
matcher.accept_string('x<tool_call>42</tool_call>y<tool_call>7</tool_call>')
matcher.get_captures()  # [("arg", b"42"), ("arg", b"7")]
```

Each completion of a captured rule records one capture, so a rule matched repeatedly (for
example inside a loop or a dispatch pattern) yields one entry per match, in completion order.
Captures are recorded when tokens or strings are accepted; `fill_next_token_bitmask` never
records anything, and `rollback` also rolls back the recorded captures.

Since the parser explores parse hypotheses in parallel, one occurrence of a captured rule may
complete at several candidate end positions (a `/[0-9]+/` body completes after every digit). By
default `get_captures` keeps only the longest completion of each occurrence, which is exact
whenever the captured rule's end is determined by a following delimiter that its body cannot
match (closing tags, quotes, brackets). If the following context can also be matched by the
rule body itself, the reported span may extend past the span of the finally accepted parse;
`get_captures(deduplicate=False)` returns the raw completion events instead.

Captures are supported on rules only (not terminals), and not on rules consumed by the dynamic
dispatch pattern (the head, tool and tail rules themselves); rules referenced from a tool's
body, like `arg` above, work as expected.
