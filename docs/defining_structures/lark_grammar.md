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

`.` matches one Unicode character. By default it does not match newline. Regular expressions
support the following trailing flags, in any order:

- `i`: make the match ASCII case-insensitive. ASCII letters in literals and character classes
  match both cases; non-ASCII characters match literally.
- `s`: make `.` match newline as well.
- `u`: explicitly select Unicode semantics. This is a no-op because XGrammar regular expressions
  already use Unicode codepoints.

The `i` flag is supported in ordinary rules, terminals, and `lazy` rules, but not on a regular
expression used with a `suffix` or `stop` attribute. The `l`, `m`, and `x` flags are not supported.

Word boundaries (`\b`, `\B`), Unicode property escapes (`\p{…}`), backreferences, and lookaround
assertions are not supported. Large bounded repetitions such as `{0,10000}` are compiled through
the grammar-level repetition mechanism and do not expand the automaton.

```text
start: /a.b/      // accepts "acb", "a😀b"; rejects "a\nb"
line: /a.b/s      // also accepts "a\nb"
word: /Σk+/i      // accepts "Σk", "ΣKK"; only ASCII letters fold, "Σ" matches literally
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

## Parametric Rules

A rule can carry one unsigned 64-bit parameter. Declare it with `::_`, pass an initial value from
an ordinary rule, and use `%if` to enable alternatives based on the current value:

```text
start: permutation::0

permutation::_: "" %if is_ones([0:3])
               | "a" permutation::set_bit(0) %if bit_clear(0)
               | "b" permutation::set_bit(1) %if bit_clear(1)
               | "c" permutation::set_bit(2) %if bit_clear(2)
```

This grammar accepts every permutation of `a`, `b`, and `c` exactly once. Parameters are compile-time
state: `Grammar.from_lark` expands only reachable `(rule, value)` pairs into ordinary grammar rules.
The resulting `Grammar` uses the existing matcher, optimizer, EBNF printer, and JSON serialization
without a parameter-aware runtime.

Parameter values may be decimal or hexadecimal (`15`, `0xf`, or
`0xffffffffffffffff`). `_` means the current value. Bit slices use `[start:end]`, where `start` is
inclusive, `end` is exclusive, and valid indices cover the complete range `[0:64]`.

### Parameter Expressions

An expression after `rule::` computes the parameter passed to that rule:

| Expression | Result for current value `p` |
| --- | --- |
| `value` | The decimal or hexadecimal constant `value` |
| `_` | `p` |
| `set_bit(k)` | `p` with bit `k` set |
| `clear_bit(k)` | `p` with bit `k` cleared |
| `bit_or(value)` | `p \| value` |
| `bit_and(value)` | `p & value` |
| `incr([start:end])` | Increment the selected field, saturating when all its bits are set |
| `decr([start:end])` | Decrement the selected field, saturating when all its bits are clear |

For example, this rule accepts exactly five `item` occurrences:

```text
start: list::0
list::_: "item" list::incr([0:3]) %if lt([0:3], 5)
      | "" %if eq([0:3], 5)
```

### Conditions

`%if condition` applies to one alternative. A false condition removes that alternative for the
current rule instance.

| Condition | Meaning |
| --- | --- |
| `true` or `true()` | Always true |
| `bit_clear(k)`, `bit_set(k)` | Bit `k` is clear or set |
| `is_zeros(slice)`, `is_ones(slice)` | Every bit in the slice is clear or set |
| `eq(slice, value)`, `ne(slice, value)` | Unsigned equality or inequality |
| `lt(slice, value)`, `le(slice, value)` | Unsigned less-than or less-than-or-equal |
| `gt(slice, value)`, `ge(slice, value)` | Unsigned greater-than or greater-than-or-equal |
| `bit_count_eq(slice, k)`, `bit_count_ne(slice, k)` | Set-bit count equality or inequality |
| `bit_count_lt(slice, k)`, `bit_count_le(slice, k)` | Set-bit count comparison |
| `bit_count_gt(slice, k)`, `bit_count_ge(slice, k)` | Set-bit count comparison |
| `and(left, right)`, `or(left, right)`, `not(condition)` | Boolean composition |

The comparison value `k` in a set-bit-count condition is an unsigned 64-bit integer. Values above
the selected slice width are valid and follow ordinary mathematical comparison rules. For example,
`bit_count_lt(_, 65)` is always true, while `bit_count_eq(_, 65)` is always false.

Use `_` wherever a condition expects a slice to select all 64 bits:

```text
start: pick::0
pick::_: "" %if bit_count_ge(_, 1)
      | "a" pick::set_bit(0) %if and(bit_clear(0), bit_count_lt(_, 3))
      | "b" pick::set_bit(1) %if and(bit_clear(1), bit_count_lt(_, 3))
      | "c" pick::set_bit(2) %if and(bit_clear(2), bit_count_lt(_, 3))
```

Parameterized calls are only valid for parametric rules, and a parametric rule must use its current
parameter. Terminals cannot contain parameterized calls or `%if`. Stop-like behavior,
`temperature`, and `max_tokens` are not supported on parametric rules. The `capture` and
`max_chars` attributes are supported and apply to every invocation of the parametric rule.

Compilation is limited to 4096 reachable rule instances per Lark document. Here, "reachable" means
reachable from any ordinary (non-parametric) rule in that document, not only from `start`. This
preserves validation of ordinary helper rules even when `start` does not reference them. Grammars
whose state transitions exceed that limit are rejected with a located error instead of consuming
unbounded time or memory. Each nested `%lark` document has its own parameter namespace and instance
limit.

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
first lexeme of the output.

`allow_invalid_utf8` (boolean, default `false`) changes regular expressions in this grammar from
Unicode codepoints to individual bytes. It permits standalone bytes such as `0x80`; string
literals continue to match their exact UTF-8 encoded bytes:

```text
%grammar_options {"allow_invalid_utf8": true}
start: /[\x80-\xFF]+/ | "é"
```

In byte mode, `.` consumes exactly one byte and still excludes newline unless the `s` flag is
present. The `i`, `s`, and `u` flags remain available; case folding is ASCII-only. `\d`, `\w`, and
`\s` use their ASCII definitions, and uppercase forms complement within all 256 bytes. Unicode
escapes and properties, non-ASCII characters inside classes, word boundaries, lookarounds, and
backreferences are rejected.

The option applies only to the grammar that declares it. Nested `%lark` blocks and named grammars
use their own options. Multiple declarations merge monotonically, so a later `false` does not
disable an earlier `true`; unknown option names are rejected.

### `%json`

`%json { ... }` embeds a JSON Schema and behaves like a rule reference: the element matches any
JSON value conforming to the schema, converted through XGrammar's JSON Schema converter.

```text
start: "<tool_call>" arguments "</tool_call>"
arguments: %json {
  "type": "object",
  "properties": {"city": {"type": "string"}},
  "required": ["city"],
  "additionalProperties": false
}
```

`%json` may appear inside sequences, alternatives, and repetition. Whitespace outside the JSON
value is controlled by the surrounding Lark grammar; whitespace inside the value follows the JSON
Schema converter's normal behavior. `%json` cannot be used inside terminals.

### Substring

The substring extension matches any contiguous sequence of a fixed list of chunks, including the
empty sequence. For compatibility with llguidance, it is written with the `%regex { ... }`
directive, although it is not a regular expression:

```text
start: %regex {"substring_chunks": ["abc", "de", "fg"]}
```

This example accepts `""`, `"abc"`, `"de"`, `"abcde"`, `"defg"`, and `"abcdefg"`, but not
`"ab"` or `"cde"`.

`substring_chars` splits a string into Unicode codepoints before applying the same operation:

```text
start: %regex {"substring_chars": "abc"}
```

It therefore accepts every codepoint-aligned substring such as `"a"`, `"bc"`, and `"abc"`.
`substring_chunks` and `substring_chars` are supported in rules and terminal definitions.
`substring_words`, which requires Unicode word-class segmentation, is not yet supported.

### `%lark`

`%lark { ... }` embeds a complete Lark grammar as one element. The nested grammar has its own
independent namespace: it must define its own `start` rule, and it may declare its own imports,
`%ignore`, and `%grammar_options` without affecting the outer grammar. Rule names may be reused
across the boundary.

```text
start: "[" %lark {
  %import common.WS
  %ignore WS
  start: item ":" %json {"type": "integer"}
  item: "x" | "(" item ")"
} "]"
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

## Rule Options

Rule options are written as a comma-separated list between a rule name and its colon:

```text
reasoning[max_tokens=512, capture="reasoning"]: TEXT
name[lazy]: /[a-z]+/
field[suffix="!", stop_capture="marker"]: /[a-z]*/
value[temperature=0.7]: /[a-z]+/
```

They are supported on lowercase rules only; uppercase terminals cannot carry options. The
available options are:

- `lazy`: commit the rule to its
  [earliest possible completion](#general-lazy-rules-committed-shortest-matching).
- `suffix=marker`: end lazily at a marker and exclude the marker from this rule's own capture.
- `stop=marker`: end lazily at a marker and exclude the marker from this rule and all enclosing
  captures.
- `stop_capture="name"`: capture a `suffix` or `stop` marker separately. See
  [The `suffix` and `stop` Attributes](#the-suffix-and-stop-attributes).
- `max_tokens=N`: give every occurrence a [token budget](#token-budgets).
- `max_chars=N`: give every occurrence a [character budget](#character-budgets).
- `capture` or `capture="name"`: record the rule's matched span as a
  [capture group](#capture-groups).
- `temperature=N`: select the [sampling temperature](#sampling-temperature) while the rule is
  active.

`suffix` and `stop` cannot be used together, and `stop_capture` requires one of them. Other
supported combinations and their boundary behavior are described below.

### Sampling Temperature

The `temperature` rule attribute selects the sampling temperature while a terminal-like rule or
named subgrammar is active:

```python
grammar = xgr.Grammar.from_lark(
    """
    start: "answer:" value
    value[temperature=0.7]: /[a-z]+/
    """
)
compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
matcher = xgr.GrammarMatcher(compiled, default_temperature=0.2)
```

`temperature` must be a finite non-negative number. It is supported on rules that can be compiled
as terminals and on rules whose body is a `%json`, `%lark`, or `@name` subgrammar. An inner
explicit temperature overrides an inherited outer temperature. When ambiguous parse paths have
different active temperatures, `matcher.temperature` emits a warning once and returns the maximum.
If no active rule has a temperature, it returns `default_temperature`; if neither is configured,
it returns `None`.

`temperature` can be combined with `capture` and `max_chars`. Combining it with `max_tokens`,
`lazy`, `suffix`, or `stop` is not supported.

`BatchGrammarMatcher.batch_fill_temperature` fills a pre-allocated one-dimensional CPU `float32`
tensor with one temperature per matcher; the entry of a matcher without an effective temperature
is set to `-1`. For speculative decoding, pass a one-dimensional CPU `float32` tensor through the
`temperatures` argument of `GrammarMatcher.traverse_draft_tree`. The tensor receives one value per
tree node; `-1` marks a node with no configured temperature or a node that was not visited.

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
head[stop="<tool>"]: TEXT          // mask-equivalent; differs only for captures
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
trigger. Text that never produces the trigger is also accepted. The `suffix="..."` and
`stop="..."` forms can also be used this way and retain this efficient TagDispatch lowering.
The regex-suffix form is only accepted inside the full dispatch pattern.

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
  `foo[lazy]: /.*/`).
- Lazy rules are compiled as lexemes: `%ignore` is not woven inside their bodies, and like
  terminals they take the ignored-token skip after them.
- Each occurrence of the rule commits independently, and the commit is exact for validation
  (`accept_string`) as well as mask generation. `rollback`/`fork`/`reset` restore the state
  across a commit exactly.
- The same attribute is available in the EBNF frontend: `name[lazy] ::= ...`, and it round-trips
  through `Grammar.__str__()` / `Grammar.from_ebnf()`.

### The `suffix` and `stop` Attributes

`suffix` and `stop` both match the rule body lazily through the first occurrence of a marker. Like
llguidance, the marker may be a string literal, a regular expression, or an uppercase terminal
name:

```text
field[suffix="<end>"]: /[a-z]*/
field[suffix=/<\/?end>/]: /[a-z]*/
field[suffix=END]: /[a-z]*/
END: "</end>" | "<end>"
```

At the mask and validation levels, these forms are equivalent to appending the marker to a lazy
rule. A general body must itself be a terminal expression; strings, regexes (including bounded
repetition), and uppercase terminal composition are compiled together into one automaton. The
marker is required during ordinary validation, and committed-shortest matching prevents the body
from continuing through an earlier match. `%ignore` remains lexeme-scoped. For an any-text body
with a fixed, case-sensitive string marker, XGrammar retains the TagDispatch fast path: the rule
completes at the marker when one appears, while text that ends without a marker is also accepted.

The distinction between the two attributes is capture scope:

```text
start[capture="outer"]: field "z"
field[capture="inner", suffix="!"]: /[a-z]*/
# "xy!z" -> inner=b"xy", outer=b"xy!z"

start[capture="outer"]: field "z"
field[capture="inner", stop="!"]: /[a-z]*/
# "xy!z" -> inner=b"xy", outer=b"xyz"

start[capture="outer"]: field "z"
field[capture="inner", suffix=/!!+/, stop_capture="marker"]: /[a-z]*/
# "xy!!z" -> marker=b"!!", inner=b"xy", outer=b"xy!!z"
```

- `suffix` excludes the marker from the annotated rule's own capture, but enclosing captures
  retain it.
- `stop` excludes the marker from the annotated rule and every enclosing capture. This only
  changes capture materialization: the accepted marker bytes are not removed from the model
  context and the matcher performs no token rollback.
- A `stop` rule does not need its own `capture` attribute for the marker to be hidden from an
  enclosing capture. Multiple markers inside one enclosing capture are all removed.
- `stop_capture="name"` captures exactly the bytes matched by either `stop` or `suffix`, before
  the marker is hidden from other captures. It works even when the annotated rule has no
  `capture` attribute. Its name follows the same restrictions as `capture`.
- `suffix` and `stop` cannot be specified on the same rule, and `stop_capture` requires one of
  them.
- String flags and regex flags follow the same support as ordinary Lark terminals. In particular,
  ASCII case-insensitive string markers such as `suffix="END"i` and dot-all regex markers such as
  `stop=/BEGIN.*END/s` are supported.
- An empty string literal marker is not accepted; in particular, the llguidance EOS shorthand
  `stop=""` is not supported. A regex or named terminal marker may still be nullable.
- Adding an explicit `lazy` attribute is allowed but redundant.
- `max_tokens` and `max_chars` may annotate the same rule as `lazy`, `suffix`, or `stop`. The first
  boundary wins: a marker that completes first keeps the normal committed-shortest and capture
  behavior; if a length budget expires first and the body can end there, the rule completes
  without a marker (and therefore produces no `stop_capture`). If the body cannot end, for example
  in the middle of a multi-token marker, the budget is relaxed until a valid boundary is reached.

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
always stays grammar-valid. When authoring a budgeted rule, prefer a body that can end at every
possible budget boundary. For example, the arbitrary-text form above can end at any position, so
its budget is exact; bodies that cannot end where the budget runs out remain best-effort.

`max_tokens` composes with committed-shortest matching. With `lazy`, whichever of the lazy
completion and the token deadline is reached first closes the occurrence. With `suffix` or
`stop`, a deadline may close through the already matched body without fabricating marker bytes;
normal suffix/stop capture scope applies only when the marker was actually consumed.

The budget applies **per occurrence**: in `(r ",")* r` every element gets its own budget, and
to bound a whole loop the budget goes on a wrapper rule (`list[max_tokens=N]: item+`). Nested
budgets combine by taking the minimum. Rules inside a budgeted rule may also be used outside of
it — the budget follows the derivation, not the rule.

The first time a budget is exceeded (a token is consumed by a derivation past its budget), a
warning is logged, once per matcher. The budget state lives in the parser state, so
`rollback()` restores it exactly and speculative decoding keeps working. `accept_string`
advances without token boundaries and is not counted (budgets constrain mask-driven
generation, not validation/prefill).

`max_tokens` must be between 1 and 1,000,000, inclusive. It can be combined with `lazy`, `suffix`,
and `stop`, but cannot be used on terminals or on rules consumed by the dynamic dispatch pattern.

## Character Budgets

A rule can be limited by Unicode codepoints, the numeric values assigned to text units by the
Unicode standard, with `max_chars`:

```text
start: <think> reasoning </think> answer
reasoning[max_chars=2048]: TEXT
answer: /[0-9]+/
TEXT: /(\n|.)*/
```

For example, `a`, `é`, and `中` each consume one character. A decomposed character such as `e`
followed by a combining accent consumes two. The count does not depend on byte length in UTF-8, a
variable-width Unicode encoding, or on model token boundaries, the units generated by the model.
This also holds when one codepoint is split across multiple tokens.

Once the budget is exhausted, the matcher forces the occurrence to end if its body can end at the
current character boundary. Otherwise, the budget is relaxed until the earliest valid end, so the
output remains grammar-valid. When authoring a character-budgeted rule, prefer a body that can end
at every possible character boundary. For example, the arbitrary-text form above can end after
every codepoint, so its budget is exact; bodies that cannot end where the budget runs out remain
best-effort.

The first time a character budget is exceeded (a codepoint is consumed by a derivation past its
budget), a warning is logged, once per matcher. The budget applies per occurrence and nested
budgets take the minimum. `max_chars` can be combined with `temperature`. It can also be combined
with `max_tokens`, `lazy`, `suffix`, and `stop`; the first applicable length or marker boundary
wins. It applies to `accept_string`, unlike `max_tokens`. Rollback, reset, forking, and speculative
decoding restore the character count exactly.

`max_chars` must be between 0 and 2,147,483,647, inclusive. A zero budget closes the rule
immediately when its body can end at the entry position. It cannot be used on terminals. On a
dynamic dispatch start rule or a rule consumed by dynamic dispatch, it is ignored and a warning
is logged.

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
body, like `arg` above, work as expected. See the `suffix` and `stop` section for their marker
exclusion rules.
