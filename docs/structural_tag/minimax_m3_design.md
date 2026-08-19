# MiniMax M3 Structural-Tag Design

## 1. Goals

MiniMax M3 recursively serializes tool arguments as namespace-prefixed XML-like elements. A JSON
property name becomes both the opening and closing element name:

```text
]<]minimax[>[<shipping>
]<]minimax[>[<city>Singapore]<]minimax[>[</city>
]<]minimax[>[</shipping>
```

The implementation must provide all of the following at the same time:

- exact byte-for-byte equality between a runtime-generated opening name and its closing name;
- normal JSON Schema constraints for values and object/array structure;
- independent semantics for every occurrence, including nested and sibling elements;
- correct grammar union, concatenation, optimization, token masking, rollback, fork, batch
  matching, jump-forward, and EOS behavior;
- lossless `Grammar` and `CompiledGrammar` JSON serialization;
- immutable compiled state that can be shared across threads;
- low overhead on the token-generation hot path.

The supported public path is:

```python
structural_tag = xgr.get_model_structural_tag("minimax_m3", ...)
compiled = xgr.GrammarCompiler(tokenizer_info).compile_structural_tag(structural_tag)
matcher = xgr.GrammarMatcher(compiled)
```

## 2. Why a plain CFG or a stack is insufficient

For an unbounded runtime name, the required sublanguage has the shape:

```text
open_prefix name open_suffix content close_prefix name close_suffix
```

A CFG can constrain both name positions to the same regular language, but it cannot require the
two arbitrary strings to be equal in the same byte order. The copy language `{w#w}` is not
context-free.

A single stack does not solve that equality either. If bytes of `abc` are pushed while reading the
opening name, popping checks `cba`, not `abc`. A stack is useful for nesting already parsed element
occurrences, but the same-order name check still needs either a captured byte span plus a forward
cursor, or an equivalent queue/persistent-string representation.

The implementation therefore extends the grammar runtime with one narrowly scoped capture and
backreference operation. It does not add a protocol-specific parser beside the grammar.

## 3. Rejected architecture: a grammar-global sidecar

The first implementation attached a `DynamicTagMatcher` configuration to the whole grammar and
intersected every accepted byte with a second protocol state machine. That design was rejected for
three reasons.

First, its state was grammar-global instead of occurrence-local. For example, XGrammar supports
grammar composition:

```text
P = "plain"
M = DynamicTag("<", name, ">", content, "</", ">")
U = P | M
```

After the CFG has accepted `plain`, the `P` branch is complete and must be allowed to terminate.
A global MiniMax sidecar cannot know that the selected derivation never entered `M`; a partial
protocol-looking suffix in ordinary text can incorrectly keep the sidecar unfinished and mask
EOS. The same scope problem appears with concatenation, nested subgrammars, optional formats, and
multiple dynamic-tag dialects.

Second, grammar dumps did not naturally contain the full semantics. Separate metadata had to be
carried through every mutator, cache key, serializer, Web binding, and composition operation.

Third, token masking duplicated much of the grammar matcher's work and required a separate set of
protocol-specific indexes and rollback checkpoints.

The final design has no grammar-global dynamic matcher and no MiniMax-specific state machine in
`GrammarMatcher`.

## 4. Grammar-local `DynamicTag` IR

The grammar IR contains a serializable macro:

```text
DynamicTag(
  open_prefix,
  name_rule,
  open_suffix,
  content_rule,
  close_prefix,
  close_suffix
)
```

Its meaning is exactly:

```text
open_prefix + captured(name_rule) + open_suffix
+ content_rule
+ close_prefix + backreference(captured_name) + close_suffix
```

`kDynamicTag` is a normal `GrammarExprType`. The EBNF parser and printer, grammar builder,
visitors, mutators, dead-code elimination, rule-reference analysis, union, concatenation, and JSON
serializer all understand it. Structure normalization materializes a nested `DynamicTag` as a
helper rule, so each compiled occurrence has one independent rule-local capture scope.

This placement fixes the composition problem by construction. A derivation in an unrelated union
branch never enters the dynamic rule and therefore owns no unfinished dynamic state.

## 5. Compilation to the existing FSM/Earley runtime

Each `DynamicTag` occurrence is lowered into the rule FSM as:

```text
fixed open prefix
  -> CaptureStart               (zero-width)
  -> delimiter-safe name DFA
  -> CaptureEnd                 (zero-width)
  -> fixed open suffix
  -> RuleRef(content_rule)
  -> fixed close prefix
  -> BackReference              (one captured byte per scan step)
  -> fixed close suffix
```

`CaptureStart`, `CaptureEnd`, and `BackReference` are serializable `FSMEdge` types. They are part
of the compiled grammar itself; there is no out-of-band metadata to synchronize.

### 5.1 Name-rule requirements

`name_rule` is compiled and inlined as a byte-regular automaton. It may contain byte strings,
character classes, regular choices/sequences, bounded or unbounded repetition, acyclic rule
references, and direct tail recursion. The compiler rejects:

- token-dependent edges (`Token`, `ExcludeToken`, or EOS);
- another `DynamicTag` inside the name language;
- mutual recursion or non-tail recursion;
- rule metadata whose runtime meaning would be lost by inlining: token/character budgets,
  captures, lazy matching, temperature, or suffix/stop metadata.

Rule lookahead is allowed because it is a token-mask boundary hint, not part of the accepted byte
language. The fixed opening suffix supplies the boundary after inlining.

Name determinization, repetition expansion, delimiter construction, and intersection are each
bounded to 100,000 states. Compilation fails explicitly instead of risking unbounded memory use.

### 5.2 Delimiter-safe name language

The caller-supplied name language is intersected with a deterministic delimiter-safety automaton.
A name must be nonempty and contain a non-ASCII-whitespace byte. It cannot:

- contain the complete opening or closing suffix;
- overlap either suffix across the `name + suffix` boundary, creating an earlier capture point;
- start with the remaining extension when one of the opening/closing prefixes is a strict prefix
  of the other.

KMP prefix states make this check linear in delimiter length. The restriction gives every opening
tag an unambiguous capture boundary and prevents an adversarial permissive name rule from
accumulating one Earley state for every previous delimiter occurrence. It also matches the M3
wire requirement that `/` cannot begin a property element and `>` cannot appear in its name.

## 6. Matcher representation

The Earley parser already distinguishes a rule occurrence by `rule_id` and `rule_start_pos`.
Dynamic-tag progress is stored in the existing hot `ParserState` fields without increasing the
state size:

- in a dynamic FSM, `sub_element_id` and `partial_codepoint` store the parser rows at
  `CaptureStart` and `CaptureEnd`;
- while scanning `BackReference`, `repeat_count` stores the next captured-byte offset.

These uses cannot overlap with UTF-8 character-class decoding or grammar repetition in a
`DynamicTag` rule. State equality and hashing already include all three fields, so ambiguous
derivations remain independent.

Each `GrammarMatcher` keeps request-local:

- `accepted_bytes_`, the accepted byte history;
- `row_byte_end_`, mapping Earley input rows to byte offsets, including atomic-token rows;
- temporary bytes used only while evaluating a candidate token or string.

The captured span is the exact byte interval between the two marker rows. This avoids deriving a
byte boundary from `rule_start_pos`, whose only job remains tracking the Earley parent node.
`BackReference` reads that span forward. Nested elements naturally use their own rule occurrence
and parent-completion state; no separate global tag stack is required.

The history also makes all existing stateful operations exact:

- `Rollback` truncates parser rows and byte history together;
- copying/forking a matcher copies request-local history while sharing the compiled grammar;
- speculative token checks and jump-forward use temporary history and restore it before return;
- atomic-token and byte paths remap after-token rows before their states are merged;
- batch matching is independent because every batch item owns a separate matcher.

EOS is accepted only when the selected Earley derivation is complete. An incomplete backreference
is a scanable grammar state, so it cannot accidentally terminate.

## 7. Token-mask algorithm

A reusable token mask cannot know the runtime captured name. The compiler therefore builds two
masks only for rules that can reach a `DynamicTag`:

1. a baseline matcher treats an unresolved backreference as rejected;
2. a wildcard matcher treats it as one or more arbitrary bytes.

Their classification has safe monotonic meaning:

- baseline-accepted tokens are definitely accepted;
- wildcard-rejected tokens are definitely rejected;
- the remaining difference is runtime-dependent and is checked against the real matcher state.

Rules that cannot reach a dynamic tag keep the ordinary rule-level cache. Reverse rule-dependency
analysis marks parent rules too, which is necessary when one tokenizer token crosses an ordinary
parent-rule boundary and then enters or exits a dynamic tag.

At a state whose only scanable edge is `BackReference`, the matcher uses a faster path:

1. obtain the next required captured byte;
2. binary-slice the sorted vocabulary to tokens beginning with that byte;
3. validate only that range, reusing longest common prefixes and tokenizer-trie subtree ranges.

Stop and special tokens are cleared unless the ordinary grammar completion rules allow them. With
the local MiniMax tokenizer (`vocab_size=200,054`), measured closing-name mask times were:

| Captured name | Median mask time |
| --- | ---: |
| ASCII | 1.958 us |
| UTF-8 Chinese | 5.208 us |
| 200-byte key | 2.750 us |

These are microseconds per mask fill, not milliseconds per accepted token.

## 8. Serialization and compatibility

Both persistence paths are lossless:

- `Grammar.serialize_json()` stores the `kDynamicTag` IR and referenced rules;
- `CompiledGrammar.serialize_json()` stores the lowered FSM, including `CaptureStart`,
  `CaptureEnd`, and `BackReference` edges.

After load, no protocol metadata or tokenizer index needs to be reconstructed outside the normal
compiled grammar. Grammar print/parse round trips also preserve `DynamicTag(...)`.

The branch uses serialization version `v18`. The version was bumped because both the grammar IR
and compiled FSM format changed; `v17` dumps are intentionally rejected in both directions rather
than being misread as compatible. JSON dumps are supported; a consumer that intentionally flattens
the macro to plain CFG productions cannot preserve runtime same-order equality.

## 9. Thread safety and ownership

The concurrency boundary is the same as for ordinary XGrammar:

- optimized `Grammar`, compiled FSMs, and adaptive token-mask caches are immutable after
  compilation and may be shared by threads;
- every mutable capture, byte history, rollback state, and mask scratch buffer belongs to one
  `GrammarMatcher`;
- different matchers, including copies made from the same compiled grammar, may run concurrently;
- mutating one `GrammarMatcher` concurrently is unsupported.

Compilation writes adaptive masks through the existing compiler mutex when multiple compiler
threads are enabled. Runtime matching performs no process-global writes and uses no lazy shared
dynamic-tag cache.

ThreadSanitizer tests cover a shared compiled grammar and batch matchers with independent runtime
names.

## 10. MiniMax M3 JSON Schema conversion

M3 is a recursive dialect of the existing `XMLToolCallingConverter`, not a separate schema
converter. This reuses `$ref` resolution, schema traversal, caching, strict mode, property order,
cardinality, and the established XML-style extension points.

| JSON Schema value | M3 representation |
| --- | --- |
| Fixed object property `k` | fixed `<k>...</k>` literals |
| Dynamic/additional property | `DynamicTag` using the generated key rule |
| Nested object | nested property elements |
| Array | repeated fixed `<item>...</item>` elements |
| String | unquoted scalar text excluding the complete M3 namespace marker |
| Integer / number | JSON numeric lexical form |
| Boolean / null | `true`, `false`, or `null` |
| `const` / `enum` object or array | recursively expanded fixed elements |
| unconstrained value | scalar text or one-or-more recursively named child elements |

Known fixed property names use literals and pay no runtime equality cost. Only names that are
actually generated at runtime use `DynamicTag`. Duplicate dynamic keys that equal a known property
are excluded by a codepoint trie.

`propertyNames` is applied to runtime keys. An explicit `additionalProperties` schema controls the
corresponding value; explicit `false` forbids it. The converter preserves the established
strict-mode behavior in which `propertyNames` by itself enables names satisfying that schema.

An absent/unconstrained tool parameter schema is normalized to an object with
`additionalProperties: true`, because an M3 invocation serializes arguments as named elements.

### 10.1 Structural-tag envelope

The built-in `minimax_m3` builder supports:

- `reasoning_mode="enabled"`: continue a `<mm:think>` opener already emitted by the prompt;
- `reasoning_mode="disabled"`: no reasoning prefix;
- `reasoning_mode="auto"`: an optional complete `<mm:think>...</mm:think>` prefix;
- automatic, required, and named/forced function-tool choice;
- one or multiple invocations where the effective tool choice allows them.

Text exclusions contain complete protocol markers. The bare namespace marker is deliberately not
excluded because it is a multi-token prefix shared by valid M3 elements; excluding it would make a
longer tool-call trigger unreachable.

## 11. Performance

The design favors generation-time performance:

- fixed names remain ordinary literals;
- a dynamic name is determinized once at compile time;
- delimiter safety is compiled into the name DFA, not checked by a second runtime parser;
- the `ParserState` size is unchanged;
- only matchers whose grammar contains a dynamic tag retain byte history;
- pure backreference masks inspect one first-byte vocabulary range;
- ordinary rules retain the existing shared token-mask cache.

Using the local MiniMax tokenizer and a forced dynamic-property schema:

- compile median: `50.631 ms`;
- compiled heap accounting: `0.148 MB`;
- no dynamic-state data is allocated per vocabulary token at request time.

In a 10,000-byte plain-text benchmark comparing otherwise equivalent auto tool-call grammars:

| Metric | Fixed-only grammar | Dynamic-capable grammar |
| --- | ---: | ---: |
| Compile | 5.883 ms | 57.346 ms |
| Compiled memory | 0.080 MB | 0.190 MB |
| Accept | 28.83 ns/byte | 31.92 ns/byte |

The approximately 10.7% plain-text accept increment comes from matcher-local byte-history tracking,
which is required for exact rollback and future dynamic occurrences. Parser history remains the
dominant per-request storage.

All numbers above are medians from this checkout on the local development machine, using the
200,054-token MiniMax tokenizer, a 16-thread compiler with its grammar cache disabled, and 10,000
ASCII bytes for the accept comparison. They are intended as regression baselines rather than
cross-machine performance claims.

## 12. Verification

The test matrix covers:

- fixed, dynamic, nested, sibling, UTF-8, long, and delimiter-adjacent names;
- exact mismatched-close rejection and all 256 possible first bytes at the generic IR level;
- grammar union, concatenation, nested helper rules, and tokens crossing every boundary;
- name-language repetition, direct tail recursion, and rejection of unsafe grammars;
- tokenizer masks, stop/special tokens, rollback, fork, reset, jump-forward, and batch matching;
- grammar and compiled-grammar dump/load round trips;
- recursive objects, arrays, refs, literals, `propertyNames`, explicit additional-property policy,
  reasoning modes, and tool choices;
- concurrent use of one compiled grammar by independent matchers.

Verified results in this checkout:

- C++ test binary: `101 passed`;
- full Python suite: `3290 passed, 678 skipped`;
- focused ThreadSanitizer concurrency tests: `4 passed`, no TSAN report.

The Web/Wasm source path carries the same IR and FSM implementation. It remains unverified in this
environment because `emcc` is unavailable.

## 13. Deliberate support boundaries

1. The runtime-name rule must be byte-regular as described above. This is an explicit safety and
   compilation-complexity boundary, not a silent fallback.
2. Recursive XML string `pattern`, recognized `format`, and length constraints are rejected when
   they cannot be intersected with the reserved namespace-marker exclusion by the current
   converter. Unknown formats remain annotations.
3. The M3 mapping emits typed scalar-or-children values, not arbitrary mixed XML content.
4. The converter inherits broader XGrammar JSON Schema limits. In particular, overlapping
   `patternProperties` do not synthesize arbitrary schema intersections, and `any_order` is not a
   new full key-set automaton.
5. Names that cannot be represented safely as M3 element names are rejected rather than escaped
   into a wire format the model and downstream parser do not define.
6. Share `CompiledGrammar`, not a mutating `GrammarMatcher`, between concurrent execution streams.

Within these explicit boundaries, runtime name equality, grammar composition, persistence,
rollback, token masking, and concurrency are first-class grammar behavior rather than MiniMax-only
side effects.
