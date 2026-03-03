# Plan: Expose `IsCompleted` Through All Layers

## Background

`IsCompleted()` currently lives only on the internal `EarleyParser` class (and
`GrammarMatcher::Impl` inherits it). It answers: **"Is the input accepted so far
a complete, valid string according to the grammar?"**

This is distinct from `IsTerminated()`, which means "the matcher is done and
will not accept more input":

| | `IsCompleted` | `IsTerminated` |
|---|---|---|
| Complete input, stop token **not** accepted | **true** | false (unless `terminate_without_stop_token`) |
| Complete input, stop token accepted | **true** | **true** |
| Incomplete input | false | false |

Users need `IsCompleted` to know whether the current output is already valid
*before* the stop token arrives (e.g. for speculative decoding, early stopping,
or streaming validation).

## Changes

### 1. C++ public header — `include/xgrammar/matcher.h`

Add next to `IsTerminated`:

```cpp
/*!
 * \brief Check if the grammar's root rule has been fully matched by the
 * input accepted so far. Unlike IsTerminated(), this does not require
 * the stop token to have been accepted.
 * \sa IsTerminated, AcceptToken
 */
bool IsCompleted() const;
```

### 2. C++ implementation — `cpp/grammar_matcher.cc`

Add the forwarding call next to the existing `IsTerminated` forwarding:

```cpp
bool GrammarMatcher::IsCompleted() const { return pimpl_->IsCompleted(); }
```

(`Impl` already inherits `EarleyParser::IsCompleted()`, so no new impl code is
needed.)

### 3. nanobind binding — `cpp/nanobind/nanobind.cc`

Add next to the `is_terminated` binding:

```cpp
.def("is_completed", &GrammarMatcher::IsCompleted)
```

### 4. Python wrapper — `python/xgrammar/matcher.py`

Add next to `is_terminated`:

```python
def is_completed(self) -> bool:
    """Check if the input accepted so far forms a complete valid string
    according to the grammar. Unlike ``is_terminated()``, this does not
    require the stop token to have been accepted.

    Returns
    -------
    completed : bool
        Whether the grammar's root rule is fully matched.
    """
    return self._handle.is_completed()
```

### 5. Web / JS binding — `web/src/xgrammar_binding.cc`

Add next to the `IsTerminated` binding:

```cpp
.function("IsCompleted", &GrammarMatcher::IsCompleted)
```

### 6. Web / TS wrapper — `web/src/xgrammar.ts`

Add next to `isTerminated()`:

```ts
/**
 * Check if the input accepted so far forms a complete valid string
 * according to the grammar. Unlike isTerminated(), this does not require
 * the stop token to have been accepted.
 */
isCompleted(): boolean {
  return this.handle.IsCompleted();
}
```

### 7. Python `__init__.py` — no change expected

`is_completed` is a method on `GrammarMatcher`, which is already exported.
Verify no re-export list needs updating.

### 8. Tests — `tests/python/test_grammar_matcher_basic.py`

Add a new test (next to `test_termination`) that exercises the semantic
difference:

```python
def test_is_completed():
    # Setup: same vocab / JSON grammar as test_termination
    # 1. Feed tokens for a complete JSON object (without stop token).
    #    - assert is_completed() == True
    #    - assert is_terminated() == False   (stop token not yet accepted)
    # 2. Accept stop token.
    #    - assert is_completed() == True
    #    - assert is_terminated() == True
    # 3. Rollback past the stop token to mid-parse.
    #    - assert is_completed() == False
    #    - assert is_terminated() == False
    # 4. Test with terminate_without_stop_token=True:
    #    - feed complete input (no stop token)
    #    - assert is_completed() == True
    #    - assert is_terminated() == True  (matches is_completed in this mode)
```

### 9. Web tests — `web/tests/grammar.test.ts`

Mirror the Python test: verify `isCompleted()` returns `true` before the stop
token is accepted, and `false` after rollback to mid-parse.

## File checklist

| # | File | Action |
|---|---|---|
| 1 | `include/xgrammar/matcher.h` | Add `IsCompleted()` declaration |
| 2 | `cpp/grammar_matcher.cc` | Add `GrammarMatcher::IsCompleted()` forwarding impl |
| 3 | `cpp/nanobind/nanobind.cc` | Bind `is_completed` |
| 4 | `python/xgrammar/matcher.py` | Add `is_completed()` method |
| 5 | `web/src/xgrammar_binding.cc` | Bind `IsCompleted` |
| 6 | `web/src/xgrammar.ts` | Add `isCompleted()` method |
| 7 | `tests/python/test_grammar_matcher_basic.py` | Add `test_is_completed` |
| 8 | `web/tests/grammar.test.ts` | Add `isCompleted` test |
