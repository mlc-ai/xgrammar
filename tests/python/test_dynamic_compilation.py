"""Dynamic compilation: masks, caches, serialization, and matcher behavior."""

import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pytest
import torch

import xgrammar as xgr
from xgrammar.testing import _get_allow_empty_rule_ids, bitmask_to_bool_mask

# Include multi-character tokens, grammar-boundary crossings, and split UTF-8.
VOCAB = [bytes([i]) for i in range(256)] + [
    b"hello",
    b"ab",
    b"abc",
    b"a>",
    b"ab>",
    b'":',
    b'"}',
    b"12",
    b"</s>",
    "中".encode(),
    "中文>".encode(),
]
STOP = VOCAB.index(b"</s>")


def mask(matcher):
    result = xgr.allocate_token_bitmask(1, len(VOCAB))
    matcher.fill_next_token_bitmask(result)
    return result


BYTE_GRAMMAR_CASES = [
    ('start: "hello" ("!" | "?")', b"hello!"),
    ('start: "(" start ")" | "ab"', b"((ab))"),
    ('start: /[a-z]{2,4}/ "!"', b"abc!"),
    ('start: "中" "!"', "中!".encode()),
    ('start: value ">"\nvalue[max_chars=2]: /[a-z]*/', b"ab>"),
]


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("source,text", BYTE_GRAMMAR_CASES)
def test_masks_match_eager_and_token_acceptance(source, text, cache_enabled):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    grammar = xgr.Grammar.from_lark(source, tokenizer_info=info)
    eager = xgr.GrammarCompiler(info, cache_enabled=cache_enabled).compile_grammar(grammar)
    compiler = xgr.GrammarCompiler(
        info, cache_enabled=cache_enabled, enable_dynamic_compilation=True
    )
    dynamic = compiler.compile_grammar(grammar)
    # The compiled grammar must own everything required after the compiler dies.
    compiler.clear_cache()
    del compiler
    gc.collect()
    matchers = [xgr.GrammarMatcher(g) for g in (eager, dynamic)]
    for offset in range(len(text) + 1):
        actual = mask(matchers[1])
        torch.testing.assert_close(actual, mask(matchers[0]), rtol=0, atol=0)
        # Independent of mask construction: replay the prefix and try every token.
        for token_id in range(len(VOCAB)):
            oracle = xgr.GrammarMatcher(eager)
            for byte in text[:offset]:
                assert oracle.accept_token(byte)
            allowed = bool((int(actual[0, token_id // 32]) >> (token_id % 32)) & 1)
            assert allowed == oracle.accept_token(token_id), (offset, token_id)
        if offset < len(text):
            for matcher in matchers:
                assert matcher.accept_token(text[offset])
    for matcher in matchers:
        assert matcher.accept_token(STOP)
        assert matcher.is_terminated()


@pytest.mark.parametrize("dynamic", [False, True])
def test_rollback_reset_and_repeated_masks(dynamic):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    compiled = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic).compile_regex("ab|ac")
    matcher = xgr.GrammarMatcher(compiled)
    initial = mask(matcher)
    assert matcher.accept_token(ord("a"))
    after_a = mask(matcher)
    size = compiled.memory_size_bytes
    for _ in range(5):
        torch.testing.assert_close(mask(matcher), after_a, rtol=0, atol=0)
        assert compiled.memory_size_bytes == size
    assert matcher.accept_token(ord("b"))
    mask(matcher)
    matcher.rollback(1)
    torch.testing.assert_close(mask(matcher), after_a, rtol=0, atol=0)
    assert matcher.accept_token(ord("c"))
    assert matcher.accept_token(STOP)
    matcher.reset()
    torch.testing.assert_close(mask(matcher), initial, rtol=0, atol=0)


@pytest.mark.parametrize("warm", [False, True])
def test_serialization_and_memory_during_lazy_generation(warm):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    compiled = xgr.GrammarCompiler(info, enable_dynamic_compilation=True).compile_regex("ab|ac")
    eager = xgr.GrammarCompiler(info).compile_regex("ab|ac")
    expected = mask(xgr.GrammarMatcher(eager))
    if warm:
        mask(xgr.GrammarMatcher(compiled))

    def exercise(index):
        if index % 3 == 0:
            restored = xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)
            return mask(xgr.GrammarMatcher(restored))
        assert compiled.memory_size_bytes > 0
        return mask(xgr.GrammarMatcher(compiled))

    with ThreadPoolExecutor(max_workers=8) as pool:
        for actual in pool.map(exercise, range(32)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_default_is_eager_and_empty_vocab_serializes():
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    default = xgr.GrammarCompiler(info).compile_regex("hello|world")
    explicit = xgr.GrammarCompiler(info, enable_dynamic_compilation=False).compile_regex(
        "hello|world"
    )
    assert default.memory_size_bytes == explicit.memory_size_bytes
    size = default.memory_size_bytes
    mask(xgr.GrammarMatcher(default))
    assert default.memory_size_bytes == size
    empty_info = xgr.TokenizerInfo([])
    empty = xgr.GrammarCompiler(empty_info, enable_dynamic_compilation=True).compile_regex("a")
    restored = xgr.CompiledGrammar.deserialize_json(empty.serialize_json(), empty_info)
    assert restored.grammar.serialize_json() == empty.grammar.serialize_json()


@pytest.mark.parametrize("dynamic", [False, True])
def test_token_budget_enforced_by_masks(dynamic):
    # max_tokens enforcement is committed by FillNextTokenBitmask. Bare token
    # acceptance may relax the budget, so it is not an oracle for this feature.
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    grammar = xgr.Grammar.from_lark(
        'start: value ">"\nvalue[max_tokens=2]: /[a-z]+/', tokenizer_info=info
    )
    compiled = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic).compile_grammar(
        grammar
    )
    matcher = xgr.GrammarMatcher(compiled)
    for byte in b"ab":
        result = mask(matcher)
        assert (int(result[0, byte // 32]) >> (byte % 32)) & 1
        assert matcher.accept_token(byte)
    result = mask(matcher)
    allowed = {i for i in range(len(VOCAB)) if (int(result[0, i // 32]) >> (i % 32)) & 1}
    assert allowed == {ord(">")}
    assert matcher.accept_token(ord(">"))
    assert matcher.accept_token(STOP)


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("edge", ["Token(97, 98)", "ExcludeToken(120, 121)"])
def test_token_edges_preserve_masks(edge, cache_enabled):
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[STOP])
    grammars = [
        xgr.GrammarCompiler(
            info, cache_enabled=cache_enabled, enable_dynamic_compilation=dynamic
        ).compile_grammar(f'root ::= {edge} "!"')
        for dynamic in (False, True)
    ]
    matchers = [xgr.GrammarMatcher(g) for g in grammars]
    for token in [97, 33, STOP]:
        torch.testing.assert_close(mask(matchers[0]), mask(matchers[1]), rtol=0, atol=0)
        for matcher in matchers:
            assert matcher.accept_token(token)


def test_dynamic_token_id_validation_is_not_deferred():
    info = xgr.TokenizerInfo(["a"], stop_token_ids=[])
    compiler = xgr.GrammarCompiler(info, enable_dynamic_compilation=True)
    with pytest.raises(RuntimeError, match="out of the vocabulary range"):
        compiler.compile_grammar("root ::= Token(10)")


VOCABULARY = [chr(value) for value in range(32, 127)] + ["Ada", "hello", "42", "<call>"]


def _mask_trace(compiled_grammar: xgr.CompiledGrammar, input_string: str):
    matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, compiled_grammar.tokenizer_info.vocab_size)
    trace = []
    for char in input_string:
        xgr.reset_token_bitmask(bitmask)
        trace.append((matcher.fill_next_token_bitmask(bitmask), bitmask.clone()))
        assert matcher.accept_string(char)
    assert matcher.is_terminated()
    return trace


def _compile_ebnf(compiler: xgr.GrammarCompiler) -> xgr.CompiledGrammar:
    return compiler.compile_grammar(
        """
root ::= greeting punctuation
greeting ::= "hello" | "hi"
punctuation ::= "!" | "?"
"""
    )


def _compile_json_schema(compiler: xgr.GrammarCompiler) -> xgr.CompiledGrammar:
    return compiler.compile_json_schema(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "scores": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["name", "scores"],
            "additionalProperties": False,
        },
        any_whitespace=True,
    )


def _compile_regex(compiler: xgr.GrammarCompiler) -> xgr.CompiledGrammar:
    return compiler.compile_regex(r"[A-Z][a-z]{2}[0-9]+")


def _compile_builtin_json(compiler: xgr.GrammarCompiler) -> xgr.CompiledGrammar:
    return compiler.compile_builtin_json_grammar()


def _compile_structural_tag(compiler: xgr.GrammarCompiler) -> xgr.CompiledGrammar:
    return compiler.compile_structural_tag(
        {
            "type": "structural_tag",
            "format": {
                "type": "dispatch",
                "rules": [["<call>", {"type": "const_string", "value": "X"}]],
                "loop": False,
                "excludes": [],
            },
        }
    )


COMPILATION_CASES = [
    (_compile_ebnf, "hello!"),
    (_compile_json_schema, '{"name":"Ada","scores":[1,2]}'),
    (_compile_regex, "Ada42"),
    (_compile_builtin_json, '{"name":"Ada"}'),
    (_compile_structural_tag, "prefix<call>X"),
]


@pytest.mark.parametrize(
    "compile_grammar,input_string",
    COMPILATION_CASES,
    ids=["ebnf", "json-schema", "regex", "builtin-json", "structural-tag"],
)
def test_dynamic_compilation_matches_eager_masks(compile_grammar, input_string):
    tokenizer_info = xgr.TokenizerInfo(VOCABULARY, stop_token_ids=[])
    eager = compile_grammar(
        xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=False)
    )
    dynamic = compile_grammar(
        xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=True)
    )

    expected = _mask_trace(eager, input_string)
    actual = _mask_trace(dynamic, input_string)
    assert len(actual) == len(expected)
    for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
        assert actual_apply == expected_apply
        torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


@pytest.mark.parametrize("repeat_range", ["{3,}", "{3,5}"])
def test_right_recursive_rule_inside_repetition_matches_eager_masks(repeat_range):
    tokenizer_info = xgr.TokenizerInfo(VOCABULARY, stop_token_ids=[])
    grammar = (
        'root ::= "[" item tail "]"\n'
        f"tail ::= unit{repeat_range}\n"
        'unit ::= ", " item\n'
        "item ::= [0-9]+"
    )
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar(grammar)

    expected = _mask_trace(eager, "[1, 2, 3, 4]")
    actual = _mask_trace(dynamic, "[1, 2, 3, 4]")
    for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
        assert actual_apply == expected_apply
        torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


@pytest.mark.parametrize("cache_enabled", [False, True], ids=["cache-off", "cache-on"])
def test_recursive_json_string_character_class_summary_matches_eager(cache_enabled):
    shared_prefix = "a" * 64
    vocabulary = [chr(value) for value in range(32, 127)] + [
        "plainascii",
        shared_prefix,
        shared_prefix + "!",
        shared_prefix + '"',
        shared_prefix + '"}',
        shared_prefix + '"}suffix',
        "中文",
        '中文"',
        '中文"}',
        "\\n",
        b"\xe4",
        b"\xe4\xb8",
        b"\xff",
    ]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=cache_enabled, enable_dynamic_compilation=False
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=cache_enabled, enable_dynamic_compilation=True
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)

    expected = _mask_trace(eager, '{"value": "中文\\nA"}')
    actual = _mask_trace(dynamic, '{"value": "中文\\nA"}')
    for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
        assert actual_apply == expected_apply
        torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)

    dynamic_matcher = xgr.GrammarMatcher(dynamic, terminate_without_stop_token=True)
    assert dynamic_matcher.accept_string('{"value": "')
    dynamic_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert dynamic_matcher.fill_next_token_bitmask(dynamic_bitmask)
    mask = bitmask_to_bool_mask(dynamic_bitmask, tokenizer_info.vocab_size)[0]
    for token_id in range(tokenizer_info.vocab_size):
        assert bool(mask[token_id]) == dynamic_matcher.fork().accept_token(token_id), token_id


def test_json_string_shape_fast_path_requires_standard_escape_rule():
    vocabulary = ["a", r"\n", r"\q", '"', 'a"', r"\\"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    grammar = r"""
root ::= "\"" string_sub
string_sub ::= "\"" | [^\x00-\x1f"\\] string_sub | "\\" weird_escape string_sub
weird_escape ::= [q]
"""
    compiled = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string('"')
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert matcher.fill_next_token_bitmask(bitmask)
    mask = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    for token_id in range(tokenizer_info.vocab_size):
        assert bool(mask[token_id]) == matcher.fork().accept_token(token_id), token_id


def test_json_string_min_length_masks_track_decoded_count_across_positions():
    # Enough quote-crossing tokens to keep them deferred to the runtime matcher and to enable the
    # runtime continuation-mask cache, which must not replay a mask computed at an earlier decoded
    # character count of the same string. The chunks step the decoded count through 0, 4, 7 and 8
    # of the required 8 characters, so quote-closing tokens flip from rejected to accepted: at
    # seven characters `"` still closes too early while `#"` closes exactly on time.
    crossing_tokens = ["x" * (index % 60 + 1) + '"' + str(index) for index in range(17000)]
    vocabulary = (
        [chr(value) for value in range(32, 127) if value != ord("#")]
        + ['#"', "abcd", "efg", "h"]
        + crossing_tokens
    )
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=True
    ).compile_json_schema('{"type": "string", "minLength": 8}')
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    sampled_token_ids = list(range(98)) + list(range(98, tokenizer_info.vocab_size, 127))
    for chunk in ['"', "abcd", "efg", "h", '#"']:
        xgr.reset_token_bitmask(bitmask)
        if matcher.fill_next_token_bitmask(bitmask):
            mask = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
            for token_id in sampled_token_ids:
                assert bool(mask[token_id]) == matcher.fork().accept_token(token_id), (
                    chunk,
                    token_id,
                )
        assert matcher.accept_string(chunk)
    assert matcher.is_terminated()


@pytest.mark.parametrize("cache_enabled", [False, True], ids=["cache-off", "cache-on"])
def test_plain_json_string_length_masks_match_eager_and_token_oracle(cache_enabled):
    vocabulary = [
        '"',
        '"}',
        "a",
        "ab",
        "abc",
        "abcd",
        'a"',
        'ab"',
        'abc"',
        'abcd"',
        "é",
        'é"',
        "中",
        '中"',
        r"\u0061",
        r'\u0061"',
        r"\ud83d\ude00",
        r'\ud83d\ude00"',
        r"\"",
        r"\\",
        b"\xe4",
        b"\xe4\xb8",
        b"\xff",
    ]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiler_options = {
        "tokenizer_info": tokenizer_info,
        "max_threads": 1,
        "cache_enabled": cache_enabled,
    }
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string", "minLength": 2, "maxLength": 3}},
        "required": ["value"],
        "additionalProperties": False,
    }
    eager = xgr.GrammarCompiler(
        **compiler_options, enable_dynamic_compilation=False
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)
    dynamic = xgr.GrammarCompiler(
        **compiler_options, enable_dynamic_compilation=True
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)

    prefixes = ['{"value": "', '{"value": "a', '{"value": "ab', '{"value": "abc']
    for prefix in prefixes:
        eager_matcher = xgr.GrammarMatcher(eager, terminate_without_stop_token=True)
        dynamic_matcher = xgr.GrammarMatcher(dynamic, terminate_without_stop_token=True)
        assert eager_matcher.accept_string(prefix)
        assert dynamic_matcher.accept_string(prefix)
        eager_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        dynamic_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert eager_matcher.fill_next_token_bitmask(eager_bitmask)
        assert dynamic_matcher.fill_next_token_bitmask(dynamic_bitmask)
        torch.testing.assert_close(dynamic_bitmask, eager_bitmask, rtol=0, atol=0)

        mask = bitmask_to_bool_mask(dynamic_bitmask, tokenizer_info.vocab_size)[0]
        for token_id in range(tokenizer_info.vocab_size):
            assert bool(mask[token_id]) == dynamic_matcher.fork().accept_token(token_id), token_id


def test_json_string_length_quote_suffix_index_matches_eager_and_token_oracle():
    safe_content_tokens = ["q" * length for length in range(1, 1101)]
    quote_crossing_tokens = [
        'a"}',
        'aa"}',
        'aaa"}',
        'aaaa"}',
        r'\u0061a"}',
        r'\ud83d\ude00a"}',
        r'a\"b"}',
        'aa"',
        'aa"}x',
        'aa"},',
    ]
    quote_crossing_tokens += [f'aa"}}invalid{index:03d}' for index in range(256)]
    vocabulary = safe_content_tokens + quote_crossing_tokens + [b"\xff"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string", "minLength": 2, "maxLength": 3}},
        "required": ["value"],
        "additionalProperties": False,
    }
    compiler_options = {"tokenizer_info": tokenizer_info, "max_threads": 1, "cache_enabled": False}
    eager = xgr.GrammarCompiler(
        **compiler_options, enable_dynamic_compilation=False
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)
    dynamic = xgr.GrammarCompiler(
        **compiler_options, enable_dynamic_compilation=True
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)

    prefix = '{"value": "'

    def fill(compiled_grammar, debug_print=False):
        matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        assert matcher.accept_string(prefix)
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert matcher.fill_next_token_bitmask(bitmask, debug_print=debug_print)
        return matcher, bitmask

    _, eager_bitmask = fill(eager)
    dynamic_matcher, dynamic_bitmask = fill(dynamic, debug_print=True)
    torch.testing.assert_close(dynamic_bitmask, eager_bitmask, rtol=0, atol=0)

    allowed = bitmask_to_bool_mask(dynamic_bitmask, tokenizer_info.vocab_size)[0]
    for token in quote_crossing_tokens:
        token_id = vocabulary.index(token)
        assert bool(allowed[token_id]) == dynamic_matcher.fork().accept_token(token_id), token

    restored = xgr.CompiledGrammar.deserialize_json(dynamic.serialize_json(), tokenizer_info)
    _, restored_bitmask = fill(restored)
    torch.testing.assert_close(restored_bitmask, dynamic_bitmask, rtol=0, atol=0)


def _json_string_length_boundary_cases():
    cases = []
    boundary_values = (
        0,
        1,
        2,
        3,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        200,
        255,
        256,
        257,
        4095,
        4096,
        4097,
        16383,
        16384,
        16385,
        32768,
        65536,
    )
    for boundary in boundary_values:
        prefix_lengths = (0, 1) if boundary == 0 else (boundary - 1, boundary, boundary + 1)
        for prefix_length in prefix_lengths:
            cases.extend(
                [
                    pytest.param(
                        boundary, None, prefix_length, id=f"min-{boundary}-seq-{prefix_length}"
                    ),
                    pytest.param(
                        None, boundary, prefix_length, id=f"max-{boundary}-seq-{prefix_length}"
                    ),
                ]
            )
    for exact in (0, 1, 32, 64, 128, 256, 4096, 16384, 65536):
        for prefix_length in sorted({max(0, exact - 1), exact, exact + 1}):
            cases.append(
                pytest.param(exact, exact, prefix_length, id=f"exact-{exact}-seq-{prefix_length}")
            )
    for minimum, maximum in ((0, 1), (31, 33), (63, 65), (127, 129), (4095, 4097)):
        for prefix_length in range(max(0, minimum - 1), maximum + 2):
            cases.append(
                pytest.param(
                    minimum,
                    maximum,
                    prefix_length,
                    id=f"range-{minimum}-{maximum}-seq-{prefix_length}",
                )
            )
    # Cover maximum-only boundaries and minimums through 128. Larger minimums
    # are sampled by the regression case at 129.
    compatible_cases = [case for case in cases if (case.values[0] is None or case.values[0] <= 128)]
    return compatible_cases + [pytest.param(129, None, 128, id="min-129-seq-128")]


@lru_cache(maxsize=None)
def _json_string_length_boundary_fixture(minimum, maximum):
    boundary_lengths = [
        1,
        2,
        3,
        31,
        32,
        33,
        63,
        64,
        65,
        126,
        127,
        128,
        129,
        255,
        256,
        257,
        4095,
        4096,
        4097,
        16383,
        16384,
        16385,
    ]
    suffixes = ['"', '"}', '"}x', '"]', '":', '"   ']
    quote_crossing_tokens = [
        "a" * content_length + suffix for content_length in boundary_lengths for suffix in suffixes
    ]
    quote_crossing_tokens += [r'\u0061"}', r'\ud83d\ude00"}', r'a\"b"}']
    vocabulary = ["a", '"', "}", "]", ":", " "] + quote_crossing_tokens + [b"\xff"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    string_schema = {"type": "string"}
    if minimum is not None:
        string_schema["minLength"] = minimum
    if maximum is not None:
        string_schema["maxLength"] = maximum
    schema = {
        "type": "object",
        "properties": {"value": string_schema},
        "required": ["value"],
        "additionalProperties": False,
    }
    compiler_options = {"tokenizer_info": tokenizer_info, "max_threads": 1, "cache_enabled": False}
    eager = xgr.GrammarCompiler(
        **compiler_options, enable_dynamic_compilation=False
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)
    dynamic = xgr.GrammarCompiler(
        **compiler_options, enable_dynamic_compilation=True
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)
    return tokenizer_info, eager, dynamic


@pytest.mark.parametrize("minimum,maximum,prefix_length", _json_string_length_boundary_cases())
def test_json_string_length_quote_suffix_index_boundaries(minimum, maximum, prefix_length):
    tokenizer_info, eager, dynamic = _json_string_length_boundary_fixture(minimum, maximum)

    prefix = '{"value": "' + "a" * prefix_length

    if maximum is not None and prefix_length > maximum:
        for compiled_grammar in (eager, dynamic):
            matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
            assert not matcher.accept_string(prefix)
        return

    def fill(compiled_grammar):
        matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        assert matcher.accept_string(prefix)
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert matcher.fill_next_token_bitmask(bitmask)
        return matcher, bitmask

    _, eager_bitmask = fill(eager)
    dynamic_matcher, dynamic_bitmask = fill(dynamic)
    torch.testing.assert_close(dynamic_bitmask, eager_bitmask, rtol=0, atol=0)

    allowed = bitmask_to_bool_mask(dynamic_bitmask, tokenizer_info.vocab_size)[0]
    if prefix_length <= 4097:
        oracle_token_ids = range(tokenizer_info.vocab_size)
    else:
        # Forking a matcher copies its parse history. Keep full token-oracle coverage on all
        # short/medium boundaries and sample every vocabulary region for very long histories.
        last_token_id = tokenizer_info.vocab_size - 1
        oracle_token_ids = sorted(
            {
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                tokenizer_info.vocab_size // 4,
                tokenizer_info.vocab_size // 2,
                3 * tokenizer_info.vocab_size // 4,
                last_token_id - 1,
                last_token_id,
            }
        )
    for token_id in oracle_token_ids:
        assert bool(allowed[token_id]) == dynamic_matcher.fork().accept_token(token_id), token_id


@pytest.mark.parametrize("with_other_length_rule", [False, True], ids=["no-counter", "counter"])
def test_unbounded_json_string_quote_suffix_index_matches_eager_and_token_oracle(
    with_other_length_rule,
):
    safe_content_tokens = ["q" * length for length in range(1, 1101)]
    suffixes = ['"', '"}', '"}x', '"]', '":', '"   ']
    quote_crossing_tokens = [
        "a" * content_length + suffix
        for content_length in (1, 2, 63, 64, 127, 128, 129, 255, 256, 4095, 4096, 4097)
        for suffix in suffixes
    ]
    quote_crossing_tokens += [rf'aa"}}invalid{index:03d}' for index in range(256)] + [
        r'\u0061"}',
        r'\ud83d\ude00"}',
        r'a\"b"}',
    ]
    vocabulary = safe_content_tokens + quote_crossing_tokens + ["a", '"', "}", "]", ":", " "]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    properties = {"value": {"type": "string"}}
    if with_other_length_rule:
        properties["optional_bounded"] = {"type": "string", "minLength": 1, "maxLength": 2}
    schema = {
        "type": "object",
        "properties": properties,
        "required": ["value"],
        "additionalProperties": False,
    }
    compiler_options = {"tokenizer_info": tokenizer_info, "max_threads": 1, "cache_enabled": False}
    eager = xgr.GrammarCompiler(
        **compiler_options, enable_dynamic_compilation=False
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)
    dynamic = xgr.GrammarCompiler(
        **compiler_options, enable_dynamic_compilation=True
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)

    for prefix_length in (0, 16384):
        prefix = '{"value": "' + "a" * prefix_length

        def fill(compiled_grammar):
            matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
            assert matcher.accept_string(prefix)
            bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
            assert matcher.fill_next_token_bitmask(bitmask)
            return matcher, bitmask

        _, eager_bitmask = fill(eager)
        dynamic_matcher, dynamic_bitmask = fill(dynamic)
        torch.testing.assert_close(dynamic_bitmask, eager_bitmask, rtol=0, atol=0)

        allowed = bitmask_to_bool_mask(dynamic_bitmask, tokenizer_info.vocab_size)[0]
        for token_id in range(tokenizer_info.vocab_size):
            assert bool(allowed[token_id]) == dynamic_matcher.fork().accept_token(
                token_id
            ), token_id

        repeated_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert dynamic_matcher.fill_next_token_bitmask(repeated_bitmask)
        torch.testing.assert_close(repeated_bitmask, dynamic_bitmask, rtol=0, atol=0)
        assert dynamic_matcher.accept_string("a")


def test_generic_local_completion_mask_matches_oracle_and_survives_serialization():
    safe_prefix_tokens = ["q" * length for length in range(1, 1101)]
    crossing_tokens = [
        "a" * (1 + index % 257) + ";" + (">" if index % 3 == 0 else f"x{index:03d}")
        for index in range(384)
    ]
    vocabulary = safe_prefix_tokens + crossing_tokens + ["a", "q", ";", ">", r"\1;>", b"\xff"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    grammar = xgr.Grammar.from_ebnf(
        r"""root ::= "<" body ">"
body ::= (";" | [a-z] body | "\\" escape body)
escape ::= [0-9]"""
    )
    options = {"tokenizer_info": tokenizer_info, "max_threads": 1, "cache_enabled": False}
    eager = xgr.GrammarCompiler(**options, enable_dynamic_compilation=False).compile_grammar(
        grammar
    )
    dynamic = xgr.GrammarCompiler(**options, enable_dynamic_compilation=True).compile_grammar(
        grammar
    )

    def fill(compiled_grammar, debug_print=False):
        matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        assert matcher.accept_string("<")
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert matcher.fill_next_token_bitmask(bitmask, debug_print=debug_print)
        return matcher, bitmask

    _, eager_bitmask = fill(eager)
    dynamic_matcher, dynamic_bitmask = fill(dynamic, debug_print=True)
    torch.testing.assert_close(dynamic_bitmask, eager_bitmask, rtol=0, atol=0)
    allowed = bitmask_to_bool_mask(dynamic_bitmask, tokenizer_info.vocab_size)[0]
    for token_id in range(tokenizer_info.vocab_size):
        assert bool(allowed[token_id]) == dynamic_matcher.fork().accept_token(token_id), token_id

    restored = xgr.CompiledGrammar.deserialize_json(dynamic.serialize_json(), tokenizer_info)
    restored_matcher, restored_bitmask = fill(restored, debug_print=True)
    torch.testing.assert_close(restored_bitmask, dynamic_bitmask, rtol=0, atol=0)
    restored_allowed = bitmask_to_bool_mask(restored_bitmask, tokenizer_info.vocab_size)[0]
    for token_id in range(tokenizer_info.vocab_size):
        assert bool(restored_allowed[token_id]) == restored_matcher.fork().accept_token(token_id)


def test_json_shaped_rule_with_non_json_escape_uses_actual_language():
    vocabulary = [r'\q"}', r'\q"x', r'\n"}', 'a"}', '"}', "a", r"\q", b"\xff"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    grammar = xgr.Grammar.from_ebnf(
        r'''root ::= "\"" basic_string_sub "}"
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" fake_escape basic_string_sub)
fake_escape ::= "q"'''
    )
    options = {"tokenizer_info": tokenizer_info, "max_threads": 1, "cache_enabled": False}
    eager = xgr.GrammarCompiler(**options, enable_dynamic_compilation=False).compile_grammar(
        grammar
    )
    dynamic = xgr.GrammarCompiler(**options, enable_dynamic_compilation=True).compile_grammar(
        grammar
    )

    def fill(compiled_grammar):
        matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        assert matcher.accept_string('"')
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert matcher.fill_next_token_bitmask(bitmask)
        return matcher, bitmask

    _, eager_bitmask = fill(eager)
    dynamic_matcher, dynamic_bitmask = fill(dynamic)
    torch.testing.assert_close(dynamic_bitmask, eager_bitmask, rtol=0, atol=0)
    allowed = bitmask_to_bool_mask(dynamic_bitmask, tokenizer_info.vocab_size)[0]
    assert allowed[vocabulary.index(r'\q"}')]
    assert not allowed[vocabulary.index(r'\n"}')]
    for token_id in range(tokenizer_info.vocab_size):
        assert bool(allowed[token_id]) == dynamic_matcher.fork().accept_token(token_id), token_id


def test_local_completion_falls_back_for_nested_nonredundant_lookahead():
    vocabulary = ["a", ";", "ax", "ax;", "a;>", ";>", ">", "z", b"\xff"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    grammar = xgr.Grammar.from_ebnf(
        r"""root ::= "<" body ">"
body ::= (";" | "a" guarded body)
guarded ::= "x" (= "z")"""
    )
    options = {"tokenizer_info": tokenizer_info, "max_threads": 1, "cache_enabled": False}
    eager = xgr.GrammarCompiler(**options, enable_dynamic_compilation=False).compile_grammar(
        grammar
    )
    dynamic = xgr.GrammarCompiler(**options, enable_dynamic_compilation=True).compile_grammar(
        grammar
    )

    eager_matcher = xgr.GrammarMatcher(eager, terminate_without_stop_token=True)
    dynamic_matcher = xgr.GrammarMatcher(dynamic, terminate_without_stop_token=True)
    assert eager_matcher.accept_string("<")
    assert dynamic_matcher.accept_string("<")
    eager_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    dynamic_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert eager_matcher.fill_next_token_bitmask(eager_bitmask)
    assert dynamic_matcher.fill_next_token_bitmask(dynamic_bitmask)
    torch.testing.assert_close(dynamic_bitmask, eager_bitmask, rtol=0, atol=0)

    allowed = bitmask_to_bool_mask(dynamic_bitmask, tokenizer_info.vocab_size)[0]
    for token_id in range(tokenizer_info.vocab_size):
        assert bool(allowed[token_id]) == dynamic_matcher.fork().accept_token(token_id), token_id


def test_local_completion_all_accepted_suffixes_match_eager_and_oracle():
    # Every crossing token has a valid suffix in the enclosing rule.
    safe_tokens = ["q" * length for length in range(1, 40)]
    crossing_tokens = ["a" * length + ";>" for length in range(1, 40)] + [";>", ";"]
    vocabulary = safe_tokens + crossing_tokens + ["a", "q", ">", b"\xff"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    grammar = xgr.Grammar.from_ebnf(
        r"""root ::= "<" body ">"
body ::= (";" | [a-z] body)"""
    )
    options = {"tokenizer_info": tokenizer_info, "max_threads": 1, "cache_enabled": False}
    eager = xgr.GrammarCompiler(**options, enable_dynamic_compilation=False).compile_grammar(
        grammar
    )
    dynamic = xgr.GrammarCompiler(**options, enable_dynamic_compilation=True).compile_grammar(
        grammar
    )

    def fill(compiled_grammar):
        matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        assert matcher.accept_string("<")
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert matcher.fill_next_token_bitmask(bitmask)
        return matcher, bitmask

    _, eager_bitmask = fill(eager)
    dynamic_matcher, dynamic_bitmask = fill(dynamic)
    torch.testing.assert_close(dynamic_bitmask, eager_bitmask, rtol=0, atol=0)
    allowed = bitmask_to_bool_mask(dynamic_bitmask, tokenizer_info.vocab_size)[0]
    assert allowed[vocabulary.index(";>")]
    assert allowed[vocabulary.index("a;>")]
    for token_id in range(tokenizer_info.vocab_size):
        assert bool(allowed[token_id]) == dynamic_matcher.fork().accept_token(token_id), token_id


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True], ids=["eager", "dynamic"])
def test_local_completion_respects_rule_own_lookahead(enable_dynamic_compilation):
    # body carries its own non-redundant lookahead (= ">") while the true successor is "!".
    # The mask must filter crossing tokens with the lookahead whether or not the rule
    # qualifies for the local-completion optimization (unique vs. two completion bytes).
    vocabulary = ["a;!", "aa;!", "a;>", "a", ";", "!", ">", b"\xff"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    optimizable = xgr.Grammar.from_ebnf(
        r"""root ::= "<" body "!"
body ::= (";" | [a-z] body) (= ">")"""
    )
    fallback = xgr.Grammar.from_ebnf(
        r"""root ::= "<" body "!"
body ::= (";" | ":" | [a-z] body) (= ">")"""
    )
    options = {"tokenizer_info": tokenizer_info, "max_threads": 1, "cache_enabled": False}
    for grammar in (optimizable, fallback):
        compiled = xgr.GrammarCompiler(
            **options, enable_dynamic_compilation=enable_dynamic_compilation
        ).compile_grammar(grammar)
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert matcher.accept_string("<")
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert matcher.fill_next_token_bitmask(bitmask)
        allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
        assert not allowed[vocabulary.index("a;!")]
        assert not allowed[vocabulary.index("aa;!")]
        assert not allowed[vocabulary.index("a;>")]
        assert allowed[vocabulary.index("a")]
        assert allowed[vocabulary.index(";")]


@pytest.mark.parametrize("cache_enabled", [False, True], ids=["cache-off", "cache-on"])
def test_email_format_regex_fsm_masks_match_eager_and_token_oracle(cache_enabled):
    shared_prefix = "a" * 64
    vocabulary = [
        "a",
        "abc",
        shared_prefix + "@example.com",
        shared_prefix + "@example.com-",
        shared_prefix + '.doe@example.com"}',
        ".doe",
        ".doe@example",
        "@example",
        "@example.com",
        ".com",
        '.com"',
        '.com"}',
        "-",
        "..",
        ".-",
        "@",
        '"',
        '\\"',
        "\\\\",
        '",',
        "é",
        "中",
        "a中",
        b"\xe4",
        b"\xe4\xb8",
        b"\xff",
    ]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiler_options = {
        "tokenizer_info": tokenizer_info,
        "max_threads": 1,
        "cache_enabled": cache_enabled,
    }
    schema = {
        "type": "object",
        "properties": {"email": {"type": "string", "format": "email"}},
        "required": ["email"],
        "additionalProperties": False,
    }
    eager = xgr.GrammarCompiler(
        **compiler_options, enable_dynamic_compilation=False
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)
    dynamic = xgr.GrammarCompiler(
        **compiler_options, enable_dynamic_compilation=True
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)

    prefixes = []
    for email_source in ["john.doe@example-domain.com", r"\"john..doe\"@example.org"]:
        prefixes.extend(
            '{"email": "' + email_source[:length] for length in range(len(email_source) + 1)
        )

    for prefix in prefixes:
        eager_matcher = xgr.GrammarMatcher(eager, terminate_without_stop_token=True)
        dynamic_matcher = xgr.GrammarMatcher(dynamic, terminate_without_stop_token=True)
        assert eager_matcher.accept_string(prefix)
        assert dynamic_matcher.accept_string(prefix)
        eager_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        dynamic_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert eager_matcher.fill_next_token_bitmask(eager_bitmask)
        assert dynamic_matcher.fill_next_token_bitmask(dynamic_bitmask)
        torch.testing.assert_close(dynamic_bitmask, eager_bitmask, rtol=0, atol=0)

        mask = bitmask_to_bool_mask(dynamic_bitmask, tokenizer_info.vocab_size)[0]
        for token_id in range(tokenizer_info.vocab_size):
            assert bool(mask[token_id]) == dynamic_matcher.fork().accept_token(token_id), token_id


def test_dynamic_masks_are_cached():
    tokenizer_info = xgr.TokenizerInfo(VOCABULARY, stop_token_ids=[])
    dynamic = _compile_builtin_json(
        xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=True)
    )
    eager = _compile_builtin_json(
        xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=False)
    )

    initial_size = dynamic.memory_size_bytes
    assert initial_size < eager.memory_size_bytes
    _mask_trace(dynamic, '{"name":"Ada"}')
    populated_size = dynamic.memory_size_bytes
    assert populated_size > initial_size
    _mask_trace(dynamic, '{"name":"Ada"}')
    assert dynamic.memory_size_bytes == populated_size


def test_serialization_materializes_dynamic_masks():
    tokenizer_info = xgr.TokenizerInfo(VOCABULARY, stop_token_ids=[])
    dynamic = _compile_ebnf(
        xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=True)
    )
    initial_size = dynamic.memory_size_bytes

    restored = xgr.CompiledGrammar.deserialize_json(dynamic.serialize_json(), tokenizer_info)
    assert dynamic.memory_size_bytes > initial_size
    for input_string in ["hello!", "hi?"]:
        expected = _mask_trace(dynamic, input_string)
        actual = _mask_trace(restored, input_string)
        for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
            assert actual_apply == expected_apply
            torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


def test_concurrent_dynamic_mask_generation():
    tokenizer_info = xgr.TokenizerInfo(VOCABULARY, stop_token_ids=[])
    dynamic = _compile_json_schema(
        xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=True)
    )
    input_string = '{"name":"Ada","scores":[1,2]}'

    with ThreadPoolExecutor(max_workers=8) as executor:
        traces = list(executor.map(lambda _: _mask_trace(dynamic, input_string), range(32)))

    expected = traces[0]
    for actual in traces[1:]:
        for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
            assert actual_apply == expected_apply
            torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


def test_shared_character_class_repeat_masks_survive_compiler_cache_clear():
    vocabulary = [">", "<", "[", "]", "a", "b", "ab", "ab<", "ab]", b"\xff"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=True)
    first = compiler.compile_grammar('root ::= ">" value "<"\nvalue ::= [a-z]{2,4}')
    expected_first = _mask_trace(first, ">ab<")

    compiler.clear_cache()
    second = compiler.compile_grammar('root ::= "[" value "]"\nvalue ::= [a-z]{2,4}')
    expected_second = _mask_trace(second, "[ab]")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_mask_trace, grammar, value)
            for _ in range(8)
            for grammar, value in ((first, ">ab<"), (second, "[ab]"))
        ]
    for index, future in enumerate(futures):
        expected = expected_first if index % 2 == 0 else expected_second
        actual = future.result()
        for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
            assert actual_apply == expected_apply
            torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


def test_cache_disabled_bounded_character_class_masks_match_eager():
    tokenizer_info = xgr.TokenizerInfo(
        [chr(value) for value in range(32, 127)] + ["ab", "abc", "abcd", "ab>", ">"],
        stop_token_ids=[],
    )
    grammar = 'root ::= value ">"\nvalue ::= [a-z]{2,4}'
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=True
    ).compile_grammar(grammar)
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=False
    ).compile_grammar(grammar)

    xgr.GrammarMatcher(dynamic, terminate_without_stop_token=True)
    expected = _mask_trace(eager, "ab>")
    actual = _mask_trace(dynamic, "ab>")
    # Bounded character-class masks must agree with eager compilation.
    for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
        assert actual_apply == expected_apply
        torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


@pytest.mark.parametrize(
    "grammar,prefixes,vocabulary",
    [
        (
            'root ::= literal "x"\nliteral[capture="literal"] ::= "abc"',
            ["", "a", "ab", "abc"],
            ["a", "ab", "abc", "abc", "abcx", "abcy", "x", "xy", "bad"],
        ),
        (
            'root ::= left "!" | right "?"\n'
            'left[capture="left"] ::= "abc"\n'
            'right[capture="right"] ::= "abd"',
            ["", "a", "ab", "abc"],
            ["a", "ab", "abc", "abc!", "abc?", "abd", "abd!", "abd?", "abe", "!", "?"],
        ),
        (
            'root ::= literal "x"\nliteral[capture="literal"] ::= "abc" | "abd"',
            ["", "a", "ab", "abc"],
            ["a", "ab", "abc", "abcx", "abc?", "abd", "abdx", "abd?", "abe", "x"],
        ),
        (
            "root ::= property\n"
            'property[capture="property"] ::= "\\"name\\":" value\n'
            "value ::= [0-9]",
            ["", '"', '"name', '"name":'],
            ['"', '"name', '"name":', '"name":1', '"name":x', "1", "x"],
        ),
        (
            "root ::= property\n"
            'property[capture="property"] ::= "\\"name\\":" value | '
            '"\\"address\\":" value\n'
            "value ::= [0-9]",
            ["", '"', '"name', '"name":', '"address', '"address":'],
            [
                '"',
                '"name',
                '"name":',
                '"name":1',
                '"address',
                '"address":',
                '"address":1',
                '"other":1',
                "1",
            ],
        ),
        (
            'root ::= literal "x"\nliteral[capture="literal"] ::= "é"',
            ["", b"\xc3", "é"],
            [b"\xc3", "é", "éx", "éy", "x", b"\xff"],
        ),
    ],
    ids=[
        "parent-continuation",
        "multiple-live-states",
        "branched-literal-rule",
        "literal-prefix-before-rule-ref",
        "branched-prefixes-before-rule-ref",
        "utf8-byte-prefix",
    ],
)
def test_cache_disabled_deterministic_byte_path_masks_match_token_oracle(
    grammar, prefixes, vocabulary
):
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=True
    ).compile_grammar(grammar)

    for prefix in prefixes:
        masks = []
        for compiled_grammar in [eager, dynamic]:
            matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
            if prefix:
                assert matcher.accept_string(prefix)
            bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
            xgr.reset_token_bitmask(bitmask)
            matcher.fill_next_token_bitmask(bitmask)
            masks.append(bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0])
        torch.testing.assert_close(masks[0], masks[1], rtol=0, atol=0)

        for token_id in range(tokenizer_info.vocab_size):
            oracle = xgr.GrammarMatcher(dynamic, terminate_without_stop_token=True)
            if prefix:
                assert oracle.accept_string(prefix)
            assert bool(masks[1][token_id]) == oracle.accept_token(token_id), (
                prefix,
                token_id,
                vocabulary[token_id],
            )


def test_additional_property_exclusion_sink_masks_match_token_oracle():
    vocabulary = [
        '"',
        "n",
        "na",
        "name",
        'name"',
        'name":',
        'name":1',
        "p",
        "profile",
        'profile":',
        "x",
        "xname",
        'xname"',
        'xname":',
        "x中",
        "x🙂",
        b"x\xe4",
        b"x\xe4\xb8",
        b'x\xe4\xb8\xad"',
        b"x\xff",
        b"x\xc0\x80",
        "x\\u0061",
        b"x\\",
        b"x\\u",
        b"x\\u0",
        b"x\\u00",
        b"x\\u006",
        b'x\\u0061"',
        b"x\\q",
        b"x\\u0g",
        b'x\\"',
        "x\n",
        " other",
        '\\"',
        "\\u0061",
        "\\n",
        b"\xff",
        ":",
        "1",
        "}",
    ]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    schema = {
        "type": "object",
        "properties": {"name": {"type": "integer"}, "profile": {"type": "integer"}},
        "additionalProperties": {"type": "integer"},
    }
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=False
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=False)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=True
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=False)

    for prefix in ['{"', '{"n', '{"na', '{"name', '{"p', '{"x', '{"xname']:
        masks = []
        for compiled_grammar in [eager, dynamic]:
            matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
            assert matcher.accept_string(prefix)
            bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
            xgr.reset_token_bitmask(bitmask)
            assert matcher.fill_next_token_bitmask(bitmask)
            masks.append(bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0])
        torch.testing.assert_close(masks[0], masks[1], rtol=0, atol=0)

        for token_id in range(tokenizer_info.vocab_size):
            oracle = xgr.GrammarMatcher(dynamic, terminate_without_stop_token=True)
            assert oracle.accept_string(prefix)
            assert bool(masks[1][token_id]) == oracle.accept_token(token_id), (
                prefix,
                token_id,
                vocabulary[token_id],
            )


def test_limited_compiler_cache_does_not_retain_growing_grammar():
    tokenizer_info = xgr.TokenizerInfo(VOCABULARY, stop_token_ids=[])
    cache_limit = 64 * 1024
    compiler = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=True,
        cache_limit_bytes=cache_limit,
        enable_dynamic_compilation=True,
    )
    dynamic = _compile_builtin_json(compiler)
    assert compiler.get_cache_size_bytes() == 0

    barrier = threading.Barrier(9)

    def generate_masks():
        barrier.wait()
        return _mask_trace(dynamic, '{"name":"Ada"}')

    def read_cache_sizes():
        barrier.wait()
        return [compiler.get_cache_size_bytes() for _ in range(100)]

    with ThreadPoolExecutor(max_workers=9) as executor:
        mask_futures = [executor.submit(generate_masks) for _ in range(8)]
        size_future = executor.submit(read_cache_sizes)
        for future in mask_futures:
            future.result()
        observed_sizes = size_future.result()

    assert all(0 <= size <= cache_limit for size in observed_sizes)
    assert 0 <= compiler.get_cache_size_bytes() <= cache_limit


def test_rule_mask_sharing_does_not_cross_context_dependent_rules():
    tokenizer_info = xgr.TokenizerInfo(VOCABULARY, stop_token_ids=[])
    dynamic_compiler = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    )
    eager_compiler = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=False
    )

    unbounded = dynamic_compiler.compile_grammar('root ::= "a" shared "z"\nshared ::= [x]+')
    _mask_trace(unbounded, "axxz")

    bounded_grammar = 'root ::= "a" shared "z"\nshared[max_tokens=1] ::= [x]+'
    dynamic_bounded = dynamic_compiler.compile_grammar(bounded_grammar)
    eager_bounded = eager_compiler.compile_grammar(bounded_grammar)
    expected = _mask_trace(eager_bounded, "axz")
    actual = _mask_trace(dynamic_bounded, "axz")
    for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
        assert actual_apply == expected_apply
        torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


@pytest.mark.parametrize(
    "repeat_range,value",
    [
        ("{0}", ""),
        ("{1}", "a"),
        ("{0,1}", ""),
        ("{1,3}", "ab"),
        ("{63,65}", "a" * 64),
        ("{127,129}", "a" * 128),
        ("{255,257}", "a" * 256),
        ("{2,}", "abc"),
    ],
)
def test_preserved_repetition_ranges_match_eager_masks(repeat_range: str, value: str):
    vocabulary = [">", "<", "a", "aa", "ab", "abc", "b", "ba", "c", b"\xc3", b"\xff"]
    grammar = f'root ::= ">" [a-z]{repeat_range} "<"'
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar(grammar)
    expected = _mask_trace(eager, ">" + value + "<")
    actual = _mask_trace(dynamic, ">" + value + "<")
    for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
        assert actual_apply == expected_apply
        expected_tokens = bitmask_to_bool_mask(expected_mask, tokenizer_info.vocab_size)
        actual_tokens = bitmask_to_bool_mask(actual_mask, tokenizer_info.vocab_size)
        torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)


@pytest.mark.parametrize("cache_enabled", [False, True], ids=["cache-off", "cache-on"])
@pytest.mark.parametrize(
    "grammar,value,vocabulary",
    [
        (
            'root ::= unit{0,1} unit "x"\nunit ::= "a"',
            "aax",
            ["a", "x", "aa", "ax", "aaa", "aax", b"\xc3", b"\xff"],
        ),
        (
            'root ::= unit unit{0,1} "x"\nunit[capture="u"] ::= "a"',
            "aax",
            ["a", "x", "aa", "ax", "aaa", "aax", b"\xc3", b"\xff"],
        ),
        (
            'root ::= unit{2,3} unit "x"\nunit ::= "a"',
            "aaaax",
            ["a", "x", "aa", "ax", "aaa", "aax", "aaaa", "aaax", "aaaax", b"\xc3", b"\xff"],
        ),
        (
            'root ::= unit{0,1} unit "x"\nunit ::= "é"',
            "ééx",
            ["é", "x", "éé", "éx", "ééx", b"\xc3", b"\xa9", b"\xff"],
        ),
    ],
    ids=[
        "repeat-then-normal",
        "normal-then-repeat",
        "multi-repeat-then-normal",
        "unicode-repeat-boundary",
    ],
)
def test_preserved_repetition_ranges_match_multitokens_across_repeat_boundaries(
    cache_enabled, grammar, value, vocabulary
):
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=cache_enabled, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=cache_enabled, enable_dynamic_compilation=True
    ).compile_grammar(grammar)

    expected = _mask_trace(eager, value)
    actual = _mask_trace(dynamic, value)
    for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
        assert actual_apply == expected_apply
        torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)

    if value == "aax":
        for compiled_grammar in [eager, dynamic]:
            matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
            assert matcher.accept_token(vocabulary.index("aa"))
            assert not matcher.is_terminated()
            assert matcher.accept_token(vocabulary.index("x"))
            assert matcher.is_terminated()

            matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
            assert matcher.accept_token(vocabulary.index("aax"))
            assert matcher.is_terminated()


@pytest.mark.parametrize(
    "character_class,value", [("[a-z]", "a"), ("[^b]", "é"), ("[a-zа-я一-龥]", "中")]
)
def test_dynamic_single_character_class_masks_match_eager(character_class, value):
    vocabulary = [">", "<", "a", "ab", "a<", "b", "é", "中", b"\xe4", b"\xe4\xb8", b"\xff"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    grammar = f'root ::= ">" value "<"\nvalue ::= {character_class}'
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar(grammar)

    expected = _mask_trace(eager, ">" + value + "<")
    actual = _mask_trace(dynamic, ">" + value + "<")
    for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
        assert actual_apply == expected_apply
        torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


@pytest.mark.parametrize(
    "character_class,value",
    [("[a-z]", "abc"), ("[a-zа-я一-龥]", "a中я"), ("[^b]", "aé中")],
    ids=["ascii-positive", "unicode-positive", "negative"],
)
def test_recursive_character_class_masks_match_eager_and_token_oracle(character_class, value):
    shared_prefix = "a" * 64
    vocabulary = [
        ">",
        "<",
        "a",
        "abc",
        "abc<",
        "b",
        "é",
        "中",
        "я",
        "a中",
        "a中<",
        shared_prefix,
        shared_prefix + "x",
        shared_prefix + "!",
        shared_prefix + "<",
        b"\xc3",
        b"\xe4",
        b"\xe4\xb8",
        b"\xff",
    ]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    grammar = (
        'root ::= ">" run "<"\n' f'run ::= ({character_class} run | {character_class}) (=("<"))'
    )
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=True
    ).compile_grammar(grammar)

    expected = _mask_trace(eager, ">" + value + "<")
    actual = _mask_trace(dynamic, ">" + value + "<")
    for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
        assert actual_apply == expected_apply
        torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)

    for prefix in [">", ">" + value[:1]]:
        eager_matcher = xgr.GrammarMatcher(eager, terminate_without_stop_token=True)
        dynamic_matcher = xgr.GrammarMatcher(dynamic, terminate_without_stop_token=True)
        assert eager_matcher.accept_string(prefix)
        assert dynamic_matcher.accept_string(prefix)
        eager_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        dynamic_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert eager_matcher.fill_next_token_bitmask(eager_bitmask)
        assert dynamic_matcher.fill_next_token_bitmask(dynamic_bitmask)
        torch.testing.assert_close(dynamic_bitmask, eager_bitmask, rtol=0, atol=0)

        mask = bitmask_to_bool_mask(dynamic_bitmask, tokenizer_info.vocab_size)[0]
        for token_id in range(tokenizer_info.vocab_size):
            assert bool(mask[token_id]) == dynamic_matcher.fork().accept_token(token_id), token_id


def test_continuation_transition_cache_isolated_between_parser_states():
    left_suffixes = [
        chr(first) + chr(second)
        for first in range(ord("a"), ord("n"))
        for second in range(ord("a"), ord("n"))
    ]
    right_suffixes = [
        chr(first) + chr(second)
        for first in range(ord("n"), ord("{"))
        for second in range(ord("n"), ord("{"))
    ]
    vocabulary = (
        [suffix + "!" for suffix in left_suffixes]
        + [suffix + "?" for suffix in right_suffixes]
        + ["a", "n", "!", "?", "x", b"\xff"]
    )
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar(
        'root ::= left "!" | right "?"\n' 'left ::= [a-m] left | ""\n' 'right ::= [n-z] right | ""'
    )

    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert matcher.fill_next_token_bitmask(bitmask, debug_print=True)

    allowed_tokens = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    for token_id in range(tokenizer_info.vocab_size):
        oracle = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert bool(allowed_tokens[token_id]) == oracle.accept_token(token_id), token_id


def test_continuation_mask_cache_classifies_tokens_independently_of_other_states():
    suffixes = [
        chr(first) + chr(second)
        for first in range(ord("a"), ord("z") + 1)
        for second in range(ord("a"), ord("z") + 1)
    ]
    vocabulary = ['"x'] + ['"x' + suffix for suffix in suffixes] + ["a", "x", "Q"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=True
    ).compile_grammar(
        'root ::= value "x" | branch\n'
        'value[temperature=0.7] ::= [a-z]* "\\""\n'
        'branch ::= "aQ" | "\\"xaa"'
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    # The branch accepts this token at the initial position, while value rejects it. The cached
    # value result must therefore be computed independently of the other live branch.
    shared_token_id = vocabulary.index('"xaa')
    xgr.reset_token_bitmask(bitmask)
    assert matcher.fill_next_token_bitmask(bitmask, debug_print=True)
    initial_mask = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert initial_mask[shared_token_id]

    # Consuming "a" keeps value in the same logical state but advances the other branch to "Q".
    # Reusing value's cached continuation must now reject the formerly shared token.
    assert matcher.accept_token(vocabulary.index("a"))
    xgr.reset_token_bitmask(bitmask)
    assert matcher.fill_next_token_bitmask(bitmask, debug_print=True)
    cached_mask = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert not cached_mask[shared_token_id]
    for token_id in range(tokenizer_info.vocab_size):
        oracle = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert oracle.accept_token(vocabulary.index("a"))
        assert bool(cached_mask[token_id]) == oracle.accept_token(token_id), token_id


def test_continuation_mask_cache_is_cleared_before_rollback_row_reuse():
    suffixes = [
        chr(first) + chr(second)
        for first in range(ord("a"), ord("m"))
        for second in range(ord("a"), ord("m"))
    ]
    vocabulary = (
        ['L"', 'R"', '"x', '"y']
        + ['"x' + suffix for suffix in suffixes]
        + ['"y' + suffix for suffix in suffixes]
        + ["a", "x", "y"]
    )
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=True
    ).compile_grammar(
        'root ::= "L\\"" wrapper "x" | "R\\"" wrapper "y"\n'
        "wrapper ::= value\n"
        'value[temperature=0.7] ::= [a-z]* "\\""'
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    def fill_mask():
        xgr.reset_token_bitmask(bitmask)
        assert matcher.fill_next_token_bitmask(bitmask, debug_print=True)
        return bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0].clone()

    assert matcher.accept_token(vocabulary.index('L"'))
    left_mask = fill_mask()
    assert left_mask[vocabulary.index('"x')]
    assert not left_mask[vocabulary.index('"y')]

    # The repeated character-class state and its parent context are logically unchanged here,
    # so the resulting mask must stay the same.
    assert matcher.accept_token(vocabulary.index("a"))
    left_cached_mask = fill_mask()
    torch.testing.assert_close(left_cached_mask, left_mask, rtol=0, atol=0)

    # Rollback reuses the same absolute parser rows for a different grandparent branch. The old
    # left-context entry must be gone before those row numbers are populated by the right branch.
    matcher.rollback(2)
    assert matcher.accept_token(vocabulary.index('R"'))
    right_mask = fill_mask()
    assert not right_mask[vocabulary.index('"x')]
    assert right_mask[vocabulary.index('"y')]
    for token_id in range(tokenizer_info.vocab_size):
        oracle = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert oracle.accept_token(vocabulary.index('R"'))
        assert bool(right_mask[token_id]) == oracle.accept_token(token_id), token_id


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True], ids=["eager", "dynamic"])
def test_shared_parser_features_preserve_budget_and_capture_behavior(enable_dynamic_compilation):
    tokenizer_info = xgr.TokenizerInfo(["ab ", "cd", " ", "</t>", "1", "<t>", "x"])
    grammar = xgr.Grammar.from_lark(
        'start: r "<t>"\nr[max_tokens=3, capture]: TEXT\nTEXT: /(\\n|.)*/',
        tokenizer_info=tokenizer_info,
    )
    compiled = xgr.GrammarCompiler(
        tokenizer_info, enable_dynamic_compilation=enable_dynamic_compilation
    ).compile_grammar(grammar)
    for _ in range(4):
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert all(matcher.accept_token(token_id) for token_id in [0, 1, 2])
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert matcher.fill_next_token_bitmask(bitmask)
        assert bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size).nonzero().tolist() == [
            [0, 5]
        ]
        assert matcher.accept_token(5)
        assert matcher.is_terminated()
        assert matcher.get_captures() == [("r", b"ab cd ")]

    grammar = xgr.Grammar.from_lark(
        'start[capture="outer"]: r "z"\n' 'r[max_chars=2, capture="inner", suffix="!"]: /[a-z]*/'
    )
    compiled = xgr.GrammarCompiler(
        xgr.TokenizerInfo([]), enable_dynamic_compilation=enable_dynamic_compilation
    ).compile_grammar(grammar)
    for _ in range(4):
        for value, expected_captures in [
            ("a!z", [("inner", b"a"), ("outer", b"a!z")]),
            ("abz", [("inner", b"ab"), ("outer", b"abz")]),
        ]:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            assert matcher.accept_string(value) and matcher.is_terminated()
            assert matcher.get_captures() == expected_captures


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize("edge_name", ["Token", "ExcludeToken"])
def test_rule_mask_sharing_distinguishes_token_edge_payloads(enable_dynamic_compilation, edge_name):
    tokenizer_info = xgr.TokenizerInfo(["a", "b", "c"], stop_token_ids=[])
    cached_compiler = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=True,
        enable_dynamic_compilation=enable_dynamic_compilation,
    )
    uncached_compiler = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    )

    def initial_mask(compiled_grammar):
        matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        xgr.reset_token_bitmask(bitmask)
        assert matcher.fill_next_token_bitmask(bitmask)
        return bitmask

    first_mask = initial_mask(cached_compiler.compile_grammar(f"root ::= {edge_name}(0)"))
    actual_mask = initial_mask(cached_compiler.compile_grammar(f"root ::= {edge_name}(1)"))
    expected_mask = initial_mask(uncached_compiler.compile_grammar(f"root ::= {edge_name}(1)"))

    assert not torch.equal(first_mask, expected_mask)
    torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True], ids=["eager", "dynamic"])
def test_rule_mask_sharing_distinguishes_lookahead_expression_boundaries(
    enable_dynamic_compilation,
):
    vocabulary = ["p", "qac", "qa\x01\x01aabb", "q"] + [f"invalid-{index}" for index in range(28)]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    cached_compiler = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=True,
        enable_dynamic_compilation=enable_dynamic_compilation,
    )
    uncached_compiler = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    )

    byte_string_suffix = r'"a\u0001\u0001aabb"'
    character_class_suffix = '"a" [^ab]'

    def grammar(suffix):
        return f'root ::= "p" target {suffix}\ntarget ::= [q] (= {suffix})'

    def mask_after_prefix(compiler, grammar_source):
        compiled_grammar = compiler.compile_grammar(grammar_source)
        matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        assert matcher.accept_token(0)
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        xgr.reset_token_bitmask(bitmask)
        assert matcher.fill_next_token_bitmask(bitmask)
        return bitmask

    first_mask = mask_after_prefix(cached_compiler, grammar(byte_string_suffix))
    actual_mask = mask_after_prefix(cached_compiler, grammar(character_class_suffix))
    expected_mask = mask_after_prefix(uncached_compiler, grammar(character_class_suffix))

    assert not torch.equal(first_mask, expected_mask)
    torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


@pytest.mark.parametrize("cache_enabled", [False, True], ids=["cache-off", "cache-on"])
@pytest.mark.parametrize(
    "grammar,accepted_values,rejected_values",
    [
        ('root ::= ("b"{0,1}){2,2}', ["", "b", "bb"], ["bbb"]),
        (
            'root ::= unit{63,64} "x"\nunit ::= inner{0,2}\ninner ::= "c"',
            ["x", "cx", "c" * 128 + "x"],
            ["c", "c" * 129 + "x"],
        ),
    ],
    ids=["nested-zero-lower-repeat", "indirect-nullable-rule"],
)
def test_preserved_repetition_ranges_complete_indirectly_nullable_children(
    cache_enabled, grammar, accepted_values, rejected_values
):
    tokenizer_info = xgr.TokenizerInfo(["b", "c", "x", "bb", "cx", b"\xff"], stop_token_ids=[])
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=cache_enabled, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=cache_enabled, enable_dynamic_compilation=True
    ).compile_grammar(grammar)
    restored = xgr.CompiledGrammar.deserialize_json(dynamic.serialize_json(), tokenizer_info)

    for value in accepted_values:
        expected_matcher = xgr.GrammarMatcher(eager, terminate_without_stop_token=True)
        assert expected_matcher.accept_string(value) and expected_matcher.is_terminated()
        for candidate in [dynamic, restored]:
            matcher = xgr.GrammarMatcher(candidate, terminate_without_stop_token=True)
            assert matcher.accept_string(value) and matcher.is_terminated()
            expected_trace = _mask_trace(eager, value)
            actual_trace = _mask_trace(candidate, value)
            for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(
                expected_trace, actual_trace
            ):
                assert actual_apply == expected_apply
                torch.testing.assert_close(
                    bitmask_to_bool_mask(actual_mask, tokenizer_info.vocab_size),
                    bitmask_to_bool_mask(expected_mask, tokenizer_info.vocab_size),
                    rtol=0,
                    atol=0,
                )

    for value in rejected_values:
        for candidate in [eager, dynamic, restored]:
            matcher = xgr.GrammarMatcher(candidate, terminate_without_stop_token=True)
            assert not (matcher.accept_string(value) and matcher.is_terminated())


def test_nested_nullable_repetition_preserves_captures_in_dynamic_mode():
    tokenizer_info = xgr.TokenizerInfo(["b", "x", "bx", "bbx"], stop_token_ids=[])
    grammar = xgr.Grammar.from_lark(
        'start: (item{2,3}){2,2} "x"\nitem[capture="item"]: "" | "b"', tokenizer_info=tokenizer_info
    )
    eager = xgr.GrammarCompiler(
        tokenizer_info, cache_enabled=False, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, cache_enabled=False, enable_dynamic_compilation=True
    ).compile_grammar(grammar)
    restored = xgr.CompiledGrammar.deserialize_json(dynamic.serialize_json(), tokenizer_info)

    for value in ["x", "bx", "bbx", "bbbx", "bbbbx"]:
        expected = xgr.GrammarMatcher(eager, terminate_without_stop_token=True)
        assert expected.accept_string(value) and expected.is_terminated()
        for candidate in [dynamic, restored]:
            matcher = xgr.GrammarMatcher(candidate, terminate_without_stop_token=True)
            assert matcher.accept_string(value) and matcher.is_terminated()
            assert matcher.get_captures() == expected.get_captures()


def test_get_allow_empty_rule_ids_with_preserved_zero_lower_repeat():
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo([]), enable_dynamic_compilation=True)
    compiled = compiler.compile_grammar('root ::= unit\nunit ::= inner{0,2}\ninner ::= "a"')
    # Repetition expansion can introduce additional nullable helper rules.
    assert 0 in _get_allow_empty_rule_ids(compiled)
    assert _compiled_accepts(compiled, "")


def _compiled_accepts(compiled_grammar: xgr.CompiledGrammar, input_str: str) -> bool:
    matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    return matcher.accept_string(input_str) and matcher.is_terminated()


def test_repeated_regex_fsm_reuse_preserves_nullable_rules():
    tokenizer_info = xgr.TokenizerInfo(["!", "a", "aa", "b"], stop_token_ids=[])
    grammar = xgr.Grammar.from_lark('start: left "!" right\nleft: /a*/\nright: /a*/')

    for enable_dynamic_compilation in [False, True]:
        compiled = xgr.GrammarCompiler(
            tokenizer_info, max_threads=1, enable_dynamic_compilation=enable_dynamic_compilation
        ).compile_grammar(grammar)
        for value in ["!", "a!", "!a", "aa!aa"]:
            assert _compiled_accepts(compiled, value)
        for value in ["", "aa", "!!", "b!"]:
            assert not _compiled_accepts(compiled, value)


def test_limited_compiler_cache_evicts_value_that_crosses_limit():
    tokenizer_info = xgr.TokenizerInfo([chr(value) for value in range(32, 127)], stop_token_ids=[])
    cache_limit = 64 * 1024
    compiler = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=4,
        cache_enabled=True,
        cache_limit_bytes=cache_limit,
        enable_dynamic_compilation=False,
    )

    compiled = compiler.compile_builtin_json_grammar()
    assert compiler.get_cache_size_bytes() <= cache_limit
    assert _compiled_accepts(compiled, '{"a": 1}')


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_bounded_word_pattern_property_rejects_invalid_nested_value_token(
    enable_dynamic_compilation: bool,
):
    schema = {
        "type": "object",
        "patternProperties": {
            r"\w{1,15}": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            }
        },
    }
    vocabulary = ['{"q": {}}', '{"q": {"x": 1}}']
    compiled = xgr.GrammarCompiler(
        xgr.TokenizerInfo(vocabulary, stop_token_ids=[]),
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)

    invalid_matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    valid_matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert not invalid_matcher.accept_token(0)
    assert valid_matcher.accept_token(1)
    assert valid_matcher.is_terminated()


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_pattern_length_filters_narrow_cached_accept_set(enable_dynamic_compilation: bool):
    vocabulary = ['"', "1", "1234567890", "12345678901", '1234567890"'] + [
        f"x{index}" for index in range(1200)
    ]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_json_schema(
        {"type": "string", "pattern": "^[0-9]{10,10}$", "minLength": 10, "maxLength": 10},
        any_whitespace=False,
    )
    restored = xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), tokenizer_info)
    for candidate in (compiled, restored):
        matcher = xgr.GrammarMatcher(candidate, terminate_without_stop_token=True)

        assert matcher.accept_string('"')
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert matcher.fill_next_token_bitmask(bitmask)
        allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
        assert bool(allowed[1])
        assert bool(allowed[2])
        assert not bool(allowed[3])
        assert bool(allowed[4])
        assert not bool(allowed[5])


def test_regex_fsm_cache_distinguishes_json_string_mode():
    tokenizer_info = xgr.TokenizerInfo([])
    grammar = xgr.Grammar.from_ebnf(
        'root ::= plain "!" plain json "!" json\n'
        'plain ::= Regex(".")\n'
        'json ::= Regex(".", json_string=true)'
    )

    for enable_dynamic_compilation in [False, True]:
        compiled = xgr.GrammarCompiler(
            tokenizer_info, max_threads=1, enable_dynamic_compilation=enable_dynamic_compilation
        ).compile_grammar(grammar)
        for value in ['"!ab!c', "\\!ab!c", "a!bc!d"]:
            assert _compiled_accepts(compiled, value)
        for value in ['a!a"!b', "a!a\\!b", "a!ab!\n"]:
            assert not _compiled_accepts(compiled, value)


def test_large_unicode_character_class_matches_eager_and_token_oracle():
    # Keep hundreds of disjoint ranges across two- and three-byte UTF-8 characters.
    # A small repetition bound avoids the separate large-pattern expansion regression.
    codepoints = list(range(0x0100, 0x0300, 2)) + list(range(0x4E00, 0x5000, 2))
    character_class = "".join(f"\\u{codepoint:04X}" for codepoint in codepoints)
    vocabulary = [
        "Ā",
        "一",
        "Ā一",
        "一Ā一",
        "Ā一Ā一",
        "Ā一Ā一Ā",
        "ā",
        "丁",
        "Ā>",
        "一>",
        ">",
        '"',
        'Ā"',
        'Ā">',
        b"\xff",
    ] + [chr(codepoint) for codepoint in codepoints if codepoint not in (0x0100, 0x4E00)]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    schema = {"type": "string", "pattern": rf"^[{character_class}]{{1,4}}$"}
    compiled_grammars = [
        xgr.GrammarCompiler(
            tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=dynamic
        ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)
        for dynamic in (False, True)
    ]

    for length in range(5):
        prefix = '"' + "Ā一Ā一"[:length]
        masks = []
        for compiled in compiled_grammars:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            assert matcher.accept_string(prefix)
            bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
            assert matcher.fill_next_token_bitmask(bitmask)
            allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
            for token_id in range(tokenizer_info.vocab_size):
                oracle = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
                assert oracle.accept_string(prefix)
                assert bool(allowed[token_id]) == oracle.accept_token(token_id), (
                    prefix,
                    vocabulary[token_id],
                )

            # Check the intended language as well as agreement between implementations.
            assert bool(allowed[0]) == (length < 4)
            assert bool(allowed[2]) == (length + 2 <= 4)
            assert bool(allowed[11]) == (length >= 1)
            for token_id in range(15, tokenizer_info.vocab_size):
                assert bool(allowed[token_id]) == (length < 4)
            for token_id in (5, 6, 7, 8, 9, 10, 13, 14):
                assert not bool(allowed[token_id]), (prefix, vocabulary[token_id])
            masks.append(bitmask)
        torch.testing.assert_close(masks[0], masks[1], rtol=0, atol=0)
