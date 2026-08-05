import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import xgrammar as xgr
from xgrammar.base import _core
from xgrammar.testing import bitmask_to_bool_mask

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


def _next_token_mask(matcher: xgr.GrammarMatcher, vocabulary_size: int) -> torch.Tensor:
    bitmask = xgr.allocate_token_bitmask(1, vocabulary_size)
    matcher.fill_next_token_bitmask(bitmask)
    return bitmask_to_bool_mask(bitmask, vocabulary_size)


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


def _compile_repeated_character_class(compiler: xgr.GrammarCompiler) -> xgr.CompiledGrammar:
    return compiler.compile_grammar("root ::= [a-z]{1,}")


CASES = [
    (_compile_ebnf, "hello!"),
    (_compile_json_schema, '{"name":"Ada","scores":[1,2]}'),
    (_compile_regex, "Ada42"),
    (_compile_builtin_json, '{"name":"Ada"}'),
    (_compile_structural_tag, "prefix<call>X"),
    (_compile_repeated_character_class, "hello"),
]


def test_native_constructor_takes_dynamic_compilation_flag():
    tokenizer_info = xgr.TokenizerInfo(["a"], stop_token_ids=[])
    for enable_dynamic_compilation in (False, True):
        compiler = _core.GrammarCompiler(
            tokenizer_info._handle, 1, True, -1, enable_dynamic_compilation
        )
        compiler.compile_builtin_json_grammar()


@pytest.mark.parametrize(
    "compile_grammar,input_string",
    CASES,
    ids=[
        "ebnf",
        "json-schema",
        "regex",
        "builtin-json",
        "structural-tag",
        "repeated-character-class",
    ],
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


def test_serialization_roundtrips_dynamic_mode():
    tokenizer_info = xgr.TokenizerInfo(VOCABULARY, stop_token_ids=[])
    dynamic = _compile_ebnf(
        xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=True)
    )
    eager = _compile_ebnf(
        xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=False)
    )
    initial_size = dynamic.memory_size_bytes

    restored = xgr.CompiledGrammar.deserialize_json(dynamic.serialize_json(), tokenizer_info)
    # Serialization does not materialize masks, and the restored grammar is dynamic again.
    assert dynamic.memory_size_bytes == initial_size
    restored_initial_size = restored.memory_size_bytes
    assert restored_initial_size < eager.memory_size_bytes
    for input_string in ["hello!", "hi?"]:
        expected = _mask_trace(eager, input_string)
        actual = _mask_trace(restored, input_string)
        for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
            assert actual_apply == expected_apply
            torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)
    assert restored.memory_size_bytes > restored_initial_size


def test_serialization_roundtrips_dynamic_tag_dispatch():
    tokenizer_info = xgr.TokenizerInfo(VOCABULARY, stop_token_ids=[])
    dynamic = _compile_structural_tag(
        xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=True)
    )
    eager = _compile_structural_tag(
        xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=False)
    )

    restored = xgr.CompiledGrammar.deserialize_json(dynamic.serialize_json(), tokenizer_info)
    expected = _mask_trace(eager, "prefix<call>X")
    actual = _mask_trace(restored, "prefix<call>X")
    for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
        assert actual_apply == expected_apply
        torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


def test_dynamic_tag_dispatch_slicing_uses_rule_level_cache():
    tokenizer_info = xgr.TokenizerInfo(VOCABULARY, stop_token_ids=[])
    cache_limit = 64 * 1024
    compiler = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=True,
        cache_limit_bytes=cache_limit,
        enable_dynamic_compilation=True,
    )

    def dispatch(value, trigger="<call>"):
        return {
            "type": "structural_tag",
            "format": {
                "type": "dispatch",
                "rules": [[trigger, {"type": "const_string", "value": value}]],
                "loop": False,
                "excludes": [],
            },
        }

    def compile_dispatch(value, trigger="<call>"):
        return compiler.compile_structural_tag(dispatch(value, trigger))

    first = compile_dispatch("X")
    size_before_first_use = compiler.get_cache_size_bytes()
    _mask_trace(first, "prefix<call>X")
    size_after_first_use = compiler.get_cache_size_bytes()
    assert size_before_first_use < size_after_first_use <= cache_limit

    second = compiler.compile_structural_tag(json.dumps(dispatch("X"), indent=2))
    size_before_second_use = compiler.get_cache_size_bytes()
    _mask_trace(second, "prefix<call>X")
    assert compiler.get_cache_size_bytes() == size_before_second_use

    distinct_patterns = [compile_dispatch("X", trigger=f"<call{index}>") for index in range(32)]
    for compiled_grammar in distinct_patterns:
        matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(bitmask)
    size_after_distinct_patterns = compiler.get_cache_size_bytes()
    assert 0 < size_after_distinct_patterns <= cache_limit

    compiler.clear_cache()
    assert compiler.get_cache_size_bytes() == 0


@pytest.mark.parametrize(
    "compile_grammar,input_string",
    [
        (_compile_json_schema, '{"name":"Ada","scores":[1,2]}'),
        (_compile_structural_tag, "prefix<call>X"),
        (_compile_repeated_character_class, "hello"),
    ],
    ids=["json-schema", "structural-tag", "repeated-character-class"],
)
def test_concurrent_dynamic_mask_generation(compile_grammar, input_string):
    tokenizer_info = xgr.TokenizerInfo(VOCABULARY, stop_token_ids=[])
    dynamic = compile_grammar(
        xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=True)
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        traces = list(executor.map(lambda _: _mask_trace(dynamic, input_string), range(32)))

    expected = traces[0]
    for actual in traces[1:]:
        for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
            assert actual_apply == expected_apply
            torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


def test_limited_compiler_cache_records_dynamic_grammar_size_on_insertion():
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
    insertion_size = compiler.get_cache_size_bytes()
    assert 0 < insertion_size <= cache_limit

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

    assert all(insertion_size <= size <= cache_limit for size in observed_sizes)
    assert insertion_size <= compiler.get_cache_size_bytes() <= cache_limit
    assert _compile_builtin_json(compiler).memory_size_bytes == dynamic.memory_size_bytes


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


def test_shared_parser_features_preserve_budget_and_capture_behavior():
    tokenizer_info = xgr.TokenizerInfo(["ab ", "cd", " ", "</t>", "1", "<t>", "x"])
    grammar = xgr.Grammar.from_lark(
        'start: r "<t>"\nr[max_tokens=3, capture]: TEXT\nTEXT: /(\\n|.)*/',
        tokenizer_info=tokenizer_info,
    )
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
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
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo([])).compile_grammar(grammar)
    for _ in range(4):
        for value, expected_captures in [
            ("a!z", [("inner", b"a"), ("outer", b"a!z")]),
            ("abz", [("inner", b"ab"), ("outer", b"abz")]),
        ]:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            assert matcher.accept_string(value) and matcher.is_terminated()
            assert matcher.get_captures() == expected_captures


def test_reusable_parser_worklists_survive_reset_fork_failure_and_rollback():
    vocabulary = ["a", "b", "c", "d", "ab", "bc", "abc", "cd", "x", b"\xff"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiled_grammar = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar('root ::= [a-c]{0,8} "d"')
    token_identifiers = {token: vocabulary.index(token) for token in ["a", "b", "c", "d", "x"]}

    def fresh_mask(prefix):
        fresh_matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        assert all(fresh_matcher.accept_token(token_identifier) for token_identifier in prefix)
        return _next_token_mask(fresh_matcher, tokenizer_info.vocab_size)

    matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    for _ in range(32):
        matcher.reset()
        prefix = []
        for token in ["a", "b"]:
            expected_mask = fresh_mask(prefix)
            torch.testing.assert_close(
                _next_token_mask(matcher, tokenizer_info.vocab_size), expected_mask, rtol=0, atol=0
            )
            torch.testing.assert_close(
                _next_token_mask(matcher, tokenizer_info.vocab_size), expected_mask, rtol=0, atol=0
            )
            assert matcher.accept_token(token_identifiers[token])
            prefix.append(token_identifiers[token])

        forked_matcher = matcher.fork()
        torch.testing.assert_close(
            _next_token_mask(forked_matcher, tokenizer_info.vocab_size),
            fresh_mask(prefix),
            rtol=0,
            atol=0,
        )
        assert forked_matcher.accept_token(token_identifiers["c"])
        assert forked_matcher.accept_token(token_identifiers["d"])
        assert forked_matcher.is_terminated()

        mask_before_failure = _next_token_mask(matcher, tokenizer_info.vocab_size)
        assert not matcher.accept_token(token_identifiers["x"])
        torch.testing.assert_close(
            _next_token_mask(matcher, tokenizer_info.vocab_size),
            mask_before_failure,
            rtol=0,
            atol=0,
        )

        matcher.rollback(1)
        prefix.pop()
        torch.testing.assert_close(
            _next_token_mask(matcher, tokenizer_info.vocab_size), fresh_mask(prefix), rtol=0, atol=0
        )
        assert matcher.accept_token(token_identifiers["c"])
        assert matcher.accept_token(token_identifiers["d"])
        assert matcher.is_terminated()


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


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True], ids=["eager", "dynamic"])
@pytest.mark.parametrize(
    "repeat_range,accepted_values,rejected_values",
    [
        ("{0}", [""], ["a"]),
        ("{0,2}", ["", "aa"], ["aaa"]),
        ("{1,3}", ["a", "aaa"], ["", "aaaa"]),
        ("{2,}", ["aa", "aaaa"], ["", "a"]),
    ],
)
def test_preserved_repetition_ranges_enforce_bounds(
    enable_dynamic_compilation, repeat_range, accepted_values, rejected_values
):
    tokenizer_info = xgr.TokenizerInfo([], stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=enable_dynamic_compilation
    ).compile_grammar(f'root ::= unit{repeat_range}\nunit ::= "a"')

    for value in accepted_values:
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert matcher.accept_string(value)
        assert matcher.is_terminated()
    for value in rejected_values:
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert not (matcher.accept_string(value) and matcher.is_terminated())


@pytest.mark.parametrize(
    "grammar,accepted_values,rejected_values",
    [
        ('root ::= unit{1} "x" | unit{1} "y"\nunit ::= "a"', ["ax", "ay"], []),
        ('root ::= unit{1} "x" | unit{2} "y"\nunit ::= "a"', ["ax", "aay"], ["aax", "ay"]),
    ],
    ids=["same-bounds", "different-bounds"],
)
def test_preserved_repetition_ranges_keep_shared_repeat_edges(
    grammar, accepted_values, rejected_values
):
    tokenizer_info = xgr.TokenizerInfo(["a", "x", "y"], stop_token_ids=[])
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar(grammar)

    for value in accepted_values:
        expected = _mask_trace(eager, value)
        actual = _mask_trace(dynamic, value)
        for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
            assert actual_apply == expected_apply
            expected_tokens = bitmask_to_bool_mask(expected_mask, tokenizer_info.vocab_size)
            actual_tokens = bitmask_to_bool_mask(actual_mask, tokenizer_info.vocab_size)
            torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)

    for value in rejected_values:
        for compiled_grammar in [eager, dynamic]:
            matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
            assert not (matcher.accept_string(value) and matcher.is_terminated())


@pytest.mark.parametrize(
    "grammar,accepted_values,rejected_values",
    [
        ('root ::= unit{2} "x" | "y"\nunit ::= "a"', ["aax", "y"], ["ay", "ax", "aay"]),
        (
            'root ::= a{1,2} | b{2,3}\na ::= "a"\nb ::= "b"',
            ["a", "aa", "bb", "bbb"],
            ["b", "ab", "ba", "aaa", "bbbb"],
        ),
        (
            'root ::= pair{2,3} "x"\npair ::= unit{1,2}\nunit ::= "a"',
            ["aax", "aaax", "aaaax", "aaaaax", "aaaaaax"],
            ["x", "ax", "aaaaaaax"],
        ),
        (
            'root ::= unit{2,3} "x"\nunit ::= "" | "a"',
            ["x", "ax", "aax", "aaax"],
            ["a", "aaaax", "xx"],
        ),
        ('root ::= unit{2,} "x"\nunit ::= "" | "a"', ["x", "ax", "aax", "aaaaax"], ["a", "xx"]),
    ],
    ids=[
        "repeat-and-literal",
        "different-repeated-rules",
        "nested-repeats",
        "bounded-nullable-repeat",
        "unbounded-nullable-repeat",
    ],
)
def test_preserved_repetition_ranges_keep_repeat_branches_isolated(
    grammar, accepted_values, rejected_values
):
    tokenizer_info = xgr.TokenizerInfo(["a", "b", "x", "y"], stop_token_ids=[])
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar(grammar)

    for value in accepted_values:
        expected = _mask_trace(eager, value)
        actual = _mask_trace(dynamic, value)
        for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
            assert actual_apply == expected_apply
            expected_tokens = bitmask_to_bool_mask(expected_mask, tokenizer_info.vocab_size)
            actual_tokens = bitmask_to_bool_mask(actual_mask, tokenizer_info.vocab_size)
            torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)

    for value in rejected_values:
        eager_matcher = xgr.GrammarMatcher(eager, terminate_without_stop_token=True)
        dynamic_matcher = xgr.GrammarMatcher(dynamic, terminate_without_stop_token=True)
        assert not (eager_matcher.accept_string(value) and eager_matcher.is_terminated())
        assert not (dynamic_matcher.accept_string(value) and dynamic_matcher.is_terminated())


def test_preserved_repetition_ranges_survive_serialization():
    grammar = 'root ::= unit{2} "x" | unit{3} "y"\nunit ::= "a"'
    tokenizer_info = xgr.TokenizerInfo(["a", "x", "y"], stop_token_ids=[])
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar(grammar)
    restored = xgr.CompiledGrammar.deserialize_json(dynamic.serialize_json(), tokenizer_info)

    for value in ["aax", "aaay"]:
        expected = _mask_trace(eager, value)
        actual = _mask_trace(restored, value)
        for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
            assert actual_apply == expected_apply
            torch.testing.assert_close(
                bitmask_to_bool_mask(actual_mask, tokenizer_info.vocab_size),
                bitmask_to_bool_mask(expected_mask, tokenizer_info.vocab_size),
                rtol=0,
                atol=0,
            )


def test_preserved_repetition_ranges_survive_rollback():
    tokenizer_info = xgr.TokenizerInfo(["a", "x"], stop_token_ids=[])
    compiled_grammar = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar('root ::= unit{1,3} "x"\nunit ::= "a"')

    def mask_after_one_a(matcher):
        assert matcher.accept_token(0)
        return _next_token_mask(matcher, tokenizer_info.vocab_size)

    matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    expected = mask_after_one_a(
        xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    )
    assert matcher.accept_token(0)
    assert matcher.accept_token(0)
    matcher.rollback(1)
    torch.testing.assert_close(
        _next_token_mask(matcher, tokenizer_info.vocab_size), expected, rtol=0, atol=0
    )


def test_rule_mask_sharing_distinguishes_repeat_bounds():
    tokenizer_info = xgr.TokenizerInfo(["p", "a", "x", "z"], stop_token_ids=[])
    cached_compiler = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=True, enable_dynamic_compilation=True
    )
    uncached_compiler = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, cache_enabled=False, enable_dynamic_compilation=True
    )

    def grammar(repeat_count):
        return (
            f'root ::= "p" recursive "z"\n'
            f'recursive ::= unit{{{repeat_count}}} recursive | "x"\n'
            'unit ::= "a"'
        )

    def mask_after_prefix(compiler, grammar_source):
        matcher = xgr.GrammarMatcher(
            compiler.compile_grammar(grammar_source), terminate_without_stop_token=True
        )
        assert matcher.accept_token(0)
        assert matcher.accept_token(1)
        return _next_token_mask(matcher, tokenizer_info.vocab_size)

    first_mask = mask_after_prefix(cached_compiler, grammar(1))
    actual_mask = mask_after_prefix(cached_compiler, grammar(2))
    expected_mask = mask_after_prefix(uncached_compiler, grammar(2))

    assert not torch.equal(first_mask, expected_mask)
    torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)


@pytest.mark.parametrize(
    "grammar",
    [r"root ::= [^\u0000]{1,}", r"root ::= [\u0000-\U0010FFFF]{1,}", r"root ::= [a-ca-b]{1,}"],
    ids=["negative", "positive", "overlapping-ranges"],
)
def test_repeated_character_class_masks_follow_fsm_semantics(grammar):
    vocabulary = [b"\xc0\x80", b"\xc0", b"\xed\xa0\x80", b"\xf5\x80\x80\x80", "a", "\x00"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar(grammar)

    def initial_mask(compiled_grammar):
        matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        xgr.reset_token_bitmask(bitmask)
        assert matcher.fill_next_token_bitmask(bitmask)
        return bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)

    torch.testing.assert_close(initial_mask(dynamic), initial_mask(eager), rtol=0, atol=0)
    for token_id in range(tokenizer_info.vocab_size):
        eager_matcher = xgr.GrammarMatcher(eager, terminate_without_stop_token=True)
        dynamic_matcher = xgr.GrammarMatcher(dynamic, terminate_without_stop_token=True)
        assert dynamic_matcher.accept_token(token_id) == eager_matcher.accept_token(token_id)


@pytest.mark.parametrize("operation", ["rollback", "reset", "fork"])
def test_repeated_character_class_masks_do_not_leak_across_state_changes(operation):
    tokenizer_info = xgr.TokenizerInfo(["p", "q", "a", "x", "y"], stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar('root ::= "p" repeated "x" | "q" repeated "y"\nrepeated ::= [a]{1,}')

    def next_token_ids(matcher):
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        xgr.reset_token_bitmask(bitmask)
        assert matcher.fill_next_token_bitmask(bitmask)
        return set(
            bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0].nonzero().flatten().tolist()
        )

    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(0)
    assert matcher.accept_token(2)
    assert next_token_ids(matcher) == {2, 3}

    if operation == "rollback":
        matcher.rollback(2)
    elif operation == "reset":
        matcher.reset()
    else:
        matcher = matcher.fork()
        matcher.rollback(2)

    assert matcher.accept_token(1)
    assert matcher.accept_token(2)
    assert next_token_ids(matcher) == {2, 4}


def test_repeated_character_class_masks_are_shared_by_compiled_grammar():
    tokenizer_info = xgr.TokenizerInfo(["a", "ab", "b", "x"], stop_token_ids=[])
    compiled = _compile_repeated_character_class(
        xgr.GrammarCompiler(tokenizer_info, max_threads=1, enable_dynamic_compilation=True)
    )

    initial_size = compiled.memory_size_bytes
    _mask_trace(compiled, "a")
    populated_size = compiled.memory_size_bytes
    assert populated_size > initial_size
    _mask_trace(compiled, "b")
    assert compiled.memory_size_bytes == populated_size


def test_serialization_rebuilds_repeated_character_class_masks():
    tokenizer_info = xgr.TokenizerInfo(["a", "ab", "b", "x"], stop_token_ids=[])
    grammar = "root ::= [a-z]{1,3}"
    dynamic = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=True
    ).compile_grammar(grammar)
    eager = xgr.GrammarCompiler(
        tokenizer_info, max_threads=1, enable_dynamic_compilation=False
    ).compile_grammar(grammar)
    restored = xgr.CompiledGrammar.deserialize_json(dynamic.serialize_json(), tokenizer_info)

    expected = _mask_trace(eager, "ab")
    actual = _mask_trace(restored, "ab")
    for (expected_apply, expected_mask), (actual_apply, actual_mask) in zip(expected, actual):
        assert actual_apply == expected_apply
        expected_tokens = bitmask_to_bool_mask(expected_mask, tokenizer_info.vocab_size)
        actual_tokens = bitmask_to_bool_mask(actual_mask, tokenizer_info.vocab_size)
        torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)
