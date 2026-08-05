import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import xgrammar as xgr
from xgrammar.base import _core

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


CASES = [
    (_compile_ebnf, "hello!"),
    (_compile_json_schema, '{"name":"Ada","scores":[1,2]}'),
    (_compile_regex, "Ada42"),
    (_compile_builtin_json, '{"name":"Ada"}'),
    (_compile_structural_tag, "prefix<call>X"),
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


@pytest.mark.parametrize(
    "compile_grammar,input_string",
    [
        (_compile_json_schema, '{"name":"Ada","scores":[1,2]}'),
        (_compile_structural_tag, "prefix<call>X"),
    ],
    ids=["json-schema", "structural-tag"],
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
