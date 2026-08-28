import re
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


def test_existing_native_constructor_remains_available():
    tokenizer_info = xgr.TokenizerInfo(["a"], stop_token_ids=[])
    compiler = _core.GrammarCompiler(tokenizer_info._handle, 1, True, -1)
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

    prefixes = [
        '{"value": "',
        '{"value": "a',
        '{"value": "ab',
        '{"value": "abc',
        r'{"value": "\u0061',
        r'{"value": "\ud83d\ude00a',
    ]
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


def test_cache_disabled_bounded_character_class_uses_repeat_masks():
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

    initial_size = dynamic.memory_size_bytes
    expected = _mask_trace(eager, "ab>")
    actual = _mask_trace(dynamic, "ab>")
    # The closing literal can materialize one ordinary mask, but each bounded repeat count must
    # use the compact repeat-mask path instead of populating another full adaptive mask.
    assert dynamic.memory_size_bytes <= initial_size + 256
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


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_rule_mask_sharing_respects_json_length_entry_and_reuses_inner_rule(
    enable_dynamic_compilation: bool,
):
    tokenizer_info = xgr.TokenizerInfo(['x"aa', 'x""', '"', "a", "aa"], stop_token_ids=[])
    compiler = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=True,
        enable_dynamic_compilation=enable_dynamic_compilation,
    )

    unbounded = compiler.compile_grammar(
        'root ::= "x" target\ntarget ::= "\\"" body "\\""\nbody ::= [a]*'
    )
    matcher = xgr.GrammarMatcher(unbounded, terminate_without_stop_token=True)
    assert matcher.accept_string('x"')
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert matcher.fill_next_token_bitmask(bitmask)

    bounded = compiler.compile_grammar(
        'root ::= "x" target\n'
        'target[json_string_min_length=0, json_string_max_length=1] ::= "\\"" body "\\""\n'
        "body ::= [a]*"
    )
    matcher = xgr.GrammarMatcher(bounded, terminate_without_stop_token=True)
    assert matcher.fill_next_token_bitmask(bitmask)
    allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert not bool(allowed[0])  # This token enters the constrained rule and exceeds maxLength.
    assert bool(allowed[1])

    assert matcher.accept_string('x"')
    assert matcher.fill_next_token_bitmask(bitmask)
    allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert bool(allowed[3])
    assert not bool(allowed[4])  # A reused body mask is still filtered by the active deadline.


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


def test_continuation_transition_cache_isolated_between_parser_states(capfd):
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
    assert capfd.readouterr().err.count("ContinuationTransitionCache(") == 2

    allowed_tokens = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    for token_id in range(tokenizer_info.vocab_size):
        oracle = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert bool(allowed_tokens[token_id]) == oracle.accept_token(token_id), token_id


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_continuation_transition_cache_stops_at_json_length_entry(
    enable_dynamic_compilation: bool, capfd
):
    suffixes = [
        chr(first) + chr(second)
        for first in range(ord("a"), ord("n"))
        for second in range(ord("a"), ord("n"))
    ]
    accepted = suffixes
    valid_continuations = [suffix + '""a"' for suffix in suffixes]
    invalid_continuations = [suffix + '""ab"' for suffix in suffixes]
    rejected = [
        chr(first) + chr(second)
        for first in range(ord("A"), ord("U"))
        for second in range(ord("A"), ord("U"))
    ]
    vocabulary = accepted + valid_continuations + invalid_continuations + rejected
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_grammar(
        "root ::= first free last\n"
        'first[json_string_min_length=1, json_string_max_length=1] ::= "\\"" [a-m] "\\""\n'
        'free ::= "\\"" [a-m]* "\\""\n'
        'last[json_string_min_length=1, json_string_max_length=1] ::= "\\"" [a-m]* "\\""'
    )
    prefix = '"a""'
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string(prefix)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert matcher.fill_next_token_bitmask(bitmask, debug_print=True)

    cache_rows = re.findall(
        r"ContinuationTransitionCache\(queries=(\d+), hits=(\d+)", capfd.readouterr().err
    )
    assert cache_rows
    assert any(int(queries) > 0 and int(hits) > 0 for queries, hits in cache_rows)

    allowed_tokens = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    for token_id in range(tokenizer_info.vocab_size):
        oracle = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert oracle.accept_string(prefix)
        assert bool(allowed_tokens[token_id]) == oracle.accept_token(token_id), token_id


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_continuation_transition_cache_filters_active_json_length(
    enable_dynamic_compilation: bool, capfd
):
    pairs = [
        chr(first) + chr(second)
        for first in range(ord("a"), ord("n"))
        for second in range(ord("a"), ord("n"))
    ]
    vocabulary = (
        ['"', 'a"', "a", "abcd", r"\u0061", r'\u0061"', r'\u0061a"', r'\u0061aa"', r'\u0061aaa"']
        + pairs
        + [pair + '"' for pair in pairs]
    )
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_grammar(
        'root[json_string_min_length=2, json_string_max_length=3] ::= "\\"" body "\\""\n'
        'body ::= [a-m] body | "\\\\u0061" body | ""'
    )

    for prefix in ['"', '"a']:
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert matcher.accept_string(prefix)
        bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        assert matcher.fill_next_token_bitmask(bitmask, debug_print=True)

        cache_rows = re.findall(
            r"ContinuationTransitionCache\(queries=(\d+), hits=(\d+)", capfd.readouterr().err
        )
        assert cache_rows
        assert any(int(queries) > 0 and int(hits) > 0 for queries, hits in cache_rows)

        allowed_tokens = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
        for token_id in range(tokenizer_info.vocab_size):
            oracle = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            assert oracle.accept_string(prefix)
            assert bool(allowed_tokens[token_id]) == oracle.accept_token(token_id), token_id


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_continuation_mask_cache_separates_json_string_lengths(enable_dynamic_compilation: bool):
    pairs = [
        chr(first) + chr(second)
        for first in range(ord("a"), ord("n"))
        for second in range(ord("a"), ord("n"))
    ]
    vocabulary = (
        ['"', 'a"', "a", "abcd", r"\u0061", r'\u0061"', r'\u0061a"', r'\u0061aa"']
        + pairs
        + [pair + '"' for pair in pairs]
    )
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_grammar(
        'root[json_string_min_length=2, json_string_max_length=3] ::= "\\"" body "\\""\n'
        'body ::= [a-m] body | "\\\\u0061" body | ""'
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    assert matcher.accept_string('"')
    assert matcher.fill_next_token_bitmask(bitmask)

    assert matcher.accept_token(vocabulary.index("a"))
    xgr.reset_token_bitmask(bitmask)
    assert matcher.fill_next_token_bitmask(bitmask)
    allowed_tokens = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    for token_id in range(tokenizer_info.vocab_size):
        assert bool(allowed_tokens[token_id]) == matcher.fork().accept_token(token_id), token_id


def test_continuation_mask_cache_classifies_tokens_independently_of_other_states(capfd):
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

    # The 677-token adaptive continuation is printed only for the first fill. Its absence from
    # the second debug trace confirms that this test exercised the matcher-local cache hit.
    assert capfd.readouterr().err.count("uncertain_num=677") == 1


def test_continuation_mask_cache_is_cleared_before_rollback_row_reuse(capfd):
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

    # The repeated character-class state and its parent context are logically unchanged here, so
    # this fill reuses the first result.
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

    # The middle fill is a hit; the first and post-rollback fills are independent misses.
    assert capfd.readouterr().err.count("uncertain_num=290") == 2


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
