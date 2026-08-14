"""JSON Schema pattern search, JSON escaping, and large-repeat regressions."""

import json

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string, bitmask_to_bool_mask


def _accepts(pattern: str, value: str) -> bool:
    grammar = xgr.Grammar.from_json_schema(
        json.dumps({"type": "string", "pattern": pattern}), any_whitespace=False
    )
    return _is_grammar_accept_string(grammar, value)


@pytest.mark.parametrize(
    "pattern,value,expected",
    [
        ("abc", '"xabcx"', True),
        ("abc", '"ab"', False),
        ("^abc", '"abcx"', True),
        ("^abc", '"xabc"', False),
        ("abc$", '"xabc"', True),
        ("abc$", '"abcx"', False),
        ("^abc$", '"abc"', True),
        ("^abc$", '"xabc"', False),
        ("^a$|^b$", '"a"', True),
        ("^a$|^b$", '"b"', True),
        ("^a$|^b$", '"ab"', False),
        ("(^a$)|(^b$)", '"b"', True),
        ("(^a$)|(^b$)", '"xb"', False),
        ("^a|b$", '"ax"', True),
        ("^a|b$", '"xb"', True),
        ("^a|b$", '"xa"', False),
        (r"\^a\$", '"x^a$y"', True),
        (r"[\^\$]+", '"x^$y"', True),
    ],
)
def test_pattern_search_and_branch_local_anchors(pattern: str, value: str, expected: bool):
    assert _accepts(pattern, value) is expected


@pytest.mark.parametrize(
    "pattern,value,expected",
    [
        (r'^a"b$', '"a\\"b"', True),
        (r"^a\\b$", '"a\\\\b"', True),
        (r"^ab$", '"a\\u0062"', True),
        (r"^你好$", '"\\u4F60\\u597D"', True),
        (r"^\u{1F600}$", '"😀"', True),
        (r"^\u{1F600}$", '"\\uD83D\\uDE00"', True),
        (r"^\u{1F600}$", '"\\ud83d\\ude00"', True),
        (r"^\u{1F600}$", '"\\uD83D"', False),
        (r"^\u{1F600}$", '"\\uDE00"', False),
        (r"^.$", '"\\q"', False),
    ],
)
def test_pattern_matches_decoded_json_string_characters(pattern: str, value: str, expected: bool):
    assert _accepts(pattern, value) is expected


@pytest.mark.parametrize(
    "schema,value,expected",
    [
        ({"type": "string", "pattern": "foo", "maxLength": 2}, '"foo"', False),
        ({"type": "string", "pattern": "foo", "maxLength": 3}, '"foo"', True),
        ({"type": "string", "pattern": "foo", "maxLength": 3}, '"xfoo"', False),
        ({"type": "string", "pattern": "foo", "minLength": 4}, '"foo"', False),
        ({"type": "string", "pattern": "foo", "minLength": 4}, '"xfoo"', True),
        ({"type": "string", "pattern": "foo", "minLength": 4}, '"foox"', True),
        (
            {"type": "string", "pattern": "^[^x]{0,3}$", "minLength": 2, "maxLength": 5},
            '"a"',
            False,
        ),
        (
            {"type": "string", "pattern": "^[^x]{0,3}$", "minLength": 2, "maxLength": 5},
            '"ab"',
            True,
        ),
        (
            {"type": "string", "pattern": "^[^x]{0,3}$", "minLength": 2, "maxLength": 5},
            '"abcd"',
            False,
        ),
        ({"type": "string", "pattern": "b", "minLength": 2, "maxLength": 2}, '"b"', False),
        ({"type": "string", "pattern": "b", "minLength": 2, "maxLength": 2}, '"ab"', True),
        ({"type": "string", "pattern": "b", "minLength": 2, "maxLength": 2}, '"a\\u0062"', True),
        ({"type": "string", "pattern": "b", "minLength": 2, "maxLength": 2}, '"\\u0062"', False),
    ],
)
def test_pattern_intersects_min_and_max_length(schema: dict, value: str, expected: bool):
    grammar = xgr.Grammar.from_json_schema(json.dumps(schema), any_whitespace=False)
    assert _is_grammar_accept_string(grammar, value) is expected


@pytest.mark.parametrize(
    "value,expected", [('"A"', False), ('"AA"', True), ('"AAA"', True), ('"AAAA"', False)]
)
def test_pattern_length_intersects_unbounded_simple_repeat(value: str, expected: bool):
    grammar = xgr.Grammar.from_json_schema(
        json.dumps({"type": "string", "pattern": "^[A]*$", "minLength": 2, "maxLength": 3}),
        any_whitespace=False,
    )
    assert _is_grammar_accept_string(grammar, value) is expected


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_pattern_max_length_masks_content_at_decoded_boundary(enable_dynamic_compilation: bool):
    vocabulary = ['"', "a", "b", 'b"', "ab"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_json_schema(
        {"type": "string", "pattern": "b", "minLength": 2, "maxLength": 2}, any_whitespace=False
    )

    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string('"a')
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert matcher.fill_next_token_bitmask(bitmask)
    allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert not bool(allowed[0])  # Too short to close.
    assert bool(allowed[2])
    assert bool(allowed[3])  # A token may supply the final character and closing quote together.

    assert matcher.accept_token(2)
    assert matcher.fill_next_token_bitmask(bitmask)
    allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert bool(allowed[0])
    assert not bool(allowed[1])
    assert not bool(allowed[2])
    assert not bool(allowed[4])

    too_short = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert too_short.accept_string('"b')
    assert too_short.fill_next_token_bitmask(bitmask)
    allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert not bool(allowed[0])  # Completion at one decoded character violates minLength.


@pytest.mark.parametrize("value", ['"😀"', '"\\uD83D\\uDE00"'])
def test_pattern_length_counts_surrogate_pair_as_one_decoded_character(value: str):
    schema = {"type": "string", "pattern": ".", "minLength": 1, "maxLength": 1}
    grammar = xgr.Grammar.from_json_schema(json.dumps(schema), any_whitespace=False)
    assert _is_grammar_accept_string(grammar, value)


def test_pattern_length_counter_resets_at_embedded_grammar_boundary():
    prefix = xgr.Grammar.from_ebnf('root ::= ["]')
    constrained = xgr.Grammar.from_json_schema(
        json.dumps({"type": "string", "pattern": "b", "minLength": 2, "maxLength": 2}),
        any_whitespace=False,
    )
    grammar = xgr.Grammar.concat(prefix, constrained)

    # The unmatched quote in the surrounding grammar must not be mistaken for the constrained
    # JSON string's opening quote.
    assert _is_grammar_accept_string(grammar, '""ab"')
    assert not _is_grammar_accept_string(grammar, '""b"')
    assert not _is_grammar_accept_string(grammar, '""abc"')


def test_pattern_length_completion_survives_tail_call_elision():
    grammar = xgr.Grammar.from_ebnf(
        'root ::= "\\"x\\"" constrained\n'
        'constrained[json_string_min_length=1, json_string_max_length=1] ::= "\\"a\\""'
    )

    # The first quoted segment increments the shared decoded-character counter. The tail-called
    # constrained rule must retain its own start position and completion-time length check.
    assert _is_grammar_accept_string(grammar, '"x""a"')


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_pattern_length_rechecks_direct_character_class_mask(enable_dynamic_compilation: bool):
    grammar = xgr.Grammar.from_ebnf(
        'root[json_string_min_length=0, json_string_max_length=0] ::= "\\"" body "\\""\n'
        'body[capture="body"] ::= [a]'
    )
    tokenizer_info = xgr.TokenizerInfo(['"', "a"], stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

    assert matcher.accept_string('"')
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert matcher.fill_next_token_bitmask(bitmask)
    allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert not bool(allowed[1])


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_pattern_length_rechecks_token_that_enters_constrained_rule(
    enable_dynamic_compilation: bool,
):
    grammar = xgr.Grammar.from_ebnf(
        'root ::= "x" constrained\n'
        'constrained[json_string_min_length=0, json_string_max_length=0] ::= "\\"" body "\\""\n'
        "body ::= [a]*"
    )
    tokenizer_info = xgr.TokenizerInfo(['x"a', 'x""'], stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert matcher.fill_next_token_bitmask(bitmask)
    allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert not bool(allowed[0])
    assert bool(allowed[1])


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_pattern_min_length_rechecks_closing_token(enable_dynamic_compilation: bool):
    tokenizer_info = xgr.TokenizerInfo(['"', "a", "b"], stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_json_schema({"type": "string", "pattern": "b", "minLength": 2}, any_whitespace=False)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

    assert matcher.accept_string('"b')
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert matcher.fill_next_token_bitmask(bitmask)
    allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert not bool(allowed[0])
    assert bool(allowed[1])


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_pattern_min_length_rechecks_quote_after_partial_unicode_escape(
    enable_dynamic_compilation: bool,
):
    tokenizer_info = xgr.TokenizerInfo(['0062"', "x"], stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_json_schema({"type": "string", "pattern": ".", "minLength": 3}, any_whitespace=False)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

    assert matcher.accept_string('"a\\u')
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert matcher.fill_next_token_bitmask(bitmask)
    allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert not bool(allowed[0])  # Four raw hex digits decode to only one character.


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_pattern_length_filters_large_cached_accept_set(enable_dynamic_compilation: bool):
    vocabulary = ['"', "b", 'b"', r'\u0062"', r'\u0062x"'] + [f"x{index}" for index in range(1200)]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_json_schema(
        {"type": "string", "pattern": "a", "minLength": 2, "maxLength": 2}, any_whitespace=False
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

    assert matcher.accept_string('"a')
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert matcher.fill_next_token_bitmask(bitmask)
    allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert not bool(allowed[0])  # Closing now violates minLength.
    assert bool(allowed[1])
    assert bool(allowed[2])
    assert bool(allowed[3])  # A six-byte JSON escape adds one decoded character.
    assert not bool(allowed[4])
    assert not bool(allowed[5])  # Cached grammar acceptance still crosses maxLength.


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


def test_pattern_length_preserves_nested_character_budget():
    grammar = xgr.Grammar.from_ebnf(
        'root[json_string_min_length=1, json_string_max_length=2] ::= "\\"" body "\\""\n'
        "body[max_chars=1] ::= [ab]+"
    )

    assert _is_grammar_accept_string(grammar, '"a"')
    assert not _is_grammar_accept_string(grammar, '"ab"')


@pytest.mark.parametrize("enable_dynamic_compilation", [False, True])
def test_pattern_length_rechecks_mask_with_nested_character_budget(
    enable_dynamic_compilation: bool,
):
    grammar = xgr.Grammar.from_ebnf(
        'root[json_string_min_length=0, json_string_max_length=0] ::= "\\"" body "\\""\n'
        "body[max_chars=1] ::= [a]"
    )
    tokenizer_info = xgr.TokenizerInfo(['"', "a"], stop_token_ids=[])
    compiled = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

    assert matcher.accept_string('"')
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert matcher.fill_next_token_bitmask(bitmask)
    allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]
    assert not bool(allowed[1])


def test_pattern_length_jump_forward_stops_at_hard_boundary():
    grammar = xgr.Grammar.from_ebnf(
        'root[json_string_min_length=0, json_string_max_length=1] ::= "\\"ab\\""'
    )
    tokenizer_info = xgr.TokenizerInfo(['"', "a", "b"], stop_token_ids=[])
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

    assert matcher.find_jump_forward_string() == '"a'
    assert matcher.accept_string('"a')
    assert not matcher.accept_string("b")


def test_pattern_length_metadata_round_trips_through_ebnf():
    schema = {"type": "string", "pattern": "b", "minLength": 2, "maxLength": 2}
    grammar = xgr.Grammar.from_json_schema(json.dumps(schema), any_whitespace=False)
    reparsed = xgr.Grammar.from_ebnf(str(grammar))
    assert _is_grammar_accept_string(reparsed, '"ab"')
    assert not _is_grammar_accept_string(reparsed, '"b"')
    assert not _is_grammar_accept_string(reparsed, '"abb"')


def test_pattern_length_metadata_survives_structural_tag_embedding():
    grammar = xgr.Grammar.from_structural_tag(
        {
            "type": "structural_tag",
            "format": {
                "type": "json_schema",
                "json_schema": {"type": "string", "pattern": "b", "minLength": 2, "maxLength": 2},
            },
        }
    )

    assert _is_grammar_accept_string(grammar, '"ab"')
    assert not _is_grammar_accept_string(grammar, '"b"')
    assert not _is_grammar_accept_string(grammar, '"abb"')


def test_pattern_properties_use_search_semantics_and_json_escapes():
    schema = {
        "type": "object",
        "patternProperties": {"x_": {"type": "integer"}},
        "additionalProperties": False,
    }
    grammar = xgr.Grammar.from_json_schema(json.dumps(schema), any_whitespace=False)
    assert _is_grammar_accept_string(grammar, '{"prefix_x_suffix": 1}')
    assert _is_grammar_accept_string(grammar, '{"prefix_\\u0078_suffix": 1}')
    assert not _is_grammar_accept_string(grammar, '{"prefix_x_suffix": "bad"}')
    assert not _is_grammar_accept_string(grammar, '{"before_y_after": 1}')


@pytest.mark.parametrize(
    "pattern,value,expected",
    [
        (r"^[A-F\d]{2,3}$", '"A\\u0039"', True),
        (r"^[A-F\d]{2,3}$", '"\\u0041F0"', True),
        (r"^[A-F\d]{2,3}$", '"\\u0041F09"', False),
        (r"^[A-F\d]{2,3}$", '"A\\u0061"', False),
        (r'^["\\/\n]{4}$', r'"\"\\\/\n"', True),
        (r'^["\\/\n]{4}$', r'"\u0022\u005c\u002f\u000a"', True),
        (r'^["\\/\n]{4}$', r'"\"\\\/x"', False),
        (r"^[A]{0}$", '""', True),
        (r"^[A]{0}$", '"A"', False),
    ],
)
def test_bounded_ascii_character_class_repeat_json_spellings(
    pattern: str, value: str, expected: bool
):
    assert _accepts(pattern, value) is expected


def _large_identifier_pattern() -> str:
    # Hundreds of disjoint Unicode ranges make a single encoded-character FSM large enough that
    # physically unrolling {1,200} would exceed the guard. The GrammarBuilder path must retain a
    # compact repeat edge instead.
    unicode_singletons = "".join(f"\\u{codepoint:04X}" for codepoint in range(0x0100, 0x0800, 2))
    return rf"^[A-Za-z0-9 _\-{unicode_singletons}]{{1,200}}$"


def _compile_large_pattern(enable_dynamic_compilation: bool):
    vocabulary = [" {", "->", "my", "-se", "gment", '",', b"\xff"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[])
    schema = {"type": "string", "pattern": _large_identifier_pattern()}
    compiled = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=1,
        cache_enabled=False,
        enable_dynamic_compilation=enable_dynamic_compilation,
    ).compile_json_schema(schema, any_whitespace=False, strict_mode=True)
    return tokenizer_info, compiled


def test_large_unicode_pattern_stays_compact_and_preserves_mask_oracle():
    eager_info, eager = _compile_large_pattern(enable_dynamic_compilation=False)
    dynamic_info, dynamic = _compile_large_pattern(enable_dynamic_compilation=True)
    assert eager_info.decoded_vocab == dynamic_info.decoded_vocab

    # A CFG expansion of the upper bound creates roughly 200 helper rules. The retained regex
    # repeat needs only the ordinary JSON rules plus one repeated subrule.
    assert len(str(eager.grammar).splitlines()) < 40
    assert len(str(dynamic.grammar).splitlines()) < 40

    expected_masks = []
    actual_masks = []
    accepted_prefix = '"'
    for next_token_id in [2, 3, 4]:
        mode_masks = []
        for compiled, tokenizer_info in [(eager, eager_info), (dynamic, dynamic_info)]:
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            assert matcher.accept_string(accepted_prefix)
            bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
            assert matcher.fill_next_token_bitmask(bitmask)
            allowed = bitmask_to_bool_mask(bitmask, tokenizer_info.vocab_size)[0]

            for token_id in range(tokenizer_info.vocab_size):
                oracle = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
                assert oracle.accept_string(accepted_prefix)
                assert bool(allowed[token_id]) == oracle.accept_token(token_id)

            # These were the two false positives in the original eager-mask failure.
            assert not bool(allowed[0])
            assert not bool(allowed[1])
            assert matcher.accept_token(next_token_id)
            mode_masks.append(allowed)

        expected_masks.append(mode_masks[0])
        actual_masks.append(mode_masks[1])
        accepted_prefix += eager_info.decoded_vocab[next_token_id].decode()

    for expected, actual in zip(expected_masks, actual_masks):
        assert expected.tolist() == actual.tolist()


@pytest.mark.parametrize(
    "value,expected",
    [
        ('"my-segment"', True),
        ('"a b"', True),
        ('"->"', False),
        ('"é"', False),
        ('"Ā"', True),
        ('"a{"', False),
        ('""', False),
    ],
)
def test_large_unicode_pattern_language(value: str, expected: bool):
    assert _accepts(_large_identifier_pattern(), value) is expected
