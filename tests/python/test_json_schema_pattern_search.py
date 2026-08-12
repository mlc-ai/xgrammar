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
