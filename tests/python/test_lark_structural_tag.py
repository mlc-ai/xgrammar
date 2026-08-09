import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, Sequence, Tuple

import pytest
import torch

import xgrammar as xgr


def _structural_tag(format_: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "structural_tag", "format": format_}


def _inline_source(payload: Dict[str, Any]) -> str:
    return "start: %structural_tag " + json.dumps(payload, ensure_ascii=False)


def _compile_pair(
    payload: Dict[str, Any], tokenizer_info: xgr.TokenizerInfo
) -> Tuple[xgr.CompiledGrammar, xgr.CompiledGrammar]:
    direct_compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)
    inline_compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)
    direct = direct_compiler.compile_structural_tag(payload)
    inline_grammar = xgr.Grammar.from_lark(_inline_source(payload), tokenizer_info=tokenizer_info)
    inline = inline_compiler.compile_grammar(inline_grammar)
    return direct, inline


def _mask_snapshot(
    matcher: xgr.GrammarMatcher, tokenizer_info: xgr.TokenizerInfo
) -> Tuple[bool, torch.Tensor]:
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    xgr.reset_token_bitmask(bitmask)
    need_apply = matcher.fill_next_token_bitmask(bitmask)
    return need_apply, bitmask.clone()


def _assert_token_trace_equal(
    direct: xgr.CompiledGrammar,
    inline: xgr.CompiledGrammar,
    tokenizer_info: xgr.TokenizerInfo,
    token_ids: Sequence[int],
    expected_accepted: bool,
) -> None:
    direct_matcher = xgr.GrammarMatcher(direct, terminate_without_stop_token=True)
    inline_matcher = xgr.GrammarMatcher(inline, terminate_without_stop_token=True)

    direct_state = direct_matcher._debug_print_internal_state()
    inline_state = inline_matcher._debug_print_internal_state()
    assert direct_matcher.validate_tokens(token_ids) == inline_matcher.validate_tokens(token_ids)
    assert direct_matcher._debug_print_internal_state() == direct_state
    assert inline_matcher._debug_print_internal_state() == inline_state

    all_tokens_accepted = True
    for token_id in token_ids:
        assert direct_matcher.is_completed() == inline_matcher.is_completed()
        assert direct_matcher.is_terminated() == inline_matcher.is_terminated()
        direct_apply, direct_mask = _mask_snapshot(direct_matcher, tokenizer_info)
        inline_apply, inline_mask = _mask_snapshot(inline_matcher, tokenizer_info)
        assert direct_apply == inline_apply
        torch.testing.assert_close(direct_mask, inline_mask, rtol=0, atol=0)
        direct_accepted = direct_matcher.accept_token(token_id)
        inline_accepted = inline_matcher.accept_token(token_id)
        assert direct_accepted == inline_accepted
        if not direct_accepted:
            all_tokens_accepted = False
            break

    assert direct_matcher.is_completed() == inline_matcher.is_completed()
    assert direct_matcher.is_terminated() == inline_matcher.is_terminated()
    actually_accepted = all_tokens_accepted and direct_matcher.is_terminated()
    assert actually_accepted == expected_accepted


def _character_tokenizer(values: Iterable[str]) -> Tuple[xgr.TokenizerInfo, Dict[str, int]]:
    characters = list(dict.fromkeys(character for value in values for character in value))
    characters.extend(character for character in ["§", "¤", "?"] if character not in characters)
    tokenizer_info = xgr.TokenizerInfo(characters, stop_token_ids=[])
    return tokenizer_info, {character: index for index, character in enumerate(characters)}


def _assert_character_language_equal(
    format_: Dict[str, Any], accepted: Sequence[str], rejected: Sequence[str]
) -> None:
    tokenizer_info, token_ids = _character_tokenizer([*accepted, *rejected])
    direct, inline = _compile_pair(_structural_tag(format_), tokenizer_info)
    for value in accepted:
        _assert_token_trace_equal(
            direct, inline, tokenizer_info, [token_ids[character] for character in value], True
        )
    for value in rejected:
        _assert_token_trace_equal(
            direct, inline, tokenizer_info, [token_ids[character] for character in value], False
        )


FORMAT_CASES = [
    pytest.param(
        {"type": "const_string", "value": "OK"}, ["OK"], ["", "O", "NO"], id="const-string"
    ),
    pytest.param(
        {"type": "any_text", "excludes": ["BAD"]},
        ["", "hello", "BA"],
        ["BAD", "xBADy"],
        id="any-text-excludes",
    ),
    pytest.param(
        {"type": "regex", "pattern": "a(?:b|c)+"},
        ["ab", "acbc"],
        ["", "a", "ad"],
        id="regex-unicode",
    ),
    pytest.param(
        {"type": "grammar", "grammar": 'root ::= "R" item\nitem ::= "x" | "x" item'},
        ["Rx", "Rxxxx"],
        ["", "R", "Ry"],
        id="grammar-recursive",
    ),
    pytest.param(
        {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            },
        },
        ['{"x":1}', '{ "x" : -2 }'],
        ["{}", '{"x":"1"}'],
        id="json-schema-json",
    ),
    pytest.param(
        {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            },
            "any_order": True,
            "max_whitespace_cnt": 1,
        },
        ['{"y":2,"x":1}', '{ "x":1, "y":2 }'],
        ['{  "x":1,"y":2}', '{"x":"1","y":2}'],
        id="json-schema-any-order-whitespace",
    ),
    pytest.param(
        {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            "style": "qwen_xml",
        },
        ["<parameter=x>v</parameter>"],
        ["<parameter=x>v"],
        id="json-schema-qwen-xml",
    ),
    pytest.param(
        {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            "style": "minimax_xml",
        },
        ['<parameter name="x">v</parameter>'],
        ["<parameter=x>v</parameter>"],
        id="json-schema-minimax-xml",
    ),
    pytest.param(
        {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            "style": "deepseek_xml",
        },
        ['<｜DSML｜parameter name="x" string="true">v</｜DSML｜parameter>'],
        ["<parameter=x>v</parameter>"],
        id="json-schema-deepseek-xml",
    ),
    pytest.param(
        {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            "style": "glm_xml",
        },
        ["<arg_key>x</arg_key><arg_value>v</arg_value>"],
        ["<parameter=x>v</parameter>"],
        id="json-schema-glm-xml",
    ),
    pytest.param(
        {
            "type": "qwen_xml_parameter",
            "json_schema": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        },
        ["<parameter=x>v</parameter>"],
        ["<parameter=y>v</parameter>"],
        id="deprecated-qwen-xml-parameter",
    ),
    pytest.param(
        {
            "type": "sequence",
            "elements": [
                {"type": "const_string", "value": "A"},
                {"type": "regex", "pattern": "[0-9]+"},
                {"type": "const_string", "value": "B"},
            ],
        },
        ["A1B", "A123B"],
        ["AB", "A1"],
        id="sequence",
    ),
    pytest.param(
        {
            "type": "or",
            "elements": [
                {"type": "const_string", "value": "LEFT"},
                {"type": "const_string", "value": "RIGHT"},
            ],
        },
        ["LEFT", "RIGHT"],
        ["", "LEFTRIGHT"],
        id="or",
    ),
    pytest.param(
        {"type": "optional", "content": {"type": "const_string", "value": "O"}},
        ["", "O"],
        ["OO", "X"],
        id="optional",
    ),
    pytest.param(
        {"type": "plus", "content": {"type": "const_string", "value": "P"}},
        ["P", "PPP"],
        ["", "Q"],
        id="plus",
    ),
    pytest.param(
        {"type": "star", "content": {"type": "const_string", "value": "S"}},
        ["", "SSS"],
        ["T", "SST"],
        id="star",
    ),
    pytest.param(
        {"type": "repeat", "min": 2, "max": 3, "content": {"type": "const_string", "value": "r"}},
        ["rr", "rrr"],
        ["", "r", "rrrr"],
        id="repeat-bounded",
    ),
    pytest.param(
        {
            "type": "tag",
            "begin": "<t>",
            "content": {"type": "any_text", "excludes": ["</t>"]},
            "end": ["</t>", "</end>"],
        },
        ["<t>x</t>", "<t>body</end>"],
        ["<t>x", "x</t>"],
        id="tag-multiple-end",
    ),
    pytest.param(
        {
            "type": "triggered_tags",
            "triggers": ["<a"],
            "tags": [
                {
                    "type": "tag",
                    "begin": "<a>",
                    "content": {"type": "const_string", "value": "X"},
                    "end": "</a>",
                }
            ],
            "at_least_one": True,
            "stop_after_first": False,
            "excludes": ["BLOCK"],
        },
        ["<a>X</a>", "<a>X</a>post<a>X</a>"],
        ["free", "pre<a>X</a>", "BLOCK<a>X</a>"],
        id="triggered-tags-dispatch",
    ),
    pytest.param(
        {
            "type": "tags_with_separator",
            "tags": [
                {
                    "type": "tag",
                    "begin": "<a>",
                    "content": {"type": "const_string", "value": "X"},
                    "end": "</a>",
                },
                {
                    "type": "tag",
                    "begin": "<b>",
                    "content": {"type": "const_string", "value": "Y"},
                    "end": "</b>",
                },
            ],
            "separator": ",",
            "at_least_one": True,
            "stop_after_first": False,
        },
        ["<a>X</a>", "<a>X</a>,<b>Y</b>"],
        ["", "<a>X</a><b>Y</b>"],
        id="tags-with-separator",
    ),
    pytest.param(
        {
            "type": "dispatch",
            "rules": [["@", {"type": "const_string", "value": "X"}]],
            "loop": False,
            "excludes": ["BLOCK"],
        },
        ["@X", "free@X"],
        ["@Y", "BLOCK@X"],
        id="string-dispatch",
    ),
]


@pytest.mark.parametrize("format_,accepted,rejected", FORMAT_CASES)
def test_lark_structural_tag_all_character_formats_match_direct_entry(
    format_: Dict[str, Any], accepted: Sequence[str], rejected: Sequence[str]
) -> None:
    _assert_character_language_equal(format_, accepted, rejected)


def _tokenizer_with_fixed_ids() -> xgr.TokenizerInfo:
    vocabulary = [f"unused-{index}" for index in range(128)]
    vocabulary[1] = "A"
    vocabulary[2] = "B"
    vocabulary[3] = "C"
    vocabulary[10] = "<trigger>"
    vocabulary[20] = "<other>"
    vocabulary[50] = "<excluded>"
    vocabulary[99] = "<end>"
    vocabulary[110] = "<trigger>"
    return xgr.TokenizerInfo(vocabulary, stop_token_ids=[127])


TOKEN_FORMAT_CASES = [
    pytest.param({"type": "token", "token": 10}, [[10]], [[20], []], id="token-id"),
    pytest.param(
        {"type": "token", "token": "<trigger>"}, [[10]], [[20], [110], []], id="token-string"
    ),
    pytest.param(
        {"type": "exclude_token", "exclude_tokens": [20, "<excluded>"]},
        [[1], [10]],
        [[20], [50], []],
        id="exclude-token",
    ),
    pytest.param(
        {"type": "any_tokens", "exclude_tokens": [20, "<excluded>"]},
        [[], [1], [1, 2, 3]],
        [[1, 20], [50]],
        id="any-tokens",
    ),
    pytest.param(
        {
            "type": "tag",
            "begin": {"type": "token", "token": "<trigger>"},
            "content": {"type": "const_string", "value": "A"},
            "end": {"type": "token", "token": 99},
        },
        [[10, 1, 99]],
        [[110, 1, 99], [10, 2, 99], [10, 1]],
        id="token-tag-string-resolution",
    ),
    pytest.param(
        {
            "type": "token_dispatch",
            "rules": [
                [10, {"type": "const_string", "value": "A"}],
                [20, {"type": "const_string", "value": "B"}],
            ],
            "loop": False,
            "exclude_tokens": [50],
        },
        [[10, 1], [20, 2], [3, 10, 1]],
        [[10, 2], [50, 10, 1]],
        id="token-dispatch",
    ),
    pytest.param(
        {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<trigger>", 20],
            "tags": [
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<trigger>"},
                    "content": {"type": "const_string", "value": "A"},
                    "end": {"type": "token", "token": 99},
                },
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": 20},
                    "content": {"type": "const_string", "value": "B"},
                    "end": {"type": "token", "token": 99},
                },
            ],
            "exclude_tokens": [50],
            "at_least_one": True,
            "stop_after_first": True,
        },
        [[10, 1, 99], [20, 2, 99]],
        [[3], [110, 1, 99], [10, 2, 99], [50, 10, 1, 99]],
        id="token-triggered-tags",
    ),
]


@pytest.mark.parametrize("format_,accepted,rejected", TOKEN_FORMAT_CASES)
def test_lark_structural_tag_all_token_formats_match_direct_entry(
    format_: Dict[str, Any], accepted: Sequence[Sequence[int]], rejected: Sequence[Sequence[int]]
) -> None:
    tokenizer_info = _tokenizer_with_fixed_ids()
    direct, inline = _compile_pair(_structural_tag(format_), tokenizer_info)
    for token_ids in accepted:
        _assert_token_trace_equal(direct, inline, tokenizer_info, token_ids, True)
    for token_ids in rejected:
        _assert_token_trace_equal(direct, inline, tokenizer_info, token_ids, False)


def test_lark_structural_tag_lexing_composition_and_cache() -> None:
    compact = '{"format":{"value":"X","type":"const_string"}}'
    reordered = '{ "format" : { "type" : "const_string", "value" : "X" } }'
    source = f"""\
start: "A" item (item | "Y") "B"
item: %structural_tag{compact}
    | (%structural_tag {reordered})
"""
    grammar = xgr.Grammar.from_lark(source)
    printed = str(grammar)
    assert printed.count("const_string ::=") == 1
    tokenizer_info, token_ids = _character_tokenizer(["AXXB", "AXYB", "AYB"])
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    for value in ["AXXB", "AXYB"]:
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert all(matcher.accept_token(token_ids[char]) for char in value)
        assert matcher.is_terminated()
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert not (
        all(matcher.accept_token(token_ids[char]) for char in "AYB") and matcher.is_terminated()
    )

    distinct = xgr.Grammar.from_lark(
        f"start: (%structural_tag{compact}) (%structural_tag "
        '{"format":{"type":"const_string","value":"Z"}})'
    )
    distinct_printed = str(distinct)
    assert distinct_printed.count("const_string ::=") == 1
    assert distinct_printed.count("const_string_1 ::=") == 1

    tight = xgr.Grammar.from_lark(f'start: (%structural_tag{compact})"Y"')
    tokenizer_info, token_ids = _character_tokenizer(["XY", "X", "Y"])
    tight_compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(tight)
    for value, accepted in [("XY", True), ("X", False), ("Y", False)]:
        matcher = xgr.GrammarMatcher(tight_compiled, terminate_without_stop_token=True)
        result = all(matcher.accept_token(token_ids[character]) for character in value)
        assert (result and matcher.is_terminated()) == accepted


def test_lark_structural_tag_nested_named_ignore_and_scope_isolation() -> None:
    payload = _structural_tag({"type": "const_string", "value": "AB"})
    payload_json = json.dumps(payload)
    source = f"""\
%import common.WS_INLINE
%ignore WS_INLINE
start: "[" %lark {{ start: %structural_tag {payload_json} }} @named "]"
"""
    grammar = xgr.Grammar.from_lark(
        source, named_grammars={"named": f"start: %structural_tag {payload_json}"}
    )
    tokenizer_info, token_ids = _character_tokenizer(["[ABAB]", "[ AB AB ]", "[A BAB]"])
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    for value, accepted in [("[ABAB]", True), ("[ AB AB ]", True), ("[A BAB]", False)]:
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        result = all(matcher.accept_token(token_ids[char]) for char in value)
        assert (result and matcher.is_terminated()) == accepted


@pytest.mark.parametrize(
    "attribute",
    [
        "lazy",
        'suffix="!"',
        'stop="!"',
        'suffix="!", stop_capture="marker"',
        'stop="!", stop_capture="marker"',
    ],
)
@pytest.mark.parametrize(
    "format_",
    [
        {"type": "const_string", "value": "X"},
        {"type": "dispatch", "rules": [["@", {"type": "const_string", "value": "X"}]]},
        {"type": "token", "token": 1},
    ],
)
def test_lark_structural_tag_rejects_terminal_like_attributes(
    attribute: str, format_: Dict[str, Any]
) -> None:
    payload = json.dumps(_structural_tag(format_))
    with pytest.raises(RuntimeError, match="cannot be used with lazy, suffix, or stop") as exc_info:
        xgr.Grammar.from_lark(f"start: embedded\nembedded[{attribute}]: %structural_tag {payload}")
    assert "line 2, column " in str(exc_info.value)
    assert "^" in str(exc_info.value)


def test_lark_structural_tag_rule_options_capture_max_chars_and_temperature() -> None:
    any_text = json.dumps(_structural_tag({"type": "any_text", "excludes": ["!"]}))
    tokenizer_info = xgr.TokenizerInfo(["é", "a", "b", "!"], stop_token_ids=[])
    grammar = xgr.Grammar.from_lark(
        f"""\
start: embedded "!"
embedded[capture="whole", max_chars=2, temperature=0.7]: %structural_tag {any_text}
""",
        tokenizer_info=tokenizer_info,
    )
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.temperature == pytest.approx(0.7)
    assert matcher.accept_token(0)
    forked = matcher.fork()
    assert matcher.accept_token(1)
    assert matcher.temperature is None
    assert matcher.accept_token(3) and matcher.is_terminated()
    assert matcher.get_captures() == [("whole", "éa".encode())]
    matcher.rollback(2)
    assert matcher.get_captures() == [("whole", "é".encode())]
    assert matcher.accept_token(2) and matcher.accept_token(3) and matcher.is_terminated()
    assert matcher.get_captures() == [("whole", "éb".encode())]
    assert forked.accept_token(3) and forked.is_terminated()
    assert forked.get_captures() == [("whole", "é".encode())]


def test_lark_structural_tag_max_tokens_is_per_occurrence() -> None:
    any_text = json.dumps(_structural_tag({"type": "any_text", "excludes": [",", "!"]}))
    tokenizer_info = xgr.TokenizerInfo(["a", "b", "c", ",", "!"], stop_token_ids=[])
    grammar = xgr.Grammar.from_lark(
        f"""\
start: embedded "," embedded "!"
embedded[max_tokens=1]: %structural_tag {any_text}
""",
        tokenizer_info=tokenizer_info,
    )
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(0)
    _, mask = _mask_snapshot(matcher, tokenizer_info)
    assert (int(mask[0, 0]) >> 3) & 1
    assert not ((int(mask[0, 0]) >> 2) & 1)
    assert matcher.accept_token(3)
    assert matcher.accept_token(1)
    assert matcher.accept_token(4) and matcher.is_terminated()


def test_lark_structural_tag_preserves_inner_and_outer_metadata_round_trip() -> None:
    payload = _structural_tag(
        {
            "type": "grammar",
            "grammar": 'root ::= inner\ninner[capture="inner", max_chars=1, temperature=0.2] ::= "x"',
        }
    )
    tokenizer_info = xgr.TokenizerInfo(["x"], stop_token_ids=[])
    grammar = xgr.Grammar.from_lark(
        f' start: outer\nouter[capture="outer", max_chars=2, temperature=0.8]: '
        f"%structural_tag {json.dumps(payload)}",
        tokenizer_info=tokenizer_info,
    )
    candidates = [
        grammar,
        xgr.Grammar.from_ebnf(str(grammar)),
        xgr.Grammar.deserialize_json(grammar.serialize_json()),
    ]
    for candidate in candidates:
        compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(
            candidate
        )
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert matcher.temperature == pytest.approx(0.2)
        assert matcher.accept_token(0) and matcher.is_terminated()
        assert matcher.get_captures() == [("inner", b"x"), ("outer", b"x")]


def test_lark_structural_tag_parametric_dead_branch_and_4096_cache_reuse() -> None:
    valid = json.dumps(_structural_tag({"type": "const_string", "value": "x"}))
    invalid_semantics = json.dumps({"type": "structural_tag"})
    dead_source = f"""\
start: state::0
state::_: %structural_tag {valid} %if bit_clear(0)
       | %structural_tag {invalid_semantics} %if bit_set(0)
"""
    grammar = xgr.Grammar.from_lark(dead_source)
    assert "const_string" in str(grammar)

    source_4096 = f"""\
start: counter::0
counter::_: %structural_tag {valid} counter::incr([0:12]) %if lt([0:12], 4095)
         | %structural_tag {valid} %if eq([0:12], 4095)
"""
    grammar_4096 = xgr.Grammar.from_lark(source_4096)
    assert str(grammar_4096).count("const_string ::=") == 1

    malformed = dead_source.replace(invalid_semantics, '{"type":"structural_tag"')
    with pytest.raises(RuntimeError, match="failed to parse JSON value after %structural_tag"):
        xgr.Grammar.from_lark(malformed)


def test_lark_structural_tag_parametric_capture_and_max_chars() -> None:
    payload = json.dumps(_structural_tag({"type": "const_string", "value": "x"}))
    source = f"""\
start: item::0 item::1 item::2
item::_[capture="part", max_chars=1]: %structural_tag {payload} %if lt(_, 3)
"""
    tokenizer_info = xgr.TokenizerInfo(["x"], stop_token_ids=[])
    grammar = xgr.Grammar.from_lark(source, tokenizer_info=tokenizer_info)
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert all(matcher.accept_token(0) for _ in range(3))
    assert matcher.is_terminated()
    assert matcher.get_captures() == [("part", b"x"), ("part", b"x"), ("part", b"x")]


def test_lark_structural_tag_byte_mode_does_not_leak_and_explicit_byte_mode_works() -> None:
    unicode_payload = _structural_tag({"type": "regex", "pattern": "."})
    byte_payload = _structural_tag(
        {"type": "grammar", "grammar": 'root ::= Regex(".", byte_mode=true)'}
    )
    tokenizer_info = xgr.TokenizerInfo([b"\x80", "a", "é"], stop_token_ids=[])
    for payload, expected in [
        (unicode_payload, [False, True, True]),
        (byte_payload, [True, True, False]),
    ]:
        source = (
            '%grammar_options {"allow_invalid_utf8": true}\nstart: %structural_tag '
            + json.dumps(payload)
        )
        grammar = xgr.Grammar.from_lark(source, tokenizer_info=tokenizer_info)
        compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
        for token_id, should_accept in enumerate(expected):
            matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
            assert (matcher.accept_token(token_id) and matcher.is_terminated()) == should_accept


def test_lark_structural_tag_byte_parametric_validation_rollback_and_serialization() -> None:
    byte_payload = _structural_tag(
        {"type": "grammar", "grammar": r'root ::= Regex("\\x80", byte_mode=true)'}
    )
    unicode_payload = _structural_tag({"type": "regex", "pattern": "."})
    source = f"""\
%grammar_options {{"allow_invalid_utf8": true}}
start[capture="all"]: state::0
state::_: %structural_tag {json.dumps(byte_payload)} state::set_bit(0) %if bit_clear(0)
       | %structural_tag {json.dumps(unicode_payload)} %if bit_set(0)
"""
    tokenizer_info = xgr.TokenizerInfo([b"\x80", "a", b"\xff"], stop_token_ids=[])
    grammar = xgr.Grammar.from_lark(source, tokenizer_info=tokenizer_info)
    for candidate in [
        grammar,
        xgr.Grammar.from_ebnf(str(grammar)),
        xgr.Grammar.deserialize_json(grammar.serialize_json()),
    ]:
        compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(
            candidate
        )
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        initial_state = matcher._debug_print_internal_state()
        assert matcher.validate_tokens([0, 1]) == 2
        assert matcher.validate_tokens([0, 2]) == 1
        assert matcher._debug_print_internal_state() == initial_state
        assert matcher.accept_token(0)
        accepted_state = matcher._debug_print_internal_state()
        assert matcher.validate_tokens([1]) == 1
        assert matcher._debug_print_internal_state() == accepted_state
        assert matcher.accept_token(1) and matcher.is_terminated()
        assert matcher.get_captures() == [("all", b"\x80a")]
        matcher.rollback(1)
        assert not matcher.is_terminated()
        assert matcher.get_captures() == []
        assert matcher.accept_token(1) and matcher.is_terminated()


def test_lark_structural_tag_dispatch_round_trips_and_validate_tokens() -> None:
    character_payload = _structural_tag(
        {
            "type": "triggered_tags",
            "triggers": ["<a"],
            "tags": [
                {
                    "type": "tag",
                    "begin": "<a>",
                    "content": {"type": "const_string", "value": "X"},
                    "end": "</a>",
                }
            ],
            "at_least_one": False,
            "stop_after_first": True,
        }
    )
    tokenizer_info, ids = _character_tokenizer(["pre<a>X</a>"])
    direct, _ = _compile_pair(character_payload, tokenizer_info)
    inline_grammar = xgr.Grammar.from_lark(
        _inline_source(character_payload), tokenizer_info=tokenizer_info
    )
    token_sequence = [ids[character] for character in "pre<a>X</a>"]
    for candidate in [
        inline_grammar,
        xgr.Grammar.from_ebnf(str(inline_grammar)),
        xgr.Grammar.deserialize_json(inline_grammar.serialize_json()),
    ]:
        compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(
            candidate
        )
        _assert_token_trace_equal(direct, compiled, tokenizer_info, token_sequence, True)

    token_payload = _structural_tag(
        {
            "type": "token_dispatch",
            "rules": [[10, {"type": "const_string", "value": "A"}]],
            "loop": False,
            "exclude_tokens": [50],
        }
    )
    token_info = _tokenizer_with_fixed_ids()
    token_direct, _ = _compile_pair(token_payload, token_info)
    token_inline = xgr.Grammar.from_lark(_inline_source(token_payload), tokenizer_info=token_info)
    for candidate in [
        token_inline,
        xgr.Grammar.from_ebnf(str(token_inline)),
        xgr.Grammar.deserialize_json(token_inline.serialize_json()),
    ]:
        compiled = xgr.GrammarCompiler(token_info, cache_enabled=False).compile_grammar(candidate)
        _assert_token_trace_equal(token_direct, compiled, token_info, [3, 10, 1], True)


def test_lark_structural_tag_nullable_recursive_and_repeated_nullable_terminate() -> None:
    nullable = _structural_tag(
        {"type": "grammar", "grammar": 'root ::= item\nitem ::= "" | "x" item'}
    )
    source = f'start: (%structural_tag {json.dumps(nullable)})* "z"'
    tokenizer_info, ids = _character_tokenizer(["z", "xxxz", "x"])
    grammar = xgr.Grammar.from_lark(source, tokenizer_info=tokenizer_info)
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    for value, accepted in [("z", True), ("xxxz", True), ("x", False)]:
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        result = all(matcher.accept_token(ids[character]) for character in value)
        assert (result and matcher.is_terminated()) == accepted


def test_lark_structural_tag_diagnostics_preserve_outer_location_and_context() -> None:
    invalid_values = ["null", "[]", '"text"', "{}", '{"type":"tag","begin":"<a>"}']
    for value in invalid_values:
        source = f'first: "ok"\nstart: %structural_tag {value}'
        with pytest.raises(RuntimeError, match="failed to compile inline structural tag") as exc:
            xgr.Grammar.from_lark(source)
        error = str(exc.value)
        assert "line 2, column 8" in error
        assert f"start: %structural_tag {value}" in error
        assert "       ^" in error

    with pytest.raises(RuntimeError, match="failed to parse JSON value after %structural_tag"):
        xgr.Grammar.from_lark('start: %structural_tag {"format":')
    with pytest.raises(RuntimeError, match="cannot be used in terminals"):
        xgr.Grammar.from_lark(
            "start: TOKEN\nTOKEN: %structural_tag "
            + json.dumps(_structural_tag({"type": "const_string", "value": "X"}))
        )
    with pytest.raises(RuntimeError, match="cannot be used in terminals"):
        xgr.Grammar.from_lark(
            "%ignore %structural_tag "
            + json.dumps(_structural_tag({"type": "const_string", "value": "X"}))
            + '\nstart: "X"'
        )
    with pytest.raises(RuntimeError, match="stop_capture requires stop or suffix"):
        xgr.Grammar.from_lark(
            'start: item\nitem[stop_capture="marker"]: %structural_tag '
            + json.dumps(_structural_tag({"type": "const_string", "value": "X"}))
        )
    with pytest.raises(RuntimeError, match="no start rule found"):
        xgr.Grammar.from_lark(
            "item: %structural_tag "
            + json.dumps(_structural_tag({"type": "const_string", "value": "X"}))
        )

    nested = """\
start: %lark {
  start: %structural_tag {"type":"structural_tag"}
}
"""
    with pytest.raises(RuntimeError, match="failed to compile nested Lark grammar") as exc:
        xgr.Grammar.from_lark(nested)
    assert "line 2, column 10" in str(exc.value)


def test_lark_structural_tag_string_token_requires_tokenizer_info() -> None:
    payload = _structural_tag({"type": "token", "token": "<missing>"})
    with pytest.raises(
        RuntimeError, match="Token string resolution requires tokenizer_info"
    ) as exc:
        xgr.Grammar.from_lark(_inline_source(payload))
    assert "line 1, column 8" in str(exc.value)


def test_lark_structural_tag_concurrent_compile_and_match() -> None:
    formats = [
        (
            {"type": "const_string", "value": f"value-{index}"}
            if index % 3 == 0
            else (
                {
                    "type": "tag",
                    "begin": f"<t{index}>",
                    "content": {"type": "regex", "pattern": "[a-z]+"},
                    "end": "</t>",
                }
                if index % 3 == 1
                else {
                    "type": "dispatch",
                    "rules": [[f"@{index}", {"type": "const_string", "value": "X"}]],
                    "loop": False,
                }
            )
        )
        for index in range(24)
    ]
    tokenizer_info = xgr.TokenizerInfo(
        list(dict.fromkeys(character for character in "value-0123456789<t>/abc@X")),
        stop_token_ids=[],
    )
    shared_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=16, cache_enabled=True)

    def compile_one(index: int) -> str:
        payload = _structural_tag(formats[index % len(formats)])
        grammar = xgr.Grammar.from_lark(_inline_source(payload), tokenizer_info=tokenizer_info)
        compiled = shared_compiler.compile_grammar(grammar)
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        _mask_snapshot(matcher, tokenizer_info)
        return str(grammar)

    with ThreadPoolExecutor(max_workers=16) as executor:
        first = list(executor.map(compile_one, range(192)))
    with ThreadPoolExecutor(max_workers=16) as executor:
        second = list(executor.map(compile_one, range(192)))
    assert first == second
