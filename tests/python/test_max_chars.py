from collections.abc import Sequence

import pytest

import xgrammar as xgr


def _compile(
    grammar: str, tokenizer_info: xgr.TokenizerInfo, *, cache_enabled: bool = False
) -> xgr.CompiledGrammar:
    grammar_object = xgr.Grammar.from_lark(grammar, tokenizer_info=tokenizer_info)
    return xgr.GrammarCompiler(tokenizer_info, cache_enabled=cache_enabled).compile_grammar(
        grammar_object
    )


def _allowed_token_ids(matcher: xgr.GrammarMatcher, tokenizer_info: xgr.TokenizerInfo) -> list[int]:
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(bitmask)
    return [
        token_id
        for token_id in range(tokenizer_info.vocab_size)
        if (int(bitmask[0, token_id // 32]) >> (token_id % 32)) & 1
    ]


def _accepts_tokens(compiled: xgr.CompiledGrammar, token_ids: Sequence[int]) -> bool:
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    return all(matcher.accept_token(token_id) for token_id in token_ids) and matcher.is_terminated()


def _accepts_tokens_with_masks(
    compiled: xgr.CompiledGrammar, tokenizer_info: xgr.TokenizerInfo, token_ids: Sequence[int]
) -> bool:
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for token_id in token_ids:
        if token_id not in _allowed_token_ids(matcher, tokenizer_info):
            return False
        assert matcher.accept_token(token_id)
    return matcher.is_terminated()


def test_max_chars_counts_codepoints_across_token_boundaries() -> None:
    tokenizer_info = xgr.TokenizerInfo(["a", "ab", "abc", "a>", "ab>", ">", "中", "中文", "中文>"])
    compiled = _compile(
        'start: reasoning ">"\nreasoning[max_chars=2]: TEXT\nTEXT: /(\\n|.)*/', tokenizer_info
    )

    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert _allowed_token_ids(matcher, tokenizer_info) == [0, 1, 3, 4, 5, 6, 7, 8]
    assert matcher.accept_token(4)
    assert matcher.is_terminated()

    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(8)
    assert matcher.is_terminated()

    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert not matcher.accept_token(2)
    assert matcher.accept_token(4)
    assert matcher.is_terminated()


def test_max_chars_boundary_with_accepted_bitset() -> None:
    accepted_tokens = [chr(0x1000 + index) for index in range(1024)]
    rejected_tokens = [chr(0x2000 + index) for index in range(1024)]
    vocab = [token for pair in zip(accepted_tokens, rejected_tokens) for token in pair] + [">"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    compiled = _compile(
        f'start: value ">"\nvalue[max_chars=1]: /[{accepted_tokens[0]}-{accepted_tokens[-1]}]*/',
        tokenizer_info,
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

    assert _allowed_token_ids(matcher, tokenizer_info) == list(range(0, 2048, 2)) + [2048]


def test_max_chars_accept_string_is_transactional() -> None:
    tokenizer_info = xgr.TokenizerInfo([])
    compiled = _compile(
        'start: reasoning ">"\nreasoning[max_chars=2]: TEXT\nTEXT: /(\\n|.)*/', tokenizer_info
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert not matcher.accept_string("中文a>")
    assert matcher.accept_string("中文>")
    assert matcher.is_terminated()


def test_max_chars_zero_closes_at_rule_entry() -> None:
    tokenizer_info = xgr.TokenizerInfo(["a", ">"])
    compiled = _compile(
        'start: value ">"\nvalue[max_chars=0]: TEXT\nTEXT: /(\\n|.)*/', tokenizer_info
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

    assert _allowed_token_ids(matcher, tokenizer_info) == [1]
    assert not matcher.accept_token(0)
    assert matcher.accept_token(1)
    assert matcher.is_terminated()


def test_max_chars_codepoint_split_across_tokens() -> None:
    tokenizer_info = xgr.TokenizerInfo([b"\xe4", b"\xb8", b"\xad", b">"])
    compiled = _compile(
        'start: reasoning ">"\nreasoning[max_chars=1]: TEXT\nTEXT: /(\\n|.)*/', tokenizer_info
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

    assert _allowed_token_ids(matcher, tokenizer_info) == [0, 3]
    assert matcher.accept_token(0)
    assert 1 in _allowed_token_ids(matcher, tokenizer_info)
    assert matcher.accept_token(1)
    assert 2 in _allowed_token_ids(matcher, tokenizer_info)
    assert matcher.accept_token(2)
    assert _allowed_token_ids(matcher, tokenizer_info) == [3]
    assert matcher.accept_token(3)
    assert matcher.is_terminated()


def test_max_chars_nested_budget_and_per_occurrence() -> None:
    tokenizer_info = xgr.TokenizerInfo(["x", "a", "b", "c", ":", ">"])
    nested = _compile(
        """
        start: outer ">"
        outer[max_chars=3]: "x" inner
        inner[max_chars=5]: TEXT
        TEXT: /(\\n|.)*/
        """,
        tokenizer_info,
    )
    matcher = xgr.GrammarMatcher(nested, terminate_without_stop_token=True)
    assert matcher.accept_string("xab>")
    assert matcher.is_terminated()
    matcher.reset()
    assert not matcher.accept_string("xabc>")

    repeated = _compile(
        """
        start: item ":" item ">"
        item[max_chars=2]: /[a-z]*/
        """,
        tokenizer_info,
    )
    matcher = xgr.GrammarMatcher(repeated, terminate_without_stop_token=True)
    assert matcher.accept_string("ab:ab>")
    assert matcher.is_terminated()


def test_max_chars_combines_with_max_tokens() -> None:
    tokenizer_info = xgr.TokenizerInfo(["a", "ab", ">"])
    compiled = _compile(
        """
        start: reasoning ">"
        reasoning[max_tokens=2, max_chars=3]: TEXT
        TEXT: /(\\n|.)*/
        """,
        tokenizer_info,
    )
    assert _accepts_tokens_with_masks(compiled, tokenizer_info, [1, 0, 2])
    assert not _accepts_tokens_with_masks(compiled, tokenizer_info, [0, 0, 0, 2])
    assert not _accepts_tokens_with_masks(compiled, tokenizer_info, [1, 1, 2])


def test_max_chars_combines_with_temperature() -> None:
    tokenizer_info = xgr.TokenizerInfo(["a", "ab", "abc", ">"])
    compiled = _compile(
        """
        start: value ">"
        value[max_chars=2, temperature=0.7]: /[a-z]*/
        """,
        tokenizer_info,
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

    assert matcher.temperature == pytest.approx(0.7)
    assert _allowed_token_ids(matcher, tokenizer_info) == [0, 1, 3]
    assert matcher.accept_token(1)
    assert matcher.temperature is None
    assert _allowed_token_ids(matcher, tokenizer_info) == [3]
    assert matcher.accept_token(3)
    assert matcher.is_terminated()


def test_max_chars_suffix_can_close_inside_token() -> None:
    tokenizer_info = xgr.TokenizerInfo(["aa!", "a>!", "aa>!"])
    compiled = _compile(
        """
        start: reasoning "!"
        reasoning[max_chars=2, suffix=">"]: TEXT
        TEXT: /(\\n|.)*/
        """,
        tokenizer_info,
    )
    assert _accepts_tokens(compiled, [0])
    assert _accepts_tokens(compiled, [1])
    assert not _accepts_tokens(compiled, [2])

    regex_tokenizer_info = xgr.TokenizerInfo(["a", "!"])
    regex_compiled = _compile(
        """
        start: reasoning "!"
        reasoning[max_chars=2, suffix=">"]: /[a-z]*/
        """,
        regex_tokenizer_info,
    )
    matcher = xgr.GrammarMatcher(regex_compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(0)
    assert matcher.accept_token(0)
    assert _allowed_token_ids(matcher, regex_tokenizer_info) == [1]
    assert matcher.accept_token(1)
    assert matcher.is_terminated()


def test_max_chars_stop_capture_distinguishes_forced_close() -> None:
    tokenizer_info = xgr.TokenizerInfo(["aa!", "a>!"])
    compiled = _compile(
        """
        start[capture="outer"]: reasoning "!"
        reasoning[max_chars=2, capture="inner", stop=">", stop_capture="marker"]: TEXT
        TEXT: /(\\n|.)*/
        """,
        tokenizer_info,
    )

    forced = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert forced.accept_token(0)
    assert forced.is_terminated()
    assert forced.get_captures() == [("inner", b"aa"), ("outer", b"aa!")]

    marker = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert marker.accept_token(1)
    assert marker.is_terminated()
    assert marker.get_captures() == [("marker", b">"), ("inner", b"a"), ("outer", b"a!")]

    stop_tokenizer_info = xgr.TokenizerInfo(["a", "<eos>"], stop_token_ids=[1])
    stop_compiled = _compile(
        """
        start: reasoning
        reasoning[max_chars=2, suffix=">"]: /[a-z]*/
        """,
        stop_tokenizer_info,
    )
    stop_matcher = xgr.GrammarMatcher(stop_compiled)
    assert stop_matcher.accept_token(0)
    assert stop_matcher.accept_token(0)
    assert _allowed_token_ids(stop_matcher, stop_tokenizer_info) == [1]
    assert stop_matcher.accept_token(1)
    assert stop_matcher.is_terminated()
    stop_matcher.rollback(1)
    assert not stop_matcher.is_terminated()
    assert _allowed_token_ids(stop_matcher, stop_tokenizer_info) == [1]


def test_max_chars_rollback_reset_and_fork() -> None:
    tokenizer_info = xgr.TokenizerInfo(["a", "b", ">"])
    compiled = _compile(
        'start: reasoning ">"\nreasoning[max_chars=2]: TEXT\nTEXT: /(\\n|.)*/', tokenizer_info
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(0)
    forked = matcher.fork()

    assert matcher.accept_token(1)
    assert _allowed_token_ids(matcher, tokenizer_info) == [2]
    matcher.rollback(1)
    assert 0 in _allowed_token_ids(matcher, tokenizer_info)

    assert forked.accept_token(1)
    assert _allowed_token_ids(forked, tokenizer_info) == [2]
    matcher.reset()
    assert 0 in _allowed_token_ids(matcher, tokenizer_info)


def test_max_chars_atomic_token_is_best_effort() -> None:
    tokenizer_info = xgr.TokenizerInfo(["<|wide|>", "!"])
    compiled = _compile('start: value "!"\nvalue[max_chars=1]: <|wide|>', tokenizer_info)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert _allowed_token_ids(matcher, tokenizer_info) == [0]
    assert matcher.accept_token(0)
    assert matcher.accept_token(1)
    assert matcher.is_terminated()


def test_max_chars_round_trip_and_validation() -> None:
    grammar = xgr.Grammar.from_lark("start: value\nvalue[max_chars=7, capture]: /[a-z]*/")
    printed = str(grammar)
    assert 'value[max_chars=7, capture="value"] ::=' in printed
    assert str(xgr.Grammar.from_ebnf(printed)) == printed
    assert str(xgr.Grammar.deserialize_json(grammar.serialize_json())) == printed

    for value in (0, 1_000_001, 2_147_483_647):
        assert f"max_chars={value}" in str(xgr.Grammar.from_lark(f"start[max_chars={value}]: /a*/"))
        assert f"max_chars={value}" in str(
            xgr.Grammar.from_ebnf(f"root[max_chars={value}] ::= [a]*")
        )


def test_max_chars_is_ignored_on_dynamic_dispatch_rules(capfd: pytest.CaptureFixture[str]) -> None:
    grammar = xgr.Grammar.from_lark(
        r"""
        start[max_chars=5]: tool* tail
        tail: TEXT
        tool_head[lazy]: TEXT "<tool>"
        tool[max_chars=3]: tool_head /[0-9]+/ "</tool>"
        TEXT: /(\n|.)*/
        """
    )
    captured = capfd.readouterr()
    warnings = captured.out + captured.err

    assert "Ignoring max_chars on dynamic dispatch start rule 'start'." in warnings
    assert (
        "Ignoring max_chars on rule 'tool' because it is consumed by dynamic dispatch." in warnings
    )
    assert "max_chars=" not in str(grammar)


def test_max_chars_compiler_cache_and_serialization() -> None:
    tokenizer_info = xgr.TokenizerInfo(["ab>", "abc"])
    grammar = xgr.Grammar.from_ebnf('root ::= value ">"\nvalue[max_chars=2] ::= [^]*')
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=True).compile_grammar(grammar)
    recovered_tokenizer_info = xgr.TokenizerInfo.deserialize_json(tokenizer_info.serialize_json())
    recovered = xgr.CompiledGrammar.deserialize_json(
        compiled.serialize_json(), recovered_tokenizer_info
    )

    for candidate in (compiled, recovered):
        matcher = xgr.GrammarMatcher(candidate, terminate_without_stop_token=True)
        assert _allowed_token_ids(matcher, tokenizer_info) == [0]
        assert matcher.accept_token(0)
        assert matcher.is_terminated()
