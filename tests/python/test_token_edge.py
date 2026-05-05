"""Tests for Token() edge support in grammar parsing, matching, and bitmask generation."""

import sys

import pytest

import xgrammar as xgr
from xgrammar.testing import (
    _ebnf_to_grammar_no_normalization,
    _get_masked_tokens_from_bitmask,
    _get_matcher_from_grammar_and_tokenizer_info,
)

# --- Parser / Printer roundtrip tests ---


def test_parse_token_basic():
    before = "root ::= Token(1, 2, 3)\n"
    expected = "root ::= ((Token(1, 2, 3)))\n"
    grammar = _ebnf_to_grammar_no_normalization(before)
    assert str(grammar) == expected


def test_parse_token_single():
    before = "root ::= Token(42)\n"
    expected = "root ::= ((Token(42)))\n"
    grammar = _ebnf_to_grammar_no_normalization(before)
    assert str(grammar) == expected


def test_parse_token_sorted_deduped():
    before = "root ::= Token(3, 1, 2, 1, 3)\n"
    expected = "root ::= ((Token(1, 2, 3)))\n"
    grammar = _ebnf_to_grammar_no_normalization(before)
    assert str(grammar) == expected


def test_parse_token_in_sequence():
    before = 'root ::= Token(1, 2) "hello"\n'
    expected = 'root ::= ((Token(1, 2) "hello"))\n'
    grammar = _ebnf_to_grammar_no_normalization(before)
    assert str(grammar) == expected


def test_parse_token_in_alternation():
    before = 'root ::= Token(1) | "hello"\n'
    expected = 'root ::= ((Token(1)) | ("hello"))\n'
    grammar = _ebnf_to_grammar_no_normalization(before)
    assert str(grammar) == expected


def test_parse_exclude_token_basic():
    before = "root ::= ExcludeToken(1, 2, 3)\n"
    expected = "root ::= ((ExcludeToken(1, 2, 3)))\n"
    grammar = _ebnf_to_grammar_no_normalization(before)
    assert str(grammar) == expected


def test_parse_exclude_token_sorted_deduped():
    before = "root ::= ExcludeToken(3, 1, 2, 1)\n"
    expected = "root ::= ((ExcludeToken(1, 2, 3)))\n"
    grammar = _ebnf_to_grammar_no_normalization(before)
    assert str(grammar) == expected


# --- Matcher accept_token tests ---


STOP_TOKEN_ID = 1  # "</s>" in our test vocab


def _make_matcher(vocab, grammar_str):
    """Create a matcher with a custom vocab and grammar."""
    tokenizer_info = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    return _get_matcher_from_grammar_and_tokenizer_info(grammar, tokenizer_info)


def test_accept_token_basic():
    """Token(2, 4) should accept token IDs 2 and 4 but reject others."""
    vocab = ["<s>", "</s>", "aa", "bb", "cc", "dd"]
    #         0      1       2     3     4     5
    matcher = _make_matcher(vocab, "root ::= Token(2, 4)\n")

    assert matcher.accept_token(2)
    assert matcher.accept_token(STOP_TOKEN_ID)
    assert matcher.is_terminated()


def test_accept_token_reject():
    """Tokens not in the Token() set should be rejected."""
    vocab = ["<s>", "</s>", "aa", "bb", "cc", "dd"]
    matcher = _make_matcher(vocab, "root ::= Token(2, 4)\n")

    assert not matcher.accept_token(3)
    assert not matcher.accept_token(5)
    assert matcher.accept_token(4)
    assert matcher.accept_token(STOP_TOKEN_ID)
    assert matcher.is_terminated()


def test_token_then_string():
    """Token followed by string literal: Token(2) "bb" ."""
    vocab = ["<s>", "</s>", "aa", "bb", "cc"]
    matcher = _make_matcher(vocab, 'root ::= Token(2) "bb"\n')

    assert matcher.accept_token(2)  # Token(2) = "aa"
    assert matcher.accept_token(3)  # "bb"
    assert matcher.accept_token(STOP_TOKEN_ID)
    assert matcher.is_terminated()


def test_token_or_string():
    """Alternation: Token(2) | "bb" ."""
    vocab = ["<s>", "</s>", "aa", "bb", "cc"]

    # Accept via token path
    matcher = _make_matcher(vocab, 'root ::= Token(2) | "bb"\n')
    assert matcher.accept_token(2)
    assert matcher.accept_token(STOP_TOKEN_ID)
    assert matcher.is_terminated()

    # Accept via string path
    matcher2 = _make_matcher(vocab, 'root ::= Token(2) | "bb"\n')
    assert matcher2.accept_token(3)  # "bb"
    assert matcher2.accept_token(STOP_TOKEN_ID)
    assert matcher2.is_terminated()


# --- Bitmask tests ---


def test_bitmask_token_only():
    """FillNextTokenBitmask should allow only tokens in Token() set (and stop token)."""
    vocab = ["<s>", "</s>", "aa", "bb", "cc", "dd"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf("root ::= Token(2, 4)\n")
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, tokenizer_info)

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected = set(_get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size))

    assert rejected == {0, 1, 3, 5}


def test_bitmask_token_and_string():
    """Bitmask for Token(2) | "bb" should allow token 2 and token whose text is "bb"."""
    vocab = ["<s>", "</s>", "aa", "bb", "cc"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf('root ::= Token(2) | "bb"\n')
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, tokenizer_info)

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected = set(_get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size))

    assert rejected == {0, 1, 4}


def test_bitmask_after_token():
    """After accepting a Token, the bitmask should reflect the next expected tokens."""
    vocab = ["<s>", "</s>", "aa", "bb", "cc"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf('root ::= Token(2) "bb"\n')
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, tokenizer_info)

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected = set(_get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size))
    assert rejected == {0, 1, 3, 4}

    assert matcher.accept_token(2)

    matcher.fill_next_token_bitmask(token_bitmask)
    rejected2 = set(_get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size))
    assert rejected2 == {0, 1, 2, 4}


def test_token_multiple_choices():
    """Token set with multiple IDs in alternation with other rules."""
    vocab = ["<s>", "</s>", "x", "y", "z", "w"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf('root ::= Token(2, 3, 4) | "w"\n')
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, tokenizer_info)

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected = set(_get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size))

    assert rejected == {0, 1}


# --- 4.3 char-then-token sequence tests ---


def test_char_then_token_sequence():
    """String literal followed by Token: "A" Token(4, 5)."""
    vocab = ["<s>", "</s>", "A", "B", "hello", "world"]
    matcher = _make_matcher(vocab, 'root ::= "A" Token(4, 5)\n')
    assert matcher.accept_token(2)  # "A"
    assert matcher.accept_token(4)  # Token(4) = "hello"
    assert matcher.accept_token(STOP_TOKEN_ID)
    assert matcher.is_terminated()


# --- TokenTagDispatch + excludes tests ---


def _make_bitmask_helper(vocab, grammar_str):
    """Create matcher, tokenizer_info, and bitmask for a grammar."""
    tokenizer_info = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, tokenizer_info)
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    return matcher, tokenizer_info, token_bitmask


def _get_accepted(matcher, token_bitmask, vocab_size):
    """Fill bitmask and return accepted token IDs."""
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected = _get_masked_tokens_from_bitmask(token_bitmask, vocab_size)
    return set(range(vocab_size)) - set(rejected)


def test_token_tag_dispatch_exclude_no_triggers():
    """ExcludeToken self-loop accepts all tokens except excluded ones."""
    vocab = ["<s>", "</s>", "hello", "world", "blocked_1", "blocked_2"]
    grammar_str = """root ::= TokenTagDispatch(
      excludes=(4, 5)
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)

    for _ in range(3):
        assert _get_accepted(matcher, bitmask, ti.vocab_size) == {0, 1, 2, 3}
        matcher.accept_token(2)


def test_token_tag_dispatch_exclude_basic():
    """ExcludeToken edge blocks excluded tokens."""
    vocab = ["<s>", "</s>", "hello", "world", "bad"]
    grammar_str = """root ::= TokenTagDispatch(
      excludes=(4,)
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)
    assert _get_accepted(matcher, bitmask, ti.vocab_size) == {0, 1, 2, 3}


def test_token_tag_dispatch_reject_enforced_by_parser():
    """accept_token must reject tokens excluded by kExcludeToken edge."""
    vocab = ["<s>", "</s>", "hello", "world", "blocked"]
    grammar_str = """root ::= TokenTagDispatch(
      excludes=(4,)
    )"""
    ti = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, ti)
    assert not matcher.accept_token(4), "parser must reject excluded token"
    assert matcher.accept_token(2)  # "hello" still accepted


def test_token_tag_dispatch_trigger_and_exclude():
    """TokenTagDispatch with trigger and exclude."""
    vocab = ["<s>", "</s>", "A", "AB", "blocked"]
    grammar_str = """
    rule1 ::= "done"
    root ::= TokenTagDispatch(
      (3, rule1),
      excludes=(4,)
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)
    assert _get_accepted(matcher, bitmask, ti.vocab_size) == {0, 1, 2, 3}


# --- TokenTagDispatch + trigger tests ---


def test_token_tag_dispatch_trigger():
    """Token trigger dispatches to a rule."""
    vocab = ["<s>", "</s>", "hello", "trigger_tok", "content"]
    grammar_str = """
    triggered_rule ::= Token(4)
    root ::= TokenTagDispatch(
      (3, triggered_rule)
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)
    assert _get_accepted(matcher, bitmask, ti.vocab_size) == {0, 1, 2, 3, 4}

    assert matcher.accept_token(3)  # dispatch trigger
    assert _get_accepted(matcher, bitmask, ti.vocab_size) == {4}


def test_token_tag_dispatch_multiple_triggers():
    """Multiple token triggers in TokenTagDispatch."""
    vocab = ["<s>", "</s>", "A", "B", "<tool>", "content"]
    grammar_str = """
    tool_body ::= Token(5)
    other_body ::= Token(5)
    root ::= TokenTagDispatch(
      (3, tool_body),
      (4, other_body)
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)
    assert _get_accepted(matcher, bitmask, ti.vocab_size) == {0, 1, 2, 3, 4, 5}

    assert matcher.accept_token(3)  # dispatch to tool_body
    assert _get_accepted(matcher, bitmask, ti.vocab_size) == {5}


def test_token_tag_dispatch_trigger_loop():
    """Token trigger with loop_after_dispatch returns to start after body completes."""
    vocab = ["<s>", "</s>", "hello", "trigger", "content"]
    grammar_str = """
    body ::= Token(4)
    root ::= TokenTagDispatch(
      (3, body),
      loop_after_dispatch=true
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)

    assert matcher.accept_token(3)  # trigger dispatches to body
    assert matcher.accept_token(4)  # Token(4) completes body
    assert _get_accepted(matcher, bitmask, ti.vocab_size) == {0, 1, 2, 3, 4}


def test_token_tag_dispatch_trigger_and_exclude_no_overlap():
    """Token trigger IDs and excludes must not overlap."""
    grammar_str = """
    body ::= Token(2)
    root ::= TokenTagDispatch(
      (3, body),
      excludes=(3,)
    )"""
    with pytest.raises(Exception):
        xgr.Grammar.from_ebnf(grammar_str)


def test_token_tag_dispatch_trigger_in_bitmask():
    """Trigger tokens accepted via kToken edge, others via ExcludeToken self-loop."""
    vocab = ["<s>", "</s>", "hello", "trigger", "content"]
    grammar_str = """
    body ::= Token(4)
    root ::= TokenTagDispatch(
      (3, body)
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)
    assert _get_accepted(matcher, bitmask, ti.vocab_size) == {0, 1, 2, 3, 4}

    assert matcher.accept_token(3)  # dispatch trigger
    assert _get_accepted(matcher, bitmask, ti.vocab_size) == {4}


def test_token_tag_dispatch_full_combo():
    """Token triggers + excludes all working together."""
    vocab = ["<s>", "</s>", "hello", "B", "<tool>", "content", "blocked"]
    grammar_str = """
    tool_body ::= Token(5)
    other_body ::= Token(5)
    root ::= TokenTagDispatch(
      (3, tool_body),
      (4, other_body),
      excludes=(6,)
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)
    assert _get_accepted(matcher, bitmask, ti.vocab_size) == {0, 1, 2, 3, 4, 5}


# --- Lookahead Assertion + kToken tests ---


def test_lookahead_exact_with_token_set():
    """Exact lookahead containing kToken: tokens matching the rule are accepted."""
    vocab = ["<s>", "</s>", "abc", "abcd", "X"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(
        """
    rule_a ::= [a-z]+
    root ::= rule_a Token(4)
    """
    )
    matcher = xgr.GrammarMatcher(compiled)
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected = set(_get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size))
    assert rejected == {0, 1, 4}


def test_lookahead_token_set_suffix_nonempty_rejected():
    """Token that partially matches rule but has bytes left at kToken boundary → rejected."""
    vocab = ["<s>", "</s>", "ab", "a", "X"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(
        """
    rule_a ::= "a"
    root ::= rule_a Token(4)
    """
    )
    matcher = xgr.GrammarMatcher(compiled)
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected = set(_get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size))
    assert rejected == {0, 1, 2, 4}


def test_lookahead_mixed_char_and_token():
    """Lookahead with char elements before kToken."""
    vocab = ["<s>", "</s>", "abc", "abc!", "X"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(
        """
    rule_a ::= [a-z]+
    root ::= rule_a "!" Token(4)
    """
    )
    matcher = xgr.GrammarMatcher(compiled)
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected = set(_get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size))
    assert rejected == {0, 1, 4}


# --- End-to-end tests ---


def test_e2e_complex():
    """TokenTagDispatch with two trigger paths: tool (char grammar) and code (Token grammar)."""
    # fmt: off
    vocab = [
        "<s>",       # 0
        "</s>",      # 1
        "<tool>",    # 2   (trigger -> tool_body)
        "<code>",    # 3   (trigger -> code_body)
        "<blocked>", # 4   (excluded)
        "hello",     # 5
        "he",        # 6   (prefix of "hello")
        "name",      # 7
        "val",       # 8
        "x",         # 9
        "y",         # 10
        "{",         # 11
        "}",         # 12
        ":",         # 13
        ",",         # 14
        "[",         # 15
        "]",         # 16
        ";",         # 17
        "42",        # 18
        "a:",        # 19  (crosses [a-z]+ / ":" boundary)
        "{a",        # 20  (crosses "{" / [a-z]+ boundary)
        "a}",        # 21  (crosses [a-z]+ / "}" boundary)
        "a;",        # 22  (crosses [a-z]+ / ";" boundary)
        "fn(",       # 23  (matches "fn(" exactly)
        ")",         # 24
    ]
    # fmt: on
    grammar_str = """
value ::= [a-z]+ | [0-9]+
entry ::= [a-z]+ ":" value
inner ::= entry (";" entry)*
body ::= "{" inner "}" | "[" inner "]"
tool_body ::= body ("," body)*
arg ::= [a-z]+
call ::= "fn(" Token(9, 10) "," arg ")"
code_body ::= call (";" call)*
root ::= TokenTagDispatch(
    (2, tool_body),
    (3, code_body),
    excludes=(4,)
)
"""
    ti = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    A = set(range(len(vocab))) - {4}
    AZ = {5, 6, 7, 8, 9, 10}

    # fmt: off
    paths = [
        [   # tool path: trigger -> nested char grammar with uncertainty tokens
            (None, A),                      # initial
            (2,    {11, 15, 20}),            # <tool> -> body: need "{" or "["
            (20,   AZ | {13, 19}),           # {a -> entry key continues
            (19,   AZ | {18, 21, 22}),       # a: -> cross-boundary: key done + ":"
            (18,   {12, 17, 18}),            # 42 -> [0-9]+ value; then "}" or ";"
            (17,   AZ | {19}),              # ; -> second entry
            (7,    AZ | {13, 19}),           # name -> entry key
            (13,   AZ | {18, 21, 22}),       # : -> value
            (8,    AZ | {12, 17, 21, 22}),   # val -> [a-z]+ value; then "}" or ";"
            (12,   A),                       # } -> tool_body complete, back to self-loop
        ],
        [   # code path: trigger -> text-nested-Token grammar fn(Token,arg)
            (None, A),            # initial
            (3,    {23}),          # <code> -> call: need "fn("
            (23,   {9, 10}),       # fn( -> Token(9, 10) position
            (9,    {14}),          # x (Token 9) -> need ","
            (14,   AZ),           # , -> arg: [a-z]+
            (7,    AZ | {24}),    # name -> arg continues or ")"
            (24,   A),            # ) -> call complete, back to self-loop
        ],
    ]
    # fmt: on

    for steps in paths:
        matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, ti)
        bitmask = xgr.allocate_token_bitmask(1, ti.vocab_size)
        for token_id, expected in steps:
            if token_id is not None:
                assert matcher.accept_token(token_id)
            assert _get_accepted(matcher, bitmask, ti.vocab_size) == expected
        assert matcher.accept_token(1)  # </s>
        assert matcher.is_terminated()


def test_e2e_nested_dispatch():
    """Nested TokenTagDispatch: outer excludes 4, inner excludes 5.

    Token 4 (o_block): rejected at outer, but accepted inside inner dispatch.
    Token 5 (i_block): rejected at inner, but accepted at outer dispatch.
    """
    vocab = [
        "<s>",  # 0
        "</s>",  # 1
        "<outer>",  # 2
        "<inner>",  # 3
        "<o_block>",  # 4
        "<i_block>",  # 5
        "hello",  # 6
        "world",  # 7
        "fn(",  # 8
        ")",  # 9
        "x",  # 10
        "y",  # 11
    ]
    grammar_str = """
    leaf ::= Token(10, 11)
    inner ::= TokenTagDispatch((3, leaf), excludes=(5,))
    tool_fn ::= "fn(" inner ")"
    root ::= TokenTagDispatch((2, tool_fn), excludes=(4,))
    """
    ALL = set(range(len(vocab)))
    OUTER = ALL - {4}
    INNER = ALL - {1, 5}

    ti = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf(grammar_str)

    def fresh():
        m = _get_matcher_from_grammar_and_tokenizer_info(grammar, ti)
        b = xgr.allocate_token_bitmask(1, ti.vocab_size)
        return m, b

    # fmt: off
    paths = [
        [   # Path A: outer trigger -> fn( -> inner -> leaf -> )
            (None, OUTER), (2, {8}), (8, INNER), (6, INNER),
            (3, {10, 11}), (10, INNER), (9, ALL), (1, None),
        ],
        [   # Path B: outer loop only, <o_block>(4) rejected
            (None, OUTER), (6, OUTER), (5, OUTER), (4, False), (1, None),
        ],
        [   # Path C1: <o_block>(4) rejected at outer
            (None, OUTER), (4, False),
        ],
        [   # Path C2: <o_block>(4) accepted inside inner
            (2, {8}), (8, INNER), (4, INNER), (9, ALL), (1, None),
        ],
    ]
    # fmt: on

    for steps in paths:
        m, b = fresh()
        for token_id, expected in steps:
            if token_id is None:
                assert _get_accepted(m, b, ti.vocab_size) == expected
            elif expected is False:
                assert not m.accept_token(token_id)
            elif expected is None:
                assert m.accept_token(token_id)
                assert m.is_terminated()
            else:
                assert m.accept_token(token_id)
                assert _get_accepted(m, b, ti.vocab_size) == expected


def test_e2e_nested_exclude_loop():
    """Nested ExcludeToken-only loop: [a-z]+ loop Token(5) [a-z]+.

    Token 4 ("###") is excluded from loop AND not [a-z]+ -> always rejected.
    Token 5 ("<END>") is excluded from loop but accepted via Token(5) after loop.
    Token 0 ("<s>") is consumed by loop (non-[a-z], non-excluded).
    """
    vocab = ["<s>", "</s>", "hello", "world", "###", "<END>", "foo", "done"]
    #         0      1       2        3        4      5        6      7
    grammar_str = """
    loop ::= TokenTagDispatch(excludes=(4, 5))
    root ::= [a-z]+ loop Token(5) [a-z]+
    """
    ti = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    AZ = {2, 3, 6, 7}
    LOOP = {0, 2, 3, 5, 6, 7}

    # fmt: off
    paths = [
        [   # Main path: [a-z]+ -> loop -> Token(5) -> [a-z]+ -> end
            (None, AZ), (4, False), (2, LOOP), (0, LOOP), (3, LOOP),
            (5, AZ), (7, {1} | AZ), (1, None),
        ],
    ]
    # fmt: on

    for steps in paths:
        matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, ti)
        bitmask = xgr.allocate_token_bitmask(1, ti.vocab_size)
        for token_id, expected in steps:
            if token_id is None:
                assert _get_accepted(matcher, bitmask, ti.vocab_size) == expected
            elif expected is False:
                assert not matcher.accept_token(token_id)
            elif expected is None:
                assert matcher.accept_token(token_id)
                assert matcher.is_terminated()
            else:
                assert matcher.accept_token(token_id)
                assert _get_accepted(matcher, bitmask, ti.vocab_size) == expected


def test_e2e_mixed_tag_and_token_dispatch():
    """Three-layer nesting: TagDispatch -> TokenTagDispatch -> TagDispatch.

    Outer (TagDispatch): trigger "<call>", excludes string "<bad>"
    Mid   (TokenTagDispatch): trigger token 3, excludes token 4
    Inner (TagDispatch): trigger "<end>", excludes string "<bad>"

    Key: token 6 ("<bad>") rejected by string-based excludes (outer+inner),
    but temporarily accepted inside mid (token-based, doesn't exclude 6).
    """
    vocab = [
        "<s>",  # 0
        "</s>",  # 1
        "<call>",  # 2
        "<mid>",  # 3
        "<skip>",  # 4
        "<end>",  # 5
        "<bad>",  # 6
        "hello",  # 7
        "world",  # 8
        "x",  # 9
        "y",  # 10
        "done",  # 11
    ]
    grammar_str = """
    leaf ::= [a-z]+
    inner ::= TagDispatch(("<end>", leaf), excludes=("<bad>"))
    mid_body ::= Token(9, 10) inner
    mid ::= TokenTagDispatch((3, mid_body), excludes=(4,))
    root ::= TagDispatch(("<call>", mid), excludes=("<bad>"))
    """
    ALL = set(range(len(vocab)))
    OUTER = ALL - {6}

    ti = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf(grammar_str)

    def fresh():
        m = _get_matcher_from_grammar_and_tokenizer_info(grammar, ti)
        b = xgr.allocate_token_bitmask(1, ti.vocab_size)
        return m, b

    # expected=False means reject; expected=None means accept+terminated
    # fmt: off
    paths = [
        [   # Path A: full traversal through all 3 layers
            (None, OUTER), (7, OUTER), (2, ALL), (3, OUTER), (9, OUTER),
            (8, OUTER), (5, OUTER), (11, OUTER), (1, None),
        ],
        [   # Path B: outer loop only, <bad>(6) rejected
            (None, OUTER), (7, OUTER), (8, OUTER), (6, False), (1, None),
        ],
        [   # Path C1: <bad>(6) rejected at outer
            (None, OUTER), (6, False),
        ],
        [   # Path C2: <bad>(6) accepted inside mid
            (2, ALL), (6, ALL),
        ],
        [   # Path C3: <bad>(6) rejected by inner excludes
            (2, ALL), (3, OUTER), (9, OUTER), (6, False),
        ],
        [   # Path D1: <skip>(4) accepted at outer
            (4, OUTER),
        ],
        [   # Path D2: <skip>(4) accepted at inner
            (2, ALL), (3, OUTER), (9, OUTER), (4, OUTER),
        ],
    ]
    # fmt: on

    for steps in paths:
        m, b = fresh()
        for token_id, expected in steps:
            if token_id is None:
                assert _get_accepted(m, b, ti.vocab_size) == expected
            elif expected is False:
                assert not m.accept_token(token_id)
            elif expected is None:
                assert m.accept_token(token_id)
                assert m.is_terminated()
            else:
                assert m.accept_token(token_id)
                assert _get_accepted(m, b, ti.vocab_size) == expected


def test_rollback():
    """Rollback restores mask and accept_token behavior for token edges."""
    vocab = ["<s>", "</s>", "<tool>", "<code>", "hello", "world", "fn(", ")", "x", "y"]
    #          0      1       2         3        4        5       6      7    8    9
    grammar_str = """
    arg ::= [a-z]+
    call ::= "fn(" Token(8, 9) "," arg ")"
    root ::= TokenTagDispatch(
      (2, call),
      excludes=(3,)
    )
    """
    ti = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    m = _get_matcher_from_grammar_and_tokenizer_info(grammar, ti)
    b = xgr.allocate_token_bitmask(1, ti.vocab_size)

    mask_0 = _get_accepted(m, b, ti.vocab_size)
    assert m.accept_token(2)  # <tool> trigger
    mask_1 = _get_accepted(m, b, ti.vocab_size)
    assert m.accept_token(6)  # fn(
    mask_2 = _get_accepted(m, b, ti.vocab_size)
    assert m.accept_token(8)  # x (Token edge)
    mask_3 = _get_accepted(m, b, ti.vocab_size)

    # Rollback all 3 tokens
    m.rollback(3)
    assert _get_accepted(m, b, ti.vocab_size) == mask_0

    # Re-accept and verify masks match
    assert m.accept_token(2)
    assert _get_accepted(m, b, ti.vocab_size) == mask_1
    assert m.accept_token(6)
    assert _get_accepted(m, b, ti.vocab_size) == mask_2
    assert m.accept_token(8)
    assert _get_accepted(m, b, ti.vocab_size) == mask_3

    # Rollback 2, then continue on a different path
    m.rollback(2)
    assert _get_accepted(m, b, ti.vocab_size) == mask_1
    assert m.accept_token(6)
    assert m.accept_token(9)  # y instead of x
    assert _get_accepted(m, b, ti.vocab_size) == mask_3  # same: need ","

    # Rollback 1 past the token edge, re-accept
    m.rollback(1)
    assert _get_accepted(m, b, ti.vocab_size) == mask_2
    assert m.accept_token(8)
    assert _get_accepted(m, b, ti.vocab_size) == mask_3


# --- Structural tag acceptance tests ---

STAG_VOCAB = [
    "<s>",  # 0
    "</s>",  # 1
    "<tool>",  # 2
    "<code>",  # 3
    "<end>",  # 4
    "<think>",  # 5
    "<think_end>",  # 6
    "hello",  # 7
    "world",  # 8
    "{",  # 9
    "}",  # 10
    "fn(",  # 11
    ")",  # 12
    "x",  # 13
    "y",  # 14
    ",",  # 15
    "<bad>",  # 16
]
STAG_STOP = 1


def _stag_matcher(stag_json):
    ti = xgr.TokenizerInfo(STAG_VOCAB)
    compiler = xgr.GrammarCompiler(ti)
    compiled = compiler.compile_structural_tag(stag_json)
    m = xgr.GrammarMatcher(compiled)
    b = xgr.allocate_token_bitmask(1, ti.vocab_size)
    return m, b, ti


def _accept_tokens(m, tokens):
    for t in tokens:
        ok = m.accept_token(t)
        assert ok, f"Failed to accept token {t} (vocab: {STAG_VOCAB[t]!r})"


def _accept_and_stop(m, tokens):
    _accept_tokens(m, tokens)
    assert m.accept_token(STAG_STOP), "Failed to accept stop token"
    assert m.is_terminated()


def test_stag_token_begin_end():
    """Tag with token begin/end wrapping JSON content."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": {"type": "token", "token": "<tool>"},
            "content": {"type": "const_string", "value": "hello"},
            "end": {"type": "token", "token": "<end>"},
        },
    }
    m, b, ti = _stag_matcher(stag)
    _accept_and_stop(m, [2, 7, 4])  # <tool> hello <end>


def test_stag_exclude_token_basic():
    """ExcludeTokenFormat rejects specified tokens and auto-detected end."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": {"type": "token", "token": "<tool>"},
            "content": {"type": "exclude_token", "exclude_tokens": [16]},  # exclude <bad>
            "end": {"type": "token", "token": "<end>"},
        },
    }
    m, b, ti = _stag_matcher(stag)
    _accept_tokens(m, [2])  # <tool>
    assert not m.accept_token(16)  # <bad> excluded
    assert not m.accept_token(4)  # <end> auto-excluded
    assert m.accept_token(7)  # hello accepted
    _accept_and_stop(m, [4])  # <end>


def test_stag_any_tokens_loop():
    """AnyTokensFormat accepts arbitrary tokens until end, excluding specified."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": {"type": "token", "token": "<think>"},
            "content": {"type": "any_tokens", "exclude_tokens": [16]},  # exclude <bad>
            "end": {"type": "token", "token": "<think_end>"},
        },
    }
    m, b, ti = _stag_matcher(stag)
    _accept_tokens(m, [5])  # <think>
    _accept_tokens(m, [7, 8, 13, 14, 9, 10])  # hello world x y { }
    assert not m.accept_token(16)  # <bad> excluded
    # token 6 (<think_end>) is accepted: ends any_tokens (zero more) then matches end
    _accept_tokens(m, [0, 3, 2])  # <s> <code> <tool> all ok
    _accept_and_stop(m, [6])  # <think_end>


def test_stag_any_tokens_empty():
    """AnyTokensFormat can match zero tokens (go straight to end)."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": {"type": "token", "token": "<tool>"},
            "content": {"type": "any_tokens"},
            "end": {"type": "token", "token": "<end>"},
        },
    }
    m, b, ti = _stag_matcher(stag)
    _accept_and_stop(m, [2, 4])  # <tool> <end> (zero content tokens)


def test_stag_token_triggered_tags_basic():
    """TokenTriggeredTagsFormat dispatches to different tags based on trigger tokens."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<tool>", "<code>"],
            "tags": [
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<tool>"},
                    "content": {"type": "const_string", "value": "hello"},
                    "end": {"type": "token", "token": "<end>"},
                },
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<code>"},
                    "content": {"type": "const_string", "value": "world"},
                    "end": {"type": "token", "token": "<end>"},
                },
            ],
            "exclude_tokens": [16],
        },
    }
    m, b, ti = _stag_matcher(stag)
    assert not m.accept_token(16)  # <bad> excluded
    _accept_tokens(m, [7, 8])  # hello world (free tokens)
    _accept_tokens(m, [2, 7, 4])  # <tool> hello <end>
    _accept_tokens(m, [13, 14])  # x y (free tokens)
    _accept_tokens(m, [3, 8, 4])  # <code> world <end>
    assert m.accept_token(STAG_STOP)
    assert m.is_terminated()


def test_stag_token_triggered_stop_after_first():
    """TokenTriggeredTagsFormat with stop_after_first stops after one tag."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<tool>", "<code>"],
            "tags": [
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<tool>"},
                    "content": {"type": "const_string", "value": "x"},
                    "end": {"type": "token", "token": "<end>"},
                },
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<code>"},
                    "content": {"type": "const_string", "value": "y"},
                    "end": {"type": "token", "token": "<end>"},
                },
            ],
            "stop_after_first": True,
        },
    }
    m, b, ti = _stag_matcher(stag)
    _accept_tokens(m, [7])  # hello (free)
    _accept_and_stop(m, [2, 13, 4])  # <tool> x <end>


def test_stag_token_triggered_at_least_one():
    """TokenTriggeredTagsFormat with at_least_one+stop_after_first = exactly one tag."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<tool>"],
            "tags": [
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<tool>"},
                    "content": {"type": "const_string", "value": "x"},
                    "end": {"type": "token", "token": "<end>"},
                }
            ],
            "at_least_one": True,
            "stop_after_first": True,
        },
    }
    m, b, ti = _stag_matcher(stag)
    # Must start with trigger — free tokens not accepted
    _accept_and_stop(m, [2, 13, 4])  # <tool> x <end>


def test_stag_nested_token_tags_with_any_tokens():
    """Nested: token_triggered_tags containing tags with any_tokens content.

    Outer: token_triggered_tags with triggers <tool> and <code>.
    <tool> tag: any_tokens content (free tokens until <end>).
    <code> tag: sequence of exclude_token + const_string.
    """
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<tool>", "<code>"],
            "tags": [
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<tool>"},
                    "content": {"type": "any_tokens", "exclude_tokens": [16]},
                    "end": {"type": "token", "token": "<end>"},
                },
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<code>"},
                    "content": {
                        "type": "sequence",
                        "elements": [
                            {"type": "exclude_token", "exclude_tokens": [16]},
                            {"type": "const_string", "value": "x"},
                        ],
                    },
                    "end": {"type": "token", "token": "<end>"},
                },
            ],
            "exclude_tokens": [16],
        },
    }
    m, b, ti = _stag_matcher(stag)
    # Free tokens, then trigger <tool>
    _accept_tokens(m, [7, 8])  # hello world
    # <tool> -> any_tokens (multiple) -> <end>
    _accept_tokens(m, [2, 9, 10, 13, 14, 7, 4])  # <tool> { } x y hello <end>
    # Free tokens, then trigger <code>
    _accept_tokens(m, [14])  # y
    # <code> -> exclude_token (single) + "x" -> <end>
    _accept_tokens(m, [3, 7, 13, 4])  # <code> hello(=single exclude_token) x <end>
    assert m.accept_token(STAG_STOP)
    assert m.is_terminated()


def test_stag_sequence_of_token_formats():
    """Sequence of multiple token formats and string content."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {"type": "token", "token": "<tool>"},
                {"type": "const_string", "value": "fn("},
                {"type": "exclude_token", "exclude_tokens": [16, 4]},
                {"type": "const_string", "value": ")"},
                {"type": "token", "token": "<end>"},
            ],
        },
    }
    m, b, ti = _stag_matcher(stag)
    _accept_tokens(m, [2])  # <tool>
    _accept_tokens(m, [11])  # fn(
    assert not m.accept_token(16)  # <bad> excluded
    assert not m.accept_token(4)  # <end> excluded
    _accept_tokens(m, [7])  # hello (single exclude_token)
    _accept_tokens(m, [12])  # )
    _accept_and_stop(m, [4])  # <end>


def test_stag_or_token_and_string_paths():
    """Or format choosing between token-level and string-level paths."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "or",
            "elements": [
                {
                    "type": "sequence",
                    "elements": [
                        {"type": "token", "token": "<tool>"},
                        {"type": "const_string", "value": "hello"},
                    ],
                },
                {"type": "const_string", "value": "world"},
            ],
        },
    }
    # Path A: token path
    m, b, ti = _stag_matcher(stag)
    _accept_and_stop(m, [2, 7])  # <tool> hello

    # Path B: string path
    m2, _, _ = _stag_matcher(stag)
    _accept_and_stop(m2, [8])  # world


def test_stag_complex_multi_dispatch():
    """Complex: outer triggered_tags (string) wrapping inner token_triggered_tags.

    Outer: triggered_tags with string trigger "<tool>" dispatching to a tag whose
    content is a token_triggered_tags.
    Inner: token_triggered_tags with triggers <code> and <think>, each wrapping
    any_tokens content.
    """
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<tool>"],
            "tags": [
                {
                    "type": "tag",
                    "begin": "<tool>",
                    "content": {
                        "type": "token_triggered_tags",
                        "trigger_tokens": ["<code>", "<think>"],
                        "tags": [
                            {
                                "type": "tag",
                                "begin": {"type": "token", "token": "<code>"},
                                "content": {"type": "any_tokens"},
                                "end": {"type": "token", "token": "<end>"},
                            },
                            {
                                "type": "tag",
                                "begin": {"type": "token", "token": "<think>"},
                                "content": {"type": "any_tokens", "exclude_tokens": [16]},
                                "end": {"type": "token", "token": "<think_end>"},
                            },
                        ],
                    },
                    "end": "<end>",
                }
            ],
        },
    }
    m, b, ti = _stag_matcher(stag)
    # Free text before outer trigger
    _accept_tokens(m, [7, 8])  # hello world
    # Outer trigger: <tool> (string match)
    _accept_tokens(m, [2])  # <tool>
    # Now inside token_triggered_tags: free tokens (any except triggers)
    _accept_tokens(m, [13, 14])  # x y
    # Inner trigger: <code> -> any_tokens -> <end>
    _accept_tokens(m, [3, 7, 8, 9, 10, 4])  # <code> hello world { } <end>
    # More free tokens inside token_triggered_tags
    _accept_tokens(m, [7])  # hello
    # Inner trigger: <think> -> any_tokens (exclude <bad>) -> <think_end>
    _accept_tokens(m, [5, 13, 14, 7])  # <think> x y hello
    assert not m.accept_token(16)  # <bad> excluded inside think
    _accept_tokens(m, [6])  # <think_end>
    # End outer tag with string "<end>"
    _accept_tokens(m, [4])  # <end> (string)
    # Free text after outer tag
    _accept_tokens(m, [8])  # world
    assert m.accept_token(STAG_STOP)
    assert m.is_terminated()


def test_stag_star_of_token_sequence():
    """Star wrapping a sequence of token + string elements."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "star",
            "content": {
                "type": "sequence",
                "elements": [
                    {"type": "token", "token": "<tool>"},
                    {"type": "const_string", "value": "x"},
                    {"type": "token", "token": "<end>"},
                ],
            },
        },
    }
    m, b, ti = _stag_matcher(stag)
    # Repeat the pattern 3 times
    for _ in range(3):
        _accept_tokens(m, [2, 13, 4])  # <tool> x <end>
    assert m.accept_token(STAG_STOP)
    assert m.is_terminated()


def test_stag_star_of_token_sequence_zero():
    """Star can match zero repetitions."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "star",
            "content": {
                "type": "sequence",
                "elements": [
                    {"type": "token", "token": "<tool>"},
                    {"type": "const_string", "value": "x"},
                ],
            },
        },
    }
    m, b, ti = _stag_matcher(stag)
    assert m.accept_token(STAG_STOP)
    assert m.is_terminated()


def test_stag_multiple_triggered_tags_rounds():
    """TokenTriggeredTags: multiple rounds of dispatches interleaved with free tokens."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<tool>", "<code>", "<think>"],
            "tags": [
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<tool>"},
                    "content": {"type": "const_string", "value": "hello"},
                    "end": {"type": "token", "token": "<end>"},
                },
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<code>"},
                    "content": {"type": "exclude_token", "exclude_tokens": []},
                    "end": {"type": "token", "token": "<end>"},
                },
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<think>"},
                    "content": {"type": "any_tokens"},
                    "end": {"type": "token", "token": "<think_end>"},
                },
            ],
        },
    }
    m, b, ti = _stag_matcher(stag)
    # Round 1: <tool> hello <end>
    _accept_tokens(m, [2, 7, 4])
    # Free tokens
    _accept_tokens(m, [13, 14, 7])  # x y hello
    # Round 2: <code> (single token) <end>
    _accept_tokens(m, [3, 8, 4])  # <code> world <end>
    # Round 3: <think> any_tokens <think_end>
    _accept_tokens(m, [5, 7, 8, 13, 14, 9, 10, 6])  # <think> ... <think_end>
    # Round 4: <tool> again
    _accept_tokens(m, [2, 7, 4])  # <tool> hello <end>
    # More free tokens then stop
    _accept_tokens(m, [8])  # world
    assert m.accept_token(STAG_STOP)
    assert m.is_terminated()


def test_stag_exclude_token_with_string_excludes():
    """ExcludeTokenFormat with string-based token references in exclude_tokens."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": {"type": "token", "token": "<tool>"},
            "content": {"type": "exclude_token", "exclude_tokens": ["<bad>", "<end>", "<think>"]},
            "end": {"type": "token", "token": "<end>"},
        },
    }
    m, b, ti = _stag_matcher(stag)
    _accept_tokens(m, [2])  # <tool>
    assert not m.accept_token(16)  # <bad> excluded
    assert not m.accept_token(4)  # <end> excluded
    assert not m.accept_token(5)  # <think> excluded
    _accept_tokens(m, [7])  # hello accepted
    _accept_and_stop(m, [4])  # <end>


def test_stag_token_triggered_string_token_refs():
    """TokenTriggeredTags using string references for trigger_tokens and exclude_tokens."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<tool>", "<code>"],
            "tags": [
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<tool>"},
                    "content": {"type": "any_tokens", "exclude_tokens": ["<bad>"]},
                    "end": {"type": "token", "token": "<end>"},
                },
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<code>"},
                    "content": {"type": "const_string", "value": "y"},
                    "end": {"type": "token", "token": "<end>"},
                },
            ],
            "exclude_tokens": ["<bad>"],
        },
    }
    m, b, ti = _stag_matcher(stag)
    assert not m.accept_token(16)  # <bad> excluded at top level
    _accept_tokens(m, [7])  # hello
    # <tool> -> any_tokens (exclude <bad>) -> <end>
    _accept_tokens(m, [2])  # <tool>
    assert not m.accept_token(16)  # <bad> excluded inside any_tokens
    _accept_tokens(m, [8, 13, 4])  # world x <end>
    # <code> -> "y" -> <end>
    _accept_tokens(m, [3, 14, 4])  # <code> y <end>
    assert m.accept_token(STAG_STOP)
    assert m.is_terminated()


def test_stag_tag_with_sequence_content_mixed():
    """Tag content is a complex sequence: token + exclude_token + const_string + any_tokens."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": {"type": "token", "token": "<tool>"},
            "content": {
                "type": "sequence",
                "elements": [
                    {"type": "token", "token": "<code>"},
                    {"type": "exclude_token", "exclude_tokens": [16]},
                    {"type": "const_string", "value": "x"},
                    {"type": "any_tokens", "exclude_tokens": [16]},
                ],
            },
            "end": {"type": "token", "token": "<end>"},
        },
    }
    m, b, ti = _stag_matcher(stag)
    _accept_tokens(m, [2])  # <tool>
    _accept_tokens(m, [3])  # <code> (token)
    assert not m.accept_token(16)  # <bad> excluded
    _accept_tokens(m, [7])  # hello (single exclude_token)
    _accept_tokens(m, [13])  # x (const_string)
    assert not m.accept_token(16)  # <bad> excluded in any_tokens
    _accept_tokens(m, [8, 14, 5])  # world y <think> (any_tokens loop)
    _accept_and_stop(m, [4])  # <end>


def test_stag_or_between_token_triggered_and_string_triggered():
    """Or between token_triggered_tags and triggered_tags paths."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "or",
            "elements": [
                {
                    "type": "token_triggered_tags",
                    "trigger_tokens": ["<tool>"],
                    "tags": [
                        {
                            "type": "tag",
                            "begin": {"type": "token", "token": "<tool>"},
                            "content": {"type": "const_string", "value": "hello"},
                            "end": {"type": "token", "token": "<end>"},
                        }
                    ],
                    "stop_after_first": True,
                    "at_least_one": True,
                },
                {
                    "type": "triggered_tags",
                    "triggers": ["fn("],
                    "tags": [
                        {"type": "tag", "begin": "fn(", "content": {"type": "any_text"}, "end": ")"}
                    ],
                    "stop_after_first": True,
                    "at_least_one": True,
                },
            ],
        },
    }
    # Path A: token triggered
    m, b, ti = _stag_matcher(stag)
    _accept_and_stop(m, [2, 7, 4])  # <tool> hello <end>

    # Path B: string triggered
    m2, _, _ = _stag_matcher(stag)
    _accept_tokens(m2, [11])  # fn(
    _accept_tokens(m2, [7, 8])  # hello world (any_text)
    _accept_and_stop(m2, [12])  # )


def test_stag_deeply_nested_three_layers():
    """Three layers: triggered_tags -> tag -> token_triggered_tags -> tag -> any_tokens.

    Layer 1: String triggered_tags with trigger "fn("
    Layer 2: Tag with string begin "fn(" wrapping token_triggered_tags
    Layer 3: Token triggered tag <think> wrapping any_tokens, and <code> wrapping exclude_token
    """
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["fn("],
            "tags": [
                {
                    "type": "tag",
                    "begin": "fn(",
                    "content": {
                        "type": "token_triggered_tags",
                        "trigger_tokens": ["<think>", "<code>"],
                        "tags": [
                            {
                                "type": "tag",
                                "begin": {"type": "token", "token": "<think>"},
                                "content": {"type": "any_tokens", "exclude_tokens": ["<bad>"]},
                                "end": {"type": "token", "token": "<think_end>"},
                            },
                            {
                                "type": "tag",
                                "begin": {"type": "token", "token": "<code>"},
                                "content": {"type": "exclude_token", "exclude_tokens": ["<bad>"]},
                                "end": {"type": "token", "token": "<end>"},
                            },
                        ],
                        "exclude_tokens": ["<bad>"],
                    },
                    "end": ")",
                }
            ],
        },
    }
    m, b, ti = _stag_matcher(stag)
    # Free text before string trigger
    _accept_tokens(m, [7, 8])  # hello world
    # String trigger: fn(
    _accept_tokens(m, [11])  # fn(
    # Inside token_triggered_tags: free tokens
    _accept_tokens(m, [13])  # x
    assert not m.accept_token(16)  # <bad> excluded at token_triggered level
    # Trigger <think> -> any_tokens (exclude <bad>) -> <think_end>
    _accept_tokens(m, [5, 7, 8, 13, 14])  # <think> hello world x y
    assert not m.accept_token(16)  # <bad> excluded inside any_tokens
    _accept_tokens(m, [6])  # <think_end>
    # More free tokens
    _accept_tokens(m, [14])  # y
    # Trigger <code> -> exclude_token (single, exclude <bad>) -> <end>
    _accept_tokens(m, [3])  # <code>
    assert not m.accept_token(16)  # <bad> excluded inside exclude_token
    _accept_tokens(m, [7])  # hello (single token)
    _accept_tokens(m, [4])  # <end>
    # End outer tag with string ")"
    _accept_tokens(m, [12])  # )
    # Free text after
    _accept_tokens(m, [8])  # world
    assert m.accept_token(STAG_STOP)
    assert m.is_terminated()


def test_stag_any_tokens_all_excluded_except_end():
    """AnyTokensFormat where almost all tokens are excluded, forcing direct end."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": {"type": "token", "token": "<tool>"},
            "content": {
                "type": "any_tokens",
                "exclude_tokens": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            },
            "end": {"type": "token", "token": "<end>"},
        },
    }
    m, b, ti = _stag_matcher(stag)
    _accept_tokens(m, [2])  # <tool>
    # Almost everything excluded; only end token (4) should work to proceed
    _accept_and_stop(m, [4])  # <end>


def test_stag_token_tag_with_or_content():
    """Tag with token begin/end wrapping or content of different types."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": {"type": "token", "token": "<tool>"},
            "content": {
                "type": "or",
                "elements": [
                    {"type": "const_string", "value": "hello"},
                    {
                        "type": "sequence",
                        "elements": [
                            {"type": "exclude_token", "exclude_tokens": [16]},
                            {"type": "const_string", "value": "world"},
                        ],
                    },
                ],
            },
            "end": {"type": "token", "token": "<end>"},
        },
    }
    # Path A: const_string "hello"
    m, b, ti = _stag_matcher(stag)
    _accept_and_stop(m, [2, 7, 4])  # <tool> hello <end>

    # Path B: exclude_token + "world"
    m2, _, _ = _stag_matcher(stag)
    _accept_tokens(m2, [2])  # <tool>
    _accept_tokens(m2, [13])  # x (single exclude_token)
    _accept_tokens(m2, [8])  # world
    _accept_and_stop(m2, [4])  # <end>


def test_stag_mixed_begin_end_types():
    """Tags with different begin/end type combinations within token_triggered_tags."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<tool>", "<code>", "<think>"],
            "tags": [
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<tool>"},
                    "content": {"type": "const_string", "value": "hello"},
                    "end": "<end>",
                },
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<code>"},
                    "content": {"type": "exclude_token", "exclude_tokens": ["<bad>"]},
                    "end": {"type": "token", "token": "<end>"},
                },
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<think>"},
                    "content": {"type": "any_tokens", "exclude_tokens": ["<bad>"]},
                    "end": "<think_end>",
                },
            ],
        },
    }
    m, b, ti = _stag_matcher(stag)
    # Tag 1: token begin, string end
    _accept_tokens(m, [2])  # <tool>
    _accept_tokens(m, [7])  # hello
    _accept_tokens(m, [4])  # <end> (as string)
    # Free tokens
    _accept_tokens(m, [13])  # x
    # Tag 2: token begin, token end
    _accept_tokens(m, [3])  # <code>
    assert not m.accept_token(16)  # <bad> excluded
    _accept_tokens(m, [7])  # hello (single exclude_token)
    _accept_tokens(m, [4])  # <end>
    # Tag 3: token begin, string end
    _accept_tokens(m, [5])  # <think>
    assert not m.accept_token(16)  # <bad> excluded
    _accept_tokens(m, [7, 8, 13])  # hello world x
    _accept_tokens(m, [6])  # <think_end> (as string)
    assert m.accept_token(STAG_STOP)
    assert m.is_terminated()


def test_stag_repeated_token_triggered_tags_different_tags():
    """Cycle through all 3 tag types multiple times in token_triggered_tags."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<tool>", "<code>", "<think>"],
            "tags": [
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<tool>"},
                    "content": {"type": "const_string", "value": "x"},
                    "end": {"type": "token", "token": "<end>"},
                },
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<code>"},
                    "content": {"type": "const_string", "value": "y"},
                    "end": {"type": "token", "token": "<end>"},
                },
                {
                    "type": "tag",
                    "begin": {"type": "token", "token": "<think>"},
                    "content": {"type": "const_string", "value": "hello"},
                    "end": {"type": "token", "token": "<think_end>"},
                },
            ],
        },
    }
    m, b, ti = _stag_matcher(stag)
    for _ in range(3):
        # <tool> x <end>
        _accept_tokens(m, [2, 13, 4])
        # free
        _accept_tokens(m, [7])  # hello
        # <code> y <end>
        _accept_tokens(m, [3, 14, 4])
        # <think> hello <think_end>
        _accept_tokens(m, [5, 7, 6])
    assert m.accept_token(STAG_STOP)
    assert m.is_terminated()


def test_stag_any_tokens_excludes_allow_empty_end():
    """Tags with any_tokens content + exclude_tokens should allow end: "".

    Models a dispatch loop where channels have no explicit end delimiter —
    the content stops when an excluded trigger token appears, and the
    dispatch loop handles the transition.

    Pattern:
        <tool>  hello world  <tool>  x y  </s>
        ^trigger  ^content    ^next trigger

    Each tag is: begin=<tool>, content=any_tokens(exclude=[<tool>,<bad>]), end=""
    The content stops naturally when the next <tool> appears. The dispatch
    loop re-triggers.
    """
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<tool>"],
            "tags": [
                {
                    "begin": {"type": "token", "token": "<tool>"},
                    "content": {"type": "any_tokens", "exclude_tokens": ["<tool>", "<bad>"]},
                    "end": "",
                }
            ],
        },
    }
    m, b, ti = _stag_matcher(stag)

    # First dispatch: <tool> hello world
    _accept_tokens(m, [2, 7, 8])  # <tool> hello world

    # Second dispatch: <tool> x y
    _accept_tokens(m, [2, 13, 14])  # <tool> x y

    # Stop
    assert m.accept_token(STAG_STOP)
    assert m.is_terminated()


def test_stag_any_tokens_exclude_redispatch():
    """any_tokens(exclude=[<tool>]) stops content when <tool> appears, allowing re-dispatch."""
    stag = {
        "type": "structural_tag",
        "format": {
            "type": "token_triggered_tags",
            "trigger_tokens": ["<tool>"],
            "tags": [
                {
                    "begin": {"type": "token", "token": "<tool>"},
                    "content": {"type": "any_tokens", "exclude_tokens": ["<tool>"]},
                    "end": "",
                }
            ],
        },
    }
    m, b, ti = _stag_matcher(stag)

    # First tag: <tool> hello
    _accept_tokens(m, [2, 7])  # <tool> hello

    # <tool> re-triggers (content excluded <tool>, so dispatch takes over)
    _accept_tokens(m, [2, 8])  # <tool> world

    # Third round then stop
    _accept_tokens(m, [2, 13])  # <tool> x
    assert m.accept_token(STAG_STOP)
    assert m.is_terminated()


if __name__ == "__main__":
    pytest.main(sys.argv)
