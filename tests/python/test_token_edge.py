"""Tests for Token() edge support in grammar parsing, matching, and bitmask generation."""

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
