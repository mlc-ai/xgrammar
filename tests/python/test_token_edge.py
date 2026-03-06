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
    rejected = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)

    assert 2 not in rejected
    assert 4 not in rejected
    assert 0 in rejected  # <s>
    assert 3 in rejected  # "bb"
    assert 5 in rejected  # "dd"


def test_bitmask_token_and_string():
    """Bitmask for Token(2) | "bb" should allow token 2 and token whose text is "bb"."""
    vocab = ["<s>", "</s>", "aa", "bb", "cc"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf('root ::= Token(2) | "bb"\n')
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, tokenizer_info)

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)

    assert 2 not in rejected  # Token(2) via kTokenSet
    assert 3 not in rejected  # "bb" via char path


def test_bitmask_after_token():
    """After accepting a Token, the bitmask should reflect the next expected tokens."""
    vocab = ["<s>", "</s>", "aa", "bb", "cc"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf('root ::= Token(2) "bb"\n')
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, tokenizer_info)

    # First: only token 2 should be allowed
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert 2 not in rejected

    # Accept token 2
    assert matcher.accept_token(2)

    # Second: "bb" tokens should be allowed
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected2 = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert 3 not in rejected2  # "bb" token


def test_token_multiple_choices():
    """Token set with multiple IDs in alternation with other rules."""
    vocab = ["<s>", "</s>", "x", "y", "z", "w"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    grammar = xgr.Grammar.from_ebnf('root ::= Token(2, 3, 4) | "w"\n')
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar, tokenizer_info)

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    rejected = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)

    assert 2 not in rejected  # "x" via kTokenSet
    assert 3 not in rejected  # "y" via kTokenSet
    assert 4 not in rejected  # "z" via kTokenSet
    assert 5 not in rejected  # "w" via char path
    assert 0 in rejected  # "<s>" rejected


# --- 4.3 char-then-token sequence tests ---


def test_char_then_token_sequence():
    """String literal followed by Token: "A" Token(4, 5)."""
    vocab = ["<s>", "</s>", "A", "B", "hello", "world"]
    matcher = _make_matcher(vocab, 'root ::= "A" Token(4, 5)\n')
    assert matcher.accept_token(2)  # "A"
    assert matcher.accept_token(4)  # Token(4) = "hello"
    assert matcher.accept_token(STOP_TOKEN_ID)
    assert matcher.is_terminated()


# --- 4.5 TagDispatch + excludes (int) tests ---


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


def test_tag_dispatch_excludes_token_no_tags():
    """Pure free text with token blocking via kTokenExclude filter."""
    vocab = ["<s>", "</s>", "hello", "world", "blocked_1", "blocked_2"]
    grammar_str = """root ::= TagDispatch(
      stop_eos=true,
      excludes=(4, 5)
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)

    for _ in range(3):
        accepted = _get_accepted(matcher, bitmask, ti.vocab_size)
        assert 4 not in accepted  # blocked_1
        assert 5 not in accepted  # blocked_2
        assert 2 in accepted  # hello
        assert 3 in accepted  # world
        assert 1 in accepted  # </s> (stop_eos)
        matcher.accept_token(2)


def test_tag_dispatch_excludes_token_blocks_despite_char_edges():
    """kTokenExclude filter blocks tokens even when catchall char edges accept them."""
    vocab = ["<s>", "</s>", "hello", "world", "bad"]
    grammar_str = """root ::= TagDispatch(
      stop_eos=true,
      excludes=(4)
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)
    accepted = _get_accepted(matcher, bitmask, ti.vocab_size)
    assert 4 not in accepted  # "bad" blocked by filter
    assert 2 in accepted  # "hello" accepted


# --- 4.6 TagDispatch + token trigger tests ---


def test_tag_dispatch_token_trigger():
    """Token trigger dispatches to a rule."""
    vocab = ["<s>", "</s>", "hello", "trigger_tok", "content"]
    grammar_str = """
    triggered_rule ::= Token(4)
    root ::= TagDispatch(
      (3, triggered_rule),
      stop_eos=true
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)

    accepted = _get_accepted(matcher, bitmask, ti.vocab_size)
    assert 3 in accepted  # trigger accepted
    assert 2 in accepted  # free text accepted

    assert matcher.accept_token(3)  # dispatch trigger

    accepted2 = _get_accepted(matcher, bitmask, ti.vocab_size)
    assert accepted2 == {4}  # only Token(4) from triggered_rule


def test_tag_dispatch_mixed_triggers():
    """String trigger and token trigger in same TagDispatch."""
    vocab = ["<s>", "</s>", "A", "B", "<tool>", "content"]
    grammar_str = """
    tool_body ::= Token(5)
    other_body ::= Token(5)
    root ::= TagDispatch(
      ("<tool>", tool_body),
      (3, other_body),
      stop_eos=true
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)
    accepted = _get_accepted(matcher, bitmask, ti.vocab_size)
    assert 3 in accepted  # token trigger "B" accepted

    assert matcher.accept_token(3)  # dispatch to other_body

    accepted2 = _get_accepted(matcher, bitmask, ti.vocab_size)
    assert accepted2 == {5}  # only Token(5)


def test_tag_dispatch_token_trigger_loop():
    """Token trigger with loop_after_dispatch returns to start after body completes."""
    vocab = ["<s>", "</s>", "hello", "trigger", "content"]
    grammar_str = """
    body ::= Token(4)
    root ::= TagDispatch(
      (3, body),
      stop_eos=true,
      loop_after_dispatch=true
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)

    assert matcher.accept_token(3)  # trigger dispatches to body
    assert matcher.accept_token(4)  # Token(4) completes body

    accepted = _get_accepted(matcher, bitmask, ti.vocab_size)
    assert 3 in accepted  # trigger accepted again
    assert 2 in accepted  # free text accepted again
    assert 1 in accepted  # </s> accepted (stop_eos)


def test_tag_dispatch_trigger_and_exclude_no_overlap():
    """Token trigger IDs and excludes (int) must not overlap."""
    grammar_str = """
    body ::= Token(2)
    root ::= TagDispatch(
      (3, body),
      stop_eos=true,
      excludes=(3,)
    )"""
    with pytest.raises(Exception):
        xgr.Grammar.from_ebnf(grammar_str)


def test_tag_dispatch_trigger_excluded_from_free_text():
    """Trigger tokens are excluded from catchall free text path via kTokenExclude,
    but still accepted via Token dispatch path."""
    vocab = ["<s>", "</s>", "hello", "trigger", "content"]
    grammar_str = """
    body ::= Token(4)
    root ::= TagDispatch(
      (3, body),
      stop_eos=true
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)

    accepted = _get_accepted(matcher, bitmask, ti.vocab_size)
    assert 3 in accepted  # trigger accepted via Token edge

    assert matcher.accept_token(3)  # dispatch trigger

    accepted2 = _get_accepted(matcher, bitmask, ti.vocab_size)
    assert accepted2 == {4}  # only body's Token(4), not free text


def test_tag_dispatch_full_combo():
    """String trigger + token trigger + excludes (int) all working together."""
    vocab = ["<s>", "</s>", "hello", "B", "<tool>", "content", "blocked"]
    grammar_str = """
    tool_body ::= Token(5)
    other_body ::= Token(5)
    root ::= TagDispatch(
      ("<tool>", tool_body),
      (3, other_body),
      stop_eos=true,
      excludes=(6)
    )"""
    matcher, ti, bitmask = _make_bitmask_helper(vocab, grammar_str)
    accepted = _get_accepted(matcher, bitmask, ti.vocab_size)
    assert 6 not in accepted  # "blocked" excluded
    assert 3 in accepted  # token trigger "B" accepted
    assert 2 in accepted  # free text "hello" accepted
    assert 1 in accepted  # </s> accepted (stop_eos)


# --- 4.7 Lookahead Assertion + kTokenSet tests ---


def test_lookahead_exact_with_token_set():
    """Exact lookahead containing kTokenSet: tokens matching the rule are accepted."""
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
    rejected = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert 2 not in rejected  # "abc" matches rule_a
    assert 3 not in rejected  # "abcd" matches rule_a


def test_lookahead_token_set_suffix_nonempty_rejected():
    """Token that partially matches rule but has bytes left at kTokenSet boundary → rejected."""
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
    rejected = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert 3 not in rejected  # "a" exactly fills rule_a → accepted
    assert 2 in rejected  # "ab" has suffix "b" that can't enter Token(4) → rejected


def test_lookahead_mixed_char_and_token():
    """Lookahead with char elements before kTokenSet."""
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
    rejected = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert 3 not in rejected  # "abc!" matches rule_a="abc" + "!"
