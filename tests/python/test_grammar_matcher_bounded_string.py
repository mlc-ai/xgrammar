"""Regression tests for JSON strings with ``minLength``/``maxLength`` (issue #852).

A length-bounded string compiles into a counted repetition of the string body.
The remaining repetition budget lives in the parser state, while the compiled
token mask is shared by every repeat count, so the matcher has to keep rejecting
a token that would push the string past its bound even when the token only
contains characters the body accepts. These tests pin that behaviour and check
that the mask agrees with a fresh replay of the same prefix, including exactly
at the bound.
"""

from __future__ import annotations

import pytest

import xgrammar as xgr

# Tokens that only contain characters accepted by the string body, tokens that
# close the string or the object, and multi-byte characters so that the bound is
# checked in codepoints rather than bytes.
VOCAB = [
    "{",
    "}",
    '"',
    '"}',
    '"content": "',
    "content",
    ":",
    " ",
    "a",
    "aa",
    "aaa",
    "aaaa",
    "b",
    "bb",
    "中",
    "中文",
    "\n",
]

TOKEN_ID = {token: index for index, token in enumerate(VOCAB)}

STRING_START = '{"content": "'


def _tokenizer_info() -> xgr.TokenizerInfo:
    return xgr.TokenizerInfo(list(VOCAB))


def _compile(max_length: int | None, min_length: int | None = None) -> xgr.CompiledGrammar:
    content: dict[str, object] = {"type": "string"}
    if max_length is not None:
        content["maxLength"] = max_length
    if min_length is not None:
        content["minLength"] = min_length
    schema = {"type": "object", "properties": {"content": content}, "required": ["content"]}
    tokenizer_info = _tokenizer_info()
    return xgr.GrammarCompiler(tokenizer_info).compile_json_schema(schema)


def _matcher_after(
    compiled: xgr.CompiledGrammar, text: str
) -> xgr.GrammarMatcher:
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string(text), f"grammar rejected the prefix {text!r}"
    return matcher


def _allowed_token_ids(
    matcher: xgr.GrammarMatcher, tokenizer_info: xgr.TokenizerInfo
) -> set[int]:
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(bitmask)
    return {
        token_id
        for token_id in range(tokenizer_info.vocab_size)
        if (int(bitmask[0, token_id // 32]) >> (token_id % 32)) & 1
    }


@pytest.mark.parametrize("max_length", [0, 1, 2, 3, 8])
def test_bounded_string_rejects_tokens_past_the_bound(max_length: int) -> None:
    """At the bound the string may close, but no token may extend it."""
    tokenizer_info = _tokenizer_info()
    compiled = _compile(max_length)
    matcher = _matcher_after(compiled, STRING_START + "a" * max_length)

    allowed = _allowed_token_ids(matcher, tokenizer_info)
    assert TOKEN_ID['"'] in allowed, "the string must still be closable at the bound"
    for token in ("a", "aa", "aaa", "aaaa", "b", "bb"):
        assert TOKEN_ID[token] not in allowed, (
            f"maxLength={max_length} allowed {token!r} after the bound"
        )


@pytest.mark.parametrize(
    ("max_length", "consumed", "token", "expected"),
    [
        (8, 0, "aaaa", True),
        (8, 4, "aaaa", True),
        (8, 7, "a", True),
        (8, 7, "aa", False),
        (8, 7, "aaaa", False),
        (4, 1, "aaa", True),
        (4, 1, "aaaa", False),
        (1, 0, "a", True),
        (1, 0, "aa", False),
        (2, 0, "中文", True),
        (2, 1, "中文", False),
        (2, 2, "中", False),
    ],
)
def test_bounded_string_budget_is_checked_in_characters(
    max_length: int, consumed: int, token: str, expected: bool
) -> None:
    tokenizer_info = _tokenizer_info()
    compiled = _compile(max_length)
    matcher = _matcher_after(compiled, STRING_START + "a" * consumed)
    allowed = _allowed_token_ids(matcher, tokenizer_info)
    assert (TOKEN_ID[token] in allowed) is expected


def test_min_length_delays_the_exit_but_not_the_repetition() -> None:
    """``minLength`` constrains when the string may close, not how many characters fit."""
    tokenizer_info = _tokenizer_info()
    compiled = _compile(4, 2)
    allowed = _allowed_token_ids(_matcher_after(compiled, STRING_START), tokenizer_info)
    assert TOKEN_ID['"'] not in allowed, "the string is shorter than minLength"
    assert TOKEN_ID["aa"] in allowed, "repetition must not be blocked by minLength"

    allowed = _allowed_token_ids(_matcher_after(compiled, STRING_START + "aa"), tokenizer_info)
    assert TOKEN_ID['"'] in allowed, "the string reached minLength and may close"
    assert TOKEN_ID["aa"] in allowed, "the string may still grow up to maxLength"
    assert TOKEN_ID["aaa"] not in allowed, "that token would exceed maxLength"


@pytest.mark.parametrize("max_length", [0, 1, 2, 8])
def test_mask_agrees_with_a_fresh_replay_at_the_bound(max_length: int) -> None:
    """Every mask bit must match what a fresh matcher does with that token."""
    tokenizer_info = _tokenizer_info()
    compiled = _compile(max_length)
    prefixes = [
        STRING_START + "a" * min(consumed, max_length)
        for consumed in (0, 1, 2, max_length)
    ] + [STRING_START + "a" * max_length + '"']
    for text in prefixes:
        allowed = _allowed_token_ids(_matcher_after(compiled, text), tokenizer_info)
        for token_id in range(tokenizer_info.vocab_size):
            accepted = _matcher_after(compiled, text).accept_token(token_id)
            assert (token_id in allowed) is accepted, (
                f"maxLength={max_length} prefix={text!r} token={VOCAB[token_id]!r}: "
                f"mask={token_id in allowed} accept={accepted}"
            )


def test_unbounded_and_bounded_masks_agree_away_from_the_bound() -> None:
    """Away from the bound the bound must not change which tokens are allowed.

    The bound is large enough that no vocabulary token can reach it from these
    prefixes; a shorter bound legitimately forbids the long tokens.
    """
    tokenizer_info = _tokenizer_info()
    unbounded = _compile(None)
    bounded = _compile(32)
    for text in (STRING_START, STRING_START + "a", STRING_START + "aaa"):
        assert _allowed_token_ids(_matcher_after(unbounded, text), tokenizer_info) == (
            _allowed_token_ids(_matcher_after(bounded, text), tokenizer_info)
        ), text


def test_string_at_the_bound_still_completes_the_object() -> None:
    compiled = _compile(4)
    matcher = _matcher_after(compiled, STRING_START + "aaaa")
    assert matcher.accept_token(TOKEN_ID['"'])
    assert matcher.accept_token(TOKEN_ID["}"])
    assert matcher.is_terminated()


def test_bound_is_enforced_in_codepoints_for_multibyte_tokens() -> None:
    """A 2-codepoint token counts as two repetitions, not as its byte length."""
    tokenizer_info = _tokenizer_info()
    compiled = _compile(2)
    matcher = _matcher_after(compiled, STRING_START)
    allowed = _allowed_token_ids(matcher, tokenizer_info)
    assert TOKEN_ID["中文"] in allowed
    assert TOKEN_ID["中"] in allowed
    assert matcher.accept_token(TOKEN_ID["中文"])
    allowed = _allowed_token_ids(matcher, tokenizer_info)
    assert TOKEN_ID['"'] in allowed
    assert TOKEN_ID["中"] not in allowed
    assert TOKEN_ID["中文"] not in allowed
