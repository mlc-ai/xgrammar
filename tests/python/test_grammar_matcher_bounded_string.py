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

import json

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


def _matcher_after(compiled: xgr.CompiledGrammar, text: str) -> xgr.GrammarMatcher:
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string(text), f"grammar rejected the prefix {text!r}"
    return matcher


def _allowed_token_ids(matcher: xgr.GrammarMatcher, tokenizer_info: xgr.TokenizerInfo) -> set[int]:
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
        assert (
            TOKEN_ID[token] not in allowed
        ), f"maxLength={max_length} allowed {token!r} after the bound"


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
        STRING_START + "a" * min(consumed, max_length) for consumed in (0, 1, 2, max_length)
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


# Tokens that end the repetition on their last byte, tokens that end inside a codepoint, and
# tokens that run past the repetition into what follows it.
EXIT_VOCAB = [b"a", b"aa", b"aaa", b"aaaa", b'"', b'a"', b'aa"', b'aaa"', b'aa",', b",", b"[", b"]"]
EXIT_VOCAB += [b" ", b"\n", b"a\n", b"\xe4", b"a\xe4", b"aa\xe4", b"\xe4\xb8\xad", b"\xb8\xad"]


def _assert_mask_matches_replay(
    compiled: xgr.CompiledGrammar, tokenizer_info: xgr.TokenizerInfo, text: str
) -> None:
    allowed = _allowed_token_ids(_matcher_after(compiled, text), tokenizer_info)
    for token_id in range(tokenizer_info.vocab_size):
        accepted = _matcher_after(compiled, text).accept_token(token_id)
        assert (token_id in allowed) is accepted, (
            f"prefix={text!r} token={tokenizer_info.decoded_vocab[token_id]!r}: "
            f"mask={token_id in allowed} accept={accepted}"
        )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "maxLength": 3},
        {"type": "array", "items": {"type": "string", "maxLength": 3}},
        {"anyOf": [{"type": "string", "maxLength": 1}, {"type": "string", "maxLength": 3}]},
        {"anyOf": [{"type": "string", "maxLength": 1}, {"type": "string"}]},
        {"type": "string", "minLength": 2, "maxLength": 3},
        {"type": "string", "minLength": 2},
        {"anyOf": [{"type": "string", "minLength": 3}, {"type": "string", "minLength": 1}]},
    ],
)
def test_tokens_that_leave_the_string_respect_the_bound(schema: dict[str, object]) -> None:
    """A token that closes the string, or ends inside a codepoint, consumes budget too."""
    tokenizer_info = xgr.TokenizerInfo(list(EXIT_VOCAB))
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(schema)
    start = '["' if schema.get("type") == "array" else '"'
    for consumed in range(4):
        _assert_mask_matches_replay(compiled, tokenizer_info, start + "a" * consumed)
    if start == '["':
        _assert_mask_matches_replay(compiled, tokenizer_info, '["aaa", "a')


@pytest.mark.parametrize(
    ("grammar", "prefixes"),
    [
        # What follows the repetition starts with characters of the class.
        ('root ::= [^x]{0,3} "ab"', ["", "a", "aa", "aaa"]),
        ('root ::= "(" [^()]{0,3} ")" | "[" [^()]{0,3} "]"', ["(", "(a", "[aa"]),
        # A rule that is one character class but is not only used as a repetition body.
        ('root ::= d d "x" | d "y"\nd ::= [0-9]', ["", "1"]),
        # The repetition ends its rule, so what follows it is decided by the parent.
        ('root ::= a "!"\na ::= "<" [^!]{0,3}', ["<", "<a", "<aaa"]),
        ('root ::= "<" [^>]{0,3} ">" [a-z]*', ["<", "<aa", "<aaa"]),
        # Two parents of one repetition body that have done different numbers of repetitions.
        ('root ::= r | "a" r\nr ::= [^()]{2,4} ")"', ["a", "aa", "aaa"]),
        ('root ::= r | "aa" r\nr ::= [^()]{3,} ")"', ["a", "aa", "aab", "aaaa"]),
        # A repetition nested in a repetition.
        ('root ::= "<" s{1,2} ">"\ns ::= [^<>]{2,3}', ["<", "<a", "<aa", "<aaaa"]),
        # Repetitions of one class followed by different bytes.
        ('root ::= "(" [^()]{2,} ")" | "<" [^()]{0,3} "("', ["(", "(aa", "<", "<a"]),
    ],
)
def test_counted_repetitions_in_ebnf_match_the_replay(grammar: str, prefixes: list[str]) -> None:
    vocab = ["a", "aa", "aaa", "aaaa", "ab", "aab", "abab", "b", "x", "(", ")", "[", "]", "a)"]
    vocab += ["a]", "aa]", "<", ">", "a>", "a>b", "!", "a!", "1", "12", "1x", "y", "1y", "d"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
    for text in prefixes:
        _assert_mask_matches_replay(compiled, tokenizer_info, text)


@pytest.mark.parametrize(
    ("compile_args", "accepted", "rejected"),
    [
        (
            {"schema": {"anyOf": [{"const": "aaa"}, {"type": "string", "maxLength": 2}]}},
            ['"aaa"', '"ab"', '"b"'],
            ['"baaa"', '"baa"'],
        ),
        ({"grammar": 'root ::= "<" ("ab" | [^"x]{0,2} "\\"")'}, ["<ab", '<aa"'], ["<aab", "<Zab"]),
    ],
)
def test_counted_repetition_does_not_continue_into_a_sibling_alternative(
    compile_args: dict[str, object], accepted: list[str], rejected: list[str]
) -> None:
    """After a repetition the parser returns to the repeat edge, which is not a branch point."""
    tokenizer_info = xgr.TokenizerInfo(["a", "b", "Z", '"', "<"])
    compiler = xgr.GrammarCompiler(tokenizer_info)
    if "schema" in compile_args:
        compiled = compiler.compile_json_schema(compile_args["schema"])
    else:
        compiled = compiler.compile_grammar(compile_args["grammar"])
    for text in accepted + rejected:
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert (matcher.accept_string(text) and matcher.is_terminated()) is (text in accepted), text
    for text in ('"', '"b', '"ba') if "schema" in compile_args else ("<", "<a", "<aa"):
        _assert_mask_matches_replay(compiled, tokenizer_info, text)


@pytest.mark.parametrize(
    ("grammar", "first_token"),
    [
        (
            'root ::= a | b\na[max_tokens=1] ::= "p" [^x]{0,3} "x"\nb[max_tokens=2] ::= [^x]{0,3} "x"',
            "pp",
        ),
        # Repetitions with different bounds or different following bytes do not share a body
        # between budgeted parents.
        (
            'root ::= a | b\na[max_tokens=1] ::= [^xy]{0,2} "x"\nb[max_tokens=2] ::= [^xy]{0,3} "y"',
            "a",
        ),
        (
            'root ::= a | b\na[max_tokens=1] ::= [^xy]{1,2} "x"\nb[max_tokens=2] ::= [^xy]{1,3} "y"',
            "a",
        ),
        (
            'root ::= a | b\na[max_tokens=1] ::= [^xy]{0,2} "x" "!"\n'
            'b[max_tokens=2] ::= [^xy]{0,3} "x" "?"',
            "a",
        ),
        # One body with the same bounds and following byte, entered from an expired and a live
        # parent: replaying from the live one must not complete into the expired one.
        ('root ::= a | b\na[max_tokens=1] ::= [^x]{2,} "x!"\nb ::= "a" [^x]{2,} "x?"', "a"),
        ('root ::= a | b\na[max_tokens=1] ::= [^x]{0,5} "x!"\nb ::= "a" [^x]{0,5} "x?"', "a"),
        # The same with a plain rule shared by two budgeted parents.
        (
            'root ::= a | b\na[max_tokens=1] ::= c "x"\nb[max_tokens=2] ::= c "y"\n'
            "c ::= [^xy] [^xy]?",
            "a",
        ),
    ],
)
def test_expired_parent_does_not_lend_its_repetitions(grammar: str, first_token: str) -> None:
    """Once the budget is enforced, only the repetitions of a live parent count."""
    vocab = ["a", "aa", "b", "ba", '"', 'a"', "p", "pp", "x", "ax", "y", "ay", "ax!", "ax?"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
    for token_id in range(len(vocab)):
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        assert matcher.accept_token(vocab.index(first_token))
        # The budget is enforced by the accept that follows a fill.
        allowed = token_id in _allowed_token_ids(matcher, tokenizer_info)
        assert allowed is bool(matcher.accept_token(token_id)), vocab[token_id]


@pytest.mark.parametrize("min_length", [0, 2])
def test_deserialized_bounded_string_masks_match(min_length: int) -> None:
    tokenizer_info = xgr.TokenizerInfo(list(EXIT_VOCAB))
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(
        {"type": "string", "minLength": min_length, "maxLength": 3}
    )
    restored = xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), tokenizer_info)
    for consumed in range(4):
        text = '"' + "a" * consumed
        assert _allowed_token_ids(_matcher_after(restored, text), tokenizer_info) == (
            _allowed_token_ids(_matcher_after(compiled, text), tokenizer_info)
        ), text
        _assert_mask_matches_replay(restored, tokenizer_info, text)


def test_deserialize_rejects_a_mask_for_an_unknown_rule() -> None:
    tokenizer_info = xgr.TokenizerInfo(list(EXIT_VOCAB))
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(
        {"type": "string", "maxLength": 3}
    )
    serialized = json.loads(compiled.serialize_json())
    serialized["adaptive_token_mask_cache"][0][0][0] = 2**31 - 1
    with pytest.raises(xgr.exception.DeserializeFormatError):
        xgr.CompiledGrammar.deserialize_json(json.dumps(serialized), tokenizer_info)


def test_nested_repetition_keeps_the_expanded_repetition() -> None:
    """The byte after the outer repetition does not follow the inner one, so the inner one is not
    turned into a counted edge that the matcher cannot take the fast path for."""
    tokenizer_info = xgr.TokenizerInfo(list(EXIT_VOCAB))
    grammar = 'root ::= "<" s{0,2} "\\""\ns ::= [^"]{2,3}'
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
    assert "{2, 3}" not in str(compiled.grammar)


@pytest.mark.parametrize(
    "grammar",
    [
        'root ::= s ","\ns[lazy] ::= "\\"" [^"]{2} "\\""',
        'root ::= s ","\ns[lazy] ::= "\\"" [^"]{3,} "\\""',
        'root ::= s ","\ns[lazy] ::= "\\"" [^"]{0,} "\\""',
        'root ::= s ","\ns[lazy] ::= "\\"" t\nt ::= [^"]{3,} "\\""',
    ],
)
def test_lazy_rule_keeps_the_expanded_repetition(grammar: str) -> None:
    """A lazy body is flattened, which a counted edge would prevent, so it keeps the expansion."""
    tokenizer_info = xgr.TokenizerInfo(list(EXIT_VOCAB))
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
    for consumed in range(3):
        _assert_mask_matches_replay(compiled, tokenizer_info, '"' + "a" * consumed)


def test_lazy_rule_elsewhere_keeps_the_counted_repetition() -> None:
    """A repetition with `lower == 0` and an upper bound stays counted next to a lazy rule."""
    tokenizer_info = xgr.TokenizerInfo(list(EXIT_VOCAB))
    grammar = 'root ::= "\\"" [^"]{0,3} "\\"" t\nt[lazy] ::= ","'
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
    assert "{0, 3}" in str(compiled.grammar)
    for consumed in range(4):
        _assert_mask_matches_replay(compiled, tokenizer_info, '"' + "a" * consumed)


def test_character_budget_grammar_keeps_the_expanded_repetition() -> None:
    """With a character budget anywhere the matcher never uses the counted fast path, so the
    repetition is expanded as before instead of becoming a counted edge on the slow path."""
    tokenizer_info = xgr.TokenizerInfo(list(EXIT_VOCAB))
    grammar = 'root ::= "\\"" [^"\\\\]{0,3} "\\"" | t\nt[max_chars=2] ::= [a-z]*'
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
    assert "{0, 3}" not in str(compiled.grammar)
    for consumed in range(4):
        _assert_mask_matches_replay(compiled, tokenizer_info, '"' + "a" * consumed)
