"""Correctness tests for building FSMs from grammar sequences.

The FSM builder constructs sequence FSMs by streaming each element (byte string, rule
reference, character class, repetition, token edge) directly into one target FSM. These
tests verify the resulting matcher behavior on real grammars covering every element type,
including exhaustive comparisons against reference regexes.
"""

import itertools
import re
import sys
from typing import List

import pytest

import xgrammar as xgr
from xgrammar.testing import (
    _get_masked_tokens_from_bitmask,
    _get_matcher_from_grammar_and_tokenizer_info,
)


def _make_string_matcher(grammar_str: str) -> xgr.GrammarMatcher:
    tokenizer_info = xgr.TokenizerInfo([])
    compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)
    compiled = compiler.compile_grammar(grammar_str)
    return xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)


def _matcher_accepts(matcher: xgr.GrammarMatcher, input_str: str) -> bool:
    matcher.reset()
    return matcher.accept_string(input_str) and matcher.is_terminated()


# --- Exhaustive comparison against reference regexes ---

# Each entry: (grammar, equivalent regex, alphabet, max enumerated length)
grammar_regex_alphabet_cases = [
    # Byte strings separated by character classes: the sequence FSM path with
    # multiple simple elements.
    ('root ::= "a" [0-9] "b"', r"a[0-9]b", "a0b", 4),
    ('root ::= "ab" [xy] "cd"', r"ab[xy]cd", "abxycd", 6),
    # Character class star inside a sequence.
    ('root ::= "<" [a-c]* ">"', r"<[a-c]*>", "<abc>", 5),
    # Negated character class inside a sequence.
    ('root ::= "<" [^>] ">"', r"<[^>]>", "<ab>", 4),
    # Rule reference between byte strings, including an empty alternative.
    ('root ::= "a" mid "c"\nmid ::= "b" | "x" | ""', r"a(b|x|)c", "abxc", 4),
    # Repetition range in a sequence.
    ('root ::= "a" [xy]{1, 3} "z"', r"a[xy]{1,3}z", "axyz", 6),
    # Nested alternation followed by sequence elements.
    ('root ::= ("a" | "bb") [0-9] "z"', r"(a|bb)[0-9]z", "ab09z", 5),
    # Single-element sequences of each type.
    ('root ::= "abc"', r"abc", "abc", 4),
    ("root ::= [a-z]", r"[a-z]", "az", 3),
    ("root ::= [a-b]*", r"[a-b]*", "ab", 4),
]


@pytest.mark.parametrize("grammar_str, regex, alphabet, max_len", grammar_regex_alphabet_cases)
def test_sequence_matches_reference_regex(
    grammar_str: str, regex: str, alphabet: str, max_len: int
):
    """Enumerate every string up to max_len over the alphabet and compare the matcher
    against Python's regex engine."""
    matcher = _make_string_matcher(grammar_str)
    pattern = re.compile(regex)
    checked = 0
    for length in range(max_len + 1):
        for candidate_chars in itertools.product(alphabet, repeat=length):
            candidate = "".join(candidate_chars)
            expected = pattern.fullmatch(candidate) is not None
            assert _matcher_accepts(matcher, candidate) == expected, (
                f"Mismatch for input {candidate!r}: grammar={grammar_str!r} regex={regex!r} "
                f"expected={expected}"
            )
            checked += 1
    assert checked > 1


# --- UTF-8 multi-byte content in sequences ---


def test_sequence_with_utf8_byte_strings():
    grammar_str = 'root ::= "你" [好世] "界"'
    matcher = _make_string_matcher(grammar_str)
    assert _matcher_accepts(matcher, "你好界")
    assert _matcher_accepts(matcher, "你世界")
    assert not _matcher_accepts(matcher, "你界")
    assert not _matcher_accepts(matcher, "你好世界")
    assert not _matcher_accepts(matcher, "好界")
    assert not _matcher_accepts(matcher, "你好")


# --- Long mixed sequences stress the streaming concatenation loop ---


def _build_long_sequence_grammar(num_segments: int) -> (str, str):
    """A single rule whose body alternates byte strings and character classes."""
    elements: List[str] = []
    valid_parts: List[str] = []
    for index in range(num_segments):
        literal = f"s{index:02d}"
        elements.append(f'"{literal}"')
        elements.append("[0-9]")
        valid_parts.append(literal)
        valid_parts.append(str(index % 10))
    grammar_str = "root ::= " + " ".join(elements)
    return grammar_str, "".join(valid_parts)


def test_long_mixed_sequence():
    grammar_str, valid_input = _build_long_sequence_grammar(64)
    matcher = _make_string_matcher(grammar_str)
    assert _matcher_accepts(matcher, valid_input)
    # Truncations must be rejected.
    assert not _matcher_accepts(matcher, valid_input[:-1])
    assert not _matcher_accepts(matcher, valid_input[: len(valid_input) // 2])
    # Any extension must be rejected.
    assert not _matcher_accepts(matcher, valid_input + "0")
    # Mutating one character at several positions must be rejected.
    for position in range(0, len(valid_input), 17):
        original_char = valid_input[position]
        replacement = "x" if original_char != "x" else "y"
        mutated = valid_input[:position] + replacement + valid_input[position + 1 :]
        assert not _matcher_accepts(matcher, mutated), f"position {position}"


# --- Recursive rule references inside sequences ---


def test_recursive_rule_ref_in_sequence():
    grammar_str = 'root ::= "(" root ")" | ""'
    matcher = _make_string_matcher(grammar_str)
    for depth in (0, 1, 2, 8, 32):
        assert _matcher_accepts(matcher, "(" * depth + ")" * depth)
    assert not _matcher_accepts(matcher, "(")
    assert not _matcher_accepts(matcher, "(()")
    assert not _matcher_accepts(matcher, "())")


# --- Empty sequence ---


def test_empty_sequence():
    matcher = _make_string_matcher('root ::= ""')
    assert _matcher_accepts(matcher, "")
    assert not _matcher_accepts(matcher, "a")


# --- Large repetition ranges that stay as repeat edges ---


def test_large_repetition_in_sequence():
    grammar_str = 'root ::= "a" [xy]{2, 100} "z"'
    matcher = _make_string_matcher(grammar_str)
    assert not _matcher_accepts(matcher, "axz")
    assert _matcher_accepts(matcher, "a" + "xy" * 1 + "z")
    assert _matcher_accepts(matcher, "a" + "x" * 100 + "z")
    assert not _matcher_accepts(matcher, "a" + "x" * 101 + "z")
    assert not _matcher_accepts(matcher, "a" + "x" * 50)


# --- Token and ExcludeToken edges inside sequences ---

TOKEN_TEST_VOCAB = ["<s>", "</s>", "aa", "bb", "cc", "dd"]
#                    0      1       2     3     4     5
STOP_TOKEN_ID = 1


def _make_token_matcher(grammar_str: str) -> xgr.GrammarMatcher:
    tokenizer_info = xgr.TokenizerInfo(TOKEN_TEST_VOCAB)
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    return _get_matcher_from_grammar_and_tokenizer_info(grammar, tokenizer_info)


def test_token_edge_between_byte_strings():
    matcher = _make_token_matcher('root ::= "aa" Token(3, 4) "dd"\n')
    assert matcher.accept_token(2)  # "aa"
    assert not matcher.accept_token(5)  # "dd" not in Token(3, 4)
    assert matcher.accept_token(3)  # "bb"
    assert matcher.accept_token(5)  # "dd"
    assert matcher.accept_token(STOP_TOKEN_ID)
    assert matcher.is_terminated()


def test_exclude_token_edge_in_sequence():
    matcher = _make_token_matcher('root ::= "aa" ExcludeToken(3) "dd"\n')
    assert matcher.accept_token(2)  # "aa"
    assert not matcher.accept_token(3)  # excluded
    assert matcher.accept_token(4)  # "cc"
    assert matcher.accept_token(5)  # "dd"
    assert matcher.accept_token(STOP_TOKEN_ID)
    assert matcher.is_terminated()


# --- Token bitmask consistency with string acceptance ---

mask_consistency_grammars = [
    'root ::= "ab" [0-9] "cd"',
    'root ::= "a" mid "c"\nmid ::= "b" | "x" | ""',
    'root ::= "<" [a-c]* ">"',
    'root ::= "a" [xy]{1, 3} "z"',
]


@pytest.mark.parametrize("grammar_str", mask_consistency_grammars)
def test_bitmask_matches_string_acceptance(grammar_str: str):
    """At every step of a generation, the bitmask must allow exactly the vocabulary
    pieces the matcher would accept as the next characters."""
    vocab = ["a", "b", "ab", "0", "5", "c", "d", "cd", "x", "y", "z", "<", ">", "bc"]
    tokenizer_info = xgr.TokenizerInfo(vocab)
    compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)
    compiled = compiler.compile_grammar(grammar_str)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    valid_inputs = {
        'root ::= "ab" [0-9] "cd"': "ab5cd",
        'root ::= "a" mid "c"\nmid ::= "b" | "x" | ""': "axc",
        'root ::= "<" [a-c]* ">"': "<abc>",
        'root ::= "a" [xy]{1, 3} "z"': "axyz",
    }
    remaining = valid_inputs[grammar_str]

    while True:
        matcher.fill_next_token_bitmask(bitmask)
        rejected = set(_get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size))
        for token_id, piece in enumerate(vocab):
            fork = matcher.fork()
            piece_accepted = fork.accept_string(piece)
            assert piece_accepted == (token_id not in rejected), (
                f"Bitmask disagrees with accept_string for piece {piece!r} "
                f"after consuming {valid_inputs[grammar_str][: -len(remaining) or None]!r}"
            )
        if not remaining:
            break
        assert matcher.accept_string(remaining[0])
        remaining = remaining[1:]
    assert matcher.is_terminated()


if __name__ == "__main__":
    pytest.main(sys.argv)
