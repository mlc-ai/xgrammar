"""Correctness tests for building FSMs from grammar sequences.

The FSM builder constructs sequence FSMs by streaming each element (byte string, rule
reference, character class, repetition, token edge) directly into one target FSM. These
tests verify the resulting matcher behavior on real grammars covering every element type,
including exhaustive comparisons against reference regexes. A targeted EBNF corpus also
guards the exact FSM layout produced by these construction paths.
"""

import hashlib
import itertools
import re
import sys
from typing import List, Tuple

import pytest

import xgrammar as xgr
from xgrammar.testing import (
    GrammarFunctor,
    _ebnf_to_grammar_no_normalization,
    _get_masked_tokens_from_bitmask,
    _get_matcher_from_grammar_and_tokenizer_info,
    _is_rule_fsm_accept_string,
    _print_grammar_fsms,
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


# --- Unnormalized nested expressions ---


def test_build_arbitrarily_nested_expressions_without_normalization():
    grammar = _ebnf_to_grammar_no_normalization(
        """
root ::= (("a" | "bb") ("c" | ("d" ("e" | "f")))) | (Regex("[0-9]+") "z")
"""
    )
    grammar = GrammarFunctor.fsm_builder(grammar)

    for accepted in ("ac", "bbc", "ade", "adf", "bbde", "bbdf", "0z", "123z"):
        assert _is_rule_fsm_accept_string(grammar, 0, accepted)
    for rejected in ("", "a", "bc", "ad", "z", "12", "1zz", "ace"):
        assert not _is_rule_fsm_accept_string(grammar, 0, rejected)


@pytest.mark.parametrize(
    "grammar_str",
    [
        """
root ::= "a" TagDispatch(("tag", body), loop_after_dispatch=false) "z"
body ::= "b"
""",
        """
root ::= Token(2) TokenTagDispatch((3, body), excludes=(5,)) Token(6)
body ::= Token(4)
""",
    ],
)
def test_build_nested_dispatch_without_normalization(grammar_str: str):
    grammar = _ebnf_to_grammar_no_normalization(grammar_str)
    grammar = GrammarFunctor.fsm_builder(grammar)
    assert "None" not in _print_grammar_fsms(grammar)


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


def _build_long_sequence_grammar(num_segments: int) -> Tuple[str, str]:
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


# --- Exact EBNF FSM layout stability ---

fsm_structure_cases = [
    ('root ::= "a" [0-9] "b"', "55b4f598dd003190ab972e2c89246b46b7ad102506fdfa785003c270866e8572"),
    (
        'root ::= "hello" [a-zA-Z_] [0-9]* "world"',
        "bf85be1292ab8bfcbdb08c09e4508ffe855d110cc19b36fd849c6724897c6a50",
    ),
    ('root ::= "<" [a-c]* ">"', "19cef267fb8eddbc9c5ac8fba92c135600fc720258abf0bd58cb6c346fe75c5a"),
    (
        'root ::= "x" [^0-9] "y" [^a-z]* "z"',
        "fd5ab210c129fe4da573c175bfd0e97440d30a88d038a160e5bcbeb2f9fc00d0",
    ),
    (
        'root ::= "(" inner ")" inner\ninner ::= [0-9] [0-9]',
        "4de2c028805f73a42e2eb3766defb94a7b6a9cfb735fd9d9a7bf8c610e4b4012",
    ),
    (
        'root ::= "a" item{2,5} "b"\nitem ::= [0-9]',
        "eddead662499416bbd37a4b70d70912ace125deaeaa8fc1f48fbe61dde26b8c7",
    ),
    (
        'root ::= item{3,}\nitem ::= "ab" [xy]',
        "e80cfab029429aa5111b5b1b95c002f7fa4bc7b0a4e88ecae2d507067dba7b6d",
    ),
    (
        'root ::= "ab" [0-9] | "cd" sub | sub sub\nsub ::= [a-f] "q"',
        "29090db9590a3b07af82126f08ec25663f657fde2b659c7d172bf6da227861ba",
    ),
    (
        'root ::= "abcdefghijklmnopqrstuvwxyz" [0-9] "ABCDEFGHIJKLMNOPQRSTUVWXYZ"',
        "58b89fec9137ba57ef63aba58efa201d7a64b9c949b80d0e4376467613779aab",
    ),
    (
        'root ::= "中文" [\\u4e00-\\u9fff] "端"',
        "a2d7f36e1d3a0453d4267fb3937ed07a4abaac7361e051d3aaaaf7a6a58de462",
    ),
    ('root ::= "only"', "7880ea9fd56d9cca1976659fa1ccd4b7b347ab2688800e96bdafcfaa591019e5"),
    ("root ::= [0-9]", "e210303c507b9b23c732079d40b3f0b01af0de6deb86c8974148dbb92b92896c"),
    ('root ::= "" "a" ""', "8698019eb93735ab521cfd930bf7688a276b4ebfb33ccd7cea20f8139f99fbeb"),
    (
        'root ::= a b c\na ::= "x" [0-9]*\nb ::= a "y" | [^xyz]\nc ::= b{1,3} "end"',
        "692de2ea3a886cc7aca953da58fe7dc66c26bc39a71b01fdeee53cfb58e89561",
    ),
]


@pytest.fixture(scope="module")
def structure_compiler():
    vocabulary = [chr(character) for character in range(33, 127)] + ["中", "文", "端"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary)
    return xgr.GrammarCompiler(tokenizer_info, max_threads=1, cache_enabled=False)


@pytest.mark.parametrize("grammar_str, expected_digest", fsm_structure_cases)
def test_compiled_ebnf_fsm_structure_stable(
    structure_compiler, grammar_str: str, expected_digest: str
):
    """Detect changes to state numbers, edge order, endpoints, or complete-FSM layout."""
    compiled = structure_compiler.compile_grammar(grammar_str)
    printed_fsm = _print_grammar_fsms(compiled.grammar)
    actual_digest = hashlib.sha256(printed_fsm.encode()).hexdigest()
    assert actual_digest == expected_digest


if __name__ == "__main__":
    pytest.main(sys.argv)
