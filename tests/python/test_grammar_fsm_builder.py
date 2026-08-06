"""Correctness tests for building FSMs from grammar sequences.

The FSM builder constructs sequence FSMs by streaming each element (byte string, rule
reference, character class, repetition, token edge) directly into one target FSM. These
tests verify the resulting matcher behavior on real grammars covering every element type,
including exhaustive comparisons against reference regexes. A targeted EBNF corpus also
guards the exact FSM layout produced by these construction paths.
"""

import itertools
import re
import sys
from textwrap import dedent
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


def test_large_byte_prefix_rule_ref_suffix_choices():
    prefixes = ["a" * length for length in range(1, 65)]
    alternatives = []
    for index, prefix in enumerate(prefixes):
        suffix = "!" if index % 2 == 0 else "?"
        alternatives.append(f'"{prefix}" body "{suffix}"')
    grammar = "root ::= " + " | ".join(alternatives) + "\nbody ::= [0-9]"
    matcher = _make_string_matcher(grammar)

    for index, prefix in enumerate(prefixes):
        suffix = "!" if index % 2 == 0 else "?"
        assert _matcher_accepts(matcher, prefix + "7" + suffix)
        assert not _matcher_accepts(matcher, prefix + "7" + ("?" if suffix == "!" else "!"))
        assert not _matcher_accepts(matcher, prefix + suffix)


def test_nested_byte_prefix_rule_ref_suffix_choices_merge_outer_fsm():
    grammar = _ebnf_to_grammar_no_normalization(
        'root ::= ("a" body "x" | "ab" body "x") ("cd" | "ed")\nbody ::= [0-9]'
    )
    grammar = GrammarFunctor.fsm_builder(grammar)

    assert "Rule 0: root, FSM: CompactFSM(num_states=9" in _print_grammar_fsms(grammar)


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


def _fsm_snapshot(snapshot: str) -> str:
    return dedent(snapshot).lstrip("\n")


fsm_structure_cases = [
    pytest.param(
        'root ::= "a" [0-9] "b"',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=4, start=2, end=[3], edges=[
            0: [[0-9]->1]
            1: ['b'->3]
            2: ['a'->0]
            3: []
            ])
            """
        ),
        id="literal-character-class-literal",
    ),
    pytest.param(
        'root ::= "hello" [a-zA-Z_] [0-9]* "world"',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=12, start=2, end=[11], edges=[
            0: [[A-Z]->1, '_'->1, [a-z]->1]
            1: [[0-9]->1, 'w'->7]
            2: ['h'->3]
            3: ['e'->4]
            4: ['l'->5]
            5: ['l'->6]
            6: ['o'->0]
            7: ['o'->8]
            8: ['r'->9]
            9: ['l'->10]
            10: ['d'->11]
            11: []
            ])
            """
        ),
        id="mixed-character-classes",
    ),
    pytest.param(
        'root ::= "<" [a-c]* ">"',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=3, start=1, end=[2], edges=[
            0: ['>'->2, [a-c]->0]
            1: ['<'->0]
            2: []
            ])
            """
        ),
        id="starred-character-class",
    ),
    pytest.param(
        'root ::= "x" [^0-9] "y" [^a-z]* "z"',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=11, start=7, end=[10], edges=[
            0: [[\x80-\xbf]->2]
            1: [[\x80-\xbf]->3]
            2: [[\x80-\xbf]->5]
            3: [[\x80-\xbf]->6]
            4: [[\0-/]->5, [:-\x7f]->5, [\xc0-\xdf]->2, [\xe0-\xef]->0, [\xf0-\xf7]->8]
            5: ['y'->6]
            6: [[\0-`]->6, 'z'->10, [{-\x7f]->6, [\xc0-\xdf]->3, [\xe0-\xef]->1, [\xf0-\xf7]->9]
            7: ['x'->4]
            8: [[\x80-\xbf]->0]
            9: [[\x80-\xbf]->1]
            10: []
            ])
            """
        ),
        id="negated-character-classes",
    ),
    pytest.param(
        'root ::= "(" inner ")" inner\ninner ::= [0-9] [0-9]',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=8, start=3, end=[4], edges=[
            0: [Rule(1)->1]
            1: [')'->2]
            2: [Rule(1)->4]
            3: ['('->0]
            4: []
            ])
            Rule 1: inner, FSM: CompactFSM(num_states=8, start=6, end=[7], edges=[
            5: [[0-9]->7]
            6: [[0-9]->5]
            7: []
            ])
            """
        ),
        id="repeated-rule-reference",
    ),
    pytest.param(
        'root ::= "a" item{2,5} "b"\nitem ::= [0-9]',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=7, start=1, end=[4], edges=[
            0: ['b'->4]
            1: ['a'->2]
            2: [Eps->3]
            3: [Repeat(rule=1, min=2, max=5)->0]
            4: []
            ])
            Rule 1: item, FSM: CompactFSM(num_states=7, start=5, end=[6], edges=[
            5: [[0-9]->6]
            6: []
            ])
            """
        ),
        id="bounded-rule-repetition",
    ),
    pytest.param(
        'root ::= item{3,}\nitem ::= "ab" [xy]',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=6, start=0, end=[1], edges=[
            0: [Repeat(rule=1, min=3, max=-1)->1]
            1: []
            ])
            Rule 1: item, FSM: CompactFSM(num_states=6, start=3, end=[5], edges=[
            2: ['x'->5, 'y'->5]
            3: ['a'->4]
            4: ['b'->2]
            5: []
            ])
            """
        ),
        id="unbounded-rule-repetition",
    ),
    pytest.param(
        'root ::= "ab" [0-9] | "cd" sub | sub sub\nsub ::= [a-f] "q"',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=10, start=2, end=[1], edges=[
            0: [Rule(1)->1]
            1: []
            2: ['a'->5, [a-f]->4, 'c'->6]
            3: [[0-9]->1]
            4: ['q'->0]
            5: ['b'->3]
            6: ['d'->0]
            ])
            Rule 1: sub, FSM: CompactFSM(num_states=10, start=8, end=[9], edges=[
            7: ['q'->9]
            8: [[a-f]->7]
            9: []
            ])
            """
        ),
        id="choices-with-rule-references",
    ),
    pytest.param(
        'root ::= "abcdefghijklmnopqrstuvwxyz" [0-9] "ABCDEFGHIJKLMNOPQRSTUVWXYZ"',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=54, start=2, end=[53], edges=[
            0: [[0-9]->1]
            1: ['A'->28]
            2: ['a'->3]
            3: ['b'->4]
            4: ['c'->5]
            5: ['d'->6]
            6: ['e'->7]
            7: ['f'->8]
            8: ['g'->9]
            9: ['h'->10]
            10: ['i'->11]
            11: ['j'->12]
            12: ['k'->13]
            13: ['l'->14]
            14: ['m'->15]
            15: ['n'->16]
            16: ['o'->17]
            17: ['p'->18]
            18: ['q'->19]
            19: ['r'->20]
            20: ['s'->21]
            21: ['t'->22]
            22: ['u'->23]
            23: ['v'->24]
            24: ['w'->25]
            25: ['x'->26]
            26: ['y'->27]
            27: ['z'->0]
            28: ['B'->29]
            29: ['C'->30]
            30: ['D'->31]
            31: ['E'->32]
            32: ['F'->33]
            33: ['G'->34]
            34: ['H'->35]
            35: ['I'->36]
            36: ['J'->37]
            37: ['K'->38]
            38: ['L'->39]
            39: ['M'->40]
            40: ['N'->41]
            41: ['O'->42]
            42: ['P'->43]
            43: ['Q'->44]
            44: ['R'->45]
            45: ['S'->46]
            46: ['T'->47]
            47: ['U'->48]
            48: ['V'->49]
            49: ['W'->50]
            50: ['X'->51]
            51: ['Y'->52]
            52: ['Z'->53]
            53: []
            ])
            """
        ),
        id="long-byte-strings",
    ),
    pytest.param(
        'root ::= "中文" [\\u4e00-\\u9fff] "端"',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=14, start=3, end=[13], edges=[
            0: [[\x80-\xbf]->2]
            1: ['\xe4'->9, [\xe5-\xe9]->10]
            2: ['\xe7'->11]
            3: ['\xe4'->4]
            4: ['\xb8'->5]
            5: ['\xad'->6]
            6: ['\xe6'->7]
            7: ['\x96'->8]
            8: ['\x87'->1]
            9: [[\xb8-\xbf]->0]
            10: [[\x80-\xbf]->0]
            11: ['\xab'->12]
            12: ['\xaf'->13]
            13: []
            ])
            """
        ),
        id="utf8-byte-strings",
    ),
    pytest.param(
        'root ::= "only"',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=5, start=0, end=[4], edges=[
            0: ['o'->1]
            1: ['n'->2]
            2: ['l'->3]
            3: ['y'->4]
            4: []
            ])
            """
        ),
        id="single-literal",
    ),
    pytest.param(
        "root ::= [0-9]",
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=2, start=0, end=[1], edges=[
            0: [[0-9]->1]
            1: []
            ])
            """
        ),
        id="single-character-class",
    ),
    pytest.param(
        'root ::= "" "a" ""',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=2, start=0, end=[1], edges=[
            0: ['a'->1]
            1: []
            ])
            """
        ),
        id="empty-elements",
    ),
    pytest.param(
        'root ::= a b c\na ::= "x" [0-9]*\nb ::= a "y" | [^xyz]\nc ::= b{1,3} "end"',
        _fsm_snapshot(
            r"""
            Rule 0: root, FSM: CompactFSM(num_states=15, start=2, end=[3], edges=[
            0: [Rule(1)->1, [0-9]->0]
            1: [Rule(2)->3]
            2: ['x'->0]
            3: []
            ])
            Rule 1: b, FSM: CompactFSM(num_states=15, start=7, end=[5], edges=[
            4: [[\x80-\xbf]->6]
            5: []
            6: [[\x80-\xbf]->5]
            7: [[\0-w]->5, 'x'->8, [{-\x7f]->5, [\xc0-\xdf]->6, [\xe0-\xef]->4, [\xf0-\xf7]->9]
            8: [[0-9]->8, 'y'->5]
            9: [[\x80-\xbf]->4]
            ])
            Rule 2: c, FSM: CompactFSM(num_states=15, start=11, end=[14], edges=[
            10: ['e'->12]
            11: [Repeat(rule=1, min=1, max=3)->10]
            12: ['n'->13]
            13: ['d'->14]
            14: []
            ])
            """
        ),
        id="nested-rule-references",
    ),
]


@pytest.fixture(scope="module")
def structure_compiler():
    vocabulary = [chr(character) for character in range(33, 127)] + ["中", "文", "端"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary)
    return xgr.GrammarCompiler(tokenizer_info, max_threads=1, cache_enabled=False)


@pytest.mark.parametrize("grammar_str, expected_fsm", fsm_structure_cases)
def test_compiled_ebnf_fsm_structure_stable(
    structure_compiler, grammar_str: str, expected_fsm: str
):
    """Detect changes to state numbers, edge order, endpoints, or complete-FSM layout."""
    compiled = structure_compiler.compile_grammar(grammar_str)
    printed_fsm = _print_grammar_fsms(compiled.grammar)
    assert printed_fsm == expected_fsm


if __name__ == "__main__":
    pytest.main(sys.argv)
