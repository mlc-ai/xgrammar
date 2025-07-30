"""Tests the macro features of the grammar parser."""

import sys
from typing import Optional

import pytest

import xgrammar as xgr
from xgrammar.testing import GrammarFunctor, _ebnf_to_grammar_no_normalization


def test_tag_dispatch():
    """Test TagDispatch functionality."""
    before = """root ::= TagDispatch(
    ("tag1", rule1),
    ("tag2", rule2),
    stop_eos = false,
    stop_str = ("abc", "def"),
    loop_after_dispatch = false
)
rule1 ::= "a"
rule2 ::= "b"
"""
    expected = """root ::= ((TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  stop_eos=false,
  stop_str=("abc", "def"),
  loop_after_dispatch=false
)))
rule1 ::= (("a"))
rule2 ::= (("b"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_tag_dispatch_default_parameters():
    """Test TagDispatch functionality."""
    before = """root ::= TagDispatch(("tag1", rule1), ("tag2", rule2))
rule1 ::= "a"
rule2 ::= "b"
"""
    expected = """root ::= ((TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)))
rule1 ::= (("a"))
rule2 ::= (("b"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_lookahead_assertion_analyzer_tag_dispatch():
    # tag dispatch disables lookahead assertion detection
    before = r"""root ::= TagDispatch(("tag1", rule1), ("tag2", rule2), ("tag3", rule3), ("tag4", rule4), ("tag5", rule5))
rule1 ::= "b"
rule2 ::= "c"
rule3 ::= "" | "d" rule3
rule4 ::= "" | "e" rule4 "f"
rule5 ::= "" | "g" rule5 "h"
"""
    expected = r"""root ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  ("tag3", rule3),
  ("tag4", rule4),
  ("tag5", rule5),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)
rule1 ::= (("b"))
rule2 ::= (("c"))
rule3 ::= ("" | ("d" rule3))
rule4 ::= ("" | ("e" rule4 "f"))
rule5 ::= ("" | ("g" rule5 "h"))
"""

    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.structure_normalizer(grammar)
    grammar = GrammarFunctor.byte_string_fuser(grammar)
    grammar = GrammarFunctor.lookahead_assertion_analyzer(grammar)
    after = str(grammar)
    assert after == expected


def test_tag_dispatch_end_to_end():
    before = """root ::= TagDispatch(("tag1", rule1), ("tag2", rule2), ("tag3", rule3))
rule1 ::= "a"
rule2 ::= "b"
rule3 ::= "c"
"""
    expected = """root ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  ("tag3", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)
rule1 ::= (("a"))
rule2 ::= (("b"))
rule3 ::= (("c"))
"""
    grammar = xgr.Grammar.from_ebnf(before)
    after = str(grammar)
    assert after == expected


def test_tag_dispatch_end_to_end_complex():
    before = """root ::= TagDispatch(("tag1", rule1), ("tag2", rule2), ("tag3", rule3))
rule1 ::= ("a" TagDispatch(("tag1", rule2), ("tag2", rule3)) | "zzz")
rule2 ::= TagDispatch(("tag1", rule2), ("tag2", rule3)) | TagDispatch(("tag3", rule2), ("tag4", rule3))
rule3 ::= "c"
"""
    expected = """root ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  ("tag3", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)
rule1 ::= (("a" rule1_1) | ("zzz"))
rule2 ::= ((rule2_1) | (rule2_2))
rule3 ::= (("c"))
rule1_1 ::= TagDispatch(
  ("tag1", rule2),
  ("tag2", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)
rule2_1 ::= TagDispatch(
  ("tag1", rule2),
  ("tag2", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)
rule2_2 ::= TagDispatch(
  ("tag3", rule2),
  ("tag4", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)
"""
    grammar = xgr.Grammar.from_ebnf(before)
    after = str(grammar)
    assert after == expected


def test_e2e_tag_dispatch_roundtrip():
    """Checks the printed result can be parsed, and the parsing-printing process is idempotent."""
    before = r"""root ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  ("tag3", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false
)
rule1 ::= (("a"))
rule2 ::= (("b"))
rule3 ::= (("c"))
"""
    grammar_1 = xgr.Grammar.from_ebnf(before)
    output_string_1 = str(grammar_1)
    grammar_2 = xgr.Grammar.from_ebnf(output_string_1)
    output_string_2 = str(grammar_2)
    assert before == output_string_1
    assert output_string_1 == output_string_2


ebnf_str__expected_error_regex__test_tag_dispatch_parser_errors = [
    (
        'root ::= TagDispatch(("", rule1))\nrule1 ::= "a"',
        "EBNF parser error at line 1, column 21: Tag must be a non-empty string literal",
    ),
    (
        'root ::= TagDispatch(("tag1", undefined_rule))',
        'EBNF parser error at line 1, column 21: Rule "undefined_rule" is not defined',
    ),
    (
        'root ::= TagDispatch("tag1", rule1)',
        "EBNF parser error at line 1, column 21: Each tag dispatch element must be a tuple",
    ),
    (
        'root ::= TagDispatch(("tag1" rule1))',
        "EBNF parser error at line 1, column 30: Expect , or \\) in tuple",
    ),
    (
        'root ::= TagDispatch(("tag1", rule1), stop_str=true)\nrule1 ::= "a"',
        "EBNF parser error at line 1, column 21: Stop strings must be a tuple",
    ),
    (
        'root ::= TagDispatch(("tag1", rule1), stop_eos=false)\nrule1 ::= "a"',
        "EBNF parser error at line 1, column 21: The TagDispatch must have stop_eos=true or stop_str is not empty",
    ),
]


@pytest.mark.parametrize(
    "ebnf_str, expected_error_regex",
    ebnf_str__expected_error_regex__test_tag_dispatch_parser_errors,
)
def test_tag_dispatch_parser_errors(ebnf_str: str, expected_error_regex: Optional[str]):
    with pytest.raises(RuntimeError, match=expected_error_regex):
        _ebnf_to_grammar_no_normalization(ebnf_str)


if __name__ == "__main__":
    pytest.main(sys.argv)
