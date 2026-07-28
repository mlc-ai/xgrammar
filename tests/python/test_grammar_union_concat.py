"""This test uses the optimized JSON grammar provided by the grammar library."""

import sys

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


def test_grammar_union():
    grammar1 = xgr.Grammar.from_ebnf(
        """root ::= r1 | r2
r1 ::= "true" | ""
r2 ::= "false" | ""
"""
    )

    grammar2 = xgr.Grammar.from_ebnf(
        """root ::= "abc" | r1
r1 ::= "true" | r1
"""
    )

    grammar3 = xgr.Grammar.from_ebnf(
        """root ::= r1 | r2 | r3
r1 ::= "true" | r3
r2 ::= "false" | r3
r3 ::= "abc" | ""
"""
    )

    expected = """root ::= ((root_1) | (root_2) | (root_3))
root_1 ::= ((r1) | (r2))
r1 ::= ("" | ("true"))
r2 ::= ("" | ("false"))
root_2 ::= (("abc") | (r1_1))
r1_1 ::= (("true") | (r1_1))
root_3 ::= ((r1_2) | (r2_1) | (r3))
r1_2 ::= (("true") | (r3))
r2_1 ::= (("false") | (r3))
r3 ::= ("" | ("abc"))
"""

    union_grammar = xgr.Grammar.union(grammar1, grammar2, grammar3)
    assert str(union_grammar) == expected


def test_grammar_concat():
    grammar1 = xgr.Grammar.from_ebnf(
        """root ::= r1 | r2
r1 ::= "true" | ""
r2 ::= "false" | ""
"""
    )

    grammar2 = xgr.Grammar.from_ebnf(
        """root ::= "abc" | r1
r1 ::= "true" | r1
"""
    )

    grammar3 = xgr.Grammar.from_ebnf(
        """root ::= r1 | r2 | r3
r1 ::= "true" | r3
r2 ::= "false" | r3
r3 ::= "abc" | ""
"""
    )

    expected = """root ::= ((root_1 root_2 root_3))
root_1 ::= ((r1) | (r2))
r1 ::= ("" | ("true"))
r2 ::= ("" | ("false"))
root_2 ::= (("abc") | (r1_1))
r1_1 ::= (("true") | (r1_1))
root_3 ::= ((r1_2) | (r2_1) | (r3))
r1_2 ::= (("true") | (r3))
r2_1 ::= (("false") | (r3))
r3 ::= ("" | ("abc"))
"""

    concat_grammar = xgr.Grammar.concat(grammar1, grammar2, grammar3)
    assert str(concat_grammar) == expected


def test_grammar_union_with_stag():
    start = "start"
    schema = {"type": "object", "properties": {"arg": {"type": "string"}}}
    end = "end"
    tag = xgr.StructuralTagItem(begin=start, schema=schema, end=end)
    triggers = [start]
    stag_grammar = xgr.Grammar.from_structural_tag([tag], triggers)
    start_grammar = xgr.Grammar.from_ebnf("root ::= [a-z] root | [a-z]")

    grammar_union = xgr.Grammar.union(stag_grammar, start_grammar)
    assert _is_grammar_accept_string(grammar_union, 'start{"arg": "x"}end')
    assert _is_grammar_accept_string(grammar_union, "abc")
    assert not _is_grammar_accept_string(grammar_union, 'start{"arg": "x"}endabc')

    grammar_concat = xgr.Grammar.concat(stag_grammar, start_grammar)
    assert _is_grammar_accept_string(grammar_concat, 'start{"arg": "x"}endabc')
    assert not _is_grammar_accept_string(grammar_concat, 'start{"arg": "x"}end')
    assert not _is_grammar_accept_string(grammar_concat, "abc")


def test_grammar_union_concat_compiled_semantics():
    """Check end-state mapping after union and concatenation are optimized into FSMs."""
    nullable = xgr.Grammar.from_ebnf('root ::= "a" | ""')
    alternatives = xgr.Grammar.from_ebnf('root ::= "bc" | "d"')
    repeated = xgr.Grammar.from_ebnf("root ::= [x-z]+")

    union = xgr.Grammar.union(nullable, alternatives, repeated)
    for input_str in ["", "a", "bc", "d", "x", "xyz"]:
        assert _is_grammar_accept_string(union, input_str)
    for input_str in ["ab", "bd", "1", "ax"]:
        assert not _is_grammar_accept_string(union, input_str)

    concatenated = xgr.Grammar.concat(nullable, alternatives, repeated)
    for input_str in ["bcx", "dxyz", "abcx", "adxyz"]:
        assert _is_grammar_accept_string(concatenated, input_str)
    for input_str in ["", "a", "bc", "x", "abdx", "abc"]:
        assert not _is_grammar_accept_string(concatenated, input_str)


if __name__ == "__main__":
    pytest.main(sys.argv)
