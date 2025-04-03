"""This test uses the optimized JSON grammar provided by the grammar library."""

import sys

import pytest

import xgrammar as xgr


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


if __name__ == "__main__":
    pytest.main(sys.argv)
