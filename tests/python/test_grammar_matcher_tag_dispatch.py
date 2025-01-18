"""This test is adopted from test_builtin_grammar_json.py, but the grammar is parsed from
a unoptimized, non-simplified EBNF string. This is to test the robustness of the grammar matcher.
"""

import sys
import time
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


def test_simple():
    grammar_str = """root ::= TagDispatch(("tag1", rule1), ("tag2", rule2))
rule1 ::= "abcd"
rule2 ::= "efg"
"""

    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, "tag1abcd")
    assert _is_grammar_accept_string(grammar, "tag1abcdtag2efg")
    assert _is_grammar_accept_string(grammar, "tag1abcdqqqqtag2efg")
    assert not _is_grammar_accept_string(grammar, "tag1abc")
    assert not _is_grammar_accept_string(grammar, "tag1abce")


def test_complex_rule():
    grammar_str = """root ::= TagDispatch(("tag1", rule1), ("tag2", rule2))
rule1 ::= "abcd" [p]*
rule2 ::= "efg" [t]*
"""

    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, "tag1abcd")
    assert _is_grammar_accept_string(grammar, "tag1abcdppppptag2efg")
    assert _is_grammar_accept_string(grammar, "tag2efgtttttag1abc")


if __name__ == "__main__":
    pytest.main(sys.argv)
