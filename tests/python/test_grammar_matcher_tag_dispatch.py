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
    grammar_str = """root ::= TagDispatch(("a", rule1), ("b", rule2))
rule1 ::= "abcd"
rule2 ::= "efg"
"""

    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, "aabcd")
    assert _is_grammar_accept_string(grammar, "aabcdbefg")
    assert _is_grammar_accept_string(grammar, "aabcdqqqq")
    assert not _is_grammar_accept_string(grammar, "aabc")
    assert not _is_grammar_accept_string(grammar, "aabce")


if __name__ == "__main__":
    pytest.main(sys.argv)
