"""This test is adopted from test_builtin_grammar_json.py, but the grammar is parsed from
a unoptimized, non-simplified EBNF string. This is to test the robustness of the grammar matcher.
"""

import sys
from typing import List

import pytest

import xgrammar as xgr
from xgrammar.testing import _get_masked_tokens_from_bitmask, _is_grammar_accept_string


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
    assert not _is_grammar_accept_string(grammar, "ttag1abd")


def test_complex_rule():
    grammar_str = """root ::= TagDispatch(("tag1", rule1), ("tag2", rule2))
rule1 ::= "abcd" [p]*
rule2 ::= "efg" [t]*
"""

    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, "tag1abcd")
    assert _is_grammar_accept_string(grammar, "tag1abcdppppptag2efg")
    assert _is_grammar_accept_string(grammar, "tag2efgtttttag1abc")


def test_tag_dispatch_mask_generation_correctness():
    grammar_str = """root ::= TagDispatch(("tag1", rule1), ("tag2", rule2))
rule1 ::= "abc"
rule2 ::= "dg"
"""
    tokens = [
        # fmt: off
        "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2a", "2dgt",
        "2dgtag1a", "2dgtag1b", "tag1a", "tag1b", "c哈哈t", "q", "abcdef"
        # fmt: on
    ]
    input_str = "tag1abcqqtag2dgq"
    expected_accepted_tokens = [
        # fmt: off
        ['a', 'b', 'c', 'd', 'g', 't', '1', '2', '1a', '2d', '2a', '2dgt', '2dgtag1a', 'tag1a', 'c哈哈t', 'q', 'abcdef'],
        ['a', 'b', 'c', 'd', 'g', 't', '1', '2', '1a', '2d', '2a', '2dgt', '2dgtag1a', 'tag1a', 'c哈哈t', 'q', 'abcdef'],
        ['a', 'b', 'c', 'd', 'g', 't', '1', '2', '1a', '2d', '2a', '2dgt', '2dgtag1a', 'tag1a', 'c哈哈t', 'q', 'abcdef'],
        ['a', 'b', 'c', 'd', 'g', 't', '1', '2', '1a', '2d', '2dgt', '2dgtag1a', 'tag1a', 'c哈哈t', 'q', 'abcdef'],
        ['a', 'abcdef'],
        ['b'],
        ['c哈哈t', 'c'],
        ['a', 'b', 'c', 'd', 'g', 't', '1', '2', '1a', '2d', '2a', '2dgt', '2dgtag1a', 'tag1a', 'c哈哈t', 'q', 'abcdef'],
        ['a', 'b', 'c', 'd', 'g', 't', '1', '2', '1a', '2d', '2a', '2dgt', '2dgtag1a', 'tag1a', 'c哈哈t', 'q', 'abcdef'],
        ['a', 'b', 'c', 'd', 'g', 't', '1', '2', '1a', '2d', '2a', '2dgt', '2dgtag1a', 'tag1a', 'c哈哈t', 'q', 'abcdef'],
        ['a', 'b', 'c', 'd', 'g', 't', '1', '2', '1a', '2d', '2a', '2dgt', '2dgtag1a', 'tag1a', 'c哈哈t', 'q', 'abcdef'],
        ['a', 'b', 'c', 'd', 'g', 't', '1', '2', '1a', '2d', '2a', '2dgt', '2dgtag1a', 'tag1a', 'c哈哈t', 'q', 'abcdef'],
        ['a', 'b', 'c', 'd', 'g', 't', '1', '2', '1a', '2d', '2dgt', '2dgtag1a', 'tag1a', 'c哈哈t', 'q', 'abcdef'],
        ['d'],
        ['g'],
        ['a', 'b', 'c', 'd', 'g', 't', '1', '2', '1a', '2d', '2a', '2dgt', '2dgtag1a', 'tag1a', 'c哈哈t', 'q', 'abcdef'],
        ['a', 'b', 'c', 'd', 'g', 't', '1', '2', '1a', '2d', '2a', '2dgt', '2dgtag1a', 'tag1a', 'c哈哈t', 'q', 'abcdef']
        # fmt: on
    ]

    grammar = xgr.Grammar.from_ebnf(grammar_str)
    tokenizer_info = xgr.TokenizerInfo(tokens)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    mask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    # pad a dummy char to check the final bitmask after accepting the input string
    for i, c in enumerate(input_str + "0"):
        matcher.fill_next_token_bitmask(mask)
        rejected_indices = _get_masked_tokens_from_bitmask(mask, tokenizer_info.vocab_size)
        accepted_indices = list(set(range(tokenizer_info.vocab_size)) - set(rejected_indices))
        accepted_tokens = [tokens[id] for id in accepted_indices]
        if i < len(input_str):
            assert matcher._debug_accept_string(c)
        assert accepted_tokens == expected_accepted_tokens[i]


# def test_tag_dispatch_mask_generation_real():
#     grammar_str = """root ::= TagDispatch(("<function=func1>", rule1), ("<function=func2>", rule2))
# rule1 ::= "abc"
# rule2 ::= "def"
# """
#     grammar = xgr.Grammar.from_ebnf(grammar_str)
#     tokenizer_info = xgr.TokenizerInfo(tokens)
#     compiler = xgr.GrammarCompiler(tokenizer_info)
#     compiled_grammar = compiler.compile_grammar(grammar)
#     matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)

if __name__ == "__main__":
    pytest.main(sys.argv)
