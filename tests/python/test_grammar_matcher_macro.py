import sys

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
    assert not _is_grammar_accept_string(grammar, "tag1efg")


def test_no_loop_after_dispatch():
    grammar_str = """root ::= TagDispatch(("tag1", rule1), ("tag2", rule2), loop_after_dispatch=false)
rule1 ::= "abcd" [p]*
rule2 ::= "efg" [t]*
"""

    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, "tag1abcd")
    assert _is_grammar_accept_string(grammar, "tag2efgttt")
    assert not _is_grammar_accept_string(grammar, "tag1abcdppppptag2")
    assert not _is_grammar_accept_string(grammar, "tag2efgtag1")


def test_stop_str():
    grammar_str = """root ::= root1 "w"
root1 ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  stop_eos=false,
  stop_str=("tag3", "ll")
)
rule1 ::= "abcd" [p]*
rule2 ::= "efg" [t]*
"""

    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, "tag1abcdllw", debug_print=True)
    assert _is_grammar_accept_string(grammar, "tag1abcdtag3w")
    assert _is_grammar_accept_string(grammar, "tag1abcdqqqtag2efgtag3w")
    assert _is_grammar_accept_string(grammar, "tag1abcd", require_termination=False)
    assert _is_grammar_accept_string(grammar, "tag2efgttt", require_termination=False)
    assert not _is_grammar_accept_string(grammar, "tag1abcd")
    assert not _is_grammar_accept_string(grammar, "tag2efgttt")
    assert not _is_grammar_accept_string(grammar, "tag1abce")
    assert not _is_grammar_accept_string(grammar, "tag1abcdlltag3w", require_termination=False)


def test_stop_str_no_loop():
    grammar_str = """root ::= root1 "w"
root1 ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  stop_eos=false,
  stop_str=("tag3", "ll"),
  loop_after_dispatch=false
)
rule1 ::= "abcd" [p]*
rule2 ::= "efg" [t]*
"""

    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, "tag1abcdllw")
    assert _is_grammar_accept_string(grammar, "tag1abcdtag3w")
    assert _is_grammar_accept_string(grammar, "tag1abcd", require_termination=False)
    assert _is_grammar_accept_string(grammar, "tag2efgttt", require_termination=False)
    assert not _is_grammar_accept_string(grammar, "tag1abcdqqqtag2efgtag3w")
    assert not _is_grammar_accept_string(grammar, "tag1abcd")
    assert not _is_grammar_accept_string(grammar, "tag2efgttt")
    assert not _is_grammar_accept_string(grammar, "tag1abce")
    assert not _is_grammar_accept_string(grammar, "tag1abcdlltag3w", require_termination=False)


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
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1)
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
            assert matcher.accept_string(c)
        assert accepted_tokens == expected_accepted_tokens[i]


def test_regression_multiple_tag_dispatch():
    grammar_str = """root ::= root1 "w"
root1 ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false
)
rule1 ::= TagDispatch(
  ("tag1", rule2),
  ("tag2", rule3),
  stop_eos=false,
  stop_str=("tag3", "ll"),
  loop_after_dispatch=true
)
rule2 ::= "efg" [t]*
rule3 ::= "abcd" [p]*
"""
    assert _is_grammar_accept_string(grammar_str, "tag1tag1efgllw")
    assert _is_grammar_accept_string(grammar_str, "tag1tag2abcdtag3w")
    assert not _is_grammar_accept_string(grammar_str, "tag1Ktag2abcdtag3tag1")
    assert _is_grammar_accept_string(grammar_str, "tag1tag3w")
    assert not _is_grammar_accept_string(grammar_str, "tag1tag3tag2abcdll")


if __name__ == "__main__":
    pytest.main(sys.argv)
