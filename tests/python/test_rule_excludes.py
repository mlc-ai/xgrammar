"""Tests for the excludes rule attribute: name[excludes=("a", "b")] ::= ... forbids the
substrings in the text matched by the rule, enforced at runtime by the parser."""

import sys

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

STRING_GRAMMAR = r"""
root ::= "\"" body "\"" tail
body[excludes=("<|open|>", "ab")] ::= [^"\\]* | [^"\\]* "\\" esc body
esc ::= ["\\/bfnrt] | "u" [0-9a-fA-F]{4}
tail ::= "<|open|>" | ""
"""


def test_attribute_parse_and_print_roundtrip():
    grammar = xgr.Grammar.from_ebnf(STRING_GRAMMAR)
    printed = str(grammar)
    assert 'body[excludes=("<|open|>", "ab")] ::=' in printed
    assert str(xgr.Grammar.from_ebnf(printed)) == printed


@pytest.mark.parametrize(
    "excludes, printed",
    [
        (r'"\""', r'excludes=("\"")'),
        (r'"\\"', r'excludes=("\\")'),
        (r'"\n", "\t"', r'excludes=("\t", "\n")'),
        (r'"你好", "\u00e9"', r'excludes=("\xe9", "\u4f60\u597d")'),
        (r'"b", "a", "b"', r'excludes=("a", "b")'),
    ],
)
def test_attribute_escapes_and_normalization(excludes, printed):
    grammar = xgr.Grammar.from_ebnf(f"root[excludes=({excludes})] ::= [a-z]*")
    assert printed in str(grammar)
    assert str(xgr.Grammar.from_ebnf(str(grammar))) == str(grammar)


def test_attribute_rejects_empty_string():
    with pytest.raises(RuntimeError):
        xgr.Grammar.from_ebnf('root[excludes=("a", "")] ::= [a-z]*')


def test_attribute_combines_with_other_attributes():
    grammar = xgr.Grammar.from_ebnf('root[max_chars=5, excludes=("ab")] ::= [a-z]*')
    assert 'root[max_chars=5, excludes=("ab")] ::=' in str(grammar)
    assert _is_grammar_accept_string(grammar, "axbxc")
    assert not _is_grammar_accept_string(grammar, "axbxcd")
    assert not _is_grammar_accept_string(grammar, "xab")


@pytest.mark.parametrize(
    "text, accepted",
    [
        ('"xy"', True),
        ('"xaby"', False),
        ('"axb"', True),
        ('"x<|open|>"', False),
        # The excluded substring continues outside the excluding rule: allowed.
        ('"x<|ope"<|open|>', True),
        # Escapes are matched as emitted text, so \u0061b does not contain "ab" ...
        ('"\\u0061b"', True),
        ('"a\\nb"', True),
        # ... but the escape bytes themselves are subject to the exclusion.
        ('"\\"ab"', False),
    ],
)
def test_string_exclusion_semantics(text, accepted):
    grammar = xgr.Grammar.from_ebnf(STRING_GRAMMAR)
    assert _is_grammar_accept_string(grammar, text) == accepted


def test_recursive_rule_detects_multi_byte_exclusion():
    # A right-recursive rule predicts a new occurrence of itself for every character; the
    # automaton state must carry over between them.
    grammar = xgr.Grammar.from_ebnf(
        r"""
root ::= "<" sub
sub[excludes=("abc")] ::= ">" | [a-z] sub
"""
    )
    assert _is_grammar_accept_string(grammar, "<xyz>")
    assert _is_grammar_accept_string(grammar, "<abxc>")
    assert not _is_grammar_accept_string(grammar, "<xabcx>")
    assert not _is_grammar_accept_string(grammar, "<abc>")


def test_exclusion_scope_resets_between_occurrences():
    grammar = xgr.Grammar.from_ebnf(
        r"""
root ::= item ("," item)*
item[excludes=("ab")] ::= [a-z]+
"""
    )
    assert _is_grammar_accept_string(grammar, "a,b")
    assert _is_grammar_accept_string(grammar, "xa,bx")
    assert not _is_grammar_accept_string(grammar, "xab,x")
    assert not _is_grammar_accept_string(grammar, "x,ab")


def test_separate_regions_have_separate_excludes():
    grammar = xgr.Grammar.from_ebnf(
        r"""
root ::= first "|" second
first[excludes=("aa")] ::= [a-z]*
second[excludes=("bb")] ::= [a-z]*
"""
    )
    assert _is_grammar_accept_string(grammar, "bb|aa")
    assert not _is_grammar_accept_string(grammar, "aa|bb")
    assert not _is_grammar_accept_string(grammar, "a|bb")
    assert not _is_grammar_accept_string(grammar, "aa|b")


def test_nested_rule_with_different_excludes_is_rejected():
    # The check runs when the grammar is optimized for matching, i.e. at compile time.
    grammar = xgr.Grammar.from_ebnf(
        r"""
root[excludes=("aa")] ::= inner*
inner[excludes=("bb")] ::= [a-z]
"""
    )
    with pytest.raises(RuntimeError, match="nested exclusions"):
        _is_grammar_accept_string(grammar, "ab")
    # The same excludes may be nested: the inner occurrence continues the outer automaton.
    grammar = xgr.Grammar.from_ebnf(
        r"""
root[excludes=("ab")] ::= inner*
inner[excludes=("ab")] ::= [a-z]
"""
    )
    assert _is_grammar_accept_string(grammar, "ba")
    assert not _is_grammar_accept_string(grammar, "ab")


def test_root_rule_excludes():
    grammar = xgr.Grammar.from_ebnf('root[excludes=("ab")] ::= [a-z]*')
    assert _is_grammar_accept_string(grammar, "ba")
    assert not _is_grammar_accept_string(grammar, "xabx")


def _check_mask_matches_accept_token(compiled, vocab, prefixes):
    """The bitmask after every prefix must equal accept_token over the whole vocabulary."""
    for prefix in prefixes:
        matcher = xgr.GrammarMatcher(compiled)
        for token in prefix:
            assert matcher.accept_token(vocab.index(token)), (prefix, token)
        bitmask = xgr.allocate_token_bitmask(1, len(vocab))
        matcher.fill_next_token_bitmask(bitmask)
        bits = int(bitmask[0, 0])
        for token_id, token in enumerate(vocab):
            trial = xgr.GrammarMatcher(compiled)
            for prev in prefix:
                trial.accept_token(vocab.index(prev))
            assert bool(bits >> token_id & 1) == trial.accept_token(token_id), (prefix, token)
        yield prefix, bits


def test_token_mask_matches_accept_token():
    grammar = xgr.Grammar.from_ebnf(STRING_GRAMMAR)
    vocab = [
        '"', "x", "a", "b", "ab", "xa", "by", "<|", "open|>", "<|open|>", '"<|open|>', "\\",
        "u0061", "n", "a\\", "\\u0061b", "[EOS]",
    ]  # fmt: skip
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[len(vocab) - 1])
    compiled = xgr.GrammarCompiler(info, cache_enabled=False).compile_grammar(grammar)
    results = dict(
        _check_mask_matches_accept_token(
            compiled, vocab, [('"',), ('"', "a"), ('"', "x", "<|"), ('"', "a", "\\"), ('"', "xa")]
        )
    )
    after_a = results[('"', "a")]
    assert not after_a >> vocab.index("b") & 1
    assert not after_a >> vocab.index("by") & 1
    assert after_a >> vocab.index('"<|open|>') & 1  # leaves the region before the marker
    after_open = results[('"', "x", "<|")]
    assert not after_open >> vocab.index("open|>") & 1
    assert after_open >> vocab.index("<|") & 1


def test_token_mask_keeps_tokens_that_leave_the_region_first():
    # "," is excluded inside the string, but a token that closes the string and then emits ","
    # is legal. The compiled mask marks it accepted through the string rule's lookahead, and the
    # exclusion filter must re-verify instead of dropping it.
    grammar = xgr.Grammar.from_ebnf(
        r"""
root ::= "[" str ("," str)* "]"
str ::= "\"" sub
sub[excludes=(",")] ::= "\"" | [^"\\] sub
"""
    )
    vocab = ['"', "x", ",", 'x",', '",', '"]', "[", "]", "x,", "[EOS]"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[len(vocab) - 1])
    compiled = xgr.GrammarCompiler(info, cache_enabled=False).compile_grammar(grammar)
    results = dict(_check_mask_matches_accept_token(compiled, vocab, [("[", '"'), ("[", '"', "x")]))
    bits = results[("[", '"', "x")]
    assert bits >> vocab.index('x",') & 1
    assert bits >> vocab.index('",') & 1
    assert not bits >> vocab.index(",") & 1
    assert not bits >> vocab.index("x,") & 1


def test_rollback_and_reset_restore_exclusion_state():
    grammar = xgr.Grammar.from_ebnf('root ::= "<" sub ">"\nsub[excludes=("ab")] ::= [a-z]*')
    vocab = ["<", ">", "a", "b", "[EOS]"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[4])
    compiled = xgr.GrammarCompiler(info, cache_enabled=False).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    assert matcher.accept_token(2)
    assert not matcher.accept_token(3)
    matcher.rollback(1)
    assert matcher.accept_token(3)
    assert matcher.accept_token(2)
    matcher.reset()
    assert matcher.accept_token(0)
    assert matcher.accept_token(3)
    assert matcher.accept_token(2)
    assert not matcher.accept_token(3)


def test_compiled_grammar_serialization_roundtrip():
    grammar = xgr.Grammar.from_ebnf(STRING_GRAMMAR)
    vocab = ['"', "x", "a", "b", "ab", "<|open|>", "[EOS]"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[len(vocab) - 1])
    compiled = xgr.GrammarCompiler(info, cache_enabled=False).compile_grammar(grammar)
    restored = xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)
    assert str(restored.grammar) == str(compiled.grammar)
    for compiled_grammar in (compiled, restored):
        matcher = xgr.GrammarMatcher(compiled_grammar)
        assert matcher.accept_string('"xa')
        bitmask = xgr.allocate_token_bitmask(1, len(vocab))
        matcher.fill_next_token_bitmask(bitmask)
        bits = int(bitmask[0, 0])
        assert not bits >> vocab.index("b") & 1
        assert not bits >> vocab.index("ab") & 1
        assert bits >> vocab.index("x") & 1
        assert not matcher.accept_token(vocab.index("b"))


def test_excluding_rule_is_not_inlined():
    grammar = xgr.Grammar.from_ebnf(
        r"""
root ::= "<" sub ">"
sub[excludes=("ab")] ::= [a-z]*
"""
    )
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo(["<", ">", "a", "b"])).compile_grammar(grammar)
    assert 'sub[excludes=("ab")] ::=' in str(compiled.grammar)
    assert not _is_grammar_accept_string(grammar, "<ab>")


def test_grammar_without_excludes_is_unchanged():
    ebnf = 'root ::= "<" [a-z]* ">"'
    grammar = xgr.Grammar.from_ebnf(ebnf)
    assert "excludes" not in str(grammar)
    serialized = grammar.serialize_json()
    assert '"exclusion_transitions":[]' in serialized
    assert '"rule_exclusion_start_states":[]' in serialized


if __name__ == "__main__":
    pytest.main(sys.argv)
