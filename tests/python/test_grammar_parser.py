import sys
from typing import Optional

import pytest

import xgrammar as xgr
from xgrammar.testing import (
    GrammarFunctor,
    _ebnf_to_grammar_no_normalization,
    _get_allow_empty_rule_ids,
    _is_grammar_accept_string,
)


def test_basic_string_literal():
    """Test basic string literals in grammar rules."""
    before = """root ::= "hello"
"""
    expected = """root ::= (("hello"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_empty_string():
    """Test empty string literals."""
    before = """root ::= ""
"""
    expected = """root ::= ((""))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def _regex_matches_bytes(grammar: xgr.Grammar, value: bytes) -> bool:
    tokenizer_info = xgr.TokenizerInfo([value] if value else [], stop_token_ids=[])
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    return (not value or matcher.accept_token(0)) and matcher.is_terminated()


def test_regex_byte_mode_round_trip_and_matching():
    grammar = xgr.Grammar.from_ebnf(r'root ::= Regex("\\x80[^\\x00-\\x7F]", byte_mode=true)')
    assert "byte_mode=true" in str(grammar)
    restored_ebnf = xgr.Grammar.from_ebnf(str(grammar))
    restored_json = xgr.Grammar.deserialize_json(grammar.serialize_json())

    for candidate in [grammar, restored_ebnf, restored_json]:
        assert _regex_matches_bytes(candidate, b"\x80\xff")
        assert not _regex_matches_bytes(candidate, b"\x80a")
        assert not _regex_matches_bytes(candidate, b"\xc2\x80")


def test_regex_byte_mode_differs_from_default_and_preserves_flags():
    byte_grammar = xgr.Grammar.from_ebnf(
        r'root ::= Regex("[^\\x00-\\x7F]a.", flags="isu", byte_mode=true)'
    )
    unicode_grammar = xgr.Grammar.from_ebnf(r'root ::= Regex("[^\\x00-\\x7F]")')

    assert _regex_matches_bytes(byte_grammar, b"\x80A\n")
    assert not _regex_matches_bytes(byte_grammar, b"\xc2\x80A\n")
    assert _regex_matches_bytes(unicode_grammar, b"\xc2\x80")
    assert not _regex_matches_bytes(unicode_grammar, b"\x80")


@pytest.mark.parametrize("byte_first", [True, False])
def test_regex_byte_and_unicode_fsm_cache_entries_are_isolated(byte_first: bool):
    byte_rule = 'byte ::= Regex(".", flags="s", byte_mode=true)'
    unicode_rule = 'unicode ::= Regex(".", flags="s")'
    definitions = [byte_rule, unicode_rule] if byte_first else [unicode_rule, byte_rule]
    grammar = xgr.Grammar.from_ebnf('root ::= ("B" byte | "U" unicode)\n' + "\n".join(definitions))

    for value in [b"B\x80", b"Ba", b"Ua", b"U\xc3\xa9"]:
        assert _regex_matches_bytes(grammar, value), value
    for value in [b"B\xc3\xa9", b"U\x80", b"U\xc3"]:
        assert not _regex_matches_bytes(grammar, value), value


@pytest.mark.parametrize(
    "grammar, message",
    [
        ('root ::= Regex("a", byte_mode=1)', "byte_mode must be a boolean"),
        (
            'root ::= Regex("a", json_string=true, byte_mode=true)',
            "Regex does not support json_string together with byte_mode",
        ),
        ('root ::= Regex("a", encoding=true)', "does not support the named argument"),
    ],
)
def test_regex_byte_mode_errors(grammar: str, message: str):
    with pytest.raises(RuntimeError, match=message):
        xgr.Grammar.from_ebnf(grammar)


def test_character_class():
    """Test character class expressions."""
    before = """root ::= [a-z]
"""
    expected = """root ::= (([a-z]))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_negated_character_class():
    """Test negated character class expressions."""
    before = """root ::= [^a-z]
"""
    expected = """root ::= (([^a-z]))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_complex_character_class():
    """Test complex character class with multiple ranges and individual characters."""
    before = r"""root ::= [a-zA-Z0-9_-] [\r\n$\x10-o\]\--]
"""
    expected = r"""root ::= (([a-zA-Z0-9_\-] [\r\n$\x10-o\]\-\-]))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_sequence():
    """Test sequence of expressions."""
    before = """root ::= "a" "b" "c"
"""
    expected = """root ::= (("a" "b" "c"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_choice():
    """Test choice between expressions."""
    before = """root ::= "a" | "b" | "c"
"""
    expected = """root ::= (("a") | ("b") | ("c"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_grouping():
    """Test grouping with parentheses."""
    before = """root ::= ("a" "b") | ("c" "d")
"""
    expected = """root ::= (((("a" "b"))) | ((("c" "d"))))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_star_quantifier_simple():
    """Test star (*) quantifier."""
    before = """root ::= "a"*
"""
    expected = """root ::= ((root_1))
root_1 ::= ("" | ("a" root_1))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_plus_quantifier():
    """Test plus (+) quantifier."""
    before = """root ::= "a"+
"""
    expected = """root ::= ((root_1))
root_1 ::= (("a" root_1) | "a")
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_question_quantifier():
    """Test question (?) quantifier."""
    before = """root ::= "a"?
"""
    expected = """root ::= ((root_1))
root_1 ::= ("" | "a")
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_character_class_star():
    """Test star (*) quantifier with character class."""
    before = """root ::= [a-z]*
"""
    expected = """root ::= (([a-z]*))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_intersection():
    """Test '&' intersection: binds tighter than '|' and looser than sequence."""
    before = """root ::= ("a" | "b") & "b"
"""
    expected = """root ::= ((((("a") | ("b"))) & ("b")))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_complement():
    """Test prefix '~' complement of an element."""
    before = """root ::= ~"bad" "!"
"""
    expected = """root ::= ((~("bad") "!"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_intersection_complement_matching():
    """Test that '&' and '~' compile into FSMs with the expected languages."""
    grammar = xgr.Grammar.from_ebnf('root ::= (("a" | "b") & "b") ~"bad" "!"')
    for instance in ["b!", "bx!", "bbadx!", "b你好!"]:
        assert _is_grammar_accept_string(grammar, instance), instance
    for instance in ["", "a!", "bbad!", "b!extra"]:
        assert not _is_grammar_accept_string(grammar, instance), instance


def test_intersection_of_concatenations_matching():
    """Intersection operands whose concatenations produce epsilon edges in the product NFA."""
    # [b]+ "c" concatenates two automata with an epsilon edge in between; the product with
    # [bc]* must accept exactly b+c and in particular reject "c" (a start-state regression).
    grammar = xgr.Grammar.from_ebnf('root ::= [b]* "b" "c" & [bc]*')
    for instance in ["bc", "bbc", "bbbc"]:
        assert _is_grammar_accept_string(grammar, instance), instance
    for instance in ["", "c", "b", "cc", "bcb"]:
        assert not _is_grammar_accept_string(grammar, instance), instance

    # Both operands are concatenations of nullable pieces.
    grammar = xgr.Grammar.from_ebnf('root ::= [a]* [b]* & ("ab" | "a" | "b")')
    for instance in ["a", "b", "ab"]:
        assert _is_grammar_accept_string(grammar, instance), instance
    for instance in ["", "ba", "aab", "abb"]:
        assert not _is_grammar_accept_string(grammar, instance), instance

    # Chained intersection with three operands.
    grammar = xgr.Grammar.from_ebnf('root ::= [ab]* & [bc]* & ("b" | "bb")')
    for instance in ["b", "bb"]:
        assert _is_grammar_accept_string(grammar, instance), instance
    for instance in ["", "a", "c", "bbb"]:
        assert not _is_grammar_accept_string(grammar, instance), instance


def test_complement_of_intersection_matching():
    """A complement whose operand is an intersection determinizes the NFA product."""
    grammar = xgr.Grammar.from_ebnf("root ::= ~([ab]* & [bc]*)")
    # The operand's language is b*, so the complement accepts anything that is not b*.
    for instance in ["a", "c", "ab", "cb", "abc"]:
        assert _is_grammar_accept_string(grammar, instance), instance
    for instance in ["", "b", "bb", "bbb"]:
        assert not _is_grammar_accept_string(grammar, instance), instance


def test_intersection_operand_with_repetition_range_is_rejected():
    """String quantifiers inside '&'/'~' operands compile into repetition ranges, which cannot
    become a single leaf automaton and must raise a clear error."""
    tokenizer_info = xgr.TokenizerInfo([])
    compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)
    with pytest.raises(RuntimeError, match="repetition ranges"):
        compiler.compile_grammar(xgr.Grammar.from_ebnf('root ::= "a"* & [ab]*'))


def test_complement_with_quantifier():
    """Test that a quantifier applies to the complemented element."""
    grammar = xgr.Grammar.from_ebnf('root ::= ~("," | ";")*')
    # Each repetition matches one string that is neither "," nor ";"; the concatenations cover
    # every string, including ones containing commas, so only direct rejection is observable
    # for the single-repetition rule below.
    single = xgr.Grammar.from_ebnf('root ::= ~("," | ";")')
    for instance in ["", "a", "你", ",,"]:
        assert _is_grammar_accept_string(single, instance), instance
    for instance in [",", ";"]:
        assert not _is_grammar_accept_string(single, instance), instance
    assert _is_grammar_accept_string(grammar, "a,b")


def test_repetition_range_exact():
    """Test repetition range with exact count {n}."""
    before = """root ::= "a"{3}
"""
    expected = """root ::= ((root_1{3, 3}))
root_1 ::= "a"
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_repetition_range_min_max():
    """Test repetition range with min and max {n,m}."""
    before = """root ::= "a"{2,4}
"""
    expected = """root ::= ((root_1{2, 4}))
root_1 ::= "a"
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_repetition_range_min_only():
    """Test repetition range with only min {n,}."""
    before = """root ::= "a"{2,}
"""
    expected = """root ::= ((root_1{2, -1}))
root_1 ::= "a"
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_repetition_range_unbounded_roundtrip():
    """Printed {n, -1} can be re-parsed (str -> compile_grammar round-trip)."""
    before = """root ::= "a"{2,}
"""
    grammar_1 = xgr.Grammar.from_ebnf(before)
    output_1 = str(grammar_1)
    assert "{2, -1}" in output_1
    output_2 = str(xgr.Grammar.from_ebnf(output_1))
    assert output_1 == output_2


def test_repetition_range_unbounded_json_schema():
    """JSON schema minLength produces {n, -1} which round-trips through the parser."""
    import json

    schema = json.dumps({"type": "string", "minLength": 2})
    grammar_1 = xgr.Grammar.from_json_schema(schema)
    output_1 = str(grammar_1)
    assert "{2, -1}" in output_1
    output_2 = str(xgr.Grammar.from_ebnf(output_1))
    assert output_1 == output_2


def test_lookahead_assertion_simple():
    """Test lookahead assertion."""
    before = """root ::= "a" (="b")
"""
    expected = """root ::= (("a")) (=(("b")))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_complex_lookahead():
    """Test complex lookahead assertion."""
    before = """root ::= "a" (="b" "c" [0-9])
"""
    expected = """root ::= (("a")) (=(("b" "c" [0-9])))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_escape_sequences():
    """Test escape sequences in string literals."""
    before = r"""root ::= "\n\t\r\"\\"
"""
    expected = r"""root ::= (("\n\t\r\"\\"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_unicode_escape():
    """Test Unicode escape sequences."""
    before = r"""root ::= "\u0041\u0042\u0043\u00A9\u2603"
"""
    expected = r"""root ::= (("ABC\xa9\u2603"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_forward_slash_escape_in_string_literal():
    # Regression: the EBNF lexer used to reject "\/" in string literals
    # with "Invalid escape sequence", because the C-style escape table did
    # not include "/". JSON allows "\/" as an alias for "/", so xgrammar
    # should accept it consistently.
    before = r"""root ::= "a\/b"
"""
    expected = r"""root ::= (("a/b"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    assert str(grammar) == expected


def test_complex_grammar():
    """Test a more complex grammar with multiple features."""
    before = """root ::= expr
expr ::= term ("+" term | "-" term)*
term ::= factor ("*" factor | "/" factor)*
factor ::= number | "(" expr ")"
number ::= [0-9]+ ("." [0-9]+)?
"""
    expected = """root ::= ((expr))
expr ::= ((term expr_1))
term ::= ((factor term_1))
factor ::= ((number) | ("(" expr ")"))
number ::= ((number_1 number_3))
expr_1 ::= ("" | ((("+" term) | ("-" term)) expr_1))
term_1 ::= ("" | ((("*" factor) | ("/" factor)) term_1))
number_1 ::= (([0-9] number_1) | [0-9])
number_2 ::= (([0-9] number_2) | [0-9])
number_3 ::= ("" | (("." number_2)))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_nested_quantifiers():
    """Test nested quantifiers in expressions."""
    before = """root ::= ("a"*)+
"""
    expected = """root ::= ((root_2))
root_1 ::= ("" | ("a" root_1))
root_2 ::= ((((root_1)) root_2) | ((root_1)))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_combined_features():
    """Test combination of various grammar features."""
    before = """root ::= "start" (rule1 | rule2)+ "end"
rule1 ::= [a-z]{1,3} (=":")
rule2 ::= [0-9]+ "." [0-9]*
"""
    expected = """root ::= (("start" root_1 "end"))
rule1 ::= ((rule1_1{1, 3})) (=((":")))
rule2 ::= ((rule2_1 "." [0-9]*))
root_1 ::= ((((rule1) | (rule2)) root_1) | ((rule1) | (rule2)))
rule1_1 ::= [a-z]
rule2_1 ::= (([0-9] rule2_1) | [0-9])
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_bnf_comment():
    before = """# top comment
root ::= a b # inline comment
a ::= "a"
b ::= "b"
# bottom comment
"""
    expected = """root ::= ((a b))
a ::= (("a"))
b ::= (("b"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_star_quantifier():
    before = """root ::= b c d
b ::= [b]*
c ::= "b"*
d ::= ([b] [c] [d] | ([p] [q]))*
e ::= [e]* [f]* | [g]*
"""

    expected = """root ::= ((b c d))
b ::= (([b]*))
c ::= ((c_1))
d ::= ((d_1))
e ::= (([e]* [f]*) | ([g]*))
c_1 ::= ("" | ("b" c_1))
d_1 ::= ("" | (d_1_1 d_1))
d_1_1 ::= (("b" "c" "d") | ("p" "q"))
"""

    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.structure_normalizer(grammar)
    after = str(grammar)
    assert after == expected

    # Here rule1 can be empty
    before = """root ::= [a]* [b]* rule1
rule1 ::= [abc]* [def]*
"""
    expected = """root ::= (([a]* [b]* rule1))
rule1 ::= (([abc]* [def]*))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.structure_normalizer(grammar)
    after = str(grammar)
    assert after == expected


def test_repetition_range():
    before = """root ::= a b c d e f g
a ::= [a]{1,2}
b ::= (a | "b"){1, 5}
c ::= "c" {0 , 2}
d ::= "d" {0,}
e ::= "e" {2, }
f ::= "f" {3}
g ::= "g" {0}
"""

    expected = """root ::= ((a b c d e f g))
a ::= ((a_1{1, 2}))
b ::= ((b_1{1, 5}))
c ::= ((c_1{0, 2}))
d ::= ((d_1{0, -1}))
e ::= ((e_1{2, -1}))
f ::= ((f_1{3, 3}))
g ::= ((g_1{0, 0}))
a_1 ::= (("a"))
b_1 ::= ((a) | ("b"))
c_1 ::= (("c"))
d_1 ::= (("d"))
e_1 ::= (("e"))
f_1 ::= (("f"))
g_1 ::= (("g"))
"""

    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.structure_normalizer(grammar)
    after = str(grammar)
    assert after == expected


def test_lookahead_assertion_with_normalizer():
    before = """root ::= ((b c d))
b ::= (("abc" [a-z])) (=("abc"))
c ::= (("a") | ("b")) (=[a-z] "b")
d ::= (("ac") | ("b" d_choice)) (="abc")
d_choice ::= (("e") | ("d"))
"""
    expected = """root ::= ((b c d))
b ::= (("abc" [a-z])) (=("abc"))
c ::= (("a") | ("b")) (=([a-z] "b"))
d ::= (("ac") | ("b" d_choice)) (=("abc"))
d_choice ::= (("e") | ("d"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.structure_normalizer(grammar)
    after = str(grammar)
    assert after == expected


def test_char():
    before = r"""root ::= [a-z] [A-z] "\u0234" "\U00000345\xff" [-A-Z] [--] [^a] rest
rest ::= [a-zA-Z0-9-] [\u0234-\U00000345] [测-试] [\--\]]  rest1
rest1 ::= "\?\"\'测试あc" "👀" "" [a-a] [b-b]
"""
    expected = r"""root ::= (([a-z] [A-z] "\u0234" "\u0345\xff" [\-A-Z] [\-\-] [^a] rest))
rest ::= (([a-zA-Z0-9\-] [\u0234-\u0345] [\u6d4b-\u8bd5] [\--\]] rest1))
rest1 ::= (("\?\"\'\u6d4b\u8bd5\u3042c" "\U0001f440" "a" "b"))
"""
    # Disable unwrap_nesting_rules to expose the result before unwrapping.
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.structure_normalizer(grammar)
    after = str(grammar)
    assert after == expected


def test_space():
    before = """

root::="a"  "b" ("c""d"
"e") |

"f" | "g"
"""
    expected = """root ::= (("a" "b" "c" "d" "e") | ("f") | ("g"))
"""
    grammar = xgr.Grammar.from_ebnf(before)
    after = str(grammar)
    assert after == expected


def test_nest():
    before = """root::= "a" ("b" | "c" "d") | (("e" "f"))
"""
    expected = """root ::= (("a" root_1) | ("e" "f"))
root_1 ::= (("b") | ("c" "d"))
"""
    grammar = xgr.Grammar.from_ebnf(before)
    after = str(grammar)
    assert after == expected


def test_empty_parentheses():
    before = """root ::= "a" ( ) "b"
"""
    expected = """root ::= (("a" "b"))
"""
    grammar = xgr.Grammar.from_ebnf(before)
    after = str(grammar)
    assert after == expected

    before = """root ::= "a" rule1
rule1 ::= ( )
"""
    expected = """root ::= (("a" rule1))
rule1 ::= ("")
"""
    grammar = xgr.Grammar.from_ebnf(before)
    after = str(grammar)
    assert after == expected


def test_lookahead_assertion_analyzer():
    before = r"""root ::= "a" rule1 "b" rule3 rule5 rule2
rule1 ::= "b"
rule2 ::= "c"
rule3 ::= "" | "d" rule3
rule4 ::= "" | "e" rule4 "f"
rule5 ::= "" | "g" rule5 "h"
"""
    expected = r"""root ::= (("a" rule1 "b" rule3 rule5 rule2))
rule1 ::= (("b")) (=("b" rule3 rule5 rule2))
rule2 ::= (("c"))
rule3 ::= (("") | ("d" rule3)) (=(rule5 rule2))
rule4 ::= (("") | ("e" rule4 "f")) (=("f"))
rule5 ::= (("") | ("g" rule5 "h"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.lookahead_assertion_analyzer(grammar)
    after = str(grammar)
    assert after == expected


def test_flatten():
    before = """root ::= or_test sequence_test nested_test empty_test
or_test ::= ([a] | "b") | "de" | "" | or_test | [^a-z]
sequence_test ::= [a] "a" ("b" ("c" | "d")) ("d" "e") sequence_test ""
nested_test ::= ("a" ("b" ("c" "d"))) | ("a" | ("b" | "c")) | nested_rest
nested_rest ::= ("a" | ("b" "c" | ("d" | "e" "f"))) | ((("g")))
empty_test ::= "d" | (("" | "" "") "" | "a" "") | ("" ("" | "")) "" ""
"""
    expected = """root ::= ((or_test sequence_test nested_test empty_test))
or_test ::= ("" | ("a") | ("b") | ("de") | (or_test) | ([^a-z]))
sequence_test ::= (("a" "a" "b" sequence_test_1 "d" "e" sequence_test))
nested_test ::= (("a" "b" "c" "d") | ("a") | ("b") | ("c") | (nested_rest))
nested_rest ::= (("a") | ("b" "c") | ("d") | ("e" "f") | ("g"))
empty_test ::= ("" | ("d") | ("a"))
sequence_test_1 ::= (("c") | ("d"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.structure_normalizer(grammar)
    after = str(grammar)
    assert after == expected


before__expected__test_rule_inliner = [
    (
        r"""root ::= rule1 | rule2
rule1 ::= "a" | "b"
rule2 ::= "b" | "c"
""",
        r"""root ::= (("a") | ("b") | ("b") | ("c"))
rule1 ::= (("a") | ("b"))
rule2 ::= (("b") | ("c"))
""",
    ),
    (
        r"""root ::= rule1 "a" [a-z]* | rule2 "b" "c"
rule1 ::= "a" [a-z]* | "b"
rule2 ::= "b" | "c" [b-c]
""",
        r"""root ::= (("a" [a-z]* "a" [a-z]*) | ("b" "a" [a-z]*) | ("b" "b" "c") | ("c" [b-c] "b" "c"))
rule1 ::= (("a" [a-z]*) | ("b"))
rule2 ::= (("b") | ("c" [b-c]))
""",
    ),
    (
        r"""root ::= rule1 (rule2 "y")
rule1 ::= "a" | "b"
rule2 ::= "c" | "d"
""",
        r"""root ::= (("a" (("c" "y") | ("d" "y"))) | ("b" (("c" "y") | ("d" "y"))))
rule1 ::= (("a") | ("b"))
rule2 ::= (("c") | ("d"))
""",
    ),
]


@pytest.mark.parametrize("before, expected", before__expected__test_rule_inliner)
def test_rule_inliner(before: str, expected: str):
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.rule_inliner(grammar)
    after = str(grammar)
    assert after == expected


before__expected__test_rule_inliner_no_rewrite = [
    # A rule with an empty-string choice is not inlined.
    (
        r"""root ::= rule1 "x"
rule1 ::= "a" | ""
""",
        r"""root ::= ((rule1 "x"))
rule1 ::= ("" | ("a"))
""",
    ),
    # A rule whose body contains a rule reference is not inlined.
    (
        r"""root ::= rule1 "x"
rule1 ::= "a" rule2 | "b"
rule2 ::= "c"
""",
        r"""root ::= ((rule1 "x"))
rule1 ::= (("a" rule2) | ("b"))
rule2 ::= (("c"))
""",
    ),
    # Only a rule reference at the start of a sequence triggers inlining.
    (
        r"""root ::= "x" rule1
rule1 ::= "a" | "b"
""",
        r"""root ::= (("x" rule1))
rule1 ::= (("a") | ("b"))
""",
    ),
]


@pytest.mark.parametrize("before, expected", before__expected__test_rule_inliner_no_rewrite)
def test_rule_inliner_no_rewrite(before: str, expected: str):
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.structure_normalizer(grammar)
    grammar = GrammarFunctor.rule_inliner(grammar)
    after = str(grammar)
    assert after == expected


def test_rule_inliner_preserves_lookahead_assertion():
    """When the inliner rewrites a rule body, the rule's lookahead assertion must be kept."""
    before = r"""root ::= rule2
rule2 ::= rule1 "x" (= "q" "r")
rule1 ::= "a" | "b"
"""
    expected = r"""root ::= ((rule2))
rule2 ::= (("a" "x") | ("b" "x")) (=(("q" "r")))
rule1 ::= (("a") | ("b"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.rule_inliner(grammar)
    assert str(grammar) == expected


before__expected__test_byte_string_fuser = [
    # A whole run of adjacent byte strings is fused into one.
    (
        r"""root ::= "a" "b" "c"
""",
        r"""root ::= (("abc"))
""",
    ),
    # Runs are split by non-byte-string elements; fusing also applies inside choices.
    (
        r"""root ::= "a" "b" [0-9] "c" "d" rule1
rule1 ::= "x" "y" | [a-z]
""",
        r"""root ::= (("ab" [0-9] "cd" rule1))
rule1 ::= (("xy") | ([a-z]))
""",
    ),
    # Nothing to fuse: the grammar stays unchanged.
    (
        r"""root ::= "a" [0-9] "b" | rule1
rule1 ::= [x]
""",
        r"""root ::= (("a" [0-9] "b") | (rule1))
rule1 ::= (([x]))
""",
    ),
]


@pytest.mark.parametrize("before, expected", before__expected__test_byte_string_fuser)
def test_byte_string_fuser(before: str, expected: str):
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.byte_string_fuser(grammar)
    assert str(grammar) == expected


def test_byte_string_fuser_lookahead_assertion():
    """Byte strings inside lookahead assertions are fused as well."""
    before = r"""root ::= "x" rule1
rule1 ::= "a" "b" (= "c" "d")
"""
    expected = r"""root ::= (("x" rule1))
rule1 ::= (("ab")) (=(("cd")))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.byte_string_fuser(grammar)
    assert str(grammar) == expected


def test_byte_string_fuser_removes_empty_byte_strings():
    structural_tag = {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<t>"],
            "tags": [
                {
                    "type": "tag",
                    "begin": "<t>",
                    "content": {"type": "const_string", "value": ""},
                    "end": "",
                }
            ],
        },
    }
    grammar = xgr.Grammar.from_structural_tag(structural_tag)
    fused = GrammarFunctor.byte_string_fuser(grammar)
    assert "triggered_tags_group ::= ((const_string))\n" in str(fused)

    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo([])).compile_grammar(grammar)
    assert _get_allow_empty_rule_ids(compiled) == [0, 1, 2, 3]


def test_optimizer_passes_do_not_mutate_input():
    """The passes rewrite a copy: the caller's grammar object must stay unchanged."""
    fuser_before = r"""root ::= "a" "b" rule1
rule1 ::= "c" | "d"
"""
    grammar = _ebnf_to_grammar_no_normalization(fuser_before)
    before_str = str(grammar)
    fused = GrammarFunctor.byte_string_fuser(grammar)
    assert str(grammar) == before_str
    assert str(fused) != before_str

    inliner_before = r"""root ::= rule1 "x"
rule1 ::= "a" | "b"
"""
    grammar = _ebnf_to_grammar_no_normalization(inliner_before)
    before_str = str(grammar)
    inlined = GrammarFunctor.rule_inliner(grammar)
    assert str(grammar) == before_str
    assert str(inlined) != before_str

    optimizer_before = r"""root ::= rule1 "x" | "y"
rule1 ::= "a" | "b"
"""
    grammar = xgr.Grammar.from_ebnf(optimizer_before)
    before_str = str(grammar)
    optimized = GrammarFunctor.grammar_optimizer(grammar)
    assert str(grammar) == before_str
    assert str(optimized) == 'root ::= (("a" "x") | ("b" "x") | ("y"))\n'

    # Compiling must not mutate the caller's grammar either.
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo([]))
    compiler.compile_grammar(grammar)
    assert str(grammar) == before_str


before__expected__test_dead_code_eliminator = [
    # Test basic dead code elimination
    (
        r"""root ::= rule1 | rule2
rule1 ::= "a" | "b"
rule2 ::= "b" | "c"
unused ::= "x" | "y"
""",
        r"""root ::= ((rule1) | (rule2))
rule1 ::= (("a") | ("b"))
rule2 ::= (("b") | ("c"))
""",
    ),
    # Test recursive rule references
    (
        r"""root ::= rule1 | rule2
unused1 ::= unused2 | "x"
unused2 ::= unused1 | "y"
rule1 ::= "a" rule2 | "b"
rule2 ::= "c" rule1 | "d"
""",
        r"""root ::= ((rule1) | (rule2))
rule1 ::= (("a" rule2) | ("b"))
rule2 ::= (("c" rule1) | ("d"))
""",
    ),
    # Test complex nested rules with unused branches
    (
        r"""root ::= rule1 "x" | rule2
rule1 ::= "a" rule3 | "b"
rule2 ::= "c" | "d" rule4
rule3 ::= "e" | "f"
rule4 ::= "g" | "h"
unused1 ::= "i" unused2
unused2 ::= "j" unused3
unused3 ::= "k" | "l"
""",
        r"""root ::= ((rule1 "x") | (rule2))
rule1 ::= (("a" rule3) | ("b"))
rule2 ::= (("c") | ("d" rule4))
rule3 ::= (("e") | ("f"))
rule4 ::= (("g") | ("h"))
""",
    ),
]


@pytest.mark.parametrize("before, expected", before__expected__test_dead_code_eliminator)
def test_dead_code_eliminator(before: str, expected: str):
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = xgr.testing.GrammarFunctor.dead_code_eliminator(grammar)
    assert str(after) == expected


def test_e2e_json_grammar():
    before = r"""root ::= (
    "{" [ \n\t]* members_and_embrace |
    "[" [ \n\t]* elements_or_embrace
)
value_non_str ::= (
    "{" [ \n\t]* members_and_embrace |
    "[" [ \n\t]* elements_or_embrace |
    "0" fraction exponent |
    [1-9] [0-9]* fraction exponent |
    "-" [0-9] fraction exponent |
    "-" [1-9] [0-9]* fraction exponent |
    "true" |
    "false" |
    "null"
) (= [ \n\t,}\]])
members_and_embrace ::= ("\"" characters_and_colon [ \n\t]* members_suffix | "}") (= [ \n\t,}\]])
members_suffix ::= (
    value_non_str [ \n\t]* member_suffix_suffix |
    "\"" characters_and_embrace |
    "\"" characters_and_comma [ \n\t]* "\"" characters_and_colon [ \n\t]* members_suffix
) (= [ \n\t,}\]])
member_suffix_suffix ::= (
    "}" |
    "," [ \n\t]* "\"" characters_and_colon [ \n\t]* members_suffix
) (= [ \n\t,}\]])
elements_or_embrace ::= (
    "{" [ \n\t]* members_and_embrace elements_rest [ \n\t]* "]" |
    "[" [ \n\t]* elements_or_embrace elements_rest [ \n\t]* "]" |
    "\"" characters_item elements_rest [ \n\t]* "]" |
    "0" fraction exponent elements_rest [ \n\t]* "]" |
    [1-9] [0-9]* fraction exponent elements_rest [ \n\t]* "]" |
    "-" "0" fraction exponent elements_rest [ \n\t]* "]" |
    "-" [1-9] [0-9]* fraction exponent elements_rest [ \n\t]* "]" |
    "true" elements_rest [ \n\t]* "]" |
    "false" elements_rest [ \n\t]* "]" |
    "null" elements_rest [ \n\t]* "]" |
    "]"
)
elements ::= (
    "{" [ \n\t]* members_and_embrace elements_rest |
    "[" [ \n\t]* elements_or_embrace elements_rest |
    "\"" characters_item elements_rest |
    "0" fraction exponent elements_rest |
    [1-9] [0-9]* fraction exponent elements_rest |
    "-" [0-9] fraction exponent elements_rest |
    "-" [1-9] [0-9]* fraction exponent elements_rest |
    "true" elements_rest |
    "false" elements_rest |
    "null" elements_rest
)
elements_rest ::= (
    "" |
    [ \n\t]* "," [ \n\t]* elements
)
characters_and_colon ::= (
    "\"" [ \n\t]* ":" |
    [^"\\\x00-\x1F] characters_and_colon |
    "\\" escape characters_and_colon
) (=[ \n\t]* [\"{[0-9tfn-])
characters_and_comma ::= (
    "\"" [ \n\t]* "," |
    [^"\\\x00-\x1F] characters_and_comma |
    "\\" escape characters_and_comma
) (=[ \n\t]* "\"")
characters_and_embrace ::= (
    "\"" [ \n\t]* "}" |
    [^"\\\x00-\x1F] characters_and_embrace |
    "\\" escape characters_and_embrace
) (=[ \n\t]* [},])
characters_item ::= (
    "\"" |
    [^"\\\x00-\x1F] characters_item |
    "\\" escape characters_item
) (= [ \n\t]* [,\]])
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
fraction ::= "" | "." [0-9] [0-9]*
exponent ::= "" |  "e" sign [0-9] [0-9]* | "E" sign [0-9] [0-9]*
sign ::= "" | "+" | "-"
"""

    expected = r"""root ::= (("{" [ \n\t]* members_and_embrace) | ("[" [ \n\t]* elements_or_embrace))
value_non_str ::= (("{" [ \n\t]* members_and_embrace) | ("[" [ \n\t]* elements_or_embrace) | ("0" fraction exponent) | ([1-9] [0-9]* fraction exponent) | ("-" [0-9] fraction exponent) | ("-" [1-9] [0-9]* fraction exponent) | ("true") | ("false") | ("null")) (=([ \n\t,}\]]))
members_and_embrace ::= (("\"" characters_and_colon [ \n\t]* members_suffix) | ("}")) (=([ \n\t,}\]]))
members_suffix ::= ((value_non_str [ \n\t]* member_suffix_suffix) | ("\"" characters_and_embrace) | ("\"" characters_and_comma [ \n\t]* "\"" characters_and_colon [ \n\t]* members_suffix)) (=([ \n\t,}\]]))
member_suffix_suffix ::= (("}") | ("," [ \n\t]* "\"" characters_and_colon [ \n\t]* members_suffix)) (=([ \n\t,}\]]))
elements_or_embrace ::= (("{" [ \n\t]* members_and_embrace elements_rest [ \n\t]* "]") | ("[" [ \n\t]* elements_or_embrace elements_rest [ \n\t]* "]") | ("\"" characters_item elements_rest [ \n\t]* "]") | ("0" fraction exponent elements_rest [ \n\t]* "]") | ([1-9] [0-9]* fraction exponent elements_rest [ \n\t]* "]") | ("-0" fraction exponent elements_rest [ \n\t]* "]") | ("-" [1-9] [0-9]* fraction exponent elements_rest [ \n\t]* "]") | ("true" elements_rest [ \n\t]* "]") | ("false" elements_rest [ \n\t]* "]") | ("null" elements_rest [ \n\t]* "]") | ("]"))
elements ::= (("{" [ \n\t]* members_and_embrace elements_rest) | ("[" [ \n\t]* elements_or_embrace elements_rest) | ("\"" characters_item elements_rest) | ("0" fraction exponent elements_rest) | ([1-9] [0-9]* fraction exponent elements_rest) | ("-" [0-9] fraction exponent elements_rest) | ("-" [1-9] [0-9]* fraction exponent elements_rest) | ("true" elements_rest) | ("false" elements_rest) | ("null" elements_rest))
elements_rest ::= ("" | ([ \n\t]* "," [ \n\t]* elements))
characters_and_colon ::= (("\"" [ \n\t]* ":") | ([^\"\\\0-\x1f] characters_and_colon) | ("\\" escape characters_and_colon)) (=([ \n\t]* [\"{[0-9tfn\-]))
characters_and_comma ::= (("\"" [ \n\t]* ",") | ([^\"\\\0-\x1f] characters_and_comma) | ("\\" escape characters_and_comma)) (=([ \n\t]* "\""))
characters_and_embrace ::= (("\"" [ \n\t]* "}") | ([^\"\\\0-\x1f] characters_and_embrace) | ("\\" escape characters_and_embrace)) (=([ \n\t]* [},]))
characters_item ::= (("\"") | ([^\"\\\0-\x1f] characters_item) | ("\\" escape characters_item)) (=([ \n\t]* [,\]]))
escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
fraction ::= ("" | ("." [0-9] [0-9]*))
exponent ::= ("" | ("e" sign [0-9] [0-9]*) | ("E" sign [0-9] [0-9]*))
sign ::= ("" | ("+") | ("-"))
"""

    grammar = xgr.Grammar.from_ebnf(before)
    grammar = GrammarFunctor.grammar_optimizer(grammar)
    after = str(grammar)
    assert after == expected


def test_e2e_to_string_roundtrip():
    """Checks the printed result can be parsed, and the parsing-printing process is idempotent."""
    before = r"""root ::= ((b c) | (b root))
b ::= ((b_1 d))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ("b" b_1)) (=(d))
c_1 ::= (([acep-z] c_1) | ([acep-z])) (=("d"))
d_1 ::= ("" | ("d"))
"""
    grammar_1 = xgr.Grammar.from_ebnf(before)
    output_string_1 = str(grammar_1)
    grammar_2 = xgr.Grammar.from_ebnf(output_string_1)
    output_string_2 = str(grammar_2)
    assert before == output_string_1
    assert output_string_1 == output_string_2


ebnf_str__expected_error_regex__test_lexer_parser_errors = [
    (r'root ::= "a" "', 'EBNF lexer error at line 1, column 15: Expect " in string literal'),
    (
        "root ::= [a\n]",
        "EBNF lexer error at line 1, column 12: Character class should not contain newline",
    ),
    (r'root ::= "\@"', "EBNF lexer error at line 1, column 11: Invalid escape sequence"),
    (r'root ::= "\uFF"', "EBNF lexer error at line 1, column 11: Invalid escape sequence"),
    (r'::= "a"', "EBNF lexer error at line 1, column 1: Assign should not be the first token"),
    (r"root ::= a b", 'EBNF parser error at line 1, column 10: Rule "a" is not defined'),
    (r'root ::= "a" |', "EBNF parser error at line 1, column 15: Expect element"),
    (
        r"root ::= [Z-A]",
        "EBNF parser error at line 1, column 11: Invalid character class: lower bound is larger "
        "than upper bound",
    ),
    (
        'root ::= "a"\nroot ::= "b"',
        'EBNF parser error at line 2, column 1: Rule "root" is defined multiple times',
    ),
    (
        r'a ::= "a"',
        'EBNF parser error at line 1, column 1: The root rule with name "root" is not found',
    ),
    (r'root ::= "a" (="a") (="b")', "EBNF parser error at line 1, column 21: Expect rule name"),
]


@pytest.mark.parametrize(
    "ebnf_str, expected_error_regex", ebnf_str__expected_error_regex__test_lexer_parser_errors
)
def test_lexer_parser_errors(ebnf_str: str, expected_error_regex: Optional[str]):
    with pytest.raises(RuntimeError, match=expected_error_regex):
        _ebnf_to_grammar_no_normalization(ebnf_str)


ebnf_str__expected_error_regex__test_end_to_end_errors = [
    (r'root ::= "a" (=("a" | "b"))', "Choices in lookahead assertion are not supported yet")
]


@pytest.mark.parametrize(
    "ebnf_str, expected_error_regex", ebnf_str__expected_error_regex__test_end_to_end_errors
)
def test_end_to_end_errors(ebnf_str: str, expected_error_regex: Optional[str]):
    with pytest.raises(RuntimeError, match=expected_error_regex):
        xgr.Grammar.from_ebnf(ebnf_str)


def test_error_consecutive_quantifiers():
    grammar_str = """root ::= "a"{1,3}{1,3}
"""
    with pytest.raises(
        RuntimeError, match="EBNF parser error at line 1, column 18: Expect element, but got {"
    ):
        xgr.Grammar.from_ebnf(grammar_str)

    grammar_str = """root ::= "a"++
"""
    with pytest.raises(
        RuntimeError, match="EBNF parser error at line 1, column 14: Expect element, but got +"
    ):
        xgr.Grammar.from_ebnf(grammar_str)

    grammar_str = """root ::= "a"??
"""
    with pytest.raises(
        RuntimeError, match="EBNF parser error at line 1, column 14: Expect element, but got ?"
    ):
        xgr.Grammar.from_ebnf(grammar_str)


def test_repetition_normalizer():
    """Test the repetition normalizer. If the context is nullable, then the min repetition time will be reduced to 0."""
    before = "root ::= ([0-9]*){200, 1000}"
    expected_grammar = r"""root ::= ((root_2))
root_repeat_1 ::= (([0-9]*)) (=([0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]*))
root_repeat_1_inner ::= ((root_repeat_1{0, 872})) (=([0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]*))
root_2 ::= ((root_repeat_1_inner [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]*))
"""
    grammar = xgr.Grammar.from_ebnf(before)
    print(grammar)
    grammar = GrammarFunctor.grammar_optimizer(grammar)
    assert expected_grammar == str(grammar)

    before = "root ::= ([0-9]){200, 1000}"
    expected_grammar = r"""root ::= ((root_2))
root_repeat_1 ::= (([0-9])) (=([0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9]))
root_repeat_1_inner ::= ((root_repeat_1{72, 872})) (=([0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9]))
root_2 ::= ((root_repeat_1_inner [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9]))
"""
    grammar = xgr.Grammar.from_ebnf(before)
    grammar = GrammarFunctor.grammar_optimizer(grammar)
    assert expected_grammar == str(grammar)


if __name__ == "__main__":
    pytest.main(sys.argv)
