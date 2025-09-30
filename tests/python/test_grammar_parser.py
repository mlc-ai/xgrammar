import sys
from typing import Optional

import pytest

import xgrammar as xgr
from xgrammar.testing import GrammarFunctor, _ebnf_to_grammar_no_normalization


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


def test_repetition_range_exact():
    """Test repetition range with exact count {n}."""
    before = """root ::= "a"{3}
"""
    expected = """root ::= (((("a" "a" "a"))))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_repetition_range_min_max():
    """Test repetition range with min and max {n,m}."""
    before = """root ::= "a"{2,4}
"""
    expected = """root ::= (((("a" "a") | ("a" "a" "a") | ("a" "a" "a" "a"))))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


def test_repetition_range_min_only():
    """Test repetition range with only min {n,}."""
    before = """root ::= "a"{2,}
"""
    expected = """root ::= ((("a" "a" root_repeat_inf)))
root_repeat_inf ::= ("" | ("a" root_repeat_inf))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    after = str(grammar)
    assert after == expected


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
rule1 ::= (((([a-z]) | ([a-z] [a-z]) | ([a-z] [a-z] [a-z])))) (=((":")))
rule2 ::= ((rule2_1 "." [0-9]*))
root_1 ::= ((((rule1) | (rule2)) root_1) | ((rule1) | (rule2)))
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
a ::= (("a") | ("a" "a"))
b ::= ((a) | ("b") | (b_1 b_2) | (b_3 b_4 b_5) | (b_6 b_7 b_8 b_9) | (b_10 b_11 b_12 b_13 b_14))
c ::= ("" | ("c") | ("c" "c"))
d ::= ((d_repeat_inf))
e ::= (("e" "e" e_repeat_inf))
f ::= (("f" "f" "f"))
g ::= ("")
d_repeat_inf ::= ("" | ("d" d_repeat_inf))
e_repeat_inf ::= ("" | ("e" e_repeat_inf))
b_1 ::= ((a) | ("b"))
b_2 ::= ((a) | ("b"))
b_3 ::= ((a) | ("b"))
b_4 ::= ((a) | ("b"))
b_5 ::= ((a) | ("b"))
b_6 ::= ((a) | ("b"))
b_7 ::= ((a) | ("b"))
b_8 ::= ((a) | ("b"))
b_9 ::= ((a) | ("b"))
b_10 ::= ((a) | ("b"))
b_11 ::= ((a) | ("b"))
b_12 ::= ((a) | ("b"))
b_13 ::= ((a) | ("b"))
b_14 ::= ((a) | ("b"))
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
]


@pytest.mark.parametrize("before, expected", before__expected__test_rule_inliner)
def test_rule_inliner(before: str, expected: str):
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.rule_inliner(grammar)
    after = str(grammar)
    assert after == expected


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
    before = "root ::= ([0-9]*){100, 1000}"
    expected_grammar = r"""root ::= ((root_repeat_1{0, 985} [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]*))
root_repeat_1 ::= (([0-9]*)) (=([0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]* [0-9]*))
"""
    grammar = xgr.Grammar.from_ebnf(before)
    grammar = GrammarFunctor.grammar_optimizer(grammar)
    assert expected_grammar == str(grammar)

    before = "root ::= ([0-9]){100, 1000}"
    expected_grammar = r"""root ::= ((root_repeat_1{85, 985} [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9]))
root_repeat_1 ::= (([0-9])) (=([0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9] [0-9]))
"""
    grammar = xgr.Grammar.from_ebnf(before)
    grammar = GrammarFunctor.grammar_optimizer(grammar)
    assert expected_grammar == str(grammar)


if __name__ == "__main__":
    pytest.main(sys.argv)
