import sys

import pytest

import xgrammar as xgr
from xgrammar.testing import GrammarFunctor, _ebnf_to_grammar_no_normalization


def test_bnf_simple():
    before = """root ::= b c
b ::= "b"
c ::= "c"
"""
    expected = """root ::= ((b c))
b ::= (("b"))
c ::= (("c"))
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


def test_ebnf():
    before = """root ::= b c | b root
b ::= "ab"*
c ::= [acep-z]+
d ::= "d"?
"""
    expected = """root ::= ((b c) | (b root))
b ::= ((b_1))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ("ab" b_1))
c_1 ::= (([acep-z] c_1) | [acep-z])
d_1 ::= ("" | "d")
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
d_1 ::= ("" | (d_1_choice d_1))
d_1_choice ::= (("bcd") | ("pq"))
"""

    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.structure_normalizer(grammar)
    grammar = GrammarFunctor.byte_string_fuser(grammar)
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
    grammar = GrammarFunctor.byte_string_fuser(grammar)
    after = str(grammar)
    assert after == expected


def test_consecutive_quantifiers():
    grammar_str = """root ::= "a"{1,3}{1,3}
"""
    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 18: Expect element, but got character: {",
    ):
        xgr.Grammar.from_ebnf(grammar_str)

    grammar_str = """root ::= "a"++
"""
    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 14: Expect element, but got character: +",
    ):
        xgr.Grammar.from_ebnf(grammar_str)

    grammar_str = """root ::= "a"??
"""
    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 14: Expect element, but got character: ?",
    ):
        xgr.Grammar.from_ebnf(grammar_str)


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
a ::= (("a" a_1))
b ::= ((b_choice b_1))
c ::= ((c_1))
d ::= ((d_1))
e ::= (("ee" e_1))
f ::= (("fff"))
g ::= (())
a_1 ::= ("" | ("a"))
b_1 ::= ("" | (b_1_choice b_2))
b_2 ::= ("" | (b_2_choice b_3))
b_3 ::= ("" | (b_3_choice b_4))
b_4 ::= ("" | (a) | ("b"))
c_1 ::= ("" | ("c" c_2))
c_2 ::= ("" | ("c"))
d_1 ::= ("" | ("d" d_1))
e_1 ::= ("" | ("e" e_1))
b_choice ::= ((a) | ("b"))
b_1_choice ::= ((a) | ("b"))
b_2_choice ::= ((a) | ("b"))
b_3_choice ::= ((a) | ("b"))
"""

    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.structure_normalizer(grammar)
    grammar = GrammarFunctor.byte_string_fuser(grammar)
    after = str(grammar)
    assert after == expected


def test_lookahead_assertion():
    before = """root ::= ((b c d))
b ::= (("abc" [a-z])) (=("abc"))
c ::= (("a") | ("b")) (=([a-z] "b"))
d ::= (("ac") | ("b" d_choice)) (=("abc"))
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
    grammar = GrammarFunctor.byte_string_fuser(grammar)
    after = str(grammar)
    assert after == expected


def test_char():
    before = r"""root ::= [a-z] [A-z] "\u0234" "\U00000345\xff" [-A-Z] [--] [^a] rest
rest ::= [a-zA-Z0-9-] [\u0234-\U00000345] [测-试] [\--\]]  rest1
rest1 ::= "\?\"\'测试あc" "👀" "" [a-a] [b-b]
"""
    expected = r"""root ::= (([a-z] [A-z] "\u0234\u0345\xff" [\-A-Z] [\-\-] [^a] rest))
rest ::= (([a-zA-Z0-9\-] [\u0234-\u0345] [\u6d4b-\u8bd5] [\--\]] rest1))
rest1 ::= (("\?\"\'\u6d4b\u8bd5\u3042c\U0001f440ab"))
"""
    # Disable unwrap_nesting_rules to expose the result before unwrapping.
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.structure_normalizer(grammar)
    grammar = GrammarFunctor.byte_string_fuser(grammar)
    after = str(grammar)
    assert after == expected


def test_space():
    before = """

root::="a"  "b" ("c""d"
"e") |

"f" | "g"
"""
    expected = """root ::= (("abcde") | ("f") | ("g"))
"""
    grammar = xgr.Grammar.from_ebnf(before)
    after = str(grammar)
    assert after == expected


def test_nest():
    before = """root::= "a" ("b" | "c" "d") | (("e" "f"))
"""
    expected = """root ::= (("a" root_choice) | ("ef"))
root_choice ::= (("b") | ("cd"))
"""
    grammar = xgr.Grammar.from_ebnf(before)
    after = str(grammar)
    assert after == expected


def test_empty_parentheses():
    before = """root ::= "a" ( ) "b"
"""
    expected = """root ::= (("ab"))
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


def test_tag_dispatch():
    before = """root ::= TagDispatch(("tag1", rule1), ("tag2", rule2), ("tag3", rule3))
rule1 ::= "a"
rule2 ::= "b"
rule3 ::= "c"
"""
    expected = """root ::= TagDispatch(("tag1", rule1), ("tag2", rule2), ("tag3", rule3))
rule1 ::= (("a"))
rule2 ::= (("b"))
rule3 ::= (("c"))
"""
    grammar = xgr.Grammar.from_ebnf(before)
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
sequence_test ::= (("aab" sequence_test_choice "de" sequence_test))
nested_test ::= (("abcd") | ("a") | ("b") | ("c") | (nested_rest))
nested_rest ::= (("a") | ("bc") | ("d") | ("ef") | ("g"))
empty_test ::= ("" | ("d") | ("a"))
sequence_test_choice ::= (("c") | ("d"))
"""
    grammar = _ebnf_to_grammar_no_normalization(before)
    grammar = GrammarFunctor.structure_normalizer(grammar)
    grammar = GrammarFunctor.byte_string_fuser(grammar)
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
    after = str(grammar)
    assert after == expected


def test_e2e_to_string_roundtrip():
    """Checks the printed result can be parsed, and the parsing-printing process is idempotent."""
    before = r"""root ::= ((b c) | (b root))
b ::= ((b_1 d))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ("b" b_1))
c_1 ::= (([acep-z] c_1) | ([acep-z])) (=("d"))
d_1 ::= ("" | ("d"))
"""
    grammar_1 = xgr.Grammar.from_ebnf(before)
    output_string_1 = str(grammar_1)
    grammar_2 = xgr.Grammar.from_ebnf(output_string_1)
    output_string_2 = str(grammar_2)
    assert before == output_string_1
    assert output_string_1 == output_string_2


def test_e2e_tag_dispatch_roundtrip():
    """Checks the printed result can be parsed, and the parsing-printing process is idempotent."""
    before = r"""root ::= TagDispatch(("tag1", rule1), ("tag2", rule2), ("tag3", rule3))
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


def test_error():
    with pytest.raises(
        RuntimeError, match='EBNF parse error at line 1, column 11: Rule "a" is not defined'
    ):
        xgr.Grammar.from_ebnf("root ::= a b")

    with pytest.raises(RuntimeError, match="EBNF parse error at line 1, column 15: Expect element"):
        xgr.Grammar.from_ebnf('root ::= "a" |')

    with pytest.raises(RuntimeError, match='EBNF parse error at line 1, column 15: Expect "'):
        xgr.Grammar.from_ebnf('root ::= "a" "')

    with pytest.raises(
        RuntimeError, match="EBNF parse error at line 1, column 1: Expect rule name"
    ):
        xgr.Grammar.from_ebnf('::= "a"')

    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 12: Character class should not contain newline",
    ):
        xgr.Grammar.from_ebnf("root ::= [a\n]")

    with pytest.raises(
        RuntimeError, match="EBNF parse error at line 1, column 11: Invalid escape sequence"
    ):
        xgr.Grammar.from_ebnf(r'root ::= "\@"')

    with pytest.raises(
        RuntimeError, match="EBNF parse error at line 1, column 11: Invalid escape sequence"
    ):
        xgr.Grammar.from_ebnf(r'root ::= "\uFF"')

    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 14: Invalid character class: "
        "lower bound is larger than upper bound",
    ):
        xgr.Grammar.from_ebnf(r"root ::= [Z-A]")

    with pytest.raises(RuntimeError, match="EBNF parse error at line 1, column 6: Expect ::="):
        xgr.Grammar.from_ebnf(r'root := "a"')

    with pytest.raises(
        RuntimeError,
        match='EBNF parse error at line 2, column 9: Rule "root" is defined multiple times',
    ):
        xgr.Grammar.from_ebnf('root ::= "a"\nroot ::= "b"')

    with pytest.raises(
        RuntimeError,
        match='EBNF parse error at line 1, column 10: The root rule with name "root" is not found.',
    ):
        xgr.Grammar.from_ebnf('a ::= "a"')

    with pytest.raises(
        RuntimeError, match="EBNF parse error at line 1, column 21: Unexpected lookahead assertion"
    ):
        xgr.Grammar.from_ebnf('root ::= "a" (="a") (="b")')


def test_error_tag_dispatch():
    # Test empty tag
    with pytest.raises(RuntimeError):
        xgr.Grammar.from_ebnf(
            """root ::= TagDispatch(("", rule1))
rule1 ::= "a"
"""
        )

    # Test undefined rule
    with pytest.raises(RuntimeError):
        xgr.Grammar.from_ebnf(
            """root ::= TagDispatch(("tag1", undefined_rule))
"""
        )

    # Test using root rule as tag target
    with pytest.raises(RuntimeError):
        xgr.Grammar.from_ebnf(
            """root ::= TagDispatch(("tag1", root))
"""
        )

    # Test invalid TagDispatch syntax
    with pytest.raises(RuntimeError):
        xgr.Grammar.from_ebnf(
            """root ::= TagDispatch("tag1", rule1)
rule1 ::= "a"
"""
        )

    with pytest.raises(RuntimeError):
        xgr.Grammar.from_ebnf(
            """root ::= TagDispatch(("tag1" rule1))
rule1 ::= "a"
"""
        )

    # Test TagDispatch in non-root rule
    with pytest.raises(RuntimeError):
        xgr.Grammar.from_ebnf(
            """root ::= rule1
rule1 ::= TagDispatch(("tag1", rule2))
rule2 ::= "a"
"""
        )


if __name__ == "__main__":
    pytest.main(sys.argv)
