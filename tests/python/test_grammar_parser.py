import sys

import pytest

import xgrammar as xgr


def test_bnf_simple():
    before = """root ::= b c
b ::= "b"
c ::= "c"
"""
    expected = """root ::= ((b c))
b ::= (("b"))
c ::= (("c"))
"""
    grammar = xgr.Grammar.from_ebnf(before)
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
    grammar = xgr.Grammar.from_ebnf(before)
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
c_1 ::= (([acep-z] c_1) | ([acep-z]))
d_1 ::= ("" | ("d"))
"""
    grammar = xgr.Grammar.from_ebnf(before)
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
    grammar = xgr.Grammar.from_ebnf(before)
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

    grammar = xgr.Grammar.from_ebnf(before)
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
    grammar = xgr.Grammar.from_ebnf(before)
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
    grammar = xgr.Grammar.from_ebnf(before)
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
    grammar = xgr.Grammar.from_ebnf(before)
    after = str(grammar)
    assert after == expected


def test_json_grammar():
    # Adopted from https://www.crockford.com/mckeeman.html. Not optimized
    before = r"""root ::= element
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws "}" | "{" members "}"
members ::= member | member "," members
member ::= ws string ws ":" element
array ::= "[" ws "]" | "[" elements "]"
elements ::= element | element "," elements
element ::= ws value ws
string ::= "\"" characters "\""
characters ::= "" | character characters
character ::= [^"\\] | "\\" escape
escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" hex hex hex hex
hex ::= [A-Fa-f0-9]
number ::= integer fraction exponent
integer ::= digit | onenine digits | "-" digit | "-" onenine digits
digits ::= digit | digit digits
digit ::= [0-9]
onenine ::= [1-9]
fraction ::= "" | "." digits
exponent ::= "" | ("e" | "E") ("" | "+" | "-") digits
ws ::= "" | "\u0020" ws | "\u000A" ws | "\u000D" ws | "\u0009" ws
"""

    expected = r"""root ::= ((element))
value ::= ((object) | (array) | (string) | (number) | ("true") | ("false") | ("null"))
object ::= (("{" ws "}") | ("{" members "}"))
members ::= ((member) | (member "," members))
member ::= ((ws string ws ":" element))
array ::= (("[" ws "]") | ("[" elements "]"))
elements ::= ((element) | (element "," elements))
element ::= ((ws value ws))
string ::= (("\"" characters "\""))
characters ::= ("" | (character characters))
character ::= (([^\"\\]) | ("\\" escape))
escape ::= (("\"") | ("\\") | ("/") | ("b") | ("f") | ("n") | ("r") | ("t") | ("u" hex hex hex hex))
hex ::= (([A-Fa-f0-9]))
number ::= ((integer fraction exponent))
integer ::= ((digit) | (onenine digits) | ("-" digit) | ("-" onenine digits))
digits ::= ((digit) | (digit digits))
digit ::= (([0-9]))
onenine ::= (([1-9]))
fraction ::= ("" | ("." digits))
exponent ::= ("" | (exponent_choice exponent_choice_1 digits))
ws ::= ("" | (" " ws) | ("\n" ws) | ("\r" ws) | ("\t" ws))
exponent_choice ::= (("e") | ("E"))
exponent_choice_1 ::= ("" | ("+") | ("-"))
"""

    grammar = xgr.Grammar.from_ebnf(before)
    after = str(grammar)
    assert after == expected


def test_to_string_roundtrip():
    """Checks the printed result can be parsed, and the parsing-printing process is idempotent."""
    before = r"""root ::= ((b c) | (b root))
b ::= ((b_1 d))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ("b" b_1))
c_1 ::= ((c_2 c_1) | (c_2)) (=("abc" [a-z]))
c_2 ::= (([acep-z]))
d_1 ::= ("" | ("d"))
"""
    grammar_1 = xgr.Grammar.from_ebnf(before)
    output_string_1 = str(grammar_1)
    grammar_2 = xgr.Grammar.from_ebnf(output_string_1)
    output_string_2 = str(grammar_2)
    assert before == output_string_1
    assert output_string_1 == output_string_2


def test_error():
    with pytest.raises(
        RuntimeError,
        match='EBNF parse error at line 1, column 11: Rule "a" is not defined',
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
        match="EBNF parse error at line 1, column 12: Character class should not contain "
        "newline",
    ):
        xgr.Grammar.from_ebnf("root ::= [a\n]")

    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 11: Invalid escape sequence",
    ):
        xgr.Grammar.from_ebnf(r'root ::= "\@"')

    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 11: Invalid escape sequence",
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
        match="EBNF parse error at line 1, column 10: "
        'The root rule with name "root" is not found.',
    ):
        xgr.Grammar.from_ebnf('a ::= "a"')

    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 21: Unexpected lookahead assertion",
    ):
        xgr.Grammar.from_ebnf('root ::= "a" (="a") (="b")')


if __name__ == "__main__":
    pytest.main(sys.argv)
