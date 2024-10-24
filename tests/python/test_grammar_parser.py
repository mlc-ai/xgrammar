import json
import sys

import pytest

from xgrammar import BNFGrammar


def test_bnf_simple():
    before = """root ::= b c
b ::= "b"
c ::= "c"
"""
    expected = """root ::= ((b c))
b ::= (("b"))
c ::= (("c"))
"""
    bnf_grammar = BNFGrammar(before, root_rule="root")
    after = bnf_grammar.to_string()
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
    bnf_grammar = BNFGrammar(before, root_rule="root")
    after = bnf_grammar.to_string()
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
    bnf_grammar = BNFGrammar(before, root_rule="root")
    after = bnf_grammar.to_string()
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
    bnf_grammar = BNFGrammar(before, root_rule="root")
    after = bnf_grammar.to_string()
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
    bnf_grammar = BNFGrammar(before, root_rule="root")
    after = bnf_grammar.to_string()
    assert after == expected


def test_space():
    before = """

root::="a"  "b" ("c""d"
"e") |

"f" | "g"
"""
    expected = """root ::= (("abcde") | ("f") | ("g"))
"""
    bnf_grammar = BNFGrammar(before, root_rule="root")
    after = bnf_grammar.to_string()
    assert after == expected


def test_nest():
    before = """root::= "a" ("b" | "c" "d") | (("e" "f"))
"""
    expected = """root ::= (("a" root_choice) | ("ef"))
root_choice ::= (("b") | ("cd"))
"""
    bnf_grammar = BNFGrammar(before, root_rule="root")
    after = bnf_grammar.to_string()
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
    bnf_grammar = BNFGrammar(before, root_rule="root")
    after = bnf_grammar.to_string()
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

    bnf_grammar = BNFGrammar(before, root_rule="root")
    after = bnf_grammar.to_string()
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
    bnf_grammar_1 = BNFGrammar(before, root_rule="root")
    output_string_1 = bnf_grammar_1.to_string()
    bnf_grammar_2 = BNFGrammar(output_string_1, root_rule="root")
    output_string_2 = bnf_grammar_2.to_string()
    assert before == output_string_1
    assert output_string_1 == output_string_2


def test_error():
    with pytest.raises(
        RuntimeError,
        match='EBNF parse error at line 1, column 11: Rule "a" is not defined',
    ):
        BNFGrammar("root ::= a b")

    with pytest.raises(RuntimeError, match="EBNF parse error at line 1, column 15: Expect element"):
        BNFGrammar('root ::= "a" |')

    with pytest.raises(RuntimeError, match='EBNF parse error at line 1, column 15: Expect "'):
        BNFGrammar('root ::= "a" "')

    with pytest.raises(
        RuntimeError, match="EBNF parse error at line 1, column 1: Expect rule name"
    ):
        BNFGrammar('::= "a"')

    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 12: Character class should not contain "
        "newline",
    ):
        BNFGrammar("root ::= [a\n]")

    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 11: Invalid escape sequence",
    ):
        BNFGrammar(r'root ::= "\@"')

    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 11: Invalid escape sequence",
    ):
        BNFGrammar(r'root ::= "\uFF"')

    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 14: Invalid character class: "
        "lower bound is larger than upper bound",
    ):
        BNFGrammar(r"root ::= [Z-A]")

    with pytest.raises(RuntimeError, match="EBNF parse error at line 1, column 6: Expect ::="):
        BNFGrammar(r'root := "a"')

    with pytest.raises(
        RuntimeError,
        match='EBNF parse error at line 2, column 9: Rule "root" is defined multiple times',
    ):
        BNFGrammar('root ::= "a"\nroot ::= "b"')

    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 10: "
        'The root rule with name "root" is not found.',
    ):
        BNFGrammar('a ::= "a"')

    with pytest.raises(
        RuntimeError,
        match="EBNF parse error at line 1, column 21: Unexpected lookahead assertion",
    ):
        BNFGrammar('root ::= "a" (="a") (="b")')


def test_to_json():
    before = """root ::= b c | b root
b ::= "bcd"
c ::= [a-z]
"""
    expected_obj = {
        "rules": [
            {"body_expr_id": 6, "name": "root"},
            {"body_expr_id": 9, "name": "b"},
            {"body_expr_id": 12, "name": "c"},
        ],
        "rule_expr_indptr": [0, 3, 6, 10, 13, 16, 20, 24, 29, 32, 35, 40, 43],
        "rule_expr_data": [
            # fmt: off
            4,1,1,4,1,2,5,2,0,1,4,1,1,4,1,0,5,2,3,4,6,2,2,5,0,3,98,99,
            100,5,1,7,6,1,8,1,3,0,97,122,5,1,10,6,1,11,
            # fmt: on
        ],
    }
    bnf_grammar = BNFGrammar(before, root_rule="root")
    after_str = bnf_grammar.serialize(prettify=False)
    after_obj = json.loads(after_str)
    assert after_obj == expected_obj


def test_to_json_roundtrip():
    before = r"""root ::= ((b c) | (b root))
b ::= ((b_1 d [a]*))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ("b" b_1))
c_1 ::= ((c_2 c_1) | (c_2))
c_2 ::= (([acep-z]))
d_1 ::= ("" | ("d"))
"""
    bnf_grammar_1 = BNFGrammar(before, root_rule="root")
    output_json_1 = bnf_grammar_1.serialize(prettify=False)
    bnf_grammar_2 = BNFGrammar.deserialize(output_json_1)
    output_json_2 = bnf_grammar_2.serialize(prettify=False)
    output_str = bnf_grammar_2.to_string()
    assert output_json_1 == output_json_2
    assert output_str == before


if __name__ == "__main__":
    pytest.main(sys.argv)
