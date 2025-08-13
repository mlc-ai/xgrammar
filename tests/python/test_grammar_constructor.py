import sys

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


def test_grammar_constructor_empty():
    expected_grammar = 'root ::= ("")\n'
    empty_grammar = xgr.Grammar.empty()
    assert empty_grammar is not None
    empty_grammar_str = str(empty_grammar)
    assert empty_grammar_str == expected_grammar


input_test_string_expected_grammar_test_grammar_constructor_string = (
    ("our beautiful Earth", "our beautiful Earth", 'root ::= (("our beautiful Earth"))\n'),
    ("wqepqw\n", "wqepqw\n", 'root ::= (("wqepqw\\n"))\n'),
    ("世界", "世界", 'root ::= (("\\u4e16\\u754c"))\n'),
    ("ぉ", "ぉ", 'root ::= (("\\u3049"))\n'),
    (
        "123werlkここぉ1q12",
        "123werlkここぉ1q12",
        'root ::= (("123werlk\\u3053\\u3053\\u30491q12"))\n',
    ),
)


@pytest.mark.parametrize(
    "input, test_string, expected_grammar",
    input_test_string_expected_grammar_test_grammar_constructor_string,
)
def test_grammar_constructor_string(input: str, test_string: str, expected_grammar: str):
    grammar = xgr.Grammar.string(input)
    assert grammar is not None
    grammar_str = str(grammar)
    assert _is_grammar_accept_string(grammar, test_string)
    assert grammar_str == expected_grammar


def test_grammar_union():
    grammar1 = xgr.Grammar.from_ebnf(
        """root ::= r1 | r2
r1 ::= "true" | ""
r2 ::= "false" | ""
"""
    )

    grammar2 = xgr.Grammar.from_ebnf(
        """root ::= "abc" | r1
r1 ::= "true" | r1
"""
    )

    grammar3 = xgr.Grammar.from_ebnf(
        """root ::= r1 | r2 | r3
r1 ::= "true" | r3
r2 ::= "false" | r3
r3 ::= "abc" | ""
"""
    )

    expected = """root ::= ((root_1) | (root_2) | (root_3))
root_1 ::= ((r1) | (r2))
r1 ::= ("" | ("true"))
r2 ::= ("" | ("false"))
root_2 ::= (("abc") | (r1_1))
r1_1 ::= (("true") | (r1_1))
root_3 ::= ((r1_2) | (r2_1) | (r3))
r1_2 ::= (("true") | (r3))
r2_1 ::= (("false") | (r3))
r3 ::= ("" | ("abc"))
"""

    union_grammar = xgr.Grammar.union(grammar1, grammar2, grammar3)
    assert str(union_grammar) == expected


def test_grammar_concat():
    grammar1 = xgr.Grammar.from_ebnf(
        """root ::= r1 | r2
r1 ::= "true" | ""
r2 ::= "false" | ""
"""
    )

    grammar2 = xgr.Grammar.from_ebnf(
        """root ::= "abc" | r1
r1 ::= "true" | r1
"""
    )

    grammar3 = xgr.Grammar.from_ebnf(
        """root ::= r1 | r2 | r3
r1 ::= "true" | r3
r2 ::= "false" | r3
r3 ::= "abc" | ""
"""
    )

    expected = """root ::= ((root_1 root_2 root_3))
root_1 ::= ((r1) | (r2))
r1 ::= ("" | ("true"))
r2 ::= ("" | ("false"))
root_2 ::= (("abc") | (r1_1))
r1_1 ::= (("true") | (r1_1))
root_3 ::= ((r1_2) | (r2_1) | (r3))
r1_2 ::= (("true") | (r3))
r2_1 ::= (("false") | (r3))
r3 ::= ("" | ("abc"))
"""

    concat_grammar = xgr.Grammar.concat(grammar1, grammar2, grammar3)
    assert str(concat_grammar) == expected


input_expected_accepted_test_grammar_constructor_character_class = (
    ("a", True),
    ("b", True),
    ("c", True),
    ("-", True),
    ("1", False),
    (" ", False),
    ("A", True),
)


@pytest.mark.parametrize(
    "input, expected_accept", input_expected_accepted_test_grammar_constructor_character_class
)
def test_grammar_character_constructor_class(input: str, expected_accept: bool):
    expected_grammar = "root ::= (([\\-a-zA-Z]))\n"
    grammar_str = "[-a-zA-Z]"
    grammar = xgr.Grammar.character_class(grammar_str)
    assert grammar is not None
    assert str(grammar) == expected_grammar
    assert _is_grammar_accept_string(grammar, input) == expected_accept


input_expected_accepted_test_grammar_constructor_character_class_negated = (
    ("a", False),
    ("b", False),
    ("c", False),
    ("-", False),
    ("1", True),
    (" ", True),
    ("A", False),
    ("好", True),
)


@pytest.mark.parametrize(
    "input, expected_accept",
    input_expected_accepted_test_grammar_constructor_character_class_negated,
)
def test_grammar_constructor_character_class_negated(input: str, expected_accept: bool):
    expected_grammar = "root ::= (([^\\-a-zA-Z]))\n"
    grammar_str = "[^-a-zA-Z]"
    grammar = xgr.Grammar.character_class(grammar_str)
    assert grammar is not None
    assert str(grammar) == expected_grammar
    assert _is_grammar_accept_string(grammar, input) == expected_accept


input_expected_accepted_test_grammar_constructor_tag_dispatch = (
    ("eeeee", True),
    ("tag114", True),
    ("tag1", False),
    ("tag2123", False),
    ("tag11tag20.3", True),
    ("tag20.", False),
    ("tag11111tag20.2", True),
)


@pytest.mark.parametrize(
    "input, expected_accept", input_expected_accepted_test_grammar_constructor_tag_dispatch
)
def test_grammar_constructor_tag_dispatch(input: str, expected_accept: bool):
    expected_grammar = """root ::= TagDispatch(
  ("tag1", trigger_rule_0),
  ("tag2", trigger_rule_1),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)
trigger_rule_0 ::= ((root_1))
root_1 ::= (("[a-z]+") | (rule2))
rule2 ::= ((rule2_1))
rule2_1 ::= (([0-9] rule2_1) | ([0-9]))
trigger_rule_1 ::= ((root_2))
root_2 ::= (("0." rule2_2))
rule2_2 ::= ((rule2_1_1))
rule2_1_1 ::= (([0-9] rule2_1_1) | ([0-9]))
"""

    grammar_1_str = """
        root ::= rule1 | rule2
        rule1 ::= \"[a-z]+\"
        rule2 ::= [0-9]+
    """

    grammar_2_str = """
        root ::= rule1 rule2
        rule1 ::= "0."
        rule2 ::= [0-9]+
    """

    tag1 = "tag1"
    tag2 = "tag2"

    grammar_1 = xgr.Grammar.from_ebnf(grammar_1_str)
    grammar_2 = xgr.Grammar.from_ebnf(grammar_2_str)

    grammar_list = [grammar_1, grammar_2]
    tag_list = [tag1, tag2]
    grammar = xgr.Grammar.tag_dispatch(tags=tag_list, grammars=grammar_list)
    assert grammar is not None
    assert str(grammar) == expected_grammar
    assert _is_grammar_accept_string(grammar, input) == expected_accept


def test_grammar_constructor_plus():
    grammar1 = xgr.Grammar.string("0.")
    grammar2 = xgr.Grammar.plus(xgr.Grammar.character_class("[0-9]"))
    grammar = xgr.Grammar.concat(grammar1, grammar2)
    expected_grammar = """root ::= ((root_1 root_2))
root_1 ::= (("0."))
root_2 ::= ((root_1_1 root_2_1))
root_1_1 ::= (([0-9]))
root_2_1 ::= ("" | (root_1_1_1 root_2_1))
root_1_1_1 ::= (([0-9]))
"""

    assert str(grammar) == expected_grammar
    assert _is_grammar_accept_string(grammar, "0.1234567890")
    assert not _is_grammar_accept_string(grammar, "0.123456789a")
    assert not _is_grammar_accept_string(grammar, "0.")
    assert _is_grammar_accept_string(grammar, "0.0")


def test_grammar_constructor_star():
    grammar1 = xgr.Grammar.string('"')
    grammar2 = xgr.Grammar.star(xgr.Grammar.character_class("[a-z]"))
    grammar = xgr.Grammar.concat(grammar1, grammar2, grammar1)
    expected_grammar = """root ::= ((root_1 root_2 root_3))
root_1 ::= (("\\\""))
root_2 ::= ("" | (root_1_1 root_2))
root_1_1 ::= (([a-z]))
root_3 ::= (("\\\""))
"""

    assert str(grammar) == expected_grammar
    assert _is_grammar_accept_string(grammar, '"azazaz"')
    assert _is_grammar_accept_string(grammar, '""')
    assert not _is_grammar_accept_string(grammar, "azazaz")
    assert not _is_grammar_accept_string(grammar, '"azazaz')
    assert not _is_grammar_accept_string(grammar, '"a--"a')


def test_grammar_constructor_optional():
    grammar1 = xgr.Grammar.union(xgr.Grammar.string("-"), xgr.Grammar.string("+"))
    grammar2 = xgr.Grammar.plus(xgr.Grammar.character_class("[0-9]"))
    grammar1_optional = xgr.Grammar.optional(grammar1)
    grammar = xgr.Grammar.concat(grammar1_optional, grammar2)
    expected_grammar = """root ::= ((root_1 root_4))
root_1 ::= ((root_1_1) | (root_3))
root_1_1 ::= ((root_1_1_1) | (root_2))
root_1_1_1 ::= (("-"))
root_2 ::= (("+"))
root_3 ::= ("")
root_4 ::= ((root_1_2 root_2_1))
root_1_2 ::= (([0-9]))
root_2_1 ::= ("" | (root_1_1_2 root_2_1))
root_1_1_2 ::= (([0-9]))
"""

    assert str(grammar) == expected_grammar
    assert _is_grammar_accept_string(grammar, "123")
    assert _is_grammar_accept_string(grammar, "-123")
    assert _is_grammar_accept_string(grammar, "+123")
    assert not _is_grammar_accept_string(grammar, "++123")
    assert not _is_grammar_accept_string(grammar, "123-")
    assert not _is_grammar_accept_string(grammar, "+-123")


def test_grammar_constructor_tag_dispatch_with_concatenation():
    grammar_1_str = """
root ::= rule1 | rule2
rule1 ::= \"[a-z]+\"
rule2 ::= [0-9]+
"""
    grammar1 = xgr.Grammar.from_ebnf(grammar_1_str)
    tag = "tag1"
    grammar = xgr.Grammar.tag_dispatch(
        tags=[tag], grammars=[grammar1], stop_eos=False, stop_str=["end"], loop_after_dispatch=False
    )
    test_grammar = xgr.Grammar.concat(grammar, xgr.Grammar.string("between"), grammar)
    assert test_grammar is not None
    assert _is_grammar_accept_string(test_grammar, "tag1123endbetween123end")
    assert _is_grammar_accept_string(test_grammar, "endbetweenend")
    assert _is_grammar_accept_string(test_grammar, "endbetweentag123end")
    assert not _is_grammar_accept_string(test_grammar, "tag1endbetween123end")
    assert not _is_grammar_accept_string(test_grammar, "endend")
    assert not _is_grammar_accept_string(test_grammar, "tag1abcendbetween123endextra")


def test_grammar_constructor_tag_dispatch_with_union():
    grammar_1_str = """
root ::= [a-z]+
"""
    grammar1 = xgr.Grammar.from_ebnf(grammar_1_str)

    tagdispatch_a = xgr.Grammar.tag_dispatch(
        tags=["tag1"],
        grammars=[grammar1],
        stop_eos=False,
        stop_str=["end"],
        loop_after_dispatch=False,
    )

    grammar = xgr.Grammar.union(tagdispatch_a, xgr.Grammar.string("Interesting"))

    assert grammar is not None
    assert _is_grammar_accept_string(grammar, "tag1abcend")
    assert _is_grammar_accept_string(grammar, "Interesting")
    assert _is_grammar_accept_string(grammar, "tag1abcendextraend")
    assert _is_grammar_accept_string(grammar, "Interesting")
    assert not _is_grammar_accept_string(grammar, "tag1abcI")
    assert not _is_grammar_accept_string(grammar, "Interestingextra")


if __name__ == "__main__":
    pytest.main(sys.argv)
