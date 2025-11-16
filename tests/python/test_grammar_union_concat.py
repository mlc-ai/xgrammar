"""This test uses the optimized JSON grammar provided by the grammar library."""

import sys

import pytest

import xgrammar as xgr


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


def test_grammar_union_with_stag():
    expected_grammar_union = r"""root ::= ((root_1_1) | (root_2))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"arg\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("" root_1 "end"))
triggered_tags ::= TagDispatch(
  ("start", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)
root_1_1 ::= ((triggered_tags))
root_2 ::= (([a-z] root_2) | ([a-z]))
"""

    expected_grammar_concat = r"""root ::= ((root_1_1 root_2))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"arg\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("" root_1 "end"))
triggered_tags ::= TagDispatch(
  ("start", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)
root_1_1 ::= ((triggered_tags))
root_2 ::= (([a-z] root_2) | ([a-z]))
"""
    start = "start"
    schema = {"type": "object", "properties": {"arg": {"type": "string"}}}
    end = "end"
    tag = xgr.StructuralTagItem(begin=start, schema=schema, end=end)
    triggers = [start]
    stag_grammar = xgr.Grammar.from_structural_tag([tag], triggers)
    start_grammar = xgr.Grammar.from_ebnf("root ::= [a-z] root | [a-z]")
    grammar_union = xgr.Grammar.union(stag_grammar, start_grammar)
    assert str(grammar_union) == expected_grammar_union
    grammar_concat = xgr.Grammar.concat(stag_grammar, start_grammar)
    assert str(grammar_concat) == expected_grammar_concat


if __name__ == "__main__":
    pytest.main(sys.argv)
