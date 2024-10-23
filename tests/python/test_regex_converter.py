import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pytest
from pydantic import BaseModel, Field, TypeAdapter

from xgrammar import BuiltinGrammar, GrammarMatcher
from xgrammar.xgrammar import BNFGrammar


def match_string_to_grammar(grammar_str: str, input_str: str) -> bool:
    grammar = BNFGrammar(grammar_str)
    matcher = GrammarMatcher(grammar, terminate_without_stop_token=True)
    can_accept = matcher.accept_string(input_str, verbose=False)
    can_terminate = matcher.is_terminated()
    return can_accept and can_terminate


def test_basic():
    regex = "123"
    expected = ...
    grammar_str = BuiltinGrammar._regex_to_ebnf(regex)
    assert grammar_str == expected
    assert match_string_to_grammar(grammar_str, "123")
    assert not match_string_to_grammar(grammar_str, "1234")


def test_unicode():
    regex = "ww我😁"
    grammar_str = BuiltinGrammar._regex_to_ebnf(regex)
    assert match_string_to_grammar(grammar_str, regex)


#   static const std::unordered_map<char, TCodepoint> CUSTOM_ESCAPE_MAP = {
#       {'^', '^'}, {'$', '$'}, {'.', '.'}, {'*', '*'}, {'+', '+'}, {'?', '?'}, {'\\', '\\'},
#       {'(', '('}, {')', ')'}, {'[', '['}, {']', ']'}, {'{', '{'}, {'}', '}'}, {'|', '|'},
#       {'/', '/'}
#   };
#   static const std::unordered_map<char, TCodepoint> kEscapeToCodepoint = {
#       {'\'', '\''},
#       {'\"', '\"'},
#       {'\?', '\?'},
#       {'\\', '\\'},
#       {'a', '\a'},
#       {'b', '\b'},
#       {'f', '\f'},
#       {'n', '\n'},
#       {'r', '\r'},
#       {'t', '\t'},
#       {'v', '\v'},
#       {'0', '\0'},
#       {'e', '\x1B'}
#   };


def test_escape():
    # regex1 = r"\^\$\.\*\+\?\\\(\)\[\]\{\}\|\/"
    # instance1 = "^$.*+?\\()[]{}|/"
    # grammar_str = BuiltinGrammar._regex_to_ebnf(regex1)
    # assert match_string_to_grammar(grammar_str, instance1)
    # regex2 = r"\"\'\a\b\f\n\r\t\v\0\e"
    # instance2 = '"\'\a\b\f\n\r\t\v\0\x1B'
    # grammar_str = BuiltinGrammar._regex_to_ebnf(regex2)
    # assert match_string_to_grammar(grammar_str, instance2)
    regex3 = r"\u{20BB7}\u0300\x1F\cJ"
    instance3 = "\U00020BB7\u0300\x1F\n"
    grammar_str = BuiltinGrammar._regex_to_ebnf(regex3)
    assert match_string_to_grammar(grammar_str, instance3)
    regex4 = r"[\r\n]"
    instance3 = "\U00020BB7\u0300\x1F\n"
    grammar_str = BuiltinGrammar._regex_to_ebnf(regex3)
    assert match_string_to_grammar(grammar_str, instance3)


test_escape()
exit()


def test_escaped_char_class(): ...


def test_char_class(): ...


def test_boundary(): ...


def test_disjunction(): ...


def test_quantifier(): ...


def test_group(): ...


def test_ipv4():
    regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    grammar_str = BuiltinGrammar._regex_to_ebnf(regex)
    print(grammar_str)
    assert match_string_to_grammar(grammar_str, "123.45.67.89")


def test_date(): ...


def test_mask_generation(): ...


test_ipv4()
exit()


if __name__ == "__main__":
    pytest.main([__file__])
