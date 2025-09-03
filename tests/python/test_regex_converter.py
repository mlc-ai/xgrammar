import sys
import time

import pytest
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string, _regex_to_ebnf


def test_basic():
    regex = "123"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= "1" "2" "3"
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, "123")
    assert not _is_grammar_accept_string(grammar_str, "1234")


def test_unicode():
    regex = "ww我😁"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= "w" "w" "\u6211" "\U0001f601"
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, regex)


regex_expected_grammar_instance = [
    (
        r"\^\$\.\*\+\?\\\(\)\[\]\{\}\|\/",
        r"""root ::= "^" "$" "." "*" "+" "\?" "\\" "(" ")" "[" "]" "{" "}" "|" "/"
""",
        "^$.*+?\\()[]{}|/",
    ),
    (
        r"\"\'\a\f\n\r\t\v\0\e",
        r"""root ::= "\"" "\'" "\a" "\f" "\n" "\r" "\t" "\v" "\0" "\e"
""",
        "\"'\a\f\n\r\t\v\0\x1b",
    ),
    (
        r"\u{20BB7}\u0300\x1F\cJ",
        r"""root ::= "\U00020bb7" "\u0300" "\x1f" "\n"
""",
        "\U00020bb7\u0300\x1f\n",
    ),
    (
        r"[\r\n\$\u0010-\u006F\]\--]+",
        r"""root ::= [\r\n$\x10-o\]\--]+
""",
        "\r\n$\u0020-",  # TODO(yixin): add unicode tests
    ),
]


@pytest.mark.parametrize("regex, expected_grammar, instance", regex_expected_grammar_instance)
def test_escape(regex: str, expected_grammar: str, instance: str):
    grammar_str = _regex_to_ebnf(regex)
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, instance)


def test_escaped_char_class():
    # TODO(yixin): add unicode tests
    # TODO(yixin): add tests for escaped char class nested in char class
    regex = r"\w\w\W\d\D\s\S"
    instance = "A_ 1b 0"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= [a-zA-Z0-9_] [a-zA-Z0-9_] [^a-zA-Z0-9_] [0-9] [^0-9] [\f\n\r\t\v\u0020\u00a0] [^[\f\n\r\t\v\u0020\u00a0]
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, instance)


def test_char_class():
    regex = r"[-a-zA-Z+--]+"
    instance = "a-+"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= [-a-zA-Z+--]+
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, instance)


def test_boundary():
    regex = r"^abc$"
    instance = "abc"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= "a" "b" "c"
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, instance)


def test_disjunction():
    regex = r"abc|de(f|g)"
    instance = "deg"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= "a" "b" "c" | "d" "e" ( "f" | "g" )
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, instance)


def test_space():
    regex = r" abc | df | g "
    instance = " df "
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= " " "a" "b" "c" " " | " " "d" "f" " " | " " "g" " "
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, instance)


def test_quantifier():
    regex = r"(a|b)?[a-z]+(abc)*"
    instance = "adddabcabc"
    instance1 = "z"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= ( "a" | "b" )? [a-z]+ ( "a" "b" "c" )*
"""
    # TODO(yixin): add tests for repetition range
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, instance)
    assert _is_grammar_accept_string(grammar_str, instance1)


def test_consecutive_quantifiers():
    regex = "a{1,3}?{1,3}"
    with pytest.raises(RuntimeError, match="Two consecutive repetition modifiers are not allowed."):
        _regex_to_ebnf(regex)

    regex = "a???"
    with pytest.raises(RuntimeError, match="Two consecutive repetition modifiers are not allowed."):
        _regex_to_ebnf(regex)

    regex = "a++"
    with pytest.raises(RuntimeError, match="Two consecutive repetition modifiers are not allowed."):
        _regex_to_ebnf(regex)

    regex = "a+?{1,3}"
    with pytest.raises(RuntimeError, match="Two consecutive repetition modifiers are not allowed."):
        _regex_to_ebnf(regex)


def test_group():
    regex = r"(a|b)(c|d)"
    instance = "ac"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= ( "a" | "b" ) ( "c" | "d" )
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, instance)


def test_any():
    regex = r".+a.+"
    instance = "bbbabb"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= [\u0000-\U0010FFFF]+ "a" [\u0000-\U0010FFFF]+
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, instance)


def test_ipv4():
    regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = (
        r"""root ::= ( ( "2" "5" [0-5] | "2" [0-4] [0-9] | [01]? [0-9] [0-9]? ) """
        r"""[\u0000-\U0010FFFF] ) ( ( "2" "5" [0-5] | "2" [0-4] [0-9] | [01]? [0-9] """
        r"""[0-9]? ) [\u0000-\U0010FFFF] ) ( ( "2" "5" [0-5] | "2" [0-4] [0-9] | [01]? [0-9] """
        r"""[0-9]? ) [\u0000-\U0010FFFF] ) ( "2" "5" [0-5] | "2" [0-4] [0-9] | [01]? [0-9] [0-9]? )
"""
    )
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, "123.45.67.89")


date_time_instances_accepted = [
    ("2024-05-19T14:23:45Z", True),
    ("2019-11-30T08:15:27+05:30", True),
    ("2030-02-01T22:59:59-07:00", True),
    ("2021-07-04T00:00:00.123456Z", True),
    ("2022-12-31T23:45:12-03:00", True),
    ("2024-13-15T14:30:00Z", False),
    ("2023-02-2010:59:59Z", False),
    ("2021-11-05T24:00:00+05:30", False),
    ("2022-08-20T12:61:10-03:00", False),
]


@pytest.mark.parametrize("instance, accepted", date_time_instances_accepted)
def test_date_time(instance: str, accepted: bool):
    regex = r"^\d\d\d\d-(0[1-9]|1[0-2])-([0-2]\d|3[01])T([01]\d|2[0123]):[0-5]\d:[0-5]\d(\.\d+)?(Z|[+-]([01]\d|2[0123]):[0-5]\d)$"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = (
        r"""root ::= [0-9] [0-9] [0-9] [0-9] "-" ( "0" [1-9] | "1" [0-2] ) "-" ( [0-2] [0-9] """
        r"""| "3" [01] ) "T" ( [01] [0-9] | "2" [0123] ) ":" [0-5] [0-9] ":" [0-5] [0-9] """
        r"""( "." [0-9]+ )? ( "Z" | [+-] ( [01] [0-9] | "2" [0123] ) ":" [0-5] [0-9] )
"""
    )
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, instance) == accepted


date_instances_accepted = [
    ("0024-05-19", True),
    ("2019-11-30", True),
    ("2022-12-31", True),
    ("2024-13-15", False),
    ("2024-12-32", False),
]


@pytest.mark.parametrize("instance, accepted", date_instances_accepted)
def test_date(instance: str, accepted: bool):
    regex = r"^\d\d\d\d-(0[1-9]|1[0-2])-([0-2]\d|3[01])$"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = (
        r"""root ::= [0-9] [0-9] [0-9] [0-9] "-" ( "0" [1-9] | "1" [0-2] ) "-" """
        r"""( [0-2] [0-9] | "3" [01] )
"""
    )
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, instance) == accepted


time_instances_accepted = [
    ("14:23:45Z", True),
    ("08:15:27+05:30", True),
    ("22:59:59-07:00", True),
    ("00:00:00.123456Z", True),
    ("10:59:59ZA", False),
    ("24:00:00+05:30", False),
    ("12:15:10-03:60", False),
]


@pytest.mark.parametrize("instance, accepted", time_instances_accepted)
def test_time(instance: str, accepted: bool):
    regex = r"^([01]\d|2[0123]):[0-5]\d:[0-5]\d(\.\d+)?(Z|[+-]([01]\d|2[0123]):[0-5]\d)$"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = (
        r"""root ::= ( [01] [0-9] | "2" [0123] ) ":" [0-5] [0-9] ":" [0-5] [0-9] """
        r"""( "." [0-9]+ )? ( "Z" | [+-] ( [01] [0-9] | "2" [0123] ) ":" [0-5] [0-9] )
"""
    )
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, instance) == accepted


email_instances_accepted = [
    ("simple@example.com", True),
    ("very.common@example.com", True),
    ("user_name+123@example.co.uk", True),
    ('"john.doe"@example.org', True),
    ("mail-host@online-shop.biz", True),
    ("customer/department=shipping@example.com", True),
    ("$A12345@example.non-profit.org", True),
    ('"!def!xyz%abc"@example.com', True),
    ("support@192.168.1.1", True),
    ("plainaddress", False),
    ("@missingusername.com", False),
    ("user@.com.my", False),
    ("user@com", False),
    ("user@-example.com", False),
]


@pytest.mark.parametrize("instance, accepted", email_instances_accepted)
def test_email(instance: str, accepted: bool):
    regex = (
        r"""^([\w!#$%&'*+/=?^_`{|}~-]+(\.[\w!#$%&'*+/=?^_`{|}~-]+)*"""
        r"""|"([\w!#$%&'*+/=?^_`{|}~\-(),:;<>@[\].]|\\")+")@(([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+"""
        r"""[a-z0-9]([a-z0-9-]*[a-z0-9])?)$"""
    )
    grammar_str = _regex_to_ebnf(regex)
    assert _is_grammar_accept_string(grammar_str, instance) == accepted


def test_empty_character_class():
    regex = "[]"
    with pytest.raises(RuntimeError, match="Empty character class is not allowed in regex."):
        _regex_to_ebnf(regex)


def test_group_modifiers():
    # Test non-capturing group
    regex = "(?:abc)"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= ( "a" "b" "c" )
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, "abc")

    # Test named capturing group
    regex = "(?<name>abc)"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= ( "a" "b" "c" )
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, "abc")

    # Test unsupported group modifiers
    unsupported_regexes = [
        "(?=abc)",  # Positive lookahead
        "(?!abc)",  # Negative lookahead
        "(?<=abc)",  # Positive lookbehind
        "(?<!abc)",  # Negative lookbehind
        "(?i)abc",  # Case-insensitive flag
    ]

    for regex in unsupported_regexes:
        with pytest.raises(RuntimeError):
            _regex_to_ebnf(regex)


def test_unmatched_parentheses():
    # Test unmatched closing parenthesis
    regex = "abc)"
    with pytest.raises(RuntimeError, match="Unmatched '\\)'"):
        _regex_to_ebnf(regex)

    regex = "abc((a)"
    with pytest.raises(RuntimeError, match="The parenthesis is not closed."):
        _regex_to_ebnf(regex)


def test_empty_parentheses():
    regex = "()"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= ( )
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, "")

    regex = "a()b"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= "a" ( ) "b"
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, "ab")


def test_empty_alternative():
    regex = "(a|)"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= ( "a" | "" )
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, "a")
    assert _is_grammar_accept_string(grammar_str, "")
    assert not _is_grammar_accept_string(grammar_str, "b")

    # Nested case
    regex = "ab(c|)"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= "a" "b" ( "c" | "" )
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, "abc")
    assert _is_grammar_accept_string(grammar_str, "ab")
    assert not _is_grammar_accept_string(grammar_str, "abd")


def test_non_greedy_quantifier():
    regex = "a{1,3}?"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= "a"{1,3}
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, "a")
    assert _is_grammar_accept_string(grammar_str, "aa")
    assert _is_grammar_accept_string(grammar_str, "aaa")
    assert not _is_grammar_accept_string(grammar_str, "aaaa")

    regex = "a+?"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= "a"+
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, "a")
    assert _is_grammar_accept_string(grammar_str, "aa")
    assert _is_grammar_accept_string(grammar_str, "aaa")
    assert not _is_grammar_accept_string(grammar_str, "")

    regex = "a*?"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= "a"*
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, "a")
    assert _is_grammar_accept_string(grammar_str, "aa")
    assert _is_grammar_accept_string(grammar_str, "aaa")
    assert _is_grammar_accept_string(grammar_str, "")

    regex = "a??"
    grammar_str = _regex_to_ebnf(regex)
    expected_grammar = r"""root ::= "a"?
"""
    assert grammar_str == expected_grammar
    assert _is_grammar_accept_string(grammar_str, "a")
    assert _is_grammar_accept_string(grammar_str, "")
    assert not _is_grammar_accept_string(grammar_str, "aa")


tokenizer_paths = ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct"]
regex_instances = [
    (r".+a.+", "bbbabb"),
    (
        r"((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)(25[0-5]|2[0-4]\d|[01]?\d\d?)",
        "123.45.67.89",
    ),
    (
        r"^\d\d\d\d-(0[1-9]|1[0-2])-([0-2]\d|3[01])T([01]\d|2[0123]):[0-5]\d:[0-5]\d(\.\d+)?(Z|[+-]([01]\d|2[0123]):[0-5]\d)$",
        "2024-05-19T14:23:45Z",
    ),
    (r"^\d\d\d\d-(0[1-9]|1[0-2])-([0-2]\d|3[01])$", "2024-05-19"),
    (
        r"^([01]\d|2[0123]):[0-5]\d:[0-5]\d(\.\d+)?(Z|[+-]([01]\d|2[0123]):[0-5]\d)$",
        "00:00:00.123456Z",
    ),
    (
        (
            r"""^([\w!#$%&'*+/=?^_`{|}~-]+(\.[\w!#$%&'*+/=?^_`{|}~-]+)*"""
            r"""|"([\w!#$%&'*+/=?^_`{|}~\-(),:;<>@[\].]|\\")+")@(([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+"""
            r"""[a-z0-9]([a-z0-9-]*[a-z0-9])?)$"""
        ),
        "customer/department=shipping@test.example.test-example.com",
    ),
]
tokenizer_path_regex_instance = [(t, *ri) for t in tokenizer_paths for ri in regex_instances]


@pytest.mark.hf_token_required
@pytest.mark.parametrize("tokenizer_path, regex, instance", tokenizer_path_regex_instance)
def test_mask_generation(tokenizer_path: str, regex: str, instance: str):
    print(f"Tokenizer: {tokenizer_path}, regex: {regex}, instance: {instance}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)

    time_start = time.monotonic_ns()
    matcher_compiled_grammar = grammar_compiler.compile_grammar(_regex_to_ebnf(regex))
    time_end = time.monotonic_ns()
    print(f"Time for preprocessing: {(time_end - time_start) / 1e3} us")
    matcher = xgr.GrammarMatcher(matcher_compiled_grammar)
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    for c in instance.encode("utf-8"):
        time_start = time.monotonic_ns()
        matcher.fill_next_token_bitmask(token_bitmask)
        time_end = time.monotonic_ns()
        print(f"Time for fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")
        accepted = matcher.accept_string(bytes([c]))
        assert accepted
        print(f"Accepting {c}")

    time_start = time.monotonic_ns()
    matcher.fill_next_token_bitmask(token_bitmask)
    time_end = time.monotonic_ns()
    print(f"Time for fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")

    assert matcher.accept_token(tokenizer.eos_token_id)
    assert matcher.is_terminated()


empty_regex = ["", "^$", "(())", "()", "^", "$", "()|()"]


@pytest.mark.parametrize("regex", empty_regex)
def test_empty(regex: str):
    grammar = xgr.Grammar.from_regex(regex)
    expected_grammar = 'root ::= ("")\n'
    assert str(grammar) == expected_grammar
    assert _is_grammar_accept_string(grammar, "")
    assert not _is_grammar_accept_string(grammar, "a")


if __name__ == "__main__":
    pytest.main(sys.argv)
