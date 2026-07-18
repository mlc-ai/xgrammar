from typing import List

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


def _assert_language(grammar: str, accepted: List[str], rejected: List[str]) -> None:
    compiled = xgr.Grammar.from_lark(grammar)
    for value in accepted:
        assert _is_grammar_accept_string(compiled, value), value
    for value in rejected:
        assert not _is_grammar_accept_string(compiled, value), value


@pytest.mark.parametrize(
    "grammar, accepted, rejected",
    [
        ('start: item{2,3}\nitem: "a" | "b"', ["aa", "ab", "bbb"], ["", "a", "aaaa"]),
        ('start: ["-"] DIGIT+\nDIGIT: "0".."9"', ["0", "123", "-42"], ["", "-", "+1", "1a"]),
        (
            'start: value ("," value)*\nvalue: "x" | "(" start ")"',
            ["x", "x,x", "(x,x)", "x,(x,x)"],
            ["", "x,", "(x", "y"],
        ),
        ('start: ab~2..4\nab: "a" | "b"', ["aa", "aba", "bbbb"], ["", "a", "aaaaa"]),
        ('start: foo-bar // comment\nfoo-bar: "ok" # another comment', ["ok"], ["foo-bar", ""]),
    ],
)
def test_lark_core_languages(grammar: str, accepted: List[str], rejected: List[str]) -> None:
    _assert_language(grammar, accepted, rejected)


def test_lark_import_ignore_and_initial_skip() -> None:
    grammar = """
        %import common.WS
        %ignore WS
        start: "a" "b"
    """
    _assert_language(grammar, ["ab", "a b", "a\n b  "], [" ab", "a c"])

    grammar_with_initial_skip = '%llguidance {"allow_initial_skip": true}\n' + grammar
    _assert_language(grammar_with_initial_skip, [" ab", "\n a b"], [" a c"])


@pytest.mark.parametrize(
    "grammar, accepted, rejected",
    [
        ('start: "a"? "b"* "c"+', ["c", "ac", "bbcc", "abbc"], ["", "a", "b"]),
        ('start: "x"{2}', ["xx"], ["", "x", "xxx"]),
        ('start: "x"{2,}', ["xx", "xxxx"], ["", "x"]),
        ('start: "x"{,2}', ["", "x", "xx"], ["xxx"]),
    ],
)
def test_lark_repetition_forms(grammar: str, accepted: List[str], rejected: List[str]) -> None:
    _assert_language(grammar, accepted, rejected)


def test_lark_multiline_choice_and_import_alias() -> None:
    grammar = """
        %import common.INT -> NUMBER
        start: "none"
             | NUMBER
    """
    _assert_language(grammar, ["none", "0", "123"], ["", "number", "1.2"])


def test_lark_inline_json_and_nested_lark() -> None:
    grammar = r"""
        start: "payload=" payload ";" %lark {
          start: "yes" | "no"
        }
        payload: %json {
          "type": "object",
          "properties": {"x": {"type": "integer"}},
          "required": ["x"],
          "additionalProperties": false
        }
    """
    _assert_language(
        grammar,
        ['payload={"x":1};yes', 'payload={ "x" : -2 };no'],
        ['payload={"x":"bad"};yes', 'payload= {"x":1};yes', 'payload={"x":1};maybe'],
    )


def test_lark_numeric_and_named_special_tokens() -> None:
    tokenizer_info = xgr.TokenizerInfo(
        ["a", "<|tool|>", "b", "</s>"], vocab_size=4, stop_token_ids=[3]
    )
    grammar = xgr.Grammar.from_lark("start: <[0,2]> | <|tool|>", tokenizer_info=tokenizer_info)
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)

    for token_id in [0, 1, 2]:
        matcher = xgr.GrammarMatcher(compiled)
        assert matcher.accept_token(token_id)
        assert matcher.is_completed()

    matcher = xgr.GrammarMatcher(compiled)
    assert not matcher.accept_token(3)


@pytest.mark.parametrize(
    "lark, accepted, rejected",
    [("start: <[^1,3]>", [0, 2], [1, 3]), ("start: <[*]>", [0, 1, 2, 3], [])],
)
def test_lark_special_token_exclusion_and_wildcard(
    lark: str, accepted: List[int], rejected: List[int]
) -> None:
    tokenizer_info = xgr.TokenizerInfo(["a", "b", "c", "d"])
    grammar = xgr.Grammar.from_lark(lark, tokenizer_info=tokenizer_info)
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)

    for token_id in accepted:
        matcher = xgr.GrammarMatcher(compiled)
        assert matcher.accept_token(token_id), token_id
        assert matcher.is_completed()
    for token_id in rejected:
        matcher = xgr.GrammarMatcher(compiled)
        assert not matcher.accept_token(token_id), token_id


TOOL_CALL_GRAMMAR = r"""
    start: tool* tail
    tail: TEXT

    tool_head[lazy]: TEXT "<tool_call>"
    tool: tool_head %json {
      "type": "object",
      "properties": {"x": {"type": "integer"}},
      "required": ["x"],
      "additionalProperties": false
    } "</tool_call>"

    TEXT: /(\n|.)*/
"""


def test_lark_dynamic_tool_call_optional_and_repeated() -> None:
    _assert_language(
        TOOL_CALL_GRAMMAR,
        [
            "",
            "plain text",
            '<tool_call>{"x":1}</tool_call>',
            'before<tool_call>{"x":1}</tool_call>after',
            '<tool_call>{"x":1}</tool_call><tool_call>{"x":2}</tool_call>',
            'line 1\nline 2<tool_call>{ "x" : -3 }</tool_call>tail',
            # A trigger prefix is still ordinary text until the full trigger is present.
            "text <tool_cal and more",
        ],
        [
            '<tool_call>{"x":"bad"}</tool_call>',
            '<tool_call>{"x":1}',
            "before<tool_call>free text</tool_call>after",
            '<tool_call> {"x":1}</tool_call>',
            '<tool_call>{"x":1} </tool_call>',
        ],
    )


def test_lark_dynamic_shared_trigger_dispatch() -> None:
    grammar = r"""
        start: (foo | bar)* tail
        tail: TEXT

        foo_head[lazy]: TEXT "<function"
        foo: foo_head "=foo>" /[a-z]+/ "</function>"

        bar_head[lazy]: TEXT "<function"
        bar: bar_head "=bar>" /[A-Z]+/ "</function>"

        TEXT: /(\n|.)*/
    """
    _assert_language(
        grammar,
        [
            "free text",
            "a<function=foo>abc</function>b",
            "<function=bar>ABC</function><function=foo>xyz</function>",
        ],
        [
            "<function=baz>abc</function>",
            "<function=foo>ABC</function>",
            "<function=bar>abc</function>",
            "<function=foo>abc",
        ],
    )


def test_lark_dynamic_special_token_trigger() -> None:
    tokenizer_info = xgr.TokenizerInfo(
        ["plain", "<|tool|>", "{", '"x"', ":", "1", "}", "</tool>", "bad", "</s>"],
        stop_token_ids=[9],
    )
    grammar = xgr.Grammar.from_lark(
        r"""
        start: tool* tail
        tail: TEXT
        tool: TEXT <|tool|> %json {
          "type": "object",
          "properties": {"x": {"const": 1}},
          "required": ["x"],
          "additionalProperties": false
        } "</tool>"
        TEXT: /(\n|.)*/
        """,
        tokenizer_info=tokenizer_info,
    )
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(grammar)

    matcher = xgr.GrammarMatcher(compiled)
    for token_id in [0, 1, 2, 3, 4, 5, 6, 7, 0]:
        assert matcher.accept_token(token_id), token_id

    invalid_matcher = xgr.GrammarMatcher(compiled)
    assert invalid_matcher.accept_token(1)
    assert not invalid_matcher.accept_token(8)


def test_lark_serialization_round_trip() -> None:
    grammar = xgr.Grammar.from_lark('start: "a" ("b" | "c")?')
    restored = xgr.Grammar.deserialize_json(grammar.serialize_json())
    for value in ["a", "ab", "ac"]:
        assert _is_grammar_accept_string(restored, value)
    assert not _is_grammar_accept_string(restored, "abc")


@pytest.mark.parametrize(
    "grammar, message",
    [
        ('item: "a"', "no start rule"),
        ("start: missing", "unknown name 'missing'"),
        ('start: foo\nfoo: "a"\nfoo: "b"', "duplicate rule or terminal 'foo'"),
        ("start: FOO\nFOO: BAR\nBAR: FOO", "circular reference in terminal"),
        ('start[capture]: "a"', "attribute 'capture' is not supported"),
        ("start: /abc/i", "regular-expression flags are not supported"),
        ('start: A & B\nA: "a"\nB: "b"', "intersection '&' is not supported"),
        ('start: %regex {"substring_chars":"abc"}', "structured %regex is not supported"),
        ("start: @other", "multiple grammar references are not supported"),
        ("start: foo::0", "parametric grammar is not supported"),
        ("start: <[1-2-3]>", "invalid numeric special-token range"),
        ("start: <[,]>", "empty numeric special-token range"),
        ('start: "unterminated', "unterminated string literal"),
    ],
)
def test_lark_errors_are_explicit_and_located(grammar: str, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        xgr.Grammar.from_lark(grammar)

    try:
        xgr.Grammar.from_lark(grammar)
    except RuntimeError as error:
        assert "line " in str(error)
        assert "column " in str(error)
