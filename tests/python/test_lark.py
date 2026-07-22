import json
from typing import Optional, Sequence

import pytest

import xgrammar as xgr


def _compile_lark(
    grammar: str, tokenizer_info: Optional[xgr.TokenizerInfo] = None
) -> xgr.CompiledGrammar:
    tokenizer_info = tokenizer_info or xgr.TokenizerInfo([])
    grammar_obj = xgr.Grammar.from_lark(grammar, tokenizer_info=tokenizer_info)
    return xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar_obj)


def _matches_string(compiled: xgr.CompiledGrammar, value: str) -> bool:
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    return matcher.accept_string(value) and matcher.is_terminated()


def _assert_grammar_language(
    grammar: xgr.Grammar,
    accepted: Sequence[str],
    rejected: Sequence[str],
    tokenizer_info: Optional[xgr.TokenizerInfo] = None,
) -> None:
    tokenizer_info = tokenizer_info or xgr.TokenizerInfo([])
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    for value in accepted:
        assert _matches_string(compiled, value), value
    for value in rejected:
        assert not _matches_string(compiled, value), value


def _assert_language(
    grammar: str,
    accepted: Sequence[str],
    rejected: Sequence[str],
    tokenizer_info: Optional[xgr.TokenizerInfo] = None,
) -> None:
    grammar_obj = xgr.Grammar.from_lark(grammar, tokenizer_info=tokenizer_info)
    _assert_grammar_language(grammar_obj, accepted, rejected, tokenizer_info)


def _matches_token_sequence(compiled: xgr.CompiledGrammar, token_ids: Sequence[int]) -> bool:
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for token_id in token_ids:
        if not matcher.accept_token(token_id):
            return False
    return matcher.is_terminated()


def _assert_token_language(
    grammar: str,
    tokenizer_info: xgr.TokenizerInfo,
    accepted: Sequence[Sequence[int]],
    rejected: Sequence[Sequence[int]],
) -> None:
    compiled = _compile_lark(grammar, tokenizer_info)
    for token_ids in accepted:
        assert _matches_token_sequence(compiled, token_ids), token_ids
    for token_ids in rejected:
        assert not _matches_token_sequence(compiled, token_ids), token_ids


def _assert_lark_error(
    grammar: str, message: str, tokenizer_info: Optional[xgr.TokenizerInfo] = None
) -> str:
    with pytest.raises(RuntimeError) as exc_info:
        xgr.Grammar.from_lark(grammar, tokenizer_info=tokenizer_info)
    error = str(exc_info.value)
    assert message in error
    assert "Lark error at line " in error
    assert ", column " in error
    return error


@pytest.mark.parametrize(
    "grammar, accepted, rejected",
    [
        pytest.param(
            'start: "a" "b" | "c" ("d" | "e")',
            ["ab", "cd", "ce"],
            ["", "a", "c", "ade"],
            id="sequence-choice-precedence",
        ),
        pytest.param(
            'start: ("a" | "b") ["c"]',
            ["a", "b", "ac", "bc"],
            ["", "c", "abc"],
            id="groups-and-optional-group",
        ),
        pytest.param(
            'start: | "a" | "b"', ["", "a", "b"], ["ab", "c"], id="empty-left-alternative"
        ),
        pytest.param('start: "a" |', ["", "a"], ["aa", "b"], id="empty-right-alternative"),
        pytest.param("start:", [""], ["a", " "], id="empty-rule"),
        pytest.param('start: "" "a" ""', ["a"], ["", "aa"], id="empty-literals"),
        pytest.param(
            '?item: "a"\n!suffix: "b"\nstart: item suffix',
            ["ab"],
            ["", "a", "b"],
            id="lark-rule-prefixes",
        ),
        pytest.param(
            'start: _item _TOKEN\n_item: "a"\n_TOKEN: "b"',
            ["ab"],
            ["a", "b", "_item_TOKEN"],
            id="hidden-rule-and-terminal-names",
        ),
        pytest.param(
            'start: "a" -> first\n     | "b" -> second',
            ["a", "b"],
            ["", "first", "second"],
            id="alternative-aliases",
        ),
        pytest.param(
            'start: value\nvalue: "x" | "(" value ")" | "[" values? "]"\nvalues: value ("," value)*',
            ["x", "(x)", "((x))", "[]", "[x]", "[x,(x),[x]]"],
            ["", "()", "[", "[x,]", "[(])"],
            id="forward-references-and-recursion",
        ),
        pytest.param(
            'start: sequence\nsequence: "x" | "a" sequence "b"',
            ["x", "axb", "aaxbb"],
            ["", "ab", "aaxb"],
            id="recursive-sequence",
        ),
        pytest.param(
            'start: SIGNED\nSIGNED: SIGN? DIGIT+\nSIGN: "+" | "-"\nDIGIT: "0".."9"',
            ["0", "123", "+7", "-42"],
            ["", "+", "--1", "1a"],
            id="terminal-composition",
        ),
        pytest.param(
            "start: \"'foo'\" /a+/ | STRING /b+/\nSTRING: /'[^']*'/",
            ["'foo'a", "'foo'aaa", "'bar'b", "'bar'bbb", "'foo'bb"],
            ["'bar'a", "'bar'c", "foo"],
            id="literal-terminal-ambiguity",
        ),
        pytest.param(
            'start: /.../ "abc" /.../',
            ["abcabcabc", "aaaabcccc", "🔵🟠✅abc❌🟠🔵"],
            ["aaabcccc", "aaaaabcccc", "🔵🟠abc🟠🔵"],
            id="regex-dot-counts-unicode-codepoints",
        ),
        pytest.param(
            r"start: /a\/b/ /[0-9]{2,4}/",
            ["a/b12", "a/b1234"],
            ["a/b1", "a/b12345", "a-b12"],
            id="regex-escaped-delimiter-and-repeat",
        ),
        pytest.param(
            "start: /a.b/",
            ["acb", "a b", "a😀b"],
            ["ab", "a\nb", "a\n\nb"],
            id="regex-dot-excludes-newline",
        ),
        pytest.param(
            "start: /a.b/s", ["acb", "a\nb", "a😀b"], ["ab", "a\n\nb"], id="regex-dotall-flag"
        ),
        pytest.param(
            r"start: /a\.b/s", ["a.b"], ["acb", "a\nb"], id="regex-dotall-preserves-escaped-dot"
        ),
        pytest.param(
            "start: /a[.]b/s",
            ["a.b"],
            ["acb", "a\nb"],
            id="regex-dotall-preserves-character-class-dot",
        ),
        pytest.param(
            'start: "a".."z"+ "0".."9"?',
            ["a", "xyz", "hello7"],
            ["", "A", "7", "abc78"],
            id="ascii-character-ranges",
        ),
        pytest.param(
            'start: "α".."γ"+', ["α", "βγ", "γα"], ["", "δ", "a"], id="unicode-character-ranges"
        ),
        pytest.param(
            'start: "😀" "é" "中文"',
            ["😀é中文"],
            ["", "😀é", "😀e中文"],
            id="unicode-string-literals",
        ),
        pytest.param(
            'start: "Ab-C9"i',
            ["Ab-C9", "ab-c9", "AB-C9", "aB-c9"],
            ["", "ab-c", "ab_c9", "äb-c9"],
            id="ascii-case-insensitive-string",
        ),
        pytest.param(
            'start: TOKEN\nTOKEN: "Yes"i | "no"',
            ["yes", "YES", "YeS", "no"],
            ["", "y", "No"],
            id="case-insensitive-string-in-terminal",
        ),
        pytest.param(
            r'''start: "\n" "\t" "\\" "\"" "\u03bb"''',
            ['\n\t\\"λ'],
            [r"\n\t\"λ", "\n\tλ"],
            id="json-style-string-escapes",
        ),
        pytest.param(
            r'''start: "\b" "\f" "\r"''',
            ["\b\f\r"],
            ["bfr", "\b\f\n"],
            id="control-character-string-escapes",
        ),
        pytest.param(
            'start: foo-bar FOO-BAR\nfoo-bar: "a"\nFOO-BAR: "b"',
            ["ab"],
            ["", "a-b", "foo-barFOO-BAR"],
            id="hyphenated-identifiers",
        ),
        pytest.param(
            'start: item // top-level comment\nitem: "ok" # rule comment',
            ["ok"],
            ["", "item", "ok#"],
            id="comment-styles",
        ),
    ],
)
def test_lark_core_languages(
    grammar: str, accepted: Sequence[str], rejected: Sequence[str]
) -> None:
    _assert_language(grammar, accepted, rejected)


@pytest.mark.parametrize(
    "grammar, accepted, rejected",
    [
        pytest.param('start: "a"?', ["", "a"], ["aa"], id="question"),
        pytest.param('start: "a"*', ["", "a", "aaaa"], ["b", "aaab"], id="star"),
        pytest.param('start: "a"+', ["a", "aaaa"], ["", "b"], id="plus"),
        pytest.param('start: "a"~2', ["aa"], ["", "a", "aaa"], id="tilde-exact"),
        pytest.param(
            'start: "a"~2..4', ["aa", "aaa", "aaaa"], ["", "a", "aaaaa"], id="tilde-range"
        ),
        pytest.param('start: "a"{2}', ["aa"], ["", "a", "aaa"], id="brace-exact"),
        pytest.param(
            'start: "a"{2,4}', ["aa", "aaa", "aaaa"], ["", "a", "aaaaa"], id="brace-range"
        ),
        pytest.param('start: "a"{2,}', ["aa", "aaaaaa"], ["", "a"], id="brace-open-end"),
        pytest.param('start: "a"{,2}', ["", "a", "aa"], ["aaa"], id="brace-open-start"),
        pytest.param('start: "a"{0}', [""], ["a"], id="zero-exact"),
        pytest.param('start: "a"{0,0}', [""], ["a"], id="zero-range"),
        pytest.param(
            'start: ("a" | "bc"){2,3}',
            ["aa", "abc", "bca", "bcbcbc"],
            ["", "a", "bcbc bcbc", "aaaa"],
            id="group-repeat",
        ),
        pytest.param(
            'start: ITEM{2,3}\nITEM: "x" | "y"',
            ["xx", "xy", "yyy"],
            ["", "x", "xxxx"],
            id="terminal-repeat",
        ),
        pytest.param(
            'start: item{2,3}\nitem: "x" | "(" item ")"',
            ["xx", "x(x)", "(x)(x)(x)"],
            ["", "x", "xxxx"],
            id="recursive-rule-repeat",
        ),
    ],
)
def test_lark_repetition_forms(
    grammar: str, accepted: Sequence[str], rejected: Sequence[str]
) -> None:
    _assert_language(grammar, accepted, rejected)


@pytest.mark.parametrize(
    "common_name, accepted, rejected",
    [
        pytest.param("DIGIT", ["0", "9"], ["", "a", "10"], id="DIGIT"),
        pytest.param("HEXDIGIT", ["0", "a", "F"], ["", "g", "ff"], id="HEXDIGIT"),
        pytest.param("INT", ["0", "123"], ["", "-1", "1.0"], id="INT"),
        pytest.param("SIGNED_INT", ["0", "+12", "-3"], ["", "+", "1.0"], id="SIGNED_INT"),
        pytest.param("DECIMAL", ["1.", "1.5", ".25"], ["", "1", "."], id="DECIMAL"),
        pytest.param("_EXP", ["e1", "E+12", "e-3"], ["", "1", "e"], id="_EXP"),
        pytest.param("FLOAT", ["1.", ".5", "1e3", "1.2e-3"], ["", "1", "e3"], id="FLOAT"),
        pytest.param(
            "SIGNED_FLOAT", ["-1.", "+.5", "-1e3", "1.2e-3"], ["", "1", "+"], id="SIGNED_FLOAT"
        ),
        pytest.param("NUMBER", ["0", "12", ".5", "1e3"], ["", "-1", "x"], id="NUMBER"),
        pytest.param(
            "SIGNED_NUMBER", ["0", "-12", "+.5", "-1e3"], ["", "+", "x"], id="SIGNED_NUMBER"
        ),
        pytest.param(
            "ESCAPED_STRING",
            ['""', '"abc"', '"a\\"b"', '"a\\nb"'],
            ["", "abc", '"unterminated'],
            id="ESCAPED_STRING",
        ),
        pytest.param("LCASE_LETTER", ["a", "z"], ["", "A", "aa"], id="LCASE_LETTER"),
        pytest.param("UCASE_LETTER", ["A", "Z"], ["", "a", "AA"], id="UCASE_LETTER"),
        pytest.param("LETTER", ["a", "Z"], ["", "1", "ab"], id="LETTER"),
        pytest.param("WORD", ["a", "AbCd"], ["", "a1", "a_b"], id="WORD"),
        pytest.param("CNAME", ["a", "_a1", "A_2"], ["", "1a", "a-b"], id="CNAME"),
        pytest.param("WS_INLINE", [" ", "\t", " \t "], ["", "\n", "a"], id="WS_INLINE"),
        pytest.param("WS", [" ", "\n", "\t\f\r\n"], ["", "a"], id="WS"),
        pytest.param("CR", ["\r"], ["", "\n", "\r\n"], id="CR"),
        pytest.param("LF", ["\n"], ["", "\r", "\r\n"], id="LF"),
        pytest.param("NEWLINE", ["\n", "\r\n", "\r\n\n"], ["", "\r", "a"], id="NEWLINE"),
        pytest.param("SH_COMMENT", ["#", "# hello"], ["", "// hello", "# a\n"], id="SH_COMMENT"),
        pytest.param(
            "CPP_COMMENT", ["//", "// hello"], ["", "# hello", "// a\n"], id="CPP_COMMENT"
        ),
        pytest.param(
            "C_COMMENT",
            ["/**/", "/* hello */", "/* ** x **/"],
            ["", "// hello", "/* open"],
            id="C_COMMENT",
        ),
        pytest.param(
            "SQL_COMMENT", ["--", "-- hello"], ["", "- hello", "-- a\n"], id="SQL_COMMENT"
        ),
    ],
)
def test_lark_common_imports(
    common_name: str, accepted: Sequence[str], rejected: Sequence[str]
) -> None:
    _assert_language(f"%import common.{common_name}\nstart: {common_name}", accepted, rejected)


def test_lark_multi_import_alias_and_forward_import() -> None:
    grammar = """
        %ignore WS_INLINE
        start: NAME "=" NUMBER
        %import common (CNAME, WS_INLINE)
        %import common.INT -> NUMBER
        NAME: CNAME
    """
    _assert_language(grammar, ["x=1", "name = 42", "_x\t=\t0"], ["", "1x=2", "x=-1"])


def test_lark_ignore_is_inserted_between_and_after_lexemes() -> None:
    grammar = """
        %import common.WS
        %ignore WS
        start: "a" DIGIT "c".."d"
        DIGIT: "0".."9"
    """
    _assert_language(grammar, ["a1c", "a 1 d", "a\n1\n c  "], [" a1c", "a 1 e", "a x c"])


def test_lark_multiple_ignore_declarations() -> None:
    grammar = """
        %import common (WS, CPP_COMMENT, SH_COMMENT)
        %ignore WS
        %ignore CPP_COMMENT
        %ignore SH_COMMENT
        start: "a" "b"
    """
    _assert_language(
        grammar,
        ["ab", "a // comment\n b", "a# comment\nb", "a b // trailing"],
        [" // initial\na b", "a x b"],
    )


def test_lark_ignore_inline_regex() -> None:
    grammar = r"""
        %ignore /[ _]+/
        start: "a" "b"
    """
    _assert_language(grammar, ["ab", "a_b", "a _ b___"], ["_ab", "a-b"])


def test_lark_allow_initial_skip_options() -> None:
    grammar = """
        %grammar_options {"allow_initial_skip": false}
        %grammar_options {"allow_initial_skip": true, "no_forcing": false, "allow_invalid_utf8": false}
        %import common.WS
        %ignore WS
        start: "a" "b"
    """
    _assert_language(grammar, ["ab", " a b", "\n\ta\n b  "], ["a c", " xab"])


def test_lark_default_disallows_initial_skip_but_allows_trailing_skip() -> None:
    grammar = """
        %import common.WS_INLINE
        %ignore WS_INLINE
        start: "a" "b"
    """
    _assert_language(grammar, ["ab", "a b", "ab  "], [" ab", "a\nb"])


@pytest.mark.parametrize(
    "schema, accepted, rejected",
    [
        pytest.param(
            '{"type":"string"}',
            ['""', '"hello"', '"λ"'],
            ["hello", "1", '"unterminated'],
            id="string",
        ),
        pytest.param('{"type":"integer"}', ["0", "-12", "123"], ["1.0", '"1"', "+1"], id="integer"),
        pytest.param('{"const":"fixed"}', ['"fixed"'], ['"other"', "fixed"], id="const"),
        pytest.param(
            '{"enum":["red","green",3]}', ['"red"', '"green"', "3"], ['"blue"', "4"], id="enum"
        ),
        pytest.param(
            '{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":3}',
            ["[1]", "[1,2]", "[ 1, 2, 3 ]"],
            ["[]", "[1,2,3,4]", '["1"]'],
            id="array",
        ),
        pytest.param(
            '{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}',
            ['{"x":1}', '{ "x" : -2 }'],
            ["{}", '{"x":"1"}', '{"x":1,"y":2}'],
            id="object",
        ),
        pytest.param(
            '{"anyOf":[{"type":"integer"},{"type":"boolean"}]}',
            ["1", "-2", "true", "false"],
            ['"1"', "null"],
            id="any-of",
        ),
        pytest.param(
            '{"type":"string","pattern":"^a[0-9]+$"}',
            ['"a0"', '"a123"'],
            ['"a"', '"ba1"', '"a1x"'],
            id="string-pattern",
        ),
    ],
)
def test_lark_inline_json_schemas(
    schema: str, accepted: Sequence[str], rejected: Sequence[str]
) -> None:
    _assert_language(f"start: %json {schema}", accepted, rejected)


def test_lark_inline_json_inside_sequence_and_repeat() -> None:
    grammar = r"""
        start: "values=" value (";" value)* "."
        value: %json {"type":"integer"}
    """
    _assert_language(
        grammar, ["values=1.", "values=1;-2;3."], ["values=.", 'values="1".', "values=1;."]
    )


def test_lark_nested_lark_has_an_independent_namespace() -> None:
    grammar = """
        start: item %lark {
          start: item
          item: "b"
        } item
        item: "a"
    """
    _assert_language(grammar, ["aba"], ["aaa", "abb", "bbb"])


def test_lark_nested_lark_supports_recursion_json_and_ignore() -> None:
    grammar = r"""
        start: "[" %lark {
          %grammar_options {"allow_initial_skip": true}
          %import common.WS
          %ignore WS
          start: item ":" %json {"type":"integer"}
          item: "x" | "(" item ")"
        } "]"
    """
    _assert_language(grammar, ["[x:1]", "[ ((x)) : -2 ]"], ["[():1]", '[x:"1"]', "[x 1]"])


def test_lark_multiple_nested_grammars() -> None:
    grammar = """
        start: %lark { start: "a" | "b" } %lark { start: "1" | "2" }
    """
    _assert_language(grammar, ["a1", "a2", "b1", "b2"], ["", "a", "1a", "c1"])


def test_lark_numeric_and_named_special_tokens() -> None:
    tokenizer_info = xgr.TokenizerInfo(["a", "<|tool|>", "b", "</s>"], stop_token_ids=[3])
    _assert_token_language(
        "start: <[0,2]> | <|tool|>",
        tokenizer_info,
        accepted=[[0], [1], [2]],
        rejected=[[3], [0, 1], []],
    )


@pytest.mark.parametrize(
    "grammar, accepted, rejected",
    [
        pytest.param("start: <[0-2,1-3,3]>", [[0], [1], [2], [3]], [[4]], id="merged-ranges"),
        pytest.param("start: <[^1,3]>", [[0], [2], [4]], [[1], [3]], id="excluded-set"),
        pytest.param("start: <[*]>", [[0], [1], [2], [3], [4]], [], id="wildcard"),
        pytest.param("start: <[0]> <[2-3]>", [[0, 2], [0, 3]], [[0], [2, 0]], id="token-sequence"),
    ],
)
def test_lark_numeric_special_token_sets(
    grammar: str, accepted: Sequence[Sequence[int]], rejected: Sequence[Sequence[int]]
) -> None:
    tokenizer_info = xgr.TokenizerInfo(["a", "b", "c", "d", "e"])
    _assert_token_language(grammar, tokenizer_info, accepted, rejected)


def test_lark_named_special_token_matches_every_exact_vocab_entry() -> None:
    tokenizer_info = xgr.TokenizerInfo(["<dup>", "x", "<dup>", "dup", "<other>"])
    _assert_token_language(
        "start: <dup>", tokenizer_info, accepted=[[0], [2]], rejected=[[1], [3], [4]]
    )


def test_lark_special_token_and_literal_sequence() -> None:
    tokenizer_info = xgr.TokenizerInfo(["<|tool|>", "x", "y"])
    _assert_token_language(
        'start: <|tool|> "x"', tokenizer_info, accepted=[[0, 1]], rejected=[[0], [0, 2], [1]]
    )


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


def test_lark_dynamic_tool_call_optional_repeated_and_committed() -> None:
    _assert_language(
        TOOL_CALL_GRAMMAR,
        [
            "",
            "plain text",
            "text <tool_cal and more",
            '<tool_call>{"x":1}</tool_call>',
            'before<tool_call>{"x":1}</tool_call>after',
            '<tool_call>{"x":1}</tool_call><tool_call>{"x":2}</tool_call>',
            'line 1\nline 2<tool_call>{ "x" : -3 }</tool_call>tail',
        ],
        [
            "<tool_call>",
            '<tool_call>{"x":"bad"}</tool_call>',
            '<tool_call>{"x":1}',
            "before<tool_call>free text</tool_call>after",
            '<tool_call> {"x":1}</tool_call>',
            '<tool_call>{"x":1} </tool_call>',
        ],
    )


def test_lark_dynamic_distinct_string_triggers() -> None:
    grammar = r"""
        start: (foo | bar)* tail
        tail: TEXT

        foo_head[lazy]: TEXT "<foo>"
        foo: foo_head /[a-z]+/ "</foo>"

        bar_head[lazy]: TEXT "<bar>"
        bar: bar_head /[0-9]+/ "</bar>"

        TEXT: /(\n|.)*/
    """
    _assert_language(
        grammar,
        ["free text", "partial <fo remains text", "x<foo>abc</foo>y", "<bar>12</bar><foo>x</foo>"],
        ["<foo>", "<foo>12</foo>", "<bar>x</bar>", "<bar>12"],
    )


def test_lark_dynamic_lazy_regex_suffix() -> None:
    grammar = r"""
        start: tool* tail
        tail: TEXT
        head[lazy]: /(\n|.)*<call>/
        tool: head /[0-9]+/ "</call>"
        TEXT: /(\n|.)*/
    """
    _assert_language(
        grammar,
        ["", "free text", "partial <cal", "x<call>12</call>y"],
        ["<call>", "<call>x</call>", "<call>12", "x<call>12</call><call>"],
    )


def test_lark_dynamic_lazy_dotall_regex_suffix() -> None:
    grammar = r"""
        start: tool* tail
        tail: TEXT
        head[lazy]: /.*<call>/s
        tool: head "ok" "</call>"
        TEXT: /.*/s
    """
    _assert_language(
        grammar,
        ["line 1\nline 2", "x\n<call>ok</call>tail"],
        ["<call>", "<call>bad</call>", "x\n<call>ok"],
    )


def test_lark_dynamic_fixed_string_suffix_attribute() -> None:
    grammar = r"""
        start: tool* tail
        tail: TEXT
        head[suffix="<tool>"]: TEXT
        tool: head /[a-z]+/ "</tool>"
        TEXT: /(\n|.)*/
    """
    _assert_language(
        grammar,
        ["free", "x<tool>abc</tool>y", "partial <too"],
        ["<tool>", "<tool>123</tool>", "<tool>abc"],
    )


def test_lark_dynamic_lazy_regex_escaped_newline_trigger() -> None:
    grammar = r"""
        start: tool* tail
        tail: TEXT
        head[lazy]: /(\n|.)*\n>>>>>>>/
        tool: head "replacement"
        TEXT: /(\n|.)*/
    """
    _assert_language(
        grammar,
        ["free >>>>>>> text", "before\n>>>>>>>replacementafter"],
        ["\n>>>>>>>", "before\n>>>>>>>wrong"],
    )


def test_lark_dynamic_lazy_regex_escaped_metacharacter_trigger() -> None:
    grammar = r"""
        start: tool* tail
        tail: TEXT
        head[lazy]: /(\n|.)*END\./
        tool: head "ok"
        TEXT: /(\n|.)*/
    """
    _assert_language(
        grammar,
        ["free END text", "beforeEND.okafter"],
        ["END.", "beforeEND.not-ok", "beforeEND.okEND."],
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
            "<function",
            "<function=baz>abc</function>",
            "<function=foo>ABC</function>",
            "<function=bar>abc</function>",
            "<function=foo>abc",
        ],
    )


def test_lark_dynamic_any_text_can_be_referenced_through_terminals() -> None:
    grammar = r"""
        start: tool* tail
        tail: FREE
        head[lazy]: FREE "<call>"
        tool: head /[0-9]+/ "</call>"
        FREE: TEXT
        TEXT: /(\n|.)*/
    """
    _assert_language(
        grammar,
        ["free", "x<call>12</call>y", "partial <cal"],
        ["<call>", "<call>x</call>", "<call>12"],
    )


def test_lark_standalone_lazy_rule() -> None:
    grammar = r"""
        start: head
        head[lazy]: TEXT "<end>"
        TEXT: /(\n|.)*/
    """
    _assert_language(grammar, ["", "plain", "<end>", "plain<end>"], ["<end>x", "a<end>b"])


def test_lark_dynamic_special_token_trigger() -> None:
    tokenizer_info = xgr.TokenizerInfo(
        ["plain", "<|tool|>", "{", '"x"', ":", "1", "}", "</tool>", "bad", "</s>"],
        stop_token_ids=[9],
    )
    grammar = r"""
        start: tool* tail
        tail: TEXT
        tool: TEXT <|tool|> %json {
          "type": "object",
          "properties": {"x": {"const": 1}},
          "required": ["x"],
          "additionalProperties": false
        } "</tool>"
        TEXT: /(\n|.)*/
    """
    _assert_token_language(
        grammar,
        tokenizer_info,
        accepted=[[0], [1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7, 0]],
        rejected=[[1], [1, 8], [1, 2, 3, 4, 8]],
    )


def test_lark_serialization_round_trip_for_core_and_dynamic_grammars() -> None:
    core = xgr.Grammar.from_lark('start: "a" ("b" | "c")?')
    restored_core = xgr.Grammar.deserialize_json(core.serialize_json())
    _assert_grammar_language(restored_core, ["a", "ab", "ac"], ["", "abc"])

    dynamic = xgr.Grammar.from_lark(TOOL_CALL_GRAMMAR)
    restored_dynamic = xgr.Grammar.deserialize_json(dynamic.serialize_json())
    _assert_grammar_language(
        restored_dynamic,
        ["text", '<tool_call>{"x":1}</tool_call>tail'],
        ["<tool_call>", '<tool_call>{"x":"bad"}</tool_call>'],
    )


def test_lark_serialization_round_trip_for_token_dispatch() -> None:
    tokenizer_info = xgr.TokenizerInfo(["plain", "<|call|>", "x", "</call>"])
    grammar = xgr.Grammar.from_lark(
        r"""
        start: call* tail
        tail: TEXT
        call: TEXT <|call|> "x" "</call>"
        TEXT: /(\n|.)*/
        """,
        tokenizer_info=tokenizer_info,
    )
    restored = xgr.Grammar.deserialize_json(grammar.serialize_json())
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(restored)
    assert _matches_token_sequence(compiled, [0])
    assert _matches_token_sequence(compiled, [1, 2, 3])
    assert not _matches_token_sequence(compiled, [1, 3])


def test_lark_grammar_union_and_concat_integration() -> None:
    left = xgr.Grammar.from_lark('start: "a" | "b"')
    right = xgr.Grammar.from_lark("start: /[0-9]+/")

    _assert_grammar_language(xgr.Grammar.union(left, right), ["a", "b", "0", "123"], ["a1", "c"])
    _assert_grammar_language(xgr.Grammar.concat(left, right), ["a0", "b123"], ["a", "1", "c1"])


def test_lark_named_grammar_references() -> None:
    item = xgr.Grammar.from_lark('start: "x" | "y"')
    grammar = xgr.Grammar.from_lark(
        'start: "[" @item ("," @item)* "]"', named_grammars={"item": item}
    )
    _assert_grammar_language(grammar, ["[x]", "[y,x]", "[x,y,x]"], ["[]", "[z]", "[x,]"])


def test_lark_named_grammar_reference_in_nested_lark() -> None:
    value = xgr.Grammar.from_regex("[0-9]+")
    grammar = xgr.Grammar.from_lark(
        'start: "outer:" %lark { start: "inner:" @value }', named_grammars={"value": value}
    )
    _assert_grammar_language(grammar, ["outer:inner:0", "outer:inner:123"], ["inner:1", "outer:1"])


def test_lark_named_grammar_string_references() -> None:
    grammar = xgr.Grammar.from_lark(
        "start: @pair", named_grammars={"pair": 'start: @item ":" @item', "item": "start: /[a-z]+/"}
    )
    _assert_grammar_language(grammar, ["a:b", "hello:world"], ["a:", ":b", "a:1"])


def test_lark_named_grammar_string_can_reference_grammar_object() -> None:
    item = xgr.Grammar.from_regex("[0-9]+")
    grammar = xgr.Grammar.from_lark(
        "start: @wrapper", named_grammars={"wrapper": 'start: "[" @item "]"', "item": item}
    )
    _assert_grammar_language(grammar, ["[0]", "[123]"], ["[]", "[x]"])


def test_lark_named_grammar_string_cycle() -> None:
    with pytest.raises(
        RuntimeError, match=r"circular named grammar reference: @left -> @right -> @left"
    ):
        xgr.Grammar.from_lark(
            "start: @left", named_grammars={"left": "start: @right", "right": "start: @left"}
        )


def test_lark_named_grammar_argument_validation() -> None:
    item = xgr.Grammar.from_lark('start: "x"')
    with pytest.raises(TypeError, match="must be a dictionary"):
        xgr.Grammar.from_lark("start: @item", named_grammars=[item])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="names must be strings"):
        xgr.Grammar.from_lark("start: @item", named_grammars={1: item})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="values must be Grammar instances or Lark strings"):
        xgr.Grammar.from_lark("start: @item", named_grammars={"item": 1})  # type: ignore[dict-item]
    with pytest.raises(RuntimeError, match="Invalid named grammar name"):
        xgr.Grammar.from_lark("start: @item", named_grammars={"bad name": item})


def test_lark_large_choice_grammar() -> None:
    options = [f"option-{index:03d}" for index in range(256)]
    choices = " | ".join(json.dumps(option) for option in options)
    grammar = f"start: option\noption: {choices}"
    _assert_language(
        grammar, [options[0], options[127], options[-1]], ["", "option-256", "option-12"]
    )


@pytest.mark.parametrize(
    "grammar, message",
    [
        pytest.param('item: "a"', "no start rule", id="missing-start"),
        pytest.param("start: missing", "unknown name 'missing'", id="unknown-rule"),
        pytest.param(
            'start: foo\nfoo: "a"\nfoo: "b"',
            "duplicate rule or terminal 'foo'",
            id="duplicate-rule",
        ),
        pytest.param(
            'start: FOO\nFOO: "a"\nFOO: "b"',
            "duplicate rule or terminal 'FOO'",
            id="duplicate-terminal",
        ),
        pytest.param(
            "start: FOO\nFOO: BAR\nBAR: FOO", "circular reference in terminal", id="terminal-cycle"
        ),
        pytest.param(
            'start: TOKEN\nTOKEN: rule\nrule: "a"',
            "terminal 'TOKEN' cannot reference rule 'rule'",
            id="terminal-references-rule",
        ),
        pytest.param(
            'start[budget=10]: "a"', "attribute 'budget' is not supported", id="unknown-attribute"
        ),
        pytest.param(
            'start[stop="x"]: "a"', "attribute 'stop' is not supported", id="stop-attribute"
        ),
        pytest.param(
            'start[capture=""]: "a"', "capture name must not be empty", id="empty-capture-name"
        ),
        pytest.param(
            'start[capture="a b"]: "a"',
            "capture name must only contain letters, digits",
            id="invalid-capture-name",
        ),
        pytest.param(
            'start[capture, capture]: "a"',
            "capture attribute is specified more than once",
            id="duplicate-capture-attribute",
        ),
        pytest.param(
            "TOKEN[lazy]: /a/\nstart: TOKEN",
            "attributes are only supported on rules",
            id="terminal-attribute",
        ),
        pytest.param('start.1: "a"', "priorities are not supported", id="priority"),
        pytest.param('start{x}: "a"', "Lark templates are not supported", id="template-definition"),
        pytest.param(
            'start: foo{x}\nfoo: "a"', "Lark templates are not supported", id="template-reference"
        ),
        pytest.param(
            'start::0: "a"', "parametric grammar is not supported", id="parametric-definition"
        ),
        pytest.param(
            "start: foo::0", "parametric grammar is not supported", id="parametric-reference"
        ),
        pytest.param(
            'start: "a" %if enabled',
            "parametric %if conditions are not supported",
            id="parametric-if",
        ),
        pytest.param(
            'start: A & B\nA: "a"\nB: "b"', "intersection '&' is not supported", id="intersection"
        ),
        pytest.param("start: ~/[a]/", "complement '~' is not supported", id="complement"),
        pytest.param(
            'start: "\\u00c4"i',
            "case-insensitive string literals currently support ASCII characters only",
            id="non-ascii-string-flag",
        ),
        pytest.param(
            "start: /abc/i",
            "only the regular-expression flag 's' is currently supported",
            id="unsupported-regex-flag",
        ),
        pytest.param(
            "start: /abc/l", "regular-expression flag 'l' is not supported", id="unsupported-l-flag"
        ),
        pytest.param(
            'start: "a"i.."z"', "flags are not allowed on character ranges", id="range-start-flag"
        ),
        pytest.param(
            'start: "a".."z"i', "flags are not allowed on character ranges", id="range-end-flag"
        ),
        pytest.param("start: /[abc/", "failed to compile regular expression", id="invalid-regex"),
        pytest.param('start: "\\q"', "invalid string literal", id="invalid-string-escape"),
        pytest.param(
            'start: "unterminated', "unterminated string literal", id="unterminated-string"
        ),
        pytest.param(
            "start: /unterminated", "unterminated regular expression", id="unterminated-regex"
        ),
        pytest.param(
            "start: <unterminated", "unterminated special token", id="unterminated-special-token"
        ),
        pytest.param("start: <bad token>", "invalid special token", id="special-token-whitespace"),
        pytest.param("start: @", "empty grammar reference", id="empty-grammar-reference"),
        pytest.param("start: $", "unexpected character '$'", id="unexpected-character"),
        pytest.param("start: -", "unexpected '-' character", id="unexpected-minus"),
        pytest.param("start: !", "unexpected '!' character", id="unexpected-bang"),
        pytest.param('start "a"', "expected ':' after rule name", id="missing-colon"),
        pytest.param('start: ("a"', "expected ')' after group", id="unclosed-group"),
        pytest.param('start: ["a"', "expected ']' after optional group", id="unclosed-optional"),
        pytest.param('start: "a" ->', "expected alias name after '->'", id="missing-alias-name"),
        pytest.param(
            "%declare TOKEN\nstart: TOKEN",
            "directive %declare is not supported",
            id="declare-directive",
        ),
        pytest.param(
            '%override start\nstart: "a"',
            "directive %override is not supported",
            id="override-directive",
        ),
        pytest.param(
            "%import common.UNKNOWN\nstart: UNKNOWN",
            "unknown common import",
            id="unknown-common-import",
        ),
        pytest.param(
            '%import common.INT\nINT: "x"\nstart: INT',
            "duplicate rule or terminal 'INT'",
            id="import-name-conflict",
        ),
        pytest.param('start: "a"{3,2}', "repetition end must be greater", id="repetition-reversed"),
        pytest.param('start: "a"{-1,}', "invalid repetition count", id="negative-repetition"),
        pytest.param(
            'start: "a"{999999999999999999999}',
            "invalid repetition count",
            id="repetition-overflow",
        ),
        pytest.param(
            'start: "ab".."c"', "range endpoints must be one character", id="range-start-too-long"
        ),
        pytest.param(
            'start: "a".."bc"', "range endpoints must be one character", id="range-end-too-long"
        ),
        pytest.param('start: "z".."a"', "range start must not exceed end", id="range-reversed"),
        pytest.param(
            "start: %json {",
            "failed to parse JSON value after %json",
            id="malformed-json-directive",
        ),
        pytest.param(
            "start: %json []", "failed to compile inline JSON schema", id="invalid-json-schema"
        ),
        pytest.param(
            'start: %lark { item: "a" }',
            "failed to compile nested Lark grammar",
            id="nested-no-start",
        ),
        pytest.param(
            'start: %regex {"substring_chars":"abc"}',
            "structured %regex is not supported",
            id="structured-regex",
        ),
        pytest.param("start: @other", "unknown named grammar '@other'", id="unknown-named-grammar"),
        pytest.param(
            "start: TOKEN\nTOKEN: <[1]>",
            "special tokens cannot be used in terminals",
            id="special-in-terminal",
        ),
        pytest.param(
            "start: TOKEN\nTOKEN: %json {}",
            "%json cannot be used in terminals",
            id="json-in-terminal",
        ),
        pytest.param(
            'start: TOKEN\nTOKEN: %lark { start: "a" }',
            "nested %lark cannot be used in terminals",
            id="nested-lark-in-terminal",
        ),
        pytest.param(
            "start: <[1-2-3]>", "invalid numeric special-token range", id="multiple-range-dashes"
        ),
        pytest.param(
            "start: <[3-1]>", "invalid numeric special-token range", id="numeric-range-reversed"
        ),
        pytest.param("start: <[,]>", "empty numeric special-token range", id="empty-token-range"),
        pytest.param(
            "start: <[*]>",
            "wildcard special token requires tokenizer_info",
            id="wildcard-needs-tokenizer",
        ),
        pytest.param(
            "start: <[^*]>",
            "negated wildcard special token is not supported",
            id="negated-wildcard",
        ),
        pytest.param("start: <[*,1]>", "wildcard cannot be mixed", id="mixed-wildcard-range"),
        pytest.param(
            "start: <|tool|>",
            "named special token <|tool|> requires tokenizer_info",
            id="named-needs-tokenizer",
        ),
        pytest.param(
            "start: <[0-1000001]>", "special-token range is too large", id="token-range-too-large"
        ),
        pytest.param(
            '%grammar_options {"allow_initial_skip": 1}\nstart: "a"',
            "allow_initial_skip must be a boolean",
            id="initial-skip-type",
        ),
        pytest.param(
            '%grammar_options {"no_forcing": true}\nstart: "a"',
            "%grammar_options option 'no_forcing' is not supported",
            id="no-forcing-option",
        ),
        pytest.param(
            '%grammar_options {"allow_invalid_utf8": true}\nstart: "a"',
            "%grammar_options option 'allow_invalid_utf8' is not supported",
            id="invalid-utf8-option",
        ),
        pytest.param(
            '%grammar_options {"unknown": false}\nstart: "a"',
            "unknown %grammar_options option 'unknown'",
            id="unknown-grammar-options-option",
        ),
        pytest.param(
            '%grammar_options []\nstart: "a"',
            "%grammar_options value must be an object",
            id="grammar-options-not-object",
        ),
        pytest.param(
            'start: thing\nthing[lazy]: "a" thing | "b"',
            "terminal cannot reference rule",
            id="lazy-rule-reference",
        ),
        pytest.param(
            'start: head\nhead[suffix=""]: TEXT\nTEXT: /(\\n|.)*/',
            "suffix must not be empty",
            id="empty-suffix",
        ),
        pytest.param(
            "start: head\nhead[suffix=/x/]: TEXT\nTEXT: /(\\n|.)*/",
            "expected string literal after suffix=",
            id="non-string-suffix",
        ),
        pytest.param(
            'start: head\nhead[suffix="x",suffix="y"]: TEXT\nTEXT: /(\\n|.)*/',
            "suffix attribute is specified more than once",
            id="duplicate-suffix",
        ),
        pytest.param(
            'start: head\nhead[suffix="x"i]: TEXT\nTEXT: /(\\n|.)*/',
            "case-insensitive flags are not supported on suffix",
            id="case-insensitive-suffix",
        ),
        pytest.param(
            'start: head\nhead[suffix="x"]: /[a-z]+/',
            "suffix is only supported on an ANY_TEXT head used by dynamic dispatch",
            id="suffix-requires-any-text",
        ),
        pytest.param(
            "start: head\nhead[lazy]: /(\\n|.)*END/",
            "lazy regex suffix is only supported on a head used by dynamic dispatch",
            id="standalone-lazy-regex-suffix",
        ),
        pytest.param(
            '%ignore MISSING\nstart: "a"', "unknown name 'MISSING'", id="unknown-ignore-name"
        ),
    ],
)
def test_lark_errors_are_explicit_and_located(grammar: str, message: str) -> None:
    _assert_lark_error(grammar, message)


def test_lark_named_special_token_error_with_tokenizer() -> None:
    tokenizer_info = xgr.TokenizerInfo(["<known>", "text"])
    _assert_lark_error("start: <unknown>", "unknown special token <unknown>", tokenizer_info)


def test_lark_dynamic_trigger_levels_cannot_be_mixed() -> None:
    tokenizer_info = xgr.TokenizerInfo(["<|bar|>", "x"])
    grammar = r"""
        start: (foo | bar)* tail
        tail: TEXT
        foo_head[lazy]: TEXT "<foo>"
        foo: foo_head "x"
        bar: TEXT <|bar|> "x"
        TEXT: /(\n|.)*/
    """
    _assert_lark_error(grammar, "cannot mix string and token triggers", tokenizer_info)


def test_lark_lazy_and_dynamic_special_token_triggers_cannot_be_negated() -> None:
    tokenizer_info = xgr.TokenizerInfo(["a", "b", "c"])
    _assert_lark_error(
        "start: head\nhead[lazy]: TEXT <[^1]>\nTEXT: /(\\n|.)*/",
        "lazy special-token trigger cannot be negated",
        tokenizer_info,
    )
    _assert_lark_error(
        'start: tool* tail\ntail: TEXT\ntool: TEXT <[^1]> "x"\nTEXT: /(\\n|.)*/',
        "dynamic special-token trigger cannot be negated",
        tokenizer_info,
    )


def test_lark_error_reports_crlf_line_column_and_source_context() -> None:
    error = _assert_lark_error(
        '# comment\r\nstart: "a"\r\nitem missing', "expected ':' after rule name"
    )
    assert "line 3, column 6" in error
    assert "item missing" in error
    assert "     ^" in error


MAX_TOKENS_TOKENIZER = xgr.TokenizerInfo(["ab ", "cd", " ", "</t>", "1", "<t>", "x"])


def _allowed_token_ids(matcher: xgr.GrammarMatcher, tokenizer_info: xgr.TokenizerInfo) -> list:
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(bitmask)
    return [
        i for i in range(tokenizer_info.vocab_size) if (int(bitmask[0, i // 32]) >> (i % 32)) & 1
    ]


def _accepts_and_terminates(compiled: xgr.CompiledGrammar, token_ids: Sequence[int]) -> bool:
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    return all(matcher.accept_token(t) for t in token_ids) and matcher.is_terminated()


ANY_TEXT_BUDGET_GRAMMAR = r"""
    start: r "<t>"
    r[max_tokens=3]: TEXT
    TEXT: /(\n|.)*/
"""


def test_lark_max_tokens_any_text_budget() -> None:
    # An arbitrary-text body can end at every position, so the runtime budget is exact: once
    # the budget is exhausted the mask only allows leaving the region.
    compiled = _compile_lark(ANY_TEXT_BUDGET_GRAMMAR, MAX_TOKENS_TOKENIZER)
    assert _accepts_and_terminates(compiled, [5])
    assert _accepts_and_terminates(compiled, [0, 1, 2, 5])
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for token_id in [0, 1, 2]:
        assert matcher.accept_token(token_id)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [5]
    # The mask enforcement commits: a fourth in-region token is rejected.
    assert not matcher.accept_token(0)
    assert matcher.accept_token(5) and matcher.is_terminated()


def test_lark_max_tokens_any_text_keeps_budget_metadata() -> None:
    grammar = xgr.Grammar.from_lark(ANY_TEXT_BUDGET_GRAMMAR, tokenizer_info=MAX_TOKENS_TOKENIZER)
    printed = str(grammar)
    assert "r[max_tokens=3] ::=" in printed
    assert "ExcludeToken(" not in printed


BUDGET_GRAMMAR = r"""
    start: "<t>" r "</t>" ans
    r[max_tokens=2]: /([a-z]* )+/
    ans: "1"
"""


def test_lark_budget_enforced_at_closable_position() -> None:
    compiled = _compile_lark(BUDGET_GRAMMAR, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(5)
    assert matcher.accept_token(0)
    assert matcher.accept_token(0)
    # Budget exhausted and the region can end here (trailing space): only the terminator is
    # allowed. Refilling is idempotent.
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [3]
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [3]
    assert matcher.accept_token(3)
    assert matcher.accept_token(4)
    assert matcher.is_terminated()


def test_lark_budget_relaxed_when_region_cannot_end() -> None:
    compiled = _compile_lark(BUDGET_GRAMMAR, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(5)
    assert matcher.accept_token(1)
    assert matcher.accept_token(1)
    # Mid-group at exhaustion: the region cannot end, so the budget is relaxed for this step.
    allowed = _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER)
    assert 3 not in allowed and 1 in allowed and 2 in allowed
    assert matcher.accept_token(2)
    # Earliest closable position: enforced right after the group terminates.
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [3]
    assert matcher.accept_token(3)
    assert matcher.accept_token(4) and matcher.is_terminated()


def test_lark_budget_relaxed_over_multiple_steps() -> None:
    grammar = r"""
        start: "<t>" r "</t>" ans
        r[max_tokens=1]: /([a-z]* )+/
        ans: "1"
    """
    compiled = _compile_lark(grammar, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(5)
    assert matcher.accept_token(1)
    bitmask = xgr.allocate_token_bitmask(1, MAX_TOKENS_TOKENIZER.vocab_size)
    # The rule cannot end mid-word: the budget stays relaxed step after step until it can.
    matcher.fill_next_token_bitmask(bitmask)
    assert matcher.accept_token(6)
    matcher.fill_next_token_bitmask(bitmask)
    assert matcher.accept_token(6)
    assert matcher.accept_token(2)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [3]


def test_lark_budget_rollback_across_close() -> None:
    compiled = _compile_lark(BUDGET_GRAMMAR, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for token_id in [5, 0, 0]:
        assert matcher.accept_token(token_id)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [3]
    assert matcher.accept_token(3)
    matcher.rollback(2)
    assert matcher.accept_token(1)
    assert matcher.accept_token(2)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [3]
    assert matcher.accept_token(3) and matcher.accept_token(4) and matcher.is_terminated()


def test_lark_budget_accept_string_is_not_counted() -> None:
    compiled = _compile_lark(BUDGET_GRAMMAR, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string("<t>ab ab ab ab ")
    assert matcher.accept_token(0)
    assert matcher.accept_token(0)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [3]


def test_lark_budget_reset_and_fork() -> None:
    compiled = _compile_lark(BUDGET_GRAMMAR, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for token_id in [5, 0, 0]:
        assert matcher.accept_token(token_id)
    forked = matcher.fork()
    assert _allowed_token_ids(forked, MAX_TOKENS_TOKENIZER) == [3]
    matcher.reset()
    assert matcher.accept_token(5) and matcher.accept_token(0)
    allowed = _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER)
    assert 0 in allowed and 3 in allowed
    assert _allowed_token_ids(forked, MAX_TOKENS_TOKENIZER) == [3]


def test_lark_budget_per_occurrence() -> None:
    # The budget bounds each occurrence of the rule; separate occurrences in a loop each get
    # their own budget.
    grammar = 'start: r+ "1"\nr[max_tokens=2]: /[a-z]+ /'
    compiled = _compile_lark(grammar, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for _ in range(5):
        assert matcher.accept_token(0)
    allowed = _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER)
    assert 0 in allowed and 4 in allowed
    assert matcher.accept_token(4) and matcher.is_terminated()
    # A single occurrence spanning more tokens than its budget still expires.
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(1) and matcher.accept_token(1)
    allowed = _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER)
    assert 4 not in allowed
    assert matcher.accept_token(2)
    assert 4 in _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER)


def test_lark_budget_nested_regions_take_minimum() -> None:
    grammar = 'start: a "1"\na[max_tokens=3]: "x" b\nb[max_tokens=9]: /([a-z]* )+/'
    compiled = _compile_lark(grammar, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(6)
    assert matcher.accept_token(0)
    assert matcher.accept_token(0)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [4]


def test_lark_budget_shared_subrule() -> None:
    # A rule inside a budgeted rule may also be used outside of it: the budget follows the
    # derivation, not the rule.
    grammar = xgr.Grammar.from_ebnf(
        'root ::= a " " b\na[max_tokens=1] ::= sub\nb ::= sub\nsub ::= [a-z] sub | [a-z]'
    )
    compiled = xgr.GrammarCompiler(MAX_TOKENS_TOKENIZER, cache_enabled=False).compile_grammar(
        grammar
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(1)
    # a's budget (1 token) is exhausted and a can end: forced to close, continuing with " ".
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [2]
    assert matcher.accept_token(2)
    # b shares sub but carries no budget: more than one token is fine.
    assert matcher.accept_token(1) and matcher.accept_token(1)
    assert matcher.is_completed()


def test_lark_budget_round_trip_and_cache() -> None:
    grammar = xgr.Grammar.from_lark(BUDGET_GRAMMAR, tokenizer_info=MAX_TOKENS_TOKENIZER)
    ebnf = str(grammar)
    assert "r[max_tokens=2] ::=" in ebnf

    for candidate in [
        xgr.Grammar.from_ebnf(ebnf),
        xgr.Grammar.deserialize_json(grammar.serialize_json()),
    ]:
        compiled = xgr.GrammarCompiler(MAX_TOKENS_TOKENIZER, cache_enabled=False).compile_grammar(
            candidate
        )
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        for token_id in [5, 0, 0]:
            assert matcher.accept_token(token_id)
        assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [3]

    cached_compiler = xgr.GrammarCompiler(MAX_TOKENS_TOKENIZER, cache_enabled=True)
    compiled = cached_compiler.compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for token_id in [5, 0, 0]:
        assert matcher.accept_token(token_id)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [3]


@pytest.mark.parametrize(
    "grammar, message",
    [
        pytest.param(
            "start[max_tokens=0]: TEXT\nTEXT: /(\\n|.)*/",
            "max_tokens must be positive",
            id="zero-budget",
        ),
        pytest.param(
            "start[max_tokens=3, max_tokens=4]: TEXT\nTEXT: /(\\n|.)*/",
            "max_tokens attribute is specified more than once",
            id="duplicate",
        ),
        pytest.param(
            'start[max_tokens=3, lazy]: TEXT "<t>"\nTEXT: /(\\n|.)*/',
            "max_tokens cannot be combined with lazy",
            id="combined-with-lazy",
        ),
        pytest.param(
            'start[max_tokens=3, suffix="<t>"]: TEXT\nTEXT: /(\\n|.)*/',
            "max_tokens cannot be combined with lazy or suffix",
            id="combined-with-suffix",
        ),
        pytest.param(
            'TOK[max_tokens=3]: "a"\nstart: TOK',
            "attributes are only supported on rules",
            id="on-terminal",
        ),
    ],
)
def test_lark_max_tokens_errors(grammar: str, message: str) -> None:
    _assert_lark_error(grammar, message, MAX_TOKENS_TOKENIZER)


def test_lark_max_tokens_rejected_on_dispatch_rules() -> None:
    grammar = r"""
        start: tool* tail
        tail: TEXT
        tool_head[lazy]: TEXT "<t>"
        tool[max_tokens=3]: tool_head /[0-9]+/ "</t>"
        TEXT: /(\n|.)*/
    """
    _assert_lark_error(grammar, "max_tokens is not supported on rules consumed by dynamic dispatch")


def test_lark_max_tokens_works_without_tokenizer_info() -> None:
    grammar = xgr.Grammar.from_lark('start: r "<t>"\nr[max_tokens=2]: TEXT\nTEXT: /(\\n|.)*/')
    assert "r[max_tokens=2] ::=" in str(grammar)
    tokenizer_info = xgr.TokenizerInfo(["a", "<t>"])
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(0) and matcher.accept_token(0)
    assert _allowed_token_ids(matcher, tokenizer_info) == [1]
    assert matcher.accept_token(1) and matcher.is_terminated()


def _get_captures(
    grammar: str, value: str, tokenizer_info: Optional[xgr.TokenizerInfo] = None
) -> list:
    compiled = _compile_lark(grammar, tokenizer_info)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string(value)
    return matcher.get_captures()


def test_lark_capture_simple() -> None:
    assert _get_captures('start: "a" v "b"\nv[capture]: /[0-9]+/', "a123b") == [("v", b"123")]


def test_lark_capture_named_and_nested() -> None:
    grammar = 'start: outer\nouter[capture="o"]: "x" inner "!"\ninner[capture="i"]: /[a-z]+/'
    assert _get_captures(grammar, "xabc!") == [("i", b"abc"), ("o", b"xabc!")]


def test_lark_capture_repeated_occurrences() -> None:
    grammar = 'start: (item ",")* item\nitem[capture]: /[0-9]+/'
    assert _get_captures(grammar, "1,22,333") == [("item", b"1"), ("item", b"22"), ("item", b"333")]


def test_lark_capture_right_recursion() -> None:
    # The right-recursion optimization elides parent completions; it must be disabled for
    # captured rules so that every recursion level still records its span.
    grammar = 'start: lst\nlst[capture]: ITEM "," lst | ITEM\nITEM: /[0-9]+/'
    assert _get_captures(grammar, "1,2,3") == [("lst", b"3"), ("lst", b"2,3"), ("lst", b"1,2,3")]


def test_lark_capture_on_root_rule() -> None:
    assert _get_captures('start[capture="all"]: "a" /[0-9]+/ "b"', "a12b") == [("all", b"a12b")]


def test_lark_capture_raw_events_and_coalescing() -> None:
    compiled = _compile_lark('start: "a" v "b"\nv[capture]: /[0-9]+/')
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string("a123b")
    # The /[0-9]+/ body completes after every digit; deduplication keeps the longest
    # completion of the occurrence, the raw event list keeps all of them.
    assert matcher.get_captures() == [("v", b"123")]
    assert matcher.get_captures(deduplicate=False) == [("v", b"1"), ("v", b"12"), ("v", b"123")]


def test_lark_capture_dynamic_tool_call() -> None:
    grammar = r"""
        start: tool* tail
        tail: TEXT

        tool_head[lazy]: TEXT "<tool_call>"
        tool: tool_head arg "</tool_call>"
        arg[capture]: /[0-9]+/

        TEXT: /(\n|.)*/
    """
    value = "before<tool_call>42</tool_call>mid<tool_call>7</tool_call>after"
    assert _get_captures(grammar, value) == [("arg", b"42"), ("arg", b"7")]


def test_lark_capture_special_token_atomic_path() -> None:
    tokenizer_info = xgr.TokenizerInfo(["a", "<|tool|>", "b", "</s>"], stop_token_ids=[3])
    compiled = _compile_lark('start: wrap\nwrap[capture]: "a" <|tool|> "b"', tokenizer_info)
    matcher = xgr.GrammarMatcher(compiled)
    for token_id in [0, 1, 2]:
        assert matcher.accept_token(token_id)
    assert matcher.get_captures() == [("wrap", b"a<|tool|>b")]
    # Rollback across the atomic special-token row and re-accept.
    matcher.rollback(2)
    assert matcher.get_captures() == []
    for token_id in [1, 2]:
        assert matcher.accept_token(token_id)
    assert matcher.get_captures() == [("wrap", b"a<|tool|>b")]


def test_lark_capture_rollback_and_reaccept() -> None:
    tokenizer_info = xgr.TokenizerInfo(["a", "1", "2", "b", "</s>"], stop_token_ids=[4])
    compiled = _compile_lark('start: "a" v "b"\nv[capture]: /[0-9]+/', tokenizer_info)
    matcher = xgr.GrammarMatcher(compiled)
    for token_id in [0, 1, 3]:
        assert matcher.accept_token(token_id)
    assert matcher.get_captures() == [("v", b"1")]
    matcher.rollback(2)
    for token_id in [2, 3]:
        assert matcher.accept_token(token_id)
    assert matcher.get_captures() == [("v", b"2")]


def test_lark_capture_mask_computation_records_nothing() -> None:
    tokenizer_info = xgr.TokenizerInfo(["a", "1", "b", "</s>"], stop_token_ids=[3])
    compiled = _compile_lark('start: "a" v "b"\nv[capture]: /[0-9]+/', tokenizer_info)
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    assert matcher.accept_token(0)
    matcher.fill_next_token_bitmask(bitmask)
    assert matcher.get_captures(deduplicate=False) == []
    assert matcher.accept_token(1)
    matcher.fill_next_token_bitmask(bitmask)
    before = matcher.get_captures(deduplicate=False)
    matcher.fill_next_token_bitmask(bitmask)
    assert matcher.get_captures(deduplicate=False) == before
    assert matcher.accept_token(2)
    assert matcher.get_captures() == [("v", b"1")]


def test_lark_capture_reset_and_fork() -> None:
    compiled = _compile_lark('start: "a" v "b"\nv[capture]: /[0-9]+/')
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string("a5b")
    forked = matcher.fork()
    assert forked.get_captures() == [("v", b"5")]
    matcher.reset()
    assert matcher.get_captures() == []
    assert matcher.accept_string("a6b")
    assert matcher.get_captures() == [("v", b"6")]
    assert forked.get_captures() == [("v", b"5")]


def test_lark_capture_survives_ebnf_round_trip_and_cache() -> None:
    grammar = xgr.Grammar.from_lark('start: "a" v "b"\nv[capture="num"]: /[0-9]+/')
    ebnf = str(grammar)
    assert 'v[capture="num"] ::=' in ebnf

    tokenizer_info = xgr.TokenizerInfo([])
    reparsed = xgr.Grammar.from_ebnf(ebnf)
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(reparsed)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string("a77b")
    assert matcher.get_captures() == [("num", b"77")]

    # The cached compile path re-parses the grammar from its ToString() form.
    cached_compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=True)
    compiled_cached = cached_compiler.compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled_cached, terminate_without_stop_token=True)
    assert matcher.accept_string("a88b")
    assert matcher.get_captures() == [("num", b"88")]


def test_lark_capture_serialization_round_trip() -> None:
    grammar = xgr.Grammar.from_lark('start: "a" v "b"\nv[capture="num"]: /[0-9]+/')
    deserialized = xgr.Grammar.deserialize_json(grammar.serialize_json())
    assert 'v[capture="num"] ::=' in str(deserialized)


def test_lark_capture_on_dispatch_consumed_rule_is_rejected() -> None:
    grammar = r"""
        start: tool* tail
        tail: TEXT
        tool_head[lazy]: TEXT "<t>"
        tool[capture]: tool_head /[0-9]+/ "</t>"
        TEXT: /(\n|.)*/
    """
    _assert_lark_error(grammar, "capture is not supported on rules consumed by dynamic dispatch")


def test_lark_capture_no_capture_grammar_returns_empty() -> None:
    compiled = _compile_lark('start: "ab"')
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string("ab")
    assert matcher.get_captures() == []


CAPTURE_BUDGET_ANY_TEXT_GRAMMAR = r"""
    start: r "<t>"
    r[max_tokens=3, capture]: TEXT
    TEXT: /(\n|.)*/
"""


def test_lark_capture_with_max_tokens_any_text() -> None:
    # Both attributes on one rule, arbitrary-text body (exact token-wildcard strategy): the
    # budget masks to the terminator and the capture spans the whole region.
    compiled = _compile_lark(CAPTURE_BUDGET_ANY_TEXT_GRAMMAR, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for token_id in [0, 1, 2]:
        assert matcher.accept_token(token_id)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [5]
    assert matcher.accept_token(5) and matcher.is_terminated()
    assert matcher.get_captures() == [("r", b"ab cd ")]


def test_lark_capture_with_max_tokens_cfg_body() -> None:
    # Both attributes on one rule, CFG body (runtime-deadline strategy), plus a captured rule
    # after the budgeted region.
    grammar = r"""
        start: "<t>" r "</t>" ans
        r[max_tokens=2, capture="think"]: /([a-z]* )+/
        ans[capture]: "1"
    """
    compiled = _compile_lark(grammar, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for token_id in [5, 0, 0]:
        assert matcher.accept_token(token_id)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [3]
    assert matcher.accept_token(3) and matcher.accept_token(4) and matcher.is_terminated()
    assert matcher.get_captures() == [("think", b"ab ab "), ("ans", b"1")]


def test_lark_capture_inside_max_tokens_region() -> None:
    # The captured rule is nested inside the budgeted rule: the budget follows the outer
    # derivation while the capture records the inner span.
    grammar = r"""
        start: outer "1"
        outer[max_tokens=3]: "x" inner
        inner[capture]: /([a-z]* )+/
    """
    compiled = _compile_lark(grammar, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for token_id in [6, 0, 0]:
        assert matcher.accept_token(token_id)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [4]
    assert matcher.accept_token(4) and matcher.is_terminated()
    assert matcher.get_captures() == [("inner", b"ab ab ")]


def test_lark_capture_with_max_tokens_per_occurrence() -> None:
    # Each loop occurrence gets its own budget and its own capture.
    grammar = 'start: r+ "1"\nr[capture, max_tokens=2]: /[a-z]+ /'
    compiled = _compile_lark(grammar, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for _ in range(5):
        assert matcher.accept_token(0)
    assert matcher.accept_token(4) and matcher.is_terminated()
    assert matcher.get_captures() == [("r", b"ab ")] * 5
    # A single occurrence exceeding its budget mid-word: the budget is relaxed until the
    # region can close, and the capture still reports the full span.
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(1) and matcher.accept_token(1)
    assert 4 not in _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER)
    assert matcher.accept_token(2)
    assert 4 in _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER)
    assert matcher.accept_token(4) and matcher.is_terminated()
    assert matcher.get_captures() == [("r", b"cdcd ")]


def test_lark_capture_with_max_tokens_rollback() -> None:
    grammar = r"""
        start: "<t>" r "</t>" ans
        r[max_tokens=2, capture]: /([a-z]* )+/
        ans: "1"
    """
    compiled = _compile_lark(grammar, MAX_TOKENS_TOKENIZER)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for token_id in [5, 0, 0]:
        assert matcher.accept_token(token_id)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [3]
    assert matcher.accept_token(3)
    matcher.rollback(2)
    assert matcher.accept_token(1) and matcher.accept_token(2)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [3]
    assert matcher.accept_token(3) and matcher.accept_token(4) and matcher.is_terminated()
    assert matcher.get_captures() == [("r", b"ab cd ")]


def test_lark_capture_with_max_tokens_round_trip_and_cache() -> None:
    # Both attributes must survive the printer -> EBNF-lexer round trip together, since the
    # cached compile path re-parses the grammar from its ToString() form.
    grammar = xgr.Grammar.from_lark(
        CAPTURE_BUDGET_ANY_TEXT_GRAMMAR, tokenizer_info=MAX_TOKENS_TOKENIZER
    )
    ebnf = str(grammar)
    assert 'r[max_tokens=3, capture="r"] ::=' in ebnf
    assert 'r[max_tokens=3, capture="r"] ::=' in str(xgr.Grammar.from_ebnf(ebnf))
    compiler = xgr.GrammarCompiler(MAX_TOKENS_TOKENIZER, cache_enabled=True)
    compiled = compiler.compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for token_id in [0, 1, 2]:
        assert matcher.accept_token(token_id)
    assert _allowed_token_ids(matcher, MAX_TOKENS_TOKENIZER) == [5]
    assert matcher.accept_token(5) and matcher.is_terminated()
    assert matcher.get_captures() == [("r", b"ab cd ")]


LAZY_TOKENIZER = xgr.TokenizerInfo(["<", ">", "a", "b", "ab", "abb", " "])


def _lazy_matcher(grammar_obj: xgr.Grammar) -> xgr.GrammarMatcher:
    compiled = xgr.GrammarCompiler(LAZY_TOKENIZER, cache_enabled=False).compile_grammar(grammar_obj)
    return xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)


def _lazy_allowed_token_ids(matcher: xgr.GrammarMatcher) -> list:
    bitmask = xgr.allocate_token_bitmask(1, LAZY_TOKENIZER.vocab_size)
    matcher.fill_next_token_bitmask(bitmask)
    return [
        i for i in range(LAZY_TOKENIZER.vocab_size) if (int(bitmask[0, i // 32]) >> (i % 32)) & 1
    ]


def test_lark_lazy_committed_shortest_regex() -> None:
    _assert_language(
        'start: "<" r ">"\nr[lazy]: /[a-z]+/', ["<a>", "<b>"], ["<ab>", "<>", "<a", "a>"]
    )
    # Greedy control: the same grammar without lazy accepts longer matches.
    _assert_language('start: "<" r ">"\nr: /[a-z]+/', ["<a>", "<ab>"], ["<>"])


def test_lark_lazy_matches_exactly_one_unit() -> None:
    _assert_language('start: r "a"\nr[lazy]: /[b]+/', ["ba"], ["bba", "a"])


def test_lark_lazy_nullable_always_matches_empty() -> None:
    _assert_language('start: "<" r ">"\nr[lazy]: /[a-z]*/', ["<>"], ["<a>", "<ab>"])
    _assert_language('start: "<" r ">"\nr[lazy]: /[a-z]?/', ["<>"], ["<a>"])


def test_lark_lazy_choices_commit_at_shortest() -> None:
    _assert_language('start: "<" r ">"\nr[lazy]: "ab" | "abc"', ["<ab>"], ["<abc>"])
    # Prefix-free alternatives are unaffected by the commit.
    _assert_language('start: "<" r ">"\nr[lazy]: "aa" | "bb"', ["<aa>", "<bb>"], ["<a>", "<ab>"])


def test_lark_lazy_composed_of_terminals() -> None:
    _assert_language('start: "<" r ">"\nr[lazy]: SUB SUB\nSUB: /[a-z]/', ["<ab>"], ["<a>", "<abc>"])


def test_ebnf_lazy_committed_shortest_and_plus_desugar() -> None:
    for body in ("[a-z] [a-z]*", "[a-z]+"):
        grammar_obj = xgr.Grammar.from_ebnf(f'root ::= "<" r ">"\nr[lazy] ::= {body}')
        _assert_grammar_language(grammar_obj, ["<a>"], ["<ab>", "<>"])


def test_ebnf_lazy_attribute_round_trips() -> None:
    grammar_obj = xgr.Grammar.from_lark('start: "<" r ">"\nr[lazy]: /[a-z]+/')
    printed = str(grammar_obj)
    assert "r[lazy] ::=" in printed
    _assert_grammar_language(xgr.Grammar.from_ebnf(printed), ["<a>"], ["<ab>"])
    deserialized = xgr.Grammar.deserialize_json(grammar_obj.serialize_json())
    _assert_grammar_language(deserialized, ["<a>"], ["<ab>"])
    # The compiler cache path re-parses ToString(); the attribute must survive it.
    compiled = xgr.GrammarCompiler(LAZY_TOKENIZER, cache_enabled=True).compile_grammar(grammar_obj)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert not matcher.accept_string("<ab>")


def test_lark_lazy_mask_is_exit_only_after_commit() -> None:
    # Tokens: 0 "<", 1 ">", 2 "a", 3 "b", 4 "ab", 5 "abb", 6 " "
    grammar_obj = xgr.Grammar.from_lark('start: "<" r "b"\nr[lazy]: /[ab]+/')
    matcher = _lazy_matcher(grammar_obj)
    assert matcher.accept_token(0)
    allowed = _lazy_allowed_token_ids(matcher)
    # "ab" = one region char, then the closing literal: allowed. "abb" would extend the
    # region past the commit point: rejected.
    assert 4 in allowed and 5 not in allowed
    assert matcher.accept_token(4) and matcher.is_terminated()
    # After committing via a single region char, only the closing literal remains.
    matcher = _lazy_matcher(grammar_obj)
    assert matcher.accept_token(0) and matcher.accept_token(2)
    assert _lazy_allowed_token_ids(matcher) == [3]


def test_lark_lazy_per_occurrence_commit() -> None:
    _assert_language('start: r " " r\nr[lazy]: /[a-z]+/', ["a b"], ["ab b", "a bb", "a b c"])


def test_lark_lazy_root_anchored_occurrence() -> None:
    grammar_obj = xgr.Grammar.from_lark('start: "<" r\nr[lazy]: /[a-z]+/')
    matcher = _lazy_matcher(grammar_obj)
    assert matcher.accept_token(0) and matcher.accept_token(2)
    assert _lazy_allowed_token_ids(matcher) == []
    _assert_grammar_language(grammar_obj, ["<a"], ["<ab", "<"])


def test_lark_lazy_rollback_reset_fork() -> None:
    grammar_obj = xgr.Grammar.from_lark('start: "<" r ">"\nr[lazy]: /[a-z]+/')
    matcher = _lazy_matcher(grammar_obj)
    assert matcher.accept_token(0) and matcher.accept_token(2)
    assert _lazy_allowed_token_ids(matcher) == [1]
    forked = matcher.fork()
    matcher.rollback(1)
    assert 2 in _lazy_allowed_token_ids(matcher)
    assert _lazy_allowed_token_ids(forked) == [1]
    matcher.reset()
    assert matcher.accept_string("<a>")


def test_lark_lazy_accept_string_and_tokens_agree() -> None:
    grammar_obj = xgr.Grammar.from_lark('start: "<" r ">"\nr[lazy]: /[a-z]+/')
    matcher = _lazy_matcher(grammar_obj)
    assert not matcher.accept_string("<ab>")
    matcher.reset()
    assert matcher.accept_token(0) and matcher.accept_token(2)
    assert not matcher.accept_token(3)
    assert matcher.accept_token(1) and matcher.is_terminated()


def test_lark_lazy_ignore_is_not_woven_into_lazy_rules() -> None:
    grammar = 'start: "<" r ">"\nr[lazy]: /[a-z]+/\n%ignore " "'
    _assert_language(grammar, ["<a>", "< a >"], ["<ab>"])


def test_lark_lazy_dispatch_subset_keeps_tag_dispatch() -> None:
    grammar_obj = xgr.Grammar.from_lark('start: head\nhead[lazy]: TEXT "<end>"\nTEXT: /(\\n|.)*/')
    printed = str(grammar_obj)
    assert "TagDispatch" in printed
    assert "[lazy]" not in printed


def test_lark_lazy_non_terminal_like_bodies_are_rejected() -> None:
    _assert_lark_error('start: "<" r ">"\nr[lazy]: "a" r | "b"', "terminal cannot reference rule")
    _assert_lark_error("start: r\nR[lazy]: /[a-z]+/\nr: R", "attributes are only supported")
    for grammar_obj in (
        xgr.Grammar.from_lark('start: "<" r ">"\nr[lazy]: /([a-z]+ )+/'),
        xgr.Grammar.from_ebnf('root ::= "<" r ">"\nr[lazy] ::= sub\nsub ::= sub [a-z] | [a-z]'),
    ):
        with pytest.raises(RuntimeError, match="terminal-like"):
            xgr.GrammarCompiler(LAZY_TOKENIZER, cache_enabled=False).compile_grammar(grammar_obj)
