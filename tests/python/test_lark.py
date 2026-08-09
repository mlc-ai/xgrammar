import json
from typing import Optional, Sequence, Union

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


def _assert_byte_language(
    grammar: Union[str, xgr.Grammar], accepted: Sequence[bytes], rejected: Sequence[bytes]
) -> None:
    vocabulary = list(dict.fromkeys(value for value in [*accepted, *rejected] if value))
    tokenizer_info = xgr.TokenizerInfo(vocabulary, vocab_size=len(vocabulary), stop_token_ids=[])
    grammar_obj = (
        xgr.Grammar.from_lark(grammar, tokenizer_info=tokenizer_info)
        if isinstance(grammar, str)
        else grammar
    )
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar_obj)
    token_ids = {value: token_id for token_id, value in enumerate(vocabulary)}

    def matches(value: bytes) -> bool:
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        if value and not matcher.accept_token(token_ids[value]):
            return False
        return matcher.is_terminated()

    for value in accepted:
        assert matches(value), value
    for value in rejected:
        assert not matches(value), value


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
    "attributes, message, location, caret",
    [
        pytest.param(
            "temperature=0.7, max_tokens=2",
            "max_tokens cannot be combined with temperature",
            "line 2, column 24",
            "                       ^",
            id="temperature-max-tokens",
        ),
        pytest.param(
            "lazy, temperature=0.7",
            "temperature cannot be combined with lazy, suffix, or stop",
            "line 2, column 1",
            "^",
            id="temperature-lazy",
        ),
        pytest.param(
            'suffix="!", temperature=0.7',
            "temperature cannot be combined with lazy, suffix, or stop",
            "line 2, column 1",
            "^",
            id="temperature-suffix",
        ),
        pytest.param(
            'stop="!", temperature=0.7',
            "temperature cannot be combined with lazy, suffix, or stop",
            "line 2, column 1",
            "^",
            id="temperature-stop",
        ),
    ],
)
def test_lark_incompatible_rule_limits_have_precise_diagnostics(
    attributes: str, message: str, location: str, caret: str
) -> None:
    rule_line = f'value[{attributes}]: "x"'
    error = _assert_lark_error(f"start: value\n{rule_line}", message)
    assert location in error
    assert f"\n{rule_line}\n{caret}" in error


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
            r"start: /hello[0-2]\x21/iu",
            ["hello0!", "HELLO1!", "HeLlO2!"],
            ["hello3!", "hello1", "héllo1!"],
            id="regex-case-insensitive-and-unicode-flags",
        ),
        pytest.param(
            "start: /Żółw|Σ|k|ß/iu",
            ["Żółw", "Σ", "k", "K", "ß"],
            ["żółw", "ŻÓŁW", "zółw", "σ", "ς", "ẞ", "\u212a", "ss", "SS"],
            id="regex-case-insensitive-folds-ascii-only",
        ),
        pytest.param(
            "start: /[^kσ]+/i",
            ["A", "ż", "😀", "Σ", "ς", "\u212a"],
            ["k", "K", "σ", "aK"],
            id="regex-case-insensitive-negative-class-ascii-folding",
        ),
        pytest.param(
            "start: /[À-Ö]+/i",
            ["À", "Ö", "ÀÉÖ"],
            ["à", "ö", "×", "÷", "A", ""],
            id="regex-case-insensitive-non-ascii-range-not-folded",
        ),
        pytest.param(
            "start: /a.b/is",
            ["a b", "A😀B", "a\nb"],
            ["ab", "a\n\nb", "ä\nb"],
            id="regex-case-insensitive-dotall-flags",
        ),
        pytest.param(
            "start: /[^a-c]+/i",
            ["Z", "09", "Ä"],
            ["a", "B", "xyzC"],
            id="regex-case-insensitive-negative-class",
        ),
        pytest.param(
            "start: TOKEN\nTOKEN: /[A-Cx-z]+/i",
            ["abc", "ABC", "XyZ", "cZ"],
            ["", "d", "w", "abcd"],
            id="regex-case-insensitive-terminal-range",
        ),
        pytest.param(
            "start: /(ab){2,300}c/i",
            ["abABc", "aBab" * 150 + "c"],
            ["c", "abc", "aBab" * 150 + "abc"],
            id="regex-case-insensitive-large-repeat-subrule",
        ),
        pytest.param(
            # Physically unrolling this repetition would exceed the FSM state limit, so this
            # only compiles if the repetition becomes a grammar-level repeat subrule.
            "start: /(ab){2,50000}c/i",
            ["abABc", "aBab" * 150 + "c"],
            ["c", "abc", "ab" * 150],
            id="regex-case-insensitive-huge-repeat-not-unrolled",
        ),
        pytest.param(
            "start: /(ab){200,}c/i",
            ["ab" * 200 + "c", "aB" * 321 + "c"],
            ["ab" * 199 + "c", "c"],
            id="regex-case-insensitive-unbounded-large-repeat",
        ),
        pytest.param(
            # The repeated atom is nullable, so the lower bound is relaxed to zero.
            "start: /(a?){2,300}b/i",
            ["b", "ab", "A" * 300 + "b"],
            ["a" * 301 + "b", ""],
            id="regex-case-insensitive-nullable-large-repeat",
        ),
        pytest.param(
            r"start: /\x41\u0062\u{43}/i",
            ["AbC", "abc", "ABC"],
            ["abd", "ab"],
            id="regex-case-insensitive-folds-escapes",
        ),
        pytest.param(
            r"start: /\u{1F600}+/i", ["😀", "😀😀"], ["", "😁"], id="regex-unicode-codepoint-escape"
        ),
        pytest.param(
            r"start: /[\u0041-\u0043x]+/i",
            ["ABC", "abc", "X"],
            ["d", "D", ""],
            id="regex-class-unicode-escape-folded",
        ),
        pytest.param(
            r"start: /x\cAy/i", ["x\x01y", "X\x01Y"], ["xy", "xay"], id="regex-control-escape"
        ),
        pytest.param(
            r"start: /a\sb/i",
            ["a b", "A\tb", "a\nB", "a\fb", "a\vb"],
            ["ab", "a\x00b", "a\x01b"],
            id="regex-standard-whitespace-class",
        ),
        pytest.param(
            r"start: /\S+/i",
            ["好!x", "\x00"],
            ["a b", ""],
            id="regex-non-whitespace-codepoint-domain",
        ),
        pytest.param("start: /a(?=b)c/i", ["ac", "AC"], ["abc", "a"], id="regex-lookahead-ignored"),
        pytest.param(
            "start: /(?<name>ab)+(?P<other>c)/i",
            ["abc", "ABabC"],
            ["c", "ab"],
            id="regex-named-groups-ignore-name",
        ),
        pytest.param("start: /(?:ab)+c/i", ["abc", "ababC"], ["c"], id="regex-non-capturing-group"),
        pytest.param("start: /a^b$c/i", ["abc", "ABC"], ["ac"], id="regex-mid-anchors-ignored"),
        pytest.param(
            "start: /a+?b??c/i",
            ["ac", "abc", "AAC"],
            ["c", "abbc"],
            id="regex-non-greedy-quantifiers",
        ),
        pytest.param(
            "start: /(a|)b|c|/i", ["ab", "b", "c", ""], ["a"], id="regex-empty-alternatives"
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


def test_lark_parametric_permutations() -> None:
    grammar = """
        start: perm::0x0
        perm::_: "" %if is_ones([0:3])
              | "a" perm::set_bit(0) %if bit_clear(0)
              | "b" perm::set_bit(1) %if bit_clear(1)
              | "c" perm::set_bit(2) %if bit_clear(2)
    """
    _assert_language(
        grammar, ["abc", "acb", "bac", "bca", "cab", "cba"], ["", "a", "ab", "aabc", "abcc", "abcd"]
    )


def test_lark_parametric_saturating_counters_and_bit_operations() -> None:
    grammar = """
        start: counter::0 "|" bits::0
        counter::_: "b" counter::incr([0:3]) %if lt([0:3], 5)
                 | "" %if eq([0:3], 5)
        bits::_: "x" bits::bit_or(0x3) %if is_zeros([0:2])
              | "y" bits::clear_bit(0) %if eq([0:2], 3)
              | "" %if eq(_, 2)
    """
    _assert_language(grammar, ["bbbbb|xy"], ["|xy", "bbbb|xy", "bbbbb|x", "bbbbb|xyy"])


def test_lark_parametric_conditions_and_round_trips() -> None:
    grammar = r"""
        start: state::0
        state::_: "a" state::set_bit(0) %if and(bit_clear(0), bit_count_lt(_, 2))
               | "b" state::set_bit(1) %if and(bit_clear(1), bit_count_lt(_, 2))
               | "!" state::bit_and(0x2) %if and(bit_set(0), or(is_zeros([1:2]), bit_set(1)))
               | "" %if not(ne(_, 2))
    """
    grammar_obj = xgr.Grammar.from_lark(grammar)
    restored_ebnf = xgr.Grammar.from_ebnf(str(grammar_obj))
    restored_json = xgr.Grammar.deserialize_json(grammar_obj.serialize_json())
    accepted = ["b", "ab!", "ba!", "a!b"]
    rejected = ["", "a", "aa", "bb", "ab", "b!", "ab!!"]
    for candidate in [grammar_obj, restored_ebnf, restored_json]:
        _assert_grammar_language(candidate, accepted, rejected)


def test_lark_parametric_is_local_to_nested_grammar() -> None:
    grammar = """
        start: "A" %lark {
          start: choose::0
          choose::_: "x" choose::set_bit(0) %if bit_clear(0)
                  | "y" %if bit_set(0)
        } "B"
    """
    _assert_language(grammar, ["AxyB"], ["AyB", "AxB", "AxxyB"])


def test_lark_parametric_decrement_self_reference_and_high_bit() -> None:
    grammar = """
        start: countdown::3 "|" high::0
        countdown::_: "d" countdown::decr([0:2]) %if gt([0:2], 0)
                    | bridge::_ %if eq([0:2], 0)
        bridge::_: "z" %if eq(_, 0)
        high::_: "h" high::set_bit(63) %if bit_clear(63)
              | "" %if bit_set(63)
    """
    _assert_language(grammar, ["dddz|h"], ["ddz|h", "ddddz|h", "dddz|", "dddz|hh"])


def test_lark_parametric_increment_and_decrement_saturate() -> None:
    grammar = """
        start: high::3 low::0
        high::_: "H" high_done::incr([0:2])
        high_done::_: "+" %if eq(_, 3)
        low::_: "L" low_done::decr([0:2])
        low_done::_: "-" %if eq(_, 0)
    """
    _assert_language(grammar, ["H+L-"], ["H+L", "HL-", "H+L--"])


def test_lark_parametric_nested_empty_rule_chain() -> None:
    grammar = """
        start: perm::0 "X"
        perm::_: empty_a::_ empty_b::_ %if is_zeros([10:12])
              | "a" perm::set_bit(0) %if bit_clear(0)
              | "b" perm::set_bit(1) %if bit_clear(1)
              | "c" perm::set_bit(2) %if bit_clear(2)
        empty_a::_: "" %if is_ones([0:1])
        empty_b::_: empty_c::_ %if is_ones([1:2])
        empty_c::_: "" %if is_ones([2:3])
    """
    _assert_language(
        grammar, ["abcX", "acbX", "bacX", "bcaX", "cabX", "cbaX"], ["X", "abX", "abc", "aabcX"]
    )


def test_lark_parametric_unconditional_recursion() -> None:
    grammar = """
        start: seen::0
        seen::_: "" %if is_ones([0:3])
              | "a" seen::set_bit(0)
              | "b" seen::set_bit(1)
              | "c" seen::set_bit(2)
    """
    _assert_language(grammar, ["abc", "caba", "aaabbbc"], ["", "a", "ab", "aaaa", "abcaax"])


def test_lark_parametric_independent_bit_slice_counters() -> None:
    grammar = """
        start: counts::0
        counts::_: "a" counts::incr([0:2]) %if lt([0:2], 2)
                | "b" counts::incr([2:4]) %if lt([2:4], 2)
                | "c" counts::incr([4:7]) %if lt([4:7], 3)
                | "" %if and(eq([0:2], 2), and(eq([2:4], 2), eq([4:7], 3)))
    """
    _assert_language(
        grammar, ["aabbccc", "cabcbac", "ccbabac"], ["", "aabbcc", "aaabbbccc", "aabbcccc"]
    )


def test_lark_parametric_long_counter() -> None:
    count = 900
    grammar = f"""
        start: count_up::0 "X"
        count_up::_: "a" count_up::incr([0:10]) %if lt([0:10], {count})
                  | count_down::_ %if eq([0:10], {count})
        count_down::_: "b" count_down::decr([0:10]) %if gt([0:10], 0)
                    | "" %if eq([0:10], 0)
    """
    accepted = "a" * count + "b" * count + "X"
    _assert_language(
        grammar,
        [accepted],
        ["a" * (count - 1) + "b" * count + "X", "a" * count + "b" * (count - 1) + "X"],
    )


def test_lark_parametric_conditions_inside_groups_and_repetitions() -> None:
    grammar = """
        start: choice::0 "|" choice::1
        choice::_: ("a" %if bit_clear(0) | "b" %if bit_set(0)) ("x" %if bit_set(0))?
    """
    _assert_language(grammar, ["a|b", "a|bx"], ["b|b", "ax|b", "a|a", "a|bxx"])


def test_lark_parametric_capture_and_max_chars_attributes() -> None:
    capture_grammar = """
        start: item::0
        item::_[capture="item"]: "a" item::set_bit(0) %if bit_clear(0)
                                | "b" %if bit_set(0)
    """
    compiled = _compile_lark(capture_grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string("ab")
    assert matcher.is_terminated()
    assert matcher.get_captures() == [("item", b"b"), ("item", b"ab")]

    budget_grammar = """
        start: item::0
        item::_[max_chars=2]: "a" item::incr(_) %if lt(_, 3)
                            | ""
    """
    _assert_language(budget_grammar, ["", "a", "aa"], ["aaa", "aaaa"])


def test_lark_parametric_false_only_state_is_empty_language() -> None:
    grammar = """
        start: dead::0 | "ok"
        dead::_: "bad" %if bit_set(0)
    """
    grammar_obj = xgr.Grammar.from_lark(grammar)
    candidates = [
        grammar_obj,
        xgr.Grammar.from_ebnf(str(grammar_obj)),
        xgr.Grammar.deserialize_json(grammar_obj.serialize_json()),
    ]
    for candidate in candidates:
        _assert_grammar_language(candidate, ["ok"], ["", "bad", "badok", "😀"])


def test_lark_parametric_unsigned_comparisons() -> None:
    grammar = """
        start: checks::5
        checks::_: "eq" %if eq([0:3], 5)
                | "ne" %if ne([0:3], 4)
                | "lt" %if lt([0:3], 6)
                | "le" %if le([0:3], 5)
                | "gt" %if gt([0:3], 4)
                | "ge" %if ge([0:3], 5)
                | "always" %if true()
                | "bare" %if true
                | "eq_false" %if eq([0:3], 4)
                | "ne_false" %if ne([0:3], 5)
                | "lt_false" %if lt([0:3], 5)
                | "le_false" %if le([0:3], 4)
                | "gt_false" %if gt([0:3], 5)
                | "ge_false" %if ge([0:3], 6)
    """
    _assert_language(
        grammar,
        ["eq", "ne", "lt", "le", "gt", "ge", "always", "bare"],
        ["eq_false", "ne_false", "lt_false", "le_false", "gt_false", "ge_false"],
    )


def test_lark_parametric_bit_count_comparisons_and_u64_literal() -> None:
    grammar = """
        start: counts::0xb full::0xffffffffffffffff
        counts::_: "E" %if bit_count_eq([0:4], 3)
                | "N" %if bit_count_ne([0:4], 2)
                | "L" %if bit_count_lt([0:4], 4)
                | "l" %if bit_count_le([0:4], 3)
                | "G" %if bit_count_gt([0:4], 2)
                | "g" %if bit_count_ge([0:4], 3)
                | "x" %if bit_count_eq([0:4], 2)
                | "y" %if bit_count_ne([0:4], 3)
        full::_: "F" %if is_ones(_)
    """
    _assert_language(grammar, ["EF", "NF", "LF", "lF", "GF", "gF"], ["xF", "yF", "F"])


@pytest.mark.parametrize("reference", ["_", "[0:64]"])
@pytest.mark.parametrize(
    "state, count", [("0", 0), ("0x7fffffffffffffff", 63), ("0xffffffffffffffff", 64)]
)
def test_lark_parametric_bit_count_full_width_boundaries(
    reference: str, state: str, count: int
) -> None:
    comparisons = {
        "eq": lambda left, right: left == right,
        "ne": lambda left, right: left != right,
        "lt": lambda left, right: left < right,
        "le": lambda left, right: left <= right,
        "gt": lambda left, right: left > right,
        "ge": lambda left, right: left >= right,
    }
    alternatives = []
    accepted = []
    rejected = []
    for operation, compare in comparisons.items():
        for boundary in (0, 63, 64):
            literal = f"{operation}_{boundary}"
            alternatives.append(f'"{literal}" %if bit_count_{operation}({reference}, {boundary})')
            (accepted if compare(count, boundary) else rejected).append(literal)
    grammar = f"start: checks::{state}\nchecks::_: " + "\n        | ".join(alternatives)
    _assert_language(grammar, accepted, rejected)


def test_lark_parametric_bit_count_high_bit_slices_and_large_thresholds() -> None:
    grammar = """
        start: high::0x8000000000000000 "|" low::0x7fffffffffffffff
        high::_: "H" %if and(bit_count_eq([63:64], 1), bit_count_eq([0:63], 0))
        low::_: "L" %if and(bit_count_eq([63:64], 0), bit_count_eq([0:63], 63))
    """
    _assert_language(grammar, ["H|L"], ["L|H", "H|H", "L|L", ""])

    large_threshold = "18446744073709551615"
    grammar = f"""
        start: checks::0xffffffffffffffff
        checks::_: "lt" %if bit_count_lt(_, {large_threshold})
                | "eq" %if bit_count_eq(_, 65)
                | "ne" %if bit_count_ne(_, 65)
                | "ge" %if bit_count_ge(_, 65)
    """
    _assert_language(grammar, ["lt", "ne"], ["eq", "ge"])


def test_lark_parametric_full_width_arithmetic_saturates_without_overflow() -> None:
    grammar = """
        start: top::0xffffffffffffffff "|" bottom::0
        top::_: "T" top_done::incr(_)
        top_done::_: "!" %if eq(_, 0xffffffffffffffff)
        bottom::_: "B" bottom_done::decr(_)
        bottom_done::_: "!" %if eq(_, 0)
    """
    _assert_language(grammar, ["T!|B!"], ["T|B!", "T!|B", "T!!|B!"])


def test_lark_parametric_exactly_4096_instances_succeeds() -> None:
    grammar = """
        start: counter::0
        counter::_: "x" counter::incr([0:12]) %if lt([0:12], 4095)
                 | "" %if eq([0:12], 4095)
    """
    xgr.Grammar.from_lark(grammar)


def test_lark_parametric_nested_documents_have_independent_instance_limits() -> None:
    grammar = """
        start: outer::0
        outer::_: "o" outer::incr([0:12]) %if lt([0:12], 4095)
                | %lark {
                    start: inner::0
                    inner::_: "i" inner::incr([0:12]) %if lt([0:12], 4095)
                            | "" %if eq([0:12], 4095)
                  } %if eq([0:12], 4095)
    """
    xgr.Grammar.from_lark(grammar)


def test_lark_parametric_combines_regex_flags_and_structured_substring() -> None:
    grammar = r"""
        start: item::0
        item::_: /a.b/isu item::set_bit(0) %if bit_clear(0)
              | %regex {"substring_chars":"xy"} %if bit_set(0)
    """
    _assert_language(
        grammar, ["A\nB", "A\nBx", "A\nBxy", "a0by"], ["", "A\nBxx", "A\nByx", "a0bz", "aB"]
    )


def test_lark_parametric_generated_names_do_not_collide() -> None:
    grammar = """
        start: pick::0
        pick::_: "x" pick::set_bit(0) %if bit_clear(0)
               | "" %if bit_set(0)
        pick__param_0000000000000000: "reserved"
    """
    _assert_language(grammar, ["x"], ["", "reserved", "xx"])


@pytest.mark.parametrize(
    "grammar, message",
    [
        pytest.param(
            'start: foo::0\nfoo: "a"',
            "rule 'foo' is not parametric",
            id="argument-to-ordinary-rule",
        ),
        pytest.param(
            'start: foo\nfoo::_: "a" %if true',
            "does not depend on its parameter",
            id="unused-parameter",
        ),
        pytest.param(
            'start: foo\nfoo: bar\nbar::_: "a" %if bit_set(0)',
            "requires a parameter",
            id="missing-argument",
        ),
        pytest.param(
            "start: foo\nfoo: bar::_\nbar::_: bar::_",
            "requires a caller parameter",
            id="caller-has-no-parameter",
        ),
        pytest.param(
            'BAR: "b" %if true\nstart: BAR',
            "%if cannot be used in terminals",
            id="condition-in-terminal",
        ),
        pytest.param(
            'BAR: foo::0\nstart: BAR\nfoo::_: "x" %if eq(_, 0)',
            "parameterized rule references cannot be used in terminals",
            id="argument-in-terminal",
        ),
        pytest.param(
            'start: foo::0\nfoo::_: "a" %if eq([3:3], 0)',
            "must be > start bit index",
            id="empty-bit-range",
        ),
        pytest.param(
            'start: foo::0\nfoo::_: "a" %if eq([64:64], 0)',
            "must be <= 63",
            id="bit-range-start-too-large",
        ),
        pytest.param(
            'start: foo::0\nfoo::_: "a" %if eq([0:65], 0)',
            "must be <= 64",
            id="bit-range-end-too-large",
        ),
        pytest.param(
            'start: foo::0\nfoo::_: "a" %if eq(0, 0)',
            "expected '_' or '[start_bit:stop_bit]'",
            id="bit-range-brackets-required",
        ),
        pytest.param(
            'start: foo::0\nfoo::_: "a" %if unknown(_, 0)',
            "unknown condition 'unknown'",
            id="unknown-condition",
        ),
        pytest.param(
            "start: foo::unknown(0)\nfoo::_: foo::_",
            "unknown parameter expression 'unknown'",
            id="unknown-expression",
        ),
        pytest.param(
            "start: foo::set_bit(64)\nfoo::_: foo::_", "must be <= 63", id="bit-index-too-large"
        ),
        pytest.param(
            "start: foo::18446744073709551616\nfoo::_: foo::_",
            "invalid 64-bit parameter value",
            id="parameter-overflow",
        ),
        pytest.param(
            "start: foo::-1\nfoo::_: foo::_",
            "parameter values must be unsigned",
            id="negative-parameter",
        ),
        pytest.param(
            "start: foo::0x\nfoo::_: foo::_",
            "expected hexadecimal digits after '0x'",
            id="missing-hexadecimal-digits",
        ),
        pytest.param(
            'start: foo::0\nfoo::_: "ok" %if bit_count_eq(_, 18446744073709551616)',
            "invalid 64-bit parameter value",
            id="bit-count-threshold-overflow",
        ),
        pytest.param(
            'start: foo::0\nfoo::_: "ok" %if bit_count_eq(_, -1)',
            "parameter values must be unsigned",
            id="negative-bit-count-threshold",
        ),
        pytest.param(
            'start: state::0\nstate::_: "ok" %if bit_clear(0) | missing %if bit_set(0)',
            "unknown name 'missing'",
            id="unknown-name-in-false-branch",
        ),
        pytest.param(
            'start: state::0\nstate::_: "ok" %if bit_clear(0) | plain::_ %if bit_set(0)\nplain: "x"',
            "rule 'plain' is not parametric",
            id="argument-to-ordinary-rule-in-false-branch",
        ),
        pytest.param(
            'start: state::0\nstate::_: "ok" %if bit_clear(0) | other %if bit_set(0)\n'
            'other::_: "x" %if bit_set(0)',
            "parametric rule 'other' requires a parameter",
            id="missing-argument-in-false-branch",
        ),
        pytest.param(
            'start: state::0\nstate::_: "ok" %if bit_clear(0) | other::_ %if bit_set(0)\n'
            "other::_: missing %if bit_set(0)",
            "unknown name 'missing'",
            id="invalid-rule-reachable-only-through-false-branch",
        ),
        pytest.param(
            "start::_: start::_", "start rule cannot be parametric", id="parametric-start"
        ),
        pytest.param(
            'start: foo::0\nfoo::_[stop="x"]: foo::_',
            "stop-like behavior is not supported",
            id="parametric-stop",
        ),
        pytest.param(
            "start: foo::0\nfoo::_[temperature=1]: foo::_",
            "temperature is not supported",
            id="parametric-temperature",
        ),
        pytest.param(
            "start: foo::0\nfoo::_[max_tokens=1]: foo::_",
            "max_tokens is not supported",
            id="parametric-max-tokens",
        ),
        pytest.param(
            'start: counter::0\ncounter::_: "x" counter::incr(_)',
            "exceeds the limit of 4096 reachable rule instances",
            id="reachable-instance-limit",
        ),
        pytest.param(
            'start: "ok"\nunused: counter::0\n'
            'counter::_: "x" counter::incr([0:13]) %if lt([0:13], 4096)\n'
            '          | "" %if eq([0:13], 4096)',
            "exceeds the limit of 4096 reachable rule instances",
            id="unused-ordinary-rule-instance-limit",
        ),
        pytest.param(
            'start: "ok"\nunused: state::0\n'
            'state::_: "ok" %if bit_clear(0) | missing %if bit_set(0)',
            "unknown name 'missing'",
            id="unused-ordinary-rule-false-branch-validation",
        ),
    ],
)
def test_lark_parametric_validation_errors(grammar: str, message: str) -> None:
    _assert_lark_error(grammar, message)


def test_lark_allow_invalid_utf8_dot_flags_and_unicode_default() -> None:
    option = '%grammar_options {"allow_invalid_utf8": true}\n'
    _assert_byte_language(
        option + "start: /./",
        accepted=[b"a", b"\x80", b"\xc2"],
        rejected=[b"", b"\n", b"\xc2\xa2", "é".encode()],
    )
    _assert_byte_language(
        option + "start: /a./isu",
        accepted=[b"A\n", b"a\x80"],
        rejected=[b"a", b"a\xc2\xa2", b"b\n"],
    )
    _assert_byte_language(
        "start: /./",
        accepted=[b"a", b"\xc2\xa2", "é".encode()],
        rejected=[b"", b"\n", b"\x80", b"\xc2"],
    )


def test_lark_allow_invalid_utf8_byte_classes_escapes_and_literals() -> None:
    option = '%grammar_options {"allow_invalid_utf8": true}\n'
    _assert_byte_language(
        option + r"start: /[\x80-\xFF]+/",
        accepted=[b"\x80", b"\xff", b"\xc2\xa2", "é".encode()],
        rejected=[b"", b"a", b"\x7f"],
    )
    _assert_byte_language(
        option + r"start: /[^\x00-\x7F]/",
        accepted=[b"\x80", b"\xff"],
        rejected=[b"", b"a", b"\xc2\xa2"],
    )
    _assert_byte_language(
        option + r"start: /\x80\xFFé/",
        accepted=[b"\x80\xff\xc3\xa9"],
        rejected=[b"\x80\xff", b"\xc2\x80\xff\xc3\xa9"],
    )
    _assert_byte_language(
        option + r"start: /[^\x00-\xFF]/", accepted=[], rejected=[b"", b"a", b"\x80", b"\xff"]
    )


@pytest.mark.parametrize(
    "pattern, accepted, rejected",
    [
        (r"\d", [b"0"], [b"a", b"\x80"]),
        (r"\D", [b"a", b"\x80"], [b"0", b"ab"]),
        (r"\w", [b"a", b"A", b"0", b"_"], [b" ", b"\x80"]),
        (r"\W", [b" ", b"!", b"\x80"], [b"a", b"_", b"ab"]),
        (r"\s", [b" ", b"\t", b"\n", b"\r", b"\x0b", b"\x0c"], [b"a", b"\x80"]),
        (r"\S", [b"a", b"!", b"\x80"], [b" ", b"\t", b"\n"]),
    ],
)
def test_lark_allow_invalid_utf8_ascii_shorthand_classes(
    pattern: str, accepted: Sequence[bytes], rejected: Sequence[bytes]
) -> None:
    grammar = f'%grammar_options {{"allow_invalid_utf8": true}}\nstart: /{pattern}/'
    _assert_byte_language(grammar, accepted, rejected)


def test_lark_allow_invalid_utf8_empty_groups_and_large_repeat_subrules() -> None:
    option = '%grammar_options {"allow_invalid_utf8": true}\n'
    # Regression: discarding () would make '*' bind to the preceding 'a', turning this into a*.
    _assert_byte_language(option + "start: /a()*/", accepted=[b"a"], rejected=[b"", b"aa"])
    _assert_byte_language(
        option + "start: /(?:){2,50000}a/iu", accepted=[b"a", b"A"], rejected=[b"", b"aa"]
    )
    _assert_byte_language(
        option + "start: /(ab){2,50000}c/iu",
        accepted=[b"abABc", b"aB" * 150 + b"C"],
        rejected=[b"abc", b"c", b"ab" * 150],
    )


def test_lark_allow_invalid_utf8_options_ignore_and_nested_scope() -> None:
    grammar = r"""
        %grammar_options {"allow_invalid_utf8": false}
        %grammar_options {"allow_invalid_utf8": true}
        %grammar_options {"allow_invalid_utf8": false}
        %ignore /[\x80-\xFF]+/
        start: "a" /\x00/ /[^a]/ "z"
    """
    _assert_byte_language(
        grammar,
        accepted=[b"a\x00!z", b"a\x80\x00!\xffz"],
        rejected=[b"a\x00az", b"\x80a\x00!z", b"a\x00!"],
    )

    outer_byte_mode = r"""
        %grammar_options {"allow_invalid_utf8": true}
        start: "A" %lark {
          start: /./
        } "B"
    """
    _assert_byte_language(
        outer_byte_mode, accepted=[b"AaB", b"A\xc2\xa2B"], rejected=[b"A\x80B", b"A\xc2B"]
    )

    nested_byte_mode = r"""
        start: "A" %lark {
          %grammar_options {"allow_invalid_utf8": true}
          start: /./
        } "B"
    """
    _assert_byte_language(
        nested_byte_mode, accepted=[b"AaB", b"A\x80B", b"A\xc2B"], rejected=[b"A\xc2\xa2B"]
    )

    byte_regex_suffix = r"""
        %grammar_options {"allow_invalid_utf8": true}
        start: head
        head[suffix=/\x80/]: /./s
    """
    _assert_byte_language(
        byte_regex_suffix,
        accepted=[b"a\x80", b"\xff\x80"],
        rejected=[b"", b"\x80", b"aa\x80", b"a\xc2\x80"],
    )
    _assert_lark_error(
        '%grammar_options {"allow_invalid_utf8": true}\nstart: "a".."é"',
        "non-ASCII character ranges are not available",
    )


def test_lark_allow_invalid_utf8_round_trips_with_structured_regex() -> None:
    source = r"""
        %grammar_options {"allow_invalid_utf8": true}
        start: RAW | SUB
        RAW: /\x80(?:[^\x00-\x7F]|\x00){1,2}/
        SUB: %regex {"substring_chunks":["ab","cd"]}
    """
    grammar = xgr.Grammar.from_lark(source)
    restored_ebnf = xgr.Grammar.from_ebnf(str(grammar))
    restored_json = xgr.Grammar.deserialize_json(grammar.serialize_json())
    accepted = [b"", b"ab", b"cd", b"abcd", b"\x80\x80", b"\x80\xff\x00"]
    rejected = [b"a", b"ad", b"\x80", b"\x80a", b"\x80\x00a"]
    assert "byte_mode=true" in str(grammar)
    assert "Substring(" in str(grammar)
    for candidate in [grammar, restored_ebnf, restored_json]:
        _assert_byte_language(candidate, accepted, rejected)


@pytest.mark.parametrize(
    "pattern, message",
    [
        (r"\p{L}", "Unicode character classes are not available"),
        (r"\P{Letter}", "Unicode character classes are not available"),
        (r"[é]", "non-ASCII characters are not available in byte character classes"),
        (r"a^b", "start anchor is only allowed at the beginning"),
        (r"a$b", "end anchor is only allowed at the end"),
        (r"\bword\b", "word-boundary assertions are not supported"),
        (r"(?=a)", "lookaround assertions are not supported"),
        (r"\1", "backreferences are not supported"),
        (r"\x{80}", "Unicode character escapes are not available"),
        (r"\xG0", "must contain exactly two hexadecimal digits"),
        (r"[a-\d]", "range endpoint must be a single byte"),
        (r"a{2,1}", "lower bound 2 is larger than the upper bound 1"),
        (r"\q", "unrecognized byte escape"),
    ],
)
def test_lark_allow_invalid_utf8_regex_diagnostics(pattern: str, message: str) -> None:
    _assert_lark_error(
        f'%grammar_options {{"allow_invalid_utf8": true}}\nstart: /{pattern}/', message
    )


def test_lark_byte_mode_parametric_branches_and_serialization() -> None:
    source = r"""
        %grammar_options {"allow_invalid_utf8": true}
        start: state::0
        state::_: /\x80/ state::set_bit(0) %if bit_clear(0)
               | /\xFF/ %if bit_set(0)
               | /\p{L}/ %if bit_set(1)
    """
    grammar = xgr.Grammar.from_lark(source)
    serialized = grammar.serialize_json()
    assert json.loads(serialized)["__VERSION__"] == xgr.get_serialization_version()
    assert "byte_mode=true" in str(grammar)

    for candidate in [
        grammar,
        xgr.Grammar.from_ebnf(str(grammar)),
        xgr.Grammar.deserialize_json(serialized),
    ]:
        _assert_byte_language(
            candidate, accepted=[b"\x80\xff"], rejected=[b"", b"\x80", b"\xff", b"\xc2\x80\xff"]
        )

    # Byte-regex semantics are applied only after parameter expansion. The invalid byte-dialect
    # property escape is harmless while its branch is dead, but is diagnosed when state 2 keeps it.
    _assert_lark_error(source.replace("state::0", "state::2", 1), "Unicode character classes")


def test_lark_byte_mode_parametric_exactly_4096_instances() -> None:
    source = r"""
        %grammar_options {"allow_invalid_utf8": true}
        start: counter::0
        counter::_: /\x80/ counter::incr([0:12]) %if lt([0:12], 4095)
                 | /\xFF/ %if eq([0:12], 4095)
    """
    grammar = xgr.Grammar.from_lark(source)
    assert str(grammar).count("byte_mode=true") == 4096
    _assert_byte_language(
        grammar,
        accepted=[b"\x80" * 4095 + b"\xff"],
        rejected=[b"\x80" * 4094 + b"\xff", b"\x80" * 4095 + b"a"],
    )


def test_lark_byte_mode_parametric_nested_options_are_isolated() -> None:
    outer_byte = r"""
        %grammar_options {"allow_invalid_utf8": true}
        start: outer::0
        outer::_: /\x80/ %lark {
            start: inner::0
            inner::_: /./ %if eq(_, 0)
        } %if eq(_, 0)
    """
    _assert_byte_language(
        outer_byte, accepted=[b"\x80a", b"\x80\xc3\xa9"], rejected=[b"\x80\x80", b"\x80\xc3"]
    )

    inner_byte = r"""
        start: /./ %lark {
            %grammar_options {"allow_invalid_utf8": true}
            start: inner::0
            inner::_: /\x80/ %if eq(_, 0)
        }
    """
    _assert_byte_language(
        inner_byte, accepted=[b"a\x80", b"\xc3\xa9\x80"], rejected=[b"\x80\x80", b"\xc3\x80"]
    )


def test_lark_byte_mode_validate_tokens_stop_override_and_state_isolation() -> None:
    vocabulary = [b"\x80", b"\xc3", b"\xa9", b"x", b"y", b"\xff", b"<old>", b"", b"z"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[6])
    grammar = xgr.Grammar.from_lark(
        r"""
        %grammar_options {"allow_invalid_utf8": true}
        start[capture="raw"]: item::0 tail
        item::_: /\x80/ %if eq(_, 0)
              | /\xC3\xA9/ %if eq(_, 0)
        tail: %regex {"substring_chars":"xy"}
        """,
        tokenizer_info=tokenizer_info,
    )
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, override_stop_tokens=[5])
    initial_state = matcher._debug_print_internal_state()

    assert matcher.validate_tokens([0, 3, 4, 5, 8]) == 4
    assert matcher.validate_tokens([1, 2, 4, 5]) == 4
    assert matcher.validate_tokens([1, 5]) == 1
    assert matcher.validate_tokens([2]) == 0
    assert matcher.validate_tokens([0, 4, 3]) == 2
    assert matcher.validate_tokens([5]) == 0
    assert matcher.validate_tokens([6]) == 0
    assert matcher.validate_tokens([7]) == 0
    assert matcher.validate_tokens([-1]) == 0
    assert matcher.validate_tokens([len(vocabulary)]) == 0
    assert matcher.validate_tokens([1 << 40]) == 0
    assert matcher._debug_print_internal_state() == initial_state
    assert matcher.get_captures() == []

    assert matcher.accept_token(1)
    partial_state = matcher._debug_print_internal_state()
    assert matcher.validate_tokens([2, 5]) == 2
    assert matcher.validate_tokens([0, 5]) == 0
    assert matcher._debug_print_internal_state() == partial_state
    matcher.rollback(1)
    assert matcher._debug_print_internal_state() == initial_state

    for token_id in [0, 3, 4]:
        assert matcher.accept_token(token_id)
    assert matcher.is_completed() and not matcher.is_terminated()
    assert matcher.get_captures() == [("raw", b"\x80xy")]
    completed_state = matcher._debug_print_internal_state()
    assert matcher.validate_tokens([5, 8]) == 1
    assert matcher._debug_print_internal_state() == completed_state
    assert matcher.accept_token(5) and matcher.is_terminated()
    matcher.rollback(1)
    assert matcher.is_completed() and not matcher.is_terminated()
    assert matcher.get_captures() == [("raw", b"\x80xy")]


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


def test_lark_structured_regex_substring_chunks() -> None:
    grammar = r"""
        start: "A" SUB "B"
        SUB: %regex {"substring_chunks":["abc","de","fg"]}
    """
    _assert_language(
        grammar,
        ["AB", "AabcB", "AdeB", "AfgB", "AabcdeB", "AdefgB", "AabcdefgB"],
        ["AabB", "AcdeB", "AabcfgB", "AdeabcB"],
    )


def test_lark_structured_regex_substring_repeated_and_empty_chunks() -> None:
    grammar = r'start: %regex {"substring_chunks":["a","","b","a","b"]}'
    _assert_language(
        grammar, ["", "a", "b", "ab", "ba", "aba", "bab", "abab"], ["aa", "bb", "abba", "baba"]
    )


def test_lark_structured_regex_substring_chars_ascii_and_unicode() -> None:
    _assert_language(
        'start: %regex {"substring_chars":"The fox."}',
        ["", "The fox.", "he fo", "fox."],
        ["Thefox", "he fx", "The fox.."],
    )
    _assert_language(
        'start: %regex {"substring_chars":"a빠🙂b"}',
        ["", "a", "빠", "🙂", "빠🙂", "🙂b", "a빠🙂b"],
        ["a🙂", "빠b", "a빠b"],
    )


def test_lark_structured_regex_substring_chars_preserves_nul() -> None:
    grammar = r'start: %regex {"substring_chars":"a\u0000b"}'
    _assert_language(grammar, ["", "a", "\0", "b", "a\0", "\0b", "a\0b"], ["ab", "a\0\0b"])


def test_lark_structured_regex_round_trip_and_serialization() -> None:
    grammar = xgr.Grammar.from_lark(
        'start: "A" %regex {"substring_chunks":["foo"," bar"," baz"]} "B"'
    )
    accepted = ["AB", "AfooB", "A bar bazB", "Afoo bar bazB"]
    rejected = ["AfoB", "Afoo bazB"]
    _assert_grammar_language(grammar, accepted, rejected)
    _assert_grammar_language(xgr.Grammar.from_ebnf(str(grammar)), accepted, rejected)
    _assert_grammar_language(
        xgr.Grammar.deserialize_json(grammar.serialize_json()), accepted, rejected
    )


def test_lark_structured_regex_large_substring_chars() -> None:
    source = "".join(chr(0x1000 + index) for index in range(4_000))
    grammar = xgr.Grammar.from_lark(f'start: %regex {{"substring_chars":{json.dumps(source)}}}')
    candidate = source[50:100]
    _assert_grammar_language(grammar, [candidate], [candidate + "missing"])


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


def test_lark_dynamic_lazy_dotall_unicode_regex_flags() -> None:
    grammar = r"""
        start: tool* tail
        tail: TEXT
        head[lazy]: /.*<call>/su
        tool: head "ok" "</call>"
        TEXT: /.*/su
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


def test_lark_lazy_rule_starred_terminal() -> None:
    grammar = r"""
        start: head "!"
        head[lazy]: TEXT* "<end>"
        TEXT: /[a-z]/
    """
    _assert_language(grammar, ["<end>!", "abc<end>!"], ["ABC<end>!", "a<end>b<end>!", "abc<end>"])


def test_lark_lazy_rule_starred_any_text() -> None:
    grammar = r"""
        start: head "!"
        head[lazy]: TEXT* "<end>"
        TEXT: /(\n|.)*/
    """
    _assert_language(
        grammar, ["<end>!", "abc<end>!", "a\nb<end>!", "你好<end>!"], ["a<end>b<end>!"]
    )


def test_lark_lazy_rule_plus_terminal() -> None:
    grammar = r"""
        start: head "!"
        head[lazy]: TEXT+ "<end>"
        TEXT: /[a-z]/
    """
    _assert_language(grammar, ["a<end>!", "abc<end>!"], ["<end>!", "a<end>b<end>!"])


def test_lark_lazy_rule_quantified_alternation() -> None:
    grammar = r"""
        start: head "!"
        head[lazy]: AB+ "<end>"
        AB: "a" | "b"
    """
    _assert_language(grammar, ["ab<end>!", "b<end>!"], ["<end>!", "c<end>!", "a<end>b<end>!"])
    _assert_language(
        'start: head "!"\nhead[lazy]: ("a"|"b")* "<end>"',
        ["<end>!", "ba<end>!"],
        ["c<end>!", "a<end>b<end>!"],
    )


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


def test_lark_regex_flags_ebnf_and_serialization_round_trip() -> None:
    grammar = xgr.Grammar.from_lark("start: /Żółw[^k]/isu")
    accepted = ["Żółwz", "ŻółwZ", "Żółw\n"]
    rejected = ["żółwz", "Żółwk", "ŻółwK", "Żółw"]
    _assert_grammar_language(grammar, accepted, rejected)

    ebnf_restored = xgr.Grammar.from_ebnf(str(grammar))
    _assert_grammar_language(ebnf_restored, accepted, rejected)

    json_restored = xgr.Grammar.deserialize_json(grammar.serialize_json())
    _assert_grammar_language(json_restored, accepted, rejected)


def test_lark_regex_large_repeat_ebnf_and_serialization_round_trip() -> None:
    # The large repetition is compiled into a repeat subrule at automaton build time; the
    # grammar representation (and thus printing and serialization) keeps the raw pattern.
    grammar = xgr.Grammar.from_lark("start: /(ab){2,50000}c/i")
    accepted = ["abABc", "aB" * 150 + "c"]
    rejected = ["abc", "c", "ab" * 150]
    _assert_grammar_language(grammar, accepted, rejected)

    ebnf_restored = xgr.Grammar.from_ebnf(str(grammar))
    _assert_grammar_language(ebnf_restored, accepted, rejected)

    json_restored = xgr.Grammar.deserialize_json(grammar.serialize_json())
    _assert_grammar_language(json_restored, accepted, rejected)


def test_lark_regex_case_insensitive_token_bitmask() -> None:
    from xgrammar.testing import _get_masked_tokens_from_bitmask

    tokenizer_info = xgr.TokenizerInfo(["ab", "AB", "aB", "cd", "C", "c"])
    compiled = _compile_lark("start: /(ab){2,300}c/i", tokenizer_info)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    # At the start only the "ab"-like tokens may begin the repetition.
    matcher.fill_next_token_bitmask(bitmask)
    assert set(_get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)) == {3, 4, 5}
    assert matcher.accept_token(0)  # "ab"

    # After one repetition the lower bound 2 is not reached yet, so "c" stays forbidden.
    matcher.fill_next_token_bitmask(bitmask)
    assert set(_get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)) == {3, 4, 5}
    assert matcher.accept_token(1)  # "AB"

    # Now the tail "c" (case-folded) becomes possible; "cd" is still impossible.
    matcher.fill_next_token_bitmask(bitmask)
    assert set(_get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)) == {3}
    assert matcher.accept_token(4)  # "C"
    assert matcher.is_terminated()


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
        pytest.param('start::0: "a"', "expected '_' after '::'", id="parametric-definition"),
        pytest.param("start: foo::0", "unknown name 'foo'", id="parametric-reference"),
        pytest.param('start: "a" %if enabled', "unknown condition 'enabled'", id="parametric-if"),
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
            "start: /abc/m",
            "regular-expression flag 'm' is not supported",
            id="unsupported-regex-flag",
        ),
        pytest.param(
            "start: /abc/l", "regular-expression flag 'l' is not supported", id="unsupported-l-flag"
        ),
        pytest.param(
            "start: /abc/x",
            "regular-expression flag 'x' is not supported",
            id="unsupported-verbose-regex-flag",
        ),
        pytest.param(
            "start: item\nitem[suffix=/end/i]: /[a-z]*/",
            "regular-expression flag 'i' is not supported with suffix or stop attributes",
            id="case-insensitive-regex-suffix",
        ),
        pytest.param(
            'start: "a"i.."z"', "flags are not allowed on character ranges", id="range-start-flag"
        ),
        pytest.param(
            'start: "a".."z"i', "flags are not allowed on character ranges", id="range-end-flag"
        ),
        pytest.param("start: /[abc/", "failed to compile regular expression", id="invalid-regex"),
        pytest.param(r"start: /a\b/i", "Word boundary assertion", id="regex-word-boundary-error"),
        pytest.param(
            r"start: /\p{L}/i", "Unicode property escape", id="regex-unicode-property-error"
        ),
        pytest.param(r"start: /(a)\1/i", "Backreference", id="regex-backreference-error"),
        pytest.param("start: /[]/i", "Empty character class", id="regex-empty-class-error"),
        pytest.param("start: /(?<=a)b/i", "Lookbehind assertion", id="regex-lookbehind-error"),
        pytest.param(
            r"start: /\uZZ/i",
            "must be followed by four hexadecimal digits",
            id="regex-bad-unicode-escape-error",
        ),
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
            'start: %regex {"substring_words":"abc def"}',
            "substring_words is not supported yet",
            id="substring-words",
        ),
        pytest.param(
            "start: %regex []", "%regex value must be an object", id="structured-regex-not-object"
        ),
        pytest.param(
            'start: %regex {"unknown":"abc"}',
            "unknown field 'unknown' in %regex",
            id="structured-regex-unknown-field",
        ),
        pytest.param(
            "start: %regex {}", "no fields set on %regex", id="structured-regex-no-fields"
        ),
        pytest.param(
            'start: %regex {"substring_chars":"abc","substring_chunks":["a"]}',
            "only one field can be set on %regex",
            id="structured-regex-multiple-fields",
        ),
        pytest.param(
            'start: %regex {"substring_chars":["a"]}',
            "substring_chars must be a string",
            id="structured-regex-chars-type",
        ),
        pytest.param(
            'start: %regex {"substring_chunks":"abc"}',
            "substring_chunks must be an array of strings",
            id="structured-regex-chunks-type",
        ),
        pytest.param(
            'start: %regex {"substring_chunks":["a",1]}',
            "substring_chunks must be an array of strings",
            id="structured-regex-chunk-type",
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
            '%grammar_options {"allow_invalid_utf8": 1}\nstart: "a"',
            "allow_invalid_utf8 must be a boolean",
            id="invalid-utf8-option-type",
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
            "start: head\nhead[suffix=end]: TEXT\nTEXT: /(\\n|.)*/",
            "suffix terminal name must be uppercase",
            id="lowercase-suffix-terminal",
        ),
        pytest.param(
            'start: head\nhead[suffix="x",suffix="y"]: TEXT\nTEXT: /(\\n|.)*/',
            "suffix attribute is specified more than once",
            id="duplicate-suffix",
        ),
        pytest.param(
            "start: head\nhead[suffix=MISSING]: TEXT\nTEXT: /(\\n|.)*/",
            "unknown name 'MISSING'",
            id="unknown-suffix-terminal",
        ),
        pytest.param(
            'start: head\nhead[stop=""]: TEXT\nTEXT: /(\\n|.)*/',
            "stop must not be empty",
            id="empty-stop",
        ),
        pytest.param(
            "start: head\nhead[stop=end]: TEXT\nTEXT: /(\\n|.)*/",
            "stop terminal name must be uppercase",
            id="lowercase-stop-terminal",
        ),
        pytest.param(
            'start: head\nhead[stop="x",stop="y"]: TEXT\nTEXT: /(\\n|.)*/',
            "stop attribute is specified more than once",
            id="duplicate-stop",
        ),
        pytest.param(
            'start: head\nhead[stop_capture="marker"]: TEXT\nTEXT: /(\\n|.)*/',
            "stop_capture requires stop or suffix",
            id="stop-capture-without-marker",
        ),
        pytest.param(
            'start: head\nhead[stop="x", stop_capture="bad name"]: TEXT\nTEXT: /(\\n|.)*/',
            "capture name must only contain",
            id="invalid-stop-capture-name",
        ),
        pytest.param(
            'start: head\nhead[stop="x", stop_capture=MARKER]: TEXT\nTEXT: /(\\n|.)*/',
            "expected string literal after stop_capture=",
            id="non-string-stop-capture",
        ),
        pytest.param(
            'start: head\nhead[stop="x", stop_capture="a", stop_capture="b"]: TEXT\n'
            "TEXT: /(\\n|.)*/",
            "stop_capture attribute is specified more than once",
            id="duplicate-stop-capture",
        ),
        pytest.param(
            'start: head\nhead[stop="x", suffix="y"]: TEXT\nTEXT: /(\\n|.)*/',
            "suffix cannot be combined with stop",
            id="stop-then-suffix",
        ),
        pytest.param(
            'start: head\nhead[suffix="y", stop="x"]: TEXT\nTEXT: /(\\n|.)*/',
            "stop cannot be combined with suffix",
            id="suffix-then-stop",
        ),
        pytest.param(
            'STOP[stop="x"]: /a/\nstart: STOP',
            "attributes are only supported on rules",
            id="stop-on-terminal",
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


def test_lark_budget_shared_subrule_keeps_parent_contexts_separate() -> None:
    tokenizer_info = xgr.TokenizerInfo(["p", "a", "b", "X", "Y"])
    compiled = _compile_lark(
        """
        start: tight "X" | loose "Y"
        tight[max_tokens=2]: shared
        loose: shared
        shared: "p" sub
        sub: /[ab]+/
        """,
        tokenizer_info,
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)

    assert matcher.accept_token(0)
    assert matcher.accept_token(1)
    assert 2 in _allowed_token_ids(matcher, tokenizer_info)
    assert matcher.accept_token(2)
    assert _allowed_token_ids(matcher, tokenizer_info) == [1, 2, 4]
    assert not matcher.accept_token(3)
    assert matcher.accept_token(4) and matcher.is_terminated()


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


def test_lark_max_tokens_with_lazy_marker_or_budget_wins() -> None:
    tokenizer_info = xgr.TokenizerInfo(["x", "!", "z", "x!"])
    grammar = xgr.Grammar.from_lark(
        'start: head "z"\nhead[max_tokens=2, lazy, capture]: TEXT "!"\nTEXT: /(\\n|.)*/',
        tokenizer_info=tokenizer_info,
    )
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)

    # The lazy marker arrives first.
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(3)
    assert _allowed_token_ids(matcher, tokenizer_info) == [2]
    assert matcher.accept_token(2) and matcher.is_terminated()
    assert matcher.get_captures() == [("head", b"x!")]

    # Otherwise max_tokens closes the lazy region at its body boundary.
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(0) and matcher.accept_token(0)
    assert _allowed_token_ids(matcher, tokenizer_info) == [2]
    assert matcher.accept_token(2) and matcher.is_terminated()
    assert matcher.get_captures() == [("head", b"xx")]


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
        xgr.Grammar.from_lark('start: "<" r ">"\nr[lazy]: T* "x"\nT: /ab/'),
        xgr.Grammar.from_ebnf('root ::= "<" r ">"\nr[lazy] ::= sub\nsub ::= sub [a-z] | [a-z]'),
    ):
        with pytest.raises(RuntimeError, match="terminal-like"):
            xgr.GrammarCompiler(LAZY_TOKENIZER, cache_enabled=False).compile_grammar(grammar_obj)


# Tokens: 0 "<", 1 ">", 2 "a", 3 "b", 4 "ab", 5 "a>", 6 "ab>", 7 "b>", 8 "bb", 9 " "
LAZY_MASK_TOKENIZER = xgr.TokenizerInfo(["<", ">", "a", "b", "ab", "a>", "ab>", "b>", "bb", " "])


def _lazy_mask_matcher(grammar_obj: xgr.Grammar) -> xgr.GrammarMatcher:
    compiled = xgr.GrammarCompiler(LAZY_MASK_TOKENIZER, cache_enabled=False).compile_grammar(
        grammar_obj
    )
    return xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)


def _mask_allowed_token_ids(matcher: xgr.GrammarMatcher) -> list:
    bitmask = xgr.allocate_token_bitmask(1, LAZY_MASK_TOKENIZER.vocab_size)
    matcher.fill_next_token_bitmask(bitmask)
    return [
        i
        for i in range(LAZY_MASK_TOKENIZER.vocab_size)
        if (int(bitmask[0, i // 32]) >> (i % 32)) & 1
    ]


def test_lark_lazy_mask_commit_inside_token() -> None:
    # A token may cross the commit point: the region char commits the rule mid-token and the
    # rest of the token must match what follows the rule.
    grammar_obj = xgr.Grammar.from_lark('start: "<" r ">"\nr[lazy]: /[a-z]+/')
    matcher = _lazy_mask_matcher(grammar_obj)
    assert matcher.accept_token(0)
    # "a", "b" open the region; "a>"/"b>" commit mid-token and exit. "ab" and "ab>" would
    # extend the region past the commit; ">" needs a non-empty region first.
    assert _mask_allowed_token_ids(matcher) == [2, 3, 5, 7]
    # Refilling must give the identical mask.
    assert _mask_allowed_token_ids(matcher) == [2, 3, 5, 7]
    assert matcher.accept_token(5) and matcher.is_terminated()


def test_lark_lazy_mask_exit_only_after_commit() -> None:
    grammar_obj = xgr.Grammar.from_lark('start: "<" r ">"\nr[lazy]: /[a-z]+/')
    matcher = _lazy_mask_matcher(grammar_obj)
    assert matcher.accept_token(0) and matcher.accept_token(2)
    assert _mask_allowed_token_ids(matcher) == [1]
    assert matcher.accept_token(1) and matcher.is_terminated()


def test_lark_lazy_mask_greedy_control() -> None:
    # The same grammar without lazy: region tokens may extend freely.
    grammar_obj = xgr.Grammar.from_lark('start: "<" r ">"\nr: /[a-z]+/')
    matcher = _lazy_mask_matcher(grammar_obj)
    assert matcher.accept_token(0)
    assert _mask_allowed_token_ids(matcher) == [2, 3, 4, 5, 6, 7, 8]
    assert matcher.accept_token(2)
    assert _mask_allowed_token_ids(matcher) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_lark_lazy_mask_choices_commit_kills_longer_alternative() -> None:
    # After "<a", "b" completes the "ab" alternative and the commit removes the "abb" branch
    # of the same occurrence: "bb" must be masked out, "b"/"b>" stay legal.
    grammar_obj = xgr.Grammar.from_lark('start: "<" r ">"\nr[lazy]: "ab" | "abb"')
    matcher = _lazy_mask_matcher(grammar_obj)
    assert matcher.accept_token(0) and matcher.accept_token(2)
    assert _mask_allowed_token_ids(matcher) == [3, 7]
    assert not matcher.accept_token(8)
    assert matcher.accept_token(7) and matcher.is_terminated()


def test_lark_lazy_mask_fresh_budget_per_occurrence() -> None:
    grammar_obj = xgr.Grammar.from_lark('start: r " " r\nr[lazy]: /[a-z]+/')
    matcher = _lazy_mask_matcher(grammar_obj)
    # First occurrence commits after one char: only the separator remains.
    assert matcher.accept_token(2)
    assert _mask_allowed_token_ids(matcher) == [9]
    # The second occurrence is fresh: region chars are allowed again, but still commit at one
    # char ("ab" spans two region chars and stays masked out).
    assert matcher.accept_token(9)
    assert _mask_allowed_token_ids(matcher) == [2, 3]
    assert matcher.accept_token(3) and matcher.is_terminated()


def test_ebnf_lazy_mask_matches_lark_form() -> None:
    grammar_obj = xgr.Grammar.from_ebnf('root ::= "<" r ">"\nr[lazy] ::= [a-z] [a-z]*')
    matcher = _lazy_mask_matcher(grammar_obj)
    assert matcher.accept_token(0)
    assert _mask_allowed_token_ids(matcher) == [2, 3, 5, 7]
    assert matcher.accept_token(2)
    assert _mask_allowed_token_ids(matcher) == [1]


# Combined attributes: lazy + max_tokens + capture interacting in one grammar.

COMBINED_TOKENIZER = xgr.TokenizerInfo(["x", "<t>", ">", "a", "b", "ab"])


def _combined_allowed_token_ids(matcher: xgr.GrammarMatcher) -> list:
    bitmask = xgr.allocate_token_bitmask(1, COMBINED_TOKENIZER.vocab_size)
    matcher.fill_next_token_bitmask(bitmask)
    return [
        i
        for i in range(COMBINED_TOKENIZER.vocab_size)
        if (int(bitmask[0, i // 32]) >> (i % 32)) & 1
    ]


ALL_THREE_GRAMMAR = r"""
    start: reasoning "<t>" val ">"
    reasoning[max_tokens=2, capture]: TEXT
    val[lazy, capture="value"]: /[ab]+/
    TEXT: /(\n|.)*/
"""


def test_lark_lazy_with_capture_records_committed_span() -> None:
    # The lazy commit makes the capture exact: the occurrence cannot complete again past the
    # committed end, so only the shortest span is recorded.
    grammar = "start: r rest\nr[lazy, capture]: /[a-z]+/\nrest: /[a-z]*/"
    compiled = _compile_lark(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string("abc")
    assert matcher.get_captures() == [("r", b"a")]
    # Greedy control: without lazy, coalescing keeps the longest completion of the occurrence.
    compiled = _compile_lark("start: r rest\nr[capture]: /[a-z]+/\nrest: /[a-z]*/")
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string("abc")
    assert matcher.get_captures() == [("r", b"abc")]


def test_lark_lazy_with_capture_round_trip_and_cache() -> None:
    grammar_obj = xgr.Grammar.from_lark(
        'start: "<" r ">"\nr[lazy, capture="x"]: /[a-z]+/', tokenizer_info=COMBINED_TOKENIZER
    )
    printed = str(grammar_obj)
    assert 'r[capture="x", lazy] ::=' in printed
    assert 'r[capture="x", lazy] ::=' in str(xgr.Grammar.from_ebnf(printed))
    deserialized = xgr.Grammar.deserialize_json(grammar_obj.serialize_json())
    assert str(deserialized) == printed
    # The compiler cache path re-parses ToString(); both attributes must survive it.
    compiler = xgr.GrammarCompiler(COMBINED_TOKENIZER, cache_enabled=True)
    compiled = compiler.compile_grammar(grammar_obj)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert not matcher.accept_string("<ab>")
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string("<a>") and matcher.is_terminated()
    assert matcher.get_captures() == [("x", b"a")]


def test_ebnf_all_three_attributes_round_trip() -> None:
    # The EBNF frontend round-trips all three attributes in one bracket group, in any order.
    ebnf = 'root ::= "<" r ">"\nr[max_tokens=3, capture="x", lazy] ::= [a-z] [a-z]*'
    grammar_obj = xgr.Grammar.from_ebnf(ebnf)
    assert 'r[max_tokens=3, capture="x", lazy] ::=' in str(grammar_obj)
    reordered = xgr.Grammar.from_ebnf(
        'root ::= "<" r ">"\nr[lazy, capture="x", max_tokens=3] ::= [a-z] [a-z]*'
    )
    assert str(reordered) == str(grammar_obj)
    deserialized = xgr.Grammar.deserialize_json(grammar_obj.serialize_json())
    assert str(deserialized) == str(grammar_obj)


def test_lark_all_three_features_in_one_grammar_masks() -> None:
    # Tokens: 0 "x", 1 "<t>", 2 ">", 3 "a", 4 "b", 5 "ab"
    compiled = xgr.GrammarCompiler(COMBINED_TOKENIZER, cache_enabled=False).compile_grammar(
        xgr.Grammar.from_lark(ALL_THREE_GRAMMAR, tokenizer_info=COMBINED_TOKENIZER)
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(0) and matcher.accept_token(0)
    # Budget of the reasoning region exhausted: the mask only allows leaving it.
    assert _combined_allowed_token_ids(matcher) == [1]
    assert matcher.accept_token(1)
    # Inside the lazy value: single region chars are allowed, "ab" would extend the occurrence
    # past its commit point.
    assert _combined_allowed_token_ids(matcher) == [3, 4]
    assert matcher.accept_token(3)
    # The lazy commit: only the closing literal remains.
    assert _combined_allowed_token_ids(matcher) == [2]
    assert matcher.accept_token(2) and matcher.is_terminated()
    assert matcher.get_captures() == [("reasoning", b"xx"), ("value", b"a")]


def test_lark_all_three_features_rollback_and_fork() -> None:
    compiled = xgr.GrammarCompiler(COMBINED_TOKENIZER, cache_enabled=False).compile_grammar(
        xgr.Grammar.from_lark(ALL_THREE_GRAMMAR, tokenizer_info=COMBINED_TOKENIZER)
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    # Generation-style loop: the fill before each accept enforces the budget close.
    for token_id in [0, 0, 1, 3]:
        assert token_id in _combined_allowed_token_ids(matcher)
        assert matcher.accept_token(token_id)
    forked = matcher.fork()
    assert _combined_allowed_token_ids(forked) == [2]
    # Rollback across the lazy commit and the budget close, then replay.
    matcher.rollback(2)
    assert _combined_allowed_token_ids(matcher) == [1]
    assert matcher.accept_token(1) and matcher.accept_token(4)
    assert matcher.accept_token(2) and matcher.is_terminated()
    assert matcher.get_captures() == [("reasoning", b"xx"), ("value", b"b")]
    assert forked.accept_token(2) and forked.is_terminated()
    assert forked.get_captures() == [("reasoning", b"xx"), ("value", b"a")]


def test_lark_all_three_features_serialization_round_trip() -> None:
    grammar_obj = xgr.Grammar.from_lark(ALL_THREE_GRAMMAR, tokenizer_info=COMBINED_TOKENIZER)
    printed = str(grammar_obj)
    assert 'reasoning[max_tokens=2, capture="reasoning"] ::=' in printed
    assert 'val[capture="value", lazy] ::=' in printed
    deserialized = xgr.Grammar.deserialize_json(grammar_obj.serialize_json())
    compiled = xgr.GrammarCompiler(COMBINED_TOKENIZER, cache_enabled=False).compile_grammar(
        deserialized
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(0) and matcher.accept_token(0)
    assert _combined_allowed_token_ids(matcher) == [1]
    for token_id in [1, 3, 2]:
        assert matcher.accept_token(token_id)
    assert matcher.is_terminated()
    assert matcher.get_captures() == [("reasoning", b"xx"), ("value", b"a")]


# General suffix/stop and their interaction with lazy, capture, max_tokens, and TagDispatch.


def _captures_for_string(
    grammar: xgr.Grammar,
    value: str,
    tokenizer_info: Optional[xgr.TokenizerInfo] = None,
    *,
    cache_enabled: bool = False,
) -> list:
    tokenizer_info = tokenizer_info or xgr.TokenizerInfo([])
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=cache_enabled).compile_grammar(
        grammar
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string(value) and matcher.is_terminated()
    return matcher.get_captures()


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_general_body_commits_at_first_marker(attribute: str) -> None:
    # The marker is also in the body's alphabet. Committed-shortest matching must select the
    # first marker; it is required for general (non-TagDispatch) bodies.
    grammar = f'start: r "c"\nr[{attribute}="b"]: /[a-z]*/'
    _assert_language(grammar, ["aabc", "bc"], ["abbc", "aac", "abab", "aab"])

    grammar_obj = xgr.Grammar.from_lark(grammar)
    printed = str(grammar_obj)
    assert "r[" in printed and "lazy" in printed
    assert f"capture_hidden_{attribute}_bytes=1" in printed


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_regex_marker(attribute: str) -> None:
    grammar = f"""
        start: "<" r ">"
        r[{attribute}=/!!+/]: /[a-z]{{1,4}}/
    """
    # /!!+/ first becomes accepting after two exclamation marks, so committed-shortest
    # matching leaves any third exclamation mark outside the rule.
    _assert_language(grammar, ["<a!!>", "<abcd!!>"], ["<!!>", "<a!>", "<a!!!>", "<abcde!!>"])


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_named_terminal_marker(attribute: str) -> None:
    grammar = f"""
        start: "<" r ">"
        r[{attribute}=END]: /[a-z]+/
        END: /!!+/ | "??"
    """
    _assert_language(grammar, ["<a!!>", "<word??>"], ["<a!>", "<a!!!>", "<word?>", "<word???>"])


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_case_insensitive_string_marker(attribute: str) -> None:
    grammar = f'start: r "z"\nr[{attribute}="END"i]: /[a-z]*/'
    _assert_language(grammar, ["abcENDz", "abcendz", "abcEnDz"], ["abcENz", "abcENDENDz"])


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_dotall_regex_marker(attribute: str) -> None:
    grammar = f'start: r "z"\nr[{attribute}=/X.Y/s]: /[a-z]*/'
    _assert_language(grammar, ["abcX\nYz", "abcXaYz"], ["abcX\nZz", "abcXYz"])


@pytest.mark.parametrize(
    "attribute, expected_outer",
    [pytest.param("suffix", b"ab!!z", id="suffix"), pytest.param("stop", b"abz", id="stop")],
)
def test_lark_suffix_stop_regex_marker_capture_and_stop_capture(
    attribute: str, expected_outer: bytes
) -> None:
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="outer"]: r "z"
        r[capture="inner", {attribute}=/!!+/, stop_capture="marker"]: /[a-z]*/
        """
    )
    assert _captures_for_string(grammar, "ab!!z") == [
        ("marker", b"!!"),
        ("inner", b"ab"),
        ("outer", expected_outer),
    ]


@pytest.mark.parametrize(
    "attribute, expected_outer",
    [pytest.param("suffix", b"baZ", id="suffix"), pytest.param("stop", b"bZ", id="stop")],
)
def test_lark_suffix_stop_regex_marker_recovers_valid_body_split(
    attribute: str, expected_outer: bytes
) -> None:
    # The marker /b?a/ also matches the entire "ba", but that split would leave the required
    # body "b" empty. Boundary recovery must therefore select b|a rather than |ba.
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="outer"]: r "Z"
        r[capture="inner", {attribute}=/b?a/, stop_capture="marker"]: "b"
        """
    )
    assert _captures_for_string(grammar, "baZ") == [
        ("marker", b"a"),
        ("inner", b"b"),
        ("outer", expected_outer),
    ]


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_nullable_regex_marker(attribute: str) -> None:
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="outer"]: r "z"
        r[capture="inner", {attribute}=/x?/, stop_capture="marker"]: "a"
        """
    )
    # A regex marker may be nullable even though the special empty string literal marker is not.
    assert _captures_for_string(grammar, "az") == [
        ("marker", b""),
        ("inner", b"a"),
        ("outer", b"az"),
    ]


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_terminal_composition(attribute: str) -> None:
    grammar = f"""
        start: "<" r ">"
        r[{attribute}="!"]: AB+
        AB: "a" | "b"
    """
    _assert_language(grammar, ["<a!>", "<ab!>"], ["<!>", "<ab!!>", "<a!b!>"])


@pytest.mark.parametrize(
    "attribute, expected_outer",
    [
        pytest.param("suffix", b"xy!z", id="suffix-own-level-only"),
        pytest.param("stop", b"xyz", id="stop-all-levels"),
    ],
)
def test_lark_suffix_stop_capture_scope(attribute: str, expected_outer: bytes) -> None:
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="outer"]: r "z"
        r[capture="inner", {attribute}="!"]: /[a-z]*/
        """
    )
    assert _captures_for_string(grammar, "xy!z") == [("inner", b"xy"), ("outer", expected_outer)]


@pytest.mark.parametrize("deduplicate", [True, False], ids=["deduplicated", "raw-events"])
def test_lark_stop_capture_scope_ignores_unrelated_earley_branch(deduplicate: bool) -> None:
    grammar = xgr.Grammar.from_lark(
        """
        start: a | c
        a[capture="a"]: b "q"
        b[stop="!"]: "x"
        c[capture="c"]: "x!"
        """
    )
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo([]), cache_enabled=False).compile_grammar(
        grammar
    )
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string("x!") and matcher.is_terminated()

    # b has no capture of its own. Its stop marker belongs only to the enclosing a occurrence,
    # which ultimately fails on the missing "q"; the overlapping successful c branch must keep it.
    assert matcher.get_captures(deduplicate=deduplicate) == [("c", b"x!")]


@pytest.mark.parametrize(
    "attribute, expected_outer",
    [
        pytest.param("suffix", "xy💥z".encode(), id="suffix"),
        pytest.param("stop", b"xyz", id="stop"),
    ],
)
def test_lark_suffix_stop_capture_multibyte_marker(attribute: str, expected_outer: bytes) -> None:
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="outer"]: r "z"
        r[capture="inner", {attribute}="💥"]: /[a-z]*/
        """
    )
    assert _captures_for_string(grammar, "xy💥z") == [("inner", b"xy"), ("outer", expected_outer)]


@pytest.mark.parametrize(
    "attribute, expected_outer",
    [
        pytest.param("suffix", b"a!b!z", id="suffix-keeps-markers-in-parent"),
        pytest.param("stop", b"abz", id="stop-hides-markers-in-parent"),
    ],
)
def test_lark_suffix_stop_hidden_events_without_inner_capture(
    attribute: str, expected_outer: bytes
) -> None:
    # A stop rule must produce hidden-span events even without its own capture name. Repetition
    # also verifies that multiple hidden spans are removed in order and never leak as captures.
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="outer"]: item item "z"
        item[{attribute}="!"]: /[ab]*/
        """
    )
    assert _captures_for_string(grammar, "a!b!z") == [("outer", expected_outer)]


@pytest.mark.parametrize(
    "attribute, marker_expected, plain_expected",
    [
        pytest.param(
            "suffix",
            [("inner", b"foo"), ("outer", b"foo<end>")],
            [("inner", b"plain"), ("outer", b"plain")],
            id="suffix",
        ),
        pytest.param(
            "stop",
            [("inner", b"foo"), ("outer", b"foo")],
            [("inner", b"plain"), ("outer", b"plain")],
            id="stop",
        ),
    ],
)
def test_lark_suffix_stop_any_text_tag_dispatch_capture(
    attribute: str, marker_expected: list, plain_expected: list
) -> None:
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="outer"]: r
        r[capture="inner", {attribute}="<end>"]: TEXT
        TEXT: /(\\n|.)*/
        """
    )
    printed = str(grammar)
    assert "TagDispatch" in printed
    assert "[lazy]" not in printed
    # The no-marker completion must not blindly subtract marker-length bytes.
    assert _captures_for_string(grammar, "plain") == plain_expected
    assert _captures_for_string(grammar, "foo<end>") == marker_expected


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_any_text_tag_dispatch_stop_capture(attribute: str) -> None:
    grammar = xgr.Grammar.from_lark(
        f"""
        start: r
        r[{attribute}="<end>", stop_capture="marker"]: TEXT
        TEXT: /(\\n|.)*/
        """
    )
    # stop_capture alone enables capture tracking. The TagDispatch no-marker completion must not
    # synthesize a capture, while the post-dispatch completion captures exactly the trigger.
    assert _captures_for_string(grammar, "plain") == []
    assert _captures_for_string(grammar, "foo<end>") == [("marker", b"<end>")]


def test_lark_dynamic_fixed_string_stop_attribute() -> None:
    grammar = r"""
        start: tool* tail
        tail: TEXT
        head[stop="<tool>"]: TEXT
        tool: head /[a-z]+/ "</tool>"
        TEXT: /(\n|.)*/
    """
    _assert_language(
        grammar,
        ["free", "x<tool>abc</tool>y", "partial <too"],
        ["<tool>", "<tool>123</tool>", "<tool>abc"],
    )


def test_lark_dynamic_stop_preserves_enclosing_capture() -> None:
    grammar = xgr.Grammar.from_lark(
        r"""
        start[capture="outer"]: tool* tail
        tail: TEXT
        head[stop="<t>"]: TEXT
        tool: head "x"
        TEXT: /(\n|.)*/
        """
    )
    assert "loop_after_dispatch=true" in str(grammar)
    assert _captures_for_string(grammar, "a<t>xb") == [("outer", b"axb")]


@pytest.mark.parametrize(
    "attribute, expected_outer",
    [
        pytest.param("suffix", b"a<t>xb<t>xc", id="suffix"),
        pytest.param("stop", b"axbxc", id="stop"),
    ],
)
def test_lark_dynamic_suffix_stop_preserves_stop_capture(
    attribute: str, expected_outer: bytes
) -> None:
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="outer"]: tool* tail
        tail: TEXT
        head[{attribute}="<t>", stop_capture="marker"]: TEXT
        tool: head "x"
        TEXT: /(\\n|.)*/
        """
    )
    assert "loop_after_dispatch=true" in str(grammar)
    for candidate in (
        grammar,
        xgr.Grammar.from_ebnf(str(grammar)),
        xgr.Grammar.deserialize_json(grammar.serialize_json()),
    ):
        assert _captures_for_string(candidate, "a<t>xb<t>xc", cache_enabled=True) == [
            ("marker", b"<t>"),
            ("marker", b"<t>"),
            ("outer", expected_outer),
        ]


def test_lark_dynamic_stop_hides_marker_from_named_grammar_parent_capture() -> None:
    dynamic = r"""
        start: tool* tail
        tail: TEXT
        head[stop="<t>", stop_capture="marker"]: TEXT
        tool: head "x"
        TEXT: /(\n|.)*/
    """
    grammar = xgr.Grammar.from_lark(
        'start[capture="outer"]: @dynamic', named_grammars={"dynamic": dynamic}
    )
    assert _captures_for_string(grammar, "a<t>xb") == [("marker", b"<t>"), ("outer", b"axb")]


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_mask_commit_and_exit(attribute: str) -> None:
    # LAZY_MASK_TOKENIZER: 0 "<", 1 ">", 2 "a", 3 "b", 4 "ab", 5 "a>", 6 "ab>",
    # 7 "b>", 8 "bb", 9 " ".
    grammar = xgr.Grammar.from_lark(f'start: r "b"\nr[{attribute}=">"]: /[ab]*/')
    matcher = _lazy_mask_matcher(grammar)
    assert _mask_allowed_token_ids(matcher) == [1, 2, 3, 4, 5, 6, 7, 8]
    # A token may cross the committed marker. Once it does, only the rule's following literal
    # remains; suffix and stop are deliberately mask-identical.
    assert matcher.accept_token(6)
    assert _mask_allowed_token_ids(matcher) == [3]
    assert matcher.accept_token(3) and matcher.is_terminated()


@pytest.mark.parametrize(
    "marker, value, marker_capture",
    [
        pytest.param('"!"', "a!z", b"!", id="fixed-marker"),
        pytest.param("/!!+/", "a!!z", b"!!", id="regex-marker"),
    ],
)
@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_atomic_token_crosses_marker_and_following_rule(
    attribute: str, marker: str, value: str, marker_capture: bytes
) -> None:
    tokenizer_info = xgr.TokenizerInfo([value])
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="outer"]: r "z"
        r[capture="inner", {attribute}={marker}, stop_capture="marker"]: /a*/
        """,
        tokenizer_info=tokenizer_info,
    )
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert _allowed_token_ids(matcher, tokenizer_info) == [0]
    assert matcher.accept_token(0) and matcher.is_terminated()
    expected_outer = value.encode() if attribute == "suffix" else b"az"
    assert matcher.get_captures() == [
        ("marker", marker_capture),
        ("inner", b"a"),
        ("outer", expected_outer),
    ]


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_ignore_is_lexeme_scoped(attribute: str) -> None:
    grammar = f'start: "<" r "b"\nr[{attribute}=">"]: /[a-z]*/\n%ignore " "'
    _assert_language(grammar, ["<a>b", "< a> b"], ["<a a>b"])


STOP_SUFFIX_ROLLBACK_TOKENIZER = xgr.TokenizerInfo(["a", "b", "!", "z", "a!", "b!"])
VARIABLE_MARKER_ROLLBACK_TOKENIZER = xgr.TokenizerInfo(["a!!", "b!!", "z", "a!", "!"])


@pytest.mark.parametrize(
    "attribute, first_outer, second_outer",
    [
        pytest.param("suffix", b"a!z", b"b!z", id="suffix"),
        pytest.param("stop", b"az", b"bz", id="stop"),
    ],
)
def test_lark_suffix_stop_capture_rollback_and_fork(
    attribute: str, first_outer: bytes, second_outer: bytes
) -> None:
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="outer"]: r "z"
        r[capture="inner", lazy, {attribute}="!"]: /[ab]*/
        """
    )
    compiled = xgr.GrammarCompiler(
        STOP_SUFFIX_ROLLBACK_TOKENIZER, cache_enabled=False
    ).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert 4 in _allowed_token_ids(matcher, STOP_SUFFIX_ROLLBACK_TOKENIZER)
    assert matcher.accept_token(4)  # "a!" crosses the commit inside one token.
    forked = matcher.fork()

    matcher.rollback(1)
    assert matcher.get_captures() == []
    assert matcher.accept_token(5) and matcher.accept_token(3)
    assert matcher.is_terminated()
    assert matcher.get_captures() == [("inner", b"b"), ("outer", second_outer)]

    assert forked.accept_token(3) and forked.is_terminated()
    assert forked.get_captures() == [("inner", b"a"), ("outer", first_outer)]


@pytest.mark.parametrize(
    "attribute, first_outer, second_outer",
    [
        pytest.param("suffix", b"a!!z", b"b!!z", id="suffix"),
        pytest.param("stop", b"az", b"bz", id="stop"),
    ],
)
def test_lark_suffix_stop_variable_marker_mask_capture_rollback_and_fork(
    attribute: str, first_outer: bytes, second_outer: bytes
) -> None:
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="outer"]: r "z"
        r[capture="inner", {attribute}=/!!+/, stop_capture="marker"]: /[ab]*/
        """
    )
    compiled = xgr.GrammarCompiler(
        VARIABLE_MARKER_ROLLBACK_TOKENIZER, cache_enabled=False
    ).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert 0 in _allowed_token_ids(matcher, VARIABLE_MARKER_ROLLBACK_TOKENIZER)
    assert matcher.accept_token(0)  # "a!!" crosses a variable marker inside one token.
    assert _allowed_token_ids(matcher, VARIABLE_MARKER_ROLLBACK_TOKENIZER) == [2]
    forked = matcher.fork()

    matcher.rollback(1)
    assert matcher.get_captures() == []
    assert matcher.accept_token(1) and matcher.accept_token(2)
    assert matcher.is_terminated()
    assert matcher.get_captures() == [("marker", b"!!"), ("inner", b"b"), ("outer", second_outer)]

    assert forked.accept_token(2) and forked.is_terminated()
    assert forked.get_captures() == [("marker", b"!!"), ("inner", b"a"), ("outer", first_outer)]


@pytest.mark.parametrize(
    "attribute, expected_outer",
    [pytest.param("suffix", b"xy!z", id="suffix"), pytest.param("stop", b"xyz", id="stop")],
)
@pytest.mark.parametrize("tag_dispatch", [False, True], ids=["general", "tag-dispatch"])
def test_lark_suffix_stop_round_trip_serialization_and_cache(
    attribute: str, expected_outer: bytes, tag_dispatch: bool
) -> None:
    if tag_dispatch:
        source = f"""
            start[capture="outer"]: r
            r[capture="inner", {attribute}="!"]: TEXT
            TEXT: /(\\n|.)*/
        """
        value = "xy!"
        expected_outer = b"xy!" if attribute == "suffix" else b"xy"
    else:
        source = f"""
            start[capture="outer"]: r "z"
            r[capture="inner", {attribute}="!"]: /[a-z]*/
        """
        value = "xy!z"

    grammar = xgr.Grammar.from_lark(source)
    printed = str(grammar)
    hidden_field = f"capture_hidden_{attribute}_bytes=1"
    assert hidden_field in printed
    assert hidden_field in str(xgr.Grammar.from_ebnf(printed))

    serialized = grammar.serialize_json()
    serialized_obj = json.loads(serialized)
    rule_id = next(i for i, item in enumerate(serialized_obj["rules"]) if item[0] == "r")
    suffix_stop_info = next(
        item for item in serialized_obj["suffix_stop_infos"] if item[0] == rule_id
    )
    assert suffix_stop_info[1 if attribute == "suffix" else 2] == 1

    for candidate in (
        grammar,
        xgr.Grammar.from_ebnf(printed),
        xgr.Grammar.deserialize_json(serialized),
    ):
        assert _captures_for_string(candidate, value, cache_enabled=True) == [
            ("inner", b"xy"),
            ("outer", expected_outer),
        ]


@pytest.mark.parametrize(
    "attribute, expected_outer",
    [pytest.param("suffix", b"xy!!z", id="suffix"), pytest.param("stop", b"xyz", id="stop")],
)
def test_lark_suffix_stop_variable_marker_round_trip_serialization_and_cache(
    attribute: str, expected_outer: bytes
) -> None:
    source = f"""
        start[capture="outer"]: r "z"
        r[capture="inner", {attribute}=END, stop_capture="marker"]: /[a-z]*/
        END: /!!+/
    """
    grammar = xgr.Grammar.from_lark(source)
    printed = str(grammar)
    assert "capture_hidden_body_rule_id=" in printed
    assert "capture_hidden_marker_rule_id=" in printed
    assert 'stop_capture="marker"' in printed

    serialized = grammar.serialize_json()
    serialized_obj = json.loads(serialized)
    rule_id = next(i for i, item in enumerate(serialized_obj["rules"]) if item[0] == "r")
    suffix_stop_info = next(
        item for item in serialized_obj["suffix_stop_infos"] if item[0] == rule_id
    )
    assert suffix_stop_info[3] >= 0 and suffix_stop_info[4] >= 0
    assert suffix_stop_info[5] == "marker"

    for candidate in (
        grammar,
        xgr.Grammar.from_ebnf(printed),
        xgr.Grammar.deserialize_json(serialized),
    ):
        assert _captures_for_string(candidate, "xy!!z", cache_enabled=True) == [
            ("marker", b"!!"),
            ("inner", b"xy"),
            ("outer", expected_outer),
        ]


STOP_SUFFIX_COMBINED_TOKENIZER = xgr.TokenizerInfo(["x", "<", ">", "a!", "b!", "a", "b", "!"])


@pytest.mark.parametrize(
    "attribute, expected_all",
    [pytest.param("suffix", b"xx<a!>", id="suffix"), pytest.param("stop", b"xx<a>", id="stop")],
)
def test_lark_suffix_stop_with_max_tokens_capture_and_lazy(
    attribute: str, expected_all: bytes
) -> None:
    # The features also compose independently on separate rules. Explicit lazy is accepted
    # because suffix/stop already has lazy semantics.
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="all"]: reasoning "<" value ">"
        reasoning[max_tokens=2, capture]: TEXT
        value[lazy, capture, {attribute}="!"]: /[ab]*/
        TEXT: /(\\n|.)*/
        """,
        tokenizer_info=STOP_SUFFIX_COMBINED_TOKENIZER,
    )
    compiled = xgr.GrammarCompiler(
        STOP_SUFFIX_COMBINED_TOKENIZER, cache_enabled=False
    ).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for token_id in [0, 0]:
        assert token_id in _allowed_token_ids(matcher, STOP_SUFFIX_COMBINED_TOKENIZER)
        assert matcher.accept_token(token_id)
    assert _allowed_token_ids(matcher, STOP_SUFFIX_COMBINED_TOKENIZER) == [1]
    assert matcher.accept_token(1)
    assert 3 in _allowed_token_ids(matcher, STOP_SUFFIX_COMBINED_TOKENIZER)
    assert matcher.accept_token(3)
    assert _allowed_token_ids(matcher, STOP_SUFFIX_COMBINED_TOKENIZER) == [2]
    assert matcher.accept_token(2) and matcher.is_terminated()
    assert matcher.get_captures() == [("reasoning", b"xx"), ("value", b"a"), ("all", expected_all)]


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_same_rule_max_tokens_without_capture(attribute: str) -> None:
    grammar = xgr.Grammar.from_lark(
        f"""
        start: value ">"
        value[max_tokens=2, {attribute}="!"]: TEXT
        TEXT: /(\\n|.)*/
        """,
        tokenizer_info=STOP_SUFFIX_COMBINED_TOKENIZER,
    )
    compiled = xgr.GrammarCompiler(
        STOP_SUFFIX_COMBINED_TOKENIZER, cache_enabled=False
    ).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(0) and matcher.accept_token(0)
    assert _allowed_token_ids(matcher, STOP_SUFFIX_COMBINED_TOKENIZER) == [2]
    assert matcher.accept_token(2) and matcher.is_terminated()


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_max_tokens_closes_expired_parent(attribute: str) -> None:
    tokenizer_info = xgr.TokenizerInfo(["x", "!", "z", "y"])
    grammar = xgr.Grammar.from_lark(
        f"""
        start: outer "z"
        outer[max_tokens=1]: inner | inner "y"
        inner[max_tokens=1, {attribute}="!"]: /x*/
        """,
        tokenizer_info=tokenizer_info,
    )
    compiled = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(0)
    # Forcing inner to close without its marker also reaches outer's deadline. The expired
    # outer "y" alternative must not survive beside the valid exit to "z".
    assert _allowed_token_ids(matcher, tokenizer_info) == [2]
    assert matcher.accept_token(2) and matcher.is_terminated()


@pytest.mark.parametrize(
    "attribute, expected_marker_outer",
    [pytest.param("suffix", b"ab!>", id="suffix"), pytest.param("stop", b"ab>", id="stop")],
)
def test_lark_suffix_stop_same_rule_max_tokens(
    attribute: str, expected_marker_outer: bytes
) -> None:
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="all"]: value ">"
        value[max_tokens=2, capture="body", {attribute}="!", stop_capture="marker"]: /[ab]*/
        """,
        tokenizer_info=STOP_SUFFIX_COMBINED_TOKENIZER,
    )
    printed = str(grammar)
    assert "max_tokens=2" in printed
    assert "capture_hidden_body_rule_id=" in printed
    grammar = xgr.Grammar.deserialize_json(grammar.serialize_json())
    compiled = xgr.GrammarCompiler(
        STOP_SUFFIX_COMBINED_TOKENIZER, cache_enabled=False
    ).compile_grammar(grammar)

    # Marker first: retain the existing committed-shortest marker and capture behavior.
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(3)  # "a!"
    assert _allowed_token_ids(matcher, STOP_SUFFIX_COMBINED_TOKENIZER) == [2]
    assert matcher.accept_token(2) and matcher.is_terminated()
    assert matcher.get_captures() == [
        ("marker", b"!"),
        ("body", b"a"),
        ("all", b"a!>" if attribute == "suffix" else b"a>"),
    ]

    # Budget first: close through the body without fabricating a marker capture.
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(5) and matcher.accept_token(6)  # "a", "b"
    assert _allowed_token_ids(matcher, STOP_SUFFIX_COMBINED_TOKENIZER) == [2]
    assert not matcher.accept_token(0)  # A rejection must not commit the forced close.
    assert _allowed_token_ids(matcher, STOP_SUFFIX_COMBINED_TOKENIZER) == [2]
    forked = matcher.fork()
    for candidate in [matcher, forked]:
        assert candidate.accept_token(2) and candidate.is_terminated()
        assert candidate.get_captures() == [("body", b"ab"), ("all", b"ab>")]

    # Roll back across the forced close, then take the marker-first path instead.
    matcher.rollback(2)
    assert matcher.accept_token(4)  # "b!"
    assert matcher.accept_token(2) and matcher.is_terminated()
    assert matcher.get_captures() == [
        ("marker", b"!"),
        ("body", b"ab"),
        ("all", expected_marker_outer),
    ]


@pytest.mark.parametrize(
    "attribute, expected_outer",
    [pytest.param("suffix", b"a!!>", id="suffix"), pytest.param("stop", b"a>", id="stop")],
)
def test_lark_suffix_stop_max_tokens_marker_crosses_token_boundary(
    attribute: str, expected_outer: bytes
) -> None:
    grammar = xgr.Grammar.from_lark(
        f"""
        start[capture="all"]: value ">"
        value[max_tokens=1, capture="body", {attribute}=END, stop_capture="marker"]: /[ab]*/
        END: /!!+/
        """,
        tokenizer_info=STOP_SUFFIX_COMBINED_TOKENIZER,
    )
    compiled = xgr.GrammarCompiler(
        STOP_SUFFIX_COMBINED_TOKENIZER, cache_enabled=False
    ).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_token(3)  # "a!": the marker is only partially matched at the budget.
    assert _allowed_token_ids(matcher, STOP_SUFFIX_COMBINED_TOKENIZER) == [7]
    assert matcher.accept_token(7)  # Complete "!!" across the token boundary.
    assert _allowed_token_ids(matcher, STOP_SUFFIX_COMBINED_TOKENIZER) == [2]
    assert matcher.accept_token(2) and matcher.is_terminated()
    assert matcher.get_captures() == [("marker", b"!!"), ("body", b"a"), ("all", expected_outer)]


@pytest.mark.parametrize("attribute", ["suffix", "stop"])
def test_lark_suffix_stop_body_must_be_terminal_but_supports_bounded_regex(attribute: str) -> None:
    _assert_lark_error(
        f'start: r\nr[{attribute}="!"]: sub\nsub: /[a-z]*/', "terminal cannot reference rule"
    )
    _assert_language(
        f'start: r\nr[{attribute}="!"]: /[a-z]{{2,5}}/',
        ["ab!", "abcde!"],
        ["a!", "abcdef!", "ab!!"],
    )
