"""A raw XML string argument never contains its own closing delimiter, whatever constrains it.

Parsers end a raw argument at the first parameter suffix, so a value that contains the suffix is
read as a shorter value followed by markup: further parameters, or a whole second tool call.
"""

import json

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

STYLES = [
    ("qwen_xml", "<parameter=q>", "</parameter>"),
    ("minimax_xml", '<parameter name="q">', "</parameter>"),
    ("glm_xml", "<arg_key>q</arg_key><arg_value>", "</arg_value>"),
    ("deepseek_xml", '<｜DSML｜parameter name="q" string="true">', "</｜DSML｜parameter>"),
]

SCHEMAS = {
    "plain": {"type": "string"},
    "minLength": {"type": "string", "minLength": 1},
    "maxLength": {"type": "string", "maxLength": 256},
    "pattern with markup characters": {"type": "string", "pattern": '^[A-Za-z<>/="_｜ ]+$'},
    "pattern with a dot": {"type": "string", "pattern": "^.*$"},
    "counted dot": {"type": "string", "pattern": "^.{1,256}$"},
    "counted class": {"type": "string", "pattern": "[^\\n]{0,256}"},
    "pattern alternation": {
        "type": "string",
        "pattern": '^(<[a-z_=" ]+>|</[A-Za-z_｜]+>|[a-z ])+$',
    },
}


def leaf_grammar(style, schema):
    return xgr.Grammar.from_structural_tag(
        json.dumps(
            {
                "type": "structural_tag",
                "format": {
                    "type": "json_schema",
                    "style": style,
                    "json_schema": {
                        "type": "object",
                        "properties": {"q": schema, "r": {"type": "string"}},
                        "required": ["q", "r"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    )


def r_parameter(style, begin):
    return begin.replace("q", "r", 1) if style != "glm_xml" else begin.replace(">q<", ">r<")


@pytest.mark.parametrize("style,begin,end", STYLES)
@pytest.mark.parametrize("name", list(SCHEMAS))
def test_raw_string_value_excludes_parameter_suffix(style, begin, end, name):
    grammar = leaf_grammar(style, SCHEMAS[name])
    r = r_parameter(style, begin) + "x" + end
    value = "ab c"
    assert _is_grammar_accept_string(grammar, begin + value + end + r)
    # The value runs on past its own suffix into the next parameter, then closes it again.
    hidden = "ab" + end + r_parameter(style, begin) + "ab"
    assert not _is_grammar_accept_string(grammar, begin + hidden + end + r)
    # The value repeats the suffix and closes the argument a second time.
    assert not _is_grammar_accept_string(grammar, begin + "ab" + end + "cd" + end + r)


@pytest.mark.parametrize("style,begin,end", STYLES)
def test_pattern_keeps_markup_without_the_suffix(style, begin, end):
    grammar = leaf_grammar(style, SCHEMAS["pattern alternation"])
    r = r_parameter(style, begin) + "x" + end
    assert _is_grammar_accept_string(grammar, begin + "<b>bold</b> text" + end + r)
    assert not _is_grammar_accept_string(grammar, begin + "Bold" + end + r)


@pytest.mark.parametrize("style,begin,end", STYLES)
def test_counted_pattern_keeps_its_bounds(style, begin, end):
    grammar = leaf_grammar(style, {"type": "string", "pattern": "^.{2,5}$"})
    r = r_parameter(style, begin) + "x" + end
    for value, accepted in [
        ("a", False),
        ("ab", True),
        ("a<b", True),
        ("你好🦋你好", True),
        ("abcdef", False),
        ("a</b", True),
    ]:
        assert _is_grammar_accept_string(grammar, begin + value + end + r) == accepted, value


def test_counted_pattern_is_not_unrolled():
    # The bound becomes a character budget, so the rule count does not grow with it.
    small = leaf_grammar("qwen_xml", {"type": "string", "pattern": "^.{1,8}$"})
    large = leaf_grammar("qwen_xml", {"type": "string", "pattern": "^.{1,4000}$"})
    assert str(large).count("::=") == str(small).count("::=")


def test_counted_pattern_is_framed_like_a_bounded_string():
    # One optional template newline on each side stays outside the bound, as for maxLength.
    grammar = leaf_grammar("qwen_xml", {"type": "string", "pattern": "^.{1,3}$"})
    r = "<parameter=r>x</parameter>"
    for value, accepted in [
        ("abc", True),
        ("\nabc\n", True),
        ("abcd", False),
        ("\nabcd\n", False),
        ("a<", True),
        ("", False),
    ]:
        text = "<parameter=q>" + value + "</parameter>" + r
        assert _is_grammar_accept_string(grammar, text) == accepted, repr(value)


def test_counted_pattern_that_cannot_match_the_suffix_is_unchanged():
    # A counted class without "<" keeps its grammar and its free formatting whitespace.
    grammar = leaf_grammar("qwen_xml", {"type": "string", "pattern": "^[^<]{1,80}$"})
    assert "_bounded_xml" not in str(grammar)


def test_pattern_that_only_matches_the_suffix_is_unsatisfiable():
    schema = {"type": "string", "pattern": "^a</parameter>b$"}
    grammar = leaf_grammar("qwen_xml", schema)
    r = "<parameter=r>x</parameter>"
    assert not _is_grammar_accept_string(grammar, "<parameter=q>a</parameter>b</parameter>" + r)


def test_second_tool_call_does_not_fit_in_a_pattern_value():
    # mlc-ai/xgrammar#1019: with stop_after_first, the second call hides inside the first value.
    tag = {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<tool_call>"],
            "stop_after_first": True,
            "tags": [
                {
                    "type": "tag",
                    "begin": "<tool_call>\n<function=f>\n",
                    "end": "\n</function>\n</tool_call>",
                    "content": {
                        "type": "json_schema",
                        "style": "qwen_xml",
                        "json_schema": {
                            "type": "object",
                            "properties": {"q": {"type": "string", "pattern": "^[a-z<>/=\\n_]+$"}},
                            "required": ["q"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    }
    grammar = xgr.Grammar.from_structural_tag(json.dumps(tag))
    hidden = (
        "a\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=f>\n<parameter=q>\nb"
    )
    call = "<tool_call>\n<function=f>\n<parameter=q>\n{}\n</parameter>\n</function>\n</tool_call>"
    assert _is_grammar_accept_string(grammar, call.format("a</b>"))
    assert not _is_grammar_accept_string(grammar, call.format(hidden))


def test_token_mask_rejects_suffix_inside_pattern_value():
    vocab = ["<parameter=q>", "a", "<", "</parameter>", "</parameter><parameter=q>", "<eos>"]
    info = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, stop_token_ids=[5])
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string", "pattern": "^.{1,64}$"}},
        "required": ["q"],
        "additionalProperties": False,
    }
    tag = {
        "type": "structural_tag",
        "format": {"type": "json_schema", "style": "qwen_xml", "json_schema": schema},
    }
    compiled = xgr.GrammarCompiler(info, max_threads=1).compile_structural_tag(json.dumps(tag))
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0) and matcher.accept_token(1)
    mask = xgr.allocate_token_bitmask(1, len(vocab))
    matcher.fill_next_token_bitmask(mask)
    allowed = {i for i in range(len(vocab)) if (int(mask[0, i // 32]) >> (i % 32)) & 1}
    assert {1, 2, 3} <= allowed
    assert 4 not in allowed
    assert matcher.accept_token(3)
    assert not matcher.accept_token(1)
