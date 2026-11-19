"""A raw XML delimiter commits only after all its bytes pass the old value grammar."""

import json

import pytest

import xgrammar as xgr

STYLES = [
    ("cohere_xml", '<cofl:value name="v" type="raw">', "</cofl:value>"),
    ("qwen_xml", "<parameter=v>", "</parameter>"),
    ("minimax_xml", '<parameter name="v">', "</parameter>"),
    ("glm_xml", "<arg_key>v</arg_key><arg_value>", "</arg_value>"),
    ("deepseek_xml", '<｜DSML｜parameter name="v" string="true">', "</｜DSML｜parameter>"),
    ("deepseek_v4_1_xml", '<｜DSML｜ parameter name="v" string="true">', "</｜DSML｜ parameter>"),
    ("kimi_k3_xml", '<|open|>argument key="v" type="string"<|sep|>', "<|close|>argument<|sep|>"),
]


def compile_value(value, vocab, style="qwen_xml", **options):
    schema = {
        "type": "object",
        "properties": {"v": value},
        "required": ["v"],
        "additionalProperties": False,
    }
    tag = {
        "type": "structural_tag",
        "format": {"type": "json_schema", "json_schema": schema, "style": style, **options},
    }
    info = xgr.TokenizerInfo(vocab + ["<eos>"], stop_token_ids=[len(vocab)])
    return xgr.GrammarCompiler(info).compile_structural_tag(json.dumps(tag))


def assert_mask(matcher, compiled, token, allowed):
    mask = xgr.allocate_token_bitmask(1, compiled.tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(mask)
    assert bool(mask[0, token // 32].item() & (1 << (token % 32))) == allowed
    trial = matcher.fork()
    assert trial.accept_token(token) == allowed


@pytest.mark.parametrize("style,opening,closing", STYLES)
@pytest.mark.parametrize(
    "value",
    [
        {"type": "string", "pattern": r"[\s\S]*X"},
        {"type": "string", "minLength": 100},
        {"type": "string", "pattern": r"[\s\S]*"},
    ],
)
def test_xml_marker_commits_string(style, opening, closing, value):
    # The old grammar retains delimiter-as-content, whether this value can finish or not.
    vocab = [opening, "abc", closing, "tail"]
    compiled = compile_value(value, vocab, style)
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    assert matcher.accept_token(1)
    assert_mask(matcher, compiled, 2, True)
    assert matcher.accept_token(2)
    assert matcher.is_completed()
    assert_mask(matcher, compiled, 3, False)
    assert_mask(matcher, compiled, 4, True)
    assert matcher.accept_token(4)
    assert matcher.is_terminated()


@pytest.mark.parametrize(
    "style,opening,closing,split",
    [
        (style, opening, closing, split)
        for style, opening, closing in STYLES
        for split in range(1, len(closing.encode()))
    ],
)
def test_xml_boundary_crosses_tokens_and_rollback(style, opening, closing, split):
    closing = closing.encode()
    vocab = [opening.encode(), b"abc", closing[:split], closing[split:], b"bad", b"wrong"]
    compiled = compile_value({"type": "string", "pattern": r"[\s\S]*X"}, vocab, style)
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    assert matcher.accept_token(1)
    assert matcher.accept_token(2)
    before = matcher.fork()
    # A partial UTF-8 codepoint still requires its continuation bytes.
    accepts_ascii = closing[:split].decode(errors="ignore").encode() == closing[:split]
    assert_mask(matcher, compiled, 4, accepts_ascii)
    assert_mask(matcher, compiled, 3, True)
    assert matcher.accept_token(3)
    assert matcher.is_completed()
    assert_mask(matcher, compiled, 4, False)
    matcher.rollback()
    assert_mask(matcher, compiled, 4, accepts_ascii)
    assert before.accept_token(3)
    assert before.is_completed()


def test_xml_boundary_checks_remainder_in_same_token():
    vocab = ["<parameter=v>", "abc</parameter>", "abc</parameter>garbage"]
    compiled = compile_value({"type": "string", "pattern": r"[\s\S]*X"}, vocab)
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    assert_mask(matcher, compiled, 1, True)
    assert_mask(matcher, compiled, 2, False)


@pytest.mark.parametrize("value", [{"const": "a</parameter>b"}, {"enum": ["a</parameter>b"]}])
def test_fixed_strings_keep_literal_markers(value):
    vocab = ["<parameter=v>", "a</parameter>", "b</parameter>", "<eos>"]
    compiled = compile_value(value, vocab)
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    assert matcher.accept_token(1)
    assert not matcher.is_completed()
    assert_mask(matcher, compiled, 2, True)
    assert matcher.accept_token(2)
    assert matcher.is_completed()


def test_xml_boundary_does_not_relax_prefix_constraints():
    vocab = ["<parameter=v>", "a", "</parameter>", "Z"]
    compiled = compile_value({"type": "string", "pattern": r"[a-z]+Z"}, vocab)
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    assert matcher.accept_token(1)
    assert_mask(matcher, compiled, 2, False)
    assert_mask(matcher, compiled, 3, True)


def test_xml_boundary_serialization_and_ebnf():
    vocab = ["<parameter=v>", "abc</parameter>"]
    compiled = compile_value({"type": "string", "pattern": r"[\s\S]*X"}, vocab)
    copies = [
        xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), compiled.tokenizer_info),
        xgr.GrammarCompiler(compiled.tokenizer_info).compile_grammar(str(compiled.grammar)),
    ]
    for copy in copies:
        matcher = xgr.GrammarMatcher(copy)
        assert matcher.accept_token(0)
        assert_mask(matcher, copy, 1, True)
        assert matcher.accept_token(1)
        assert matcher.is_completed()


@pytest.mark.parametrize(
    "value",
    [
        {"anyOf": [{"type": "string", "pattern": r"[\s\S]*X"}, {"type": "integer"}]},
        {"oneOf": [{"type": "string", "pattern": r"[\s\S]*X"}, {"const": "fixed"}]},
        {"type": ["string", "null"], "pattern": r"[\s\S]*X"},
    ],
)
def test_xml_boundary_preserves_alternative_types(value):
    vocab = ["<parameter=v>", "abc</parameter>", "42</parameter>", "null</parameter>"]
    compiled = compile_value(value, vocab)
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    assert_mask(matcher, compiled, 1, True)
    assert matcher.accept_token(1)
    assert matcher.is_completed()


def test_xml_boundary_next_parameter_and_nested_json():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {"type": "string", "pattern": r"[\s\S]*X"},
            "b": {
                "type": "object",
                "properties": {"literal": {"const": "</parameter>"}},
                "required": ["literal"],
                "additionalProperties": False,
            },
        },
        "required": ["a", "b"],
    }
    tag = {
        "type": "structural_tag",
        "format": {"type": "json_schema", "json_schema": schema, "style": "qwen_xml"},
    }
    vocab = [
        "<parameter=a>",
        "abc</parameter><parameter=b>",
        '{"literal":"</parameter>"}</parameter>',
        "<eos>",
    ]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[3])
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(json.dumps(tag))
    matcher = xgr.GrammarMatcher(compiled)
    for token in range(4):
        assert_mask(matcher, compiled, token, True)
        assert matcher.accept_token(token)
    assert matcher.is_terminated()


def test_xml_boundary_recursive_cohere_scope():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "inner": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"v": {"type": "string", "pattern": r"[\s\S]*X"}},
                "required": ["v"],
            },
            "tail": {"type": "integer"},
        },
        "required": ["inner", "tail"],
    }
    tag = {
        "type": "structural_tag",
        "format": {"type": "json_schema", "json_schema": schema, "style": "cohere_xml"},
    }
    vocab = [
        '<cofl:value name="inner" type="dict"><cofl:value name="v" type="raw">',
        'abc</cofl:value></cofl:value><cofl:value name="tail" type="json">',
        "1</cofl:value>",
        "<eos>",
    ]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[3])
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(json.dumps(tag))
    matcher = xgr.GrammarMatcher(compiled)
    for token in range(4):
        assert_mask(matcher, compiled, token, True)
        assert matcher.accept_token(token)
    assert matcher.is_terminated()


def test_xml_boundary_shared_reference_occurrences():
    schema = {
        "$defs": {"text": {"type": "string", "pattern": r"[\s\S]*X"}},
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"$ref": "#/$defs/text"}, "b": {"$ref": "#/$defs/text"}},
        "required": ["a", "b"],
    }
    tag = {
        "type": "structural_tag",
        "format": {"type": "json_schema", "json_schema": schema, "style": "qwen_xml"},
    }
    vocab = ["<parameter=a>", "a</parameter><parameter=b>", "b</parameter>", "<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[3])
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(json.dumps(tag))
    matcher = xgr.GrammarMatcher(compiled)
    for token in range(4):
        assert_mask(matcher, compiled, token, True)
        assert matcher.accept_token(token)
    assert matcher.is_terminated()


def test_xml_boundary_token_remainder_enters_any_tokens():
    tag = {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {
                    "type": "tag",
                    "begin": "<call>",
                    "end": "</call>",
                    "content": {
                        "type": "json_schema",
                        "style": "qwen_xml",
                        "json_schema": {
                            "type": "object",
                            "properties": {"v": {"type": "string", "pattern": r"[\s\S]*X"}},
                            "required": ["v"],
                            "additionalProperties": False,
                        },
                    },
                },
                {"type": "any_tokens", "exclude_tokens": [2]},
            ],
        },
    }
    vocab = ["<call><parameter=v>", "abc</parameter></call>free text", "<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[2])
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(json.dumps(tag))
    matcher = xgr.GrammarMatcher(compiled)
    for token in range(3):
        assert_mask(matcher, compiled, token, True)
        assert matcher.accept_token(token)
    assert matcher.is_terminated()


@pytest.mark.parametrize("budget", ["max_tokens=2", "max_tokens=100", "max_chars=2"])
def test_xml_boundary_composes_with_rule_budget(budget):
    vocab = ["<parameter=v>", "abc", "</parameter>"]
    original = compile_value({"type": "string", "pattern": r"[\s\S]*X"}, vocab)
    grammar = str(original.grammar).replace("root ::=", f"root[{budget}] ::=", 1)
    compiled = xgr.GrammarCompiler(original.tokenizer_info).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled)
    for token in range(4):
        assert_mask(matcher, compiled, token, True)
        assert matcher.accept_token(token)
    assert matcher.is_terminated()


@pytest.mark.parametrize("budget", [2, 100])
def test_xml_boundary_char_budget_shared_candidate_prefix(budget):
    vocab = ["<parameter=v>", "abc", "</parameter>", "</parameter>x", "</parameter>zz"]
    original = compile_value({"type": "string", "pattern": r"[\s\S]*X"}, vocab)
    grammar = str(original.grammar)
    assert "[end_on=" in grammar
    grammar = grammar.replace("[end_on=", f"[max_chars={budget}, end_on=")
    compiled = xgr.GrammarCompiler(original.tokenizer_info).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, max_rollback_tokens=4)
    assert matcher.accept_token(0)
    assert matcher.accept_token(1)
    # Rechecking a failed candidate must not mutate another candidate's shared prefix.
    for _ in range(2):
        assert_mask(matcher, compiled, 2, True)
        assert_mask(matcher, compiled, 3, False)
        assert_mask(matcher, compiled, 4, False)
    assert not matcher.accept_token(3)
    assert matcher.accept_token(2)
    matcher.rollback(1)
    assert_mask(matcher, compiled, 2, True)
    assert matcher.accept_token(2)
    assert matcher.is_completed()


def test_xml_boundary_char_budget_recheck_reaches_parent_atomic_edge():
    # The current row has no atomic edge. Closing a zero-character budget exposes the
    # parent's Token(1), so XML candidate rechecking must still try the atomic path.
    vocab = ["a", "</parameter>", ">", "!", "<eos>"]
    grammar = str(
        xgr.Grammar.from_lark(
            """
start: reasoning "Z" | "!" marker
reasoning[max_chars=0, suffix=">"]: /[a-z]*/
marker: /[a-z]*/ "</parameter>"
"""
        )
    )
    assert grammar.count('"Z"') == 1
    assert "\nmarker ::=" in grammar
    grammar = grammar.replace('"Z"', "Token(1)").replace(
        "\nmarker ::=", '\nmarker[end_on="</parameter>"] ::='
    )
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[4])
    compiled = xgr.GrammarCompiler(info, max_threads=1).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled)
    direct = matcher.fork()
    assert direct.accept_token(1)
    assert direct.is_completed()
    for _ in range(2):
        assert_mask(matcher, compiled, 1, True)
        assert not matcher.is_completed()
    # Speculation and an unsuccessful accept must not consume the budget-close transition.
    assert not matcher.accept_token(4)
    assert_mask(matcher, compiled, 1, True)
    assert matcher.accept_token(1)
    assert matcher.is_completed()
    assert_mask(matcher, compiled, 4, True)
    assert matcher.accept_token(4)
    assert matcher.is_terminated()
    matcher.rollback(2)
    assert not matcher.is_completed()
    assert_mask(matcher, compiled, 1, True)
    assert matcher.accept_token(1)
    assert matcher.is_completed()


def test_xml_boundary_token_budget_recheck_reaches_parent_atomic_edge():
    # Token-budget enforcement exposes the parent's atomic edge before candidate rechecking.
    vocab = ["a", "</parameter>", ">", "!", "<eos>"]
    grammar = str(
        xgr.Grammar.from_lark(
            """
start: reasoning "Z" | "!" marker
reasoning[max_tokens=1, suffix=">"]: /[a-z]*/
marker: /[a-z]*/ "</parameter>"
"""
        )
    )
    grammar = grammar.replace('"Z"', "Token(1)").replace(
        "\nmarker ::=", '\nmarker[end_on="</parameter>"] ::='
    )
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[4])
    compiled = xgr.GrammarCompiler(info, max_threads=1).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    for _ in range(2):
        assert_mask(matcher, compiled, 1, True)
        assert not matcher.is_completed()
    assert not matcher.accept_token(4)
    assert_mask(matcher, compiled, 1, True)
    assert matcher.accept_token(1)
    assert matcher.is_completed()
    assert matcher.accept_token(4)
    assert matcher.is_terminated()
    matcher.rollback(2)
    assert_mask(matcher, compiled, 1, True)
    assert matcher.accept_token(1)
    assert matcher.is_completed()


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "alternative,allows_tail",
    [
        ('"!" Token(1)', False),
        ("Token(1)", True),
        ("optional Token(1)", True),
        ("ExcludeToken(0, 2, 3)", True),
    ],
)
def test_xml_boundary_recheck_current_atomic_edges(dynamic, alternative, allows_tail):
    # A global token edge is insufficient: it must be reachable at the current row.
    # Nullable completion exposes Token(1) before scanning; ExcludeToken also needs retry.
    vocab = ["a", "</parameter>tail", "</parameter>", "!", "<eos>"]
    grammar = (
        f"root ::= boundary | {alternative}\n"
        'boundary[end_on="</parameter>"] ::= [a-z]* "</parameter>"\n'
        'optional ::= "a"?\n'
    )
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[4])
    compiled = xgr.GrammarCompiler(
        info, enable_dynamic_compilation=dynamic, max_threads=1
    ).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.fork().accept_token(1) == allows_tail
    for _ in range(2):
        assert_mask(matcher, compiled, 1, allows_tail)
        assert_mask(matcher, compiled, 2, True)
    if not allows_tail:
        assert not matcher.accept_token(1)
    chosen = 1 if allows_tail else 2
    assert matcher.accept_token(chosen)
    assert matcher.is_completed()
    assert_mask(matcher, compiled, 4, True)
    assert matcher.accept_token(4)
    assert matcher.is_terminated()
    matcher.rollback(2)
    assert_mask(matcher, compiled, 1, allows_tail)
    assert_mask(matcher, compiled, 2, True)


@pytest.mark.parametrize("allowed_token", [1, 2])
@pytest.mark.parametrize("fallback", ["mid_token", "atomic"])
def test_xml_boundary_shared_failed_bytes_keep_token_specific_fallback(allowed_token, fallback):
    # Both candidates fail at the same byte after the closing marker. Their byte result
    # can be shared, but ExcludeToken and Token fallbacks must still check each token ID.
    vocab = ["a", "</parameter>tail", "</parameter>tail", "<eos>"]
    rejected_token = 3 - allowed_token
    root = (
        f"boundary ExcludeToken({rejected_token})"
        if fallback == "mid_token"
        else f"boundary | Token({allowed_token})"
    )
    grammar = f"root ::= {root}\n" 'boundary[end_on="</parameter>"] ::= [a-z]* "</parameter>"\n'
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[3])
    compiled = xgr.GrammarCompiler(info, max_threads=1).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled)
    if fallback == "mid_token":
        assert matcher.accept_token(0)
    assert matcher.fork().accept_token(allowed_token)
    assert not matcher.fork().accept_token(rejected_token)
    for _ in range(2):
        assert_mask(matcher, compiled, rejected_token, False)
        assert_mask(matcher, compiled, allowed_token, True)
    assert not matcher.accept_token(rejected_token)
    assert matcher.accept_token(allowed_token)
    assert matcher.is_completed()
    assert matcher.accept_token(3)
    assert matcher.is_terminated()
    matcher.rollback(2)
    assert_mask(matcher, compiled, rejected_token, False)
    assert_mask(matcher, compiled, allowed_token, True)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("marker_in_helper", [False, True])
def test_xml_boundary_completion_propagates_through_recursive_owner(dynamic, marker_in_helper):
    # Closing the inner occurrence must also commit its same-owner parents before their "x".
    # A raw completed context mistaken for a partial prefix would leave that suffix live.
    vocab = ["a", ">", "!", "x", ">!", "<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[5])
    ending = "helper" if marker_in_helper else '">"'
    grammar = (
        'root ::= boundary "!"\n'
        f'boundary[end_on=">", capture="value"] ::= "a" boundary "x" | {ending}\n'
    )
    if marker_in_helper:
        # Capture prevents inlining, so completion must climb from a distinct helper rule.
        grammar += 'helper[capture="leaf"] ::= ">"\n'
    compiled = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic).compile_grammar(
        grammar
    )
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    assert matcher.accept_token(0)
    for token, allowed in enumerate([True, True, False, False, True, False]):
        assert_mask(matcher, compiled, token, allowed)
    assert matcher.get_captures() == []
    whole = matcher.fork()
    assert whole.accept_token(4)
    assert whole.is_completed()
    assert matcher.accept_token(1)
    assert_mask(matcher, compiled, 3, False)
    assert_mask(matcher, compiled, 2, True)
    expected = [("leaf", b">")] if marker_in_helper else []
    assert matcher.get_captures() == expected + [
        ("value", b">"),
        ("value", b"a>"),
        ("value", b"aa>"),
    ]
    assert matcher.accept_token(2)
    assert matcher.accept_token(5)
    assert matcher.is_terminated()
    matcher.rollback(3)
    assert matcher.get_captures() == []
    assert_mask(matcher, compiled, 4, True)
    assert matcher.accept_token(4)
    assert matcher.get_captures() == whole.get_captures()


@pytest.mark.parametrize("dynamic", [False, True])
def test_xml_boundary_completion_preserves_nested_owner_scope(dynamic):
    vocab = ["A", "a</o></i>", "B", "</o>", "!", "B</o>!", "<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[6])
    grammar = (
        'root ::= outer "!"\n'
        'outer[end_on="</o>", capture="outer"] ::= "A" inner "B" "</o>"\n'
        'inner[end_on="</i>", capture="inner"] ::= [a-z<>/]* "X"\n'
    )
    compiled = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic).compile_grammar(
        grammar
    )
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    assert_mask(matcher, compiled, 1, True)
    assert matcher.accept_token(1)
    # The outer closing text inside the inner string did not finish the outer rule.
    assert not matcher.is_completed()
    assert matcher.get_captures() == [("inner", b"a</o></i>")]
    assert_mask(matcher, compiled, 4, False)
    assert_mask(matcher, compiled, 5, True)
    assert matcher.accept_token(5)
    assert matcher.get_captures() == [("inner", b"a</o></i>"), ("outer", b"Aa</o></i>B</o>")]
    assert matcher.accept_token(6)
    assert matcher.is_terminated()


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "body,definitions",
    [
        ("helper", 'helper[capture="leaf"] ::= ">"\n'),
        ("helper", 'helper[capture="middle"] ::= leaf\nleaf[capture="leaf"] ::= ">"\n'),
        ("helper", 'helper[capture="leaf"] ::= ">" empty\nempty[capture="empty"] ::= ""\n'),
        ("helper", 'helper[capture="leaf"] ::= ">" empty{1,2}\nempty[capture="empty"] ::= ""\n'),
        ("helper", 'helper[capture="leaf"] ::= ">" "X"?\n'),
        (
            "helper",
            'helper[capture="leaf"] ::= ">" next\n' 'next[end_on="<", capture="next"] ::= "X"?\n',
        ),
        ("left | right", 'left[capture="left"] ::= ">"\nright[capture="right"] ::= ">"\n'),
    ],
    ids=["literal", "nested", "nullable", "counted-nullable", "optional", "new-owner", "parallel"],
)
def test_xml_boundary_preserves_natural_helper_captures(dynamic, body, definitions):
    vocab = [">", "!", "X", ">!", "<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[4])
    grammar = (
        'root ::= boundary "!"\n'
        f'boundary[end_on=">", capture="value"] ::= {body}\n' + definitions
    )
    compiler = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic)
    compiled = compiler.compile_grammar(grammar)
    # Remove only this boundary's annotation: nested owners retain their own semantics.
    reference = compiler.compile_grammar(grammar.replace('end_on=">", ', "", 1))
    matcher, ordinary = xgr.GrammarMatcher(compiled), xgr.GrammarMatcher(reference)
    whole, ordinary_whole = matcher.fork(), ordinary.fork()
    for token in (0, 3):
        assert_mask(matcher, compiled, token, True)
    assert matcher.get_captures(deduplicate=False) == []
    assert matcher.accept_token(0)
    assert ordinary.accept_token(0)
    for deduplicate in (False, True):
        assert matcher.get_captures(deduplicate=deduplicate) == ordinary.get_captures(
            deduplicate=deduplicate
        )
    assert_mask(matcher, compiled, 1, True)
    assert_mask(matcher, compiled, 2, False)  # A nullable child must not keep its "X" scanner.
    assert_mask(matcher, compiled, 4, False)
    for item in (matcher, ordinary):
        assert item.accept_token(1)
        assert item.accept_token(4)
        assert item.is_terminated()
    for item in (whole, ordinary_whole):
        assert item.accept_token(3)
        assert item.accept_token(4)
        assert item.is_terminated()
    for deduplicate in (False, True):
        expected = ordinary.get_captures(deduplicate=deduplicate)
        assert matcher.get_captures(deduplicate=deduplicate) == expected
        assert whole.get_captures(deduplicate=deduplicate) == expected
        assert ordinary_whole.get_captures(deduplicate=deduplicate) == expected
    matcher.rollback(3)
    assert matcher.get_captures(deduplicate=False) == []
    assert matcher.accept_token(3)
    assert matcher.accept_token(4)
    assert matcher.get_captures(deduplicate=False) == whole.get_captures(deduplicate=False)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("new_owner", [False, True])
def test_xml_boundary_does_not_capture_unfinished_helpers(dynamic, new_owner):
    vocab = [">", "!", "X", ">!", "<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[4])
    helper = (
        'helper[capture="leaf"] ::= ">" next\n' 'next[end_on="<", capture="next"] ::= "X"\n'
        if new_owner
        else 'helper[capture="leaf"] ::= ">" "X"\n'
    )
    grammar = (
        'root ::= boundary "!"\n' 'boundary[end_on=">", capture="value"] ::= helper\n' + helper
    )
    compiled = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic).compile_grammar(
        grammar
    )
    for sequence in ([0, 1, 4], [3, 4]):
        matcher = xgr.GrammarMatcher(compiled)
        assert_mask(matcher, compiled, sequence[0], True)
        assert matcher.accept_token(sequence[0])
        assert_mask(matcher, compiled, 2, False)
        for token in sequence[1:]:
            assert matcher.accept_token(token)
        assert matcher.is_terminated()
        assert matcher.get_captures(deduplicate=False) == [("value", b">")]


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("via_helper", [False, True])
def test_xml_boundary_preserves_owner_nullable_suffix_captures(dynamic, via_helper):
    vocab = [">", "!", ">!", "<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[3])
    body = "helper" if via_helper else '">"'
    grammar = (
        'root ::= boundary "!"\n'
        f'boundary[end_on=">", capture="value"] ::= {body} tail\n'
        'tail[capture="tail"] ::= ""\n'
        'helper[capture="leaf"] ::= ">"\n'
    )
    compiler = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic)
    compiled = compiler.compile_grammar(grammar)
    reference = compiler.compile_grammar(grammar.replace('end_on=">", ', "", 1))
    for sequence in ([0, 1, 3], [2, 3]):
        matcher = xgr.GrammarMatcher(compiled)
        ordinary = xgr.GrammarMatcher(reference)
        for token in sequence:
            assert_mask(matcher, compiled, token, True)
            assert matcher.accept_token(token)
            assert ordinary.accept_token(token)
            for deduplicate in (False, True):
                assert matcher.get_captures(deduplicate=deduplicate) == ordinary.get_captures(
                    deduplicate=deduplicate
                )
        assert matcher.is_terminated()
        assert ordinary.is_terminated()


@pytest.mark.parametrize("dynamic", [False, True])
def test_xml_boundary_new_nullable_occurrence_of_same_owner(dynamic):
    vocab = [">", "!", "<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[2])
    grammar = (
        'root ::= boundary "!"\n'
        'boundary[end_on=">", capture="value"] ::= helper | ""\n'
        'helper[capture="leaf"] ::= ">" boundary\n'
    )
    compiler = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic)
    compiled = compiler.compile_grammar(grammar)
    reference = compiler.compile_grammar(grammar.replace('end_on=">", ', "", 1))
    matcher, ordinary = xgr.GrammarMatcher(compiled), xgr.GrammarMatcher(reference)
    assert matcher.accept_token(0)
    assert ordinary.accept_token(0)
    for deduplicate in (False, True):
        assert matcher.get_captures(deduplicate=deduplicate) == ordinary.get_captures(
            deduplicate=deduplicate
        )
    # The new occurrence can complete via its empty alternative, but cannot consume another
    # marker or be force-completed merely because it has the same rule id as the old owner.
    assert_mask(matcher, compiled, 0, False)
    assert_mask(matcher, compiled, 1, True)
    for token in (1, 2):
        assert matcher.accept_token(token)
        assert ordinary.accept_token(token)
    assert matcher.is_terminated()
    assert matcher.get_captures(deduplicate=False) == ordinary.get_captures(deduplicate=False)


@pytest.mark.parametrize("dynamic", [False, True])
def test_xml_boundary_preserves_empty_prefix_of_unfinished_new_owner(dynamic):
    vocab = [">", "!", "X", ">!", "<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[4])
    grammar = (
        'root ::= boundary "!"\n'
        'boundary[end_on=">", capture="value"] ::= helper\n'
        'helper[capture="leaf"] ::= ">" next\n'
        'next[end_on="<", capture="next"] ::= empty "X"\n'
        'empty[capture="empty"] ::= ""\n'
    )
    compiled = xgr.GrammarCompiler(info, enable_dynamic_compilation=dynamic).compile_grammar(
        grammar
    )
    # The empty prefix completed; neither the new owner nor its enclosing helper completed.
    expected = [("empty", b""), ("value", b">")]
    for sequence in ([0, 1, 4], [3, 4]):
        matcher = xgr.GrammarMatcher(compiled)
        assert_mask(matcher, compiled, sequence[0], True)
        assert matcher.get_captures(deduplicate=False) == []
        assert matcher.accept_token(sequence[0])
        assert_mask(matcher, compiled, 2, False)
        for token in sequence[1:]:
            assert matcher.accept_token(token)
        assert matcher.is_terminated()
        assert matcher.get_captures(deduplicate=False) == expected
        matcher.rollback(len(sequence))
        assert matcher.get_captures(deduplicate=False) == []
        assert matcher.accept_token(3)
        assert matcher.accept_token(4)
        assert matcher.get_captures(deduplicate=False) == expected


@pytest.mark.parametrize(
    "value",
    [
        {"type": "string", "pattern": r"[\s\S]*X"},
        {"type": "string", "minLength": 100},
        {"type": "string", "minLength": 2, "maxLength": 100},
        {"type": "string", "pattern": r"[\s\S]{2,100}X"},
    ],
)
def test_xml_boundary_preserves_masks_before_full_marker(value):
    closing = "</parameter>"
    vocab = ["<parameter=v>", "a", "中。", "<", "/", "parameter", ">", "x", closing]
    compiled = compile_value(value, vocab)
    grammar = str(compiled.grammar)
    annotation = '[end_on="</parameter>"]'
    assert annotation in grammar
    original = xgr.GrammarCompiler(compiled.tokenizer_info).compile_grammar(
        grammar.replace(annotation, "")
    )
    new = xgr.GrammarMatcher(compiled)
    old = xgr.GrammarMatcher(original)
    prefix = ""
    for accepted_token in [0, 1, 2, 3, 4, 5, 7, 3, 4, 5]:
        assert new.accept_token(accepted_token) == old.accept_token(accepted_token)
        prefix += vocab[accepted_token]
        for token, text in enumerate(vocab):
            if closing in prefix + text:
                continue
            expected = old.fork().accept_token(token)
            assert_mask(new, compiled, token, expected)


def test_xml_boundary_preserves_capture_and_temperature():
    vocab = ["<parameter=v>", "abc", "</parameter>"]
    original = compile_value({"type": "string", "pattern": r"[\s\S]*X"}, vocab)
    grammar = (
        str(original.grammar)
        .replace("[end_on=", '[temperature=0.7, capture="parameter", end_on=')
        .replace("root ::=", 'root[capture="outer"] ::=')
    )
    compiled = xgr.GrammarCompiler(original.tokenizer_info).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled, max_rollback_tokens=4)
    assert matcher.accept_token(0)
    assert matcher.temperature == pytest.approx(0.7)
    assert matcher.accept_token(1)
    assert_mask(matcher, compiled, 2, True)
    assert matcher.get_captures() == []  # Speculative marker completion must not emit captures.
    assert matcher.accept_token(2)
    assert matcher.temperature is None
    assert matcher.get_captures() == [
        ("parameter", b"abc</parameter>"),
        ("outer", b"<parameter=v>abc</parameter>"),
    ]
    matcher.rollback(1)
    assert matcher.temperature == pytest.approx(0.7)
    assert matcher.get_captures() == []


@pytest.mark.parametrize(
    "remainder,outer_accepts",
    [
        ("\n<parameter=b>", True),  # the outer allows a newline between parameters
        ("<parameter=b>", True),  # the next parameter starts directly
        ("?!<parameter=b>", False),  # the outer rejects the remaining junk
    ],
)
def test_xml_boundary_same_token_close_then_outer_rules(remainder, outer_accepts):
    # Closing the value commits its boundary; remaining bytes must pass the outer grammar.
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {"type": "string", "pattern": r"[\s\S]*"},
            "b": {"type": "string", "pattern": r"[\s\S]*"},
        },
        "required": ["a", "b"],
    }
    tag = {
        "type": "structural_tag",
        "format": {"type": "json_schema", "json_schema": schema, "style": "qwen_xml"},
    }
    vocab = ["<parameter=a>", f"abc</parameter>{remainder}", "b</parameter>", "<eos>"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[3])
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(json.dumps(tag))
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_token(0)
    assert_mask(matcher, compiled, 1, outer_accepts)
    if outer_accepts:
        assert matcher.accept_token(1)
        assert matcher.accept_token(2)
        assert matcher.is_completed()
