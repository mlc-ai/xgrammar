"""String exclusions on schema formats, including raw K3 argument values."""

import json

import pytest

import xgrammar as xgr
from xgrammar.structural_tag import JSONSchemaFormat, SequenceFormat, StructuralTag, TagFormat
from xgrammar.testing import _is_grammar_accept_string

CONTROLS = ["<|open|>", "<|close|>", "<|sep|>"]
ARG_BEGIN = '<|open|>argument key="value" type="string"<|sep|>'
ARG_END = "<|close|>argument<|sep|>"


def schema_tag(schema, style="json", excludes=CONTROLS, **kwargs):
    return StructuralTag(
        format=JSONSchemaFormat(json_schema=schema, style=style, excludes=excludes, **kwargs)
    )


def leaf_grammar(schema, style, excludes=CONTROLS):
    if style == "kimi_k3_xml":
        schema = {
            "type": "object",
            "properties": {"value": schema},
            "required": ["value"],
            "additionalProperties": False,
        }
    return xgr.Grammar.from_structural_tag(schema_tag(schema, style, excludes))


def string_instance(value, style):
    if style == "kimi_k3_xml":
        return ARG_BEGIN + value + ARG_END
    return json.dumps(value, ensure_ascii=False)


@pytest.mark.parametrize("style", ["json", "kimi_k3_xml"])
@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string"},
        {"type": "string", "minLength": 1, "maxLength": 16},
        {"enum": ["safe", "你好🦋", *(f"x{control}y" for control in CONTROLS)]},
    ],
)
def test_string_excludes(style, schema):
    grammar = leaf_grammar(schema, style)
    for value in ("safe", "你好🦋"):
        assert _is_grammar_accept_string(grammar, string_instance(value, style))
    for value in (f"x{control}y" for control in CONTROLS):
        assert not _is_grammar_accept_string(grammar, string_instance(value, style))


@pytest.mark.parametrize("style", ["json", "kimi_k3_xml"])
@pytest.mark.parametrize(
    "schema,accepted,rejected",
    [
        ({"type": "string", "pattern": "^a[abc]*z$"}, ["aabz", "acbz"], ["aab"]),
        ({"type": "string", "pattern": "^你好[abc]*$"}, ["你好aa", "你好bc"], ["hello"]),
        ({"type": "string", "format": "email"}, ["aa@example.com", "bc@example.com"], ["aa"]),
    ],
)
def test_pattern_and_format_strings_are_not_filtered(style, schema, accepted, rejected):
    # A pattern or format is the schema's own contract for the string: it is matched as is, and
    # the exclusions do not apply to it.
    grammar = leaf_grammar(schema, style, excludes=["bc", "cb"])
    for value in accepted:
        assert _is_grammar_accept_string(grammar, string_instance(value, style))
    for value in rejected:
        assert not _is_grammar_accept_string(grammar, string_instance(value, style))


@pytest.mark.parametrize("style", ["json", "kimi_k3_xml"])
@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "minLength": 2},
        {"type": "string", "maxLength": 3},
        {"type": "string", "minLength": 2, "maxLength": 3},
    ],
)
def test_excludes_drop_length_constraints(style, schema, capfd):
    # Bounding the length of a filtered string multiplies the grammar by the bound, so the
    # length constraints are dropped when exclusions are present; the exclusions still apply.
    bounded = leaf_grammar(schema, style, excludes=[])
    assert "Ignoring" not in capfd.readouterr().err
    unbounded = leaf_grammar(schema, style, excludes=["bc", "cb"])
    warning = capfd.readouterr().err
    assert "Ignoring" in warning and "JSONSchemaFormat.excludes" in warning
    assert f"minLength={schema.get('minLength', 0)}" in warning
    assert f"maxLength={schema.get('maxLength', -1)}" in warning
    plain = leaf_grammar({"type": "string"}, style, excludes=["bc", "cb"])
    # Same rules as the unbounded string (up to the reference to the shared string rule), not
    # one rule per position and exclusion state.
    assert str(unbounded).count("::=") <= str(plain).count("::=") + 1
    for value in ("a", "abde", "ax" * 10):
        assert _is_grammar_accept_string(unbounded, string_instance(value, style))
        assert _is_grammar_accept_string(bounded, string_instance(value, style)) == (
            schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 10**9)
        )
    for value in ("bc", "acbz", "a" * 20 + "bc"):
        assert not _is_grammar_accept_string(unbounded, string_instance(value, style))


@pytest.mark.parametrize("style", ["json", "kimi_k3_xml"])
@pytest.mark.parametrize("any_order", [False, True])
def test_nested_excludes(style, any_order):
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/payload"}},
        "required": ["value"],
        "additionalProperties": False,
        "$defs": {
            "payload": {"type": "array", "items": {"$ref": "#/$defs/node"}},
            "node": {
                "type": "object",
                "properties": {"next": {"$ref": "#/$defs/node"}},
                "additionalProperties": True,
            },
        },
    }
    grammar = xgr.Grammar.from_structural_tag(schema_tag(schema, style, any_order=any_order))

    def instance(payload):
        if style == "json":
            return json.dumps({"value": payload}, ensure_ascii=False)
        return (
            ARG_BEGIN.replace('type="string"', 'type="array"')
            + json.dumps(payload, ensure_ascii=False)
            + ARG_END
        )

    assert _is_grammar_accept_string(grammar, instance([{"next": {"text": "你好", "n": 42}}]))
    assert not _is_grammar_accept_string(grammar, instance([{"next": {"text": "a<|open|>b"}}]))
    assert not _is_grammar_accept_string(grammar, instance([{"next": {"a<|close|>b": "text"}}]))


@pytest.mark.parametrize(
    "schema,filtered",
    [
        (True, True),
        ({"type": "object", "additionalProperties": True}, True),
        (
            {
                "type": "object",
                "properties": {"": {"type": "integer"}},
                "additionalProperties": True,
            },
            True,
        ),
        # Keys constrained by a pattern are matched as is, like patterned values.
        (
            {
                "type": "object",
                "patternProperties": {"^[a-z]*$": {"type": "integer"}},
                "additionalProperties": False,
            },
            False,
        ),
        (
            {
                "type": "object",
                "properties": {"fixed": {"type": "integer"}},
                "patternProperties": {"^[a-z]*$": {"type": "integer"}},
                "additionalProperties": False,
            },
            False,
        ),
        (
            {
                "type": "object",
                "propertyNames": {"pattern": "^[a-z]*$"},
                "additionalProperties": {"type": "integer"},
            },
            False,
        ),
    ],
)
def test_property_name_excludes(schema, filtered):
    grammar = xgr.Grammar.from_structural_tag(schema_tag(schema, excludes=["bad"]))
    assert _is_grammar_accept_string(grammar, '{"good":1}')
    assert _is_grammar_accept_string(grammar, '{"bad":1}') == (not filtered)
    assert _is_grammar_accept_string(grammar, '{"xbady":1}') == (not filtered)


@pytest.mark.parametrize("literal", [{"text": "BAD"}, {"BAD": "text"}, ["BAD"]])
def test_nested_literal_excludes(literal):
    grammar = xgr.Grammar.from_structural_tag(schema_tag({"enum": [literal, 42]}, excludes=["BAD"]))
    assert _is_grammar_accept_string(grammar, "42")
    assert not _is_grammar_accept_string(grammar, json.dumps(literal))


@pytest.mark.parametrize("keyword", ["const", "enum"])
@pytest.mark.parametrize(
    "key,value,excludes,encoded_key",
    [("value", 'a"b', ["\\", "BAD"], "value"), ('quo"ted', "safe", ['"', "BAD"], "quo&quot;ted")],
)
def test_k3_root_literal_excludes_use_emitted_text(keyword, key, value, excludes, encoded_key):
    literal = {key: value}
    schema = {keyword: literal if keyword == "const" else [literal, {key: "BAD"}]}
    grammar = xgr.Grammar.from_structural_tag(schema_tag(schema, "kimi_k3_xml", excludes))
    begin = f'<|open|>argument key="{encoded_key}" type="string"<|sep|>'
    assert _is_grammar_accept_string(grammar, begin + value + ARG_END)
    assert not _is_grammar_accept_string(grammar, begin + "BAD" + ARG_END)
    assert not _is_grammar_accept_string(grammar, json.dumps(literal))


def test_excludes_match_emitted_string_content_only():
    grammar = xgr.Grammar.from_structural_tag(schema_tag(True, excludes=["ab", ":", "12", "true"]))
    for value in ('{"a":12}', "true"):
        # Check punctuation and non-string literals independently of string exclusions.
        assert _is_grammar_accept_string(grammar, value)
    assert _is_grammar_accept_string(grammar, '["a","b"]')
    assert _is_grammar_accept_string(grammar, r'"\u0061b"')
    for value in ('"ab"', '":"', '"12"', '"true"'):
        assert not _is_grammar_accept_string(grammar, value)


@pytest.mark.parametrize("style", ["json", "kimi_k3_xml"])
def test_unicode_and_overlapping_excludes(style):
    grammar = leaf_grammar({"type": "string"}, style, ["aba", "bab", "你好", "aba"])
    for value in ("aaba", "bbab", "x你好y"):
        assert not _is_grammar_accept_string(grammar, string_instance(value, style))
    for value in ("ab", "你x好", "🦋"):
        assert _is_grammar_accept_string(grammar, string_instance(value, style))


def test_excludes_unicode_boundaries_and_grammar_roundtrip():
    grammar = leaf_grammar({"type": "string"}, "json", ["坏", *CONTROLS])
    for restored in (
        grammar,
        xgr.Grammar.deserialize_json(grammar.serialize_json()),
        xgr.Grammar.from_ebnf(str(grammar)),
    ):
        for codepoint in (0x7F, 0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFF, 0x10000, 0x10FFFF):
            assert _is_grammar_accept_string(restored, string_instance(chr(codepoint), "json"))
        for value in ("候", "好候"):
            assert _is_grammar_accept_string(restored, string_instance(value, "json"))
        for value in ("坏", "x<|open|>y"):
            assert not _is_grammar_accept_string(restored, string_instance(value, "json"))


@pytest.mark.parametrize("style", ["json", "kimi_k3_xml"])
def test_unrecognized_format_keeps_exclusions(style):
    # A format without a built-in regex leaves the string unconstrained; the exclusions still
    # apply, and without exclusions the grammar is the one main produces.
    schema = {"type": "string", "format": "password"}
    grammar = leaf_grammar(schema, style, excludes=["BAD"])
    assert _is_grammar_accept_string(grammar, string_instance("safe", style))
    assert not _is_grammar_accept_string(grammar, string_instance("xBADy", style))
    plain = leaf_grammar(schema, style, excludes=[])
    assert _is_grammar_accept_string(plain, string_instance("xBADy", style))
    if style == "kimi_k3_xml":
        assert "root_prop_0 ::= (([\\0-\\U0010ffff]*))" in str(plain)


@pytest.mark.parametrize(
    "style,begin,end",
    [
        ("qwen_xml", "<parameter=value>", "</parameter>"),
        ("minimax_xml", '<parameter name="value">', "</parameter>"),
        ("glm_xml", "<arg_key>value</arg_key><arg_value>", "</arg_value>"),
        ("deepseek_xml", '<｜DSML｜parameter name="value" string="true">', "</｜DSML｜parameter>"),
    ],
)
def test_excludes_other_xml_styles(style, begin, end):
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    grammar = xgr.Grammar.from_structural_tag(schema_tag(schema, style, excludes=["BAD"]))
    assert _is_grammar_accept_string(grammar, begin + "你好" + end)
    assert not _is_grammar_accept_string(grammar, begin + "BAD" + end)


@pytest.mark.parametrize("excludes", [None, "bad", [1], [""], ["ok", ""]])
def test_invalid_excludes(excludes):
    # Pass serialized JSON to exercise native validation independently of Pydantic.
    tag = {
        "type": "structural_tag",
        "format": {"type": "json_schema", "json_schema": {"type": "string"}, "excludes": excludes},
    }
    with pytest.raises(Exception, match="excludes must be an array of non-empty strings"):
        xgr.Grammar.from_structural_tag(json.dumps(tag))


@pytest.mark.parametrize("excluded", [" safe", "safe ", "\n"])
def test_xml_excludes_reject_padding_boundaries(excluded):
    with pytest.raises(RuntimeError, match="must not start or end with formatting whitespace"):
        leaf_grammar({"const": "safe"}, "kimi_k3_xml", [excluded])
    # JSON delimiter whitespace is outside strings, so these exclusions are supported there.
    grammar = leaf_grammar({"type": "string"}, "json", [excluded])
    assert _is_grammar_accept_string(grammar, '"safe"')


def test_xml_excludes_allow_internal_whitespace():
    grammar = leaf_grammar({"type": "string"}, "kimi_k3_xml", ["a b"])
    assert _is_grammar_accept_string(grammar, string_instance("safe", "kimi_k3_xml"))
    assert not _is_grammar_accept_string(grammar, string_instance("a b", "kimi_k3_xml"))


def test_excludes_default_and_cache_isolation():
    schema = {"type": "string"}
    without = {"type": "structural_tag", "format": {"type": "json_schema", "json_schema": schema}}
    assert str(xgr.Grammar.from_structural_tag(without)) == str(
        xgr.Grammar.from_structural_tag(schema_tag(schema, excludes=[]))
    )
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo([]), cache_enabled=True)
    for excludes, allowed in (([], True), (["bad"], False), (["other"], True), (["bad"], False)):
        grammar = compiler.compile_structural_tag(schema_tag(schema, excludes=excludes)).grammar
        assert _is_grammar_accept_string(grammar, '"bad"') == allowed
    tag = StructuralTag(
        format=SequenceFormat(
            elements=[
                JSONSchemaFormat(json_schema=schema, excludes=["bad"]),
                JSONSchemaFormat(json_schema=schema),
            ]
        )
    )
    grammar = xgr.Grammar.from_structural_tag(tag)
    assert _is_grammar_accept_string(grammar, '"safe""bad"')
    assert not _is_grammar_accept_string(grammar, '"bad""safe"')


def test_k3_excludes_token_mask_and_argument_end():
    grammar = leaf_grammar({"type": "string"}, "kimi_k3_xml")
    vocab = [
        ARG_BEGIN,
        "hello",
        "<|open|>",
        "<|close|>",
        "argument",
        "response",
        "<|sep|>",
        "<|op",
        "en|>",
        "hello" + ARG_END,
        "[EOS]",
    ]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=len(vocab) - 1)
    compiled = xgr.GrammarCompiler(info).compile_grammar(grammar)
    for cg in (compiled, xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)):
        matcher = xgr.GrammarMatcher(cg)
        mask = xgr.allocate_token_bitmask(1, len(vocab))

        def allowed(token):
            matcher.fill_next_token_bitmask(mask)
            return bool((int(mask[0, token // 32]) >> (token % 32)) & 1)

        assert matcher.accept_token(0)
        assert allowed(1)
        assert not allowed(2)
        assert allowed(3)  # CLOSE remains legal as the start of the argument terminator.
        assert allowed(9)  # A token can cross the boundary from value into its wrapper.
        assert matcher.accept_token(7)
        assert not allowed(8)  # Exclusions also match text split across tokens.
        matcher.rollback(1)
        assert matcher.accept_token(1)
        assert matcher.accept_token(3)
        assert allowed(4)
        assert not allowed(5)
        assert matcher.accept_token(4)
        assert matcher.accept_token(6)
        assert allowed(10)
        assert matcher.accept_token(10)
        assert matcher.is_terminated()


@pytest.mark.parametrize("style", ["json", "kimi_k3_xml"])
@pytest.mark.parametrize(
    "format_name,good,bad",
    [
        ("hostname", "example.com", "c-"),
        ("uri", "http://example.com/a", "abc://@@"),
        ("json-pointer", "/a", "a"),
        ("relative-json-pointer", "0/a", "a"),
    ],
)
def test_excludes_preserve_builtin_format_language(style, format_name, good, bad):
    schema = {"type": "string", "format": format_name}
    baseline = leaf_grammar(schema, style, [])
    filtered = leaf_grammar(schema, style, CONTROLS)
    for grammar in (baseline, filtered):
        assert _is_grammar_accept_string(grammar, string_instance(good, style))
        assert not _is_grammar_accept_string(grammar, string_instance(bad, style))


@pytest.mark.parametrize("style", ["json", "kimi_k3_xml"])
def test_excludes_preserve_unicode_pattern_and_json_escaping(style):
    schema = {"type": "string", "pattern": "^你.*好$"}
    baseline = leaf_grammar(schema, style, [])
    filtered = leaf_grammar(schema, style, CONTROLS)
    for grammar in (baseline, filtered):
        assert _is_grammar_accept_string(grammar, string_instance('你"好', style))
        assert _is_grammar_accept_string(grammar, string_instance("你好", style))
        assert not _is_grammar_accept_string(grammar, string_instance("你", style))


@pytest.mark.parametrize("style", ["json", "kimi_k3_xml"])
@pytest.mark.parametrize(
    "pattern,excludes,accepted,rejected",
    [
        ("^(BAD|safe)$", ["BAD"], ["safe", "BAD"], ["BA你", "BA🦋"]),
        ("^你.*好$", ["你"], ["你好", "你🦋好"], ["", "safe", "🦋"]),
    ],
)
def test_pattern_branches_are_not_filtered(style, pattern, excludes, accepted, rejected):
    grammar = leaf_grammar({"type": "string", "pattern": pattern}, style, excludes)
    for value in accepted:
        assert _is_grammar_accept_string(grammar, string_instance(value, style))
    for value in rejected:
        assert not _is_grammar_accept_string(grammar, string_instance(value, style))


@pytest.mark.parametrize("schema", [{"const": "BAD"}, {"enum": ["BAD"]}])
@pytest.mark.parametrize("safe_branch", [False, True])
def test_excluded_branch_mask_rejects_unicode_and_stop_token(schema, safe_branch):
    if safe_branch:
        schema = {"anyOf": [schema, {"const": "safe"}]}
    grammar = leaf_grammar(schema, "json", ["BAD"])
    vocab = ["你", "🦋", '"你"', '"BAD"', '"safe"', "[EOS]"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[len(vocab) - 1])
    compiled = xgr.GrammarCompiler(info).compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled)
    mask = xgr.allocate_token_bitmask(1, len(vocab))
    matcher.fill_next_token_bitmask(mask)
    for token_id, text in enumerate(vocab):
        expected = safe_branch and text == '"safe"'
        assert bool((int(mask[0, token_id // 32]) >> (token_id % 32)) & 1) == expected
        matcher.reset()
        assert matcher.accept_token(token_id) == expected
        assert _is_grammar_accept_string(grammar, text) == expected
    assert not matcher.is_completed()


@pytest.mark.parametrize("style", ["json", "kimi_k3_xml"])
@pytest.mark.parametrize("blocked", [{"const": "BAD"}, {"enum": ["BAD"]}])
def test_excluded_literal_branch_preserves_valid_alternative(style, blocked):
    grammar = leaf_grammar({"anyOf": [blocked, {"const": "safe"}]}, style, ["BAD"])
    assert _is_grammar_accept_string(grammar, string_instance("safe", style))
    for value in ("BAD", "你", "🦋"):
        assert not _is_grammar_accept_string(grammar, string_instance(value, style))
    if style == "json":
        assert not _is_grammar_accept_string(grammar, "你")


@pytest.mark.parametrize("required", [False, True])
def test_excluded_fixed_key(required):
    schema = {
        "type": "object",
        "properties": {"bad": {"type": "integer"}},
        "required": ["bad"] if required else [],
        "additionalProperties": False,
    }
    grammar = leaf_grammar(schema, "json", ["bad"])
    assert _is_grammar_accept_string(grammar, "{}") == (not required)
    for text in ('{"bad":1}', '{"你":1}', "{你:1}"):
        assert not _is_grammar_accept_string(grammar, text)


def test_excluded_optional_const_property_can_be_omitted():
    schema = {
        "type": "object",
        "properties": {"optional": {"const": "BAD"}, "required": {"const": "safe"}},
        "required": ["required"],
        "additionalProperties": False,
    }
    grammar = leaf_grammar(schema, "json", ["BAD"])
    assert _is_grammar_accept_string(grammar, '{"required":"safe"}')
    for text in ('{"optional":"BAD","required":"safe"}', '{"optional":你,"required":"safe"}'):
        assert not _is_grammar_accept_string(grammar, text)


@pytest.mark.parametrize("style", ["json", "kimi_k3_xml"])
def test_excludes_large_open_object(style):
    properties = {
        f"customer_{index:03d}_preferred_contact_channel": {"type": "integer"}
        for index in range(40)
    }
    schema = {"type": "object", "properties": properties, "additionalProperties": True}
    grammar = leaf_grammar(schema, style)

    def instance(value):
        text = json.dumps(value, ensure_ascii=False)
        if style == "kimi_k3_xml":
            return ARG_BEGIN.replace('type="string"', 'type="object"') + text + ARG_END
        return text

    first_key = next(iter(properties))
    assert _is_grammar_accept_string(grammar, instance({first_key: 1, "备注": "safe"}))
    assert not _is_grammar_accept_string(grammar, instance({first_key: "wrong type"}))
    for control in CONTROLS:
        assert not _is_grammar_accept_string(grammar, instance({"extra": control}))
        assert not _is_grammar_accept_string(grammar, instance({f"extra{control}": 1}))


def test_quote_excludes_token_mask_keeps_closing_quote():
    grammar = leaf_grammar({"type": "string"}, "json", ['"'])
    vocab = ['"', "hello", r"\"", r"\u0022", "[EOS]"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[4])
    matcher = xgr.GrammarMatcher(xgr.GrammarCompiler(info).compile_grammar(grammar))
    mask = xgr.allocate_token_bitmask(1, len(vocab))
    assert matcher.accept_token(0)
    matcher.fill_next_token_bitmask(mask)
    assert int(mask[0, 0]) & (1 << 0)  # Empty string may close.
    assert not int(mask[0, 0]) & (1 << 2)  # Emitted escaped quote contains the excluded quote.
    assert int(mask[0, 0]) & (1 << 3)  # Its Unicode-escape spelling does not.
    assert not matcher.accept_token(2)
    assert matcher.accept_token(3)
    assert matcher.accept_token(0)
    assert matcher.accept_token(4)
    assert matcher.is_terminated()


def test_nested_k3_three_controls_mask_and_argument_boundary():
    payload_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    grammar = leaf_grammar(payload_schema, "kimi_k3_xml", CONTROLS)
    begin = ARG_BEGIN.replace('type="string"', 'type="object"')
    vocab = [
        begin,
        '{"text":"',
        "hello",
        *CONTROLS,
        "<|op",
        "en|>",
        '"}',
        "argument",
        "response",
        'hello"}' + ARG_END,
        "[EOS]",
    ]
    ids = {text: index for index, text in enumerate(vocab)}
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[ids["[EOS]"]])
    compiled = xgr.GrammarCompiler(info).compile_grammar(grammar)
    for cg in (compiled, xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)):
        matcher = xgr.GrammarMatcher(cg)
        mask = xgr.allocate_token_bitmask(1, len(vocab))

        def allowed(text):
            matcher.fill_next_token_bitmask(mask)
            token_id = ids[text]
            return bool((int(mask[0, token_id // 32]) >> (token_id % 32)) & 1)

        assert matcher.accept_token(ids[begin])
        assert matcher.accept_token(ids['{"text":"'])
        for control in CONTROLS:
            assert not allowed(control)
            assert not matcher.accept_token(ids[control])
        assert allowed('hello"}' + ARG_END)  # One token may finish JSON and consume XML suffix.
        crossing_matcher = xgr.GrammarMatcher(cg)
        for text in (begin, '{"text":"', 'hello"}' + ARG_END, "[EOS]"):
            assert crossing_matcher.accept_token(ids[text])
        assert crossing_matcher.is_terminated()
        assert matcher.accept_token(ids["<|op"])
        assert not allowed("en|>")
        assert not matcher.accept_token(ids["en|>"])
        matcher.rollback(1)
        assert matcher.accept_token(ids["hello"])
        assert matcher.accept_token(ids['"}'])
        assert allowed("<|close|>")
        assert not allowed("<|open|>")
        assert not allowed("<|sep|>")
        assert matcher.accept_token(ids["<|close|>"])
        assert allowed("argument")
        assert not allowed("response")
        assert matcher.accept_token(ids["argument"])
        assert allowed("<|sep|>")
        assert matcher.accept_token(ids["<|sep|>"])
        assert allowed("[EOS]")
        assert matcher.accept_token(ids["[EOS]"])
        assert matcher.is_terminated()


def test_json_string_response_schema_keeps_k3_response_end():
    schema = {"type": "string"}
    end = "<|close|>response<|sep|>"
    tag = StructuralTag(
        format=TagFormat(
            begin="", content=JSONSchemaFormat(json_schema=schema, excludes=CONTROLS), end=end
        )
    )
    vocab = ['"hello', *CONTROLS, '"', end, '"' + end, "[EOS]"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[len(vocab) - 1])
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(tag)
    for closing_tokens in ([4, 5], [6]):
        matcher = xgr.GrammarMatcher(compiled)
        mask = xgr.allocate_token_bitmask(1, len(vocab))
        assert matcher.accept_token(0)
        matcher.fill_next_token_bitmask(mask)
        for index in (1, 2, 3):
            assert not int(mask[0, 0]) & (1 << index)
        for index in (4, 6):
            assert int(mask[0, 0]) & (1 << index)
        for index in [*closing_tokens, 7]:
            assert matcher.accept_token(index)
        assert matcher.is_terminated()


@pytest.mark.parametrize("style", ["cohere_xml", "minimax_m3_xml"])
def test_unsupported_style_excludes_are_not_silently_ignored(style):
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "additionalProperties": False,
    }
    xgr.Grammar.from_structural_tag(schema_tag(schema, style, excludes=[]))
    with pytest.raises(RuntimeError, match=f"excludes is not supported for {style}"):
        xgr.Grammar.from_structural_tag(schema_tag(schema, style))


@pytest.mark.parametrize("required", [False, True])
def test_deepseek_v41_excluded_fixed_key(required):
    schema = {
        "type": "object",
        "properties": {"xBADy": {"type": "integer"}},
        "required": ["xBADy"] if required else [],
        "additionalProperties": False,
    }
    grammar = xgr.Grammar.from_structural_tag(
        schema_tag(schema, "deepseek_v4_1_xml", excludes=["BAD"])
    )
    assert _is_grammar_accept_string(grammar, "") == (not required)
    assert not _is_grammar_accept_string(
        grammar, '<｜DSML｜ parameter name="xBADy" string="false">1</｜DSML｜ parameter>'
    )


def test_deepseek_v41_excludes_preserve_value_type_and_parameter_end():
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/value"}},
        "required": ["value"],
        "additionalProperties": False,
        "$defs": {
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                ]
            }
        },
    }
    grammar = xgr.Grammar.from_structural_tag(
        schema_tag(schema, "deepseek_v4_1_xml", excludes=["BAD", "</｜DSML｜"])
    )
    prefix = '<｜DSML｜ parameter name="value" string="'
    end = "</｜DSML｜ parameter>"
    for body in ('true">safe', 'false">{"text":"safe"}'):
        assert _is_grammar_accept_string(grammar, prefix + body + end)
    for body in ('true">BAD', 'false">{"text":"BAD"}', 'false">"safe"'):
        assert not _is_grammar_accept_string(grammar, prefix + body + end)
    assert not _is_grammar_accept_string(grammar, prefix + 'false">{"text":"' + end + '"}' + end)
