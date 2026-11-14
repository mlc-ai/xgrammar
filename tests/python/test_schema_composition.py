"""Composition must preserve the vocabulary and restrictions declared beside a choice."""

import copy
import json

import pytest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string, _json_schema_to_ebnf


def permission_schema():
    return {
        "type": "object",
        "properties": {
            "script": {"type": "string"},
            "note": {"type": "string"},
            "mode": {"type": "string", "enum": ["default", "extra", "external"]},
            "grants": {
                "type": "object",
                "properties": {
                    "network": {
                        "type": "object",
                        "properties": {"enabled": {"const": True}},
                        "required": ["enabled"],
                        "additionalProperties": False,
                    },
                    "files": {
                        "type": "object",
                        "properties": {
                            "read": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                            "write": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
                "anyOf": [
                    {"required": ["network"]},
                    {"properties": {"files": {"required": ["read"]}}, "required": ["files"]},
                    {"properties": {"files": {"required": ["write"]}}, "required": ["files"]},
                ],
            },
            "reason": {"type": "string"},
            "prior": {"type": "string"},
        },
        "required": ["script", "note", "mode"],
        "additionalProperties": False,
        "oneOf": [
            (
                {
                    "properties": {"mode": {"const": mode}},
                    "required": ["mode"],
                    "not": {"anyOf": [{"required": ["grants"]}, {"required": ["prior"]}]},
                }
                if mode != "extra"
                else {
                    "properties": {"mode": {"const": mode}},
                    "required": ["mode", "grants", "reason"],
                }
            )
            for mode in ["default", "extra", "external"]
        ],
    }


def accepts(schema, instance, style="json", any_order=False):
    grammar = xgr.Grammar.from_ebnf(
        _json_schema_to_ebnf(
            schema,
            json_format=style,
            any_whitespace=False,
            separators=(",", ":"),
            any_order=any_order,
        )
    )
    if style == "json":
        text = json.dumps(instance, separators=(",", ":"))
    else:
        text = "".join(
            f"<parameter={key}>"
            + (value if isinstance(value, str) else json.dumps(value, separators=(",", ":")))
            + "</parameter>"
            for key, value in instance.items()
        )
    return _is_grammar_accept_string(grammar, text)


@pytest.mark.parametrize("style", ["json", "qwen_xml"])
@pytest.mark.parametrize("mode", ["default", "external"])
def test_sibling_properties_required_and_forbidden(style, mode, capfd):
    schema = permission_schema()
    valid = {"script": "true", "note": "safe", "mode": mode}
    assert accepts(schema, valid, style)
    assert "falling back to anyOf" not in capfd.readouterr().err
    for key in ["script", "note", "mode"]:
        missing = dict(valid)
        del missing[key]
        assert not accepts(schema, missing, style)
    for key, value in [("grants", {"network": {"enabled": True}}), ("prior", "old"), ("other", 1)]:
        assert not accepts(schema, {**valid, key: value}, style)
    assert accepts(schema, {**valid, "reason": "optional"}, style)


@pytest.mark.parametrize("style", ["json", "qwen_xml"])
@pytest.mark.parametrize(
    "grants",
    [
        {"network": {"enabled": True}},
        {"files": {"read": ["/tmp"]}},
        {"files": {"write": ["/tmp"]}},
        {"network": {"enabled": True}, "files": {"read": ["/tmp"], "write": ["/tmp"]}},
    ],
)
def test_nested_required_only_anyof(style, grants):
    schema = permission_schema()
    valid = {"script": "true", "note": "safe", "mode": "extra", "grants": grants, "reason": "need"}
    assert accepts(schema, valid, style)
    # anyOf is an inclusive union: matching more than one grant branch is legal.
    for key in ["grants", "reason"]:
        missing = dict(valid)
        del missing[key]
        assert not accepts(schema, missing, style)
    for bad in [{}, {"files": {}}, {"network": {"enabled": False}}, {"files": {"read": []}}]:
        invalid = dict(valid, grants=bad)
        assert not accepts(schema, invalid, style)


def test_additional_properties_scope_is_not_widened():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
        "anyOf": [{"properties": {"a": {"minimum": 2}}, "additionalProperties": False}],
    }
    assert accepts(schema, {"a": 2})
    assert not accepts(schema, {"a": 1})
    assert not accepts(schema, {"a": 2, "b": "x"})
    schema["required"] = ["b"]
    with pytest.raises(Exception, match="no satisfiable branches"):
        accepts(schema, {"b": "x"})


@pytest.mark.parametrize("any_order", [False, True])
def test_legal_properties_in_each_order(any_order):
    schema = permission_schema()
    value = {"script": "true", "note": "safe", "mode": "default"}
    if any_order:
        value = dict(reversed(list(value.items())))
    assert accepts(schema, value, "qwen_xml", any_order=any_order)
    # Required identity and duplicate-key enforcement with any_order are a separate feature.


@pytest.mark.parametrize(
    "schema, accepted, rejected",
    [
        (
            {"type": "integer", "minimum": 3, "anyOf": [{"maximum": 5}, {"const": 8}]},
            [3, 5, 8],
            [2, 6],
        ),
        (
            {"type": "string", "enum": ["a", "b"], "anyOf": [{"const": "b"}, {"const": "c"}]},
            ["b"],
            ["a", "c"],
        ),
        ({"type": "integer", "anyOf": [{"type": "number"}, {"type": "string"}]}, [1], ["x", 1.5]),
        (
            {"type": "string", "minLength": 2, "anyOf": [{"maxLength": 3}, {"minLength": 5}]},
            ["ab", "abc", "abcde"],
            ["a", "abcd"],
        ),
    ],
)
def test_scalar_siblings(schema, accepted, rejected):
    for value in accepted:
        assert accepts(schema, value)
    for value in rejected:
        assert not accepts(schema, value)


def test_conflicting_intersection_is_explicit():
    with pytest.raises(Exception, match="Cannot conjoin.*pattern"):
        accepts({"type": "string", "pattern": "a", "anyOf": [{"pattern": "b"}]}, "ab")


def test_oneof_discriminator_conflict_prunes_only_impossible_branch():
    schema = permission_schema()
    schema["properties"]["mode"]["enum"] = ["default", "external"]
    assert accepts(schema, {"script": "true", "note": "safe", "mode": "default"})
    assert not accepts(schema, {"script": "true", "note": "safe", "mode": "extra"})


def test_composition_does_not_mutate_input():
    schema = permission_schema()
    before = copy.deepcopy(schema)
    assert accepts(schema, {"script": "true", "note": "safe", "mode": "default"})
    assert schema == before


@pytest.mark.parametrize("branch", [True, {}])
def test_finite_siblings_with_unconstrained_arm(branch):
    schema = {"type": "string", "enum": ["a", "b", 1], "anyOf": [False, branch]}
    assert accepts(schema, "a")
    assert not accepts(schema, 1)
    with pytest.raises(Exception, match="no satisfiable branches"):
        accepts({"const": 1, "type": "string", "anyOf": [branch]}, 1)


def test_process_style_choice_keeps_all_sibling_properties(capfd):
    schema = {
        "type": "object",
        "properties": {
            "note": {"type": "string"},
            "action": {"enum": ["list", "poll", "kill"]},
            "id": {"type": "string"},
            "cursor": {"type": "integer"},
        },
        "required": ["note", "action"],
        "oneOf": [
            {"properties": {"action": {"const": "list"}}, "required": ["action"]},
            {"properties": {"action": {"const": "poll"}}, "required": ["action", "id", "cursor"]},
            {"properties": {"action": {"const": "kill"}}, "required": ["action", "id"]},
        ],
    }
    for value in [
        {"note": "ok", "action": "list"},
        {"note": "ok", "action": "poll", "id": "x", "cursor": 0},
        {"note": "ok", "action": "kill", "id": "x"},
    ]:
        assert accepts(schema, value, "qwen_xml")
    assert "falling back to anyOf" not in capfd.readouterr().err
    assert not accepts(schema, {"note": "ok", "action": "poll", "id": "x"}, "qwen_xml")
    assert not accepts(schema, {"action": "list"}, "qwen_xml")


def test_preexisting_overlapping_oneof_fallback_is_explicit(capfd):
    assert accepts({"oneOf": [{"type": "integer"}, {"type": "number"}]}, 1)
    assert "falling back to anyOf semantics" in capfd.readouterr().err


def test_reference_dispatch_preserves_preexisting_sibling_limitation():
    schema = {
        "$defs": {"text": {"type": "string"}},
        "$ref": "#/$defs/text",
        "anyOf": [{"minLength": 2}],
    }
    # Existing $ref precedence ignores siblings; general reference intersections
    # remain outside this choice-composition fix.
    assert accepts(schema, "x")


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_unsatisfiable_arm_does_not_discard_valid_alternative(keyword):
    schema = {"type": "integer", "minimum": 5, keyword: [{"maximum": 3}, {"const": 8}]}
    assert accepts(schema, 8)
    assert not accepts(schema, 3)
    assert not accepts(schema, 6)
    schema[keyword] = [{"maximum": 3}]
    with pytest.raises(Exception, match="no satisfiable branches"):
        accepts(schema, 8)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_missing_required_property_in_closed_arm_is_unsatisfiable(keyword):
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "additionalProperties": False,
        keyword: [{"required": ["missing"]}, {"required": ["a"]}],
    }
    assert accepts(schema, {"a": 1})
    assert not accepts(schema, {})


def test_invalid_arm_is_not_discarded_as_unsatisfiable():
    with pytest.raises(Exception, match="must be a number"):
        accepts({"type": "integer", "anyOf": [{"minimum": "bad"}, {"const": 8}]}, 8)


@pytest.mark.parametrize("keyword", ["$defs", "definitions"])
def test_unused_definitions_do_not_block_finite_conjunction(keyword):
    schema = {
        keyword: {"unused": {"type": "string"}},
        "$id": "https://example.com/schema",
        "enum": ["a", "b"],
        "anyOf": [{"const": "a"}, {"const": "b"}],
    }
    assert accepts(schema, "a")
    assert accepts(schema, "b")
    assert not accepts(schema, "c")


@pytest.mark.parametrize("required", ["bad", [1]])
def test_malformed_required_with_forbidden_property_is_reported(required):
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": False,
        "required": required,
        "anyOf": [{"not": {"required": ["a"]}}],
    }
    with pytest.raises(Exception, match="required.*must be"):
        accepts(schema, {})


@pytest.mark.parametrize("type_spec", ["string", ["object", "string"]])
def test_object_only_constraints_do_not_prune_scalar_choice(type_spec):
    schema = {
        "required": ["missing"],
        "additionalProperties": False,
        "anyOf": [{"type": type_spec}],
    }
    assert accepts(schema, "x")
    assert not accepts(schema, {})


def test_forbidden_required_object_retains_nonobject_type():
    schema = {
        "type": ["object", "string"],
        "properties": {"a": False},
        "required": ["a"],
        "additionalProperties": False,
        "anyOf": [{"properties": {"a": {"type": "integer"}}}],
    }
    assert accepts(schema, "x")
    assert not accepts(schema, {"a": 1})


@pytest.mark.parametrize("finite", [{"const": 1}, {"enum": [1]}])
def test_unrelated_allof_dispatch_keeps_existing_finite_behavior(finite):
    schema = {**finite, "allOf": [{"type": "integer"}]}
    assert accepts(schema, 1)
    assert not accepts(schema, 2)


def test_unsatisfiable_type_member_preserves_other_members():
    schema = {
        "type": ["object", "string"],
        "additionalProperties": False,
        "anyOf": [{"minProperties": 1}],
    }
    assert accepts(schema, "x")
    assert not accepts(schema, {})


def test_finite_numeric_intersection_is_exact():
    with pytest.raises(Exception, match="no satisfiable branches"):
        accepts(
            {"const": 9007199254740993, "anyOf": [{"const": 9007199254740992.0}]}, 9007199254740993
        )
    schema = {"enum": [1.5, 2.5], "anyOf": [{"const": 2.5}]}
    assert accepts(schema, 2.5)
    assert not accepts(schema, 1.5)
    schema = {
        "type": "integer",
        "minimum": 9007199254740992,
        "anyOf": [{"minimum": 9007199254740993}],
    }
    assert accepts(schema, 9007199254740993)
    assert not accepts(schema, 9007199254740992)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
@pytest.mark.parametrize("branch", [{"type": "string", "enum": ["a"], "pattern": "^a$"}])
def test_choice_without_sibling_constraints_preserves_finite_schema(keyword, branch):
    schema = {keyword: [branch]}
    assert accepts(schema, "a")
    assert not accepts(schema, "b")


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
@pytest.mark.parametrize("type_spec", ["integer", "number", ["string", "integer"]])
def test_redundant_type_choice_preserves_reference(keyword, type_spec):
    schema = {
        "$defs": {"n": {"type": "integer", "minimum": 2}},
        "$ref": "#/$defs/n",
        keyword: [{"type": type_spec}],
    }
    assert accepts(schema, 2)
    assert not accepts(schema, 1)
    assert not accepts(schema, 2.5)
    assert not accepts(schema, "x")


def test_reference_dispatch_retains_existing_type_sibling_behavior():
    schema = {
        "$defs": {"n": {"type": "number"}},
        "$ref": "#/$defs/n",
        "anyOf": [{"type": "integer"}],
    }
    assert accepts(schema, 2)
    assert accepts(schema, 2.5)


def test_redundant_reference_type_does_not_recursively_parse_target():
    schema = {
        "$defs": {
            "node": {
                "type": "object",
                "properties": {"next": {"$ref": "#/$defs/node", "anyOf": [{"type": "object"}]}},
            }
        },
        "$ref": "#/$defs/node",
    }
    assert accepts(schema, {})
    assert accepts(schema, {"next": {"next": {}}})
    assert not accepts(schema, {"next": 1})


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_finite_object_alternatives_keep_other_property_constraints(keyword):
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["mode", "n"],
        keyword: [
            {"properties": {"mode": {"const": "a"}, "n": {"minimum": 2}}},
            {"properties": {"mode": {"const": "b"}, "n": {"minimum": 2}}},
            {"properties": {"mode": {"const": "c"}, "n": {"maximum": 0}}},
        ],
    }
    for value in [{"mode": "a", "n": 2}, {"mode": "b", "n": 2}, {"mode": "c", "n": 0}]:
        assert accepts(schema, value)
    for value in [{"mode": "a", "n": 0}, {"mode": "c", "n": 2}, {"mode": "d", "n": 2}]:
        assert not accepts(schema, value)


def test_overlapping_finite_object_oneof_keeps_existing_fallback_warning(capfd):
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string"}},
        "required": ["mode"],
        "oneOf": [
            {"properties": {"mode": {"enum": ["a", "b"]}}},
            {"properties": {"mode": {"const": "a"}}},
        ],
    }
    assert accepts(schema, {"mode": "a"})
    assert "falling back to anyOf semantics" in capfd.readouterr().err


def test_object_choice_preserves_nested_property_order():
    schema = {
        "type": "object",
        "required": ["mode", "value"],
        "oneOf": [
            {
                "properties": {
                    "mode": {"const": mode},
                    "value": {
                        "type": "object",
                        "properties": {key: {"type": "integer"} for key in order},
                        "required": ["x", "y"],
                    },
                }
            }
            for mode, order in [("a", ["x", "y"]), ("b", ["y", "x"])]
        ],
    }
    assert accepts(schema, {"mode": "a", "value": {"x": 1, "y": 2}})
    assert accepts(schema, {"mode": "b", "value": {"y": 2, "x": 1}})


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_any_order_retains_choice_branch_consistency(keyword):
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["mode", "n"],
        keyword: [{"properties": {"mode": {"const": mode}}} for mode in ["a", "b"]],
    }
    assert accepts(schema, {"mode": "a", "n": 1}, "qwen_xml", any_order=True)
    assert accepts(schema, {"mode": "b", "n": 1}, "qwen_xml", any_order=True)
    grammar = xgr.Grammar.from_ebnf(
        _json_schema_to_ebnf(schema, json_format="qwen_xml", any_order=True)
    )
    # Duplicate/required identity enforcement is deferred, but a sequence must
    # still stay within one original branch, even when repeating a property.
    assert not _is_grammar_accept_string(
        grammar, "<parameter=mode>a</parameter><parameter=mode>b</parameter>"
    )


@pytest.mark.parametrize("target", [{"type": "integer"}, {"const": 1}])
@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_reference_dispatch_preserves_multihop_and_finite_targets(target, keyword):
    schema = {
        "$defs": {"n": {"$ref": "#/$defs/value"}, "value": target},
        "$ref": "#/$defs/n",
        keyword: [{"type": "integer"}],
    }
    assert accepts(schema, 1)
    assert not accepts(schema, "x")
    assert not accepts(schema, 1.5)
    if "const" in target:
        assert not accepts(schema, 2)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_finite_pattern_with_redundant_type_choice_preserves_existing_behavior(keyword):
    schema = {"type": "string", "enum": ["a"], "pattern": "^a$", keyword: [{"type": "string"}]}
    assert accepts(schema, "a")
    assert not accepts(schema, "b")
    assert not accepts(schema, 1)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_type_sibling_with_finite_pattern_choice_preserves_existing_behavior(keyword):
    schema = {"type": "string", keyword: [{"type": "string", "enum": ["a"], "pattern": "^a$"}]}
    assert accepts(schema, "a")
    assert not accepts(schema, "b")
    assert not accepts(schema, 1)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
@pytest.mark.parametrize("finite", [{"enum": ["a"]}, {"const": "a"}])
@pytest.mark.parametrize("finite_is_sibling", [False, True])
def test_finite_values_can_imply_redundant_choice_type(keyword, finite, finite_is_sibling):
    finite = {**finite, "pattern": "^a$"}
    typed = {"type": "string"}
    sibling, branch = (finite, typed) if finite_is_sibling else (typed, finite)
    schema = {**sibling, keyword: [branch]}
    assert accepts(schema, "a")
    assert not accepts(schema, "b")
    assert not accepts(schema, 1)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_finite_identity_requires_every_candidate_to_match_type(keyword):
    schema = {"enum": ["a", 2], "pattern": "^a$", keyword: [{"type": "string"}]}
    with pytest.raises(Exception, match="Cannot conjoin.*finite values"):
        accepts(schema, "a")
    schema[keyword] = [{"type": ["string", "integer"]}]
    assert accepts(schema, "a")
    assert accepts(schema, 2)
    assert not accepts(schema, 3)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_impossible_finite_type_arm_preserves_valid_alternative(keyword):
    schema = {"enum": ["a"], "pattern": "^a$", keyword: [{"type": "integer"}, {"type": "string"}]}
    assert accepts(schema, "a")
    assert not accepts(schema, "b")
    assert not accepts(schema, 1)
    schema[keyword] = [{"type": "integer"}]
    with pytest.raises(Exception, match="no satisfiable branches"):
        accepts(schema, "a")


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
@pytest.mark.parametrize(
    "metadata, other_metadata",
    [
        ({"$defs": {"unused": {"type": "integer"}}}, {"$defs": {"unused": {"type": "string"}}}),
        (
            {"definitions": {"unused": {"type": "integer"}}},
            {"definitions": {"unused": {"type": "string"}}},
        ),
        ({"$id": "https://example.com/a"}, {"$id": "https://example.com/b"}),
        ({"title": "A type constraint"}, {"title": "A finite constraint"}),
    ],
)
def test_finite_type_proof_ignores_existing_neutral_metadata(keyword, metadata, other_metadata):
    finite = {"enum": ["a"], "pattern": "^a$"}
    for added in [{}, other_metadata]:
        for constraint in [metadata, {"type": "string", **metadata}]:
            for sibling, branch in [
                ({**finite, **added}, constraint),
                (constraint, {**finite, **added}),
            ]:
                schema = {**sibling, keyword: [branch]}
                assert accepts(schema, "a")
                assert not accepts(schema, "b")
                assert not accepts(schema, 1)
        schema = {
            **finite,
            **added,
            keyword: [{"type": "integer", **metadata}, {"type": "string", **metadata}],
        }
        assert accepts(schema, "a")
        assert not accepts(schema, 1)
        schema[keyword] = [{"type": "integer", **metadata}]
        with pytest.raises(Exception, match="no satisfiable branches"):
            accepts(schema, "a")


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_finite_type_identity_does_not_bypass_explicit_type_conflicts(keyword):
    for finite_type, branch_type in [
        ("integer", "string"),
        (["integer", "null"], ["string", "boolean"]),
    ]:
        finite = {"type": finite_type, "enum": ["a"], "pattern": "^a$"}
        typed = {"type": branch_type}
        for sibling, branch in [(finite, typed), (typed, finite)]:
            with pytest.raises(Exception, match="no satisfiable branches"):
                accepts({**sibling, keyword: [branch]}, "a")
    for finite_type, branch_type in [("integer", "number"), ("number", "integer")]:
        schema = {
            "type": finite_type,
            "enum": [1],
            "pattern": "^a$",
            keyword: [{"type": branch_type}],
        }
        assert accepts(schema, 1)
        assert not accepts(schema, 2)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
@pytest.mark.parametrize(
    "type_name, values, bounds, valid, invalid",
    [
        ("string", ["a", "bbb"], {"minLength": 2}, "bbb", "a"),
        ("string", ["a", "bbb"], {"maxLength": 2}, "a", "bbb"),
        ("integer", [1, 3], {"minimum": 2}, 3, 1),
        ("integer", [1, 3], {"maximum": 2}, 1, 3),
        ("number", [1, 2.5], {"exclusiveMinimum": 1}, 2.5, 1),
        ("number", [1, 2.5], {"exclusiveMaximum": 2.5}, 1, 2.5),
    ],
)
def test_finite_fallback_keeps_normalized_bound_filtering(
    keyword, type_name, values, bounds, valid, invalid
):
    for irrelevant in [
        {"properties": {}},
        {"required": ["unused"]},
        {"additionalProperties": False},
        {"patternProperties": {}},
        {"propertyNames": {"const": "unused"}},
        {"unevaluatedProperties": False},
        {"minProperties": 1},
        {"maxProperties": 0},
    ]:
        finite = {"enum": values, **bounds, **irrelevant}
        typed = {"type": type_name}
        for schema in [
            {**finite, keyword: [typed]},
            {**typed, keyword: [finite]},
            {**finite, **typed, keyword: [True]},
            {**finite, **typed, keyword: [{}]},
        ]:
            assert accepts(schema, valid)
            assert not accepts(schema, invalid)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_metadata_collision_does_not_merge_general_definition_scopes(keyword):
    first = {"$defs": {"value": {"type": "string"}}}
    second = {"$defs": {"value": {"type": "integer"}}}
    schemas = [
        {**first, "type": "string", keyword: [{**second, "type": "string"}]},
        {**first, "enum": ["a"], keyword: [{**second, "$ref": "#/$defs/value"}]},
        {**first, "enum": ["a"], keyword: [{**second, "pattern": "^a$"}]},
    ]
    for schema in schemas:
        with pytest.raises(Exception, match="Cannot conjoin"):
            accepts(schema, "a")
