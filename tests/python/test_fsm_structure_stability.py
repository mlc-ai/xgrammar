"""Guard the exact FSM layout produced by grammar compilation."""

import hashlib
import json

import pytest

import xgrammar as xgr
from xgrammar.testing import _print_grammar_fsms

FSM_STRUCTURE_CASES = [
    (
        "ebnf",
        'root ::= "a" [0-9] "b"',
        "55b4f598dd003190ab972e2c89246b46b7ad102506fdfa785003c270866e8572",
    ),
    (
        "ebnf",
        'root ::= "hello" [a-zA-Z_] [0-9]* "world"',
        "bf85be1292ab8bfcbdb08c09e4508ffe855d110cc19b36fd849c6724897c6a50",
    ),
    (
        "ebnf",
        'root ::= "<" [a-c]* ">"',
        "19cef267fb8eddbc9c5ac8fba92c135600fc720258abf0bd58cb6c346fe75c5a",
    ),
    (
        "ebnf",
        'root ::= "x" [^0-9] "y" [^a-z]* "z"',
        "fd5ab210c129fe4da573c175bfd0e97440d30a88d038a160e5bcbeb2f9fc00d0",
    ),
    (
        "ebnf",
        'root ::= "(" inner ")" inner\ninner ::= [0-9] [0-9]',
        "4de2c028805f73a42e2eb3766defb94a7b6a9cfb735fd9d9a7bf8c610e4b4012",
    ),
    (
        "ebnf",
        'root ::= "a" item{2,5} "b"\nitem ::= [0-9]',
        "eddead662499416bbd37a4b70d70912ace125deaeaa8fc1f48fbe61dde26b8c7",
    ),
    (
        "ebnf",
        'root ::= item{3,}\nitem ::= "ab" [xy]',
        "e80cfab029429aa5111b5b1b95c002f7fa4bc7b0a4e88ecae2d507067dba7b6d",
    ),
    (
        "ebnf",
        'root ::= "ab" [0-9] | "cd" sub | sub sub\nsub ::= [a-f] "q"',
        "29090db9590a3b07af82126f08ec25663f657fde2b659c7d172bf6da227861ba",
    ),
    (
        "ebnf",
        'root ::= "abcdefghijklmnopqrstuvwxyz" [0-9] "ABCDEFGHIJKLMNOPQRSTUVWXYZ"',
        "58b89fec9137ba57ef63aba58efa201d7a64b9c949b80d0e4376467613779aab",
    ),
    (
        "ebnf",
        'root ::= "中文" [\\u4e00-\\u9fff] "端"',
        "a2d7f36e1d3a0453d4267fb3937ed07a4abaac7361e051d3aaaaf7a6a58de462",
    ),
    ("ebnf", 'root ::= "only"', "7880ea9fd56d9cca1976659fa1ccd4b7b347ab2688800e96bdafcfaa591019e5"),
    ("ebnf", "root ::= [0-9]", "e210303c507b9b23c732079d40b3f0b01af0de6deb86c8974148dbb92b92896c"),
    (
        "ebnf",
        'root ::= "" "a" ""',
        "8698019eb93735ab521cfd930bf7688a276b4ebfb33ccd7cea20f8139f99fbeb",
    ),
    (
        "ebnf",
        'root ::= a b c\na ::= "x" [0-9]*\nb ::= a "y" | [^xyz]\nc ::= b{1,3} "end"',
        "692de2ea3a886cc7aca953da58fe7dc66c26bc39a71b01fdeee53cfb58e89561",
    ),
    (
        "json_schema",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "nested": {
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                },
            },
            "required": ["name", "age", "tags", "nested"],
            "additionalProperties": False,
        },
        "4d050454cf0d4a9a579a7f0d33803c6a98a0354649945be7e4830505c473ab1c",
    ),
    (
        "json_schema",
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": "[a-f0-9]{8}"},
                    "kind": {"enum": ["alpha", "beta", "gamma"]},
                },
                "required": ["id", "kind"],
            },
            "minItems": 1,
            "maxItems": 4,
        },
        "d0284724594dcee4009b06e2ef7d210b7614f062482caa5df925a10619af4323",
    ),
    (
        "structural_tag",
        {
            "type": "structural_tag",
            "format": {
                "type": "dispatch",
                "rules": [
                    ["<tool=alpha>", {"type": "const_string", "value": "A"}],
                    ["<tool=beta>", {"type": "json_schema", "json_schema": {"type": "object"}}],
                ],
                "loop": True,
                "excludes": [],
            },
        },
        "61e2ccf44a3c589547de488dfbd6cd84a9a4fa39ce86ed3f71a11e388cc4050c",
    ),
]


@pytest.fixture(scope="module")
def compiler():
    vocabulary = [chr(character) for character in range(33, 127)] + ["中", "文", "端"]
    tokenizer_info = xgr.TokenizerInfo(vocabulary)
    return xgr.GrammarCompiler(tokenizer_info, max_threads=1, cache_enabled=False)


@pytest.mark.parametrize("kind, source, expected_digest", FSM_STRUCTURE_CASES)
def test_compiled_fsm_structure_stable(compiler, kind, source, expected_digest):
    """Detect changes to state numbers, edge order, endpoints, or complete-FSM layout."""
    if kind == "ebnf":
        compiled = compiler.compile_grammar(source)
    elif kind == "json_schema":
        compiled = compiler.compile_json_schema(
            json.dumps(source, separators=(",", ":")), any_whitespace=True
        )
    else:
        compiled = compiler.compile_structural_tag(source)

    printed_fsm = _print_grammar_fsms(compiled.grammar)
    actual_digest = hashlib.sha256(printed_fsm.encode()).hexdigest()
    assert actual_digest == expected_digest
