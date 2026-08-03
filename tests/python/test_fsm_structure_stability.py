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
        "c078f43b05f1f6a2e2e62d5aa2ad8ee28b3aa9358ef26091041dcc89a9669e67",
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
        "52f3bf57ed3057820c5e096486022f11daed6f44b1f0838a821d10fb28588ce5",
    ),
    (
        "ebnf",
        'root ::= "abcdefghijklmnopqrstuvwxyz" [0-9] "ABCDEFGHIJKLMNOPQRSTUVWXYZ"',
        "58b89fec9137ba57ef63aba58efa201d7a64b9c949b80d0e4376467613779aab",
    ),
    (
        "ebnf",
        'root ::= "中文" [\\u4e00-\\u9fff] "端"',
        "31f8a746759114a682fef038356e409a388be3e73bb327fff5f24bae4e0b1754",
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
        "ad034a35713202f53ea01d16b48486edb9e475a1a79fcc44b2cf157cfd06663d",
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
        "42c5b5b7fd66abeae7946a0d2aadd427a7d237a52b6711a8ae3097f138b89b04",
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
        "76fa4d2475e1afcc9f230fabdb2c87655922e1786638db853478a5a1a2ce62d1",
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
        "542f8b15f4229b4fb842bb57d4372c066e6757dba14675c2d0de25dfb661a300",
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
