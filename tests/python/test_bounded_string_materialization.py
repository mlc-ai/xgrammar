"""Compare materialized string masks with the original counted-character grammar."""

import json
from itertools import product

import pytest
import torch

import xgrammar as xgr
from xgrammar.testing import _get_masked_tokens_from_bitmask


@pytest.mark.parametrize(
    "lower,upper",
    [
        (0, 0),
        (0, 1),
        (1, 1),
        (0, 32),
        (16, 32),
        (32, 32),
        (0, 128),
        (0, 129),
        (113, 129),
        (0, 256),
        (240, 256),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_bounded_string_materialization(lower, upper, nested):
    # Include long tokens, split UTF-8, forbidden characters, and tokens which leave
    # the bounded string. No downloaded tokenizer is needed for the differential test.
    vocabulary = [bytes([b]) for b in range(256)]
    vocabulary += [b"a" * n for n in (2, 16, 31, 32, 33, 127, 128, 129)]
    vocabulary += [
        "é".encode(),
        "中".encode(),
        "😀".encode(),
        b'aa"',
        b'",',
        b'","tail":"x"}',
        b'a","tail":"x"}',
        b"\\n",
        b'\\"',
        b"\\u0061",
        b"<eos>",
    ]
    stop_id = len(vocabulary) - 1
    info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[stop_id])
    compiler = xgr.GrammarCompiler(info)
    string_schema = {"type": "string", "minLength": lower, "maxLength": upper}
    prefix, suffix = ('{"content":', ',"tail":"x"}') if nested else ("", "")
    schema = (
        {
            "type": "object",
            "properties": {"content": string_schema, "tail": {"const": "x"}},
            "required": ["content", "tail"],
            "additionalProperties": False,
        }
        if nested
        else string_schema
    )
    character = r'[^"\\\r\n]'
    bounded = " ".join([character] * lower)
    rules = []
    if upper > lower:
        bounded += " rest0"
        for i in range(upper - lower):
            next_rule = f" rest{i + 1}" if i + 1 < upper - lower else ""
            rules.append(f'rest{i} ::= "" | {character}{next_rule}')
    reference = (
        "root ::= "
        + json.dumps(prefix + '"')
        + " "
        + bounded
        + " "
        + json.dumps('"' + suffix)
        + "\n"
        + "\n".join(rules)
    )
    contexts = [
        compiler.compile_json_schema(schema, any_whitespace=False, separators=(",", ":")),
        compiler.compile_grammar(reference),
    ]
    masks = [xgr.allocate_token_bitmask(1, len(vocabulary)) for _ in contexts]
    # Both ends of the bound and multibyte characters at the boundary.
    values = {"a" * lower, "a" * upper}
    if upper:
        values.add("a" * (upper - 1) + "😀")
        values.add("é" * upper)
    for value in values:
        matchers = [xgr.GrammarMatcher(context) for context in contexts]
        encoded = (prefix + json.dumps(value, ensure_ascii=False) + suffix).encode()
        for token in encoded:
            for matcher, mask in zip(matchers, masks):
                matcher.fill_next_token_bitmask(mask)
            assert torch.equal(*masks), (lower, upper, nested, value, token)
            before = masks[0].clone()
            for matcher in matchers:
                assert matcher.accept_token(token)
                # Speculative traversal must restore the same capacity and mask.
                matcher.rollback(1)
            for matcher, mask in zip(matchers, masks):
                matcher.fill_next_token_bitmask(mask)
                assert torch.equal(mask, before)
                assert matcher.accept_token(token)
        for matcher, mask in zip(matchers, masks):
            matcher.fill_next_token_bitmask(mask)
            assert matcher.accept_token(stop_id)
        assert torch.equal(*masks)


@pytest.mark.parametrize("upper", [1024, 4096, 100000])
@pytest.mark.parametrize("near_lower_bound", [False, True])
def test_large_string_capacity_and_rollback(upper, near_lower_bound):
    lower = upper - 16 if near_lower_bound else 0
    # A direct length oracle independent of the grammar compiler. The closing
    # tokens also finish a following field, exercising cross-rule continuations.
    candidates = [
        (body, closes)
        for body in ["a", "a" * 128, "a" * 129, "a" * 256, "中", "😀", "中" * 256]
        for closes in [False, True]
    ] + [("", True)]
    vocabulary = [
        (body + ('", "tail": "x"}' if closes else "")).encode() for body, closes in candidates
    ] + [b"<eos>"]
    info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[len(candidates)])
    schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "minLength": lower, "maxLength": upper},
            "tail": {"const": "x"},
        },
        "required": ["content", "tail"],
        "additionalProperties": False,
    }
    matcher = xgr.GrammarMatcher(xgr.GrammarCompiler(info).compile_json_schema(schema))
    assert matcher.accept_string('{"content": "')
    mask = xgr.allocate_token_bitmask(1, len(vocabulary))
    previous = 0
    for position in sorted(
        {
            p
            for p in [
                0,
                1,
                127,
                128,
                129,
                255,
                256,
                257,
                upper // 2,
                lower - 1,
                lower,
                upper - 257,
                upper - 256,
                upper - 129,
                upper - 128,
                upper - 1,
                upper,
            ]
            if 0 <= p <= upper
        }
    ):
        assert matcher.accept_string("a" * (position - previous))
        previous = position
        matcher.fill_next_token_bitmask(mask)
        for token_id, (body, closes) in enumerate(candidates):
            length = position + len(body)
            expected = length <= upper and (not closes or length >= lower)
            allowed = bool((int(mask[0, token_id // 32]) >> (token_id % 32)) & 1)
            assert allowed == expected, (lower, upper, position, body, closes)
        before = mask.clone()
        # Roll back a multi-token speculative branch across the materialized/counter
        # transition where possible, and check that the complete mask is restored.
        accepted = min(3, upper - position)
        for _ in range(accepted):
            assert matcher.accept_token(0)  # one ASCII character
        if accepted:
            matcher.rollback(accepted)
            matcher.fill_next_token_bitmask(mask)
            assert torch.equal(mask, before)
    assert matcher.accept_token(len(candidates) - 1)
    assert matcher.accept_token(len(candidates))
    assert matcher.is_terminated()


def test_duplicate_bitsets_preserve_uncertain_alternatives_after_serialization():
    # Thousands of accepted and rejected tokens force bitset storage. The two
    # alternatives share accepted bitsets but have different length/exit rules;
    # deduplicating the bitset union must not skip either uncertain-token trial.
    vocabulary = [
        "".join(chars)
        for alphabet in ["abcdefghijklm", "ABCDEFGHIJKLM"]
        for chars in product(alphabet, repeat=3)
    ]
    vocabulary += [
        body + ending for body in ["", "a", "a" * 128, "a" * 300] for ending in ["!", "?"]
    ]
    vocabulary += ["a", "<eos>"]
    info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[len(vocabulary) - 1])
    grammar = 'root ::= left "!" | right "?"\nleft ::= [a-z]{0,256}\nright ::= [a-z]{200,512}'
    compiled = xgr.GrammarCompiler(info).compile_grammar(grammar)
    serialized = compiled.serialize_json()
    assert any(
        mask[1]["store_type"] == 2 for mask in json.loads(serialized)["adaptive_token_mask_cache"]
    )
    contexts = [compiled, xgr.CompiledGrammar.deserialize_json(serialized, info)]
    mask = xgr.allocate_token_bitmask(1, len(vocabulary))
    for context in contexts:
        matcher = xgr.GrammarMatcher(context)
        previous = 0
        for position in [0, 1, 127, 128, 129, 199, 200, 255, 256, 257, 511, 512]:
            assert matcher.accept_string("a" * (position - previous))
            previous = position
            matcher.fill_next_token_bitmask(mask)
            expected_rejected = []
            for token_id, token in enumerate(vocabulary):
                end = token[-1] if token[-1] in "!?" else ""
                body = token[:-1] if end else token
                length = position + len(body)
                allowed = all("a" <= c <= "z" for c in body)
                allowed &= (
                    length <= 256
                    if end == "!"
                    else 200 <= length <= 512 if end == "?" else length <= 512
                )
                if not allowed:
                    expected_rejected.append(token_id)
            assert _get_masked_tokens_from_bitmask(mask, len(vocabulary)) == expected_rejected
            before = mask.clone()
            matcher.fill_next_token_bitmask(mask)
            assert torch.equal(mask, before)
        assert matcher.accept_token(vocabulary.index("?"))
        assert matcher.accept_token(len(vocabulary) - 1)
        matcher.reset()
        matcher.fill_next_token_bitmask(mask)
        assert matcher.accept_token(vocabulary.index("!"))
