"""Check bounded strings and skipped fields against a length/order oracle."""

import pytest
import torch

import xgrammar as xgr


@pytest.mark.parametrize("count", [4, 16, 40])
@pytest.mark.parametrize("upper", [16, 4096])
def test_optional_string_continuations(count, upper):
    fields = [f"field_{i}" for i in range(count)]
    schema = {
        "type": "object",
        "properties": {
            **{key: {"type": "string", "maxLength": upper} for key in fields},
            "tail": {"const": True},
        },
        "required": ["tail"],
        "additionalProperties": False,
    }
    bodies = ["", "a", "a" * 127, "a" * 256, "中", "😀"]
    # Each candidate either stays inside the string, closes the object, skips
    # ahead to another field, or illegally repeats an earlier field.
    candidates = [(body, None) for body in bodies if body]
    candidates += [(body, next_field) for body in bodies for next_field in range(count + 1)]
    vocabulary = []
    for body, next_field in candidates:
        suffix = ""
        if next_field is not None:
            suffix = '",'
            if next_field < count:
                suffix += f'"{fields[next_field]}":"",'
            suffix += '"tail":true}'
        vocabulary.append((body + suffix).encode())
    vocabulary.append(b"<eos>")
    info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[len(candidates)])
    ctx = xgr.GrammarCompiler(info).compile_json_schema(
        schema, any_whitespace=False, separators=(",", ":")
    )
    mask = xgr.allocate_token_bitmask(1, len(vocabulary))
    for index in [0, count // 2, count - 1]:
        for position in sorted({0, min(16, upper), upper - 1, upper}):
            matcher = xgr.GrammarMatcher(ctx)
            assert matcher.accept_string('{"' + fields[index] + '":"' + "a" * position)
            matcher.fill_next_token_bitmask(mask)
            before = mask.clone()
            for token_id, (body, next_field) in enumerate(candidates):
                expected = position + len(body) <= upper and (
                    next_field is None or next_field > index
                )
                allowed = bool((int(mask[0, token_id // 32]) >> (token_id % 32)) & 1)
                assert allowed == expected, (index, position, body, next_field)
                if expected and next_field is not None:
                    assert matcher.accept_token(token_id)
                    matcher.rollback(1)
                    matcher.fill_next_token_bitmask(mask)
                    assert torch.equal(mask, before)
