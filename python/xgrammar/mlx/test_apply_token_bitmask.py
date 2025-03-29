import sys

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX tests only run on macOS")
def test_apply_token_bitmask():
    import mlx.core as mx  # import mlx and extension only if running on macOS

    import xgrammar.mlx.extension

    bitmask = mx.array([5], dtype=mx.uint32)  # 5 = 0b0101
    logits = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float16)
    result = xgrammar.mlx.extension.apply_token_bitmask(bitmask, logits)
    expected = mx.array([1.0, -float("inf"), 3.0, -float("inf")], dtype=mx.float16)
    assert mx.all(result == expected), f"result: {result}, expected: {expected}"

    bitmask = mx.array([0xFFFFFFFF, 0x0], dtype=mx.uint32)
    logits = mx.zeros((64,), dtype=mx.float32)
    expected = mx.array([0.0] * 32 + [-float("inf")] * 32, dtype=mx.float32)
    result = xgrammar.mlx.extension.apply_token_bitmask(bitmask, logits)
    assert mx.all(result == expected), f"result: {result}, expected: {expected}"
