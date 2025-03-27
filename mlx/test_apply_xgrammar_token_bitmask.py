import mlx.core as mx
from mlx_sample_extensions import _ext

bitmask = mx.array([5], dtype=mx.uint32)  # 5 = 0b0101
logits = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float16)
result = _ext.apply_xgrammar_token_bitmask(bitmask, logits)
expected = mx.array([1.0, -float("inf"), 3.0, -float("inf")], dtype=mx.float16)
assert mx.all(result == expected), f"result: {result}, expected: {expected}"

bitmask = mx.array([0xFFFFFFFF, 0x0], dtype=mx.uint32)
logits = mx.zeros((64,), dtype=mx.float32)
expected = mx.array([0.0] * 32 + [-float("inf")] * 32, dtype=mx.float32)
result = _ext.apply_xgrammar_token_bitmask(bitmask, logits)
assert mx.all(result == expected), f"result: {result}, expected: {expected}"
