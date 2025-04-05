"""MLX kernel for applying token bitmasks."""

import itertools

import mlx.core as mx


@mx.compile
def apply_token_bitmask_mlx(bitmask: mx.array, logits: mx.array, vocab_size: int):
    """Apply a token bitmask to logits using MLX for Metal GPUs.

    Args:
        bitmask: A tensor of shape (batch_size, (vocab_size + 31) // 32) containing
            the bitmask. Each bit in the bitmask determines whether the corresponding
            token is allowed (1) or not (0).
        logits: A tensor of shape (batch_size, vocab_size) containing the logits.

    Returns:
        The logits with -inf for tokens that are not allowed.
    """
    bitmap = mx.array(
        [l[::-1] for l in itertools.product(*[[float("-inf"), 0]] * 8)], dtype=logits.dtype
    )
    bitmask = bitmask.view(mx.uint8)
    return logits[..., :vocab_size] + bitmap[bitmask].flatten(-2)[..., :vocab_size]
