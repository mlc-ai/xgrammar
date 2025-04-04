from typing import List, Optional, Union

import torch

try:
    import mlx.core as mx
    from mlx.core.fast import metal_kernel
except ImportError as err:
    raise ImportError("MLX is not installed") from err


def apply_token_bitmask_inplace_metal(bitmask: torch.Tensor, logits: mx.array) -> mx.array:
    """Apply a bitmask to logits using Metal. The bitmask is a 01 bitwise compressed tensor,
    where 0 means the token is masked and 1 means the token is not masked. After applying the bitmask,
    the masked logits will be set to -inf.

    Parameters
    ----------
    bitmask : torch.Tensor
        The bitmask tensor to apply. Should be of type int32.

    logits : mx.array
        The logits tensor to apply the bitmask to.

    Returns
    -------
    mx.array
        The masked logits where invalid tokens have their logits set to -inf.
    """
    # Check input dtype
    assert bitmask.dtype == torch.int32, "bitmask must be of type int32"

    # Convert PyTorch bitmask to MLX array
    bitmask_mx = mx.array(bitmask.detach().cpu().numpy())

    # Ensure 2D tensors for logits and bitmask
    logits = logits.reshape((1, logits.size)) if len(logits.shape) == 1 else logits
    bitmask_mx = (
        bitmask_mx.reshape((1, bitmask_mx.size)) if len(bitmask_mx.shape) == 1 else bitmask_mx
    )

    # Vocabulary size is from logits shape
    vocab_size = logits.shape[1]

    # Check bitmask size
    bits_per_int32 = 32
    required_bitmask_width = (vocab_size + bits_per_int32 - 1) // bits_per_int32
    assert bitmask_mx.shape[1] >= required_bitmask_width, (
        f"Bitmask width too small: need at least {required_bitmask_width} int32s for "
        f"logits' width {vocab_size}, but got {bitmask_mx.shape[1]}"
    )

    # Check batch sizes match
    assert (
        logits.shape[0] == bitmask_mx.shape[0]
    ), f"Batch size mismatch: logits {logits.shape[0]} vs bitmask {bitmask_mx.shape[0]}"
    batch_size = logits.shape[0]

    # Define the Metal kernel source
    source = """
    // Get thread position in grid
    uint token_idx = thread_position_in_grid.x;
    uint batch_idx = thread_position_in_grid.y;

    // Get actual scalar values from input parameters
    int logits_stride_val = logits_stride[0];
    int bitmask_stride_val = bitmask_stride[0];

    // Calculate bitmask position
    int bitmask_block_idx = token_idx / 32;
    int bit_position = token_idx % 32;

    // Calculate logits index
    int logits_idx = batch_idx * logits_stride_val + token_idx;

    // Always copy the value first
    result[logits_idx] = logits[logits_idx];

    // Apply mask if within bitmask bounds
    if (bitmask_block_idx < bitmask_stride_val) {
        int packed_bitmask = bitmask[batch_idx * bitmask_stride_val + bitmask_block_idx];
        bool is_masked = ((packed_bitmask >> bit_position) & 1) == 0;

        if (is_masked) {
            result[logits_idx] = -INFINITY;
        }
    }
    """

    # Compile and execute the kernel
    kernel = metal_kernel(
        name="apply_token_bitmask",
        source=source,
        input_names=["logits", "bitmask", "logits_stride", "bitmask_stride"],
        output_names=["result"],
    )

    # Prepare inputs and execute kernel
    inputs = [
        logits,
        bitmask_mx,
        mx.array([vocab_size], dtype=mx.int32),
        mx.array([bitmask_mx.shape[1]], dtype=mx.int32),
    ]

    # Calculate thread group size - for Metal, 256 threads per group is typically good
    threads_per_group = 256

    outputs = kernel(
        inputs=inputs,
        output_shapes=[logits.shape],
        output_dtypes=[logits.dtype],
        grid=(vocab_size, batch_size, 1),  # Launch one thread per token-row pair
        threadgroup=(threads_per_group, 1, 1),  # Use optimal thread group size
    )

    return outputs[0]
