"""CPU implementation for in-place applying token mask."""

import torch


def apply_token_bitmask_inplace_cpu(logits: torch.Tensor, bitmask: torch.Tensor) -> None:
    """Exactly the same as `apply_token_bitmask_inplace()`, but `logits` is on the CPU.
    So we use CPU implementation rather than launching a CUDA kernel.
    """
    if logits.device.type != "cpu":
        raise ValueError("logits must be on CPU")
    if bitmask.device != logits.device:
        bitmask = bitmask.to(logits.device)

    def int32_to_bits(x: torch.Tensor) -> torch.Tensor:
        bits_per_block = 32
        x_shape = x.shape
        x = x.view(-1, 1)  # Flatten and add a dimension for bits
        shifts = torch.arange(
            bits_per_block, device=x.device, dtype=torch.int32
        )  # (bits_per_block,)
        bits = (x >> shifts) & 1  # Extract bits
        bits = bits.view(*x_shape, bits_per_block)  # Reshape back to original shape with bits
        return bits

    # Determine if batch dimension is present
    if logits.dim() == 1:
        # No batch dimension
        vocab_size = logits.size(0)
        bitmask_size = bitmask.size(0)
        # Expand bitmask to bits
        bits = int32_to_bits(bitmask)
        bits = bits.view(-1)  # Flatten
        bits = bits[:vocab_size]  # Truncate to vocab_size
        mask = bits == 0
        logits[mask] = -float("inf")
    elif logits.dim() == 2:
        batch_size, vocab_size = logits.size()
        batch_size_bm, bitmask_size = bitmask.size()
        if batch_size != batch_size_bm:
            raise ValueError("Batch size of logits and bitmask must match")
        # Expand bitmask to bits
        bits = int32_to_bits(bitmask)  # Shape (batch_size, bitmask_size, bits_per_block)
        bits = bits.view(batch_size, -1)  # Shape (batch_size, bitmask_size * bits_per_block)
        bits = bits[:, :vocab_size]  # Truncate to vocab_size
        mask = bits == 0  # Shape (batch_size, vocab_size)
        logits[mask] = -float("inf")
    else:
        raise ValueError("Unsupported logits dimensions: {}".format(logits.dim()))
