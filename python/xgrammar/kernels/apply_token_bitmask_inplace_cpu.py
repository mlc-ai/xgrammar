"""CPU implementation for in-place applying token mask."""

from typing import List, Optional, Union

import torch


def _bitmask_to_bool_mask(bitmask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    bits_per_block = 32
    bitmask_size = bitmask.size(-1)
    # Expand bitmask to bits
    shifts = torch.arange(bits_per_block, device=bitmask.device, dtype=torch.int32)
    bits = (bitmask.unsqueeze(-1) >> shifts) & 1  # Shape (*, bits_per_block)
    bits = bits.view(bitmask.size(0), -1)  # Shape (batch_size, bitmask_size * bits_per_block)
    bool_mask = bits[:, :vocab_size].to(torch.bool)  # Truncate to vocab_size
    return bool_mask


def apply_token_bitmask_inplace_cpu(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
) -> None:
    """Exactly the same as `apply_token_bitmask_inplace()`, but `logits` is on the CPU.
    So we use CPU implementation rather than launching a CUDA kernel.
    """
    if logits.device.type != "cpu":
        raise ValueError("logits must be on CPU")
    if bitmask.device != logits.device:
        raise ValueError("bitmask must be on the same device as logits")
    if bitmask.dim() != logits.dim():
        raise ValueError(
            f"bitmask and logits must have the same number of dimensions, but "
            + f"got {bitmask.dim()} and {logits.dim()}"
        )
    if bitmask.dim() != 1 and bitmask.dim() != 2:
        raise ValueError("Unsupported logits and bitmask dimensions: {}".format(bitmask.dim()))

    # Expand both to (batch_size, xxx_size), where batch_size is 1
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        bitmask = bitmask.unsqueeze(0)

    batch_size, vocab_size = logits.size()
    if indices is None:
        if batch_size != bitmask.size(0):
            raise ValueError("Batch size of logits and bitmask must match")
        bool_mask = _bitmask_to_bool_mask(bitmask, vocab_size)  # Shape (batch_size, vocab_size)
        logits.masked_fill_(~bool_mask, -float("inf"))
    else:
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.int32, device=logits.device)
        len_indices = len(indices)
        if len_indices != bitmask.size(0):
            raise ValueError("The length of indices and bitmask's batch size must match.")
        bool_mask = _bitmask_to_bool_mask(bitmask, vocab_size)  # Shape (len_indices, vocab_size)
        logits[indices] = logits[indices].masked_fill_(~bool_mask, -float("inf"))
