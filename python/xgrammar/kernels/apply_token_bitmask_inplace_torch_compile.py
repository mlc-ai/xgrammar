from typing import List, Optional

import torch


@torch.compile(dynamic=True)
def apply_token_bitmask_inplace_kernel_no_indices_torch_compile(
    logits: torch.Tensor, bitmask: torch.Tensor, vocab_size: int
) -> None:
    # logits: (batch_size, vocab_size)
    # bitmask: (batch_size, bitmask_size)
    # mask_expanded: (batch_size, 32 * bitmask_size)
    mask_expanded = torch.repeat_interleave(bitmask, 32, dim=-1)
    # bit_indices: (32 * bitmask_size,)
    bit_indices = torch.arange(32, device=logits.device, dtype=torch.int32).repeat(
        bitmask.shape[-1]
    )
    # bit_masks: (batch_size, 32 * bitmask_size)
    bit_masks = (mask_expanded >> bit_indices) & 1
    bit_masks = bit_masks[..., :vocab_size]
    logits[..., :vocab_size] = logits[..., :vocab_size].masked_fill_(bit_masks == 0, float("-inf"))


@torch.compile(dynamic=True)
def apply_token_bitmask_inplace_kernel_indices_torch_compile(
    logits: torch.Tensor, bitmask: torch.Tensor, vocab_size: int, indices: List[int]
) -> None:
    # logits: (batch_size, vocab_size)
    # bitmask: (batch_size, bitmask_size)
    # mask_expanded: (batch_size, 32 * bitmask_size)
    mask_expanded = torch.repeat_interleave(bitmask[indices], 32, dim=-1)
    # bit_indices: (32 * bitmask_size,)
    bit_indices = torch.arange(32, device=logits.device, dtype=torch.int32).repeat(
        bitmask.shape[-1]
    )
    bit_masks = (mask_expanded >> bit_indices) & 1
    bit_masks = bit_masks[..., :vocab_size]
    logits[indices, :vocab_size] = logits[indices, :vocab_size].masked_fill_(
        bit_masks == 0, float("-inf")
    )


def apply_token_bitmask_inplace_torch_compile(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    vocab_size: Optional[int] = None,
    indices: Optional[List[int]] = None,
) -> None:
    vocab_size = min(logits.shape[-1], bitmask.shape[-1] * 32) if vocab_size is None else vocab_size
    if indices is None:
        apply_token_bitmask_inplace_kernel_no_indices_torch_compile(logits, bitmask, vocab_size)
    else:
        apply_token_bitmask_inplace_kernel_indices_torch_compile(
            logits, bitmask, vocab_size, indices
        )
