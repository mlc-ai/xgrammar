import torch
import triton
import triton.language as tl

from typing import List, Optional, Union


@triton.jit
def apply_token_bitmask_inplace_kernel(
    logits_ptr,
    bitmask_ptr,
    indices_ptr,
    vocab_size,
    bitmask_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = tl.load(indices_ptr + pid)

    for block_offset in tl.range(0, vocab_size, BLOCK_SIZE, num_stages=3):
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        bitmask_offsets = block_offset // 32 + tl.arange(0, BLOCK_SIZE // 32)
        vocab_mask = offsets < vocab_size
        packed_bitmask_mask = bitmask_offsets < bitmask_size
        logits = tl.load(logits_ptr + batch_id * vocab_size + offsets, vocab_mask)
        packed_bitmask = tl.load(
            bitmask_ptr + pid * bitmask_size + bitmask_offsets, packed_bitmask_mask
        )
        bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
        bitmask = bitmask.reshape(BLOCK_SIZE)

        logits = tl.where(bitmask, -float("inf"), logits)
        tl.store(logits_ptr + batch_id * vocab_size + offsets, logits, vocab_mask)


def apply_token_bitmask_inplace_triton(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
):
    def ceil_div(a, b):
        return (a + b - 1) // b

    BLOCK_SIZE = 2048
    # Check input tensor shapes.
    if logits.ndim == 2:
        batch_size, vocab_size = logits.shape
    elif logits.ndim == 1:
        batch_size = 1
        (vocab_size,) = logits.shape
    else:
        raise ValueError(f"Invalid logits tensor shape {logits.shape}")

    if indices is None:
        indices = torch.arange(batch_size, dtype=torch.int32, device=logits.device)
    elif isinstance(indices, list):
        indices = torch.tensor(indices, dtype=torch.int32, device=logits.device)

    grid = lambda meta: (indices.size(0),)

    apply_token_bitmask_inplace_kernel[grid](
        logits,
        bitmask.view(torch.uint32),
        indices,
        vocab_size,
        ceil_div(vocab_size, 32),
        BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32 // (16 // logits.element_size()),
    )
