"""CPU implementation for in-place applying token mask."""

from typing import List, Optional, Union

import torch

from ..base import _core


def apply_token_bitmask_inplace_cpu(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    vocab_size: Optional[int] = None,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
) -> None:
    """Apply token bitmask in-place on CPU."""
    if logits.device.type != "cpu":
        raise ValueError("logits must be on CPU")
    if bitmask.device.type != "cpu":
        raise ValueError("bitmask must be on CPU")
    if logits.dtype != torch.float32:
        raise ValueError("logits must be of type float32")
    if bitmask.dtype != torch.int32:
        raise ValueError("bitmask must be of type int32")
    if logits.dim() != 1 and logits.dim() != 2:
        raise ValueError("logits should be 1D or 2D, but got {}D".format(logits.dim()))
    if bitmask.dim() != 1 and bitmask.dim() != 2:
        raise ValueError("bitmask should be 1D or 2D, but got {}D".format(bitmask.dim()))

    logits_shape = (1, logits.shape[0]) if logits.dim() == 1 else (logits.shape[0], logits.shape[1])
    bitmask_shape = (
        (1, bitmask.shape[0]) if bitmask.dim() == 1 else (bitmask.shape[0], bitmask.shape[1])
    )
    vocab_size = min(logits.shape[-1], bitmask.shape[-1] * 32) if vocab_size is None else vocab_size

    _core.kernels.apply_token_bitmask_inplace_cpu(
        logits.data_ptr(), logits_shape, bitmask.data_ptr(), bitmask_shape, vocab_size, indices
    )
