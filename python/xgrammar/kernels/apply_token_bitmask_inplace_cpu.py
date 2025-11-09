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
    if bitmask.dtype != torch.int32:
        raise ValueError("bitmask must be of type int32")
    if logits.dim() != 1 and logits.dim() != 2:
        raise ValueError("logits should be 1D or 2D, but got {}D".format(logits.dim()))
    if bitmask.dim() != 1 and bitmask.dim() != 2:
        raise ValueError("bitmask should be 1D or 2D, but got {}D".format(bitmask.dim()))

    logits_shape = (1, logits.shape[0]) if logits.dim() == 1 else (logits.shape[0], logits.shape[1])
    logits_stride = logits.stride()
    logits_stride = (
        (logits_stride[0], 1) if logits.dim() == 1 else (logits_stride[0], logits_stride[1])
    )

    bitmask_shape = (
        (1, bitmask.shape[0]) if bitmask.dim() == 1 else (bitmask.shape[0], bitmask.shape[1])
    )
    bitmask_stride = bitmask.stride()
    bitmask_stride = (
        (bitmask_stride[0], 1) if bitmask.dim() == 1 else (bitmask_stride[0], bitmask_stride[1])
    )

    vocab_size = min(logits.shape[-1], bitmask.shape[-1] * 32) if vocab_size is None else vocab_size

    if logits.dtype == torch.float32:
        _core.kernels.apply_token_bitmask_inplace_cpu(
            logits.data_ptr(),
            logits_shape,
            logits_stride,
            bitmask.data_ptr(),
            bitmask_shape,
            bitmask_stride,
            vocab_size,
            indices,
            "float32",
        )
    elif logits.dtype == torch.bfloat16:
        _core.kernels.apply_token_bitmask_inplace_cpu(
            logits.data_ptr(),
            logits_shape,
            logits_stride,
            bitmask.data_ptr(),
            bitmask_shape,
            bitmask_stride,
            vocab_size,
            indices,
            "bfloat16",
        )
    elif logits.dtype == torch.float16:
        _core.kernels.apply_token_bitmask_inplace_cpu(
            logits.data_ptr(),
            logits_shape,
            logits_stride,
            bitmask.data_ptr(),
            bitmask_shape,
            bitmask_stride,
            vocab_size,
            indices,
            "float16",
        )
    else:
        raise ValueError("logits must be of type float32 or bfloat16/float16")
