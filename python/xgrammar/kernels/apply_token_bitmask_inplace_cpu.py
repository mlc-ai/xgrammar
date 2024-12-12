"""CPU implementation for in-place applying token mask."""

import time
from typing import List, Optional, Union

import torch

from ..base import _core


def apply_token_bitmask_inplace_cpu(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
) -> None:
    """Apply token bitmask in-place on CPU."""
    start = time.monotonic_ns()
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

    # logits_shape = list(logits.shape)
    # bitmask_shape = list(bitmask.shape)
    # end = time.monotonic_ns()
    # print(f"Python prepare: {(end - start) / 1e3} us")

    # _core.kernels.apply_token_bitmask_inplace_cpu(
    #     logits.data_ptr(),
    #     logits_shape,
    #     bitmask.data_ptr(),
    #     bitmask_shape,
    #     indices,
    # )

    logits_shape = (logits.shape[0],) if logits.dim() == 1 else (logits.shape[0], logits.shape[1])
    bitmask_shape = (
        (bitmask.shape[0],) if bitmask.dim() == 1 else (bitmask.shape[0], bitmask.shape[1])
    )
    end = time.monotonic_ns()
    print(f"Python prepare: {(end - start) / 1e3} us")

    # start = time.monotonic_ns()
    _core.kernels.apply_token_bitmask_inplace_cpu(
        logits.data_ptr(),
        logits_shape,
        bitmask.data_ptr(),
        bitmask_shape,
        indices,
    )
    end = time.monotonic_ns()
    print(f"Python inner: {(end - start) / 1e3} us")
