"""CPU implementation for in-place applying token mask."""

from typing import List, Optional, Union

import torch

from ..base import _core


# Error messages
class BitmaskValidationError(ValueError):
    """Raised when bitmask validation fails."""

    @staticmethod
    def dims_error(tensor_name: str, dims: int) -> str:
        return f"{tensor_name} should be 1D or 2D, but got {dims}D"


LOGITS_CPU_ERROR = "logits must be on CPU"
BITMASK_CPU_ERROR = "bitmask must be on CPU"
LOGITS_TYPE_ERROR = "logits must be of type float32"
BITMASK_TYPE_ERROR = "bitmask must be of type int32"
MAX_DIMS = 2


def apply_token_bitmask_inplace_cpu(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
) -> None:
    """Apply token bitmask in-place on CPU."""
    if logits.device.type != "cpu":
        raise BitmaskValidationError(LOGITS_CPU_ERROR)
    if bitmask.device.type != "cpu":
        raise BitmaskValidationError(BITMASK_CPU_ERROR)
    if logits.dtype != torch.float32:
        raise BitmaskValidationError(LOGITS_TYPE_ERROR)
    if bitmask.dtype != torch.int32:
        raise BitmaskValidationError(BITMASK_TYPE_ERROR)
    if logits.dim() != 1 and logits.dim() != MAX_DIMS:
        raise BitmaskValidationError(
            BitmaskValidationError.dims_error("logits", logits.dim())
        )
    if bitmask.dim() != 1 and bitmask.dim() != MAX_DIMS:
        raise BitmaskValidationError(
            BitmaskValidationError.dims_error("bitmask", bitmask.dim())
        )

    logits_shape = (
        (1, logits.shape[0])
        if logits.dim() == 1
        else (logits.shape[0], logits.shape[1])
    )
    bitmask_shape = (
        (1, bitmask.shape[0])
        if bitmask.dim() == 1
        else (bitmask.shape[0], bitmask.shape[1])
    )

    _core.kernels.apply_token_bitmask_inplace_cpu(
        logits.data_ptr(),
        logits_shape,
        bitmask.data_ptr(),
        bitmask_shape,
        indices,
    )
