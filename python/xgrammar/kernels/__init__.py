"""The kernels for XGrammar."""

import torch

from .apply_token_bitmask_inplace_cpu import apply_token_bitmask_inplace_cpu

apply_token_bitmask_inplace = {
    "cpu": apply_token_bitmask_inplace_cpu,
}

__all__ = [
    "apply_token_bitmask_inplace",
]

if torch.cuda.is_available():
    from .apply_token_bitmask_inplace_cuda import apply_token_bitmask_inplace_cuda
    apply_token_bitmask_inplace["cuda"] = apply_token_bitmask_inplace_cuda

try:
    from .apply_token_bitmask_inplace_triton import apply_token_bitmask_inplace_triton  # isort:skip
    apply_token_bitmask_inplace["triton"] = apply_token_bitmask_inplace_triton
except ImportError:
    # If triton is not installed, we can still use the CPU and CUDA implementations.
    pass
