"""The kernels for XGrammar."""

import torch

from .apply_token_bitmask_inplace_cpu import apply_token_bitmask_inplace_cpu

apply_token_bitmask_inplace_kernels = {"cpu": apply_token_bitmask_inplace_cpu}

__all__ = ["apply_token_bitmask_inplace_kernels"]

try:
    if torch.cuda.is_available():
        from .apply_token_bitmask_inplace_cuda import apply_token_bitmask_inplace_cuda

        apply_token_bitmask_inplace_kernels["cuda"] = apply_token_bitmask_inplace_cuda
except ImportError:
    # If we can't find nvcc, then don't register the CUDA kernel.
    pass
except RuntimeError:
    # If we are unable to compile the CUDA kernel, then don't register the CUDA kernel.
    pass

try:
    from .apply_token_bitmask_inplace_triton import (  # isort: skip
        apply_token_bitmask_inplace_triton,
    )

    apply_token_bitmask_inplace_kernels["triton"] = apply_token_bitmask_inplace_triton
except ImportError:
    # If triton is not installed, we can still use the CPU and CUDA implementations.
    pass

try:
    from .apply_token_bitmask_inplace_metal import apply_token_bitmask_inplace_metal  # isort: skip
    # Note: The MLX Metal implementation has reversed parameter order (bitmask, logits)
    # and works with MLX arrays directly instead of PyTorch tensors.
    # Both logits and indices parameters are expected to be MLX arrays.
    apply_token_bitmask_inplace_kernels["metal"] = apply_token_bitmask_inplace_metal
except ImportError:
    # If MLX is not installed, we can still use the CPU, CUDA, and Triton implementations.
    pass
