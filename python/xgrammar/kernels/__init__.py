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
    import mlx.core as mx

    from .apply_token_bitmask_mlx import apply_token_bitmask_mlx

    # Note: MLX arrays, like JAX arrays, are immutable.  Therefore, in-place operations are not
    #       allowed. Here we reuse the variable apply_token_bitmask_inplace_kernels.
    apply_token_bitmask_inplace_kernels["metal"] = apply_token_bitmask_mlx
except ImportError:
    # If MLX is not installed, we don't register the MLX implementation
    pass
