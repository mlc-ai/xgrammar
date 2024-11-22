"""The CUDA kernel source code for in-place applying token mask."""

import ctypes
import logging
import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from xgrammar.support import logging

logging.enable_logging()
logger = logging.getLogger(__name__)


try:
    from cuda import cuda, cudart, nvrtc
except ImportError:
    cuda = None
    cudart = None
    nvrtc = None


BITS_PER_BLOCK = 32
THREADS_PER_BLOCK = 1024
ELEMENTS_PER_THREAD = 4

_apply_token_bitmask_inplace_kernel = """
#include <cuda_fp16.h>

#include <cuda/std/limits>

#define BITS_PER_BLOCK 32
#define THREADS_PER_BLOCK 1024
#define ELEMENTS_PER_THREAD 4
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define GET_BIT(data_ptr, bit_idx) \
  ((data_ptr[(bit_idx) / BITS_PER_BLOCK] >> ((bit_idx) % BITS_PER_BLOCK)) & 1)

extern "C" __global__ void __launch_bounds__(1024) ApplyTokenBitmaskInplaceKernel(
    float* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int vocab_size,
    int bitmask_size
) {
  int bid = indices[blockIdx.y];
  int tid = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

  float* logits_ptr = logits + bid * vocab_size + tid;

  for (int i = 0; i < ELEMENTS_PER_THREAD && tid + i < vocab_size; ++i) {
    // logits[bid, tid + i] = mask(..., bitmask[by, tid + i])
    if (GET_BIT(reinterpret_cast<const int32_t*>(bitmask + blockIdx.y * bitmask_size), tid + i) ==
        0) {
      logits_ptr[i] = -cuda::std::numeric_limits<float>::infinity();
    }
  }
}
""".strip()


def _find_nvcc_path() -> Optional[str]:
    return shutil.which("nvcc")


# Adapted from https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples.
def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


# Adapted from https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples.
def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(
            "CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0]))
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


# Adapted from https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/examples.
class KernelStore:
    _module = None
    _func = None

    @classmethod
    def compile(cls, device_id: int):
        if cls._func is not None:
            return cls._func

        prog = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(
                str.encode(_apply_token_bitmask_inplace_kernel), b"sourceCode.cu", 0, [], []
            )
        )
        # Get CUDA home from environment variable.
        CUDA_HOME = os.getenv("CUDA_HOME")
        if not CUDA_HOME:
            CUDA_HOME = os.getenv("CUDA_PATH")

        if not CUDA_HOME:
            # Check common installation paths
            candidate_paths = []
            nvcc_path = _find_nvcc_path()
            if nvcc_path != None:
                candidate_paths.append(str(Path(nvcc_path).parent.parent))
            if platform.system() != "Windows":
                candidate_paths += [
                    "/usr/local/cuda",
                    "/opt/cuda",
                ]
            else:
                candidate_paths += ["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"]
            for path in candidate_paths:
                if os.path.exists(path):
                    CUDA_HOME = path
                    break

        if not CUDA_HOME:
            raise RuntimeError(
                f"Cannot find CUDA home from the candidate paths: {candidate_paths}. "
                'You can specify the CUDA home via environment variable "$CUDA_HOME" after '
                "installing CUDA."
            )
        logger.info("Use CUDA home %s", CUDA_HOME)
        include_dirs = os.path.join(CUDA_HOME, "include")

        # Initialize CUDA
        checkCudaErrors(cudart.cudaFree(0))

        major = checkCudaErrors(
            cudart.cudaDeviceGetAttribute(
                cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device_id
            )
        )
        minor = checkCudaErrors(
            cudart.cudaDeviceGetAttribute(
                cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device_id
            )
        )
        prefix = "compute"
        arch_arg = bytes(f"--gpu-architecture={prefix}_{major}{minor}", "ascii")

        try:
            opts = [
                b"--fmad=true",
                arch_arg,
                "--include-path={}".format(include_dirs).encode("UTF-8"),
                b"--std=c++11",
                b"-default-device",
            ]
            checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except RuntimeError as err:
            logSize = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * logSize
            checkCudaErrors(nvrtc.nvrtcGetProgramLog(prog, log))
            # NOTE: The prints below are intended to provide the kernel compilation error.
            print(log.decode())
            print(err)
            raise RuntimeError("CUDA kernel compilation failure")

        dataSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
        data = b" " * dataSize
        checkCudaErrors(nvrtc.nvrtcGetPTX(prog, data))

        # Store into `_module` and `_func`.
        module = checkCudaErrors(cuda.cuModuleLoadData(np.char.array(data)))
        func = checkCudaErrors(cuda.cuModuleGetFunction(module, b"ApplyTokenBitmaskInplaceKernel"))
        cls._module = module
        cls._func = func
        # Return the compiled kernel.
        return func


def apply_token_bitmask_inplace_cuda(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
):
    time_start = time.monotonic_ns()
    if cuda is None or cudart is None or nvrtc is None:
        raise RuntimeError(
            "CUDA dependencies are not installed. Please follow "
            "https://xgrammar.mlc.ai/docs/start/install#cuda-dependency to install CUDA dependency."
        )

    # Check input tensor shapes.
    if logits.ndim == 2:
        batch_size, vocab_size = logits.shape
    elif logits.ndim == 1:
        batch_size = 1
        (vocab_size,) = logits.shape
    else:
        raise ValueError(f"Invalid logits tensor shape {logits.shape}")

    # Check input tensor dtypes.
    if logits.dtype != torch.float32:
        raise ValueError(
            "The logits tensor is expected to have dtype torch.float32. "
            f"However the input logits has dtype {logits.dtype}"
        )
    if bitmask.dtype != torch.int32:
        raise ValueError(
            "The bitmask tensor is expected to have dtype torch.int32. "
            f"However the input bitmask has dtype {logits.dtype}"
        )

    # Ensure that the tensors are contiguous in memory.
    if not logits.is_contiguous() or not bitmask.is_contiguous():
        raise ValueError("The logits and bitmask tensors are expected to be contiguous in memory.")

    if indices is None:
        indices = torch.arange(batch_size, dtype=torch.int32, device=logits.device)
    elif isinstance(indices, list):
        indices = torch.tensor(indices, dtype=torch.int32, device=logits.device)

    # Compile the kernel.
    kernel = KernelStore.compile(logits.device.index)

    # Setup kernel launching arguments.
    bitmask_size = math.ceil(vocab_size / BITS_PER_BLOCK)
    grid_dims = (
        math.ceil(vocab_size / (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD)),
        indices.shape[0],
        1,
    )
    block_dims = THREADS_PER_BLOCK, 1, 1
    shared_mem_bytes = 0
    stream = torch.cuda.current_stream().cuda_stream
    extra = 0
    kernelArgs = (
        (
            logits.data_ptr(),
            bitmask.data_ptr(),
            indices.data_ptr(),
            vocab_size,
            bitmask_size,
        ),
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
        ),
    )
    time_end = time.monotonic_ns()
    print(f"Time to compile kernel: {(time_end - time_start) / 1e3} us")

    # Launch the kernel.
    checkCudaErrors(
        cuda.cuLaunchKernel(
            kernel,
            *grid_dims,
            *block_dims,
            shared_mem_bytes,
            stream,
            kernelArgs,
            extra,
        )
    )
