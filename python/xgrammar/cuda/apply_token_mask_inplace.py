# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The CUDA kernel source code for in-place applying token mask."""
import ctypes
import os

import numpy as np
import torch

try:
    from cuda import cuda, cudart, nvrtc
except ImportError:
    cuda = None
    cudart = None
    nvrtc = None


BITS_PER_BLOCK = 32
THREADS_PER_BLOCK = 512

_apply_token_bitmask_inplace_kernel = """
#include <cuda_fp16.h>

#include <cuda/std/limits>

#define BITS_PER_BLOCK 32

extern "C" __global__ void __launch_bounds__(512) ApplyTokenBitmaskInplaceKernel(
    float* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    int32_t vocab_size,
    int64_t bitmask_size,
    int32_t bitmask_row_size
) {
  int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= bitmask_size) {
    return;
  }

  int32_t batch_id = gid / bitmask_row_size;
  int32_t bitmask_id = gid % bitmask_row_size;
  int32_t bitmask_val = bitmask[gid];
  float* logits_ptr = logits + batch_id * vocab_size + bitmask_id * BITS_PER_BLOCK;
  for (int i = 0; i < BITS_PER_BLOCK; ++i) {
    if (bitmask_id * BITS_PER_BLOCK + i >= vocab_size) {
      break;
    }
    if ((bitmask_val & 1) == 0) {
      logits_ptr[i] = -cuda::std::numeric_limits<float>::infinity();
    }
    bitmask_val >>= 1;
  }
}
""".strip()


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
        CUDA_HOME = os.getenv("CUDA_HOME")
        if CUDA_HOME == None:
            CUDA_HOME = os.getenv("CUDA_PATH")
        if CUDA_HOME == None:
            raise RuntimeError("Environment variable CUDA_HOME or CUDA_PATH is not set")
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


def apply_token_bitmask_inplace(logits: torch.Tensor, bitmask: torch.Tensor):
    if cuda is None or cudart is None or nvrtc is None:
        raise RuntimeError("cuda-python is not installed. Please install cuda-python first.")

    # Check input tensor shapes.
    if logits.ndim == 2:
        batch_size, vocab_size = logits.shape
    elif logits.ndim == 1:
        batch_size = 1
        (vocab_size,) = logits.shape
    else:
        raise ValueError(f"Invalid logits tensor shape {logits.shape}")
    bitmask_size = (vocab_size + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK

    # Ensure that the tensors are contiguous in memory.
    logits = logits.contiguous()
    bitmask = bitmask.contiguous()

    # Compile the kernel.
    kernel = KernelStore.compile(logits.device.index)
    # Setup kernel launching arguments.
    grid_dims = (batch_size * bitmask_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, 1, 1
    block_dims = THREADS_PER_BLOCK, 1, 1
    shared_mem_bytes = 0
    stream = cuda.CU_STREAM_LEGACY
    extra = 0
    kernelArgs = (
        (
            logits.data_ptr(),
            bitmask.data_ptr(),
            vocab_size,
            batch_size * bitmask_size,
            bitmask_size,
        ),
        (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_int32,
        ),
    )
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
