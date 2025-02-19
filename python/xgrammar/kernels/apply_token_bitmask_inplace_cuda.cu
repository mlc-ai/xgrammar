/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// clang-format off
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
// clang-format on

#ifndef CUDART_INF_FP16
#define CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif

#ifndef CUDART_INF_BF16
#define CUDART_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif

constexpr int32_t kBitsPerMaskElement = 32;
constexpr int32_t kThreadsPerBlock = 256;

template <typename T>
__device__ T NegativeInfinity() {
  return -INFINITY;
}

template <>
__device__ __half NegativeInfinity<__half>() {
  return -CUDART_INF_FP16;
}

template <>
__device__ __nv_bfloat16 NegativeInfinity<__nv_bfloat16>() {
  return -CUDART_INF_BF16;
}

template <typename T, typename PackedT>
__device__ PackedT PackedNegativeInfinity() {
  constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
  T packed[kAlignment];
#pragma unroll
  for (int i = 0; i < kAlignment; i++) {
    packed[i] = NegativeInfinity<T>();
  }
  return *reinterpret_cast<PackedT*>(packed);
}

template <typename T, typename PackedT, int32_t kBitsPerThread>
__global__ void __launch_bounds__(kThreadsPerBlock) LogitsBitmaskKernel(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int32_t vocab_size,
    int32_t bitmask_size
) {
  constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
  constexpr uint32_t kPackedMask = (1 << kAlignment) - 1;

  const int batch_idx = (indices == nullptr) ? blockIdx.y : indices[blockIdx.y];

  const int block_offset = blockIdx.x * kThreadsPerBlock * kBitsPerThread;
  T* logits_gmem_ptr = logits + batch_idx * vocab_size + block_offset;
  const int32_t* bitmask_gmem_ptr =
      bitmask + batch_idx * bitmask_size + block_offset / kBitsPerMaskElement;
  const int bitmask_inner_idx = threadIdx.x % (kBitsPerMaskElement / kAlignment);
  T logits_reg[kAlignment];

#pragma unroll
  for (int offset = threadIdx.x * kAlignment; offset < kThreadsPerBlock * kBitsPerThread;
       offset += kThreadsPerBlock * kAlignment) {
    if (block_offset + offset >= vocab_size) {
      break;
    }

    const uint32_t bitmask_val =
        (~bitmask_gmem_ptr[offset / kBitsPerMaskElement] >> (bitmask_inner_idx * kAlignment)) &
        kPackedMask;

    if (bitmask_val == 0) {
      continue;
    }

    if (bitmask_val == kPackedMask) {
      *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset) = PackedNegativeInfinity<T, PackedT>();
      continue;
    }

    *reinterpret_cast<PackedT*>(logits_reg) = *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset);
#pragma unroll
    for (int i = 0; i < kAlignment; i++) {
      if (((bitmask_val >> i) & 1)) {
        logits_reg[i] = NegativeInfinity<T>();
      }
    }
    *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset) = *reinterpret_cast<PackedT*>(logits_reg);
  }
}

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
constexpr auto CeilDiv(T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename T, typename PackedT>
void ApplyTokenBitmaskInplaceDispatchToBitsPerThread(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int32_t vocab_size,
    int32_t bitmask_size,
    int32_t batch_size
) {
  constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
  const int32_t num_blocks_per_row = CeilDiv(2048 / kThreadsPerBlock * 128, batch_size);
  const int32_t num_bits_per_thread = CeilDiv(vocab_size, kThreadsPerBlock * num_blocks_per_row);

  const dim3 block(kThreadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (num_bits_per_thread <= 4 && kAlignment <= 4) {
    const dim3 grid(CeilDiv(vocab_size, kThreadsPerBlock * 4), batch_size);
    LogitsBitmaskKernel<T, PackedT, 4>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocab_size, bitmask_size);
  } else if (num_bits_per_thread <= 8 && kAlignment <= 8) {
    const dim3 grid(CeilDiv(vocab_size, kThreadsPerBlock * 8), batch_size);
    LogitsBitmaskKernel<T, PackedT, 8>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocab_size, bitmask_size);
  } else if (num_bits_per_thread <= 16 && kAlignment <= 16) {
    const dim3 grid(CeilDiv(vocab_size, kThreadsPerBlock * 16), batch_size);
    LogitsBitmaskKernel<T, PackedT, 16>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocab_size, bitmask_size);
  } else {
    const dim3 grid(CeilDiv(vocab_size, kThreadsPerBlock * 32), batch_size);
    LogitsBitmaskKernel<T, PackedT, 32>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocab_size, bitmask_size);
  }
}

template <typename T>
void ApplyTokenBitmaskInplaceDispatchToPackedT(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int32_t vocab_size,
    int32_t bitmask_size,
    int32_t batch_size
) {
  if (vocab_size % (sizeof(float4) / sizeof(T)) == 0) {
    ApplyTokenBitmaskInplaceDispatchToBitsPerThread<T, float4>(
        logits, bitmask, indices, vocab_size, bitmask_size, batch_size
    );
  } else {
    ApplyTokenBitmaskInplaceDispatchToBitsPerThread<T, T>(
        logits, bitmask, indices, vocab_size, bitmask_size, batch_size
    );
  }
}

void ApplyTokenBitmaskInplace(
    at::Tensor logits, at::Tensor bitmask, at::optional<at::Tensor> indices = at::nullopt
) {
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor.");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous.");
  TORCH_CHECK(logits.dim() == 1 || logits.dim() == 2, "logits must be a 1D or 2D tensor.");
  int32_t batch_size = 1;
  int32_t vocab_size = logits.size(0);
  if (logits.dim() == 2) {
    batch_size = logits.size(0);
    vocab_size = logits.size(1);
  }

  TORCH_CHECK(bitmask.is_cuda(), "bitmask must be a CUDA tensor.");
  TORCH_CHECK(bitmask.is_contiguous(), "bitmask must be contiguous.");
  TORCH_CHECK(bitmask.dim() == 1 || bitmask.dim() == 2, "bitmask must be a 1D or 2D tensor.");
  int32_t bitmask_batch_size = 1;
  int32_t bitmask_size = bitmask.size(0);
  if (bitmask.dim() == 2) {
    bitmask_batch_size = bitmask.size(0);
    bitmask_size = bitmask.size(1);
  }
  TORCH_CHECK(bitmask_batch_size == batch_size, "bitmask must have the batch size same to logits.");
  TORCH_CHECK(
      bitmask_size == CeilDiv(vocab_size, kBitsPerMaskElement),
      "bitmask must have the hidden size equal to CeilDiv(vocab_size, 32), but got vocab_size=",
      vocab_size,
      " and bitmask_size=",
      bitmask_size
  );

  int32_t* indices_ptr = nullptr;
  if (indices) {
    batch_size = indices->size(0);
    indices_ptr = indices->data_ptr<int32_t>();
  }

  switch (logits.scalar_type()) {
    case torch::kFloat32: {
      ApplyTokenBitmaskInplaceDispatchToPackedT(
          logits.data_ptr<float>(),
          bitmask.data_ptr<int32_t>(),
          indices_ptr,
          vocab_size,
          bitmask_size,
          batch_size
      );
      break;
    }
    case torch::kFloat16: {
      ApplyTokenBitmaskInplaceDispatchToPackedT(
          reinterpret_cast<__half*>(logits.data_ptr<torch::Half>()),
          bitmask.data_ptr<int32_t>(),
          indices_ptr,
          vocab_size,
          bitmask_size,
          batch_size
      );
      break;
    }
    case torch::kBFloat16: {
      ApplyTokenBitmaskInplaceDispatchToPackedT(
          reinterpret_cast<__nv_bfloat16*>(logits.data_ptr<torch::BFloat16>()),
          bitmask.data_ptr<int32_t>(),
          indices_ptr,
          vocab_size,
          bitmask_size,
          batch_size
      );
      break;
    }
    default:
      TORCH_CHECK(false, "logits dtype must be float, half or bfloat16.");
      break;
  }
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def(
      "apply_token_bitmask_inplace_cuda(Tensor logits, Tensor bitmask, Tensor? indices=None) -> ()"
  );
}

TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("apply_token_bitmask_inplace_cuda", &ApplyTokenBitmaskInplace);
}
