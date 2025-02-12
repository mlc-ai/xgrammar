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

int32_t constexpr kBitsPerMaskElement = 32;
int32_t constexpr kThreadsPerBlock = 256;

template <typename T>
__device__ T negativeInfinity() {
  return -INFINITY;
}

template <>
__device__ __half negativeInfinity<__half>() {
  return -CUDART_INF_FP16;
}

template <>
__device__ __nv_bfloat16 negativeInfinity<__nv_bfloat16>() {
  return -CUDART_INF_BF16;
}

template <typename T, typename PackedT>
__device__ PackedT packedNegativeInfinity() {
  int constexpr kAlignment = sizeof(PackedT) / sizeof(T);
  T packed[kAlignment];
#pragma unroll
  for (int i = 0; i < kAlignment; i++) {
    packed[i] = negativeInfinity<T>();
  }
  return *reinterpret_cast<PackedT*>(packed);
}

template <typename T, typename PackedT, int32_t kBitsPerThread>
__global__ void __launch_bounds__(kThreadsPerBlock) logitsBitmaskKernel(
    T* __restrict__ logits,
    int32_t const* __restrict__ bitmask,
    int32_t const* __restrict__ indices,
    int32_t vocabSize,
    int32_t bitmaskSize
) {
  int constexpr kAlignment = sizeof(PackedT) / sizeof(T);
  uint32_t constexpr kPackedMask = (1 << kAlignment) - 1;

  int const batchIdx = (indices == nullptr) ? blockIdx.y : indices[blockIdx.y];

  int const blockOffset = blockIdx.x * kThreadsPerBlock * kBitsPerThread;
  T* logitsGmemPtr = logits + batchIdx * vocabSize + blockOffset;
  int32_t const* bitmaskGmemPtr =
      bitmask + batchIdx * bitmaskSize + blockOffset / kBitsPerMaskElement;
  int const bitmaskInnerIdx = threadIdx.x % (kBitsPerMaskElement / kAlignment);
  T logitsReg[kAlignment];

#pragma unroll
  for (int offset = threadIdx.x * kAlignment; offset < kThreadsPerBlock * kBitsPerThread;
       offset += kThreadsPerBlock * kAlignment) {
    if (blockOffset + offset >= vocabSize) {
      break;
    }

    uint32_t const bitmaskVal =
        (~bitmaskGmemPtr[offset / kBitsPerMaskElement] >> (bitmaskInnerIdx * kAlignment)) &
        kPackedMask;

    if (bitmaskVal == 0) {
      continue;
    }

    if (bitmaskVal == kPackedMask) {
      *reinterpret_cast<PackedT*>(logitsGmemPtr + offset) = packedNegativeInfinity<T, PackedT>();
      continue;
    }

    *reinterpret_cast<PackedT*>(logitsReg) = *reinterpret_cast<PackedT*>(logitsGmemPtr + offset);
#pragma unroll
    for (int i = 0; i < kAlignment; i++) {
      if (((bitmaskVal >> i) & 1)) {
        logitsReg[i] = negativeInfinity<T>();
      }
    }
    *reinterpret_cast<PackedT*>(logitsGmemPtr + offset) = *reinterpret_cast<PackedT*>(logitsReg);
  }
}

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
auto constexpr ceilDiv(T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename T, typename PackedT>
void applyTokenBitmaskInplaceDispatchToBitsPerThread(
    T* __restrict__ logits,
    int32_t const* __restrict__ bitmask,
    int32_t const* __restrict__ indices,
    int32_t vocabSize,
    int32_t bitmaskSize,
    int32_t batchSize
) {
  int constexpr kAlignment = sizeof(PackedT) / sizeof(T);
  int32_t const numBlocksPerRow = ceilDiv(2048 / kThreadsPerBlock * 128, batchSize);
  int32_t const numBitsPerThread = ceilDiv(vocabSize, kThreadsPerBlock * numBlocksPerRow);

  dim3 const block(kThreadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (numBitsPerThread <= 4 && kAlignment <= 4) {
    dim3 const grid(ceilDiv(vocabSize, kThreadsPerBlock * 4), batchSize);
    logitsBitmaskKernel<T, PackedT, 4>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocabSize, bitmaskSize);
  } else if (numBitsPerThread <= 8 && kAlignment <= 8) {
    dim3 const grid(ceilDiv(vocabSize, kThreadsPerBlock * 8), batchSize);
    logitsBitmaskKernel<T, PackedT, 8>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocabSize, bitmaskSize);
  } else if (numBitsPerThread <= 16 && kAlignment <= 16) {
    dim3 const grid(ceilDiv(vocabSize, kThreadsPerBlock * 16), batchSize);
    logitsBitmaskKernel<T, PackedT, 16>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocabSize, bitmaskSize);
  } else {
    dim3 const grid(ceilDiv(vocabSize, kThreadsPerBlock * 32), batchSize);
    logitsBitmaskKernel<T, PackedT, 32>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocabSize, bitmaskSize);
  }
}

template <typename T>
void applyTokenBitmaskInplaceDispatchToPackedT(
    T* __restrict__ logits,
    int32_t const* __restrict__ bitmask,
    int32_t const* __restrict__ indices,
    int32_t vocabSize,
    int32_t bitmaskSize,
    int32_t batchSize
) {
  if (vocabSize % (sizeof(float4) / sizeof(T)) == 0) {
    applyTokenBitmaskInplaceDispatchToBitsPerThread<T, float4>(
        logits, bitmask, indices, vocabSize, bitmaskSize, batchSize
    );
  } else {
    applyTokenBitmaskInplaceDispatchToBitsPerThread<T, T>(
        logits, bitmask, indices, vocabSize, bitmaskSize, batchSize
    );
  }
}

void applyTokenBitmaskInplace(
    at::Tensor logits, at::Tensor bitmask, at::optional<at::Tensor> indices = at::nullopt
) {
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor.");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous.");
  TORCH_CHECK(logits.dim() == 1 || logits.dim() == 2, "logits must be a 1D or 2D tensor.");
  int32_t batchSize = 1;
  int32_t vocabSize = logits.size(0);
  if (logits.dim() == 2) {
    batchSize = logits.size(0);
    vocabSize = logits.size(1);
  }

  TORCH_CHECK(bitmask.is_cuda(), "bitmask must be a CUDA tensor.");
  TORCH_CHECK(bitmask.is_contiguous(), "bitmask must be contiguous.");
  TORCH_CHECK(bitmask.dim() == 1 || bitmask.dim() == 2, "bitmask must be a 1D or 2D tensor.");
  int32_t bitmaskBatchSize = 1;
  int32_t bitmaskSize = bitmask.size(0);
  if (bitmask.dim() == 2) {
    bitmaskBatchSize = bitmask.size(0);
    bitmaskSize = bitmask.size(1);
  }
  TORCH_CHECK(bitmaskBatchSize == batchSize, "bitmask must have the batch size same to logits.");
  TORCH_CHECK(
      bitmaskSize == ceilDiv(vocabSize, kBitsPerMaskElement),
      "bitmask must have the hidden size equal to ceilDiv(vocabSize, 32)."
  );

  int32_t* indices_ptr = nullptr;
  if (indices) {
    batchSize = indices->size(0);
    indices_ptr = indices->data_ptr<int32_t>();
  }

  switch (logits.scalar_type()) {
    case torch::kFloat32: {
      applyTokenBitmaskInplaceDispatchToPackedT(
          logits.data_ptr<float>(),
          bitmask.data_ptr<int32_t>(),
          indices_ptr,
          vocabSize,
          bitmaskSize,
          batchSize
      );
      break;
    }
    case torch::kFloat16: {
      applyTokenBitmaskInplaceDispatchToPackedT(
          reinterpret_cast<__half*>(logits.data_ptr<torch::Half>()),
          bitmask.data_ptr<int32_t>(),
          indices_ptr,
          vocabSize,
          bitmaskSize,
          batchSize
      );
      break;
    }
    case torch::kBFloat16: {
      applyTokenBitmaskInplaceDispatchToPackedT(
          reinterpret_cast<__nv_bfloat16*>(logits.data_ptr<torch::BFloat16>()),
          bitmask.data_ptr<int32_t>(),
          indices_ptr,
          vocabSize,
          bitmaskSize,
          batchSize
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
  m.impl("apply_token_bitmask_inplace_cuda", &applyTokenBitmaskInplace);
}
