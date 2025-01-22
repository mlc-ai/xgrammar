#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

int32_t constexpr kBitsPerMaskElement = 32;
int32_t constexpr kThreadsPerBlock = 256;

template <typename T>
__device__ T GetNegativeInfinity() {
  return -INFINITY;
}

template <>
__device__ half GetNegativeInfinity<half>() {
  return __float2half(-INFINITY);
}

template <>
__device__ __nv_bfloat16 GetNegativeInfinity<__nv_bfloat16>() {
  return __float2bfloat16(-INFINITY);
}

template <typename T, typename PackedT>
__global__ void __launch_bounds__(kThreadsPerBlock) logitsBitmaskKernel(
    T* __restrict__ logits,
    int32_t const* __restrict__ bitmask,
    int32_t const* __restrict__ indices,
    int32_t vocabSize,
    int32_t bitmaskSize
) {
  int constexpr kAlignment = sizeof(PackedT) / sizeof(T);
  int const batchIdx = (indices == nullptr) ? blockIdx.y : indices[blockIdx.y];

  int const logitsGmemOffset = kThreadsPerBlock * blockIdx.x * kBitsPerMaskElement;
  T* logitsGmemPtr = logits + batchIdx * vocabSize + logitsGmemOffset;
  __shared__ T logitsSmem[kThreadsPerBlock * kBitsPerMaskElement];

#pragma unroll
  for (int offset = 0; offset < kThreadsPerBlock * kBitsPerMaskElement;
       offset += kThreadsPerBlock * kAlignment) {
    int localOffset = offset + threadIdx.x * kAlignment;
    if (logitsGmemOffset + localOffset >= vocabSize) {
      break;
    }
    *reinterpret_cast<PackedT*>(logitsSmem + localOffset) =
        *reinterpret_cast<PackedT*>(logitsGmemPtr + localOffset);
  }
  __syncthreads();

  int const bitmaskIdx = kThreadsPerBlock * blockIdx.x + threadIdx.x;
  int32_t const bitmaskVal = bitmask[batchIdx * bitmaskSize + bitmaskIdx];

#pragma unroll
  for (int i = 0; i < kBitsPerMaskElement; ++i) {
    int offset = (i + threadIdx.x) % warpSize;
    if (bitmaskIdx * kBitsPerMaskElement + offset >= vocabSize) {
      continue;
    }
    if (!((bitmaskVal >> offset) & 1)) {
      logitsSmem[threadIdx.x * kBitsPerMaskElement + offset] = GetNegativeInfinity<T>();
    }
  }
  __syncthreads();

#pragma unroll
  for (int offset = 0; offset < kThreadsPerBlock * kBitsPerMaskElement;
       offset += kThreadsPerBlock * kAlignment) {
    int localOffset = offset + threadIdx.x * kAlignment;
    if (logitsGmemOffset + localOffset >= vocabSize) {
      break;
    }
    *reinterpret_cast<PackedT*>(logitsGmemPtr + localOffset) =
        *reinterpret_cast<PackedT*>(logitsSmem + localOffset);
  }
}

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
auto constexpr ceilDiv(T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
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
  dim3 const grid(ceilDiv(bitmaskSize, kThreadsPerBlock), batchSize);
  dim3 const block(kThreadsPerBlock);

  cudaStream_t stream = 0;
  if (vocabSize % (sizeof(float4) / sizeof(T)) == 0) {
    logitsBitmaskKernel<T, float4>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocabSize, bitmaskSize);
  } else if (vocabSize % (sizeof(float2) / sizeof(T)) == 0) {
    logitsBitmaskKernel<T, float2>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocabSize, bitmaskSize);
  } else if (vocabSize % (sizeof(float) / sizeof(T)) == 0) {
    logitsBitmaskKernel<T, float>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocabSize, bitmaskSize);
  } else {
    logitsBitmaskKernel<T, T>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocabSize, bitmaskSize);
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
          reinterpret_cast<half*>(logits.data_ptr<torch::Half>()),
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apply_token_bitmask_inplace", &applyTokenBitmaskInplace, "Apply token bitmask inplace.");
}
