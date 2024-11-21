#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <vector>

#include "../support/logging.h"
#include "kernels.h"

#define XGRAMMAR_CUDA_CALL(...)                                                                    \
  do {                                                                                             \
    __VA_ARGS__;                                                                                   \
    cudaError_t err = cudaGetLastError();                                                          \
    XGRAMMAR_CHECK(err == cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err) << " (" << err \
                                       << ") " << __FILE__ << ": line " << __LINE__ << std::endl;  \
  } while (0)

#define XGRAMMAR_DISPATCH_DTYPE(dtype_flag, c_type, ...)                                         \
  do {                                                                                           \
    switch (dtype_flag) {                                                                        \
      case DTypeFlag::DTYPE_FLOAT16: {                                                           \
        using c_type = half;                                                                     \
        __VA_ARGS__;                                                                             \
        break;                                                                                   \
      }                                                                                          \
      case DTypeFlag::DTYPE_FLOAT32: {                                                           \
        using c_type = float;                                                                    \
        __VA_ARGS__;                                                                             \
        break;                                                                                   \
      }                                                                                          \
      case DTypeFlag::DTYPE_FLOAT64: {                                                           \
        using c_type = double;                                                                   \
        __VA_ARGS__;                                                                             \
        break;                                                                                   \
      }                                                                                          \
      default:                                                                                   \
        std::ostringstream oss;                                                                  \
        oss << #__VA_ARGS__ << " failed to dispatch data type " << static_cast<int>(dtype_flag); \
        XGRAMMAR_LOG(FATAL) << oss.str();                                                        \
        break;                                                                                   \
    }                                                                                            \
  } while (0)

namespace xgrammar {

#define BITS_PER_BLOCK 32
#define THREADS_PER_BLOCK 1024
#define ELEMENTS_PER_THREAD 4
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define GET_BIT(data_ptr, bit_idx) \
  ((data_ptr[(bit_idx) / BITS_PER_BLOCK] >> ((bit_idx) % BITS_PER_BLOCK)) & 1)

template <typename T>
__device__ T GetNegativeInfinity() {
  return -cuda::std::numeric_limits<T>::infinity();
}

template <>
__device__ half GetNegativeInfinity<half>() {
  return __float2half(-INFINITY);
}

__global__ void __launch_bounds__(1024) ApplyTokenBitmaskInplaceKernel(
    float* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int vocab_size,
    int bitmask_size
) {
  int bid = indices[blockIdx.y];
  // printf("accessing by: %d, bid: %d\n", blockIdx.y, bid);
  int tid = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

  float* logits_ptr = logits + bid * vocab_size + tid;

  for (int i = 0; i < ELEMENTS_PER_THREAD && tid + i < vocab_size; ++i) {
    // logits[bid, tid + i] = mask(..., bitmask[by, tid + i])
    if (GET_BIT(reinterpret_cast<const int32_t*>(bitmask + blockIdx.y * bitmask_size), tid + i) ==
        0) {
      logits_ptr[i] = GetNegativeInfinity<float>();
    }
  }
}

void ApplyTokenBitmaskInplace(
    float* logits,
    int32_t* bitmask,
    int batch_size,
    int vocab_size,
    std::optional<std::vector<int>> indices
) {
  if (indices) {
    for (int i = 0; i < indices->size(); ++i) {
      XGRAMMAR_CHECK(indices->at(i) < batch_size)
          << "index " << indices->at(i) << " is out of bounds";
    }
  } else {
    indices = std::vector<int>(batch_size);
    std::iota(indices->begin(), indices->end(), 0);
  }

  dim3 num_blocks(CEIL_DIV(vocab_size, THREADS_PER_BLOCK * ELEMENTS_PER_THREAD), indices->size());
  int num_threads = THREADS_PER_BLOCK;

  int* device_indices;
  cudaMalloc(&device_indices, indices->size() * sizeof(int));
  cudaMemcpy(
      device_indices, indices->data(), indices->size() * sizeof(int), cudaMemcpyHostToDevice
  );

  XGRAMMAR_CUDA_CALL(ApplyTokenBitmaskInplaceKernel<<<num_blocks, num_threads>>>(
      logits, bitmask, device_indices, vocab_size, CEIL_DIV(vocab_size, BITS_PER_BLOCK)
  ));
}

}  // namespace xgrammar
