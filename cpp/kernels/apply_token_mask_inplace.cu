#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/limits>
#include <sstream>

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
#define GET_BIT(data_ptr, bit_idx) \
  ((data_ptr[bit_idx / BITS_PER_BLOCK] >> (bit_idx % BITS_PER_BLOCK)) & 1)

template <typename T>
__device__ T GetNegativeInfinity() {
  return -cuda::std::numeric_limits<T>::infinity();
}

template <>
__device__ half GetNegativeInfinity<half>() {
  return __float2half(-INFINITY);
}

template <typename T>
__global__ void __launch_bounds__(512) ApplyTokenBitmaskInplaceKernel(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    int vocab_size,
    int bitmask_size,
    int bitmask_row_size
) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= bitmask_size) {
    return;
  }

  int batch_id = gid / bitmask_row_size;
  int bitmask_id = gid % bitmask_row_size;
  int bitmask_val = bitmask[gid];
  T* logits_ptr = logits + batch_id * vocab_size + bitmask_id * BITS_PER_BLOCK;
  for (int i = 0; i < BITS_PER_BLOCK; ++i) {
    if (bitmask_id * BITS_PER_BLOCK + i >= vocab_size) {
      break;
    }
    if ((bitmask_val & 1) == 0) {
      logits_ptr[i] = GetNegativeInfinity<T>();
    }
    bitmask_val >>= 1;
  }
}

#define THREADS_PER_BLOCK 512

void ApplyTokenBitmaskInplace(
    void* logits, DTypeFlag dtype_flag, int32_t* bitmask, int batch_size, int vocab_size
) {
  int bitmask_size = (vocab_size + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
  int num_blocks = (batch_size * bitmask_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int num_threads = THREADS_PER_BLOCK;

  XGRAMMAR_DISPATCH_DTYPE(dtype_flag, c_type, {
    XGRAMMAR_CUDA_CALL({
      ApplyTokenBitmaskInplaceKernel<<<num_blocks, num_threads>>>(
          reinterpret_cast<c_type*>(logits),
          bitmask,
          vocab_size,
          batch_size * bitmask_size,
          bitmask_size
      );
    });
  });
}

}  // namespace xgrammar
