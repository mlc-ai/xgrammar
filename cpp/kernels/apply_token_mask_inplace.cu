#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/limits>
#include <sstream>

#include "../support/logging.h"
#include "kernels.h"

// #ifndef NDEBUG
#define XGRAMMAR_CUDA_CALL(...)                                                                    \
  do {                                                                                             \
    __VA_ARGS__;                                                                                   \
    cudaError_t err = cudaGetLastError();                                                          \
    XGRAMMAR_CHECK(err == cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err) << " (" << err \
                                       << ") " << __FILE__ << ": line " << __LINE__ << std::endl;  \
  } while (0)
/*
      return e;                                                                                \
#else
#define XGRAMMAR_CUDA_CALL(func, ...) \
  {                                   \
    cudaError_t e = (func);           \
    if (e != cudaSuccess) {           \
      return e;                       \
    }                                 \
  }
#endif
*/

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
__global__ void __launch_bounds__(512) apply_token_bitmask_inplace_kernel(
    T* __restrict__ logits, const int32_t* __restrict__ bitmask, int batch_size, int vocab_size
) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int size = batch_size * vocab_size;
  if (gid >= size) {
    return;
  }
  int bitmask_size = (vocab_size + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
  int batch_id = gid / vocab_size;
  int vocab_id = gid % vocab_size;
  const int32_t* bitmask_row = bitmask + batch_id * bitmask_size;
  int bit = GET_BIT(bitmask_row, vocab_id);
  if (!bit) {
    logits[gid] = -cuda::std::numeric_limits<T>::infinity();
  }
}

#define THREADS_PER_BLOCK 512

void apply_token_bitmask_inplace(
    void* logits, DTypeFlag dtype_flag, int32_t* bitmask, int batch_size, int vocab_size
) {
  int num_blocks = (batch_size * vocab_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int num_threads = THREADS_PER_BLOCK;

  XGRAMMAR_DISPATCH_DTYPE(dtype_flag, c_type, {
    XGRAMMAR_CUDA_CALL({
      apply_token_bitmask_inplace_kernel<<<num_blocks, num_threads>>>(
          reinterpret_cast<c_type*>(logits), bitmask, batch_size, vocab_size
      );
    });
  });
}

}  // namespace xgrammar
