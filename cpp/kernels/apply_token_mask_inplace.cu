#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/std/limits>
#include <sstream>

#include "kernels.h"
#include "support/logging.h"

// #ifndef NDEBUG
#define XGRAMMAR_CUDA_CALL(...)                                                                    \
  do {                                                                                             \
    __VA_ARGS__;                                                                                   \
    cudaError_t err = cudaGetLastError();                                                          \
    XGRAMMAR_CHECK(err != cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err) << " (" << err \
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
__global__ void __launch_bounds__(1024) apply_token_bitmask_inplace_kernel(
    int* __restrict__ bitmask, T* __restrict__ logits, int batch_size, int vocab_size
) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int size = batch_size * vocab_size;
  int bitmask_size = (vocab_size + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
  for (int i = gid; i < size; i += stride) {
    int batch_id = i / vocab_size;
    int vocab_id = i % vocab_size;
    const int* bitmask_row = bitmask + batch_id * bitmask_size;
    int bit = GET_BIT(bitmask_row, vocab_id);
    logits[gid] = bit ? logits[gid] : -cuda::std::numeric_limits<T>::infinity();
  }
}

#define STRIDE_SIZE 1048576
#define THREADS_PER_BLOCK 1024

void apply_token_bitmask_inplace(
    int* bitmask, void* logits, DTypeFlag dtype_flag, int batch_size, int vocab_size
) {
  int num_blocks = (batch_size * vocab_size + STRIDE_SIZE - 1) / STRIDE_SIZE;
  int num_threads = THREADS_PER_BLOCK;

  XGRAMMAR_DISPATCH_DTYPE(dtype_flag, c_type, {
    XGRAMMAR_CUDA_CALL({
      apply_token_bitmask_inplace_kernel<<<num_blocks, num_threads>>>(
          bitmask, reinterpret_cast<c_type*>(logits), batch_size, vocab_size
      );
    });
  });
}

}  // namespace xgrammar

// PYBIND11_MODULE(ndarray_backend_cuda, m) {
//   namespace py = pybind11;
//   using namespace cuda;

//   m.attr("__device_name__") = "cuda";
//   m.attr("__tile_size__") = TILE;

//   py::class_<CudaArray>(m, "Array")
//       .def(py::init<size_t>(), py::return_value_policy::take_ownership)
//       .def_readonly("size", &CudaArray::size)
//       .def("ptr", &CudaArray::ptr_as_int);

//   // return numpy array, copying from CPU
//   m.def(
//       "to_numpy",
//       [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides, size_t
//       offset ) {
//         std::vector<size_t> numpy_strides = strides;
//         std::transform(
//             numpy_strides.begin(),
//             numpy_strides.end(),
//             numpy_strides.begin(),
//             [](size_t& c) { return c * ELEM_SIZE; }
//         );

//         // copy memory to host
//         scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
//         if (host_ptr == 0) throw std::bad_alloc();
//         cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE,
//         cudaMemcpyDeviceToHost); if (err != cudaSuccess) throw
//         std::runtime_error(cudaGetErrorString(err));

//         // return numpy array
//         py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
//         return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
//       }
//   );

//   // copy numpy array to GPU
//   m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
//     cudaError_t err =
//         cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
//     if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
//   });

//   m.def("fill", Fill);
//   m.def("compact", Compact);
//   m.def("ewise_setitem", EwiseSetitem);
//   m.def("scalar_setitem", ScalarSetitem);
//   m.def("ewise_add", EwiseAdd);
//   m.def("scalar_add", ScalarAdd);

//   m.def("ewise_mul", EwiseMul);
//   m.def("scalar_mul", ScalarMul);
//   m.def("ewise_div", EwiseDiv);
//   m.def("scalar_div", ScalarDiv);
//   m.def("scalar_power", ScalarPower);

//   m.def("ewise_maximum", EwiseMaximum);
//   m.def("scalar_maximum", ScalarMaximum);
//   m.def("ewise_eq", EwiseEq);
//   m.def("scalar_eq", ScalarEq);
//   m.def("ewise_ge", EwiseGe);
//   m.def("scalar_ge", ScalarGe);

//   m.def("ewise_log", EwiseLog);
//   m.def("ewise_exp", EwiseExp);
//   m.def("ewise_tanh", EwiseTanh);

//   m.def("matmul", Matmul);

//   m.def("reduce_max", ReduceMax);
//   m.def("reduce_sum", ReduceSum);
// }
