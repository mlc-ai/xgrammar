enable_language(CUDA)

set(XGRAMMAR_CUDA_ARCHITECTURES native)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER STREQUAL "NOTFOUND")
  message(
    FATAL_ERROR "CUDA compiler not found but XGRAMMAR_BUILD_CUDA_KERNELS is ON. Please install "
                "CUDA or set XGRAMMAR_BUILD_CUDA_KERNELS to OFF."
  )
endif()

message(STATUS "Found cuda, building XGrammar kernels")

if(DEFINED XGRAMMAR_CUDA_ARCHITECTURES)
  message(STATUS "CUDA architectures are set to ${XGRAMMAR_CUDA_ARCHITECTURES}.")
  set(CMAKE_CUDA_ARCHITECTURES ${XGRAMMAR_CUDA_ARCHITECTURES})
else()
  message(STATUS "CUDA architectures are set to default value ${CMAKE_CUDA_ARCHITECTURES}.")
endif()

file(GLOB_RECURSE XGRAMMAR_KERNELS_PATH ${PROJECT_SOURCE_DIR}/cpp/kernels/*.cu)
add_library(xgrammar_kernels STATIC ${XGRAMMAR_KERNELS_PATH})
target_include_directories(xgrammar_kernels PRIVATE ${PROJECT_SOURCE_DIR}/cpp)

# Expose CMAKE_CUDA_ARCHITECTURES to parent scope
set(CMAKE_CUDA_ARCHITECTURES
    ${CMAKE_CUDA_ARCHITECTURES}
    PARENT_SCOPE
)
