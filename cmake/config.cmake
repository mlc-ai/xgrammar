set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(XGRAMMAR_BUILD_PYTHON_BINDINGS ON)
set(XGRAMMAR_BUILD_CXX_TESTS OFF)
# set it to your own architecture
set(XGRAMMAR_CUDA_ARCHITECTURES
    native
    CACHE STRING "CUDA architectures"
)
