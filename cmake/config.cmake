# Only a default: an unconditional set() here would shadow the CMAKE_BUILD_TYPE cache variable and
# silently override -DCMAKE_BUILD_TYPE=... (e.g. the build type scikit-build-core passes for wheel
# builds).
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
set(XGRAMMAR_BUILD_PYTHON_BINDINGS ON)
set(XGRAMMAR_ENABLE_COVERAGE OFF)
set(XGRAMMAR_BUILD_CXX_TESTS OFF)
set(XGRAMMAR_ENABLE_CPPTRACE OFF)
set(XGRAMMAR_ENABLE_INTERNAL_CHECK OFF)
