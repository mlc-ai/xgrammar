cmake_minimum_required(VERSION 3.18)
project(xgrammar LANGUAGES CXX)

if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  message(STATUS "Config file: ${CMAKE_BINARY_DIR}/config.cmake")
  include(${CMAKE_BINARY_DIR}/config.cmake)
elseif(EXISTS ${PROJECT_SOURCE_DIR}/config.cmake)
  message(STATUS "Config file: ${PROJECT_SOURCE_DIR}/config.cmake")
  include(${PROJECT_SOURCE_DIR}/config.cmake)
elseif(EXISTS ${PROJECT_SOURCE_DIR}/cmake/config.cmake)
  message(STATUS "Config file: ${PROJECT_SOURCE_DIR}/cmake/config.cmake")
  include(${PROJECT_SOURCE_DIR}/cmake/config.cmake)
else()
  message(STATUS "No config.cmake found. Using the default config")
endif()

option(XGRAMMAR_BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(XGRAMMAR_BUILD_CXX_TESTS "Build C++ tests" OFF)
option(XGRAMMAR_ENABLE_CPPTRACE
       "Enable C++ trace (Now only support Linux, and RelWithDebugInfo or Debug build)" OFF
)
option(XGRAMMAR_ENABLE_COVERAGE "Enable code coverage with gcov" OFF)

set(XGRAMMAR_CUDA_ARCHITECTURES
    native
    CACHE STRING "CUDA architectures"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type specified; defaulting to CMAKE_BUILD_TYPE=RelWithDebugInfo.")
  set(CMAKE_BUILD_TYPE
      "RelWithDebugInfo"
      CACHE STRING "The build type" FORCE
  )
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build Python bindings: ${XGRAMMAR_BUILD_PYTHON_BINDINGS}")
message(STATUS "Build C++ tests: ${XGRAMMAR_BUILD_CXX_TESTS}")
message(STATUS "CUDA architectures: ${XGRAMMAR_CUDA_ARCHITECTURES}")
message(STATUS "Enable C++ trace: ${XGRAMMAR_ENABLE_CPPTRACE}")

if(MSVC)
  set(CMAKE_CXX_FLAGS "/Wall ${CMAKE_CXX_FLAGS}")
else()
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
  endif()

  set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wno-pedantic -Wno-unused-parameter \
-Woverloaded-virtual -flto=auto ${CMAKE_CXX_FLAGS}"
  )
endif()

set(XGRAMMAR_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/3rdparty/picojson
                          ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include
)

file(GLOB_RECURSE XGRAMMAR_SOURCES_PATH "${PROJECT_SOURCE_DIR}/cpp/*.cc")
list(FILTER XGRAMMAR_SOURCES_PATH EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/cpp/nanobind/.*\\.cc")

add_library(xgrammar STATIC ${XGRAMMAR_SOURCES_PATH})
target_include_directories(xgrammar PUBLIC include)
target_include_directories(xgrammar SYSTEM PUBLIC ${XGRAMMAR_INCLUDE_PATH})

# link to cpptrace
if(XGRAMMAR_ENABLE_CPPTRACE)
  add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/cpptrace)
  target_link_libraries(xgrammar PUBLIC cpptrace::cpptrace)
  target_compile_definitions(xgrammar PUBLIC XGRAMMAR_ENABLE_CPPTRACE=1)
else()
  target_compile_definitions(xgrammar PUBLIC XGRAMMAR_ENABLE_CPPTRACE=0)
endif()

if(XGRAMMAR_BUILD_PYTHON_BINDINGS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/cpp/nanobind)
  install(TARGETS xgrammar_bindings DESTINATION .)
endif()

if(XGRAMMAR_BUILD_CXX_TESTS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/googletest)

  file(GLOB_RECURSE XGRAMMAR_TEST_SOURCES_PATH "${PROJECT_SOURCE_DIR}/tests/cpp/*.cc")
  enable_testing()

  add_executable(xgrammar_test ${XGRAMMAR_TEST_SOURCES_PATH})
  target_include_directories(xgrammar_test PUBLIC ${PROJECT_SOURCE_DIR}/cpp)
  target_link_libraries(xgrammar_test xgrammar gtest gmock gtest_main)

  include(GoogleTest)
  gtest_discover_tests(xgrammar_test)
endif()

if(XGRAMMAR_ENABLE_COVERAGE)
  target_link_libraries(xgrammar_bindings PRIVATE gcov)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")

  if(XGRAMMAR_BUILD_PYTHON_BINDINGS)
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} --coverage")
  endif()
endif()
