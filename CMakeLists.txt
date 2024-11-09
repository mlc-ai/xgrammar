cmake_minimum_required(VERSION 3.15)
project(xgrammar LANGUAGES CXX)

if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()
endif()

option(XGRAMMAR_BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

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

if(MSVC)
  set(CMAKE_CXX_FLAGS "/Wall ${CMAKE_CXX_FLAGS}")
else()
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-Ofast ${CMAKE_CXX_FLAGS}")
  endif()

  set(CMAKE_CXX_FLAGS
      "-Wall -Wextra -Werror -pedantic -Wno-unused-parameter -flto=auto ${CMAKE_CXX_FLAGS}"
  )
endif()

set(XGRAMMAR_INCLUDE_PATH
    ${PROJECT_SOURCE_DIR}/3rdparty/picojson ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include
    ${PROJECT_SOURCE_DIR}/include
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(XGRAMMAR_COMPILE_DEFINITIONS XGRAMMAR_ENABLE_LOG_DEBUG=1)
else()
  set(XGRAMMAR_COMPILE_DEFINITIONS XGRAMMAR_ENABLE_LOG_DEBUG=0)
endif()

file(GLOB_RECURSE XGRAMMAR_SOURCES_PATH "${PROJECT_SOURCE_DIR}/cpp/*.cc")
list(FILTER XGRAMMAR_SOURCES_PATH EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/cpp/pybind/.*\\.cc")

add_library(xgrammar STATIC ${XGRAMMAR_SOURCES_PATH})
target_include_directories(xgrammar PUBLIC ${XGRAMMAR_INCLUDE_PATH})
target_compile_definitions(xgrammar PUBLIC ${XGRAMMAR_COMPILE_DEFINITIONS})

if(XGRAMMAR_BUILD_PYTHON_BINDINGS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/cpp/pybind)
endif()
