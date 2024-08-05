cmake_minimum_required(VERSION 3.15)
project(xgrammar LANGUAGES CXX)

OPTION(BUILD_BINDINGS "Build Python bindings" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  set(CMAKE_CXX_FLAGS "/Wall /WX ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -pedantic -Wno-unused-parameter ${CMAKE_CXX_FLAGS}")
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type specified; defaulting to CMAKE_BUILD_TYPE=Debug.")
  set(CMAKE_BUILD_TYPE
      "Debug"
      CACHE STRING "The build type" FORCE
  )
endif()

set(XGRAMMAR_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/3rdparty/picojson
                     ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include ${PROJECT_SOURCE_DIR}/include
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(XGRAMMAR_COMPILE_DEFINITIONS XGRAMMAR_ENABLE_LOG_DEBUG=1)
else()
  set(XGRAMMAR_COMPILE_DEFINITIONS XGRAMMAR_ENABLE_LOG_DEBUG=0)
endif()

file(GLOB_RECURSE XGRAMMAR_SOURCES_PATH "${PROJECT_SOURCE_DIR}/cpp/*.cc")
list(FILTER XGRAMMAR_SOURCES_PATH EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/cpp/pybind/.*\\.cc")

add_library(xgrammar SHARED ${XGRAMMAR_SOURCES_PATH})
target_include_directories(xgrammar PUBLIC ${XGRAMMAR_INCLUDE_PATH})
target_compile_definitions(xgrammar PUBLIC ${XGRAMMAR_COMPILE_DEFINITIONS})

if(BUILD_BINDINGS)
  message(STATUS "XGrammar Python bindings will be built")
  add_subdirectory(${PROJECT_SOURCE_DIR}/cpp/pybind)
endif()
