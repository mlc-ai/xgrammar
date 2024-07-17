cmake_minimum_required(VERSION 3.15)
project(xgrammar LANGUAGES CXX)

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
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "The build type" FORCE)
endif()

# Set pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(XGRAMMAR_INCLUDE
    ${PROJECT_SOURCE_DIR}/3rdparty/picojson
    ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include
    ${PROJECT_SOURCE_DIR}/include
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(XGRAMMAR_COMPILE_DEFINITIONS XGRAMMAR_ENABLE_LOG_DEBUG=1)
else()
    set(XGRAMMAR_COMPILE_DEFINITIONS XGRAMMAR_ENABLE_LOG_DEBUG=0)
endif()

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/cpp/*.cc")
add_library(xgrammar_objs OBJECT ${SOURCES})
target_include_directories(xgrammar_objs PUBLIC ${XGRAMMAR_INCLUDE})
target_compile_definitions(xgrammar_objs PUBLIC ${XGRAMMAR_COMPILE_DEFINITIONS})

add_library(xgrammar_lib $<TARGET_OBJECTS:xgrammar_objs>)

pybind11_add_module(xgrammar_bindings $<TARGET_OBJECTS:xgrammar_objs>)
install(TARGETS xgrammar_bindings DESTINATION .)
