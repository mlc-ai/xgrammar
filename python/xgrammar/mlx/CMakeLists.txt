cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX)

# Setup like https://github.com/ml-explore/mlx/blob/main/examples/extensions/CMakeLists.txt
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED
)

# Import CMake rule nanobind_add_module from nanobind installation directory. Also find the CMake
# library target mlx, which is to be linked to our extension.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# Import CMake rule mlx_build_metallib from mlx installation directory.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT
)
find_package(MLX CONFIG REQUIRED)

# Build extension library mlx_ext including MLX operation and primitive.
add_library(mlx_ext)
target_sources(mlx_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR}/apply_token_bitmask.cpp)
target_include_directories(mlx_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(mlx_ext PUBLIC mlx)
if(MLX_BUILD_METAL)
  # Build GPU kernel only if MLX's CMake rule tells that MLX was built with Metal support.
  mlx_build_metallib(
    TARGET
    mlx_ext_metallib
    TITLE
    mlx_ext
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/apply_token_bitmask.metal
    INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}
  )
  add_dependencies(mlx_ext mlx_ext_metallib)
endif()

# Wrap the extension library mlx_ext in a Python module.
nanobind_add_module(
  extension
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp
)
target_link_libraries(extension PRIVATE mlx_ext)
if(BUILD_SHARED_LIBS)
  target_link_options(extension PRIVATE -Wl,-rpath,@loader_path)
endif()

# Copy built results to source directory so they could be packed and imported.
add_custom_target(
  copy_mlx_extensions ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_BINARY_DIR}/extension.cpython-*-darwin.so" ${CMAKE_CURRENT_SOURCE_DIR}/
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/libmlx_ext.dylib
          ${CMAKE_CURRENT_SOURCE_DIR}/
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/mlx_ext.metallib
          ${CMAKE_CURRENT_SOURCE_DIR}/
  DEPENDS extension mlx_ext
  COMMENT "Copying MLX extensions to source directory"
)
