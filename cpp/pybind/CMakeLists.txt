# Set Pybind11
find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} "-c" "import pybind11 as pb11; print(pb11.get_cmake_dir(),end='');"
  RESULT_VARIABLE PYBIND_CMAKE_DIR_RET
  OUTPUT_VARIABLE PYBIND_CMAKE_DIR
)
if(PYBIND_CMAKE_DIR_RET MATCHES 0)
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND_CMAKE_DIR}")
else()
  message(FATAL_ERROR "pybind11 is not installed. Please install pybind11 with pip or conda first")
endif()
find_package(pybind11 CONFIG REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} "-c" "import torch; print(torch.utils.cmake_prefix_path,end='');"
  RESULT_VARIABLE TORCH_CMAKE_DIR_RET
  OUTPUT_VARIABLE TORCH_CMAKE_DIR
)
if(TORCH_CMAKE_DIR_RET MATCHES 0)
  list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
else()
  message(FATAL_ERROR "PyTorch is not installed. Please install PyTorch with pip or conda first")
endif()
find_package(Torch CONFIG REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# The compilation flags for bindings is different from the main library as torch requires
# -D_GLIBCXX_USE_CXX11_ABI=0. So we compile bindings separately.
file(GLOB_RECURSE XGRAMMAR_BINDINGS_PATH ${PROJECT_SOURCE_DIR}/cpp/*.cc)
pybind11_add_module(xgrammar_bindings ${XGRAMMAR_BINDINGS_PATH})
target_include_directories(xgrammar_bindings PUBLIC ${XGRAMMAR_INCLUDE_PATH})
target_compile_definitions(xgrammar_bindings PUBLIC ${XGRAMMAR_COMPILE_DEFINITIONS})
target_link_libraries(xgrammar_bindings PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
set_target_properties(
  xgrammar_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/xgrammar"
)

if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(
    TARGET xgrammar_bindings
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
            $<TARGET_FILE_DIR:xgrammar_bindings>
  )
endif(MSVC)
