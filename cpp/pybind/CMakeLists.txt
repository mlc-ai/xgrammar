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

file(GLOB_RECURSE XGRAMMAR_BINDINGS_PATH ${PROJECT_SOURCE_DIR}/cpp/pybind/*.cc)
pybind11_add_module(xgrammar_bindings ${XGRAMMAR_BINDINGS_PATH})
target_include_directories(xgrammar_bindings PRIVATE ${XGRAMMAR_INCLUDE_PATH})
target_compile_definitions(xgrammar_bindings PRIVATE ${XGRAMMAR_COMPILE_DEFINITIONS})
target_link_libraries(xgrammar_bindings PRIVATE xgrammar)
