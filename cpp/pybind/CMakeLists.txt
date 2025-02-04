# Setup Pybind11
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

file(GLOB_RECURSE XGRAMMAR_BINDINGS_PATH ${PROJECT_SOURCE_DIR}/cpp/*.cc)
pybind11_add_module(xgrammar_bindings ${XGRAMMAR_BINDINGS_PATH})
target_include_directories(xgrammar_bindings PUBLIC ${XGRAMMAR_INCLUDE_PATH})

if(XGRAMMAR_ENABLE_CPPTRACE)
  target_link_libraries(xgrammar_bindings PUBLIC cpptrace::cpptrace)
  target_compile_definitions(xgrammar_bindings PUBLIC XGRAMMAR_ENABLE_CPPTRACE=1)
else()
  target_compile_definitions(xgrammar_bindings PUBLIC XGRAMMAR_ENABLE_CPPTRACE=0)
endif()

set(LIB_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/xgrammar")
set_target_properties(xgrammar_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIB_OUTPUT_DIRECTORY})
set_target_properties(
  xgrammar_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG ${LIB_OUTPUT_DIRECTORY}
)
set_target_properties(
  xgrammar_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE ${LIB_OUTPUT_DIRECTORY}
)
set_target_properties(
  xgrammar_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY_REL_WITH_DEB_INFO ${LIB_OUTPUT_DIRECTORY}
)
