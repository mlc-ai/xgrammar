find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# Compile this source file seperately. Nanobind suggests to optimize bindings code for size, but
# this source file contains mostly core logic. See notes about size optimizations here:
# https://nanobind.readthedocs.io/en/latest/api_cmake.html#command:nanobind_add_module
add_library(python_methods STATIC)
target_sources(python_methods PRIVATE python_methods.cc)
target_link_libraries(python_methods PUBLIC xgrammar)

# Any code that uses nanobind directly lives here
nanobind_add_module(xgrammar_bindings LTO nanobind.cc)
target_link_libraries(xgrammar_bindings PRIVATE python_methods)

if(DEFINED SKBUILD_PROJECT_NAME)
  # Building wheel through scikit-build-core
  set(LIB_OUTPUT_DIRECTORY xgrammar)
else()
  set(LIB_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/xgrammar)
endif()

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
