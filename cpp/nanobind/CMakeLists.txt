find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED
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
  set(LIB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/xgrammar)
else()
  set(LIB_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/xgrammar)
endif()

set(STUBFILE_OUTPUTS
    ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings/__init__.pyi
    ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings/config.pyi
    ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings/kernels.pyi
    ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings/testing/__init__.pyi
    ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings/testing/grammar_functor.pyi
)

add_custom_command(
  OUTPUT ${STUBFILE_OUTPUTS}
  COMMAND "${Python_EXECUTABLE}" -m nanobind.stubgen -m xgrammar_bindings -r -O
          "${LIB_OUTPUT_DIRECTORY}"
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings/testing
  # See https://github.com/wjakob/nanobind/issues/989 - the default place that these bindings are
  # generated in is incorrect.
  COMMAND ${CMAKE_COMMAND} -E rename ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings.pyi
          ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings/__init__.pyi
  COMMAND ${CMAKE_COMMAND} -E rename ${LIB_OUTPUT_DIRECTORY}/config/__init__.pyi
          ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings/config.pyi
  COMMAND ${CMAKE_COMMAND} -E rename ${LIB_OUTPUT_DIRECTORY}/kernels/__init__.pyi
          ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings/kernels.pyi
  COMMAND ${CMAKE_COMMAND} -E rename ${LIB_OUTPUT_DIRECTORY}/testing/__init__.pyi
          ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings/testing/__init__.pyi
  COMMAND ${CMAKE_COMMAND} -E rename ${LIB_OUTPUT_DIRECTORY}/testing/grammar_functor/__init__.pyi
          ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings/testing/grammar_functor.pyi
  WORKING_DIRECTORY "$<TARGET_FILE_DIR:xgrammar_bindings>"
  DEPENDS xgrammar_bindings
)
add_custom_target(xgrammar_stubfiles ALL DEPENDS ${STUBFILE_OUTPUTS})

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

install(TARGETS xgrammar_bindings DESTINATION .)
install(DIRECTORY ${LIB_OUTPUT_DIRECTORY}/xgrammar_bindings DESTINATION .)
