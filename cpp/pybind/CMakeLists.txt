set(PYBIND11_FINDPYTHON ON)
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
