find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED
)

# Discover tvm_ffi from Python environment (used by scikit-build-core and standalone)
if(NOT DEFINED tvm_ffi_ROOT)
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" -c
      "import tvm_ffi, os; print(os.path.join(os.path.dirname(tvm_ffi.__file__), 'share', 'cmake', 'tvm_ffi'))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE tvm_ffi_ROOT
  )
endif()
find_package(tvm_ffi CONFIG REQUIRED)

# Core logic used by bindings (no tvm_ffi dependency)
add_library(python_methods STATIC python_methods.cc)
target_include_directories(python_methods PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(python_methods PUBLIC xgrammar)

# TVM-FFI shared library: C++ bindings entry point
add_library(xgrammar_bindings SHARED tvm_ffi.cc)
tvm_ffi_configure_target(xgrammar_bindings STUB_DIR "${PROJECT_SOURCE_DIR}/python" STUB_INIT ON)
install(TARGETS xgrammar_bindings DESTINATION .)
tvm_ffi_install(xgrammar_bindings DESTINATION .)
target_include_directories(xgrammar_bindings PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Use tvm_ffi's include (and its DLPack) before project includes to satisfy tvm_ffi headers
get_target_property(TVM_FFI_INCLUDES tvm_ffi::header INTERFACE_INCLUDE_DIRECTORIES)
if(TVM_FFI_INCLUDES)
  target_include_directories(xgrammar_bindings BEFORE PRIVATE ${TVM_FFI_INCLUDES})
endif()
target_link_libraries(xgrammar_bindings PRIVATE python_methods)
target_link_libraries(xgrammar_bindings PRIVATE tvm_ffi::header tvm_ffi::shared)

if(DEFINED SKBUILD_PROJECT_NAME)
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
