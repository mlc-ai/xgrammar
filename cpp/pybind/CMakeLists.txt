# Setup Pybind11
set(PYBIND11_NEWPYTHON ON)
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

install(TARGETS xgrammar_bindings LIBRARY DESTINATION .)
