"""Load the xgrammar bindings."""

from tvm_ffi.libinfo import load_lib_module

LIB = load_lib_module("xgrammar", "xgrammar_bindings")
