"""Load the xgrammar bindings."""

import os

from tvm_ffi.libinfo import load_lib_module

if os.environ.get("XGRAMMAR_BUILD_DOCS") == "1":
    # During documentation builds, skip loading the native library.
    LIB = None
else:
    LIB = load_lib_module("xgrammar", "xgrammar_bindings")
