"""Load the xgrammar bindings."""

import os

from tvm_ffi.libinfo import load_lib_module

if os.environ.get("XGRAMMAR_BUILD_DOCS") == "1":
    # During documentation builds, skip loading the native library.
    LIB = None
else:
    try:
        LIB = load_lib_module("xgrammar", "xgrammar_bindings")
    except (ImportError, OSError):
        # In environments where the native library is not available yet,
        # make this module a no-op instead of failing at import time.
        LIB = None
