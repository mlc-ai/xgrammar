"""Load the xgrammar bindings."""

import os
import sys
from inspect import signature
from pathlib import Path

from tvm_ffi.libinfo import load_lib_module

if os.environ.get("XGRAMMAR_BUILD_DOCS") == "1":
    # During documentation builds, skip loading the native library.
    LIB = None
elif "extra_lib_paths" in signature(load_lib_module).parameters:
    package = sys.modules[__package__ or "xgrammar"]
    package_paths = [Path(path) for path in getattr(package, "__path__", ())]
    LIB = load_lib_module("xgrammar", "xgrammar_bindings", extra_lib_paths=package_paths)
else:
    # Older FFI builds can still find installed wheel libraries via package metadata.
    LIB = load_lib_module("xgrammar", "xgrammar_bindings")
