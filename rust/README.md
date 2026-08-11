<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership. -->

# XGrammar Rust Bindings

Rust bindings for XGrammar with the same functionality as the Python package,
built on [apache/tvm-ffi](https://github.com/apache/tvm-ffi)'s Rust support.
They drive the same `libxgrammar_bindings` shared library as the Python
package, so grammars, compiled grammars and serialized artifacts are fully
interchangeable between the two languages.

See the [Rust API documentation page](../docs/using_xgrammar/rust_api.md) for
installation and usage; run `cargo doc --open` for the API reference.

## Quick development loop

```bash
# 1. one-time: the tvm-ffi toolchain used for linking
pip install apache-tvm-ffi        # provides tvm-ffi-config + libtvm_ffi

# 2. build the bindings library (from the repository root)
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja
cmake --build build --target xgrammar_bindings

# 3. build, test, try
cd rust
cargo test
cargo run --example constrained_decoding
```

`tvm-ffi-config` must be on `PATH` (activate the environment where
`apache-tvm-ffi` is installed). The integration tests load
the platform's `libxgrammar_bindings` file from `../python/xgrammar`, i.e. the
library built in step 2.

## How it works

- **Global-function aliases.** The C++ bindings register their API as
  reflected *type methods*, which tvm-ffi's Rust crate cannot reach (it has no
  reflection support). `cpp/tvm_ffi/tvm_ffi.cc` therefore mirrors every
  reflected method into the global function table as
  `"<type_key>.<method_name>"` (constructors as `"<type_key>.__ffi_init__"`)
  in `RegisterGlobalFunctionAliases` — a generic loop over the reflection
  tables, so new methods get aliases automatically. The Rust side then only
  needs `Function::get_global`.
- **Opaque handles.** Each C++ object type (`Grammar`, `CompiledGrammar`,
  `GrammarCompiler`, `TokenizerInfo`, `GrammarMatcher`,
  `BatchGrammarMatcher`) is wrapped as a single strong reference bound by its
  type key (`src/ffi/objects.rs`). The Rust side never mirrors C++ field
  layouts and never allocates these objects; instances only come from FFI
  factory calls.
- **Packed calls only.** All calls go through `call_packed` (see the
  `ffi_call!` macro in `src/ffi/mod.rs`), avoiding the typed path's arity
  limit and its missing `Array<T>` argument support. Function handles are
  cached per call site in thread-locals (`tvm_ffi::Function` is not `Sync`).
- **Consumer-safe runtime loading.** A small static C shim loads the
  `libtvm_ffi` selected by `tvm-ffi-config`. Unlike an rpath link argument,
  the shim is bundled through the rlib into downstream executables, so path
  and git consumers can start without setting a platform loader path. Set
  `TVM_FFI_LIBRARY_PATH` to a full library path if the installation moves
  after the Rust binary is built.
- **Zero-copy tensors.** Token bitmasks and the draft-tree/temperature
  buffers are passed as borrowed `DLTensor` views over Rust slices
  (`DlArg` in `src/ffi/mod.rs`), matching the DLPack contract of the C++
  side.
- **Pure-Rust ports.** The structural tag models (pydantic in Python) are
  serde types with a byte-for-byte identical JSON wire format
  (`src/structural_tag.rs`); the CPU bitmask helpers (torch in Python) are
  native Rust (`src/bitmask.rs`). Hugging Face fast tokenizers are loaded by
  the official Rust `tokenizers` crate and converted directly with
  `TokenizerInfo::from_huggingface`.

## Version pinning

`Cargo.toml` pins tvm-ffi to the git revision of the released version that
`libxgrammar_bindings` links against (currently `v0.1.13-post3`). Keep the pin, the
`apache-tvm-ffi` package version, and the bindings build in lockstep: all
three share one `libtvm_ffi` ABI, and mixing revisions puts two incompatible
ABIs (or two crate instantiations with split registries) into one process.
