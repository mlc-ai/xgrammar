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
pip install apache-tvm-ffi==0.1.13.post3   # the version pinned in Cargo.toml

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

- **Reflection bridge.** The C++ bindings register their API as reflected
  *type methods*, which tvm-ffi's Rust crate cannot reach (it has no
  reflection support). `src/ffi/reflection.rs` therefore resolves
  `"<type_key>.<method_name>"` names directly against the C ABI reflection
  tables (`TVMFFIGetTypeInfo(index)->methods[]`; constructors via the
  `__ffi_init__` type-attribute column), obtaining the same `Function`
  objects the Python bindings dispatch to. True global functions (the
  `testing.*`/`kernels.*`/`config.*` set) still go through
  `Function::get_global`; the resolver tries the global table first and
  falls back to reflection. No C++-side support is needed, so the crate
  works against any xgrammar bindings library, including the ones shipped
  in existing Python wheels.
- **Opaque handles.** Each C++ object type (`Grammar`, `CompiledGrammar`,
  `GrammarCompiler`, `TokenizerInfo`, `GrammarMatcher`,
  `BatchGrammarMatcher`) is held as a plain `ObjectRef` strong reference
  (pointer-identity and `Debug` helpers in `src/ffi/handle.rs`). The Rust
  side never mirrors C++ field layouts and never allocates these objects;
  instances only come from FFI factory calls.
- **Packed calls only.** All calls go through `call_packed` (see the
  `ffi_call!` macro in `src/ffi/mod.rs`), avoiding the typed path's arity
  limit and its missing `Array<T>` argument support. Function handles are
  cached per call site in thread-locals (`tvm_ffi::Function` is not `Sync`).
- **Runtime library lookup via rpath.** The build bakes an rpath to the
  `tvm-ffi-config --libdir` directory into this crate's own tests and
  examples, so they start without loader configuration. Cargo does not
  propagate link args across packages, so downstream binaries (and
  relocated deployments) point the platform loader path (e.g.
  `LD_LIBRARY_PATH`) at the `libtvm_ffi` directory instead — the same
  requirement tvm-ffi documents for its Rust support.
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
