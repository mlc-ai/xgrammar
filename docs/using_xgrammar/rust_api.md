# Rust API

Beside the Python and C++ API, XGrammar also provides a Rust API with the same
functionality as the Python package.

The Rust bindings live in [`rust/`](https://github.com/mlc-ai/xgrammar/tree/main/rust)
and drive the same `libxgrammar_bindings` shared library that backs the Python
package, through [apache/tvm-ffi](https://github.com/apache/tvm-ffi)'s Rust
support. Grammars, compiled grammars and serialized artifacts are therefore
fully interchangeable between the two languages.

## Installation

The crate is not yet published on crates.io (it pins its tvm-ffi dependency
to a git revision), so use it as a git or path dependency:

```toml
[dependencies]
xgrammar = { git = "https://github.com/mlc-ai/xgrammar.git" }
```

Building requires:

1. **The `apache-tvm-ffi` Python package**, which provides the `tvm-ffi-config`
   tool and the `libtvm_ffi` shared library the crate links against:

   ```bash
   pip install apache-tvm-ffi==0.1.13.post3   # the version pinned in rust/Cargo.toml
   ```

   `tvm-ffi-config` must be on `PATH` when running cargo (activating the
   virtual environment is enough). The build bakes an rpath to this
   installation, so binaries built on the same machine start without loader
   configuration. If the installation moves after building a binary, point
   the platform loader path (e.g. `LD_LIBRARY_PATH`) at the directory
   containing `libtvm_ffi`.

2. **The bindings library** (`libxgrammar_bindings.so` on Linux,
   `libxgrammar_bindings.dylib` on macOS, or `xgrammar_bindings.dll` on
   Windows). The crate resolves the bindings' API through tvm-ffi's
   reflection tables at runtime, so the library needs no Rust-specific
   support: either take it from an installed `xgrammar` Python wheel (it is
   inside the `xgrammar` package directory), or build it from source:

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -G Ninja
   cmake --build build --target xgrammar_bindings
   # produces the platform library under python/xgrammar/
   ```

```{note}
The Rust crate pins tvm-ffi to the git tag of a released version (currently
`v0.1.13-post3`, i.e. pip's `0.1.13.post3`; see `rust/Cargo.toml`). Keep the
pin, the installed `apache-tvm-ffi` package and the bindings build in
lockstep: all three share one `libtvm_ffi` shared library, and its ABI is
still evolving.
```

## Loading the bindings library

The library is located automatically, in this order:

1. registrations already present in the process — nothing to do when the host
   application (e.g. an embedded Python interpreter) already loaded xgrammar;
2. the path in the `XGRAMMAR_BINDINGS_LIB` environment variable;
3. the repository checkout the crate was built from (development builds);
4. the `xgrammar` Python package discovered at build time — a
   `pip install xgrammar` is enough for binaries to start without any loading
   code;
5. the platform's `xgrammar_bindings` library name via the system loader search
   path.

When the library lives somewhere unusual, point the environment variable at it
or call `xgrammar::load_library` before any other API (the platform-specific
file name is exported as `xgrammar::BINDINGS_LIBRARY_FILENAME`):

```rust
xgrammar::load_library("/path/to/the/xgrammar_bindings/library")?;
```

## Quick start

The flow mirrors the Python API: build a grammar, compile it against tokenizer
info, then drive generation with a matcher and a token bitmask.

```rust
use xgrammar::{
    tokenizers::Tokenizer, GrammarCompiler, GrammarMatcher, HuggingFaceTokenizerOptions,
    JsonSchemaOptions, TokenBitmask, TokenizerInfo,
};

fn main() -> xgrammar::Result<()> {
    // Load the same tokenizer.json used by a Hugging Face fast tokenizer.
    // Downloading from the Hub is intentionally left to `hf-hub`; loading and
    // conversion itself are offline.
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;
    let tokenizer_info = TokenizerInfo::from_huggingface(
        &tokenizer,
        &HuggingFaceTokenizerOptions {
            // Use the model config's lm_head dimension and EOS id here.
            vocab_size: Some(model_vocab_size),
            stop_token_ids: Some(vec![eos_id]),
        },
    )?;

    let compiler = GrammarCompiler::new(&tokenizer_info)?;
    let compiled = compiler.compile_json_schema(schema, &JsonSchemaOptions::default())?;

    let mut matcher = GrammarMatcher::new(&compiled)?;
    let mut bitmask = TokenBitmask::new(tokenizer_info.vocab_size()?);
    while !matcher.is_terminated()? {
        if matcher.fill_next_token_bitmask(&mut bitmask, 0)? {
            // CPU masking; for GPU inference, apply the bitmask with your
            // engine's kernel instead (the layout is the same int32 bitset
            // used by the Python API).
            bitmask.apply_to_logits(&mut logits, 0);
        }
        let token_id = sample(&logits);
        matcher.accept_token(token_id)?;
    }
    Ok(())
}
```

A runnable version of this loop is in
[`rust/examples/constrained_decoding.rs`](https://github.com/mlc-ai/xgrammar/blob/main/rust/examples/constrained_decoding.rs).

## API overview

The mapping from the Python API is mechanical:

| Python | Rust |
|---|---|
| `Grammar.from_ebnf / from_json_schema / from_regex / from_lark / from_structural_tag / builtin_json_grammar / union / concat` | `Grammar::from_ebnf / from_json_schema / from_regex / from_lark / from_structural_tag / builtin_json_grammar / union / concat` |
| `TokenizerInfo(...)`, `from_huggingface`, `from_vocab_and_metadata` | `TokenizerInfo::new`, `from_huggingface`, `from_vocab_and_metadata` |
| `GrammarCompiler(...)`, `compile_*` | `GrammarCompiler::new / with_options`, `compile_*` |
| `CompiledGrammar` properties | `CompiledGrammar` methods |
| `GrammarMatcher` methods | `GrammarMatcher` methods (`&mut self`; `fork()` instead of copying) |
| `BatchGrammarMatcher` | `BatchGrammarMatcher` (operates on `&mut [GrammarMatcher]`) |
| `allocate_token_bitmask`, `reset_token_bitmask`, `apply_token_bitmask_inplace` (CPU) | `TokenBitmask::new / with_batch / reset / apply_to_logits` (native Rust) |
| `StructuralTag` and the `*Format` pydantic models | `StructuralTag` and the `Format` serde enum (identical JSON wire format) |
| `serialize_json` / `deserialize_json` everywhere | same, interchangeable with Python-produced JSON |
| `xgrammar.testing._*`, `GrammarFunctor` | `xgrammar::testing::*`, `testing::grammar_functor` |
| `get/set_max_recursion_depth`, `max_recursion_depth` context manager | `get/set_max_recursion_depth`, `max_recursion_depth` RAII guard |
| exceptions (`InvalidJSONError`, ...) | `xgrammar::Error` variants mapped from the same error kinds |

Conventions on the Rust side:

- **Everything fallible returns `Result`** — errors raised across the FFI keep
  their kind and message (and backtrace, for unmapped kinds).
- **Keyword arguments become option structs** (`JsonSchemaOptions`,
  `TokenizerInfoOptions`, `MatcherOptions`, `GrammarCompilerOptions`) with
  `Default` implementations matching the Python defaults; use
  `..Default::default()` for the rest.
- **Token content is `[u8]`, not `str`** — vocabularies, captures and
  `accept_string` inputs are raw bytes, since tokens need not be valid UTF-8.
- **Thread safety follows the C++ semantics**: `Grammar`, `CompiledGrammar`
  and `TokenizerInfo` are immutable and `Send + Sync` (clones share the
  handle); `GrammarCompiler` is `Send` but not `Sync` while native cache reads
  remain unsynchronized; `GrammarMatcher` is a mutable state machine, `Send`
  but not `Sync`.

## Differences from the Python package

The remaining Python-only functionality is not mirrored:

- `TokenizerInfo::from_huggingface` accepts the native Rust
  `tokenizers::Tokenizer`, corresponding to a Hugging Face fast tokenizer. It
  cannot accept Python-only slow SentencePiece/tiktoken wrapper objects. Also,
  `tokenizer.json` does not contain the model's lm_head size or identify its
  EOS semantics, so pass `vocab_size` and `stop_token_ids` from the model
  config when they matter.
- GPU `apply_token_bitmask_inplace` kernels (CUDA/Triton/Metal): apply the
  bitmask with your inference engine's kernel; the bitmask layout is
  identical.
- `builtin_structural_tag` (per-model tool-call templates) and the
  HuggingFace/mlx `contrib` integrations.

## Development

Tests live in `tests/rust/` (next to the Python and C++ suites) and run
against the locally built bindings:

```bash
cmake --build build --target xgrammar_bindings   # once
cd rust && cargo test
```

The real-tokenizer parity test uses the same gated Llama 2 and Llama 3
tokenizers and rejected-token counts as Python's `hf_token_required` suite. It
is ignored by the default offline run; provide `HF_TOKEN` (or populate the
standard Hugging Face cache) and run it explicitly:

```bash
cd rust
cargo test --features hf-token-tests --test huggingface -- --ignored --nocapture
```

The test suite covers the full API surface and cross-checks the structural tag
wire format byte-for-byte against the pydantic models.
