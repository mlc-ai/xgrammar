//! Rust bindings for [XGrammar](https://github.com/mlc-ai/xgrammar), an
//! efficient, flexible and portable structured-generation engine.
//!
//! The API mirrors the Python `xgrammar` package: build a [`Grammar`] (from a
//! JSON schema, EBNF, a regex, Lark syntax or a structural tag), compile it
//! with a [`GrammarCompiler`] against a [`TokenizerInfo`], then drive
//! generation with a [`GrammarMatcher`] and a [`TokenBitmask`].
//!
//! ```no_run
//! use xgrammar::{Grammar, GrammarCompiler, GrammarMatcher, TokenBitmask, TokenizerInfo};
//!
//! # fn main() -> xgrammar::Result<()> {
//! # let (encoded_vocab, metadata): (Vec<Vec<u8>>, String) = unimplemented!();
//! # let mut logits: Vec<f32> = unimplemented!();
//! # fn sample(_logits: &[f32]) -> i64 { 0 }
//! let tokenizer_info = TokenizerInfo::from_vocab_and_metadata(&encoded_vocab, &metadata)?;
//! let compiler = GrammarCompiler::new(&tokenizer_info)?;
//! let compiled = compiler.compile_builtin_json_grammar()?;
//!
//! let mut matcher = GrammarMatcher::new(&compiled)?;
//! let mut bitmask = TokenBitmask::new(tokenizer_info.vocab_size()?);
//! while !matcher.is_terminated()? {
//!     if matcher.fill_next_token_bitmask(&mut bitmask, 0)? {
//!         bitmask.apply_to_logits(&mut logits, 0);
//!     }
//!     let token_id = sample(&logits);
//!     matcher.accept_token(token_id)?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Loading the native library
//!
//! The bindings drive the same `libxgrammar_bindings` shared library that
//! backs the Python package. It is located automatically (see
//! [`load_library`] for the search order); set the `XGRAMMAR_BINDINGS_LIB`
//! environment variable or call [`load_library`] when it lives somewhere
//! unusual.

mod bitmask;
mod compiler;
mod config;
mod error;
mod ffi;
mod grammar;
mod matcher;
pub mod structural_tag;
pub mod testing;
mod tokenizer_info;

pub use bitmask::{bitmask_words, TokenBitmask};
pub use compiler::{CompiledGrammar, GrammarCompiler, GrammarCompilerOptions};
pub use config::{
    get_max_recursion_depth, get_serialization_version, max_recursion_depth,
    set_max_recursion_depth, MaxRecursionDepthGuard,
};
pub use error::{Error, Result};
pub use ffi::{load_library, BINDINGS_LIBRARY_FILENAME};
pub use grammar::{Grammar, JsonSchemaOptions, NamedGrammar};
pub use matcher::{BatchGrammarMatcher, GrammarMatcher, MatcherOptions, MaxThreads};
pub use structural_tag::{Format, StructuralTag, StructuralTagItem, TagFormat};
pub use tokenizer_info::{
    HuggingFaceTokenizerOptions, TokenizerInfo, TokenizerInfoOptions, VocabType,
};
/// Hugging Face's native Rust tokenizer crate, re-exported so applications can
/// construct the exact [`tokenizers::Tokenizer`] accepted by
/// [`TokenizerInfo::from_huggingface`].
pub use tokenizers;
