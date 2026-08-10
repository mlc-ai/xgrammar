//! Grammar compilation: [`GrammarCompiler`] and [`CompiledGrammar`].

use std::marker::PhantomData;
use tvm_ffi::String as FfiString;

use crate::error::Result;
use crate::ffi::objects::{RawCompiledGrammar, RawGrammarCompiler};
use crate::ffi::{ffi_call, opt_any, ret};
use crate::grammar::{Grammar, JsonSchemaOptions};
use crate::tokenizer_info::TokenizerInfo;

/// A grammar compiled against a specific tokenizer vocabulary, ready to drive
/// a [`crate::GrammarMatcher`].
///
/// This class is immutable: instances can be cloned cheaply (shared handle)
/// and shared across threads. Obtain one from a [`GrammarCompiler`], or by
/// deserializing with [`CompiledGrammar::deserialize_json`].
#[derive(Clone)]
pub struct CompiledGrammar {
    pub(crate) raw: RawCompiledGrammar,
}

impl CompiledGrammar {
    pub(crate) fn from_raw(raw: RawCompiledGrammar) -> Self {
        Self { raw }
    }

    /// The grammar this object was compiled from.
    pub fn grammar(&self) -> Result<Grammar> {
        let any = ffi_call!("xgrammar.tvm_ffi_binding.CompiledGrammar.grammar", self.raw)?;
        Ok(Grammar::from_raw(ret(any)?))
    }

    /// The tokenizer info this object was compiled against.
    pub fn tokenizer_info(&self) -> Result<TokenizerInfo> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.CompiledGrammar.tokenizer_info",
            self.raw,
        )?;
        Ok(TokenizerInfo::from_raw(ret(any)?))
    }

    /// The approximate memory consumed by this compiled grammar, in bytes.
    pub fn memory_size_bytes(&self) -> Result<usize> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.CompiledGrammar.memory_size_bytes",
            self.raw,
        )?;
        Ok(ret::<i64>(any)? as usize)
    }

    /// Serialize to JSON. The tokenizer vocabulary is not embedded — only its
    /// metadata — so deserialization needs the matching [`TokenizerInfo`].
    pub fn serialize_json(&self) -> Result<String> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.CompiledGrammar.serialize_json",
            self.raw,
        )?;
        Ok(ret::<FfiString>(any)?.as_str().to_string())
    }

    /// Deserialize from the JSON produced by
    /// [`CompiledGrammar::serialize_json`]. `tokenizer_info` must describe
    /// the same tokenizer the grammar was compiled against.
    pub fn deserialize_json(json: &str, tokenizer_info: &TokenizerInfo) -> Result<Self> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.CompiledGrammar.deserialize_json",
            FfiString::from(json),
            tokenizer_info.raw,
        )?;
        Ok(Self::from_raw(ret(any)?))
    }
}

impl std::fmt::Debug for CompiledGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledGrammar({:?})", self.raw)
    }
}

/// Options for [`GrammarCompiler::with_options`], mirroring the Python
/// constructor's keyword arguments.
#[derive(Debug, Clone)]
pub struct GrammarCompilerOptions {
    /// Maximum number of threads used for compilation.
    pub max_threads: usize,
    /// Whether compiled grammars are cached.
    pub cache_enabled: bool,
    /// Cache size limit in bytes; `None` means unlimited.
    pub cache_limit_bytes: Option<usize>,
}

impl Default for GrammarCompilerOptions {
    fn default() -> Self {
        Self {
            max_threads: 8,
            cache_enabled: true,
            cache_limit_bytes: None,
        }
    }
}

/// Compiles grammars against one tokenizer vocabulary, with an internal
/// cache.
///
/// A compiler can be moved between threads (`Send`) but not shared between
/// them (`!Sync`), because all native cache reads are not yet synchronized.
/// Each compilation can still use up to `max_threads` internal worker threads.
pub struct GrammarCompiler {
    raw: RawGrammarCompiler,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

impl GrammarCompiler {
    /// Create a compiler with default options (8 threads, unlimited cache).
    pub fn new(tokenizer_info: &TokenizerInfo) -> Result<Self> {
        Self::with_options(tokenizer_info, &GrammarCompilerOptions::default())
    }

    /// Create a compiler with explicit options.
    pub fn with_options(
        tokenizer_info: &TokenizerInfo,
        options: &GrammarCompilerOptions,
    ) -> Result<Self> {
        let cache_limit = match options.cache_limit_bytes {
            Some(bytes) => bytes as i64,
            None => -1,
        };
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarCompiler.__ffi_init__",
            tokenizer_info.raw,
            options.max_threads as i64,
            options.cache_enabled,
            cache_limit,
        )?;
        Ok(Self {
            raw: ret(any)?,
            _not_sync: PhantomData,
        })
    }

    /// Compile a [`Grammar`].
    pub fn compile_grammar(&self, grammar: &Grammar) -> Result<CompiledGrammar> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarCompiler.compile_grammar_ebnf",
            self.raw,
            grammar.raw,
        )?;
        Ok(CompiledGrammar::from_raw(ret(any)?))
    }

    /// Compile a grammar given as EBNF text with root rule `root` (the
    /// string overload of Python's `compile_grammar`).
    pub fn compile_ebnf(&self, ebnf: &str) -> Result<CompiledGrammar> {
        self.compile_ebnf_with_root(ebnf, "root")
    }

    /// Compile EBNF text with an explicit root rule name.
    pub fn compile_ebnf_with_root(
        &self,
        ebnf: &str,
        root_rule_name: &str,
    ) -> Result<CompiledGrammar> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarCompiler.compile_grammar_from_strings",
            self.raw,
            FfiString::from(ebnf),
            FfiString::from(root_rule_name),
        )?;
        Ok(CompiledGrammar::from_raw(ret(any)?))
    }

    /// Compile a grammar from a JSON schema. `options.print_converted_ebnf`
    /// is ignored here (as in the Python API, which does not expose it on the
    /// compiler).
    pub fn compile_json_schema(
        &self,
        schema: &str,
        options: &JsonSchemaOptions,
    ) -> Result<CompiledGrammar> {
        let separators = options.separators.as_ref().map(|(item, key)| {
            tvm_ffi::Array::<FfiString>::new(vec![FfiString::from(item), FfiString::from(key)])
        });
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarCompiler.compile_json_schema",
            self.raw,
            FfiString::from(schema),
            options.any_whitespace,
            opt_any(options.indent),
            opt_any(separators),
            options.strict_mode,
            opt_any(options.max_whitespace_cnt),
            options.any_order,
        )?;
        Ok(CompiledGrammar::from_raw(ret(any)?))
    }

    /// Compile the built-in grammar accepting any valid JSON document.
    pub fn compile_builtin_json_grammar(&self) -> Result<CompiledGrammar> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarCompiler.compile_builtin_json_grammar",
            self.raw,
        )?;
        Ok(CompiledGrammar::from_raw(ret(any)?))
    }

    /// Compile a grammar from a regular expression.
    pub fn compile_regex(&self, regex: &str) -> Result<CompiledGrammar> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarCompiler.compile_regex",
            self.raw,
            FfiString::from(regex),
        )?;
        Ok(CompiledGrammar::from_raw(ret(any)?))
    }

    /// Compile a grammar from a structural tag; see [`crate::StructuralTag`].
    pub fn compile_structural_tag(
        &self,
        structural_tag: &crate::StructuralTag,
    ) -> Result<CompiledGrammar> {
        self.compile_structural_tag_json(&structural_tag.to_json()?)
    }

    /// Compile a grammar from a serialized structural tag (JSON text); see
    /// [`Grammar::from_structural_tag_json`].
    pub fn compile_structural_tag_json(
        &self,
        structural_tag_json: &str,
    ) -> Result<CompiledGrammar> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarCompiler.compile_structural_tag",
            self.raw,
            FfiString::from(structural_tag_json),
        )?;
        Ok(CompiledGrammar::from_raw(ret(any)?))
    }

    /// Clear the compilation cache.
    pub fn clear_cache(&self) -> Result<()> {
        ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarCompiler.clear_cache",
            self.raw
        )?;
        Ok(())
    }

    /// The current size of the compilation cache, in bytes.
    pub fn cache_size_bytes(&self) -> Result<usize> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarCompiler.get_cache_size_bytes",
            self.raw,
        )?;
        Ok(ret::<i64>(any)? as usize)
    }

    /// The cache size limit in bytes, or `None` when unlimited.
    pub fn cache_limit_bytes(&self) -> Result<Option<usize>> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarCompiler.cache_limit_bytes",
            self.raw,
        )?;
        let limit = ret::<i64>(any)?;
        Ok((limit >= 0).then_some(limit as usize))
    }
}

impl std::fmt::Debug for GrammarCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GrammarCompiler({:?})", self.raw)
    }
}
