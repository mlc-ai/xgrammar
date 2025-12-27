use autocxx::prelude::*;

use crate::{
    CxxUniquePtr, FFIGrammarCompiler,
    compiler::CompiledGrammar,
    cxx_int, cxx_longlong, cxx_utils,
    grammar::{self, StructuralTagItem},
    tokenizer_info::TokenizerInfo,
};

/// The compiler for grammars.
///
/// It is associated with a certain tokenizer info, and compiles grammars into `CompiledGrammar`
/// with the tokenizer info. It allows parallel compilation with multiple threads, and has a cache
/// to store the compilation result, avoiding compiling the same grammar multiple times.
pub struct GrammarCompiler {
    inner: CxxUniquePtr<FFIGrammarCompiler>,
}

impl GrammarCompiler {
    /// Construct the compiler.
    ///
    /// # Parameters
    ///
    /// - `tokenizer_info`: The tokenizer info.
    /// - `max_threads`: The maximum number of threads used to compile the grammar.
    /// - `cache_enabled`: Whether to enable the cache.
    /// - `cache_limit_bytes`: The maximum memory usage for the cache in bytes.
    ///   Note that the actual memory usage may slightly exceed this value.
    ///
    /// # Errors
    ///
    /// Returns an error if the grammar compiler cannot be constructed.
    pub fn new(
        tokenizer_info: &TokenizerInfo,
        max_threads: i32,
        cache_enabled: bool,
        cache_limit_bytes: isize,
    ) -> Result<Self, String> {
        cxx::let_cxx_string!(error_out_cxx = "");
        let inner = unsafe {
            cxx_utils::make_grammar_compiler(
                tokenizer_info.ffi_ref(),
                cxx_int(max_threads),
                cache_enabled,
                cxx_longlong(cache_limit_bytes as i64),
                error_out_cxx.as_mut().get_unchecked_mut(),
            )
        };
        if inner.is_null() {
            return Err(error_out_cxx.to_string());
        }
        Ok(Self {
            inner,
        })
    }

    /// Get `CompiledGrammar` from the specified JSON schema and format. The indent
    /// and separators parameters follow the same convention as in `json.dumps()`.
    ///
    /// # Parameters
    ///
    /// - `schema`: The schema string.
    /// - `any_whitespace`: Whether to use any whitespace. If true, the generated grammar will
    ///   ignore the indent and separators parameters, and allow any whitespace.
    /// - `indent`: The number of spaces for indentation. If `None`, the output will be in one line.
    /// - `separators`: Two separators used in the schema: comma and colon. Examples: `(",", ":")`,
    ///   `(", ", ": ")`. If `None`, the default separators will be used: `(",", ": ")` when the
    ///   indent is not `None`, and `(", ", ": ")` otherwise.
    /// - `strict_mode`: Whether to use strict mode. In strict mode, the generated grammar will not
    ///   allow properties and items that is not specified in the schema. This is equivalent to
    ///   setting `unevaluatedProperties` and `unevaluatedItems` to false. This helps LLM to
    ///   generate accurate output in the grammar-guided generation with JSON schema.
    /// - `max_whitespace_cnt`: The maximum number of whitespace characters allowed between
    ///   elements, such like keys, values, separators and so on. If `None`, there is no limit
    ///   on the number of whitespace characters. If specified, it will limit the number of
    ///   whitespace characters to at most `max_whitespace_cnt`. It should be a positive integer.
    ///
    /// # Returns
    ///
    /// The compiled grammar.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON schema is invalid or compilation fails.
    pub fn compile_json_schema(
        &mut self,
        schema: &str,
        any_whitespace: bool,
        indent: Option<i32>,
        separators: Option<(impl AsRef<str>, impl AsRef<str>)>,
        strict_mode: bool,
        max_whitespace_cnt: Option<i32>,
    ) -> Result<CompiledGrammar, String> {
        cxx::let_cxx_string!(schema_cxx = schema);
        let has_indent = indent.is_some();
        let indent_i32: i32 = indent.unwrap_or(0);
        let has_separators = separators.is_some();
        let (sep_comma, sep_colon) = if let Some((comma, colon)) = separators {
            (comma.as_ref().to_string(), colon.as_ref().to_string())
        } else {
            (String::new(), String::new())
        };
        cxx::let_cxx_string!(sep_comma_cxx = sep_comma.as_str());
        cxx::let_cxx_string!(sep_colon_cxx = sep_colon.as_str());

        cxx::let_cxx_string!(error_out_cxx = "");
        let unique_ptr = unsafe {
            cxx_utils::compiler_compile_json_schema(
                self.inner.as_mut().expect("GrammarCompiler inner is null"),
                &schema_cxx,
                any_whitespace,
                has_indent,
                cxx_int(indent_i32),
                has_separators,
                &sep_comma_cxx,
                &sep_colon_cxx,
                strict_mode,
                max_whitespace_cnt.is_some(),
                cxx_int(max_whitespace_cnt.unwrap_or(0)),
                error_out_cxx.as_mut().get_unchecked_mut(),
            )
        };
        if unique_ptr.is_null() {
            return Err(error_out_cxx.to_string());
        }
        Ok(CompiledGrammar::from_unique_ptr(unique_ptr))
    }

    /// Get `CompiledGrammar` from the standard JSON.
    ///
    /// # Returns
    ///
    /// The compiled grammar.
    ///
    /// # Errors
    ///
    /// Returns an error if compilation fails.
    pub fn compile_builtin_json_grammar(
        &mut self
    ) -> Result<CompiledGrammar, String> {
        cxx::let_cxx_string!(error_out_cxx = "");
        let unique_ptr = unsafe {
            cxx_utils::compiler_compile_builtin_json(
                self.inner.as_mut().expect("GrammarCompiler inner is null"),
                error_out_cxx.as_mut().get_unchecked_mut(),
            )
        };
        if unique_ptr.is_null() {
            return Err(error_out_cxx.to_string());
        }
        Ok(CompiledGrammar::from_unique_ptr(unique_ptr))
    }

    /// Get `CompiledGrammar` from the specified regex.
    ///
    /// # Parameters
    ///
    /// - `regex`: The regex string.
    ///
    /// # Returns
    ///
    /// The compiled grammar.
    ///
    /// # Errors
    ///
    /// Returns an error if the regex is invalid or compilation fails.
    pub fn compile_regex(
        &mut self,
        regex: &str,
    ) -> Result<CompiledGrammar, String> {
        cxx::let_cxx_string!(regex_cxx = regex);
        cxx::let_cxx_string!(error_out_cxx = "");
        let unique_ptr = unsafe {
            cxx_utils::compiler_compile_regex(
                self.inner.as_mut().expect("GrammarCompiler inner is null"),
                &regex_cxx,
                error_out_cxx.as_mut().get_unchecked_mut(),
            )
        };
        if unique_ptr.is_null() {
            return Err(error_out_cxx.to_string());
        }
        Ok(CompiledGrammar::from_unique_ptr(unique_ptr))
    }

    /// Compile a grammar from a structural tag. See the Structural Tag Usage in XGrammar
    /// documentation for its usage.
    ///
    /// # Parameters
    ///
    /// - `tags`: The structural tags.
    /// - `triggers`: The triggers.
    ///
    /// # Returns
    ///
    /// The compiled grammar from the structural tag.
    ///
    /// # Errors
    ///
    /// Returns an error if the structural tag is invalid or compilation fails.
    pub fn compile_structural_tag(
        &mut self,
        tags: &[StructuralTagItem],
        triggers: &[impl AsRef<str>],
    ) -> Result<CompiledGrammar, String> {
        use serde_json::json;
        let mut tag_entries = Vec::new();
        for tag in tags {
            let schema_value: serde_json::Value =
                serde_json::from_str(&tag.schema).map_err(|e| {
                    format!("Invalid JSON schema in StructuralTagItem: {}", e)
                })?;
            let content = json!({
                "type": "json_schema",
                "json_schema": schema_value
            });
            tag_entries.push(json!({
                "type": "tag",
                "begin": tag.begin,
                "content": content,
                "end": tag.end,
            }));
        }
        let triggers_vec: Vec<String> =
            triggers.iter().map(|t| t.as_ref().to_string()).collect();
        let format_obj = json!({
            "type": "triggered_tags",
            "triggers": triggers_vec,
            "tags": tag_entries,
        });
        let structural_tag_json = json!({
            "type": "structural_tag",
            "format": format_obj,
        })
        .to_string();

        cxx::let_cxx_string!(structural_tag_str = structural_tag_json);
        cxx::let_cxx_string!(error_out_cxx = "");
        let unique_ptr = unsafe {
            cxx_utils::compiler_compile_structural_tag(
                self.inner.as_mut().expect("GrammarCompiler inner is null"),
                &structural_tag_str,
                error_out_cxx.as_mut().get_unchecked_mut(),
            )
        };
        if unique_ptr.is_null() {
            return Err(error_out_cxx.to_string());
        }
        Ok(CompiledGrammar::from_unique_ptr(unique_ptr))
    }

    /// Compile a grammar object.
    ///
    /// # Parameters
    ///
    /// - `grammar`: The grammar object.
    ///
    /// # Returns
    ///
    /// The compiled grammar.
    ///
    /// # Errors
    ///
    /// Returns an error if the grammar is invalid or compilation fails.
    pub fn compile_grammar(
        &mut self,
        grammar: &grammar::Grammar,
    ) -> Result<CompiledGrammar, String> {
        cxx::let_cxx_string!(error_out_cxx = "");
        let unique_ptr = unsafe {
            cxx_utils::compiler_compile_grammar_or_error(
                self.inner.as_mut().expect("GrammarCompiler inner is null"),
                grammar.ffi_ref(),
                error_out_cxx.as_mut().get_unchecked_mut(),
            )
        };
        if unique_ptr.is_null() {
            return Err(error_out_cxx.to_string());
        }
        Ok(CompiledGrammar::from_unique_ptr(unique_ptr))
    }

    /// Compile a grammar from an EBNF string. The string should follow the format described in
    /// <https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md>
    ///
    /// # Parameters
    ///
    /// - `ebnf_string`: The grammar string in EBNF format.
    /// - `root_rule_name`: The name of the root rule in the grammar.
    ///
    /// # Returns
    ///
    /// The compiled grammar.
    ///
    /// # Errors
    ///
    /// Returns an error if the EBNF string is invalid or compilation fails.
    pub fn compile_grammar_from_ebnf(
        &mut self,
        ebnf_string: &str,
        root_rule_name: &str,
    ) -> Result<CompiledGrammar, String> {
        let grammar = grammar::Grammar::from_ebnf(ebnf_string, root_rule_name)?;
        self.compile_grammar(&grammar)
    }

    /// Clear all cached compiled grammars.
    pub fn clear_cache(&mut self) {
        self.inner
            .as_mut()
            .expect("GrammarCompiler inner is null")
            .ClearCache();
    }

    /// The approximate memory usage of the cache in bytes.
    pub fn get_cache_size_bytes(&self) -> i64 {
        self.inner
            .as_ref()
            .expect("GrammarCompiler inner is null")
            .GetCacheSizeBytes()
            .into()
    }

    /// The maximum memory usage for the cache in bytes.
    ///
    /// # Returns
    ///
    /// The cache limit in bytes. Returns -1 if the cache has no memory limit.
    pub fn cache_limit_bytes(&self) -> i64 {
        self.inner
            .as_ref()
            .expect("GrammarCompiler inner is null")
            .CacheLimitBytes()
            .into()
    }
}

impl Drop for GrammarCompiler {
    fn drop(&mut self) {}
}
