//! Testing and debugging utilities, mirroring `xgrammar.testing`.
//!
//! Like their Python counterparts (whose names carry a leading underscore),
//! these functions are internal tools: useful for tests, debugging and
//! experiments, but not part of the stable API surface.

use tvm_ffi::{Array, String as FfiString};

use crate::bitmask::TokenBitmask;
use crate::compiler::CompiledGrammar;
use crate::error::Result;
use crate::ffi::{ffi_call, opt_any, ret};
use crate::grammar::{Grammar, JsonSchemaOptions};
use crate::structural_tag::JsonSchemaStyle;

impl JsonSchemaStyle {
    fn wire_name(self) -> &'static str {
        match self {
            JsonSchemaStyle::Json => "json",
            JsonSchemaStyle::QwenXml => "qwen_xml",
            JsonSchemaStyle::MinimaxXml => "minimax_xml",
            JsonSchemaStyle::DeepseekXml => "deepseek_xml",
            JsonSchemaStyle::GlmXml => "glm_xml",
        }
    }
}

/// Convert a JSON schema to EBNF text without building a grammar
/// (`testing._json_schema_to_ebnf`). `style` selects the output dialect
/// (JSON or a model-specific XML flavor).
pub fn json_schema_to_ebnf(
    schema: &str,
    options: &JsonSchemaOptions,
    style: JsonSchemaStyle,
) -> Result<String> {
    let separators = options.separators.as_ref().map(|(item, key)| {
        Array::<FfiString>::new(vec![FfiString::from(item), FfiString::from(key)])
    });
    let any = ffi_call!(
        "xgrammar.tvm_ffi_binding.testing._json_schema_to_ebnf",
        FfiString::from(schema),
        options.any_whitespace,
        opt_any(options.indent),
        opt_any(separators),
        options.strict_mode,
        opt_any(options.max_whitespace_cnt),
        FfiString::from(style.wire_name()),
        options.any_order,
    )?;
    Ok(ret::<FfiString>(any)?.as_str().to_string())
}

/// Convert a regex to EBNF text (`testing._regex_to_ebnf`).
pub fn regex_to_ebnf(regex: &str, with_rule_name: bool) -> Result<String> {
    let any = ffi_call!(
        "xgrammar.tvm_ffi_binding.testing._regex_to_ebnf",
        FfiString::from(regex),
        with_rule_name,
    )?;
    Ok(ret::<FfiString>(any)?.as_str().to_string())
}

/// Parse EBNF text into a grammar without running the normalization passes
/// (`testing._ebnf_to_grammar_no_normalization`).
pub fn ebnf_to_grammar_no_normalization(ebnf: &str, root_rule_name: &str) -> Result<Grammar> {
    let any = ffi_call!(
        "xgrammar.tvm_ffi_binding.testing._ebnf_to_grammar_no_normalization",
        FfiString::from(ebnf),
        FfiString::from(root_rule_name),
    )?;
    Ok(Grammar::from_raw(ret(any)?))
}

/// The token ids masked out in row `index` of `bitmask`, computed by the C++
/// side (`testing._get_masked_tokens_from_bitmask`). Equivalent to
/// [`TokenBitmask::masked_tokens`], useful for cross-checking.
pub fn get_masked_tokens_from_bitmask(bitmask: &TokenBitmask, index: usize) -> Result<Vec<i64>> {
    let shape = Array::<i64>::new(vec![
        bitmask.batch_size() as i64,
        crate::bitmask::bitmask_words(bitmask.vocab_size()) as i64,
    ]);
    let any = ffi_call!(
        "xgrammar.tvm_ffi_binding.testing._get_masked_tokens_from_bitmask",
        bitmask.as_slice().as_ptr() as i64,
        shape,
        bitmask.vocab_size() as i64,
        index as i64,
    )?;
    let array = ret::<Array<i64>>(any)?;
    Ok(array.iter().collect())
}

/// Whether row `index` of `bitmask` allows exactly one token
/// (`testing._is_single_token_bitmask`); returns `(is_single, token_id)`
/// with `token_id == -1` when not single.
pub fn is_single_token_bitmask(bitmask: &TokenBitmask, index: usize) -> Result<(bool, i64)> {
    let shape = Array::<i64>::new(vec![
        bitmask.batch_size() as i64,
        crate::bitmask::bitmask_words(bitmask.vocab_size()) as i64,
    ]);
    let any = ffi_call!(
        "xgrammar.tvm_ffi_binding.testing._is_single_token_bitmask",
        bitmask.as_slice().as_ptr() as i64,
        shape,
        bitmask.vocab_size() as i64,
        index as i64,
    )?;
    let array = ret::<Array<i64>>(any)?;
    let values: Vec<i64> = array.iter().collect();
    Ok((values[0] != 0, values[1]))
}

/// The rule ids of rules that can match the empty string
/// (`testing._get_allow_empty_rule_ids`).
pub fn get_allow_empty_rule_ids(compiled_grammar: &CompiledGrammar) -> Result<Vec<i64>> {
    let any = ffi_call!(
        "xgrammar.tvm_ffi_binding.testing._get_allow_empty_rule_ids",
        compiled_grammar.raw,
    )?;
    let array = ret::<Array<i64>>(any)?;
    Ok(array.iter().collect())
}

/// A regex matching integers in `[start, end]`; `None` bounds are unbounded
/// (`testing._generate_range_regex`).
pub fn generate_range_regex(start: Option<i64>, end: Option<i64>) -> Result<String> {
    let any = ffi_call!(
        "xgrammar.tvm_ffi_binding.testing._generate_range_regex",
        opt_any(start),
        opt_any(end),
    )?;
    Ok(ret::<FfiString>(any)?.as_str().to_string())
}

/// A regex matching floats in the given range
/// (`testing._generate_float_regex`).
pub fn generate_float_regex(
    start: Option<f64>,
    end: Option<f64>,
    exclusive_start: bool,
    exclusive_end: bool,
) -> Result<String> {
    let any = ffi_call!(
        "xgrammar.tvm_ffi_binding.testing._generate_float_regex",
        opt_any(start),
        opt_any(end),
        exclusive_start,
        exclusive_end,
    )?;
    Ok(ret::<FfiString>(any)?.as_str().to_string())
}

/// A human-readable dump of the FSMs built from the grammar
/// (`testing._print_grammar_fsms`).
///
/// The grammar must come from [`CompiledGrammar::grammar`]: only compilation
/// populates the per-rule FSM table, and (as in the Python API) passing an
/// uncompiled grammar crashes the C++ side.
pub fn print_grammar_fsms(grammar: &Grammar) -> Result<String> {
    let any = ffi_call!(
        "xgrammar.tvm_ffi_binding.testing._print_grammar_fsms",
        grammar.raw,
    )?;
    Ok(ret::<FfiString>(any)?.as_str().to_string())
}

/// The grammar transformation passes exposed for testing
/// (`xgrammar.testing.GrammarFunctor`).
pub mod grammar_functor {
    use super::*;

    macro_rules! def_pass {
        ($(#[$doc:meta])* $name:ident) => {
            $(#[$doc])*
            pub fn $name(grammar: &Grammar) -> Result<Grammar> {
                let any = ffi_call!(
                    concat!(
                        "xgrammar.tvm_ffi_binding.testing.grammar_functor.",
                        stringify!($name)
                    ),
                    grammar.raw,
                )?;
                Ok(Grammar::from_raw(ret(any)?))
            }
        };
    }

    def_pass!(
        /// Normalize the grammar structure.
        structure_normalizer
    );
    def_pass!(
        /// Fuse consecutive byte strings.
        byte_string_fuser
    );
    def_pass!(
        /// Inline short rules.
        rule_inliner
    );
    def_pass!(
        /// Remove unreachable rules.
        dead_code_eliminator
    );
    def_pass!(
        /// Analyze lookahead assertions.
        lookahead_assertion_analyzer
    );
    def_pass!(
        /// The full grammar optimization pipeline.
        grammar_optimizer
    );
    def_pass!(
        /// Normalize repetition ranges.
        repetition_normalizer
    );
}
