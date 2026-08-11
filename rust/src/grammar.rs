//! The [`Grammar`] type and its factory functions.

use std::fmt;

use tvm_ffi::{Array, String as FfiString};

use crate::error::Result;
use tvm_ffi::object::ObjectRef;

use crate::ffi::{ffi_call, handle, opt_any, ret};
use crate::tokenizer_info::TokenizerInfo;

/// Options for converting a JSON schema into a grammar.
///
/// The defaults match the Python API (`Grammar.from_json_schema`).
#[derive(Debug, Clone)]
pub struct JsonSchemaOptions {
    /// Allow arbitrary whitespace between JSON tokens. When `true` (default),
    /// `indent` and `separators` are ignored.
    pub any_whitespace: bool,
    /// Number of spaces per indentation level; `None` means no line breaks.
    pub indent: Option<i64>,
    /// The `(item, key)` separators, as in Python's `json.dumps`. `None`
    /// picks `(", ", ": ")` when `indent` is `None` and `(",", ": ")`
    /// otherwise.
    pub separators: Option<(String, String)>,
    /// Reject unspecified properties/items (see the Python docs for details).
    pub strict_mode: bool,
    /// Maximum number of consecutive whitespace characters; `None` means
    /// unlimited. Only meaningful with `any_whitespace`.
    pub max_whitespace_cnt: Option<i64>,
    /// Print the converted EBNF grammar to stdout (debugging aid).
    pub print_converted_ebnf: bool,
    /// Accept object properties in any order.
    pub any_order: bool,
}

impl Default for JsonSchemaOptions {
    fn default() -> Self {
        Self {
            any_whitespace: true,
            indent: None,
            separators: None,
            strict_mode: true,
            max_whitespace_cnt: None,
            print_converted_ebnf: false,
            any_order: false,
        }
    }
}

/// A named grammar referenced from a Lark source (see [`Grammar::from_lark_with`]).
#[derive(Debug, Clone, Copy)]
pub enum NamedGrammar<'a> {
    /// An already constructed grammar.
    Grammar(&'a Grammar),
    /// A Lark grammar source with its own `start` rule.
    Lark(&'a str),
}

/// A context-free grammar in xgrammar's internal representation.
///
/// This class is immutable: instances can be cloned cheaply (shared handle)
/// and shared across threads. Construct one with the `from_*` factory
/// functions, then compile it with [`crate::GrammarCompiler`].
#[derive(Clone)]
pub struct Grammar {
    pub(crate) raw: ObjectRef,
}

impl Grammar {
    pub(crate) fn from_raw(raw: ObjectRef) -> Self {
        Self { raw }
    }

    /// Create a grammar from EBNF text, using `root` as the root rule.
    ///
    /// The EBNF string should follow the format described in
    /// <https://xgrammar.mlc.ai/docs/tutorials/ebnf_guided_generation.html>.
    pub fn from_ebnf(ebnf: &str) -> Result<Self> {
        Self::from_ebnf_with_root(ebnf, "root")
    }

    /// Create a grammar from EBNF text with an explicit root rule name.
    pub fn from_ebnf_with_root(ebnf: &str, root_rule_name: &str) -> Result<Self> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.Grammar.from_ebnf",
            FfiString::from(ebnf),
            FfiString::from(root_rule_name),
        )?;
        Ok(Self::from_raw(ret(any)?))
    }

    /// Create a grammar that matches JSON documents following a JSON schema.
    ///
    /// Use `&JsonSchemaOptions::default()` for the same behavior as the
    /// Python API's defaults.
    pub fn from_json_schema(schema: &str, options: &JsonSchemaOptions) -> Result<Self> {
        let separators = options.separators.as_ref().map(|(item, key)| {
            Array::<FfiString>::new(vec![FfiString::from(item), FfiString::from(key)])
        });
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.Grammar.from_json_schema",
            FfiString::from(schema),
            options.any_whitespace,
            opt_any(options.indent),
            opt_any(separators),
            options.strict_mode,
            opt_any(options.max_whitespace_cnt),
            options.print_converted_ebnf,
            options.any_order,
        )?;
        Ok(Self::from_raw(ret(any)?))
    }

    /// Create a grammar that matches strings accepted by a regular expression.
    ///
    /// The regex follows Python `re` syntax; see the Python documentation of
    /// `Grammar.from_regex` for the supported subset.
    pub fn from_regex(regex: &str) -> Result<Self> {
        Self::from_regex_with(regex, false)
    }

    /// Like [`Grammar::from_regex`], optionally printing the converted EBNF
    /// to stdout (debugging aid).
    pub fn from_regex_with(regex: &str, print_converted_ebnf: bool) -> Result<Self> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.Grammar.from_regex",
            FfiString::from(regex),
            print_converted_ebnf,
        )?;
        Ok(Self::from_raw(ret(any)?))
    }

    /// Create a grammar from Lark syntax. The grammar must define a `start`
    /// rule.
    pub fn from_lark(lark: &str) -> Result<Self> {
        Self::from_lark_with(lark, None, &[])
    }

    /// Create a grammar from Lark syntax, with tokenizer metadata for
    /// resolving named special tokens and/or named grammars referenced by
    /// `@name` in the source (names are given without the leading `@`).
    pub fn from_lark_with(
        lark: &str,
        tokenizer_info: Option<&TokenizerInfo>,
        named_grammars: &[(&str, NamedGrammar<'_>)],
    ) -> Result<Self> {
        let names = Array::<FfiString>::new(
            named_grammars
                .iter()
                .map(|(name, _)| FfiString::from(name))
                .collect(),
        );
        let value_anys: Vec<tvm_ffi::Any> = named_grammars
            .iter()
            .map(|(_, value)| match value {
                NamedGrammar::Grammar(g) => handle::object_arg(&g.raw),
                NamedGrammar::Lark(src) => tvm_ffi::Any::from(FfiString::from(src)),
            })
            .collect();
        let value_views: Vec<tvm_ffi::AnyView> =
            value_anys.iter().map(tvm_ffi::AnyView::from).collect();
        let values = crate::ffi::mixed_array(&value_views)?;
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.Grammar.from_lark",
            FfiString::from(lark),
            opt_any(tokenizer_info.map(|t| t.raw.clone())),
            names,
            values,
        )?;
        Ok(Self::from_raw(ret(any)?))
    }

    /// Create a grammar from a structural tag; see [`crate::StructuralTag`].
    pub fn from_structural_tag(structural_tag: &crate::StructuralTag) -> Result<Self> {
        Self::from_structural_tag_json(&structural_tag.to_json()?)
    }

    /// Create a grammar from a serialized structural tag (JSON text).
    ///
    /// Prefer [`Grammar::from_structural_tag`] with the typed
    /// [`crate::StructuralTag`] model; this function is the raw escape hatch
    /// accepting the JSON wire format directly.
    pub fn from_structural_tag_json(structural_tag_json: &str) -> Result<Self> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.Grammar.from_structural_tag",
            FfiString::from(structural_tag_json),
        )?;
        Ok(Self::from_raw(ret(any)?))
    }

    /// The built-in grammar accepting any valid JSON document.
    pub fn builtin_json_grammar() -> Result<Self> {
        let any = ffi_call!("xgrammar.tvm_ffi_binding.Grammar.builtin_json_grammar")?;
        Ok(Self::from_raw(ret(any)?))
    }

    /// A grammar matching any input accepted by at least one of `grammars`.
    pub fn union(grammars: &[Grammar]) -> Result<Self> {
        let array = Array::<ObjectRef>::new(grammars.iter().map(|g| g.raw.clone()).collect());
        let any = ffi_call!("xgrammar.tvm_ffi_binding.Grammar.union", array)?;
        Ok(Self::from_raw(ret(any)?))
    }

    /// A grammar matching the concatenation of inputs of `grammars`.
    pub fn concat(grammars: &[Grammar]) -> Result<Self> {
        let array = Array::<ObjectRef>::new(grammars.iter().map(|g| g.raw.clone()).collect());
        let any = ffi_call!("xgrammar.tvm_ffi_binding.Grammar.concat", array)?;
        Ok(Self::from_raw(ret(any)?))
    }

    /// Serialize the grammar to JSON.
    pub fn serialize_json(&self) -> Result<String> {
        let any = ffi_call!("xgrammar.tvm_ffi_binding.Grammar.serialize_json", self.raw,)?;
        Ok(ret::<FfiString>(any)?.as_str().to_string())
    }

    /// Deserialize a grammar from the JSON produced by
    /// [`Grammar::serialize_json`].
    ///
    /// Returns [`crate::Error::DeserializeVersion`] when the data was written
    /// by an incompatible xgrammar version and
    /// [`crate::Error::DeserializeFormat`]/[`crate::Error::InvalidJson`] when
    /// it is malformed.
    pub fn deserialize_json(json: &str) -> Result<Self> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.Grammar.deserialize_json",
            FfiString::from(json),
        )?;
        Ok(Self::from_raw(ret(any)?))
    }

    fn to_ebnf_string(&self) -> Result<String> {
        let any = ffi_call!("xgrammar.tvm_ffi_binding.Grammar.to_string", self.raw)?;
        Ok(ret::<FfiString>(any)?.as_str().to_string())
    }
}

/// Formats the grammar as EBNF text, like `str(grammar)` in Python.
impl fmt::Display for Grammar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_ebnf_string().map_err(|_| fmt::Error)?)
    }
}

impl fmt::Debug for Grammar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Grammar({:p})", handle::handle_ptr(&self.raw))
    }
}

/// Handle identity, like the Python `Grammar.__eq__`.
impl PartialEq for Grammar {
    fn eq(&self, other: &Self) -> bool {
        handle::same_handle(&self.raw, &other.raw)
    }
}
impl Eq for Grammar {}

// SAFETY: the underlying C++ object is immutable after construction, and the
// handle is a reference-counted pointer with atomic counts.
unsafe impl Send for Grammar {}
unsafe impl Sync for Grammar {}
