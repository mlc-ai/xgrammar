//! Token-level grammar matching: [`GrammarMatcher`] and
//! [`BatchGrammarMatcher`].

use std::marker::PhantomData;

use tvm_ffi::tvm_ffi_sys::TVMFFIAny;
use tvm_ffi::{Any, AnyCompatible, Array, Bytes as FfiBytes, String as FfiString};

use crate::bitmask::TokenBitmask;
use crate::compiler::CompiledGrammar;
use crate::error::{Error, Result};
use crate::ffi::objects::{RawBatchGrammarMatcher, RawGrammarMatcher};
use crate::ffi::{ffi_call, mixed_array, opt_any, ret, ret_opt, DlArg};

/// Options for [`GrammarMatcher::with_options`], mirroring the Python
/// constructor's keyword arguments.
#[derive(Debug, Clone, Default)]
pub struct MatcherOptions {
    /// Override the stop tokens from the tokenizer info.
    pub override_stop_tokens: Option<Vec<i64>>,
    /// Consider the matcher terminated once the grammar is completed, without
    /// requiring a stop token.
    pub terminate_without_stop_token: bool,
    /// The default sampling temperature reported by
    /// [`GrammarMatcher::temperature`].
    pub default_temperature: Option<f64>,
}

/// Matches tokens against a [`CompiledGrammar`] and produces token bitmasks
/// for constrained decoding.
///
/// A matcher is a mutable state machine and is `Send` but not `Sync`; use
/// [`GrammarMatcher::fork`] to branch the state (e.g. for beam search).
pub struct GrammarMatcher {
    pub(crate) raw: RawGrammarMatcher,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

impl GrammarMatcher {
    fn from_raw(raw: RawGrammarMatcher) -> Self {
        Self {
            raw,
            _not_sync: PhantomData,
        }
    }

    /// Create a matcher with default options.
    pub fn new(compiled_grammar: &CompiledGrammar) -> Result<Self> {
        Self::with_options(compiled_grammar, &MatcherOptions::default())
    }

    /// Create a matcher with explicit options.
    pub fn with_options(
        compiled_grammar: &CompiledGrammar,
        options: &MatcherOptions,
    ) -> Result<Self> {
        let override_stop = options
            .override_stop_tokens
            .as_ref()
            .map(|ids| Array::<i64>::new(ids.clone()));
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.__ffi_init__",
            compiled_grammar.raw,
            opt_any(override_stop),
            options.terminate_without_stop_token,
            -1i64, // max_rollback_tokens: unlimited (deprecated parameter)
            opt_any(options.default_temperature),
        )?;
        Ok(Self::from_raw(ret(any)?))
    }

    /// Accept one token and update the matcher state. Returns `false` (and
    /// leaves the state unchanged) when the token is not allowed by the
    /// grammar.
    pub fn accept_token(&mut self, token_id: i64) -> Result<bool> {
        self.accept_token_with_debug(token_id, false)
    }

    /// Like [`GrammarMatcher::accept_token`], optionally printing debug
    /// information about the acceptance to stderr.
    pub fn accept_token_with_debug(&mut self, token_id: i64, debug_print: bool) -> Result<bool> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.accept_token",
            self.raw,
            token_id,
            debug_print,
        )?;
        ret(any)
    }

    /// Accept a string (raw bytes) and update the matcher state. Returns
    /// `false` (and leaves the state unchanged) when the string is not
    /// accepted by the grammar. Intended mainly for jump-forward decoding and
    /// testing.
    pub fn accept_string(&mut self, input: impl AsRef<[u8]>) -> Result<bool> {
        self.accept_string_with_debug(input, false)
    }

    /// Like [`GrammarMatcher::accept_string`], optionally printing debug
    /// information to stderr.
    pub fn accept_string_with_debug(
        &mut self,
        input: impl AsRef<[u8]>,
        debug_print: bool,
    ) -> Result<bool> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.accept_string",
            self.raw,
            FfiBytes::from(input.as_ref()),
            debug_print,
        )?;
        ret(any)
    }

    /// Fill row `index` of `bitmask` with the tokens allowed next. Returns
    /// `true` when the row was written; `false` means every token is allowed
    /// and the row was left untouched (skip applying it to the logits).
    pub fn fill_next_token_bitmask(
        &mut self,
        bitmask: &mut TokenBitmask,
        index: usize,
    ) -> Result<bool> {
        self.fill_next_token_bitmask_with_debug(bitmask, index, false)
    }

    /// Like [`GrammarMatcher::fill_next_token_bitmask`], optionally printing
    /// debug information to stderr.
    pub fn fill_next_token_bitmask_with_debug(
        &mut self,
        bitmask: &mut TokenBitmask,
        index: usize,
        debug_print: bool,
    ) -> Result<bool> {
        let dl = bitmask.dl_arg();
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.fill_next_token_bitmask",
            self.raw,
            *dl.any(),
            index as i64,
            debug_print,
        )?;
        ret(any)
    }

    /// Match a draft tree of speculative tokens against the grammar, filling
    /// one bitmask row per draft token.
    ///
    /// `retrieve_next_token[i]` / `retrieve_next_sibling[i]` encode the tree
    /// (`-1` for none); `draft_tokens` are the token ids. `bitmask` must have
    /// at least `draft_tokens.len()` rows. `time_threshold` (seconds) bounds
    /// the traversal time (`-1.0` for unlimited); the return value is `false`
    /// when the traversal timed out. `temperatures`, when given, receives one
    /// temperature per draft token.
    pub fn traverse_draft_tree(
        &mut self,
        retrieve_next_token: &[i64],
        retrieve_next_sibling: &[i64],
        draft_tokens: &[i64],
        bitmask: &mut TokenBitmask,
        time_threshold: f64,
        temperatures: Option<&mut [f32]>,
    ) -> Result<bool> {
        let next_token = DlArg::int64_readonly(retrieve_next_token);
        let next_sibling = DlArg::int64_readonly(retrieve_next_sibling);
        let tokens = DlArg::int64_readonly(draft_tokens);
        let mask = bitmask.dl_arg();
        let temps = temperatures.map(DlArg::float32);
        let temps_any = match &temps {
            Some(dl) => dl.any().clone(),
            None => Any::new(),
        };
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.traverse_draft_tree",
            self.raw,
            *next_token.any(),
            *next_sibling.any(),
            *tokens.any(),
            *mask.any(),
            time_threshold,
            temps_any,
        )?;
        ret(any)
    }

    /// The longest string that is certainly the next output under the
    /// grammar, used for jump-forward decoding. Empty when nothing is certain.
    pub fn find_jump_forward_string(&mut self) -> Result<String> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.find_jump_forward_string",
            self.raw,
        )?;
        Ok(ret::<FfiString>(any)?.as_str().to_string())
    }

    /// Roll back the last `num_tokens` accepted tokens.
    pub fn rollback(&mut self, num_tokens: usize) -> Result<()> {
        ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.rollback",
            self.raw,
            num_tokens as i64,
        )?;
        Ok(())
    }

    /// The values captured by the grammar's capture groups so far, as
    /// `(name, value)` pairs in capture order. Values are raw bytes. When
    /// `deduplicate` is set, only the last capture of each name is kept
    /// (matching Python's default).
    pub fn get_captures(&self, deduplicate: bool) -> Result<Vec<(String, Vec<u8>)>> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.get_captures",
            self.raw,
            deduplicate,
        )?;
        // The result is Array<Array<[String, Bytes]>>, which cannot be
        // expressed as a typed `Array<T>`: walk the array objects directly
        // (layout fixed by the pinned tvm-ffi version).
        unsafe {
            let outer = array_elements(&any).ok_or_else(|| {
                Error::XGrammar("get_captures: unexpected return type".to_string())
            })?;
            let mut captures = Vec::with_capacity(outer.len());
            for entry_any in outer {
                let entry = borrow_owned_any(entry_any);
                let inner = array_elements(&entry).ok_or_else(|| {
                    Error::XGrammar("get_captures: entry is not an array".to_string())
                })?;
                if inner.len() != 2 {
                    return Err(Error::XGrammar(
                        "get_captures: entry is not a (name, value) pair".to_string(),
                    ));
                }
                let name = borrow_as::<FfiString>(&inner[0]).ok_or_else(|| {
                    Error::XGrammar("get_captures: capture name is not a string".to_string())
                })?;
                let value = borrow_as::<FfiBytes>(&inner[1]).ok_or_else(|| {
                    Error::XGrammar("get_captures: capture value is not bytes".to_string())
                })?;
                captures.push((name.as_str().to_string(), value.as_slice().to_vec()));
            }
            Ok(captures)
        }
    }

    /// Fork the matcher, duplicating its current state.
    pub fn fork(&self) -> Result<Self> {
        let any = ffi_call!("xgrammar.tvm_ffi_binding.GrammarMatcher.fork", self.raw)?;
        Ok(Self::from_raw(ret(any)?))
    }

    /// Whether the matcher has terminated (grammar completed and, unless
    /// configured otherwise, a stop token accepted).
    pub fn is_terminated(&self) -> Result<bool> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.is_terminated",
            self.raw,
        )?;
        ret(any)
    }

    /// Whether the matched input so far completes the grammar.
    pub fn is_completed(&self) -> Result<bool> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.is_completed",
            self.raw,
        )?;
        ret(any)
    }

    /// Reset the matcher to the initial state.
    pub fn reset(&mut self) -> Result<()> {
        ffi_call!("xgrammar.tvm_ffi_binding.GrammarMatcher.reset", self.raw)?;
        Ok(())
    }

    /// The maximum number of rollback tokens (`-1` means unlimited).
    pub fn max_rollback_tokens(&self) -> Result<i64> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.max_rollback_tokens",
            self.raw,
        )?;
        ret(any)
    }

    /// The sampling temperature for the current matcher position, when the
    /// grammar (or the matcher options) specifies one.
    pub fn temperature(&self) -> Result<Option<f64>> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.temperature",
            self.raw,
        )?;
        ret_opt(any)
    }

    /// The stop token ids the matcher uses.
    pub fn stop_token_ids(&self) -> Result<Vec<i64>> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher.stop_token_ids",
            self.raw,
        )?;
        let array = ret::<Array<i64>>(any)?;
        Ok(array.iter().collect())
    }

    /// A human-readable dump of the matcher's internal state (debugging aid).
    pub fn debug_print_internal_state(&self) -> Result<String> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.GrammarMatcher._debug_print_internal_state",
            self.raw,
        )?;
        Ok(ret::<FfiString>(any)?.as_str().to_string())
    }
}

impl std::fmt::Debug for GrammarMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GrammarMatcher({:?})", self.raw)
    }
}

/// Thread count for [`BatchGrammarMatcher::new`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MaxThreads {
    /// Let xgrammar pick based on the hardware (Python's `"auto"`).
    #[default]
    Auto,
    /// A fixed number of threads.
    Count(usize),
}

/// Runs matcher operations over batches of [`GrammarMatcher`]s, using an
/// internal thread pool for bitmask filling.
pub struct BatchGrammarMatcher {
    raw: RawBatchGrammarMatcher,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

impl BatchGrammarMatcher {
    /// Create a batch matcher with the given thread count.
    pub fn new(max_threads: MaxThreads) -> Result<Self> {
        let threads = match max_threads {
            MaxThreads::Auto => Any::from(FfiString::from("auto")),
            MaxThreads::Count(n) => Any::from(n as i64),
        };
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.BatchGrammarMatcher.__ffi_init__",
            threads,
        )?;
        Ok(Self {
            raw: ret(any)?,
            _not_sync: PhantomData,
        })
    }

    /// Fill the next-token bitmask for a batch of matchers in parallel.
    ///
    /// Without `indices`, matcher `i` fills bitmask row `i`. With `indices`,
    /// matcher `i` fills row `indices[i]`.
    pub fn batch_fill_next_token_bitmask(
        &mut self,
        matchers: &mut [GrammarMatcher],
        bitmask: &mut TokenBitmask,
        indices: Option<&[i64]>,
    ) -> Result<()> {
        self.batch_fill_next_token_bitmask_with_debug(matchers, bitmask, indices, false)
    }

    /// Like [`BatchGrammarMatcher::batch_fill_next_token_bitmask`],
    /// optionally printing debug information to stderr.
    pub fn batch_fill_next_token_bitmask_with_debug(
        &mut self,
        matchers: &mut [GrammarMatcher],
        bitmask: &mut TokenBitmask,
        indices: Option<&[i64]>,
        debug_print: bool,
    ) -> Result<()> {
        let matcher_array = matcher_array(matchers);
        let dl = bitmask.dl_arg();
        let indices_arr = indices.map(|ids| Array::<i64>::new(ids.to_vec()));
        ffi_call!(
            "xgrammar.tvm_ffi_binding.BatchGrammarMatcher.batch_fill_next_token_bitmask",
            self.raw,
            matcher_array,
            *dl.any(),
            opt_any(indices_arr),
            debug_print,
        )?;
        Ok(())
    }

    /// Write each matcher's current temperature into `temperatures`
    /// (`f32::NAN`-safe: rows without a temperature are left untouched).
    /// Without `indices`, matcher `i` writes `temperatures[i]`; with
    /// `indices`, matcher `i` writes `temperatures[indices[i]]`.
    pub fn batch_fill_temperature(
        matchers: &mut [GrammarMatcher],
        temperatures: &mut [f32],
        indices: Option<&[i64]>,
    ) -> Result<()> {
        let matcher_arr = matcher_array(matchers);
        let dl = DlArg::float32(temperatures);
        let indices_arr = indices.map(|ids| Array::<i64>::new(ids.to_vec()));
        ffi_call!(
            "xgrammar.tvm_ffi_binding.BatchGrammarMatcher.batch_fill_temperature",
            matcher_arr,
            *dl.any(),
            opt_any(indices_arr),
        )?;
        Ok(())
    }

    /// Accept one token per matcher. Returns per-matcher acceptance.
    pub fn batch_accept_token(
        matchers: &mut [GrammarMatcher],
        token_ids: &[i64],
    ) -> Result<Vec<bool>> {
        Self::batch_accept_token_with_debug(matchers, token_ids, false)
    }

    /// Like [`BatchGrammarMatcher::batch_accept_token`], optionally printing
    /// debug information to stderr.
    pub fn batch_accept_token_with_debug(
        matchers: &mut [GrammarMatcher],
        token_ids: &[i64],
        debug_print: bool,
    ) -> Result<Vec<bool>> {
        let matcher_arr = matcher_array(matchers);
        let tokens = Array::<i64>::new(token_ids.to_vec());
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.BatchGrammarMatcher.batch_accept_token",
            matcher_arr,
            tokens,
            debug_print,
        )?;
        let array = ret::<Array<i64>>(any)?;
        Ok(array.iter().map(|v| v != 0).collect())
    }

    /// Accept one string (raw bytes) per matcher. Returns per-matcher
    /// acceptance.
    pub fn batch_accept_string<S: AsRef<[u8]>>(
        matchers: &mut [GrammarMatcher],
        inputs: &[S],
    ) -> Result<Vec<bool>> {
        Self::batch_accept_string_with_debug(matchers, inputs, false)
    }

    /// Like [`BatchGrammarMatcher::batch_accept_string`], optionally printing
    /// debug information to stderr.
    pub fn batch_accept_string_with_debug<S: AsRef<[u8]>>(
        matchers: &mut [GrammarMatcher],
        inputs: &[S],
        debug_print: bool,
    ) -> Result<Vec<bool>> {
        let matcher_arr = matcher_array(matchers);
        let input_anys: Vec<Any> = inputs
            .iter()
            .map(|s| Any::from(FfiBytes::from(s.as_ref())))
            .collect();
        let input_views: Vec<tvm_ffi::AnyView> =
            input_anys.iter().map(tvm_ffi::AnyView::from).collect();
        let strings = mixed_array(&input_views)?;
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.BatchGrammarMatcher.batch_accept_string",
            matcher_arr,
            strings,
            debug_print,
        )?;
        let array = ret::<Array<i64>>(any)?;
        Ok(array.iter().map(|v| v != 0).collect())
    }

    /// Roll back `num_tokens[i]` tokens on matcher `i`.
    pub fn batch_rollback(matchers: &mut [GrammarMatcher], num_tokens: &[i64]) -> Result<()> {
        let matcher_arr = matcher_array(matchers);
        let nums = Array::<i64>::new(num_tokens.to_vec());
        ffi_call!(
            "xgrammar.tvm_ffi_binding.BatchGrammarMatcher.batch_rollback",
            matcher_arr,
            nums,
        )?;
        Ok(())
    }
}

impl std::fmt::Debug for BatchGrammarMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BatchGrammarMatcher({:?})", self.raw)
    }
}

fn matcher_array(matchers: &mut [GrammarMatcher]) -> Array<RawGrammarMatcher> {
    Array::new(matchers.iter().map(|m| m.raw.clone()).collect())
}

// ---- raw array walking for `get_captures` ----
// The nested Array<Array<[String, Bytes]>> return cannot be expressed as a
// typed `Array<T>` at tvm-ffi v0.1.12 (`Any` is not `AnyCompatible`), so the
// elements are read through `ArrayObj`'s public layout, exactly like
// tvm-ffi's own `collections::array` iterator does.

/// Borrow the elements of an `ffi.Array` held by `any`.
unsafe fn array_elements(any: &Any) -> Option<&[TVMFFIAny]> {
    use tvm_ffi::collections::array::ArrayObj;
    use tvm_ffi::TypeIndex;
    // Clone the Any to inspect the raw payload without consuming the input.
    let raw = Any::into_raw_ffi_any(any.clone());
    if raw.type_index != TypeIndex::kTVMFFIArray as i32 {
        return None;
    }
    // The clone's reference is released below; the caller's `any` keeps the
    // array alive while the returned slice is in use.
    let obj = raw.data_union.v_obj as *const ArrayObj;
    let _drop_clone = Any::from_raw_ffi_any(raw);
    let elements =
        std::slice::from_raw_parts((*obj).data as *const TVMFFIAny, (*obj).size as usize);
    Some(elements)
}

/// Copy a borrowed element into an owned `Any` (bumping its refcount).
unsafe fn borrow_owned_any(element: &TVMFFIAny) -> Any {
    let tmp = std::mem::ManuallyDrop::new(Any::from_raw_ffi_any(std::ptr::read(element)));
    (*tmp).clone()
}

/// Read a borrowed element as `T` without touching its refcount.
unsafe fn borrow_as<T: AnyCompatible>(element: &TVMFFIAny) -> Option<T> {
    let tmp = std::mem::ManuallyDrop::new(Any::from_raw_ffi_any(std::ptr::read(element)));
    tmp.try_as::<T>()
}
