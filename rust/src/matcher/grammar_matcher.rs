use std::pin::Pin;

use autocxx::prelude::*;

use crate::{
    CxxUniquePtr, DLTensor, FFIGrammarMatcher, compiler::CompiledGrammar,
    cxx_int, cxx_utils,
};

/// Match the output of the LLM to the specified grammar, then generate the mask for the next
/// token. This is the core class in the grammar-guided generation.
///
/// This class maintains a stateful matcher that can accept tokens and strings, then match them
/// to the specified grammar. The matcher can provide a bitmask for the next token prediction,
/// so that the output of the LLM follows the specified grammar. Its state can be reset and
/// rolled back by tokens. It also provides utilities for jump-forward decoding.
///
/// After matching the whole grammar, the matcher will accept a stop token. The token mask at
/// this time will only allow stop tokens. After accepting the stop token, the matcher will
/// terminate, then it cannot accept any new token or generate a new token mask, meaning the
/// generation is finished.
///
/// Under the hood, it utilizes a pushdown automaton with backtracking to match the grammar,
/// with optimizations specific to LLM token mask generation.
pub struct GrammarMatcher {
    inner: CxxUniquePtr<FFIGrammarMatcher>,
    stored_stop_token_ids: Box<[i32]>,
}

impl GrammarMatcher {
    /// Construct the grammar matcher.
    ///
    /// # Parameters
    ///
    /// - `compiled_grammar`: The initialization context for the grammar matcher.
    /// - `override_stop_tokens`: If not `None`, the stop tokens to override the ones in
    ///   the grammar.
    /// - `terminate_without_stop_token`: Whether to terminate the matcher without accepting
    ///   a stop token.
    /// - `max_rollback_tokens`: Deprecated. You don't need to set it and it's always unlimited
    ///   (-1). The new Earley parser significantly reduces the number of states, so we can
    ///   allow unlimited rollback.
    ///
    /// # Errors
    ///
    /// Returns an error if the grammar matcher cannot be constructed.
    pub fn new(
        compiled_grammar: &CompiledGrammar,
        override_stop_tokens: Option<&[i32]>,
        terminate_without_stop_token: bool,
        max_rollback_tokens: i32,
    ) -> Result<Self, String> {
        let stored_stop_token_ids: Box<[i32]> = match override_stop_tokens {
            Some(slice) => slice.to_vec().into_boxed_slice(),
            None => compiled_grammar.tokenizer_info().stop_token_ids(),
        };
        let (has_override, ptr, len) = match override_stop_tokens {
            Some(slice) if !slice.is_empty() => {
                (true, slice.as_ptr(), slice.len())
            },
            _ => (false, std::ptr::null(), 0usize),
        };

        cxx::let_cxx_string!(error_out_cxx = "");
        let unique_ptr = unsafe {
            cxx_utils::make_grammar_matcher(
                compiled_grammar.ffi_ref(),
                has_override,
                ptr,
                len,
                terminate_without_stop_token,
                cxx_int(max_rollback_tokens),
                error_out_cxx.as_mut().get_unchecked_mut(),
            )
        };
        if unique_ptr.is_null() {
            return Err(error_out_cxx.to_string());
        }
        Ok(Self {
            inner: unique_ptr,
            stored_stop_token_ids,
        })
    }

    /// Accept one token and update the state of the matcher.
    ///
    /// In the following cases, the matcher will not accept the token and return false:
    /// 1. The token does not match the grammar.
    /// 2. The matcher has terminated after accepting the stop token, but is trying to accept
    ///    a new token.
    /// 3. The token id is out of range.
    /// 4. The token is a special token.
    ///
    /// The user should capture the return value and handle the cases where the token is not
    /// accepted.
    ///
    /// # Parameters
    ///
    /// - `token_id`: The id of the token to accept.
    ///
    /// # Returns
    ///
    /// Whether the token is accepted.
    pub fn accept_token(
        &mut self,
        token_id: i32,
    ) -> bool {
        self.inner
            .as_mut()
            .expect("GrammarMatcher inner is null")
            .AcceptToken(token_id, false)
    }

    /// Accept one token with optional debug printing.
    ///
    /// # Parameters
    ///
    /// - `token_id`: The id of the token to accept.
    /// - `debug_print`: Whether to print information about the internal state of the matcher.
    ///   Helpful for debugging.
    ///
    /// # Returns
    ///
    /// Whether the token is accepted.
    pub fn accept_token_with_debug(
        &mut self,
        token_id: i32,
        debug_print: bool,
    ) -> bool {
        self.inner
            .as_mut()
            .expect("GrammarMatcher inner is null")
            .AcceptToken(token_id, debug_print)
    }

    /// Accept a string and update the state of the matcher. The whole string is considered
    /// as one step in rollback. It is used to complement the functionality of `accept_token`,
    /// and `accept_token` should always be used to accept tokens.
    ///
    /// # Parameters
    ///
    /// - `input`: The string to be accepted.
    /// - `debug_print`: Whether to print information about the internal state of the matcher.
    ///   Helpful for debugging.
    ///
    /// # Returns
    ///
    /// Whether the string is accepted.
    pub fn accept_string(
        &mut self,
        input: &str,
        debug_print: bool,
    ) -> bool {
        cxx::let_cxx_string!(input_cxx = input);
        self.inner
            .as_mut()
            .expect("GrammarMatcher inner is null")
            .AcceptString(&input_cxx, debug_print)
    }

    /// Fill the bitmask for the next token prediction. The input bitmask must be on CPU.
    /// `bitmask[index]` will be filled with the next token bitmask.
    ///
    /// This method does not change the matcher state.
    ///
    /// # Parameters
    ///
    /// - `bitmask`: The bitmask for the next token prediction.
    /// - `index`: The batch id of the bitmask.
    /// - `debug_print`: Whether to print information about generated bitmask.
    ///   Helpful for debugging.
    ///
    /// # Returns
    ///
    /// Whether the bitmask need to be applied (not all-true). An optimization: if false,
    /// this means the bitmask is already all-true, so no need to apply it.
    ///
    /// # Panics
    ///
    /// If the bitmask is invalid (not on CPU, not int32, shape mismatch).
    pub fn fill_next_token_bitmask(
        &mut self,
        bitmask: &mut DLTensor,
        index: i32,
        debug_print: bool,
    ) -> bool {
        unsafe {
            self.inner
                .as_mut()
                .expect("GrammarMatcher inner is null")
                .FillNextTokenBitmask(
                    bitmask as *mut _,
                    cxx_int(index),
                    debug_print,
                )
        }
    }

    /// Find the jump-forward string for jump-forward decoding. This is the longest string that
    /// certainly conforms with the current grammar from the current matcher state. This string
    /// can become the output of the LLM without requiring LLM decoding.
    ///
    /// This method does not change the matcher state.
    ///
    /// # Returns
    ///
    /// The jump-forward string.
    pub fn find_jump_forward_string(&mut self) -> String {
        self.inner
            .as_mut()
            .expect("GrammarMatcher inner is null")
            .FindJumpForwardString()
            .to_string()
    }

    /// Rollback the matcher to a previous state by several tokens.
    ///
    /// # Parameters
    ///
    /// - `num_tokens`: The number of tokens to rollback. It cannot exceed the current number
    ///   of steps, nor can it exceed the specified maximum number of rollback tokens.
    pub fn rollback(
        &mut self,
        num_tokens: i32,
    ) {
        self.inner
            .as_mut()
            .expect("GrammarMatcher inner is null")
            .Rollback(cxx_int(num_tokens));
    }

    /// Check if the matcher has terminated. If `terminate_without_stop_token` is false, the
    /// matcher will terminate if it has accepted the stop token. Otherwise, the matcher will
    /// terminate after matching the whole grammar.
    ///
    /// # Returns
    ///
    /// Whether the matcher has terminated.
    pub fn is_terminated(&self) -> bool {
        self.inner
            .as_ref()
            .expect("GrammarMatcher inner is null")
            .IsTerminated()
    }

    /// Reset the matcher to the initial state.
    pub fn reset(&mut self) {
        self.inner.as_mut().expect("GrammarMatcher inner is null").Reset();
    }

    /// Get the maximum number of rollback tokens allowed.
    ///
    /// Deprecated. Now `max_rollback_tokens` is always unlimited (-1).
    ///
    /// # Returns
    ///
    /// The maximum number of rollback tokens.
    pub fn max_rollback_tokens(&self) -> i32 {
        -1
    }

    /// The ids of the stop tokens used in the matcher. If specified, the provided stop tokens
    /// will be used. Otherwise, the stop tokens will be detected from the vocabulary.
    ///
    /// # Returns
    ///
    /// The ids of the stop tokens.
    pub fn stop_token_ids(&self) -> Box<[i32]> {
        self.stored_stop_token_ids.clone()
    }

    /// Print the internal state of the matcher. This is used for debugging. The
    /// representation of the internal state is subject to change.
    ///
    /// # Returns
    ///
    /// The internal state of the matcher.
    pub fn debug_print_internal_state(&self) -> String {
        self.inner
            .as_ref()
            .expect("GrammarMatcher inner is null")
            ._DebugPrintInternalState()
            .to_string()
    }

    pub(crate) fn ffi_ref(&self) -> &FFIGrammarMatcher {
        self.inner.as_ref().expect("GrammarMatcher inner is null")
    }
}

impl Drop for GrammarMatcher {
    fn drop(&mut self) {}
}
