use std::pin::Pin;

use autocxx::prelude::*;

use super::GrammarMatcher;
use crate::{CxxUniquePtr, DLTensor, cxx_utils};

/// A batch version of `GrammarMatcher` that can fill the next token bitmask for multiple
/// matchers in parallel. It utilizes multiple threads to speed up the computation. It is
/// especially useful when the batch size is large.
pub struct BatchGrammarMatcher {
    inner: CxxUniquePtr<crate::FFIBatchGrammarMatcher>,
}

impl BatchGrammarMatcher {
    /// Construct the batch grammar matcher.
    ///
    /// # Parameters
    ///
    /// - `max_threads`: The maximum number of threads to use for parallel processing. If set
    ///   to -1, the max_threads will be set to `std::thread::hardware_concurrency() / 2`.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch grammar matcher cannot be constructed.
    pub fn new(max_threads: i32) -> Result<Self, String> {
        cxx::let_cxx_string!(error_out_cxx = "");
        let ffi_pin = unsafe {
            cxx_utils::make_batch_grammar_matcher(
                max_threads,
                error_out_cxx.as_mut().get_unchecked_mut(),
            )
        };
        if ffi_pin.is_null() {
            return Err(error_out_cxx.to_string());
        }
        Ok(Self {
            inner: ffi_pin,
        })
    }

    /// Create a batch grammar matcher with automatic thread count.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch grammar matcher cannot be constructed.
    pub fn new_auto() -> Result<Self, String> {
        Self::new(-1)
    }

    /// Fill the next token bitmask for multiple matchers.
    ///
    /// # Parameters
    ///
    /// - `matchers`: The list of matchers to fill the bitmask for.
    /// - `bitmask`: Must be a 2-dimensional int32 tensor with shape
    ///   `(bitmask_batch_size, bitmask_size)`. `bitmask_batch_size` could be larger than the
    ///   actual batch size to allow padding. `bitmask_size` equals to `ceil(vocab_size/32)`,
    ///   and could be computed through `allocate_token_bitmask`.
    /// - `indices`: A list of indices to specify which rows in the bitmask to fill. If `None`,
    ///   fill the bitmask `[0..matchers.len())`.
    /// - `debug_print`: Whether to print information about generated bitmask.
    ///   Helpful for debugging.
    ///
    /// # Panics
    ///
    /// If the bitmask is invalid (not on CPU, not int32, shape mismatch).
    pub fn batch_fill_next_token_bitmask(
        &mut self,
        matchers: &[GrammarMatcher],
        bitmask: &mut DLTensor,
        indices: Option<&[i32]>,
        debug_print: bool,
    ) {
        let mut ffi_matcher_vec = cxx_utils::new_grammar_matcher_vector();
        {
            let mut vec_pin = ffi_matcher_vec.pin_mut();
            cxx_utils::grammar_matcher_vec_reserve(
                vec_pin.as_mut(),
                matchers.len(),
            );
            for matcher in matchers {
                cxx_utils::grammar_matcher_vec_push(
                    vec_pin.as_mut(),
                    matcher.ffi_ref(),
                );
            }
        }

        let (has_indices, indices_ptr, indices_len) = match indices {
            Some(slice) if !slice.is_empty() => {
                (true, slice.as_ptr(), slice.len())
            },
            _ => (false, std::ptr::null(), 0usize),
        };

        unsafe {
            cxx_utils::batch_matcher_batch_fill_next_token_bitmask(
                self.inner.as_mut().expect("BatchGrammarMatcher inner is null"),
                ffi_matcher_vec.as_mut().unwrap().get_unchecked_mut(),
                bitmask as *mut _,
                has_indices,
                indices_ptr,
                indices_len,
                debug_print,
            );
        }
    }

    /// Accept a batch of tokens for multiple matchers.
    ///
    /// # Parameters
    ///
    /// - `matchers`: The list of matchers to accept tokens for.
    /// - `tokens`: The list of tokens to accept.
    /// - `debug_print`: Whether to print information about generated bitmask.
    ///   Helpful for debugging.
    ///
    /// # Returns
    ///
    /// A list of booleans indicating whether each token was accepted by its corresponding
    /// matcher.
    ///
    /// # Panics
    ///
    /// If the sizes of `matchers` and `tokens` do not match.
    pub fn batch_accept_token(
        matchers: &[GrammarMatcher],
        tokens: &[i32],
        debug_print: bool,
    ) -> Box<[bool]> {
        assert_eq!(
            matchers.len(),
            tokens.len(),
            "matchers and tokens must have the same length"
        );

        let mut ffi_matcher_vec = cxx_utils::new_grammar_matcher_vector();
        {
            let mut vec_pin = ffi_matcher_vec.pin_mut();
            cxx_utils::grammar_matcher_vec_reserve(
                vec_pin.as_mut(),
                matchers.len(),
            );
            for matcher in matchers {
                cxx_utils::grammar_matcher_vec_push(
                    vec_pin.as_mut(),
                    matcher.ffi_ref(),
                );
            }
        }

        let result = unsafe {
            cxx_utils::batch_accept_token(
                ffi_matcher_vec.as_mut().unwrap().get_unchecked_mut(),
                tokens.as_ptr(),
                tokens.len(),
                debug_print,
            )
        };

        result.iter().map(|&b| b != 0).collect::<Vec<_>>().into_boxed_slice()
    }

    /// Accept a batch of strings for multiple matchers.
    ///
    /// # Parameters
    ///
    /// - `matchers`: The list of matchers to accept tokens for.
    /// - `strings`: The list of strings to accept.
    /// - `debug_print`: Whether to print information about generated bitmask.
    ///   Helpful for debugging.
    ///
    /// # Returns
    ///
    /// A list of booleans indicating whether each string was accepted by its corresponding
    /// matcher.
    ///
    /// # Panics
    ///
    /// If the sizes of `matchers` and `strings` do not match.
    pub fn batch_accept_string(
        matchers: &[GrammarMatcher],
        strings: &[impl AsRef<str>],
        debug_print: bool,
    ) -> Box<[bool]> {
        assert_eq!(
            matchers.len(),
            strings.len(),
            "matchers and strings must have the same length"
        );

        let mut ffi_matcher_vec = cxx_utils::new_grammar_matcher_vector();
        {
            let mut vec_pin = ffi_matcher_vec.pin_mut();
            cxx_utils::grammar_matcher_vec_reserve(
                vec_pin.as_mut(),
                matchers.len(),
            );
            for matcher in matchers {
                cxx_utils::grammar_matcher_vec_push(
                    vec_pin.as_mut(),
                    matcher.ffi_ref(),
                );
            }
        }

        let mut cxx_strings = cxx_utils::new_string_vector();
        {
            let mut cxx_vec_pin = cxx_strings.pin_mut();
            cxx_utils::string_vec_reserve(cxx_vec_pin.as_mut(), strings.len());
            for string in strings.iter() {
                let bytes = string.as_ref().as_bytes();
                unsafe {
                    cxx_utils::string_vec_push_bytes(
                        cxx_vec_pin.as_mut(),
                        bytes.as_ptr() as *const i8,
                        bytes.len(),
                    );
                }
            }
        }

        let result = unsafe {
            cxx_utils::batch_accept_string(
                ffi_matcher_vec.as_mut().unwrap().get_unchecked_mut(),
                cxx_strings.as_ref().unwrap(),
                debug_print,
            )
        };

        result.iter().map(|&b| b != 0).collect::<Vec<_>>().into_boxed_slice()
    }
}
