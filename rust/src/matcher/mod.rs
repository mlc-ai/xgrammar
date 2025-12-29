//! Match the output of the LLM to the specified grammar, then generate the mask for the next
//! token.

mod batch_grammar_matcher;
mod grammar_matcher;

pub use batch_grammar_matcher::BatchGrammarMatcher;
pub use grammar_matcher::GrammarMatcher;

use crate::{DLTensor, cxx_utils};

/// Return the shape of the bitmask: (batch_size, ceil(vocab_size / 32)).
pub fn get_bitmask_shape(
    batch_size: usize,
    vocab_size: usize,
) -> (usize, usize) {
    (batch_size, (vocab_size + 31) / 32)
}

/// Allocate the bitmask for the next token prediction. The bitmask is an int32 tensor on
/// CPU with shape (batch_size, ceil(vocab_size / 32)). Users who have their own needs to
/// manage CUDA memory can construct the tensor with get_bitmask_shape and bitmask_dtype
/// themselves.
///
/// The reason why we use int32 instead of uint32 is that old versions of PyTorch do not support
/// uint32.
///
/// Parameters
/// ----------
/// batch_size : int
///     The batch size of the bitmask.
///
/// vocab_size : int
///     The size of the vocabulary.
///
/// Returns
/// -------
/// bitmask : torch.Tensor
///     The shape of the bitmask.
pub fn allocate_token_bitmask(
    batch_size: usize,
    vocab_size: usize,
) -> Box<[i32]> {
    let (_, bitmask_size) = get_bitmask_shape(batch_size, vocab_size);
    let total_size = batch_size * bitmask_size;
    vec![-1i32; total_size].into_boxed_slice()
}

/// Reset the bitmask to the full mask.
pub fn reset_token_bitmask(bitmask: &mut [i32]) {
    bitmask.fill(-1i32);
}

pub fn apply_token_bitmask_inplace_cpu(
    logits: &mut DLTensor,
    bitmask: &DLTensor,
    vocab_size: Option<i32>,
    indices: Option<&[i32]>,
) -> Result<(), String> {
    let vocab_size_i32 = vocab_size.unwrap_or(-1);
    let (has_indices, indices_ptr, indices_len) = match indices {
        Some(slice) if !slice.is_empty() => (true, slice.as_ptr(), slice.len()),
        _ => (false, std::ptr::null(), 0usize),
    };
    cxx::let_cxx_string!(error_out_cxx = "");
    let ok = unsafe {
        cxx_utils::apply_token_bitmask_inplace_cpu(
            logits as *mut _,
            bitmask as *const _,
            vocab_size_i32,
            has_indices,
            indices_ptr,
            indices_len,
            error_out_cxx.as_mut().get_unchecked_mut(),
        )
    };
    if !ok {
        return Err(error_out_cxx.to_string());
    }
    Ok(())
}
