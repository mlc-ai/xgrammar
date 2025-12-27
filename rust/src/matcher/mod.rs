//! Match the output of the LLM to the specified grammar, then generate the mask for the next
//! token.

mod batch_grammar_matcher;
mod grammar_matcher;

pub use batch_grammar_matcher::BatchGrammarMatcher;
pub use grammar_matcher::GrammarMatcher;

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
