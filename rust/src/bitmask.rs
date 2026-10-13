//! Token bitmasks, the Rust counterpart of the torch helpers in the Python
//! package (`allocate_token_bitmask`, `reset_token_bitmask`,
//! `apply_token_bitmask_inplace`).
//!
//! A bitmask stores one bit per token per batch row (32 tokens per `i32`
//! word); bit = 1 means the token is allowed. The layout is identical to the
//! DLPack tensor the C++ side fills, so a [`TokenBitmask`] can be passed
//! straight to [`crate::GrammarMatcher::fill_next_token_bitmask`].

/// Number of `i32` words needed to store a bitmask over `vocab_size` tokens,
/// the second element of Python's `xgr.get_bitmask_shape`.
pub fn bitmask_words(vocab_size: usize) -> usize {
    vocab_size.div_ceil(32)
}

/// An owned CPU token bitmask of shape `[batch_size, ceil(vocab_size / 32)]`.
#[derive(Debug, Clone)]
pub struct TokenBitmask {
    data: Vec<i32>,
    batch_size: usize,
    vocab_size: usize,
    words: usize,
}

impl TokenBitmask {
    /// Allocate a single-row bitmask with every token allowed.
    pub fn new(vocab_size: usize) -> Self {
        Self::with_batch(1, vocab_size)
    }

    /// Allocate a `batch_size`-row bitmask with every token allowed.
    pub fn with_batch(batch_size: usize, vocab_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be positive");
        assert!(vocab_size > 0, "vocab_size must be positive");
        let words = bitmask_words(vocab_size);
        Self {
            data: vec![-1; batch_size * words],
            batch_size,
            vocab_size,
            words,
        }
    }

    /// Number of batch rows.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Vocabulary size the mask was allocated for.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Reset every bit to "allowed", like Python's `reset_token_bitmask`.
    pub fn reset(&mut self) {
        self.data.fill(-1);
    }

    /// Whether `token_id` is allowed in row `index`.
    pub fn is_allowed(&self, index: usize, token_id: usize) -> bool {
        assert!(index < self.batch_size && token_id < self.vocab_size);
        let word = self.data[index * self.words + token_id / 32];
        (word >> (token_id % 32)) & 1 != 0
    }

    /// The token ids masked out (disallowed) in row `index`, the native
    /// counterpart of `xgrammar.testing._get_masked_tokens_from_bitmask`.
    pub fn masked_tokens(&self, index: usize) -> Vec<i64> {
        (0..self.vocab_size)
            .filter(|&t| !self.is_allowed(index, t))
            .map(|t| t as i64)
            .collect()
    }

    /// Set logits of masked tokens in row `index` to `-inf`, on the CPU.
    ///
    /// `logits` is a single row of at least `vocab_size` entries (entries
    /// beyond `vocab_size` are left untouched, matching the Python kernels).
    pub fn apply_to_logits(&self, logits: &mut [f32], index: usize) {
        assert!(index < self.batch_size, "row index out of range");
        assert!(
            logits.len() >= self.vocab_size,
            "logits row shorter than vocab_size"
        );
        let row = &self.data[index * self.words..(index + 1) * self.words];
        for (word_idx, &word) in row.iter().enumerate() {
            if word == -1 {
                continue;
            }
            let base = word_idx * 32;
            let end = (base + 32).min(self.vocab_size);
            for bit in 0..(end - base) {
                if (word >> bit) & 1 == 0 {
                    logits[base + bit] = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// Apply the whole bitmask to a `[batch, row_len]` row-major logits
    /// buffer. `indices`, when given, selects which logits rows to touch:
    /// logits row `indices[i]` is masked with bitmask row `i` (like the
    /// `indices` argument of Python's `apply_token_bitmask_inplace`).
    pub fn apply_to_logits_2d(
        &self,
        logits: &mut [f32],
        row_len: usize,
        indices: Option<&[usize]>,
    ) {
        assert!(
            row_len >= self.vocab_size,
            "logits row shorter than vocab_size"
        );
        match indices {
            None => {
                assert!(logits.len() >= self.batch_size * row_len);
                for i in 0..self.batch_size {
                    self.apply_to_logits(&mut logits[i * row_len..(i + 1) * row_len], i);
                }
            }
            Some(indices) => {
                for (mask_row, &logits_row) in indices.iter().enumerate() {
                    assert!(logits.len() >= (logits_row + 1) * row_len);
                    self.apply_to_logits(
                        &mut logits[logits_row * row_len..(logits_row + 1) * row_len],
                        mask_row,
                    );
                }
            }
        }
    }

    /// The underlying words, row-major `[batch_size, ceil(vocab_size/32)]`.
    pub fn as_slice(&self) -> &[i32] {
        &self.data
    }

    /// Mutable access to the underlying words.
    pub fn as_mut_slice(&mut self) -> &mut [i32] {
        &mut self.data
    }

    pub(crate) fn dl_arg(&mut self) -> crate::ffi::DlArg {
        let shape = [self.batch_size as i64, self.words as i64];
        crate::ffi::DlArg::int32(&mut self.data, &shape)
    }
}
