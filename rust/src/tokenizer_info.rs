use autocxx::prelude::*;

use crate::{CxxUniquePtr, FFITokenizerInfo, VocabType, cxx_utils};

type StopTokenIds = Option<Box<[i32]>>;

/// The tokenizer info contains the vocabulary, the type of the vocabulary, and necessary
/// information for the grammar-guided generation.
///
/// Note that although some tokenizers will encode the tokens in a special format, e.g.
/// `<0x1B>` for `\u001b` in the ByteFallback tokenizer, and `Ġ` for ` ` in the Byte-Level BPE
/// tokenizer, TokenizerInfo always decodes the vocabulary to the original format (e.g. `\u001b`
/// and ` `).
///
/// Also note that some models (e.g. Phi-3 and Deepseek-V2) may pad the vocabulary to a multiple
/// of 32. In this case, the model's vocab_size is larger than the tokenizer's vocabulary size.
/// Please pass the model's vocab_size to the `vocab_size` parameter in the constructor, because
/// this information is used to determine the size of the token mask.
pub struct TokenizerInfo {
    inner: CxxUniquePtr<FFITokenizerInfo>,
}

impl TokenizerInfo {
    /// Construct the tokenizer info.
    ///
    /// # Parameters
    ///
    /// - `encoded_vocab`: The encoded vocabulary of the tokenizer.
    /// - `vocab_type`: The type of the vocabulary. See also `VocabType`.
    /// - `stop_token_ids`: The stop token ids. If `None`, the stop token ids will be auto
    ///   detected (but may not be correct).
    /// - `add_prefix_space`: Whether the tokenizer will prepend a space before the text in
    ///   the tokenization process.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer info cannot be constructed.
    pub fn new<T: AsRef<str>>(
        encoded_vocab: &[T],
        vocab_type: VocabType,
        stop_token_ids: &StopTokenIds,
        add_prefix_space: bool,
    ) -> Result<Self, String> {
        Self::new_with_vocab_size(
            encoded_vocab,
            vocab_type,
            Some(encoded_vocab.len()),
            stop_token_ids,
            add_prefix_space,
        )
    }

    /// Construct the tokenizer info with an explicit vocab size.
    ///
    /// # Parameters
    ///
    /// - `encoded_vocab`: The encoded vocabulary of the tokenizer.
    /// - `vocab_type`: The type of the vocabulary. See also `VocabType`.
    /// - `vocab_size`: The size of the vocabulary. If not provided, the vocabulary size will
    ///   be `encoded_vocab.len()`.
    /// - `stop_token_ids`: The stop token ids. If `None`, the stop token ids will be auto
    ///   detected (but may not be correct).
    /// - `add_prefix_space`: Whether the tokenizer will prepend a space before the text in
    ///   the tokenization process.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer info cannot be constructed.
    pub fn new_with_vocab_size<T: AsRef<str>>(
        encoded_vocab: &[T],
        vocab_type: VocabType,
        vocab_size: Option<usize>,
        stop_token_ids: &StopTokenIds,
        add_prefix_space: bool,
    ) -> Result<Self, String> {
        let mut cxx_vec = cxx_utils::new_string_vector();
        {
            let mut cxx_vec_pin = cxx_vec.pin_mut();
            cxx_utils::string_vec_reserve(
                cxx_vec_pin.as_mut(),
                encoded_vocab.len(),
            );
            for string in encoded_vocab.iter() {
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
        let (has_vocab_size, vocab_size_i32) = match vocab_size {
            Some(sz) => (true, sz as i32),
            None => (false, 0i32),
        };

        let (has_stop_ids, stop_ptr, stop_len) = match stop_token_ids.as_ref() {
            Some(slice) if !slice.is_empty() => {
                (true, slice.as_ptr(), slice.len())
            },
            _ => (false, std::ptr::null(), 0usize),
        };

        cxx::let_cxx_string!(error_out_cxx = "");
        let ffi_obj = unsafe {
            cxx_utils::make_tokenizer_info(
                cxx_vec.as_ref().unwrap(),
                vocab_type,
                has_vocab_size,
                vocab_size_i32,
                has_stop_ids,
                stop_ptr,
                stop_len,
                add_prefix_space,
                error_out_cxx.as_mut().get_unchecked_mut(),
            )
        };
        if ffi_obj.is_null() {
            return Err(error_out_cxx.to_string());
        }

        let inner = ffi_obj;
        Ok(Self {
            inner,
        })
    }

    /// Construct the tokenizer info from the vocabulary and the metadata string in JSON format.
    ///
    /// # Parameters
    ///
    /// - `encoded_vocab`: The encoded vocabulary of the tokenizer.
    /// - `metadata`: The metadata string in JSON format.
    pub fn from_vocab_and_metadata_bytes<I, B>(
        encoded_vocab: I,
        metadata: &str,
    ) -> Self
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let mut cxx_vec = cxx_utils::new_string_vector();
        {
            let mut cxx_vec_pin = cxx_vec.pin_mut();
            for string in encoded_vocab.into_iter() {
                let bytes = string.as_ref();
                unsafe {
                    cxx_utils::string_vec_push_bytes(
                        cxx_vec_pin.as_mut(),
                        bytes.as_ptr() as *const i8,
                        bytes.len(),
                    );
                }
            }
        }

        cxx::let_cxx_string!(metadata_cxx = metadata);
        let ffi_ptr = FFITokenizerInfo::FromVocabAndMetadata(
            cxx_vec.as_ref().unwrap(),
            &metadata_cxx,
        )
        .within_unique_ptr();
        Self {
            inner: ffi_ptr,
        }
    }

    /// The type of the vocabulary.
    pub fn vocab_type(&self) -> VocabType {
        self.inner
            .as_ref()
            .expect("FFITokenizerInfo UniquePtr was null")
            .GetVocabType()
    }

    /// The size of the vocabulary.
    pub fn vocab_size(&self) -> usize {
        let sz = usize::try_from(
            self.inner
                .as_ref()
                .expect("FFITokenizerInfo UniquePtr was null")
                .GetVocabSize()
                .0,
        )
        .expect("GetVocabSize returned a negative value");
        sz
    }

    /// Whether the tokenizer will prepend a space before the text in the tokenization process.
    pub fn add_prefix_space(&self) -> bool {
        let val = self
            .inner
            .as_ref()
            .expect("FFITokenizerInfo UniquePtr was null")
            .GetAddPrefixSpace();
        val
    }

    /// The decoded vocabulary of the tokenizer. This converts the tokens in the LLM's
    /// vocabulary back to the original format of the input text. E.g. for type ByteFallback,
    /// the token `<0x1B>` is converted back to `\u001b`.
    pub fn decoded_vocab(&self) -> Box<[Box<[u8]>]> {
        let cxx_vec = self.inner.GetDecodedVocab();
        let mut result: Vec<Box<[u8]>> = Vec::with_capacity(cxx_vec.len());
        for cxx_string in cxx_vec.iter() {
            result.push(
                cxx_string
                    .to_string_lossy()
                    .into_owned()
                    .into_bytes()
                    .into_boxed_slice(),
            );
        }
        result.into_boxed_slice()
    }

    /// The stop token ids.
    pub fn stop_token_ids(&self) -> Box<[i32]> {
        let cxx_vec = self.inner.GetStopTokenIds();
        cxx_vec.iter().copied().collect::<Vec<_>>().into_boxed_slice()
    }

    /// The special token ids. Special tokens include control tokens, reserved tokens,
    /// padded tokens, etc. Now it is automatically detected from the vocabulary.
    pub fn special_token_ids(&self) -> Box<[i32]> {
        let cxx_vec = self
            .inner
            .as_ref()
            .expect("FFITokenizerInfo UniquePtr was null")
            .GetSpecialTokenIds();
        cxx_vec.iter().copied().collect::<Vec<_>>().into_boxed_slice()
    }

    /// Dump the metadata of the tokenizer to a JSON string. It can be used to construct the
    /// tokenizer info from the vocabulary and the metadata string.
    pub fn dump_metadata(&self) -> String {
        self.inner
            .as_ref()
            .expect("FFITokenizerInfo UniquePtr was null")
            .DumpMetadata()
            .to_string()
    }

    /// Serialize the tokenizer info to a JSON string.
    ///
    /// # Returns
    ///
    /// The JSON string.
    pub fn serialize_json(&self) -> String {
        self.inner
            .as_ref()
            .expect("FFITokenizerInfo UniquePtr was null")
            .SerializeJSON()
            .to_string()
    }

    /// Deserialize a tokenizer info from a JSON string.
    ///
    /// # Parameters
    ///
    /// - `json`: The JSON string.
    ///
    /// # Returns
    ///
    /// The tokenizer info.
    ///
    /// # Errors
    ///
    /// - When the JSON string is invalid.
    /// - When the JSON string does not follow the serialization format of the tokenizer info.
    /// - When the `__VERSION__` field in the JSON string is not the same as the current version.
    pub fn deserialize_json(json: &str) -> Result<Self, String> {
        cxx::let_cxx_string!(json_cxx = json);
        cxx::let_cxx_string!(error_out_cxx = "");
        let uptr = unsafe {
            cxx_utils::tokenizer_info_deserialize_json_or_error(
                &json_cxx,
                error_out_cxx.as_mut().get_unchecked_mut(),
            )
        };
        if uptr.is_null() {
            return Err(error_out_cxx.to_string());
        }
        Ok(Self {
            inner: uptr,
        })
    }

    pub(crate) fn ffi_ref(&self) -> &FFITokenizerInfo {
        self.inner.as_ref().expect("FFITokenizerInfo UniquePtr was null")
    }

    pub(crate) fn from_unique_ptr(
        inner: cxx::UniquePtr<FFITokenizerInfo>
    ) -> Self {
        Self {
            inner,
        }
    }
}

impl Drop for TokenizerInfo {
    fn drop(&mut self) {}
}

#[cfg(feature = "tokenizers")]
impl TokenizerInfo {
    #[inline]
    fn extract_ordered_vocab(
        tokenizer: &tokenizers::Tokenizer
    ) -> Box<[String]> {
        let mut pairs: Vec<(usize, String)> = tokenizer
            .get_vocab(true)
            .into_iter()
            .map(|(tok, id)| (id as usize, tok))
            .collect();
        pairs.sort_by_key(|(id, _)| *id);
        pairs
            .into_iter()
            .map(|(_, tok)| tok)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Heuristically detect whether a tokenizer resembles a tiktoken-style tokenizer.
    ///
    /// In Python this checks `isinstance(tokenizer.tokenizer, tiktoken.Encoding)` or whether
    /// the vocab filename contains "tiktoken". In Rust we do not have those runtime types,
    /// so we approximate: if the vocabulary does NOT contain typical markers of
    /// SentencePiece (`▁`), Byte-level GPT-2 (`Ġ`), or ByteFallback (tokens like `<0x1B>`),
    /// we consider it RAW (tiktoken-like).
    pub fn _is_tiktoken_tokenizer(tokenizer: &tokenizers::Tokenizer) -> bool {
        let vocab = tokenizer.get_vocab(true);
        let mut has_sentencepiece_marker = false;
        let mut has_bytelevel_marker = false;
        let mut has_bytefallback_marker = false;
        for token in vocab.keys() {
            if !has_sentencepiece_marker && token.contains('▁') {
                has_sentencepiece_marker = true;
            }
            if !has_bytelevel_marker && token.contains('Ġ') {
                has_bytelevel_marker = true;
            }
            if !has_bytefallback_marker
                && token.starts_with("<0x")
                && token.ends_with('>')
            {
                has_bytefallback_marker = true;
            }
            if has_sentencepiece_marker
                || has_bytelevel_marker
                || has_bytefallback_marker
            {
                break;
            }
        }
        !(has_sentencepiece_marker
            || has_bytelevel_marker
            || has_bytefallback_marker)
    }

    /// Heuristically detect whether a tokenizer is SentencePiece-based.
    ///
    /// In Python this checks for a `sentencepiece.SentencePieceProcessor`. Here we look for
    /// typical SentencePiece marker `▁` in the vocabulary. This is a best-effort heuristic
    /// and may not be perfect for all models.
    pub fn _is_sentencepiece_tokenizer(
        tokenizer: &tokenizers::Tokenizer
    ) -> bool {
        let vocab = tokenizer.get_vocab(true);
        vocab.keys().any(|tok| tok.contains('▁'))
    }

    /// Construct from a `tokenizers::Tokenizer` with explicit options, preserving tokenizer
    /// indexing.
    ///
    /// This matches Python's constructor path where `encoded_vocab` is built by id order and
    /// `vocab_size` may be larger than the tokenizer's vocab (model padding), with special ids
    /// reserved in the tail range.
    ///
    /// # Parameters
    ///
    /// - `tokenizer`: The tokenizer.
    /// - `vocab_type`: The type of the vocabulary.
    /// - `vocab_size`: The vocabulary size defined by the model (not the tokenizer).
    /// - `stop_token_ids`: The stop token ids.
    /// - `add_prefix_space`: Whether the tokenizer will prepend a space before the text.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer info cannot be constructed.
    pub fn from_tokenizers_with_options(
        tokenizer: &tokenizers::Tokenizer,
        vocab_type: VocabType,
        vocab_size: Option<usize>,
        stop_token_ids: Option<&[i32]>,
        add_prefix_space: bool,
    ) -> Result<Self, String> {
        let ordered = Self::extract_ordered_vocab(tokenizer);
        let stop: Option<Box<[i32]>> =
            stop_token_ids.map(|s| s.to_vec().into_boxed_slice());
        Self::new_with_vocab_size(
            &ordered,
            vocab_type,
            vocab_size,
            &stop,
            add_prefix_space,
        )
    }

    /// Convenience: RAW vocab, detected size, no stops, no prefix space.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer info cannot be constructed.
    pub fn from_tokenizers_simple(tokenizer: &tokenizers::Tokenizer) -> Result<Self, String> {
        Self::from_tokenizers_with_options(
            tokenizer,
            VocabType::RAW,
            None,
            None,
            false,
        )
    }

    /// Construct the tokenizer info from a Hugging Face tokenizer. This constructor supports
    /// various tokenizer backends. Necessary information is automatically detected from
    /// the tokenizer.
    ///
    /// The `vocab_size` parameter is introduced to handle the misalignment between the model's
    /// vocab_size and the tokenizer's vocabulary size. User should pass the model's vocab_size
    /// (could be defined in the model config) here.
    ///
    /// The stop token ids is by default auto-detected. If there are other stop tokens, you can
    /// specify them manually.
    ///
    /// # Parameters
    ///
    /// - `tokenizer`: The tokenizer.
    /// - `vocab_size`: The vocabulary size defined by the model (not the tokenizer). This equals
    ///   to the vocab dimension of the model's lm_head. This is the size of the token mask.
    ///   It can be:
    ///   1. the same as the tokenizer's vocabulary size. This is the most common case.
    ///   2. larger than the tokenizer's vocabulary size. This happens when the model has padding
    ///      to lm_head, possibly due to aligning lm_head to the power of 2.
    ///   3. smaller than the tokenizer's vocabulary size. This happens when the tokenizer has
    ///      some added tokens that will not supported by the model.
    /// - `stop_token_ids`: The stop token ids. If `None`, they will be auto-detected.
    ///
    /// # Returns
    ///
    /// The tokenizer info.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer info cannot be constructed.
    pub fn from_huggingface(
        tokenizer: &tokenizers::Tokenizer,
        vocab_size: Option<usize>,
        stop_token_ids: Option<&[i32]>,
    ) -> Result<Self, String> {
        use crate::VocabType;

        let vocab = tokenizer.get_vocab(true);
        let has_bytefallback_marker =
            vocab.keys().any(|t| t.starts_with("<0x") && t.ends_with('>'));
        let has_sentencepiece_marker = vocab.keys().any(|t| t.contains('▁'));
        let has_bytelevel_marker = vocab.keys().any(|t| t.contains('Ġ'));

        let (vocab_type, add_prefix_space) = if has_bytefallback_marker {
            (VocabType::BYTE_FALLBACK, true)
        } else if has_sentencepiece_marker {
            (VocabType::RAW, true)
        } else if has_bytelevel_marker {
            (VocabType::BYTE_LEVEL, false)
        } else {
            (VocabType::RAW, false)
        };

        Self::from_tokenizers_with_options(
            tokenizer,
            vocab_type,
            vocab_size,
            stop_token_ids,
            add_prefix_space,
        )
    }
}
