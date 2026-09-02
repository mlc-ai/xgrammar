//! Tokenizer metadata needed to compile grammars against a model vocabulary.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;
use tvm_ffi::{Array, Bytes as FfiBytes, String as FfiString};

use crate::error::{Error, Result};
use tvm_ffi::object::ObjectRef;

use crate::ffi::{ffi_call, handle, opt_any, ret};

/// How tokens are encoded in the vocabulary passed to [`TokenizerInfo::new`].
///
/// Matches the Python `xgrammar.VocabType` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VocabType {
    /// Tokens appear in their raw form (e.g. tiktoken-style BPE tokenizers).
    Raw,
    /// Tokens use `<0xAB>`-style byte-fallback escapes (e.g. Llama-2 style
    /// sentencepiece tokenizers).
    ByteFallback,
    /// Tokens use the GPT-2 byte-to-unicode mapping (e.g. GPT-2, Llama-3).
    ByteLevel,
}

impl VocabType {
    fn to_ffi(self) -> i64 {
        match self {
            VocabType::Raw => 0,
            VocabType::ByteFallback => 1,
            VocabType::ByteLevel => 2,
        }
    }

    fn from_ffi(value: i64) -> Result<Self> {
        match value {
            0 => Ok(VocabType::Raw),
            1 => Ok(VocabType::ByteFallback),
            2 => Ok(VocabType::ByteLevel),
            _ => Err(Error::XGrammar(format!("unknown VocabType value {value}"))),
        }
    }
}

/// Optional parameters for [`TokenizerInfo::new`], mirroring the keyword
/// arguments of the Python constructor.
#[derive(Debug, Clone, Default)]
pub struct TokenizerInfoOptions {
    /// The size of the model's vocabulary, which may be larger than the
    /// encoded vocabulary (padded special tokens). `None` uses the vocabulary
    /// length.
    pub vocab_size: Option<usize>,
    /// Token ids that stop generation. `None` applies a heuristic detection.
    pub stop_token_ids: Option<Vec<i64>>,
    /// Whether the tokenizer prepends a space when tokenizing.
    pub add_prefix_space: bool,
}

/// Optional parameters for [`TokenizerInfo::from_huggingface`].
#[derive(Debug, Clone, Default)]
pub struct HuggingFaceTokenizerOptions {
    /// The size of the model's vocabulary (the output dimension of its
    /// language-model head). `None` derives the size from the tokenizer.
    ///
    /// Set this when the model pads its vocabulary, or when tokenizer-only
    /// added tokens are absent from the model head.
    pub vocab_size: Option<usize>,
    /// Token ids that stop generation. A raw [`tokenizers::Tokenizer`] does
    /// not expose which special token is the model's EOS token, so callers
    /// should normally pass it explicitly. `None` uses xgrammar's token-name
    /// heuristic.
    pub stop_token_ids: Option<Vec<i64>>,
}

#[derive(Deserialize)]
struct HuggingFaceMetadata {
    vocab_type: i64,
    add_prefix_space: bool,
}

/// The tokenizer information xgrammar needs: the decoded vocabulary and
/// metadata such as the vocabulary encoding and the stop token ids.
///
/// This class is immutable: instances can be cloned cheaply (shared handle)
/// and shared across threads.
#[derive(Clone)]
pub struct TokenizerInfo {
    pub(crate) raw: ObjectRef,
}

impl TokenizerInfo {
    pub(crate) fn from_raw(raw: ObjectRef) -> Self {
        Self { raw }
    }

    /// Construct from the encoded vocabulary (in token-id order) and the
    /// vocabulary encoding type.
    pub fn new<I, T>(
        encoded_vocab: I,
        vocab_type: VocabType,
        options: &TokenizerInfoOptions,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: AsRef<[u8]>,
    {
        let vocab = Array::<FfiBytes>::new(
            encoded_vocab
                .into_iter()
                .map(|t| FfiBytes::from(t.as_ref()))
                .collect(),
        );
        let stop_ids = options
            .stop_token_ids
            .as_ref()
            .map(|ids| Array::<i64>::new(ids.clone()));
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.TokenizerInfo.__ffi_init__",
            vocab,
            vocab_type.to_ffi(),
            opt_any(options.vocab_size.map(|v| v as i64)),
            opt_any(stop_ids),
            options.add_prefix_space,
        )?;
        Ok(Self::from_raw(ret(any)?))
    }

    /// Construct directly from a Hugging Face fast tokenizer.
    ///
    /// The vocabulary (including added tokens) is placed in token-id order,
    /// while the vocabulary encoding and prefix-space behavior are detected
    /// from the serialized `tokenizer.json` pipeline. As in the Python API,
    /// the vocabulary itself is used to correct stale or incomplete decoder
    /// metadata for byte-fallback and byte-level tokenizers.
    ///
    /// This accepts Hugging Face's native Rust [`tokenizers::Tokenizer`],
    /// which can be loaded with [`tokenizers::Tokenizer::from_file`] or
    /// [`tokenizers::Tokenizer::from_bytes`]. It corresponds to Python fast
    /// tokenizers; Python-only slow SentencePiece and tiktoken wrapper objects
    /// are outside this API.
    pub fn from_huggingface(
        tokenizer: &tokenizers::Tokenizer,
        options: &HuggingFaceTokenizerOptions,
    ) -> Result<Self> {
        if options.stop_token_ids.as_ref().is_some_and(Vec::is_empty) {
            return Err(Error::XGrammar(
                "stop_token_ids cannot be empty when constructing from a Hugging Face tokenizer"
                    .to_string(),
            ));
        }

        let vocab = tokenizer.get_vocab(true);
        let max_id = vocab.values().copied().max().ok_or_else(|| {
            Error::XGrammar("the Hugging Face tokenizer has an empty vocabulary".to_string())
        })?;
        let indexed_vocab_size = (max_id as usize).checked_add(1).ok_or_else(|| {
            Error::XGrammar("the Hugging Face tokenizer vocabulary is too large".to_string())
        })?;
        let tokenizer_vocab_size = vocab.len().max(indexed_vocab_size);
        let vocab_size = options.vocab_size.unwrap_or(tokenizer_vocab_size);
        if vocab_size == 0 {
            return Err(Error::XGrammar(
                "Hugging Face model vocab_size must be positive".to_string(),
            ));
        }

        // Preserve token ids, including holes. Limiting the vector to the
        // model vocabulary also mirrors Python for tokenizers whose added
        // tokens are not represented by the model's language-model head.
        let mut encoded_vocab = vec![Vec::new(); vocab_size];
        for (token, &token_id) in &vocab {
            if let Some(slot) = encoded_vocab.get_mut(token_id as usize) {
                *slot = token.as_bytes().to_vec();
            }
        }

        let backend = tokenizer.to_string(false)?;
        let metadata_json = Self::detect_metadata_from_hf(&backend)?;
        let metadata: HuggingFaceMetadata =
            serde_json::from_str(&metadata_json).map_err(|err| {
                Error::XGrammar(format!(
                    "invalid metadata detected from Hugging Face tokenizer: {err}"
                ))
            })?;
        let mut vocab_type = VocabType::from_ffi(metadata.vocab_type)?;
        let mut add_prefix_space = metadata.add_prefix_space;

        // A few Transformers conversions serialize incomplete decoder
        // metadata. The byte alphabet in the vocabulary is stronger evidence
        // of how those token strings must be decoded.
        if let Some(vocab_type_from_vocab) = detect_vocab_type_from_vocab(&vocab) {
            if vocab_type_from_vocab != vocab_type {
                vocab_type = vocab_type_from_vocab;
                add_prefix_space =
                    detect_add_prefix_space_by_encoding(tokenizer, vocab_type_from_vocab);
            }
        }

        Self::new(
            encoded_vocab,
            vocab_type,
            &TokenizerInfoOptions {
                vocab_size: Some(vocab_size),
                stop_token_ids: options.stop_token_ids.clone(),
                add_prefix_space,
            },
        )
    }

    /// Construct from an encoded vocabulary and a metadata JSON string, the
    /// inverse of [`TokenizerInfo::dump_metadata`].
    pub fn from_vocab_and_metadata<I, T>(encoded_vocab: I, metadata: &str) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: AsRef<[u8]>,
    {
        let vocab = Array::<FfiBytes>::new(
            encoded_vocab
                .into_iter()
                .map(|t| FfiBytes::from(t.as_ref()))
                .collect(),
        );
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.TokenizerInfo.from_vocab_and_metadata",
            vocab,
            FfiString::from(metadata),
        )?;
        Ok(Self::from_raw(ret(any)?))
    }

    /// Detect tokenizer metadata from a HuggingFace tokenizer backend string
    /// (the JSON produced by `tokenizers::Tokenizer::to_string` /
    /// `PreTrainedTokenizerFast.backend_tokenizer.to_str()`). Returns a
    /// metadata JSON string suitable for
    /// [`TokenizerInfo::from_vocab_and_metadata`].
    pub fn detect_metadata_from_hf(backend_str: &str) -> Result<String> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.TokenizerInfo._detect_metadata_from_hf",
            FfiString::from(backend_str),
        )?;
        Ok(ret::<FfiString>(any)?.as_str().to_string())
    }

    /// The vocabulary encoding type.
    pub fn vocab_type(&self) -> Result<VocabType> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.TokenizerInfo.vocab_type",
            self.raw
        )?;
        VocabType::from_ffi(ret::<i64>(any)?)
    }

    /// The size of the vocabulary (including padded special tokens).
    pub fn vocab_size(&self) -> Result<usize> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.TokenizerInfo.vocab_size",
            self.raw
        )?;
        Ok(ret::<i64>(any)? as usize)
    }

    /// Whether the tokenizer prepends a space when tokenizing.
    pub fn add_prefix_space(&self) -> Result<bool> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.TokenizerInfo.add_prefix_space",
            self.raw,
        )?;
        ret(any)
    }

    /// The decoded vocabulary in token-id order. Tokens are raw bytes and not
    /// necessarily valid UTF-8.
    pub fn decoded_vocab(&self) -> Result<Vec<Vec<u8>>> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.TokenizerInfo.decoded_vocab",
            self.raw,
        )?;
        let array = ret::<Array<FfiBytes>>(any)?;
        let mut vocab = Vec::with_capacity(array.len());
        for item in array.iter() {
            vocab.push(item.as_slice().to_vec());
        }
        Ok(vocab)
    }

    /// The token ids that stop generation.
    pub fn stop_token_ids(&self) -> Result<Vec<i64>> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.TokenizerInfo.stop_token_ids",
            self.raw,
        )?;
        let array = ret::<Array<i64>>(any)?;
        Ok(array.iter().collect())
    }

    /// The token ids of special tokens (control tokens the grammar never
    /// matches).
    pub fn special_token_ids(&self) -> Result<Vec<i64>> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.TokenizerInfo.special_token_ids",
            self.raw,
        )?;
        let array = ret::<Array<i64>>(any)?;
        Ok(array.iter().collect())
    }

    /// Dump the metadata (everything except the vocabulary) as a JSON string.
    pub fn dump_metadata(&self) -> Result<String> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.TokenizerInfo.dump_metadata",
            self.raw,
        )?;
        Ok(ret::<FfiString>(any)?.as_str().to_string())
    }

    /// Serialize to JSON (including the vocabulary).
    pub fn serialize_json(&self) -> Result<String> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.TokenizerInfo.serialize_json",
            self.raw,
        )?;
        Ok(ret::<FfiString>(any)?.as_str().to_string())
    }

    /// Deserialize from the JSON produced by
    /// [`TokenizerInfo::serialize_json`].
    pub fn deserialize_json(json: &str) -> Result<Self> {
        let any = ffi_call!(
            "xgrammar.tvm_ffi_binding.TokenizerInfo.deserialize_json",
            FfiString::from(json),
        )?;
        Ok(Self::from_raw(ret(any)?))
    }
}

impl std::fmt::Debug for TokenizerInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TokenizerInfo({:p})", handle::handle_ptr(&self.raw))
    }
}

/// Handle identity, like the Python `TokenizerInfo.__eq__`.
impl PartialEq for TokenizerInfo {
    fn eq(&self, other: &Self) -> bool {
        handle::same_handle(&self.raw, &other.raw)
    }
}
impl Eq for TokenizerInfo {}

// SAFETY: the underlying C++ object is immutable after construction, and the
// handle is a reference-counted pointer with atomic counts.
unsafe impl Send for TokenizerInfo {}
unsafe impl Sync for TokenizerInfo {}

fn detect_vocab_type_from_vocab(vocab: &HashMap<String, u32>) -> Option<VocabType> {
    let num_byte_pieces = (0..=u8::MAX)
        .filter(|byte| vocab.contains_key(&format!("<0x{byte:02X}>")))
        .count();
    if num_byte_pieces >= 128 {
        return Some(VocabType::ByteFallback);
    }

    let charset = byte_level_charset();
    let num_single_chars = charset
        .iter()
        .filter(|character| vocab.contains_key(&character.to_string()))
        .count();
    if num_single_chars < 128 {
        return None;
    }

    let num_charset_tokens = vocab
        .keys()
        .filter(|token| token.chars().all(|character| charset.contains(&character)))
        .count();
    if num_charset_tokens.saturating_mul(100) >= vocab.len().saturating_mul(99) {
        Some(VocabType::ByteLevel)
    } else {
        None
    }
}

fn byte_level_charset() -> HashSet<char> {
    let mut kept_bytes = [false; 256];
    let mut charset = HashSet::with_capacity(256);

    for byte in b'!'..=b'~' {
        kept_bytes[byte as usize] = true;
        charset.insert(char::from(byte));
    }
    for byte in 0xA1u32..=0xAC {
        kept_bytes[byte as usize] = true;
        charset.insert(char::from_u32(byte).expect("Latin-1 byte is a Unicode scalar"));
    }
    for byte in 0xAEu32..=0xFF {
        kept_bytes[byte as usize] = true;
        charset.insert(char::from_u32(byte).expect("Latin-1 byte is a Unicode scalar"));
    }

    let mut shifted = 0;
    for byte in 0..=u8::MAX {
        if !kept_bytes[byte as usize] {
            charset.insert(
                char::from_u32(256 + shifted)
                    .expect("the GPT-2 byte alphabet contains valid Unicode scalars"),
            );
            shifted += 1;
        }
    }
    charset
}

fn detect_add_prefix_space_by_encoding(
    tokenizer: &tokenizers::Tokenizer,
    vocab_type: VocabType,
) -> bool {
    let Ok(encoding) = tokenizer.encode("a", false) else {
        return false;
    };
    let Some(first_token) = encoding.get_tokens().first() else {
        return false;
    };
    let prefix = match vocab_type {
        VocabType::ByteFallback => '▁',
        VocabType::ByteLevel => 'Ġ',
        VocabType::Raw => return false,
    };
    first_token.starts_with(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_byte_fallback_from_vocabulary() {
        let vocab = (0..128u32)
            .map(|byte| (format!("<0x{byte:02X}>"), byte))
            .collect();
        assert_eq!(
            detect_vocab_type_from_vocab(&vocab),
            Some(VocabType::ByteFallback)
        );
    }

    #[test]
    fn detects_byte_level_from_vocabulary() {
        let vocab = byte_level_charset()
            .into_iter()
            .enumerate()
            .map(|(token_id, character)| (character.to_string(), token_id as u32))
            .collect();
        assert_eq!(
            detect_vocab_type_from_vocab(&vocab),
            Some(VocabType::ByteLevel)
        );
    }
}
