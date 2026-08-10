//! Tokenizer metadata needed to compile grammars against a model vocabulary.

use tvm_ffi::{Array, Bytes as FfiBytes, String as FfiString};

use crate::error::{Error, Result};
use crate::ffi::objects::RawTokenizerInfo;
use crate::ffi::{ffi_call, opt_any, ret};

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

/// The tokenizer information xgrammar needs: the decoded vocabulary and
/// metadata such as the vocabulary encoding and the stop token ids.
///
/// This class is immutable: instances can be cloned cheaply (shared handle)
/// and shared across threads.
///
/// Unlike Python's `TokenizerInfo.from_huggingface`, the Rust bindings do not
/// inspect tokenizer objects; construct instances from an encoded vocabulary
/// with [`TokenizerInfo::new`], or from a vocabulary plus a metadata string
/// with [`TokenizerInfo::from_vocab_and_metadata`].
#[derive(Clone)]
pub struct TokenizerInfo {
    pub(crate) raw: RawTokenizerInfo,
}

impl TokenizerInfo {
    pub(crate) fn from_raw(raw: RawTokenizerInfo) -> Self {
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
        write!(f, "TokenizerInfo({:?})", self.raw)
    }
}

/// Handle identity, like the Python `TokenizerInfo.__eq__`.
impl PartialEq for TokenizerInfo {
    fn eq(&self, other: &Self) -> bool {
        self.raw.same_as(&other.raw)
    }
}
impl Eq for TokenizerInfo {}
