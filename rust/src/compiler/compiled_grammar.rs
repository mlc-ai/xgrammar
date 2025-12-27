use std::pin::Pin;

use autocxx::prelude::*;

use crate::{
    CxxUniquePtr, FFICompiledGrammar, Grammar, TokenizerInfo, cxx_ulong, cxx_ulonglong,
    cxx_utils,
};

/// This is the primary object to store compiled grammar.
///
/// A `CompiledGrammar` can be used to construct `GrammarMatcher` to generate token masks
/// efficiently.
///
/// # Notes
///
/// Do not construct this class directly, instead use `GrammarCompiler` to construct the object.
pub struct CompiledGrammar {
    inner: CxxUniquePtr<FFICompiledGrammar>,
}

impl CompiledGrammar {
    /// The original grammar.
    pub fn grammar(&self) -> Grammar {
        let inner_ref = self.inner.as_ref().expect("CompiledGrammar inner is null");
        Grammar::from_unique_ptr(inner_ref.GetGrammar().within_unique_ptr())
    }

    /// The tokenizer info associated with the compiled grammar.
    pub fn tokenizer_info(&self) -> TokenizerInfo {
        let inner_ref = self.inner.as_ref().expect("CompiledGrammar inner is null");
        TokenizerInfo::from_unique_ptr(inner_ref.GetTokenizerInfo().within_unique_ptr())
    }

    /// The approximate memory usage of the compiled grammar in bytes.
    pub fn memory_size_bytes(&self) -> usize {
        trait ToUsize {
            fn to_usize(self) -> usize;
        }

        impl ToUsize for usize {
            fn to_usize(self) -> usize {
                self
            }
        }

        #[cfg(target_os = "windows")]
        impl ToUsize for cxx_ulong {
            fn to_usize(self) -> usize {
                self.0 as usize
            }
        }

        #[cfg(not(target_os = "windows"))]
        impl ToUsize for cxx_ulong {
            fn to_usize(self) -> usize {
                let val: u64 = self.into();
                val as usize
            }
        }

        #[cfg(target_os = "windows")]
        impl ToUsize for cxx_ulonglong {
            fn to_usize(self) -> usize {
                let val: u64 = self.0.into();
                val as usize
            }
        }

        #[cfg(not(target_os = "windows"))]
        impl ToUsize for cxx_ulonglong {
            fn to_usize(self) -> usize {
                let val: u64 = self.into();
                val as usize
            }
        }

        let inner_ref = self.inner.as_ref().expect("CompiledGrammar inner is null");
        let sz = inner_ref.MemorySizeBytes().to_usize();
        sz
    }

    /// Serialize the compiled grammar to a JSON string. It will serialize the compiled grammar
    /// without the tokenizer info, since the tokenizer info is shared by multiple compiled
    /// grammars.
    ///
    /// # Notes
    ///
    /// The metadata of the tokenizer info is serialized and will be checked when deserializing.
    ///
    /// # Returns
    ///
    /// The JSON string.
    pub fn serialize_json(&self) -> String {
        let inner_ref = self.inner.as_ref().expect("CompiledGrammar inner is null");
        inner_ref.SerializeJSON().to_string()
    }

    /// Deserialize the compiled grammar from a JSON string and associate it with the specified
    /// tokenizer info.
    ///
    /// # Notes
    ///
    /// This will check the metadata of the tokenizer info matching the serialized metadata in
    /// `json`. If the metadata does not match, an error will be returned.
    ///
    /// # Parameters
    ///
    /// - `json`: The JSON string.
    /// - `tokenizer_info`: The tokenizer info.
    ///
    /// # Returns
    ///
    /// The compiled grammar.
    ///
    /// # Errors
    ///
    /// - When the JSON string is invalid.
    /// - When the JSON string does not follow the serialization format of the grammar, or the
    ///   tokenizer info metadata does not match.
    /// - When the `__VERSION__` field in the JSON string is not the same as the current version.
    pub fn deserialize_json(
        json: &str,
        tokenizer_info: &TokenizerInfo,
    ) -> Result<Self, String> {
        cxx::let_cxx_string!(json_cxx = json);
        cxx::let_cxx_string!(error_out_cxx = "");
        let unique_ptr = unsafe {
            cxx_utils::compiled_grammar_deserialize_json_or_error(
                &json_cxx,
                tokenizer_info.ffi_ref(),
                error_out_cxx.as_mut().get_unchecked_mut(),
            )
        };
        if unique_ptr.is_null() {
            return Err(error_out_cxx.to_string());
        }
        Ok(Self { inner: unique_ptr })
    }

    pub(crate) fn from_unique_ptr(inner: cxx::UniquePtr<FFICompiledGrammar>) -> Self {
        Self { inner }
    }

    pub(crate) fn ffi_ref(&self) -> &FFICompiledGrammar {
        self.inner.as_ref().expect("CompiledGrammar inner is null")
    }
}

impl Drop for CompiledGrammar {
    fn drop(&mut self) {
    }
}
