//! Error types for the XGrammar Rust bindings.
//!
//! C++ exceptions cross the FFI boundary as `tvm_ffi::Error` values carrying a
//! *kind* string. The kinds raised by xgrammar (see `cpp/support/exception.h`)
//! are mapped onto dedicated variants; everything else is preserved verbatim in
//! [`Error::Ffi`], including its backtrace.

use std::fmt;

/// The error type shared by every fallible operation in this crate.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A string was not valid JSON (kind `InvalidJSONError`).
    InvalidJson(String),
    /// A JSON schema could not be converted to a grammar (kind `InvalidJSONSchemaError`).
    InvalidJsonSchema(String),
    /// A structural tag definition was rejected (kind `InvalidStructuralTagError`).
    InvalidStructuralTag(String),
    /// Serialized data did not have the expected format (kind `DeserializeFormatError`).
    DeserializeFormat(String),
    /// Serialized data was produced by an incompatible version (kind `DeserializeVersionError`).
    DeserializeVersion(String),
    /// A generic xgrammar error (kind `XGrammarError`).
    XGrammar(String),
    /// The bindings library could not be located or loaded.
    Library(String),
    /// Serialization of a structural tag failed before reaching the FFI.
    Json(serde_json::Error),
    /// Loading, serializing, or running a Hugging Face tokenizer failed.
    HuggingFaceTokenizer(tokenizers::Error),
    /// Any other error raised across the FFI boundary (e.g. `RuntimeError`).
    Ffi(tvm_ffi::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn from_ffi(err: tvm_ffi::Error) -> Self {
        let message = err.message().to_string();
        match err.kind().as_str() {
            "InvalidJSONError" => Error::InvalidJson(message),
            "InvalidJSONSchemaError" => Error::InvalidJsonSchema(message),
            "InvalidStructuralTagError" => Error::InvalidStructuralTag(message),
            "DeserializeFormatError" => Error::DeserializeFormat(message),
            "DeserializeVersionError" => Error::DeserializeVersion(message),
            "XGrammarError" => Error::XGrammar(message),
            _ => Error::Ffi(err),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidJson(m) => write!(f, "invalid JSON: {m}"),
            Error::InvalidJsonSchema(m) => write!(f, "invalid JSON schema: {m}"),
            Error::InvalidStructuralTag(m) => write!(f, "invalid structural tag: {m}"),
            Error::DeserializeFormat(m) => write!(f, "deserialize format error: {m}"),
            Error::DeserializeVersion(m) => write!(f, "deserialize version error: {m}"),
            Error::XGrammar(m) => write!(f, "{m}"),
            Error::Library(m) => write!(f, "{m}"),
            Error::Json(e) => write!(f, "structural tag serialization failed: {e}"),
            Error::HuggingFaceTokenizer(e) => write!(f, "Hugging Face tokenizer error: {e}"),
            Error::Ffi(e) => write!(f, "{}: {}", e.kind().as_str(), e.message()),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Json(e) => Some(e),
            Error::HuggingFaceTokenizer(e) => Some(e.as_ref()),
            Error::Ffi(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Json(err)
    }
}

impl From<tokenizers::Error> for Error {
    fn from(err: tokenizers::Error) -> Self {
        Error::HuggingFaceTokenizer(err)
    }
}

impl From<tvm_ffi::Error> for Error {
    fn from(err: tvm_ffi::Error) -> Self {
        Error::from_ffi(err)
    }
}
