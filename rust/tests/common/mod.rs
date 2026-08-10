//! Shared helpers for the integration tests.
#![allow(dead_code)] // not every test binary uses every helper

use std::sync::Once;

use xgrammar::{TokenizerInfo, TokenizerInfoOptions, VocabType};

#[cfg(target_os = "macos")]
const LIB_BASENAME: &str = "libxgrammar_bindings.dylib";
#[cfg(target_os = "windows")]
const LIB_BASENAME: &str = "xgrammar_bindings.dll";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const LIB_BASENAME: &str = "libxgrammar_bindings.so";

/// Load the freshly built bindings library from the repository checkout.
pub fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../python/xgrammar")
            .join(LIB_BASENAME);
        xgrammar::load_library(path.to_str().expect("bindings library path is not UTF-8"))
            .unwrap_or_else(|err| {
                panic!(
                    "cannot load {LIB_BASENAME}; build it first \
                     (cmake --build <build-dir> --target xgrammar_bindings): {err}"
                )
            });
    });
}

/// A small hand-written byte-level vocabulary good enough to exercise the
/// whole pipeline: single printable ASCII bytes plus a few multi-byte tokens
/// and an end-of-sequence token.
pub fn tiny_vocab() -> Vec<Vec<u8>> {
    let mut vocab: Vec<Vec<u8>> = Vec::new();
    vocab.push(b"</s>".to_vec()); // 0: EOS
    for b in 0x20u8..0x7f {
        vocab.push(vec![b]); // 1..=95: printable ASCII
    }
    vocab.push(b"true".to_vec()); // 96
    vocab.push(b"false".to_vec()); // 97
    vocab.push(b"null".to_vec()); // 98
    vocab.push(b"\": \"".to_vec()); // 99
    vocab.push(b"\", \"".to_vec()); // 100
    vocab
}

/// Token id of a single-byte token in [`tiny_vocab`].
pub fn byte_token(b: u8) -> i64 {
    assert!((0x20..0x7f).contains(&b));
    (b - 0x20) as i64 + 1
}

/// A [`TokenizerInfo`] over [`tiny_vocab`] with token 0 as the stop token.
pub fn tiny_tokenizer_info() -> TokenizerInfo {
    init();
    TokenizerInfo::new(
        tiny_vocab(),
        VocabType::Raw,
        &TokenizerInfoOptions {
            stop_token_ids: Some(vec![0]),
            ..Default::default()
        },
    )
    .expect("tokenizer info")
}
