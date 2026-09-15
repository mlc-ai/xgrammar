//! Shared helpers for the integration tests.
#![allow(dead_code)] // not every test binary uses every helper

use std::sync::Once;

use xgrammar::{TokenizerInfo, TokenizerInfoOptions, VocabType, BINDINGS_LIBRARY_FILENAME};

/// Load the freshly built bindings library from the repository checkout.
pub fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../python/xgrammar")
            .join(BINDINGS_LIBRARY_FILENAME);
        xgrammar::load_library(path.to_str().expect("bindings library path is not UTF-8"))
            .unwrap_or_else(|err| {
                panic!(
                    "cannot load {BINDINGS_LIBRARY_FILENAME}; build it first \
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

/// A fully offline Hugging Face `tokenizer.json` fixture. Loading it through
/// the official `tokenizers` crate exercises the same representation used by
/// downloaded fast tokenizers without making the test suite depend on the
/// network or the Hub cache.
pub fn huggingface_word_level_tokenizer() -> xgrammar::tokenizers::Tokenizer {
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {
                "id": 8,
                "content": "<extra>",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }
        ],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": {
            "type": "WordPiece",
            "prefix": "##",
            "cleanup": true
        },
        "model": {
            "type": "WordLevel",
            "vocab": {
                "</s>": 0,
                "[UNK]": 1,
                "a": 2,
                "b": 3,
                "c": 4,
                "ab": 5,
                "ac": 6,
                "x": 7
            },
            "unk_token": "[UNK]"
        }
    });
    xgrammar::tokenizers::Tokenizer::from_bytes(
        serde_json::to_vec(&tokenizer_json).expect("serialize tokenizer fixture"),
    )
    .expect("load tokenizer fixture")
}
