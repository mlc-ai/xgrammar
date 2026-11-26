mod common;

use xgrammar::{HuggingFaceTokenizerOptions, TokenizerInfo, TokenizerInfoOptions, VocabType};

#[test]
fn properties() {
    let ti = common::tiny_tokenizer_info();
    let vocab = common::tiny_vocab();

    assert_eq!(ti.vocab_size().unwrap(), vocab.len());
    assert_eq!(ti.vocab_type().unwrap(), VocabType::Raw);
    assert!(!ti.add_prefix_space().unwrap());
    assert_eq!(ti.stop_token_ids().unwrap(), vec![0]);
    assert_eq!(ti.decoded_vocab().unwrap(), vocab);
}

#[test]
fn vocab_size_padding() {
    common::init();
    // vocab_size larger than the provided vocabulary pads with empty tokens.
    let ti = TokenizerInfo::new(
        [b"a".as_slice(), b"b".as_slice()],
        VocabType::Raw,
        &TokenizerInfoOptions {
            vocab_size: Some(8),
            stop_token_ids: Some(vec![0]),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(ti.vocab_size().unwrap(), 8);
    // decoded_vocab holds only the real tokens; padding is virtual.
    assert_eq!(ti.decoded_vocab().unwrap().len(), 2);
}

#[test]
fn metadata_roundtrip() {
    let ti = common::tiny_tokenizer_info();
    let metadata = ti.dump_metadata().unwrap();
    assert!(metadata.contains("vocab_type"));

    let rebuilt = TokenizerInfo::from_vocab_and_metadata(common::tiny_vocab(), &metadata).unwrap();
    assert_eq!(rebuilt.vocab_size().unwrap(), ti.vocab_size().unwrap());
    assert_eq!(
        rebuilt.stop_token_ids().unwrap(),
        ti.stop_token_ids().unwrap()
    );
    assert_eq!(rebuilt.vocab_type().unwrap(), ti.vocab_type().unwrap());
}

#[test]
fn serialization_roundtrip() {
    let ti = common::tiny_tokenizer_info();
    let json = ti.serialize_json().unwrap();
    let restored = TokenizerInfo::deserialize_json(&json).unwrap();
    assert_eq!(restored.vocab_size().unwrap(), ti.vocab_size().unwrap());
    assert_eq!(
        restored.decoded_vocab().unwrap(),
        ti.decoded_vocab().unwrap()
    );
}

#[test]
fn detect_metadata_from_hf_rejects_garbage() {
    common::init();
    // A real HuggingFace backend string requires a tokenizer at hand; at
    // least verify the error path is a clean Rust error.
    let err = TokenizerInfo::detect_metadata_from_hf("not a tokenizer json").unwrap_err();
    assert!(!err.to_string().is_empty());
}

#[test]
fn byte_fallback_vocab_type() {
    common::init();
    let ti = TokenizerInfo::new(
        [b"<0x41>".as_slice(), b"a".as_slice()],
        VocabType::ByteFallback,
        &TokenizerInfoOptions {
            stop_token_ids: Some(vec![1]),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(ti.vocab_type().unwrap(), VocabType::ByteFallback);
    // Byte-fallback decoding turns <0x41> into the byte 0x41.
    assert_eq!(ti.decoded_vocab().unwrap()[0], b"A".to_vec());
}

#[test]
fn from_huggingface_tokenizer_json() {
    common::init();
    let tokenizer = common::huggingface_word_level_tokenizer();
    let ti = TokenizerInfo::from_huggingface(
        &tokenizer,
        &HuggingFaceTokenizerOptions {
            // Exercise a model head padded beyond tokenizer.get_vocab().
            vocab_size: Some(11),
            stop_token_ids: Some(vec![0]),
        },
    )
    .unwrap();

    assert_eq!(ti.vocab_type().unwrap(), VocabType::Raw);
    assert_eq!(ti.vocab_size().unwrap(), 11);
    assert_eq!(ti.stop_token_ids().unwrap(), vec![0]);
    let vocab = ti.decoded_vocab().unwrap();
    assert_eq!(vocab[2], b"a");
    assert_eq!(vocab[5], b"ab");
    assert_eq!(vocab[6], b"ac");
    // Added tokens come from get_vocab(true), even if they are not in the
    // model vocabulary serialized inside tokenizer.json.
    assert_eq!(vocab[8], b"<extra>");
    assert!(vocab[9].is_empty());
    assert!(vocab[10].is_empty());
}

#[test]
fn from_huggingface_rejects_explicitly_empty_stop_tokens() {
    let tokenizer = common::huggingface_word_level_tokenizer();
    let err = TokenizerInfo::from_huggingface(
        &tokenizer,
        &HuggingFaceTokenizerOptions {
            stop_token_ids: Some(vec![]),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("cannot be empty"));
}
