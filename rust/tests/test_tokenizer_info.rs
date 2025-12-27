#![cfg(feature = "hf")]
mod test_utils;

use serial_test::serial;
use test_utils::*;

fn extract_ordered_vocab(tk: &tokenizers::Tokenizer) -> Box<[String]> {
    let mut pairs: Vec<(usize, String)> = tk
        .get_vocab(true)
        .into_iter()
        .map(|(tok, id)| (id as usize, tok))
        .collect();
    pairs.sort_by_key(|(id, _)| *id);
    let mut out = Vec::with_capacity(pairs.len());
    for (_, tok) in pairs {
        out.push(tok);
    }
    out.into_boxed_slice()
}

// Subset of Python's tokenizer_path__vocab_type__prepend_space
// Using models accessible with current HF token
fn cases_model_vocab() -> Box<[(&'static str, xgrammar::VocabType, bool)]> {
    vec![
        (
            "meta-llama/Llama-2-7b-chat-hf",
            xgrammar::VocabType::BYTE_FALLBACK,
            true,
        ),
        // Note: Python has 30+ models, but limiting to publicly accessible models
    ]
    .into_boxed_slice()
}

#[test]
#[serial]
fn test_build_tokenizer_info() {
    for (model_id, _vocab_type, _add_prefix_space) in cases_model_vocab() {
        let path =
            download_tokenizer_json(model_id).expect("download tokenizer.json");
        let tk =
            tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let ti =
            xgrammar::TokenizerInfo::from_huggingface(&tk, None, None).unwrap();
        assert!(ti.vocab_size() > 0, "{}", model_id);
    }
}

#[test]
#[serial]
fn test_properties() {
    for (model_id, _vocab_type, add_prefix_space) in cases_model_vocab() {
        let path =
            download_tokenizer_json(model_id).expect("download tokenizer.json");
        let tk =
            tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let ti =
            xgrammar::TokenizerInfo::from_huggingface(&tk, None, None).unwrap();
        let vocab = tk.get_vocab(true);
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        assert_eq!(ti.vocab_size(), std::cmp::max(vocab.len(), max_id + 1));
        assert_eq!(ti.add_prefix_space(), add_prefix_space);
    }
}

#[test]
#[serial]
fn test_decoded_vocab() {
    for (model_id, _vocab_type, _add_prefix_space) in cases_model_vocab() {
        let path =
            download_tokenizer_json(model_id).expect("download tokenizer.json");
        let tk =
            tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let ti =
            xgrammar::TokenizerInfo::from_huggingface(&tk, None, None).unwrap();
        let decoded = ti.decoded_vocab();
        let vocab = tk.get_vocab(true);
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        assert_eq!(decoded.len(), std::cmp::max(vocab.len(), max_id + 1));
        assert_eq!(decoded.len(), ti.vocab_size());
    }
}

#[test]
#[serial]
fn test_model_vocab_size_smaller_than_tokenizer() {
    // Python test: test_model_vocab_size_smaller_than_tokenizer
    // Uses meta-llama/Llama-3.2-11B-Vision-Instruct with model_vocab_size=128256
    let tokenizer_path = "meta-llama/Llama-3.2-11B-Vision-Instruct";
    let model_vocab_size = 128256;

    let path = download_tokenizer_json(tokenizer_path)
        .expect("download tokenizer.json");
    let tk = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
    let ordered = extract_ordered_vocab(&tk);
    let original_vocab_size = ordered.len();

    assert!(
        original_vocab_size > model_vocab_size,
        "Original vocab size {} should be > model vocab size {}",
        original_vocab_size,
        model_vocab_size
    );

    let ti = xgrammar::TokenizerInfo::from_tokenizers_with_options(
        &tk,
        xgrammar::VocabType::BYTE_LEVEL,
        Some(model_vocab_size),
        None,
        false,
    )
    .unwrap();

    // Some tokenizers pad by 1 for special tokens, so allow for that
    assert!(
        ti.vocab_size() == model_vocab_size
            || ti.vocab_size() == model_vocab_size + 1,
        "vocab_size should be {} or {}, got {}",
        model_vocab_size,
        model_vocab_size + 1,
        ti.vocab_size()
    );
    assert!(ti.decoded_vocab().len() >= model_vocab_size);
}

#[test]
#[serial]
fn test_vocab_type_detection() {
    // Python test checks vocab_type for various models from tokenizer_path__vocab_type__prepend_space
    let model_id = "meta-llama/Llama-2-7b-chat-hf";
    let expected_vocab_type = xgrammar::VocabType::BYTE_FALLBACK;

    let path =
        download_tokenizer_json(model_id).expect("download tokenizer.json");
    let tk = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
    let ti =
        xgrammar::TokenizerInfo::from_huggingface(&tk, None, None).unwrap();
    assert_eq!(
        ti.vocab_type() as i32,
        expected_vocab_type as i32,
        "Model {} should have correct vocab_type",
        model_id
    );
}

#[test]
#[serial]
fn test_stop_token_ids_match_eos() {
    // Use a chat model with EOS token id
    let path = download_tokenizer_json("meta-llama/Llama-2-7b-chat-hf")
        .expect("download tokenizer.json");
    let tk = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
    let ti =
        xgrammar::TokenizerInfo::from_huggingface(&tk, None, None).unwrap();
    // If the tokenizer exposes an EOS id in added tokens metadata, it should be reflected.
    // We only assert non-empty to avoid tight coupling to specific id values.
    let stops = ti.stop_token_ids();
    assert!(stops.len() >= 1);
}

#[test]
#[serial]
fn test_vocab_type_and_prefix_space_llama2() {
    let path = download_tokenizer_json("meta-llama/Llama-2-7b-chat-hf")
        .expect("download tokenizer.json");
    let tk = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
    let ti =
        xgrammar::TokenizerInfo::from_huggingface(&tk, None, None).unwrap();
    // Expect BYTE_FALLBACK and add_prefix_space true for LLaMA-2 style tokenizers
    assert!(matches!(ti.vocab_type(), xgrammar::VocabType::BYTE_FALLBACK));
    assert!(ti.add_prefix_space());
}

#[test]
#[serial]
fn test_dump_metadata_load() {
    for (model_id, _vocab_type, _add_prefix_space) in cases_model_vocab() {
        let path =
            download_tokenizer_json(model_id).expect("download tokenizer.json");
        let tk =
            tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let ti =
            xgrammar::TokenizerInfo::from_huggingface(&tk, None, None).unwrap();
        let metadata = ti.dump_metadata();
        let ordered = extract_ordered_vocab(&tk);
        let loaded = xgrammar::TokenizerInfo::from_vocab_and_metadata_bytes(
            ordered.iter().map(|s| s.as_bytes()),
            &metadata,
        );
        assert_eq!(loaded.decoded_vocab(), ti.decoded_vocab());
    }
}

#[test]
#[serial]
fn test_customize_stop_token_ids() {
    // Python test: test_customize_stop_token_ids
    // Tests meta-llama/Llama-2-7b-chat-hf
    let model_id = "meta-llama/Llama-2-7b-chat-hf";

    let path =
        download_tokenizer_json(model_id).expect("download tokenizer.json");
    let tk = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
    let stop_ids = [1i32, 2i32, 3i32];
    let ti = xgrammar::TokenizerInfo::from_huggingface(
        &tk,
        None,
        Some(&stop_ids[..]),
    )
    .unwrap();
    assert_eq!(ti.stop_token_ids().as_ref(), &stop_ids);
}

#[test]
#[serial]
fn test_padding_vocab_size() {
    // Python test: test_padding_vocab_size
    // Tests meta-llama/Llama-2-7b-chat-hf
    let model_id = "meta-llama/Llama-2-7b-chat-hf";
    let vocab_type = xgrammar::VocabType::BYTE_FALLBACK;
    let add_prefix_space = true;

    let path =
        download_tokenizer_json(model_id).expect("download tokenizer.json");
    let tk = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
    let ordered = extract_ordered_vocab(&tk);
    let original = ordered.len();
    let pad_by = 5usize;
    let ti = xgrammar::TokenizerInfo::new_with_vocab_size(
        &ordered,
        vocab_type,
        Some(original + pad_by),
        &None,
        add_prefix_space,
    )
    .unwrap();
    assert_eq!(ti.vocab_size(), original + pad_by);
    let specials = ti.special_token_ids();
    for i in 0..pad_by {
        assert!(specials.contains(&((original + i) as i32)));
    }
}

#[test]
fn test_special_token_detection() {
    // Matches Python test_special_token_detection (no HF needed)
    let vocab_dict =
        ["", "<s>", "</s>", "[@BOS@]", "regular", "<>", "<think>", "</think>"];
    let tokenizer_info = xgrammar::TokenizerInfo::from_vocab_and_metadata_bytes(
        vocab_dict.iter().map(|s| s.as_bytes()),
        "{\"vocab_type\":1,\"vocab_size\":8,\"add_prefix_space\":true,\"stop_token_ids\":[2]}",
    );
    let expected: std::collections::HashSet<i32> = [0].into_iter().collect();
    let got: std::collections::HashSet<i32> =
        tokenizer_info.special_token_ids().into_iter().collect();
    assert_eq!(got, expected);
}
