#![cfg(feature = "hf")]
#![allow(clippy::type_complexity)]

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

fn cases_model_vocab() -> Box<[(&'static str, xgrammar::VocabType, bool)]> {
    vec![
        ("luodian/llama-7b-hf", xgrammar::VocabType::BYTE_FALLBACK, true),
        ("meta-llama/Llama-2-7b-chat-hf", xgrammar::VocabType::BYTE_FALLBACK, true),
        ("meta-llama/Meta-Llama-3-8B-Instruct", xgrammar::VocabType::BYTE_LEVEL, false),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", xgrammar::VocabType::BYTE_LEVEL, false),
        ("NousResearch/Hermes-2-Theta-Llama-3-70B", xgrammar::VocabType::BYTE_LEVEL, false),
        ("NousResearch/Hermes-3-Llama-3.1-8B", xgrammar::VocabType::BYTE_LEVEL, false),
        ("google/gemma-2b-it", xgrammar::VocabType::BYTE_FALLBACK, true),
        ("CohereForAI/aya-23-8B", xgrammar::VocabType::BYTE_LEVEL, false),
        ("deepseek-ai/DeepSeek-Coder-V2-Instruct", xgrammar::VocabType::BYTE_LEVEL, true),
        ("deepseek-ai/DeepSeek-V2-Chat-0628", xgrammar::VocabType::BYTE_LEVEL, true),
        ("deepseek-ai/deepseek-coder-7b-instruct-v1.5", xgrammar::VocabType::BYTE_LEVEL, true),
        ("microsoft/phi-2", xgrammar::VocabType::BYTE_LEVEL, false),
        ("microsoft/Phi-3-mini-4k-instruct", xgrammar::VocabType::BYTE_FALLBACK, true),
        ("microsoft/Phi-3.5-mini-instruct", xgrammar::VocabType::BYTE_FALLBACK, true),
        ("Qwen/Qwen1.5-4B-Chat", xgrammar::VocabType::BYTE_LEVEL, false),
        ("Qwen/Qwen2-7B-Instruct", xgrammar::VocabType::BYTE_LEVEL, false),
        ("meta-llama/Llama-3.2-1B", xgrammar::VocabType::BYTE_LEVEL, false),
        ("google/gemma-2-2b-it", xgrammar::VocabType::BYTE_FALLBACK, true),
        ("deepseek-ai/DeepSeek-V2.5", xgrammar::VocabType::BYTE_LEVEL, true),
        ("Qwen/Qwen2.5-1.5B", xgrammar::VocabType::BYTE_LEVEL, false),
        ("mistralai/Mixtral-8x22B-Instruct-v0.1", xgrammar::VocabType::BYTE_FALLBACK, true),
        ("deepseek-ai/DeepSeek-R1", xgrammar::VocabType::BYTE_LEVEL, true),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", xgrammar::VocabType::BYTE_LEVEL, true),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", xgrammar::VocabType::BYTE_LEVEL, true),
    ]
    .into_boxed_slice()
}

#[test]
#[serial]
fn test_build_tokenizer_info() {
    for (model_id, _vocab_type, _add_prefix_space) in cases_model_vocab() {
        let path =
            download_tokenizer_json(model_id).expect("download tokenizer.json");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let tokenizer_info =
            xgrammar::TokenizerInfo::from_huggingface(&tokenizer, None, None).unwrap();
        assert!(tokenizer_info.vocab_size() > 0, "{}", model_id);
    }
}

#[test]
#[serial]
fn test_properties() {
    for (model_id, _vocab_type, add_prefix_space) in cases_model_vocab() {
        let path =
            download_tokenizer_json(model_id).expect("download tokenizer.json");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let tokenizer_info =
            xgrammar::TokenizerInfo::from_huggingface(&tokenizer, None, None).unwrap();
        let vocab = tokenizer.get_vocab(true);
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        assert_eq!(tokenizer_info.vocab_size(), std::cmp::max(vocab.len(), max_id + 1));
        assert_eq!(
            tokenizer_info.add_prefix_space(),
            add_prefix_space,
            "add_prefix_space mismatch for {}",
            model_id
        );
    }
}

#[test]
#[serial]
fn test_decoded_vocab() {
    for (model_id, _vocab_type, _add_prefix_space) in cases_model_vocab() {
        let path =
            download_tokenizer_json(model_id).expect("download tokenizer.json");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let tokenizer_info =
            xgrammar::TokenizerInfo::from_huggingface(&tokenizer, None, None).unwrap();
        let decoded = tokenizer_info.decoded_vocab();
        let vocab = tokenizer.get_vocab(true);
        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        assert_eq!(decoded.len(), std::cmp::max(vocab.len(), max_id + 1));
        assert_eq!(decoded.len(), tokenizer_info.vocab_size());
    }
}

#[test]
#[serial]
fn test_model_vocab_size_smaller_than_tokenizer() {
    let cases: &[(&str, usize)] = &[
        ("meta-llama/Llama-3.2-11B-Vision-Instruct", 128256),
        ("meta-llama/Llama-Guard-3-11B-Vision", 128256),
        ("allenai/Molmo-72B-0924", 152064),
    ];

    for (tokenizer_path, model_vocab_size) in cases {
        let path = download_tokenizer_json(tokenizer_path)
            .expect("download tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let ordered = extract_ordered_vocab(&tokenizer);
        let original_vocab_size = ordered.len();

        assert!(
            original_vocab_size > *model_vocab_size,
            "Original vocab size {} should be > model vocab size {} for {}",
            original_vocab_size,
            model_vocab_size,
            tokenizer_path
        );

        let tokenizer_info = xgrammar::TokenizerInfo::from_huggingface(
            &tokenizer,
            Some(*model_vocab_size),
            None,
        )
        .unwrap();

        assert!(
            tokenizer_info.vocab_size() == *model_vocab_size
                || tokenizer_info.vocab_size() == *model_vocab_size + 1,
            "vocab_size should be {} or {}, got {} for {}",
            model_vocab_size,
            model_vocab_size + 1,
            tokenizer_info.vocab_size(),
            tokenizer_path
        );
        assert!(
            tokenizer_info.decoded_vocab().len() >= *model_vocab_size,
            "decoded_vocab len should be >= {} for {}",
            model_vocab_size,
            tokenizer_path
        );
    }
}

#[test]
#[serial]
fn test_stop_token_ids() {
    for (model_id, _vocab_type, _add_prefix_space) in cases_model_vocab() {
        let path = download_tokenizer_json(model_id)
            .expect("download tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let tokenizer_info =
            xgrammar::TokenizerInfo::from_huggingface(&tokenizer, None, None).unwrap();
        let stops = tokenizer_info.stop_token_ids();
        println!("Model: {}, stop_token_ids: {:?}", model_id, stops);
    }
}

#[test]
#[serial]
fn test_dump_metadata_load() {
    for (model_id, _vocab_type, _add_prefix_space) in cases_model_vocab() {
        let path =
            download_tokenizer_json(model_id).expect("download tokenizer.json");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let tokenizer_info =
            xgrammar::TokenizerInfo::from_huggingface(&tokenizer, None, None).unwrap();
        let metadata = tokenizer_info.dump_metadata();
        let ordered = extract_ordered_vocab(&tokenizer);
        let loaded = xgrammar::TokenizerInfo::from_vocab_and_metadata_bytes(
            ordered.iter().map(|s| s.as_bytes()),
            &metadata,
        );
        assert_eq!(loaded.decoded_vocab(), tokenizer_info.decoded_vocab());
    }
}

#[test]
#[serial]
fn test_dump_metadata_load_with_expected_metadata() {
    let cases: &[(&str, &str)] = &[
        (
            "meta-llama/Llama-2-7b-chat-hf",
            r#"{"vocab_type":1,"vocab_size":32000,"add_prefix_space":true,"stop_token_ids":[2]}"#,
        ),
        (
            "meta-llama/Meta-Llama-3-8B-Instruct",
            r#"{"vocab_type":2,"vocab_size":128256,"add_prefix_space":false,"stop_token_ids":[128001,128009]}"#,
        ),
    ];

    for (model_id, expected_metadata) in cases {
        let path = download_tokenizer_json(model_id).expect("download tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let tokenizer_info =
            xgrammar::TokenizerInfo::from_huggingface(&tokenizer, None, None).unwrap();
        assert_eq!(
            tokenizer_info.dump_metadata(),
            *expected_metadata,
            "metadata mismatch for {}",
            model_id
        );

        let ordered = extract_ordered_vocab(&tokenizer);
        let loaded = xgrammar::TokenizerInfo::from_vocab_and_metadata_bytes(
            ordered.iter().map(|s| s.as_bytes()),
            expected_metadata,
        );
        assert_eq!(loaded.decoded_vocab(), tokenizer_info.decoded_vocab());
    }
}

#[test]
#[serial]
fn test_customize_stop_token_ids() {
    let model_ids = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ];

    for model_id in model_ids {
        let path =
            download_tokenizer_json(model_id).expect("download tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let stop_ids = [1i32, 2i32, 3i32];
        let tokenizer_info = xgrammar::TokenizerInfo::from_huggingface(
            &tokenizer,
            None,
            Some(&stop_ids[..]),
        )
        .unwrap();
        assert_eq!(
            tokenizer_info.stop_token_ids().as_ref(),
            &stop_ids,
            "stop_token_ids mismatch for {}",
            model_id
        );
    }
}

#[test]
#[serial]
fn test_padding_vocab_size() {
    let model_ids = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ];

    for model_id in model_ids {
        let path =
            download_tokenizer_json(model_id).expect("download tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let vocab = tokenizer.get_vocab(true);
        let original_vocab_size = vocab.len();
        let pad_by = 5usize;

        let tokenizer_info = xgrammar::TokenizerInfo::from_huggingface(
            &tokenizer,
            Some(original_vocab_size + pad_by),
            None,
        )
        .unwrap();

        assert_eq!(
            tokenizer_info.vocab_size(),
            original_vocab_size + pad_by,
            "vocab_size mismatch for {}",
            model_id
        );

        let specials = tokenizer_info.special_token_ids();
        let last_five: Vec<i32> = (0..pad_by)
            .map(|i| (original_vocab_size + i) as i32)
            .collect();
        for expected_id in &last_five {
            assert!(
                specials.contains(expected_id),
                "special_token_ids should contain {} for {}",
                expected_id,
                model_id
            );
        }
    }
}

#[test]
fn test_special_token_detection() {
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

fn cases_model_decode_text() -> Box<[&'static str]> {
    vec![
        "luodian/llama-7b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "NousResearch/Hermes-2-Theta-Llama-3-70B",
        "NousResearch/Hermes-3-Llama-3.1-8B",
        "CohereForAI/aya-23-8B",
        "microsoft/phi-2",
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "Qwen/Qwen1.5-4B-Chat",
        "Qwen/Qwen2-7B-Instruct",
        "meta-llama/Llama-3.2-1B",
        "Qwen/Qwen2.5-1.5B",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
    ]
    .into_boxed_slice()
}

#[test]
#[serial]
fn test_decode_text() {
    let text = "Hello 你好 こんにちは 안녕하세요! 🌎🌍🌏 \u{0300}\u{0301}\u{0302} \u{1f600}\u{1f601}\u{1f602} αβγδ АБВГД عربي עברית\n\t\r Special chars: &*()_+-=[]{}|;:'\",.<>?/\\~`!@#$%^<think>haha</think>";

    for model_id in cases_model_decode_text() {
        let path = download_tokenizer_json(model_id)
            .expect("download tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let tokenizer_info = xgrammar::TokenizerInfo::from_huggingface(&tokenizer, None, None).unwrap();
        let decoded_vocab = tokenizer_info.decoded_vocab();

        let encoding = tokenizer.encode(text, false).expect("encode text");
        let token_ids = encoding.get_ids();

        let mut recovered_bytes = Vec::new();
        for &token_id in token_ids {
            recovered_bytes.extend_from_slice(&decoded_vocab[token_id as usize]);
        }
        let recovered_text = String::from_utf8(recovered_bytes).expect("valid utf-8");

        let trial_encoding = tokenizer.encode("a", false).expect("encode trial");
        let trial_ids = trial_encoding.get_ids();
        let mut trial_bytes = Vec::new();
        for &token_id in trial_ids {
            trial_bytes.extend_from_slice(&decoded_vocab[token_id as usize]);
        }
        let trial_roundtrip = String::from_utf8(trial_bytes).expect("valid utf-8");

        assert!(trial_roundtrip.ends_with('a'), "model: {}", model_id);
        let detected_prefix = &trial_roundtrip[..trial_roundtrip.len() - 1];

        let actual_adds_space = !detected_prefix.is_empty() && detected_prefix.ends_with(' ');
        assert_eq!(
            tokenizer_info.add_prefix_space(),
            actual_adds_space,
            "add_prefix_space mismatch for {}",
            model_id
        );

        let expected = format!("{}{}", detected_prefix, text);
        assert_eq!(recovered_text, expected, "recovered text mismatch for {}", model_id);
    }
}

/// Test specific token ids map to expected byte sequences
#[test]
#[serial]
fn test_vocab_conversion() {
    let cases: &[(&str, &[(i32, &[u8])])] = &[
        (
            "meta-llama/Llama-2-7b-chat-hf",
            &[
                (4, b"\x01"),
                (259, b"  "),
                (261, b"er"),
                (20565, " исследова".as_bytes()),
            ],
        ),
        (
            "meta-llama/Meta-Llama-3-8B-Instruct",
            &[
                (1, b"\""),
                (37046, "我".as_bytes()),
                (40508, " automotive".as_bytes()),
            ],
        ),
    ];

    for (model_id, token_cases) in cases {
        let path = download_tokenizer_json(model_id).expect("download tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
        let tokenizer_info = xgrammar::TokenizerInfo::from_huggingface(&tokenizer, None, None).unwrap();
        let vocab = tokenizer_info.decoded_vocab();

        for &(token_id, expected_bytes) in *token_cases {
            assert_eq!(
                &*vocab[token_id as usize],
                expected_bytes,
                "model={}, token_id={}",
                model_id,
                token_id
            );
        }
    }
}
