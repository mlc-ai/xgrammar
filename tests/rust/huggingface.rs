//! Tests against the same real Hugging Face tokenizers used by the Python
//! `hf_token_required` suite.

mod common;

use hf_hub::{api::sync::ApiBuilder, Cache};
use xgrammar::{
    GrammarCompiler, GrammarMatcher, HuggingFaceTokenizerOptions, TokenBitmask, TokenizerInfo,
    VocabType,
};

struct Scenario {
    model_id: &'static str,
    vocab_size: usize,
    vocab_type: VocabType,
    add_prefix_space: bool,
    stop_token_id: i64,
    vocab_samples: &'static [(usize, &'static [u8])],
    input: &'static str,
    expected_rejected_sizes: &'static [usize],
}

// Kept value-for-value in sync with
// tests/python/test_grammar_matcher_basic.py::test_fill_next_token_bitmask.
const LLAMA_2_REJECTED_SIZES: &[usize] = &[
    31989, 31912, 270, 270, 270, 31973, 31846, 31846, 31948, 31915, 270, 270, 270, 270, 270, 31973,
    31846, 31846, 263, 263, 263, 263, 263, 263, 263, 263, 31974, 31999,
];

const LLAMA_3_REJECTED_SIZES: &[usize] = &[
    128235, 127497, 4744, 4744, 4744, 127849, 126399, 126399, 126760, 127499, 4744, 4744, 4744,
    4744, 4744, 127849, 126399, 126399, 4694, 4694, 4694, 4694, 4694, 4694, 4694, 4694, 128066,
    128111, 4694, 128066, 128111, 4694, 127873, 128255,
];

const SCENARIOS: &[Scenario] = &[
    Scenario {
        model_id: "meta-llama/Llama-2-7b-chat-hf",
        vocab_size: 32_000,
        vocab_type: VocabType::ByteFallback,
        add_prefix_space: true,
        stop_token_id: 2,
        vocab_samples: &[
            (4, b"\x01"),
            (259, b"  "),
            (261, b"er"),
            (20565, " исследова".as_bytes()),
        ],
        input: "{\"id\": 1,\"name\": \"Example\"}",
        expected_rejected_sizes: LLAMA_2_REJECTED_SIZES,
    },
    Scenario {
        model_id: "meta-llama/Meta-Llama-3-8B-Instruct",
        vocab_size: 128_256,
        vocab_type: VocabType::ByteLevel,
        add_prefix_space: false,
        stop_token_id: 128_009,
        vocab_samples: &[
            (1, b"\""),
            (37046, "我".as_bytes()),
            (40508, b" automotive"),
        ],
        input: "{\"id\": 1,\"name\": \"Example哈哈\"}",
        expected_rejected_sizes: LLAMA_3_REJECTED_SIZES,
    },
];

fn auth_token(cache: &Cache) -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        .into_iter()
        .find_map(|name| std::env::var(name).ok().filter(|token| !token.is_empty()))
        .or_else(|| cache.token())
}

fn load_tokenizer(model_id: &str) -> xgrammar::tokenizers::Tokenizer {
    let cache = Cache::from_env();
    let tokenizer_path = match cache.model(model_id.to_string()).get("tokenizer.json") {
        Some(path) => path,
        None => {
            let token = auth_token(&cache).unwrap_or_else(|| {
                panic!(
                    "{model_id}/tokenizer.json is not cached; set HF_TOKEN before running this test"
                )
            });
            ApiBuilder::from_env()
                .with_token(Some(token))
                .with_progress(false)
                .build()
                .expect("build Hugging Face Hub client")
                .model(model_id.to_string())
                .get("tokenizer.json")
                .unwrap_or_else(|err| panic!("download {model_id}/tokenizer.json: {err}"))
        }
    };

    xgrammar::tokenizers::Tokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|err| panic!("load {}: {err}", tokenizer_path.display()))
}

/// Port of Python's real-tokenizer `test_fill_next_token_bitmask` plus the
/// representative vocabulary conversion assertions from `test_tokenizer_info`.
#[test]
#[ignore = "requires gated Hugging Face tokenizer access or a populated local cache"]
fn real_tokenizers_match_python_properties_and_mask_counts() {
    common::init();

    for scenario in SCENARIOS {
        let tokenizer = load_tokenizer(scenario.model_id);
        let tokenizer_info = TokenizerInfo::from_huggingface(
            &tokenizer,
            &HuggingFaceTokenizerOptions {
                stop_token_ids: Some(vec![scenario.stop_token_id]),
                ..Default::default()
            },
        )
        .unwrap_or_else(|err| panic!("convert {}: {err}", scenario.model_id));

        assert_eq!(
            tokenizer_info.vocab_size().unwrap(),
            scenario.vocab_size,
            "{} vocab size",
            scenario.model_id
        );
        assert_eq!(
            tokenizer_info.vocab_type().unwrap(),
            scenario.vocab_type,
            "{} vocab type",
            scenario.model_id
        );
        assert_eq!(
            tokenizer_info.add_prefix_space().unwrap(),
            scenario.add_prefix_space,
            "{} add_prefix_space",
            scenario.model_id
        );
        assert_eq!(
            tokenizer_info.stop_token_ids().unwrap(),
            vec![scenario.stop_token_id],
            "{} stop token",
            scenario.model_id
        );

        let decoded_vocab = tokenizer_info.decoded_vocab().unwrap();
        for &(token_id, expected) in scenario.vocab_samples {
            assert_eq!(
                decoded_vocab[token_id], expected,
                "{} token {token_id}",
                scenario.model_id
            );
        }

        assert_eq!(
            scenario.expected_rejected_sizes.len(),
            scenario.input.len() + 1,
            "invalid test vector for {}",
            scenario.model_id
        );
        let compiler = GrammarCompiler::new(&tokenizer_info).unwrap();
        let compiled = compiler.compile_builtin_json_grammar().unwrap();
        let mut matcher = GrammarMatcher::new(&compiled).unwrap();
        let mut bitmask = TokenBitmask::new(scenario.vocab_size);

        for (position, &byte) in scenario.input.as_bytes().iter().enumerate() {
            assert!(matcher.fill_next_token_bitmask(&mut bitmask, 0).unwrap());
            assert_eq!(
                bitmask.masked_tokens(0).len(),
                scenario.expected_rejected_sizes[position],
                "{} rejected-token count at byte {position} ({byte:#04x})",
                scenario.model_id
            );
            assert!(matcher.accept_string([byte]).unwrap());
        }

        assert!(matcher.fill_next_token_bitmask(&mut bitmask, 0).unwrap());
        assert_eq!(
            bitmask.masked_tokens(0).len(),
            *scenario.expected_rejected_sizes.last().unwrap(),
            "{} final rejected-token count",
            scenario.model_id
        );
    }
}
