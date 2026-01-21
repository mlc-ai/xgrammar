#![allow(clippy::type_complexity, clippy::useless_vec)]

mod test_utils;

use serial_test::serial;
use test_utils::*;
use xgrammar::{Grammar, TokenizerInfo, VocabType};
#[cfg(feature = "hf")]
use xgrammar::{
    BatchGrammarMatcher, GrammarCompiler, GrammarMatcher, allocate_token_bitmask,
};

#[cfg(feature = "hf")]
fn get_masked_tokens_from_bitmask(
    bitmask: &[i32],
    vocab_size: usize,
) -> Vec<usize> {
    (0..vocab_size)
        .filter(|&i| !is_token_accepted_helper(i as i32, bitmask))
        .collect()
}

#[derive(Debug)]
enum TestInput<'a> {
    Str(&'a str),
    Bytes(&'a [u8]),
}

#[test]
#[serial]
fn test_accept_string() {
    let cases: &[(&str, TestInput, bool)] = &[
        ("root ::= [^a]+", TestInput::Str("bbb"), true),
        ("root ::= [^a]+", TestInput::Str("bba"), false),
        ("root ::= [^a]+", TestInput::Str("©"), true),
        ("root ::= [^a]+", TestInput::Bytes(b"\xe2\xa1\xa1"), true),
        ("root ::= [^a]+", TestInput::Bytes(b"\xe2\xa1\xa1\xa1"), false),
        ("root ::= [^a]+", TestInput::Bytes(b"\xe2\xa1\xe2\xa1"), false),
    ];

    for (ebnf, input, accepted) in cases {
        let grammar = Grammar::from_ebnf(ebnf, "root").unwrap();
        let mut matcher = matcher_from_grammar(&grammar);
        let result = match input {
            TestInput::Str(s) => matcher.accept_string(s, false),
            TestInput::Bytes(b) => matcher.accept_bytes(b, false),
        };
        assert_eq!(result, *accepted, "ebnf: {}, input: {:?}", ebnf, input);
    }
}

#[test]
#[serial]
fn test_grammar_accept() {
    let json_grammar = Grammar::builtin_json_grammar();
    let inputs = ["{\"name\": \"John\"}", "{ \"name\" : \"John\" }"];

    for input in &inputs {
        let mut matcher = matcher_from_grammar(&json_grammar);
        assert!(matcher.accept_string(input, false));
        assert!(matcher.is_terminated());
    }
}

#[test]
#[serial]
fn test_grammar_refuse() {
    let json_grammar = Grammar::builtin_json_grammar();
    let inputs = ["{ name: \"John\" }", "{ \"name\": \"John\" } "];

    for input in &inputs {
        let mut matcher = matcher_from_grammar(&json_grammar);
        let result = matcher.accept_string(input, false);
        let terminated = matcher.is_terminated();
        assert!(!result || !terminated, "Input should be refused: {}", input);
    }
}

#[test]
#[serial]
fn test_debug_print_internal_state() {
    let json_grammar = Grammar::builtin_json_grammar();
    let mut matcher = matcher_from_grammar(&json_grammar);
    let input = "{\"name\": \"John\"}";
    for ch in input.chars() {
        let s = ch.to_string();
        assert!(
            matcher.accept_string(&s, false),
            "Failed to accept character: {:?}",
            ch
        );
        let state = matcher.debug_print_internal_state();
        assert!(!state.is_empty());
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_fill_next_token_bitmask() {
    let cases: Vec<(&str, &str, Option<Box<[usize]>>)> = vec![
        (
            "meta-llama/Llama-2-7b-chat-hf",
            r#"{"id": 1,"name": "Example"}"#,
            Some(
                [
                    31989, 31912, 270, 270, 270, 31973, 31846, 31846, 31948, 31915, 270, 270,
                    270, 270, 270, 31973, 31846, 31846, 263, 263, 263, 263, 263, 263, 263, 263,
                    31974, 31999,
                ]
                .into(),
            ),
        ),
        (
            // test for llama 3
            "meta-llama/Meta-Llama-3-8B-Instruct",
            r#"{"id": 1,"name": "Example哈哈"}"#,
            None,
        ),
    ];

    for (tokenizer_path, input_str, expected_rejected_sizes) in cases {
        let tokenizer_info = make_hf_tokenizer_info(tokenizer_path);
        let json_grammar = Grammar::builtin_json_grammar();
        let mut matcher =
            matcher_from_grammar_with_tokenizer(&json_grammar, &tokenizer_info);

        let mut token_bitmask = allocate_token_bitmask(1, tokenizer_info.vocab_size());
        let (mut tensor, _shape, _strides) = create_bitmask_dltensor(
            &mut token_bitmask,
            1,
            tokenizer_info.vocab_size(),
        );

        let input_bytes = input_str.as_bytes();
        let mut rejected_sizes = Vec::new();

        for (i, c) in input_bytes.iter().enumerate() {
            matcher.fill_next_token_bitmask(&mut tensor, 0, false);
            let rejected_token_ids =
                get_masked_tokens_from_bitmask(&token_bitmask, tokenizer_info.vocab_size());
            rejected_sizes.push(rejected_token_ids.len());
            if let Some(expected) = expected_rejected_sizes.as_ref() {
                assert_eq!(
                    rejected_sizes[rejected_sizes.len() - 1],
                    expected[i]
                );
            }
            let s = unsafe {
                std::str::from_utf8_unchecked(std::slice::from_ref(c))
            };
            assert!(matcher.accept_string(s, false));
        }

        matcher.fill_next_token_bitmask(&mut tensor, 0, false);
        let rejected_token_ids =
            get_masked_tokens_from_bitmask(&token_bitmask, tokenizer_info.vocab_size());
        rejected_sizes.push(rejected_token_ids.len());
        if let Some(expected) = expected_rejected_sizes.as_ref() {
            assert_eq!(
                rejected_sizes[rejected_sizes.len() - 1],
                expected[expected.len() - 1]
            );
        }
    }
}

#[test]
#[serial]
fn test_token_operations() {
    let vocab = vec![
        "<s>",
        "</s>",
        "a",
        "abc",
        "b\"",
        "\"",
        ":\"",
        "{",
        "}",
        ", ",
        "6",
        ":",
        "\n",
        " ",
        "\"a\":true",
    ];
    let input_splitted =
        vec!["{", "\"", "abc", "b\"", ":", "6", ", ", " ", "\"a\":true", "}"];
    let input_ids: Vec<i32> = input_splitted
        .iter()
        .map(|t| vocab.iter().position(|v| v == t).unwrap() as i32)
        .collect();

    let json_grammar = Grammar::builtin_json_grammar();
    let tokenizer_info =
        TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();
    let mut matcher =
        matcher_from_grammar_with_tokenizer(&json_grammar, &tokenizer_info);

    let expected: Vec<Vec<&str>> = vec![
        vec!["{"],
        vec!["\"", "}", "\n", " ", "\"a\":true"],
        vec![
            "<s>", "a", "abc", "b\"", "\"", ":\"", "{", "}", ", ", "6", ":",
            " ",
        ],
        vec![
            "<s>", "a", "abc", "b\"", "\"", ":\"", "{", "}", ", ", "6", ":",
            " ",
        ],
        vec![":", "\n", " ", ":\""],
        vec!["\"", "{", "6", "\n", " "],
        vec!["}", ", ", "6", "\n", " "],
        vec![" ", "\n", "\"", "\"a\":true"],
        vec![" ", "\n", "\"", "\"a\":true"],
        vec!["}", ", ", "\n", " "],
        vec!["</s>"],
    ];

    let mut result: Vec<Vec<String>> = Vec::new();

    for &id in &input_ids {
        let bitmask = get_next_token_bitmask_helper(&mut matcher, vocab.len());
        let accepted_indices =
            get_accepted_tokens_helper(&bitmask, vocab.len());
        let accepted: Vec<String> =
            accepted_indices.iter().map(|&i| vocab[i].to_string()).collect();
        result.push(accepted.clone());
        assert!(
            accepted.contains(&vocab[id as usize].to_string()),
            "Token {} should be accepted",
            vocab[id as usize]
        );
        assert!(matcher.accept_token(id));
    }

    let bitmask = get_next_token_bitmask_helper(&mut matcher, vocab.len());
    let accepted_indices = get_accepted_tokens_helper(&bitmask, vocab.len());
    let accepted: Vec<String> =
        accepted_indices.iter().map(|&i| vocab[i].to_string()).collect();
    result.push(accepted);

    for (i, (res, exp)) in result.iter().zip(expected.iter()).enumerate() {
        let mut res_sorted = res.clone();
        let mut exp_sorted: Vec<String> =
            exp.iter().map(|s| s.to_string()).collect();
        res_sorted.sort();
        exp_sorted.sort();
        assert_eq!(res_sorted, exp_sorted, "Mismatch at step {}", i);
    }
}

#[test]
#[serial]
fn test_rollback() {
    let vocab = vec![
        "<s>",
        "</s>",
        "a",
        "abc",
        "b\"",
        "\"",
        ":\"",
        "{",
        "}",
        ", ",
        "6",
        ":",
        "\n",
        " ",
        "\"a\":true",
    ];
    let input_splitted =
        vec!["{", "\"", "abc", "b\"", ":", "6", ", ", " ", "\"a\":true", "}"];
    let input_ids: Vec<i32> = input_splitted
        .iter()
        .map(|t| vocab.iter().position(|v| v == t).unwrap() as i32)
        .collect();

    let json_grammar = Grammar::builtin_json_grammar();
    let tokenizer_info =
        TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();
    let mut matcher = matcher_from_grammar_with_tokenizer_and_rollback(
        &json_grammar,
        &tokenizer_info,
        5,
    );

    assert_eq!(matcher.max_rollback_tokens(), -1);

    let input_ids_splitted: Vec<(i32, i32)> =
        input_ids.chunks(2).map(|chunk| (chunk[0], chunk[1])).collect();

    for (i_1, i_2) in input_ids_splitted {
        let bitmask1_orig =
            get_next_token_bitmask_helper(&mut matcher, vocab.len());
        assert!(matcher.accept_token(i_1));
        let bitmask2_orig =
            get_next_token_bitmask_helper(&mut matcher, vocab.len());
        assert!(matcher.accept_token(i_2));

        matcher.rollback(2);

        let bitmask1_after =
            get_next_token_bitmask_helper(&mut matcher, vocab.len());
        assert_eq!(bitmask1_orig, bitmask1_after);
        assert!(matcher.accept_token(i_1));

        let bitmask2_after =
            get_next_token_bitmask_helper(&mut matcher, vocab.len());
        assert_eq!(bitmask2_orig, bitmask2_after);
        assert!(matcher.accept_token(i_2));
    }
}

#[test]
#[serial]
fn test_graceful_rollback_failure() {
    let vocab = vec![
        "<s>",
        "</s>",
        "a",
        "abc",
        "b\"",
        "\"",
        ":\"",
        "{",
        "}",
        ", ",
        "6",
        "6:",
        ":",
        "\n",
        " ",
        "\"a\":true",
    ];
    let input_splitted = ["{", "\"", "abc", "\"", ":"];
    let input_ids: Vec<i32> = input_splitted
        .iter()
        .map(|t| vocab.iter().position(|v| v == t).unwrap() as i32)
        .collect();

    let json_grammar = Grammar::builtin_json_grammar();
    let tokenizer_info =
        TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();
    let mut matcher = matcher_from_grammar_with_tokenizer_and_rollback(
        &json_grammar,
        &tokenizer_info,
        5,
    );

    for &i in &input_ids {
        assert!(matcher.accept_token(i));
    }

    let token_6_colon = vocab.iter().position(|v| v == &"6:").unwrap() as i32;
    assert!(!matcher.accept_token(token_6_colon));

    // The matching should have accepted char '6' but failed to accept char ':'
    // A graceful revert should then occur, where char '6' is rolled back and
    // the state of the matcher is the same as before the failed call to accept_token

    let continuation = vec!["\"", "abc", "\"", " ", "}"];
    for token_str in continuation {
        let token_id =
            vocab.iter().position(|v| v == &token_str).unwrap() as i32;
        assert!(matcher.accept_token(token_id));
    }
}

#[test]
#[serial]
fn test_reset() {
    let vocab = vec![
        "<s>",
        "</s>",
        "a",
        "abc",
        "b\"",
        "\"",
        ":\"",
        "{",
        "}",
        ", ",
        "6",
        ":",
        "\n",
        " ",
        "\"a\":true",
    ];
    let input_splitted =
        vec!["{", "\"", "abc", "b\"", ":", "6", ", ", " ", "\"a\":true", "}"];
    let input_ids: Vec<i32> = input_splitted
        .iter()
        .map(|t| vocab.iter().position(|v| v == t).unwrap() as i32)
        .collect();

    let json_grammar = Grammar::builtin_json_grammar();
    let tokenizer_info =
        TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();
    let mut matcher =
        matcher_from_grammar_with_tokenizer(&json_grammar, &tokenizer_info);

    let mut orig_result = Vec::new();

    for &i in &input_ids {
        let bitmask = get_next_token_bitmask_helper(&mut matcher, vocab.len());
        orig_result.push(bitmask);
        assert!(matcher.accept_token(i));
    }

    matcher.reset();

    let mut result_after_reset = Vec::new();

    for &i in &input_ids {
        let bitmask = get_next_token_bitmask_helper(&mut matcher, vocab.len());
        result_after_reset.push(bitmask);
        assert!(matcher.accept_token(i));
    }

    for (l, r) in orig_result.iter().zip(result_after_reset.iter()) {
        assert_eq!(l, r);
    }
}

#[test]
#[serial]
fn test_termination() {
    let vocab = vec![
        "<s>", "</s>", "a", "abc", "b\"", "\"", ":\"", "{", " }", ", ", "6",
        ":", "\n", " ", "\"a\"", ":true",
    ];
    let input_splitted = vec![
        "{", "\"", "abc", "b\"", ":", "6", ", ", " ", "\"a\"", ":true", " }",
        "</s>",
    ];
    let input_ids: Vec<i32> = input_splitted
        .iter()
        .map(|t| vocab.iter().position(|v| v == t).unwrap() as i32)
        .collect();

    let json_grammar = Grammar::builtin_json_grammar();
    let tokenizer_info =
        TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();
    let mut matcher = matcher_from_grammar_with_tokenizer_and_rollback(
        &json_grammar,
        &tokenizer_info,
        5,
    );

    for (idx, &i) in input_ids.iter().enumerate() {
        let _ = get_next_token_bitmask_helper(&mut matcher, vocab.len());
        assert!(
            matcher.accept_token(i),
            "Failed to accept token {} at index {}: '{}'",
            i,
            idx,
            vocab[i as usize]
        );
    }

    assert!(matcher.is_terminated());

    assert!(!matcher.accept_token(0));

    matcher.rollback(2);

    assert!(!matcher.is_terminated());
    assert!(matcher.accept_token(input_ids[input_ids.len() - 2]));
}

#[test]
#[serial]
fn test_get_jump_forward_string() {
    let ebnf = r#"root ::= "abb" | "abbd" | other_rule
other_rule ::= "a" sub_rule "b"
sub_rule ::= "b"
"#;
    let grammar = Grammar::from_ebnf(ebnf, "root").unwrap();
    let tokenizer_info =
        TokenizerInfo::new::<&str>(&[], VocabType::RAW, &None, false).unwrap();
    let mut matcher = matcher_from_grammar_with_tokenizer(&grammar, &tokenizer_info);
    assert!(matcher.accept_string("a", false));
    assert_eq!(matcher.find_jump_forward_string(), "bb");
}

#[test]
#[serial]
fn test_vocab_size() {
    let vocab = vec![
        "<s>",
        "</s>",
        "a",
        "abc",
        "b\"",
        "\"",
        ":\"",
        "{",
        "}",
        ", ",
        "6",
        ":",
        "\n",
        " ",
        "\"a\":true",
    ];
    let json_grammar = Grammar::builtin_json_grammar();
    let tokenizer_info = TokenizerInfo::new_with_vocab_size(
        &vocab,
        VocabType::RAW,
        Some(64),
        &None,
        false,
    ).unwrap();
    let mut matcher =
        matcher_from_grammar_with_tokenizer(&json_grammar, &tokenizer_info);

    let bitmask = get_next_token_bitmask_helper(&mut matcher, 64);

    // Count rejected tokens
    let mut rejected_count = 0;
    for i in 0..64 {
        if !is_token_accepted_helper(i, &bitmask) {
            rejected_count += 1;
        }
    }

    // Only token 7 ("{") should be accepted at the start
    assert_eq!(rejected_count, 63);
    assert!(is_token_accepted_helper(7, &bitmask));
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_override_stop_tokens() {
    let cases: &[(&str, &[i32])] = &[
        ("meta-llama/Llama-2-7b-chat-hf", &[2]),
        ("meta-llama/Meta-Llama-3-8B-Instruct", &[128001, 128009]),
        ("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", &[100001]),
    ];

    for (model_id, override_stop_tokens) in cases {
        let path = test_utils::download_tokenizer_json(model_id)
            .expect("download tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");

        let tokenizer_info_with_override = TokenizerInfo::from_huggingface(
            &tokenizer,
            None,
            Some(*override_stop_tokens),
        )
        .unwrap();
        assert_eq!(
            &*tokenizer_info_with_override.stop_token_ids(),
            *override_stop_tokens,
            "tokenizer_info stop_token_ids mismatch for {}",
            model_id
        );

        let grammar = Grammar::builtin_json_grammar();
        let mut compiler = GrammarCompiler::new(&tokenizer_info_with_override, 1, false, -1).unwrap();
        let compiled = compiler.compile_grammar(&grammar).unwrap();
        let matcher_with_override = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
        assert_eq!(
            &*matcher_with_override.stop_token_ids(),
            *override_stop_tokens,
            "matcher stop_token_ids mismatch for {}",
            model_id
        );

        let tokenizer_info_without_override = TokenizerInfo::from_huggingface(&tokenizer, None, None).unwrap();
        let mut compiler_no_override = GrammarCompiler::new(&tokenizer_info_without_override, 1, false, -1).unwrap();
        let compiled_no_override = compiler_no_override.compile_grammar(&grammar).unwrap();

        let matcher_with_override_at_creation = GrammarMatcher::new(
            &compiled_no_override,
            Some(*override_stop_tokens),
            true,
            -1,
        )
        .unwrap();
        assert_eq!(
            &*matcher_with_override_at_creation.stop_token_ids(),
            *override_stop_tokens,
            "matcher override at creation mismatch for {}",
            model_id
        );
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_fill_next_token_bitmask_errors() {
    let tokenizer_info =
        make_hf_tokenizer_info("meta-llama/Meta-Llama-3-8B-Instruct");
    let json_grammar = Grammar::builtin_json_grammar();
    let mut matcher =
        matcher_from_grammar_with_tokenizer(&json_grammar, &tokenizer_info);

    let mut bitmask_data =
        allocate_token_bitmask(1, tokenizer_info.vocab_size());
    let (mut tensor_correct, _shape, _strides) = create_bitmask_dltensor(
        &mut bitmask_data,
        1,
        tokenizer_info.vocab_size(),
    );
    matcher.fill_next_token_bitmask(&mut tensor_correct, 0, false);
}


#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_batch_accept_string() {
    let cases = vec![
        (
            vec![r#"root ::= "a""#, r#"root ::= [0-9]+"#, r#"root ::= "ab""#],
            vec!["a", "123", "ab"],
            vec![true, true, true],
        ),
        (
            vec![r#"root ::= "a""#, r#"root ::= [0-9]+"#, r#"root ::= "ab""#],
            vec!["b", "123a", "d"],
            vec![false, false, false],
        ),
        (
            vec![r#"root ::= "a""#, r#"root ::= [0-9]+"#, r#"root ::= "ab""#],
            vec!["a", "123a", "ab"],
            vec![true, false, true],
        ),
        (vec![r#"root ::= "a""#], vec!["a"], vec![true]),
        (vec![r#"root ::= "a""#], vec!["b"], vec![false]),
        (
            vec![r#"root ::= "你好""#, r#"root ::= "こんにちは""#, r#"root ::= "안녕하세요""#],
            vec!["你好", "こんにちは", "안녕하세요"],
            vec![true, true, true],
        ),
    ];

    for (grammars, inputs, expecteds) in cases {
        let grammar_objs: Vec<Grammar> = grammars
            .iter()
            .map(|g| Grammar::from_ebnf(g, "root").unwrap())
            .collect();
        let matchers: Vec<GrammarMatcher> = grammar_objs
            .iter()
            .map(matcher_from_grammar)
            .collect();

        let results = BatchGrammarMatcher::batch_accept_string(&matchers, &inputs, false);
        assert_eq!(&*results, expecteds.as_slice());
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_batch_accept_token() {
    let cases = vec![
        (
            vec![r#"root ::= "a""#, r#"root ::= [0-9]+"#, r#"root ::= "ab""#],
            vec![2, 5, 2],
            vec![true, true, true],
        ),
        (
            vec![r#"root ::= "a""#, r#"root ::= [0-9]+"#, r#"root ::= "ab""#],
            vec![3, 2, 4],
            vec![false, false, false],
        ),
        (
            vec![r#"root ::= "a""#, r#"root ::= [0-9]+"#, r#"root ::= "ab""#],
            vec![2, 8, 9],
            vec![true, false, true],
        ),
        (vec![r#"root ::= "a""#], vec![2], vec![true]),
        (vec![r#"root ::= "a""#], vec![3], vec![false]),
    ];

    let vocab: Vec<&str> = vec![
        "<s>", "</s>", "a", "b", "c", "1", "2", "3", "123a", "ab",
    ];
    let tokenizer_info =
        TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();

    for (grammars, inputs, expecteds) in cases {
        let grammar_objs: Vec<Grammar> = grammars
            .iter()
            .map(|g| Grammar::from_ebnf(g, "root").unwrap())
            .collect();

        let matchers: Vec<GrammarMatcher> = grammar_objs
            .iter()
            .map(|g| matcher_from_grammar_with_tokenizer(g, &tokenizer_info))
            .collect();

        let results = BatchGrammarMatcher::batch_accept_token(&matchers, &inputs, false);
        assert_eq!(&*results, expecteds.as_slice());
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_batch_fill_next_token_bitmask() {
    let grammars = [
        r#"root ::= "a""#,
        r#"root ::= [0-9]+"#,
        r#"root ::= "ab""#,
        r#"root ::= [a-z0-9]+"#,
    ];
    let vocab: Vec<&str> = vec!["ab", "</s>", "a", "b", "c", "1", "2", "3", "123a"];
    let tokenizer_info =
        TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();

    let grammar_objs: Vec<Grammar> = grammars
        .iter()
        .map(|g| Grammar::from_ebnf(g, "root").unwrap())
        .collect();
    let matchers: Vec<GrammarMatcher> = grammar_objs
        .iter()
        .map(|g| matcher_from_grammar_with_tokenizer(g, &tokenizer_info))
        .collect();

    let batch_size = matchers.len();
    let mut token_bitmask =
        allocate_token_bitmask(batch_size, tokenizer_info.vocab_size());
    let (mut tensor, _shape, _strides) = create_bitmask_dltensor(
        &mut token_bitmask,
        batch_size,
        tokenizer_info.vocab_size(),
    );

    let input_str = ["a", "1", "a", "123a"];
    let expected_accepted_tokens = vec![
        vec![vec![2], vec![5, 6, 7], vec![0, 2], vec![0, 2, 3, 4, 5, 6, 7, 8]],
        vec![
            vec![1],
            vec![1, 5, 6, 7],
            vec![3],
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
        ],
    ];

    let mut batch_grammar_matcher = BatchGrammarMatcher::new(2).unwrap();
    batch_grammar_matcher.batch_fill_next_token_bitmask(&matchers, &mut tensor, None, false);

    for i in 0..batch_size {
        let rejected_token_ids =
            get_masked_tokens_from_bitmask(
                &token_bitmask[(i * token_bitmask.len() / batch_size)
                    ..((i + 1) * token_bitmask.len() / batch_size)],
                tokenizer_info.vocab_size(),
            );
        let mut accepted: Vec<usize> = (0..vocab.len())
            .filter(|id| !rejected_token_ids.contains(id))
            .collect();
        accepted.sort();
        assert_eq!(accepted, expected_accepted_tokens[0][i]);
    }

    assert_eq!(
        &*BatchGrammarMatcher::batch_accept_string(&matchers, &input_str, false),
        &[true, true, true, true]
    );

    batch_grammar_matcher.batch_fill_next_token_bitmask(&matchers, &mut tensor, None, false);

    for i in 0..batch_size {
        let rejected_token_ids =
            get_masked_tokens_from_bitmask(
                &token_bitmask[(i * token_bitmask.len() / batch_size)
                    ..((i + 1) * token_bitmask.len() / batch_size)],
                tokenizer_info.vocab_size(),
            );
        let mut accepted: Vec<usize> = (0..vocab.len())
            .filter(|id| !rejected_token_ids.contains(id))
            .collect();
        accepted.sort();
        assert_eq!(accepted, expected_accepted_tokens[1][i]);
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_batch_fill_next_token_bitmask_pressure() {
    let tokenizer_info = make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let input_str = r#"{"id": 1,"name": "Example"}"#;

    let grammar = Grammar::builtin_json_grammar();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();

    let mut matchers: Vec<GrammarMatcher> = Vec::new();
    for i in 0..=input_str.len() {
        let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
        let substr = &input_str[..i];
        matcher.accept_string(substr, false);
        matchers.push(matcher);
    }

    let batch_size = matchers.len();
    let vocab_size = compiled.tokenizer_info().vocab_size();
    let mut bitmask_data = allocate_token_bitmask(batch_size, vocab_size);
    let (mut tensor, _shape, _strides) =
        create_bitmask_dltensor(&mut bitmask_data, batch_size, vocab_size);

    let rejected_token_size = vec![
        31989, 31912, 270, 270, 270, 31973, 31846, 31846, 31948, 31915, 270, 270, 270, 270,
        270, 31973, 31846, 31846, 263, 263, 263, 263, 263, 263, 263, 263, 31974, 31999,
    ];

    let mut batch_matcher = BatchGrammarMatcher::new(2).unwrap();
    batch_matcher.batch_fill_next_token_bitmask(&matchers, &mut tensor, None, false);

    for i in 0..matchers.len() {
        let slice_len = bitmask_data.len() / batch_size;
        let rejected_token_ids = get_masked_tokens_from_bitmask(
            &bitmask_data[(i * slice_len)..((i + 1) * slice_len)],
            vocab_size,
        );
        assert_eq!(
            rejected_token_ids.len(),
            rejected_token_size[i],
            "index {}, rejected size {} != expected {}",
            i,
            rejected_token_ids.len(),
            rejected_token_size[i]
        );
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_batch_fill_next_token_bitmask_pressure_single_thread() {
    let tokenizer_info = make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let input_str = r#"{"id": 1,"name": "Example"}"#;

    let grammar = Grammar::builtin_json_grammar();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();

    let mut matchers: Vec<GrammarMatcher> = Vec::new();
    for i in 0..=input_str.len() {
        let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
        let substr = &input_str[..i];
        matcher.accept_string(substr, false);
        matchers.push(matcher);
    }

    let batch_size = matchers.len();
    let vocab_size = compiled.tokenizer_info().vocab_size();
    let mut bitmask_data = allocate_token_bitmask(batch_size, vocab_size);
    let (mut tensor, _shape, _strides) =
        create_bitmask_dltensor(&mut bitmask_data, batch_size, vocab_size);

    let rejected_token_size = vec![
        31989, 31912, 270, 270, 270, 31973, 31846, 31846, 31948, 31915, 270, 270, 270, 270,
        270, 31973, 31846, 31846, 263, 263, 263, 263, 263, 263, 263, 263, 31974, 31999,
    ];

    let mut batch_matcher = BatchGrammarMatcher::new(1).unwrap();
    batch_matcher.batch_fill_next_token_bitmask(&matchers, &mut tensor, None, false);

    for i in 0..matchers.len() {
        let slice_len = bitmask_data.len() / batch_size;
        let rejected_token_ids = get_masked_tokens_from_bitmask(
            &bitmask_data[(i * slice_len)..((i + 1) * slice_len)],
            vocab_size,
        );
        assert_eq!(
            rejected_token_ids.len(),
            rejected_token_size[i],
            "index {}, rejected size {} != expected {}",
            i,
            rejected_token_ids.len(),
            rejected_token_size[i]
        );
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_batch_fill_next_token_bitmask_pressure_shuffled() {
    let tokenizer_info = make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let input_str = r#"{"id": 1,"name": "Example"}"#;

    let grammar = Grammar::builtin_json_grammar();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();

    let mut matchers: Vec<GrammarMatcher> = Vec::new();
    let indices: Vec<usize> = (0..=input_str.len()).collect();
    for i in &indices {
        let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
        let substr = &input_str[..*i];
        matcher.accept_string(substr, false);
        matchers.push(matcher);
    }

    let batch_size = matchers.len();
    let vocab_size = compiled.tokenizer_info().vocab_size();
    let mut bitmask_data = allocate_token_bitmask(batch_size, vocab_size);
    let (mut tensor, _shape, _strides) =
        create_bitmask_dltensor(&mut bitmask_data, batch_size, vocab_size);

    let rejected_token_size = vec![
        31989, 31912, 270, 270, 270, 31973, 31846, 31846, 31948, 31915, 270, 270, 270, 270,
        270, 31973, 31846, 31846, 263, 263, 263, 263, 263, 263, 263, 263, 31974, 31999,
    ];

    let mut shuffled_indices: Vec<i32> = (0..batch_size).map(|i| i as i32).collect();
    shuffled_indices.reverse(); // Deterministic shuffle

    let mut batch_matcher = BatchGrammarMatcher::new(2).unwrap();
    batch_matcher.batch_fill_next_token_bitmask(&matchers, &mut tensor, Some(&shuffled_indices), false);

    for i in 0..matchers.len() {
        let output_idx = shuffled_indices[i] as usize;
        let slice_len = bitmask_data.len() / batch_size;
        let rejected_token_ids = get_masked_tokens_from_bitmask(
            &bitmask_data[(output_idx * slice_len)..((output_idx + 1) * slice_len)],
            vocab_size,
        );
        assert_eq!(
            rejected_token_ids.len(),
            rejected_token_size[i],
            "index {}, rejected size {} != expected {}",
            i,
            rejected_token_ids.len(),
            rejected_token_size[i]
        );
    }
}
