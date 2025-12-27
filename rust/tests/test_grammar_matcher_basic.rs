mod test_utils;

use serial_test::serial;
use test_utils::*;
use xgrammar::{Grammar, TokenizerInfo, VocabType};
#[cfg(feature = "hf")]
use xgrammar::{
    BatchGrammarMatcher, GrammarCompiler, GrammarMatcher, allocate_token_bitmask,
};

#[test]
#[serial]
fn test_accept_string() {
    let cases: &[(&str, &str, bool)] = &[
        ("root ::= [^a]+", "bbb", true),
        ("root ::= [^a]+", "bba", false),
        ("root ::= [^a]+", "©", true), // U+00A9
    ];

    for (ebnf, input, accepted) in cases {
        let g = Grammar::from_ebnf(ebnf, "root").unwrap();
        let mut m = matcher_from_grammar(&g);
        assert_eq!(m.accept_string(input, false), *accepted, "{}", input);
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
    let input_splitted = vec!["{", "\"", "abc", "\"", ":"];
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
    let g = Grammar::from_ebnf(ebnf, "root").unwrap();
    let tokenizer_info =
        TokenizerInfo::new::<&str>(&vec![], VocabType::RAW, &None, false).unwrap();
    let mut matcher = matcher_from_grammar_with_tokenizer(&g, &tokenizer_info);
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
    // Mirror Python: ensure stop token overrides work at both TokenizerInfo and Matcher levels
    let model_id = "meta-llama/Llama-2-7b-chat-hf";
    let override_stop_tokens: &[i32] = &[2];

    // Build tokenizers::Tokenizer and TokenizerInfo with override stops
    let path = test_utils::download_tokenizer_json(model_id)
        .expect("download tokenizer.json");
    let tk = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
    let tokenizer_info_1 = TokenizerInfo::from_huggingface(&tk, None, Some(override_stop_tokens)).unwrap();
    assert_eq!(&*tokenizer_info_1.stop_token_ids(), override_stop_tokens);

    // Compile a grammar with tokenizer_info_1 and verify matcher inherits stop ids
    let grammar = Grammar::builtin_json_grammar();
    let mut compiler = GrammarCompiler::new(&tokenizer_info_1, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();
    let matcher_1 = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
    assert_eq!(&*matcher_1.stop_token_ids(), override_stop_tokens);

    // Build TokenizerInfo without overrides
    let tokenizer_info_2 = TokenizerInfo::from_huggingface(&tk, None, None).unwrap();
    let mut compiler2 = GrammarCompiler::new(&tokenizer_info_2, 1, false, -1).unwrap();
    let compiled2 = compiler2.compile_grammar(&grammar).unwrap();

    // Override at matcher creation
    let matcher_2 = GrammarMatcher::new(&compiled2, Some(override_stop_tokens), true, -1).unwrap();
    assert_eq!(&*matcher_2.stop_token_ids(), override_stop_tokens);
}


#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_batch_accept_string_basic() {
    let grammars = vec![r#"root ::= "a""#, r#"root ::= [0-9]+"#, r#"root ::= "ab""#];
    let inputs = vec!["a", "123", "ab"];

    let grammar_objs: Vec<Grammar> = grammars
        .iter()
        .map(|g| Grammar::from_ebnf(g, "root").unwrap())
        .collect();

    let vocab: Vec<&str> = vec![
        "<s>", "</s>", "a", "b", "c", "1", "2", "3", "123a", "ab",
    ];
    let tokenizer_info = TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();

    let matchers: Vec<GrammarMatcher> = grammar_objs
        .iter()
        .map(|g| {
            let compiled = compiler.compile_grammar(g).unwrap();
            GrammarMatcher::new(&compiled, None, true, -1).unwrap()
        })
        .collect();

    let results = BatchGrammarMatcher::batch_accept_string(&matchers, &inputs, false);
    assert_eq!(&*results, &[true, true, true]);
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_batch_accept_token_basic() {
    let grammars = vec![r#"root ::= "a""#, r#"root ::= [0-9]+"#, r#"root ::= "ab""#];
    let input_ids = vec![2, 5, 9];

    let vocab: Vec<&str> = vec![
        "<s>", "</s>", "a", "b", "c", "1", "2", "3", "123a", "ab",
    ];
    let tokenizer_info = TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();

    let grammar_objs: Vec<Grammar> = grammars
        .iter()
        .map(|g| Grammar::from_ebnf(g, "root").unwrap())
        .collect();

    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();

    let matchers: Vec<GrammarMatcher> = grammar_objs
        .iter()
        .map(|g| {
            let compiled = compiler.compile_grammar(g).unwrap();
            GrammarMatcher::new(&compiled, None, true, -1).unwrap()
        })
        .collect();

    let results = BatchGrammarMatcher::batch_accept_token(&matchers, &input_ids, false);
    assert_eq!(&*results, &[true, true, true]);
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_batch_fill_next_token_bitmask_basic() {
    let grammars = vec![r#"root ::= "a""#, r#"root ::= [0-9]+"#, r#"root ::= "ab""#, r#"root ::= [a-z0-9]+"#];
    let vocab: Vec<&str> = vec!["ab", "</s>", "a", "b", "c", "1", "2", "3", "123a"];
    let tokenizer_info = TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();

    let grammar_objs: Vec<Grammar> = grammars
        .iter()
        .map(|g| Grammar::from_ebnf(g, "root").unwrap())
        .collect();

    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();

    let matchers: Vec<GrammarMatcher> = grammar_objs
        .iter()
        .map(|g| {
            let compiled = compiler.compile_grammar(g).unwrap();
            GrammarMatcher::new(&compiled, None, true, -1).unwrap()
        })
        .collect();

    let batch_size = matchers.len();
    let vocab_size = tokenizer_info.vocab_size();
    let mut bitmask_data = allocate_token_bitmask(batch_size, vocab_size);
    let (mut tensor, _shape, _strides) =
        create_bitmask_dltensor(&mut bitmask_data, batch_size, vocab_size);

    let mut batch_matcher = BatchGrammarMatcher::new(2).unwrap();
    batch_matcher.batch_fill_next_token_bitmask(&matchers, &mut tensor, None, false);
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_batch_fill_next_token_bitmask_pressure() {
    let tk = make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let input_str = r#"{"id": 1,"name": "Example"}"#;

    let grammar = Grammar::builtin_json_grammar();
    let mut compiler = GrammarCompiler::new(&tk, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();

    let mut matchers: Vec<GrammarMatcher> = Vec::new();
    for i in 0..=input_str.len() {
        let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
        let substr = &input_str[..i];
        matcher.accept_string(substr, false);
        matchers.push(matcher);
    }

    let batch_size = matchers.len();
    let vocab_size = tk.vocab_size();
    let mut bitmask_data = allocate_token_bitmask(batch_size, vocab_size);
    let (mut tensor, _shape, _strides) =
        create_bitmask_dltensor(&mut bitmask_data, batch_size, vocab_size);

    let mut batch_matcher = BatchGrammarMatcher::new(2).unwrap();
    batch_matcher.batch_fill_next_token_bitmask(&matchers, &mut tensor, None, false);
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_batch_fill_next_token_bitmask_pressure_single_thread() {
    let tk = make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let input_str = r#"{"id": 1,"name": "Example"}"#;

    let grammar = Grammar::builtin_json_grammar();
    let mut compiler = GrammarCompiler::new(&tk, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();

    let mut matchers: Vec<GrammarMatcher> = Vec::new();
    for i in 0..=input_str.len() {
        let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
        let substr = &input_str[..i];
        matcher.accept_string(substr, false);
        matchers.push(matcher);
    }

    let batch_size = matchers.len();
    let vocab_size = tk.vocab_size();
    let mut bitmask_data = allocate_token_bitmask(batch_size, vocab_size);
    let (mut tensor, _shape, _strides) =
        create_bitmask_dltensor(&mut bitmask_data, batch_size, vocab_size);

    let mut batch_matcher = BatchGrammarMatcher::new(1).unwrap();
    batch_matcher.batch_fill_next_token_bitmask(&matchers, &mut tensor, None, false);
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_batch_fill_next_token_bitmask_pressure_shuffled() {
    let tk = make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let input_str = r#"{"id": 1,"name": "Example"}"#;

    let grammar = Grammar::builtin_json_grammar();
    let mut compiler = GrammarCompiler::new(&tk, 1, false, -1).unwrap();
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
    let vocab_size = tk.vocab_size();
    let mut bitmask_data = allocate_token_bitmask(batch_size, vocab_size);
    let (mut tensor, _shape, _strides) =
        create_bitmask_dltensor(&mut bitmask_data, batch_size, vocab_size);

    let mut batch_matcher = BatchGrammarMatcher::new(2).unwrap();
    batch_matcher.batch_fill_next_token_bitmask(&matchers, &mut tensor, None, false);
}
