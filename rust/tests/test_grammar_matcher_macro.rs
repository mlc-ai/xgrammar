mod test_utils;

use serial_test::serial;
use test_utils::*;
use xgrammar::Grammar;

#[test]
#[serial]
fn test_simple() {
    let grammar_str = r#"root ::= TagDispatch(("tag1", rule1), ("tag2", rule2))
rule1 ::= "abcd"
rule2 ::= "efg"
"#;

    let grammar = Grammar::from_ebnf(grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "tag1abcd"));
    assert!(is_grammar_accept_string(&grammar, "tag1abcdtag2efg"));
    assert!(is_grammar_accept_string(&grammar, "tag1abcdqqqqtag2efg"));
    assert!(!is_grammar_accept_string(&grammar, "tag1abc"));
    assert!(!is_grammar_accept_string(&grammar, "tag1abce"));
    assert!(!is_grammar_accept_string(&grammar, "ttag1abd"));
}

#[test]
#[serial]
fn test_complex_rule() {
    let grammar_str = r#"root ::= TagDispatch(("tag1", rule1), ("tag2", rule2))
rule1 ::= "abcd" [p]*
rule2 ::= "efg" [t]*
"#;

    let grammar = Grammar::from_ebnf(grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "tag1abcd"));
    assert!(is_grammar_accept_string(&grammar, "tag1abcdppppptag2efg"));
    assert!(is_grammar_accept_string(&grammar, "tag2efgtttttag1abc"));
    assert!(!is_grammar_accept_string(&grammar, "tag1efg"));
}

#[test]
#[serial]
fn test_no_loop_after_dispatch() {
    let grammar_str = r#"root ::= TagDispatch(("tag1", rule1), ("tag2", rule2), loop_after_dispatch=false)
rule1 ::= "abcd" [p]*
rule2 ::= "efg" [t]*
"#;

    let grammar = Grammar::from_ebnf(grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "tag1abcd"));
    assert!(is_grammar_accept_string(&grammar, "tag2efgttt"));
    assert!(!is_grammar_accept_string(&grammar, "tag1abcdppppptag2"));
    assert!(!is_grammar_accept_string(&grammar, "tag2efgtag1"));
}

#[test]
#[serial]
fn test_stop_str() {
    let grammar_str = r#"root ::= root1 "w"
root1 ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  stop_eos=false,
  stop_str=("tag3", "ll")
)
rule1 ::= "abcd" [p]*
rule2 ::= "efg" [t]*
"#;

    let grammar = Grammar::from_ebnf(grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "tag1abcdllw"));
    assert!(is_grammar_accept_string(&grammar, "tag1abcdtag3w"));
    assert!(is_grammar_accept_string(&grammar, "tag1abcdqqqtag2efgtag3w"));
    // Non-terminated allowance
    assert!(matcher_from_grammar(&grammar).accept_string("tag1abcd", false));
    assert!(matcher_from_grammar(&grammar).accept_string("tag2efgttt", false));
    // But requiring termination should fail
    assert!(!is_grammar_accept_string(&grammar, "tag1abcd"));
    assert!(!is_grammar_accept_string(&grammar, "tag2efgttt"));
    assert!(!is_grammar_accept_string(&grammar, "tag1abce"));
    // This should not be accepted even without termination requirement
    assert!(
        !matcher_from_grammar(&grammar).accept_string("tag1abcdlltag3w", false)
    );
}

#[test]
#[serial]
fn test_stop_str_no_loop() {
    let grammar_str = r#"root ::= root1 "w"
root1 ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  stop_eos=false,
  stop_str=("tag3", "ll"),
  loop_after_dispatch=false
)
rule1 ::= "abcd" [p]*
rule2 ::= "efg" [t]*
"#;

    let grammar = Grammar::from_ebnf(grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "tag1abcdllw"));
    assert!(is_grammar_accept_string(&grammar, "tag1abcdtag3w"));
    assert!(matcher_from_grammar(&grammar).accept_string("tag1abcd", false));
    assert!(matcher_from_grammar(&grammar).accept_string("tag2efgttt", false));
    assert!(!is_grammar_accept_string(&grammar, "tag1abcdqqqtag2efgtag3w"));
    assert!(!is_grammar_accept_string(&grammar, "tag1abcd"));
    assert!(!is_grammar_accept_string(&grammar, "tag2efgttt"));
    assert!(!is_grammar_accept_string(&grammar, "tag1abce"));
    assert!(
        !matcher_from_grammar(&grammar).accept_string("tag1abcdlltag3w", false)
    );
}

#[test]
#[serial]
fn test_tag_dispatch_mask_generation_correctness() {
    let grammar_str = r#"root ::= TagDispatch(("tag1", rule1), ("tag2", rule2))
rule1 ::= "abc"
rule2 ::= "dg"
"#;
    let tokens = vec![
        "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2a", "2dgt",
        "2dgtag1a", "2dgtag1b", "tag1a", "tag1b", "c哈哈t", "q", "abcdef",
    ];
    let input_str = "tag1abcqqtag2dgq";
    let expected_accepted_tokens = vec![
        vec![
            "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2a", "2dgt",
            "2dgtag1a", "tag1a", "c哈哈t", "q", "abcdef",
        ],
        vec![
            "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2a", "2dgt",
            "2dgtag1a", "tag1a", "c哈哈t", "q", "abcdef",
        ],
        vec![
            "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2a", "2dgt",
            "2dgtag1a", "tag1a", "c哈哈t", "q", "abcdef",
        ],
        vec![
            "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2dgt",
            "2dgtag1a", "tag1a", "c哈哈t", "q", "abcdef",
        ],
        vec!["a", "abcdef"],
        vec!["b"],
        vec!["c哈哈t", "c"],
        vec![
            "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2a", "2dgt",
            "2dgtag1a", "tag1a", "c哈哈t", "q", "abcdef",
        ],
        vec![
            "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2a", "2dgt",
            "2dgtag1a", "tag1a", "c哈哈t", "q", "abcdef",
        ],
        vec![
            "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2a", "2dgt",
            "2dgtag1a", "tag1a", "c哈哈t", "q", "abcdef",
        ],
        vec![
            "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2a", "2dgt",
            "2dgtag1a", "tag1a", "c哈哈t", "q", "abcdef",
        ],
        vec![
            "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2a", "2dgt",
            "2dgtag1a", "tag1a", "c哈哈t", "q", "abcdef",
        ],
        vec![
            "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2dgt",
            "2dgtag1a", "tag1a", "c哈哈t", "q", "abcdef",
        ],
        vec!["d"],
        vec!["g"],
        vec![
            "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2a", "2dgt",
            "2dgtag1a", "tag1a", "c哈哈t", "q", "abcdef",
        ],
        vec![
            "a", "b", "c", "d", "g", "t", "1", "2", "1a", "2d", "2a", "2dgt",
            "2dgtag1a", "tag1a", "c哈哈t", "q", "abcdef",
        ],
    ];

    use xgrammar::{
        GrammarCompiler, GrammarMatcher, TokenizerInfo, VocabType,
        allocate_token_bitmask, testing,
    };

    let grammar = Grammar::from_ebnf(grammar_str, "root").unwrap();
    let tokenizer_info =
        TokenizerInfo::new(&tokens, VocabType::RAW, &None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let compiled_grammar = compiler.compile_grammar(&grammar).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();

    let vocab_size = tokenizer_info.vocab_size();
    let mut bitmask_data = allocate_token_bitmask(1, vocab_size);

    // pad a dummy char to check the final bitmask after accepting the input string
    let input_with_padding = format!("{}0", input_str);
    for (i, c) in input_with_padding.chars().enumerate() {
        let (mut tensor, _shape, _strides) =
            create_bitmask_dltensor(&mut bitmask_data, 1, vocab_size);

        matcher.fill_next_token_bitmask(&mut tensor, 0, false);

        let rejected_indices = testing::get_masked_tokens_from_bitmask(
            &tensor,
            vocab_size as i32,
            0,
        );
        let all_indices: std::collections::HashSet<usize> =
            (0..vocab_size).collect();
        let rejected_set: std::collections::HashSet<usize> =
            rejected_indices.iter().map(|&x| x as usize).collect();
        let accepted_indices: Vec<usize> =
            all_indices.difference(&rejected_set).copied().collect();
        let mut accepted_tokens: Vec<String> =
            accepted_indices.iter().map(|&id| tokens[id].to_string()).collect();
        accepted_tokens.sort();

        if i < input_str.len() {
            let char_str = c.to_string();
            assert!(matcher.accept_string(&char_str, false));
        }

        let mut expected_sorted = expected_accepted_tokens[i].clone();
        expected_sorted.sort();
        let expected_sorted: Vec<String> =
            expected_sorted.iter().map(|s| s.to_string()).collect();

        assert_eq!(
            accepted_tokens, expected_sorted,
            "Mismatch at step {} (char: {})",
            i, c
        );

        // Reset bitmask for next iteration
        bitmask_data.fill(-1);
    }
}

#[test]
#[serial]
fn test_regression_multiple_tag_dispatch() {
    let grammar_str = r#"root ::= root1 "w"
root1 ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false
)
rule1 ::= TagDispatch(
  ("tag1", rule2),
  ("tag2", rule3),
  stop_eos=false,
  stop_str=("tag3", "ll"),
  loop_after_dispatch=true
)
rule2 ::= "efg" [t]*
rule3 ::= "abcd" [p]*
"#;
    assert!(is_grammar_accept_string(
        &Grammar::from_ebnf(grammar_str, "root").unwrap(),
        "tag1tag1efgllw"
    ));
    assert!(is_grammar_accept_string(
        &Grammar::from_ebnf(grammar_str, "root").unwrap(),
        "tag1tag2abcdtag3w"
    ));
    assert!(!is_grammar_accept_string(
        &Grammar::from_ebnf(grammar_str, "root").unwrap(),
        "tag1Ktag2abcdtag3tag1"
    ));
    assert!(is_grammar_accept_string(
        &Grammar::from_ebnf(grammar_str, "root").unwrap(),
        "tag1tag3w"
    ));
    assert!(!is_grammar_accept_string(
        &Grammar::from_ebnf(grammar_str, "root").unwrap(),
        "tag1tag3tag2abcdll"
    ));
}
