mod common;

use common::byte_token;
use xgrammar::{
    Grammar, GrammarCompiler, GrammarMatcher, MatcherOptions, TokenBitmask, TokenizerInfo,
    TokenizerInfoOptions, VocabType,
};

fn compile(ebnf: &str) -> xgrammar::CompiledGrammar {
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    compiler.compile_ebnf(ebnf).unwrap()
}

#[test]
fn constrained_decoding_loop() {
    let compiled = compile("root ::= \"ab\" | \"cd\"");
    let ti = compiled.tokenizer_info().unwrap();
    let vocab_size = ti.vocab_size().unwrap();

    let mut matcher = GrammarMatcher::new(&compiled).unwrap();
    let mut bitmask = TokenBitmask::new(vocab_size);

    // Position 0: only 'a' or 'c' are allowed.
    assert!(matcher.fill_next_token_bitmask(&mut bitmask, 0).unwrap());
    assert!(bitmask.is_allowed(0, byte_token(b'a') as usize));
    assert!(bitmask.is_allowed(0, byte_token(b'c') as usize));
    assert!(!bitmask.is_allowed(0, byte_token(b'b') as usize));
    assert!(!bitmask.is_allowed(0, 0)); // stop token not yet allowed

    // Applying the mask sets disallowed logits to -inf and keeps the rest.
    let mut logits = vec![1.0f32; vocab_size];
    bitmask.apply_to_logits(&mut logits, 0);
    assert_eq!(logits[byte_token(b'a') as usize], 1.0);
    assert!(logits[byte_token(b'b') as usize].is_infinite());

    // Accept "a", then only "b" continues.
    assert!(matcher.accept_token(byte_token(b'a')).unwrap());
    bitmask.reset();
    assert!(matcher.fill_next_token_bitmask(&mut bitmask, 0).unwrap());
    assert!(bitmask.is_allowed(0, byte_token(b'b') as usize));
    assert!(!bitmask.is_allowed(0, byte_token(b'c') as usize));

    assert!(matcher.accept_token(byte_token(b'b')).unwrap());
    assert!(matcher.is_completed().unwrap());
    assert!(!matcher.is_terminated().unwrap());

    // Now the stop token terminates the matcher.
    bitmask.reset();
    matcher.fill_next_token_bitmask(&mut bitmask, 0).unwrap();
    assert!(bitmask.is_allowed(0, 0));
    assert!(matcher.accept_token(0).unwrap());
    assert!(matcher.is_terminated().unwrap());
}

#[test]
fn rejects_tokens_outside_grammar() {
    let compiled = compile("root ::= \"ab\"");
    let mut matcher = GrammarMatcher::new(&compiled).unwrap();
    assert!(!matcher.accept_token(byte_token(b'x')).unwrap());
    // State is unchanged: 'a' still accepted afterwards.
    assert!(matcher.accept_token(byte_token(b'a')).unwrap());
}

#[test]
fn accept_string_and_reset() {
    let compiled = compile("root ::= \"ab\" | \"cd\"");
    let mut matcher = GrammarMatcher::new(&compiled).unwrap();
    assert!(matcher.accept_string("cd").unwrap());
    assert!(matcher.is_completed().unwrap());
    assert!(!matcher.accept_string("zz").unwrap());

    matcher.reset().unwrap();
    assert!(!matcher.is_completed().unwrap());
    assert!(matcher.accept_string("ab").unwrap());
}

#[test]
fn jump_forward_string() {
    let compiled = compile("root ::= \"abc\" [0-9]");
    let mut matcher = GrammarMatcher::new(&compiled).unwrap();
    assert_eq!(matcher.find_jump_forward_string().unwrap(), "abc");
}

#[test]
fn rollback_and_fork() {
    let compiled = compile("root ::= \"ab\" | \"cd\"");
    let mut matcher = GrammarMatcher::new(&compiled).unwrap();

    assert!(matcher.accept_token(byte_token(b'a')).unwrap());
    matcher.rollback(1).unwrap();
    assert!(matcher.accept_token(byte_token(b'c')).unwrap());

    // Fork continues independently of the original.
    let mut forked = matcher.fork().unwrap();
    assert!(forked.accept_token(byte_token(b'd')).unwrap());
    assert!(forked.is_completed().unwrap());
    assert!(!matcher.is_completed().unwrap());
    assert!(matcher.accept_token(byte_token(b'd')).unwrap());
}

#[test]
fn captures_from_lark_grammar() {
    common::init();
    let ti = common::tiny_tokenizer_info();
    let grammar = Grammar::from_lark("start: \"a\" v \"b\"\nv[capture]: /[0-9]+/").unwrap();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();

    let mut matcher = GrammarMatcher::new(&compiled).unwrap();
    assert!(matcher.accept_string("a123b").unwrap());
    let captures = matcher.get_captures(true).unwrap();
    assert_eq!(captures, vec![("v".to_string(), b"123".to_vec())]);
}

#[test]
fn matcher_options_and_properties() {
    let compiled = compile("root ::= \"ab\"");

    let matcher = GrammarMatcher::new(&compiled).unwrap();
    assert_eq!(matcher.temperature().unwrap(), None);
    assert_eq!(matcher.stop_token_ids().unwrap(), vec![0]);
    assert_eq!(matcher.max_rollback_tokens().unwrap(), -1);
    assert!(!matcher.debug_print_internal_state().unwrap().is_empty());

    let matcher = GrammarMatcher::with_options(
        &compiled,
        &MatcherOptions {
            override_stop_tokens: Some(vec![5]),
            default_temperature: Some(0.75),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(matcher.stop_token_ids().unwrap(), vec![5]);
    assert_eq!(matcher.temperature().unwrap(), Some(0.75));

    // terminate_without_stop_token: completing the grammar terminates.
    let mut matcher = GrammarMatcher::with_options(
        &compiled,
        &MatcherOptions {
            terminate_without_stop_token: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matcher.accept_string("ab").unwrap());
    assert!(matcher.is_terminated().unwrap());
}

/// Mirrors `tests/python/test_speculative_decoding.py`.
#[test]
fn traverse_draft_tree() {
    common::init();
    let vocab: Vec<&[u8]> = vec![
        b"a", b"b", b"c", b"{", b"}", b"\"", b":", b",", b" ", b"true", b"false", b"null",
    ];
    let vocab_size = vocab.len();
    let ti = TokenizerInfo::new(
        vocab,
        VocabType::Raw,
        &TokenizerInfoOptions {
            vocab_size: Some(vocab_size),
            stop_token_ids: Some(vec![]),
            ..Default::default()
        },
    )
    .unwrap();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    let json = Grammar::builtin_json_grammar().unwrap();
    let compiled = compiler.compile_grammar(&json).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled).unwrap();

    // Linear tree 0 -> 1 -> 2 with draft tokens {, :, }
    let mut bitmask = TokenBitmask::with_batch(3, vocab_size);
    let ok = matcher
        .traverse_draft_tree(
            &[1, 2, -1],
            &[-1, -1, -1],
            &[3, 6, 4],
            &mut bitmask,
            -1.0,
            None,
        )
        .unwrap();
    assert!(ok);
    assert!(!bitmask.masked_tokens(0).is_empty());

    // Rejected node: draft token 'a' right after '{' is invalid, so its row
    // masks out everything.
    matcher.reset().unwrap();
    let mut bitmask = TokenBitmask::with_batch(3, vocab_size);
    let ok = matcher
        .traverse_draft_tree(
            &[1, 2, -1],
            &[-1, -1, -1],
            &[3, 0, 3],
            &mut bitmask,
            -1.0,
            None,
        )
        .unwrap();
    assert!(ok);
    assert_eq!(bitmask.masked_tokens(1).len(), vocab_size);
}
