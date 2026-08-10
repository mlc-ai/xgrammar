mod common;

use common::byte_token;
use xgrammar::{
    BatchGrammarMatcher, GrammarCompiler, GrammarMatcher, MatcherOptions, MaxThreads, TokenBitmask,
};

fn two_matchers() -> (xgrammar::CompiledGrammar, Vec<GrammarMatcher>) {
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    let compiled = compiler.compile_ebnf("root ::= \"ab\" | \"cd\"").unwrap();
    let matchers = vec![
        GrammarMatcher::new(&compiled).unwrap(),
        GrammarMatcher::new(&compiled).unwrap(),
    ];
    (compiled, matchers)
}

#[test]
fn batch_fill_next_token_bitmask() {
    let (compiled, mut matchers) = two_matchers();
    let vocab_size = compiled.tokenizer_info().unwrap().vocab_size().unwrap();

    // Put the two matchers into different states.
    assert!(matchers[0].accept_token(byte_token(b'a')).unwrap());
    assert!(matchers[1].accept_token(byte_token(b'c')).unwrap());

    let mut batch = BatchGrammarMatcher::new(MaxThreads::Auto).unwrap();
    let mut bitmask = TokenBitmask::with_batch(2, vocab_size);
    batch
        .batch_fill_next_token_bitmask(&mut matchers, &mut bitmask, None)
        .unwrap();

    assert!(bitmask.is_allowed(0, byte_token(b'b') as usize));
    assert!(!bitmask.is_allowed(0, byte_token(b'd') as usize));
    assert!(bitmask.is_allowed(1, byte_token(b'd') as usize));
    assert!(!bitmask.is_allowed(1, byte_token(b'b') as usize));
}

#[test]
fn batch_fill_with_indices() {
    let (compiled, mut matchers) = two_matchers();
    let vocab_size = compiled.tokenizer_info().unwrap().vocab_size().unwrap();
    assert!(matchers[0].accept_token(byte_token(b'a')).unwrap());
    assert!(matchers[1].accept_token(byte_token(b'c')).unwrap());

    // Swap rows via indices.
    let mut batch = BatchGrammarMatcher::new(MaxThreads::Count(2)).unwrap();
    let mut bitmask = TokenBitmask::with_batch(2, vocab_size);
    batch
        .batch_fill_next_token_bitmask(&mut matchers, &mut bitmask, Some(&[1, 0]))
        .unwrap();
    assert!(bitmask.is_allowed(1, byte_token(b'b') as usize));
    assert!(bitmask.is_allowed(0, byte_token(b'd') as usize));
}

#[test]
fn batch_accept_token_and_rollback() {
    let (_compiled, mut matchers) = two_matchers();

    let accepted = BatchGrammarMatcher::batch_accept_token(
        &mut matchers,
        &[byte_token(b'a'), byte_token(b'x')],
    )
    .unwrap();
    assert_eq!(accepted, vec![true, false]);

    BatchGrammarMatcher::batch_rollback(&mut matchers, &[1, 0]).unwrap();
    let accepted = BatchGrammarMatcher::batch_accept_token(
        &mut matchers,
        &[byte_token(b'c'), byte_token(b'c')],
    )
    .unwrap();
    assert_eq!(accepted, vec![true, true]);
}

#[test]
fn batch_accept_string() {
    let (_compiled, mut matchers) = two_matchers();
    let accepted = BatchGrammarMatcher::batch_accept_string(&mut matchers, &["ab", "zz"]).unwrap();
    assert_eq!(accepted, vec![true, false]);
    assert!(matchers[0].is_completed().unwrap());
    assert!(!matchers[1].is_completed().unwrap());
}

#[test]
fn batch_fill_temperature() {
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    let compiled = compiler.compile_ebnf("root ::= \"ab\"").unwrap();
    let mut matchers: Vec<GrammarMatcher> = [0.5f64, 0.9]
        .iter()
        .map(|&t| {
            GrammarMatcher::with_options(
                &compiled,
                &MatcherOptions {
                    default_temperature: Some(t),
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect();

    let mut temperatures = vec![0.0f32; 2];
    BatchGrammarMatcher::batch_fill_temperature(&mut matchers, &mut temperatures, None).unwrap();
    assert_eq!(temperatures, vec![0.5, 0.9]);
}
