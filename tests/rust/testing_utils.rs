mod common;

use common::byte_token;
use xgrammar::structural_tag::JsonSchemaStyle;
use xgrammar::testing::{self, grammar_functor};
use xgrammar::{Grammar, GrammarCompiler, GrammarMatcher, JsonSchemaOptions, TokenBitmask};

#[test]
fn json_schema_to_ebnf() {
    common::init();
    let ebnf = testing::json_schema_to_ebnf(
        r#"{"type": "object", "properties": {"a": {"type": "integer"}}}"#,
        &JsonSchemaOptions::default(),
        JsonSchemaStyle::Json,
    )
    .unwrap();
    assert!(ebnf.contains("root"), "unexpected ebnf: {ebnf}");
}

#[test]
fn regex_to_ebnf() {
    common::init();
    let ebnf = testing::regex_to_ebnf("[a-z]+", true).unwrap();
    assert!(ebnf.contains("root"), "unexpected ebnf: {ebnf}");
}

#[test]
fn ebnf_without_normalization() {
    common::init();
    let grammar =
        testing::ebnf_to_grammar_no_normalization("root ::= \"a\" | \"b\"", "root").unwrap();
    assert!(grammar.to_string().contains("root"));
}

#[test]
fn bitmask_helpers_agree_with_native() {
    common::init();
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    let compiled = compiler.compile_ebnf("root ::= \"ab\"").unwrap();
    let mut matcher = GrammarMatcher::new(&compiled).unwrap();
    let mut bitmask = TokenBitmask::new(ti.vocab_size().unwrap());
    matcher.fill_next_token_bitmask(&mut bitmask, 0).unwrap();

    // The C++ helper and the native Rust helper must agree.
    let from_cpp = testing::get_masked_tokens_from_bitmask(&bitmask, 0).unwrap();
    assert_eq!(from_cpp, bitmask.masked_tokens(0));

    // Only 'a' is allowed at the start: single-token bitmask.
    let (single, token) = testing::is_single_token_bitmask(&bitmask, 0).unwrap();
    assert!(single);
    assert_eq!(token, byte_token(b'a'));
}

#[test]
fn allow_empty_rule_ids() {
    common::init();
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    let compiled = compiler.compile_ebnf("root ::= \"a\"*").unwrap();
    let ids = testing::get_allow_empty_rule_ids(&compiled).unwrap();
    assert!(!ids.is_empty());
}

#[test]
fn range_regexes() {
    common::init();
    let int_regex = testing::generate_range_regex(Some(1), Some(12)).unwrap();
    let grammar = Grammar::from_regex(&int_regex).unwrap();
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled).unwrap();
    assert!(matcher.accept_string("7").unwrap());
    let mut matcher = GrammarMatcher::new(&compiled).unwrap();
    assert!(!matcher.accept_string("13").unwrap());

    let float_regex = testing::generate_float_regex(Some(0.0), Some(1.0), false, false).unwrap();
    assert!(!float_regex.is_empty());
}

#[test]
fn print_grammar_fsms() {
    common::init();
    // Must use a *compiled* grammar: only compilation populates the per-rule
    // FSM table (like the Python tests do).
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    let compiled = compiler.compile_ebnf("root ::= \"abc\" [0-9]+").unwrap();
    let dump = testing::print_grammar_fsms(&compiled.grammar().unwrap()).unwrap();
    assert!(dump.contains("Rule"), "unexpected dump: {dump}");
}

#[test]
fn grammar_functor_passes() {
    common::init();
    let grammar = testing::ebnf_to_grammar_no_normalization(
        "root ::= sub sub\nsub ::= \"a\" | \"b\"",
        "root",
    )
    .unwrap();
    let normalized = grammar_functor::structure_normalizer(&grammar).unwrap();
    let fused = grammar_functor::byte_string_fuser(&normalized).unwrap();
    let inlined = grammar_functor::rule_inliner(&fused).unwrap();
    let cleaned = grammar_functor::dead_code_eliminator(&inlined).unwrap();
    let analyzed = grammar_functor::lookahead_assertion_analyzer(&cleaned).unwrap();
    let _ = grammar_functor::repetition_normalizer(&analyzed).unwrap();

    let optimized = grammar_functor::grammar_optimizer(&normalized).unwrap();
    assert!(optimized.to_string().contains("root"));
}
