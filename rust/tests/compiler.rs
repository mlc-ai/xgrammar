mod common;

use xgrammar::{
    CompiledGrammar, Grammar, GrammarCompiler, GrammarCompilerOptions, JsonSchemaOptions,
};

#[test]
fn compile_and_inspect() {
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    let compiled = compiler.compile_ebnf("root ::= \"ab\" | \"cd\"").unwrap();

    assert!(compiled.memory_size_bytes().unwrap() > 0);
    let grammar = compiled.grammar().unwrap();
    assert!(grammar.to_string().contains("root"));
    let back = compiled.tokenizer_info().unwrap();
    assert_eq!(back.vocab_size().unwrap(), ti.vocab_size().unwrap());
}

#[test]
fn compile_all_sources() {
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();

    let grammar = Grammar::from_ebnf("root ::= [0-9]+").unwrap();
    compiler.compile_grammar(&grammar).unwrap();
    compiler.compile_builtin_json_grammar().unwrap();
    compiler.compile_regex("[a-c]+").unwrap();
    compiler
        .compile_json_schema(
            r#"{"type": "object", "properties": {"a": {"type": "string"}}}"#,
            &JsonSchemaOptions::default(),
        )
        .unwrap();
}

#[test]
fn cache_controls() {
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    assert_eq!(compiler.cache_limit_bytes().unwrap(), None);
    assert_eq!(compiler.cache_size_bytes().unwrap(), 0);

    compiler.compile_ebnf("root ::= \"ab\"").unwrap();
    let after_compile = compiler.cache_size_bytes().unwrap();
    assert!(after_compile > 0);
    compiler.clear_cache().unwrap();
    // Note: as of the current C++ implementation, `GetCacheSizeBytes` does not
    // drop back to zero after `ClearCache` (the LRU accounting is not reset);
    // the Python API behaves identically. Only assert it does not grow.
    assert!(compiler.cache_size_bytes().unwrap() <= after_compile);

    let limited = GrammarCompiler::with_options(
        &ti,
        &GrammarCompilerOptions {
            max_threads: 2,
            cache_enabled: true,
            cache_limit_bytes: Some(1 << 20),
        },
    )
    .unwrap();
    assert_eq!(limited.cache_limit_bytes().unwrap(), Some(1 << 20));
}

#[test]
fn compiled_grammar_serialization_roundtrip() {
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    let compiled = compiler.compile_ebnf("root ::= \"ab\"").unwrap();

    let json = compiled.serialize_json().unwrap();
    let restored = CompiledGrammar::deserialize_json(&json, &ti).unwrap();

    // The restored grammar must be usable for matching.
    let mut matcher = xgrammar::GrammarMatcher::new(&restored).unwrap();
    assert!(matcher.accept_string("ab").unwrap());
    assert!(matcher.is_completed().unwrap());
}

#[test]
fn compiled_grammar_deserialize_rejects_garbage() {
    let ti = common::tiny_tokenizer_info();
    let err = CompiledGrammar::deserialize_json("{definitely not valid", &ti).unwrap_err();
    assert!(!err.to_string().is_empty());
}
