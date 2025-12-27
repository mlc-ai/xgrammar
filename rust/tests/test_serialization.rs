mod test_utils;

use serial_test::serial;
use test_utils::is_grammar_accept_string;
#[cfg(feature = "hf")]
use test_utils::make_hf_tokenizer_info;
use xgrammar::{
    CompiledGrammar, Grammar, GrammarCompiler, TokenizerInfo, VocabType,
};

fn construct_grammar() -> Grammar {
    Grammar::from_ebnf(
        r#"rule1 ::= ([^0-9] rule1) | ""
root_rule ::= rule1 "a"
"#,
        "root_rule",
    ).unwrap()
}

fn construct_tokenizer_info() -> TokenizerInfo {
    let vocab = ["1", "212", "a", "A", "b", "一", "-", "aBc", "abc"];
    let stop_ids: Option<Box<[i32]>> = Some(Box::new([0, 1]));
    TokenizerInfo::new_with_vocab_size(
        &vocab,
        VocabType::BYTE_FALLBACK,
        Some(10),
        &stop_ids,
        true,
    ).unwrap()
}

fn construct_compiled_grammar() -> (CompiledGrammar, TokenizerInfo) {
    let tokenizer_info = construct_tokenizer_info();
    let grammar = construct_grammar();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();
    (compiled, tokenizer_info)
}

#[test]
#[serial]
fn test_serialize_grammar_roundtrip() {
    let orig = construct_grammar();
    let s = orig.serialize_json();
    let recovered = Grammar::deserialize_json(&s).expect("deserialize grammar");
    assert_eq!(orig.to_string(), recovered.to_string());
}

#[test]
#[serial]
fn test_get_serialization_version() {
    assert_eq!(xgrammar::get_serialization_version(), "v7");
}

#[test]
#[serial]
fn test_serialize_grammar_functional() {
    let grammar = construct_grammar();
    let s = grammar.serialize_json();
    let recovered = Grammar::deserialize_json(&s).expect("deserialize");

    let tok = construct_tokenizer_info();
    let mut compiler = GrammarCompiler::new(&tok, 1, false, -1).unwrap();
    let cg1 = compiler.compile_grammar(&grammar).unwrap();
    let cg2 = compiler.compile_grammar(&recovered).unwrap();

    let mut m1 = xgrammar::GrammarMatcher::new(&cg1, None, true, -1).unwrap();
    let mut m2 = xgrammar::GrammarMatcher::new(&cg2, None, true, -1).unwrap();
    let input = "aaa";
    assert_eq!(m1.accept_string(input, false), m2.accept_string(input, false));
}

#[test]
#[serial]
fn test_serialize_tokenizer_info_roundtrip() {
    let orig = construct_tokenizer_info();
    let s = orig.serialize_json();
    let recovered =
        TokenizerInfo::deserialize_json(&s).expect("deserialize tokenizer");
    assert_eq!(orig.vocab_type() as i32, recovered.vocab_type() as i32);
    assert_eq!(orig.vocab_size(), recovered.vocab_size());
    assert_eq!(orig.add_prefix_space(), recovered.add_prefix_space());
    assert_eq!(&*orig.stop_token_ids(), &*recovered.stop_token_ids());
    assert_eq!(&*orig.special_token_ids(), &*recovered.special_token_ids());
    // decoded vocab equality
    let dv1 = orig.decoded_vocab();
    let dv2 = recovered.decoded_vocab();
    assert_eq!(dv1.len(), dv2.len());
    for (a, b) in dv1.iter().zip(dv2.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
#[serial]
fn test_serialize_compiled_grammar_roundtrip() {
    let (orig_cg, tok) = construct_compiled_grammar();
    let s = orig_cg.serialize_json();
    let recovered = CompiledGrammar::deserialize_json(&s, &tok)
        .expect("deserialize compiled grammar");
    assert_eq!(orig_cg.serialize_json(), recovered.serialize_json());
}

#[test]
#[serial]
fn test_serialize_compiled_grammar_functional() {
    let (orig_cg, _tok) = construct_compiled_grammar();
    let s = orig_cg.serialize_json();
    let tok = construct_tokenizer_info();
    let recovered = CompiledGrammar::deserialize_json(&s, &tok)
        .expect("deserialize compiled grammar");

    let mut m1 = xgrammar::GrammarMatcher::new(&orig_cg, None, true, -1).unwrap();
    let mut m2 = xgrammar::GrammarMatcher::new(&recovered, None, true, -1).unwrap();
    let input = "aaa";
    assert_eq!(m1.accept_string(input, false), m2.accept_string(input, false));
    assert_eq!(m1.is_terminated(), m2.is_terminated());
}

#[test]
#[serial]
fn test_grammar_deserialize_errors() {
    // Invalid JSON
    assert!(Grammar::deserialize_json("not json").is_err());

    // Version mismatch or format mismatch: modify payload
    let g = construct_grammar();
    let mut v: serde_json::Value =
        serde_json::from_str(&g.serialize_json()).unwrap();
    if let Some(obj) = v.as_object_mut() {
        obj.insert("__VERSION__".to_string(), serde_json::json!("v1"));
    }
    assert!(Grammar::deserialize_json(&v.to_string()).is_err());
}

#[test]
#[serial]
fn test_compiled_grammar_deserialize_errors() {
    let (cg, tok) = construct_compiled_grammar();
    // Invalid JSON
    assert!(CompiledGrammar::deserialize_json("not json", &tok).is_err());

    // Format mismatch: alter tokenizer metadata version field by removing required key
    let mut v: serde_json::Value =
        serde_json::from_str(&cg.serialize_json()).unwrap();
    if let Some(obj) =
        v.get_mut("tokenizer_metadata").and_then(|x| x.as_object_mut())
    {
        obj.remove("vocab_size");
    }
    assert!(CompiledGrammar::deserialize_json(&v.to_string(), &tok).is_err());
}

#[test]
#[serial]
fn test_serialize_grammar() {
    let ebnf = r#"rule1 ::= ([^0-9] rule1) | ""
root_rule ::= rule1 "a"
"#;
    let grammar = Grammar::from_ebnf(ebnf, "root_rule").unwrap();
    let serialized = grammar.serialize_json();
    let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();

    assert!(parsed["__VERSION__"].is_string());
    assert!(parsed["rules"].is_array());
    assert!(parsed["grammar_expr_data"].is_array());
}

#[test]
#[serial]
fn test_serialize_tokenizer_info() {
    let vocab: Vec<&str> =
        vec!["1", "212", "a", "A", "b", "一", "-", "aBc", "abc"];
    let stop_ids: Box<[i32]> = vec![0, 1].into_boxed_slice();
    let tok = TokenizerInfo::new(
        &vocab,
        VocabType::BYTE_FALLBACK,
        &Some(stop_ids),
        true,
    ).unwrap();

    let serialized = tok.serialize_json();
    let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();

    assert!(parsed["__VERSION__"].is_string());
    assert!(!serialized.is_empty());
}

#[test]
#[serial]
fn test_serialize_tokenizer_info_functional() {
    let vocab: Vec<&str> =
        vec!["1", "212", "a", "A", "b", "一", "-", "aBc", "abc"];
    let stop_ids: Box<[i32]> = vec![0, 1].into_boxed_slice();
    let tok = TokenizerInfo::new(
        &vocab,
        VocabType::BYTE_FALLBACK,
        &Some(stop_ids),
        true,
    ).unwrap();

    let serialized = tok.serialize_json();
    let deserialized = TokenizerInfo::deserialize_json(&serialized).unwrap();

    assert_eq!(tok.vocab_size(), deserialized.vocab_size());
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_serialize_compiled_grammar() {
    let tok = make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let grammar = Grammar::builtin_json_grammar();
    let mut compiler = GrammarCompiler::new(&tok, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();

    let serialized = compiled.serialize_json();
    let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();

    assert!(parsed["__VERSION__"].is_string());
    assert!(parsed["grammar"].is_object());
}

#[test]
#[serial]
fn test_serialize_grammar_utf8() {
    let schema = r##"{"type": "string", "enum": ["こんにちは", "世界"]}"##;
    let grammar = Grammar::from_json_schema(
        schema,
        true,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();

    let serialized = grammar.serialize_json();
    let deserialized = Grammar::deserialize_json(&serialized).unwrap();

    let test_str = r#""こんにちは""#;
    assert!(is_grammar_accept_string(&grammar, test_str));
    assert!(is_grammar_accept_string(&deserialized, test_str));
}
