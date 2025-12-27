mod test_utils;

use serial_test::serial;
use test_utils::*;
#[cfg(feature = "hf")]
use xgrammar::allocate_token_bitmask;
use xgrammar::{
    Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo, VocabType,
};

#[test]
#[serial]
fn test_json_schema_find_jump_forward_string() {
    let schema = r##"{
        "type": "object",
        "properties": {
            "integer_field": {"type": "integer"},
            "number_field": {"type": "number"},
            "boolean_field": {"type": "boolean"},
            "array_field": {"type": "array", "items": {"type": "string"}},
            "object_field": {"type": "object"}
        },
        "required": ["integer_field", "number_field", "boolean_field", "array_field", "object_field"]
    }"##;

    let grammar = Grammar::from_json_schema(
        schema,
        true,
        Some(2),
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();
    let vocab: Vec<&str> = vec![];
    let tokenizer_info =
        TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();

    let jump_str = matcher.find_jump_forward_string();
    assert!(!jump_str.is_empty());
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_fill_next_token_bitmask() {
    let tk = make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let schema = r##"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }"##;

    let grammar = Grammar::from_json_schema(
        schema,
        true,
        Some(2),
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();
    let mut compiler = GrammarCompiler::new(&tk, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();

    let input_str = r##"{
  "name": "John",
  "age": 30
}"##;

    let mut bitmask_data = allocate_token_bitmask(1, tk.vocab_size());
    let (mut tensor, _shape, _strides) =
        create_bitmask_dltensor(&mut bitmask_data, 1, tk.vocab_size());

    for c in input_str.bytes() {
        matcher.fill_next_token_bitmask(&mut tensor, 0, false);
        matcher.accept_string(&String::from_utf8(vec![c]).unwrap(), false);
    }

    assert!(matcher.is_terminated());
}

#[test]
#[serial]
fn test_implicit_left_recursion_schema() {
    let schema = r##"{
        "type": "object",
        "properties": {
            "value": {"$ref": "#"}
        }
    }"##;

    let grammar = Grammar::from_json_schema(
        schema,
        true,
        None,
        None::<(&str, &str)>,
        false,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(&grammar, r#"{}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"value": {}}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"value": {"value": {}}}"#));
}

#[test]
#[serial]
fn test_json_schema_number_without_constraint() {
    let schema = r##"{"type": "number"}"##;
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

    assert!(is_grammar_accept_string(&grammar, "42"));
    assert!(is_grammar_accept_string(&grammar, "3.14"));
    assert!(is_grammar_accept_string(&grammar, "-123.456"));
    assert!(is_grammar_accept_string(&grammar, "1e10"));
    assert!(!is_grammar_accept_string(&grammar, r#""42""#));
}
