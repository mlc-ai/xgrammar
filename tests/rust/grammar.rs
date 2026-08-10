mod common;

use xgrammar::{Error, Grammar, JsonSchemaOptions};

#[test]
fn from_ebnf_and_display() {
    common::init();
    let grammar = Grammar::from_ebnf("root ::= \"a\" | \"bb\"").unwrap();
    let text = grammar.to_string();
    assert!(text.contains("root ::="), "unexpected EBNF: {text}");
}

#[test]
fn from_ebnf_reports_errors() {
    common::init();
    let err = Grammar::from_ebnf("root ::= (").unwrap_err();
    let message = err.to_string();
    assert!(!message.is_empty());
}

#[test]
fn from_json_schema_defaults() {
    common::init();
    let schema = r#"{"type": "object", "properties": {"a": {"type": "integer"}}}"#;
    let grammar = Grammar::from_json_schema(schema, &JsonSchemaOptions::default()).unwrap();
    assert!(grammar.to_string().contains("root"));
}

#[test]
fn from_json_schema_rejects_invalid_schema_type() {
    common::init();
    // Malformed JSON is rejected by an internal check on the C++ side, which
    // surfaces as a generic FFI error (same behavior as the Python API).
    let err = Grammar::from_json_schema("[1, 2", &JsonSchemaOptions::default()).unwrap_err();
    assert!(
        err.to_string().contains("Failed to parse JSON"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn serialize_roundtrip() {
    common::init();
    let grammar = Grammar::from_ebnf("root ::= \"x\" [0-9]*").unwrap();
    let json = grammar.serialize_json().unwrap();
    let restored = Grammar::deserialize_json(&json).unwrap();
    assert_eq!(grammar.to_string(), restored.to_string());
}

#[test]
fn deserialize_rejects_garbage() {
    common::init();
    let err = Grammar::deserialize_json("{not json").unwrap_err();
    assert!(
        matches!(
            err,
            Error::InvalidJson(_) | Error::DeserializeFormat(_) | Error::DeserializeVersion(_)
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn builtin_json_union_concat() {
    common::init();
    let json = Grammar::builtin_json_grammar().unwrap();
    let a = Grammar::from_ebnf("root ::= \"a\"").unwrap();
    let union = Grammar::union(&[json.clone(), a.clone()]).unwrap();
    let concat = Grammar::concat(&[a, json]).unwrap();
    assert!(union.to_string().contains("root"));
    assert!(concat.to_string().contains("root"));
}

#[test]
fn from_regex_and_lark() {
    common::init();
    let regex = Grammar::from_regex("[a-z]+@[a-z]+\\.com").unwrap();
    assert!(regex.to_string().contains("root"));

    let lark = Grammar::from_lark("start: \"hello\" WORD\nWORD: /[a-z]+/").unwrap();
    assert!(lark.to_string().contains("root"));
}

#[test]
fn clone_is_same_handle() {
    common::init();
    let grammar = Grammar::from_ebnf("root ::= \"a\"").unwrap();
    let clone = grammar.clone();
    assert_eq!(grammar, clone);
    let other = Grammar::from_ebnf("root ::= \"a\"").unwrap();
    assert_ne!(grammar, other);
}
