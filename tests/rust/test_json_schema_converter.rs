#![allow(clippy::approx_constant)]

mod test_utils;

use serial_test::serial;
use test_utils::*;
use xgrammar::Grammar;
use serde_json::{Value, json};
use xgrammar::testing::{generate_float_range_regex, generate_range_regex, json_schema_to_ebnf};
#[cfg(feature = "hf")]
use xgrammar::{GrammarCompiler, GrammarMatcher, TokenizerInfo, VocabType};

const BASIC_JSON_RULES_EBNF: &str = r#"basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
"#;

#[allow(dead_code)]
const BASIC_JSON_RULES_EBNF_NO_SPACE: &str = r#"basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" "" basic_any (", " basic_any)* "" "]") | ("[" "" "]"))
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{" "}"
"#;

fn check_schema_with_grammar(
    schema: &Value,
    expected_grammar_ebnf: &str,
    any_whitespace: bool,
    indent: Option<i32>,
    separators: Option<(&str, &str)>,
    strict_mode: bool,
) {
    let schema_json = serde_json::to_string(schema).expect("serialize schema");
    let json_schema_ebnf = json_schema_to_ebnf(
        &schema_json,
        any_whitespace,
        indent,
        separators,
        strict_mode,
        None,
    );
    assert_eq!(json_schema_ebnf, expected_grammar_ebnf);
}

fn check_schema_with_instance(
    schema: &Value,
    instance: &str,
    is_accepted: bool,
    any_whitespace: bool,
    indent: Option<i32>,
    separators: Option<(&str, &str)>,
    strict_mode: bool,
) {
    let schema_json = serde_json::to_string(schema).expect("serialize schema");
    let json_schema_grammar = Grammar::from_json_schema(
        &schema_json,
        any_whitespace,
        indent,
        separators,
        strict_mode,
        None,
        false,
    )
    .unwrap();
    assert_eq!(
        is_grammar_accept_string(&json_schema_grammar, instance),
        is_accepted
    );
}

/// Test basic JSON schema with various field types
#[test]
#[serial]
fn test_basic() {
    let schema = r#"{"type": "object", "properties": {"integer_field": {"type": "integer"}}, "required": ["integer_field"]}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#"{"integer_field": 42}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"integer_field": -123}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"integer_field": 0}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{"integer_field": 42.5}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{"integer_field": "42"}"#));
}

/// Test JSON schema with indent formatting
#[test]
#[serial]
fn test_indent() {
    let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}"#;
    let grammar = Grammar::from_json_schema(
        schema,
        false,
        Some(2),
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();

    let indented_json = r#"{
  "name": "Alice",
  "age": 30
}"#;
    assert!(is_grammar_accept_string(&grammar, indented_json));
    assert!(!is_grammar_accept_string(&grammar, r#"{"name":"Alice","age":30}"#));
}

/// Test non-strict mode (allows additional properties)
#[test]
#[serial]
fn test_non_strict() {
    let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}"#;

    let grammar_strict = Grammar::from_json_schema(
        schema,
        true,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();
    assert!(is_grammar_accept_string(&grammar_strict, r#"{"name": "Alice"}"#));
    assert!(!is_grammar_accept_string(
        &grammar_strict,
        r#"{"name": "Alice", "extra": "field"}"#
    ));

    let grammar_non_strict = Grammar::from_json_schema(
        schema,
        true,
        None,
        None::<(&str, &str)>,
        false,
        None,
        false,
    )
    .unwrap();
    assert!(is_grammar_accept_string(
        &grammar_non_strict,
        r#"{"name": "Alice"}"#
    ));
    assert!(is_grammar_accept_string(
        &grammar_non_strict,
        r#"{"name": "Alice", "extra": "field"}"#
    ));
}

/// Test enum and const constraints
#[test]
#[serial]
fn test_enum_const() {
    let schema = r#"{"type": "string", "enum": ["red", "green", "blue"]}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#""red""#));
    assert!(is_grammar_accept_string(&grammar, r#""green""#));
    assert!(is_grammar_accept_string(&grammar, r#""blue""#));
    assert!(!is_grammar_accept_string(&grammar, r#""yellow""#));

    let schema_const = r#"{"const": "fixed_value"}"#;
    let grammar_const = Grammar::from_json_schema(
        schema_const,
        true,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(&grammar_const, r#""fixed_value""#));
    assert!(!is_grammar_accept_string(&grammar_const, r#""other_value""#));
}

/// Test optional properties
#[test]
#[serial]
fn test_optional() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "required_field": {"type": "string"},
            "optional_field": {"type": "integer"}
        },
        "required": ["required_field"]
    }"#;

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

    assert!(is_grammar_accept_string(&grammar, r#"{"required_field": "value"}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"required_field": "value", "optional_field": 42}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{"optional_field": 42}"#));
}

/// Test empty object schema
#[test]
#[serial]
fn test_empty() {
    let schema = r#"{"type": "object"}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#"{}"#));

    let grammar_non_strict = Grammar::from_json_schema(
        schema,
        true,
        None,
        None::<(&str, &str)>,
        false,
        None,
        false,
    )
    .unwrap();
    assert!(is_grammar_accept_string(
        &grammar_non_strict,
        r#"{"any": "value"}"#
    ));
}

/// Test union types (anyOf)
#[test]
#[serial]
fn test_union() {
    let schema = r#"{"anyOf": [{"type": "string"}, {"type": "integer"}]}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#""hello""#));
    assert!(is_grammar_accept_string(&grammar, r#"42"#));
    assert!(!is_grammar_accept_string(&grammar, r#"true"#));
    assert!(!is_grammar_accept_string(&grammar, r#"null"#));
}

/// Test any_whitespace flag
#[test]
#[serial]
fn test_any_whitespace() {
    let schema = r#"{"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}"#;

    let grammar_any = Grammar::from_json_schema(
        schema,
        true,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();
    assert!(is_grammar_accept_string(&grammar_any, r#"{"key":"value"}"#));
    assert!(is_grammar_accept_string(&grammar_any, r#"{ "key" : "value" }"#));
    assert!(is_grammar_accept_string(
        &grammar_any,
        r#"{  "key"  :  "value"  }"#
    ));

    let grammar_no_any = Grammar::from_json_schema(
        schema,
        false,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();
    assert!(is_grammar_accept_string(&grammar_no_any, r#"{"key": "value"}"#));
    assert!(!is_grammar_accept_string(
        &grammar_no_any,
        r#"{  "key"  :  "value"  }"#
    ));
}

#[test]
#[serial]
fn test_array_schema_error_cases() {
    let schema_err_message = vec![
        (
            json!({"type": "array", "prefixItems": {"type": "string"}}),
            "prefixItems must be an array",
        ),
        (
            json!({"type": "array", "prefixItems": ["not an object"]}),
            "prefixItems must be an array of objects or booleans",
        ),
        (
            json!({"type": "array", "prefixItems": [false]}),
            "prefixItems contains false",
        ),
        (
            json!({"type": "array", "items": "not an object"}),
            "items must be a boolean or an object",
        ),
        (
            json!({"type": "array", "unevaluatedItems": "not an object"}),
            "unevaluatedItems must be a boolean or an object",
        ),
        (
            json!({"type": "array", "minItems": "not an integer"}),
            "minItems must be an integer",
        ),
        (
            json!({"type": "array", "maxItems": -1}),
            "maxItems must be a non-negative integer",
        ),
        (
            json!({"type": "array", "minItems": 5, "maxItems": 3}),
            "minItems is greater than maxItems: 5 > 3",
        ),
        (
            json!({"type": "array", "prefixItems": [{}, {}, {}], "maxItems": 2}),
            "maxItems is less than the number of prefixItems: 2 < 3",
        ),
        (
            json!({"type": "array", "prefixItems": [{}, {}], "minItems": 3, "items": false}),
            "minItems is greater than the number of prefixItems, but additional items are not allowed: 3 > 2",
        ),
    ];

    for (schema, err_message) in schema_err_message {
        let schema_json = serde_json::to_string(&schema).expect("serialize schema");
        let result = Grammar::from_json_schema(
            &schema_json,
            true,
            None,
            None::<(&str, &str)>,
            true,
            None,
            false,
        );
        match result {
            Ok(_) => panic!("expected error for schema"),
            Err(err) => assert!(
                err.contains(err_message),
                "expected error containing '{}', got '{}'",
                err_message,
                err
            ),
        }
    }
}

/// Test array schemas
#[test]
#[serial]
fn test_array_schema() {
    let schema = r#"{"type": "array", "items": {"type": "string"}}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#"[]"#));
    assert!(is_grammar_accept_string(&grammar, r#"["a"]"#));
    assert!(is_grammar_accept_string(&grammar, r#"["a", "b", "c"]"#));
    assert!(!is_grammar_accept_string(&grammar, r#"[1, 2, 3]"#));
}

/// Test array with minItems and maxItems
#[test]
#[serial]
fn test_array_schema_min_max() {
    let schema = r#"{"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 4}"#;
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

    assert!(!is_grammar_accept_string(&grammar, r#"[]"#));
    assert!(!is_grammar_accept_string(&grammar, r#"[1]"#));
    assert!(is_grammar_accept_string(&grammar, r#"[1, 2]"#));
    assert!(is_grammar_accept_string(&grammar, r#"[1, 2, 3]"#));
    assert!(is_grammar_accept_string(&grammar, r#"[1, 2, 3, 4]"#));
    assert!(!is_grammar_accept_string(&grammar, r#"[1, 2, 3, 4, 5]"#));
}

/// Test Grammar::from_json_schema with max_whitespace_cnt=2
#[test]
#[serial]
fn test_limited_whitespace_cnt() {
    let schema = r#"{"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}"#;

    let grammar = Grammar::from_json_schema(
        schema,
        true,                 // any_whitespace
        None,                 // indent
        None::<(&str, &str)>, // separators
        true,                 // strict_mode
        Some(2),              // max_whitespace_cnt=2
        false,                // print_converted_ebnf
    )
    .unwrap();

    assert!(is_grammar_accept_string(&grammar, r#"{  "key"  :  "value"  }"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"key":"value"}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{   "key"  :  "value"   }"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{    "key"  :  "value"    }"#));
}

/// Test GrammarCompiler::compile_json_schema with max_whitespace_cnt=2
#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_limited_whitespace_compile() {
    let schema = r#"{"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}"#;

    let empty_vocab: Vec<&str> = vec![];
    let stop_ids: Option<Box<[i32]>> = None;
    let tokenizer_info =
        TokenizerInfo::new(&empty_vocab, VocabType::RAW, &stop_ids, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 8, true, -1).unwrap();

    let compiled_grammar = compiler.compile_json_schema(
        schema,
        true,                 // any_whitespace
        None,                 // indent
        None::<(&str, &str)>, // separators
        true,                 // strict_mode
        Some(2),              // max_whitespace_cnt=2
    ).unwrap();

    assert!(compiled_grammar.memory_size_bytes() > 0);

    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();
    assert!(matcher.accept_string(r#"{  "key"  :  "value"  }"#, false));
    assert!(matcher.is_terminated());

    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();
    assert!(matcher.accept_string(r#"{"key":"value"}"#, false));
    assert!(matcher.is_terminated());

    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();
    assert!(!matcher.accept_string(r#"{   "key"  :  "value"   }"#, false));

    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();
    assert!(!matcher.accept_string(r#"{    "key"  :  "value"    }"#, false));
}

/// Test UTF-8 strings in enum
#[test]
#[serial]
fn test_utf8_in_enum() {
    let schema = r#"{"type": "string", "enum": ["こんにちは", "😊", "你好", "hello", "\n"]}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#""こんにちは""#));
    assert!(is_grammar_accept_string(&grammar, r#""😊""#));
    assert!(is_grammar_accept_string(&grammar, r#""你好""#));
    assert!(is_grammar_accept_string(&grammar, r#""hello""#));
    assert!(is_grammar_accept_string(&grammar, r#""\n""#));
}

/// Test UTF-8 string in const
#[test]
#[serial]
fn test_utf8_string_in_const() {
    let schema = r#"{"const": "常数constじょうすう\n\r\t"}"#;
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

    assert!(is_grammar_accept_string(
        &grammar,
        r#""常数constじょうすう\n\r\t""#
    ));
}

/// Test all properties optional
#[test]
#[serial]
fn test_all_optional() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"}
        }
    }"#;

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

    assert!(is_grammar_accept_string(&grammar, r#"{}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"name": "Alice"}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"age": 30}"#));
    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"name": "Alice", "age": 30}"#
    ));
    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"name": "Alice", "age": 30, "email": "alice@example.com"}"#
    ));
}

/// Test reference with $defs
#[test]
#[serial]
fn test_reference_schema() {
    let schema = r##"{
        "type": "object",
        "properties": {
            "value": {"$ref": "#/$defs/nested"}
        },
        "required": ["value"],
        "$defs": {
            "nested": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    }"##;

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

    assert!(is_grammar_accept_string(&grammar, r#"{"value": {"name": "John", "age": 30}}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{"value": {"name": "John"}}"#));
}

/// Test anyOf and oneOf
#[test]
#[serial]
fn test_anyof_oneof() {
    let schema_anyof = r#"{
        "anyOf": [
            {"type": "string"},
            {"type": "integer"},
            {"type": "boolean"}
        ]
    }"#;

    let grammar = Grammar::from_json_schema(
        schema_anyof,
        true,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(&grammar, r#""hello""#));
    assert!(is_grammar_accept_string(&grammar, r#"42"#));
    assert!(is_grammar_accept_string(&grammar, r#"true"#));
    assert!(!is_grammar_accept_string(&grammar, r#"null"#));
}

/// Test string with pattern restriction
#[test]
#[serial]
fn test_restricted_string() {
    let schema = r#"{
        "type": "string",
        "minLength": 3,
        "maxLength": 5
    }"#;

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

    assert!(!is_grammar_accept_string(&grammar, r#"""#));
    assert!(!is_grammar_accept_string(&grammar, r#""ab""#));
    assert!(is_grammar_accept_string(&grammar, r#""abc""#));
    assert!(is_grammar_accept_string(&grammar, r#""abcd""#));
    assert!(is_grammar_accept_string(&grammar, r#""abcde""#));
    assert!(!is_grammar_accept_string(&grammar, r#""abcdef""#));
}

/// Test number with minimum and maximum
#[test]
#[serial]
fn test_complex_restrictions() {
    let schema = r#"{
        "type": "integer",
        "minimum": 0,
        "maximum": 100
    }"#;

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

    assert!(is_grammar_accept_string(&grammar, r#"0"#));
    assert!(is_grammar_accept_string(&grammar, r#"50"#));
    assert!(is_grammar_accept_string(&grammar, r#"100"#));
}

/// Test array with only items keyword
#[test]
#[serial]
fn test_array_with_only_items_keyword() {
    let schema = r#"{
        "type": "array",
        "items": {"type": "integer"}
    }"#;

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

    assert!(is_grammar_accept_string(&grammar, r#"[]"#));
    assert!(is_grammar_accept_string(&grammar, r#"[1]"#));
    assert!(is_grammar_accept_string(&grammar, r#"[1, 2, 3]"#));
    assert!(!is_grammar_accept_string(&grammar, r#"["not", "integers"]"#));
}

/// Test object with only properties keyword
#[test]
#[serial]
fn test_object_with_only_properties_keyword() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }"#;

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

    assert!(is_grammar_accept_string(&grammar, r#"{}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"name": "Alice"}"#));
    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"name": "Alice", "age": 30}"#
    ));
    assert!(!is_grammar_accept_string(
        &grammar,
        r#"{"name": "Alice", "extra": "field"}"#
    ));
}

#[test]
#[serial]
fn test_all_optional_non_strict() {
    let schema = r##"{"type": "object", "properties": {"size": {"type": "integer", "default": 0}, "state": {"type": "boolean", "default": false}, "num": {"type": "number", "default": 0}}}"##;

    let grammar = Grammar::from_json_schema(
        schema,
        false,
        None,
        None::<(&str, &str)>,
        false,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"size": 1, "num": 1.5, "other": false}"#
    ));
    assert!(is_grammar_accept_string(&grammar, r#"{"other": false}"#));
}

#[test]
#[serial]
fn test_reference() {
    let schema = r##"{
        "type": "object",
        "properties": {
            "foo": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "size": {"type": "number"}
                },
                "required": ["count"]
            },
            "bars": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "apple": {"type": "string"},
                        "banana": {"type": "string"}
                    }
                }
            }
        },
        "required": ["foo", "bars"]
    }"##;

    let grammar = Grammar::from_json_schema(
        schema,
        false,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"foo": {"count": 42, "size": 3.14}, "bars": [{"apple": "a", "banana": "b"}, {"apple": "c", "banana": "d"}]}"#
    ));
}

#[test]
#[serial]
fn test_alias() {
    let schema = r##"{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}"##;

    let grammar = Grammar::from_json_schema(
        schema,
        false,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(&grammar, r#"{"name": "kitty"}"#));
}

#[test]
#[serial]
fn test_dynamic_model() {
    let schema = r##"{"type": "object", "properties": {"restricted_string": {"type": "string", "pattern": "[a-f]"}, "restricted_string_dynamic": {"type": "string", "pattern": "[a-x]"}}, "required": ["restricted_string", "restricted_string_dynamic"]}"##;

    let grammar = Grammar::from_json_schema(
        schema,
        false,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"restricted_string": "a", "restricted_string_dynamic": "j"}"#
    ));
}

#[test]
#[serial]
fn test_object_with_pattern_properties_and_property_names() {
    let schema = r##"{
        "type": "object",
        "patternProperties": {
            "^[a-zA-Z]+$": {"type": "string"},
            "^[0-9]+$": {"type": "integer"},
            "^[a-zA-Z]*_[0-9]*$": {"type": "object"}
        }
    }"##;

    let grammar = Grammar::from_json_schema(
        schema,
        false,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(&grammar, r#"{"aBcDe": "aaa"}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"12345": 12345}"#));
    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"abc_123": {"key": "value"}}"#
    ));
    assert!(is_grammar_accept_string(&grammar, r#"{"_": {"key": "value"}}"#));
    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"a": "value", "b": "another_value", "000": 12345, "abc_123": {"key": "value"}}"#
    ));

    assert!(!is_grammar_accept_string(&grammar, r#"{"233A": "adfa"}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{"aBcDe": 12345}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{"12345": "aaa"}"#));

    let schema_property_names = r##"{
        "type": "object",
        "propertyNames": {"pattern": "^[a-zA-Z0-9_]+$"}
    }"##;

    let grammar_prop_names = Grammar::from_json_schema(
        schema_property_names,
        false,
        None,
        None::<(&str, &str)>,
        false,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(
        &grammar_prop_names,
        r#"{"aBcDe": "aaa"}"#
    ));
    assert!(is_grammar_accept_string(
        &grammar_prop_names,
        r#"{"12345": 12345}"#
    ));
    assert!(is_grammar_accept_string(
        &grammar_prop_names,
        r#"{"abc_123": {"key": "value"}}"#
    ));

    assert!(!is_grammar_accept_string(
        &grammar_prop_names,
        r#"{"aBc?De": "aaa"}"#
    ));
    assert!(!is_grammar_accept_string(
        &grammar_prop_names,
        r#"{"1234|5": 12345}"#
    ));
}

#[test]
#[serial]
fn test_object_with_property_numbers() {
    let schema = r##"{"type": "object", "properties": {"123": {"type": "string"}, "456": {"type": "integer"}}, "required": ["123"]}"##;

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

    assert!(is_grammar_accept_string(&grammar, r#"{"123": "value"}"#));
    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"123": "value", "456": 789}"#
    ));
}

#[test]
#[serial]
fn test_object_error_handle() {
    // Test error handling for invalid object schemas
    let compile_from_schema = |schema: &Value| {
        let schema_json = serde_json::to_string(schema).expect("serialize schema");
        Grammar::from_json_schema(
            &schema_json,
            true,
            None,
            None::<(&str, &str)>,
            true,
            None,
            false,
        )
        .map(|_| ())
    };

    let schema = json!({"type": "object", "properties": "not an object"});
    let err = compile_from_schema(&schema).expect_err("expected error");
    assert!(err.contains("properties must be an object"));

    let schema = json!({"type": "object", "required": {"key": "not an array"}});
    let err = compile_from_schema(&schema).expect_err("expected error");
    assert!(err.contains("required must be an array"));

    let err = compile_from_schema(&json!({"type": "object", "patternProperties": ["not an object"]}))
        .expect_err("expected error");
    assert!(err.contains("patternProperties must be an object"));

    let err = compile_from_schema(&json!({"type": "object", "propertyNames": "not an object"}))
        .expect_err("expected error");
    assert!(err.contains("propertyNames must be an object"));

    let err = compile_from_schema(&json!({"type": "object", "propertyNames": {"type": "object"}}))
        .expect_err("expected error");
    assert!(err.contains("propertyNames must be an object that validates string"));

    let err = compile_from_schema(&json!({"type": "object", "minProperties": "not an integer"}))
        .expect_err("expected error");
    assert!(err.contains("minProperties must be an integer"));

    let err = compile_from_schema(&json!({"type": "object", "maxProperties": "not an integer"}))
        .expect_err("expected error");
    assert!(err.contains("maxProperties must be an integer"));

    let err = compile_from_schema(&json!({"type": "object", "minProperties": -1}))
        .expect_err("expected error");
    assert!(err.contains("minProperties must be a non-negative integer"));

    let err = compile_from_schema(&json!({"type": "object", "maxProperties": -1}))
        .expect_err("expected error");
    assert!(err.contains("maxProperties must be a non-negative integer"));

    let err = compile_from_schema(&json!({"type": "object", "minProperties": 5, "maxProperties": 3}))
        .expect_err("expected error");
    assert!(err.contains("minxPropertiesmax is greater than maxProperties"));

    let err = compile_from_schema(&json!({"type": "object", "maxProperties": 1, "required": ["key1", "key2"]}))
        .expect_err("expected error");
    assert!(err.contains("maxProperties is less than the number of required properties"));

    let err = compile_from_schema(&json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {"key": {"type": "string"}},
        "minProperties": 2,
    }))
    .expect_err("expected error");
    assert!(
        err.contains(
            "minProperties is greater than the number of properties, but additional properties aren't allowed"
        )
    );
}

#[test]
#[serial]
fn test_generate_range_regex() {
    // Basic range tests
    assert_eq!(
        generate_range_regex(Some(12), Some(16)).unwrap(),
        r"^((1[2-6]))$"
    );
    assert_eq!(
        generate_range_regex(Some(1), Some(10)).unwrap(),
        r"^(([1-9]|10))$"
    );
    assert_eq!(
        generate_range_regex(Some(2134), Some(3459)).unwrap(),
        r"^((2[2-9]\d{2}|2[2-9]\d{2}|21[4-9]\d{1}|213[5-9]|2134|3[0-3]\d{2}|3[0-3]\d{2}|34[0-4]\d{1}|345[0-8]|3459))$"
    );

    // Negative to positive range
    assert_eq!(
        generate_range_regex(Some(-5), Some(10)).unwrap(),
        r"^(-([1-5])|0|([1-9]|10))$"
    );

    // Pure negative range
    assert_eq!(
        generate_range_regex(Some(-15), Some(-10)).unwrap(),
        r"^(-(1[0-5]))$"
    );

    // Large ranges
    assert_eq!(
        generate_range_regex(Some(-1999), Some(-100)).unwrap(),
        r"^(-([1-9]\d{2}|1[0-8]\d{2}|19[0-8]\d{1}|199[0-8]|1999))$"
    );
    assert_eq!(
        generate_range_regex(Some(1), Some(9999)).unwrap(),
        r"^(([1-9]|[1-9]\d{1}|[1-9]\d{2}|[1-9]\d{3}))$"
    );
}

#[test]
#[serial]
fn test_min_max_length() {
    let schema = r##"{"type": "string", "minLength": 2, "maxLength": 5}"##;

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

    assert!(!is_grammar_accept_string(&grammar, r#"""#));
    assert!(!is_grammar_accept_string(&grammar, r#""a""#));
    assert!(is_grammar_accept_string(&grammar, r#""ab""#));
    assert!(is_grammar_accept_string(&grammar, r#""abc""#));
    assert!(is_grammar_accept_string(&grammar, r#""abcde""#));
    assert!(!is_grammar_accept_string(&grammar, r#""abcdef""#));
}

#[test]
#[serial]
fn test_type_array() {
    let schema = r##"{"type": ["string", "integer", "null"]}"##;

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

    assert!(is_grammar_accept_string(&grammar, r#""hello""#));
    assert!(is_grammar_accept_string(&grammar, r#"42"#));
    assert!(is_grammar_accept_string(&grammar, r#"null"#));
    assert!(!is_grammar_accept_string(&grammar, r#"true"#));
    assert!(!is_grammar_accept_string(&grammar, r#"[]"#));
}

#[test]
#[serial]
fn test_type_array_empty() {
    let schema = r##"{"type": []}"##;

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

    assert!(is_grammar_accept_string(&grammar, r#""hello""#));
    assert!(is_grammar_accept_string(&grammar, r#"42"#));
    assert!(is_grammar_accept_string(&grammar, r#"null"#));
}

#[test]
#[serial]
fn test_empty_array() {
    let schema = r##"{"type": "array", "maxItems": 0}"##;

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

    assert!(is_grammar_accept_string(&grammar, r#"[]"#));
    assert!(!is_grammar_accept_string(&grammar, r#"[1]"#));
}

#[test]
#[serial]
fn test_empty_object() {
    let schema = r##"{"type": "object", "maxProperties": 0}"##;

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

    assert!(is_grammar_accept_string(&grammar, r#"{}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{"a": 1}"#));
}

#[test]
#[serial]
fn test_primitive_type_string() {
    let schema = r##"{"type": "string"}"##;

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

    assert!(is_grammar_accept_string(&grammar, r#""hello""#));
    assert!(!is_grammar_accept_string(&grammar, r#"42"#));
}

#[test]
#[serial]
fn test_primitive_type_object() {
    let schema = r##"{"type": "object"}"##;

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
    assert!(is_grammar_accept_string(&grammar, r#"{"a": 1}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"[]"#));
}

#[test]
#[serial]
fn test_generate_float_regex() {
    assert_eq!(
        generate_float_range_regex(Some(1.0), Some(5.0)).unwrap(),
        r"^(1|5|(([2-4]))(\.\d{1,6})?|1\.\d{1,6}|5\.\d{1,6})$"
    );

    assert_eq!(
        generate_float_range_regex(Some(1.5), Some(5.75)).unwrap(),
        r"^(1.5|5.75|(([2-4]))(\.\d{1,6})?|1\.6\d{0,5}|1\.7\d{0,5}|1\.8\d{0,5}|1\.9\d{0,5}|5\.0\d{0,5}|5\.1\d{0,5}|5\.2\d{0,5}|5\.3\d{0,5}|5\.4\d{0,5}|5\.5\d{0,5}|5\.6\d{0,5}|5\.70\d{0,4}|5\.71\d{0,4}|5\.72\d{0,4}|5\.73\d{0,4}|5\.74\d{0,4})$"
    );

    assert_eq!(
        generate_float_range_regex(Some(-3.14), Some(2.71828)).unwrap(),
        r"^(-3.14|2.71828|(-([1-3])|0|(1))(\.\d{1,6})?|-3\.0\d{0,5}|-3\.10\d{0,4}|-3\.11\d{0,4}|-3\.12\d{0,4}|-3\.13\d{0,4}|2\.0\d{0,5}|2\.1\d{0,5}|2\.2\d{0,5}|2\.3\d{0,5}|2\.4\d{0,5}|2\.5\d{0,5}|2\.6\d{0,5}|2\.70\d{0,4}|2\.710\d{0,3}|2\.711\d{0,3}|2\.712\d{0,3}|2\.713\d{0,3}|2\.714\d{0,3}|2\.715\d{0,3}|2\.716\d{0,3}|2\.717\d{0,3}|2\.7180\d{0,2}|2\.7181\d{0,2}|2\.71820\d{0,1}|2\.71821\d{0,1}|2\.71822\d{0,1}|2\.71823\d{0,1}|2\.71824\d{0,1}|2\.71825\d{0,1}|2\.71826\d{0,1}|2\.71827\d{0,1})$"
    );

    assert_eq!(
        generate_float_range_regex(Some(0.5), None).unwrap(),
        r"^(0.5|0\.6\d{0,5}|0\.7\d{0,5}|0\.8\d{0,5}|0\.9\d{0,5}|([1-9]|[1-9]\d*)(\.\d{1,6})?)$"
    );

    assert_eq!(
        generate_float_range_regex(None, Some(-1.5)).unwrap(),
        r"^(-1.5|-1\.6\d{0,5}|-1\.7\d{0,5}|-1\.8\d{0,5}|-1\.9\d{0,5}|(-[3-9]|-[1-9]\d*)(\.\d{1,6})?)$"
    );
}

#[test]
#[serial]
fn test_utf8_object_array_in_enum() {
    let schema = r##"{
        "type": "object",
        "enum": [
            {"key": "こんにちは"},
            {"key": "😊"},
            {"key": "你好"},
            {"key": "hello"},
            {"key": "\n"},
            [123, "こんにちは", "😊", "你好", "hello", "\n"]
        ]
    }"##;

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

    assert!(is_grammar_accept_string(&grammar, r#"{"key":"こんにちは"}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"key":"😊"}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"key":"你好"}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"key":"hello"}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"key":"\n"}"#));
    assert!(is_grammar_accept_string(
        &grammar,
        r#"[123,"こんにちは","😊","你好","hello","\n"]"#
    ));
}

#[test]
#[serial]
fn test_utf8_object_const() {
    let schema = r##"{"type": "object", "const": {"key": "こんにちは常数constじょうすう\n\r\t"}}"##;

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

    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"key":"こんにちは常数constじょうすう\n\r\t"}"#
    ));
}

#[test]
#[serial]
fn test_utf8_array_const() {
    let schema = r##"{"type": "array", "const": ["こんにちは", "😊", "你好", "hello", "\n"]}"##;

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

    assert!(is_grammar_accept_string(
        &grammar,
        r#"["こんにちは","😊","你好","hello","\n"]"#
    ));
}

// ============================================================================
// Format Validation Tests - Matching Python test_json_schema_converter.py
// ============================================================================

/// Test email format validation
#[test]
#[serial]
fn test_email_format() {
    let schema = r#"{"type": "string", "format": "email"}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#""simple@example.com""#));
    assert!(is_grammar_accept_string(&grammar, r#""very.common@example.com""#));
    assert!(is_grammar_accept_string(&grammar, r#""FirstName.LastName@EasierReading.org""#));
    assert!(is_grammar_accept_string(&grammar, r#""x@example.com""#));
    assert!(is_grammar_accept_string(&grammar, r#""long.email-address-with-hyphens@and.subdomains.example.com""#));
    assert!(is_grammar_accept_string(&grammar, r#""user.name+tag+sorting@example.com""#));
    assert!(is_grammar_accept_string(&grammar, r#""admin@example""#));
    assert!(is_grammar_accept_string(&grammar, r#""example@s.example""#));
    assert!(!is_grammar_accept_string(&grammar, r#""abc.example.com""#));
    assert!(!is_grammar_accept_string(&grammar, r#""a@b@c@example.com""#));
}

/// Test date format validation
#[test]
#[serial]
fn test_date_format() {
    let schema = r#"{"type": "string", "format": "date"}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#""0000-01-01""#));
    assert!(is_grammar_accept_string(&grammar, r#""9999-12-31""#));
    assert!(is_grammar_accept_string(&grammar, r#""2024-06-15""#));
    assert!(!is_grammar_accept_string(&grammar, r#""10-01-01""#));
    assert!(!is_grammar_accept_string(&grammar, r#""2025-00-01""#));
    assert!(!is_grammar_accept_string(&grammar, r#""2025-13-01""#));
    assert!(!is_grammar_accept_string(&grammar, r#""2025-01-00""#));
    assert!(!is_grammar_accept_string(&grammar, r#""2025-01-32""#));
}

/// Test time format validation
#[test]
#[serial]
fn test_time_format() {
    let schema = r#"{"type": "string", "format": "time"}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#""00:00:00Z""#));
    assert!(is_grammar_accept_string(&grammar, r#""23:59:60Z""#));
    assert!(is_grammar_accept_string(&grammar, r#""12:34:56Z""#));
    assert!(is_grammar_accept_string(&grammar, r#""12:34:56+07:08""#));
    assert!(is_grammar_accept_string(&grammar, r#""12:34:56-07:08""#));
    assert!(is_grammar_accept_string(&grammar, r#""12:34:56.7Z""#));
    assert!(!is_grammar_accept_string(&grammar, r#""00:00:00""#));
    assert!(!is_grammar_accept_string(&grammar, r#""24:00:00Z""#));
    assert!(!is_grammar_accept_string(&grammar, r#""00:60:00Z""#));
}

#[test]
#[serial]
fn test_ipv6_format() {
    let instance_accepted = [
        (r"0123:4567:890a:bced:fABC:DEF0:1234:5678", true),
        (r"::6666:6666:6666:6666:6666:6666", true),
        (r"::6666:6666:6666:6666:6666", true),
        (r"::6666:6666:6666:6666", true),
        (r"::6666:6666:6666", true),
        (r"::6666:6666", true),
        (r"::6666", true),
        (r"::", true),
        (r"8888:8888:8888:8888:8888:8888::", true),
        (r"8888:8888:8888:8888:8888::", true),
        (r"8888:8888:8888:8888::", true),
        (r"8888:8888:8888::", true),
        (r"8888:8888::", true),
        (r"8888::", true),
        (r"1111::2222", true),
        (r"1111:1111::2222", true),
        (r"1111::2222:2222", true),
        (r"1111:1111:1111::2222", true),
        (r"1111:1111::2222:2222", true),
        (r"1111::2222:2222:2222", true),
        (r"1111:1111:1111:1111::2222", true),
        (r"1111:1111:1111::2222:2222", true),
        (r"1111:1111::2222:2222:2222", true),
        (r"1111::2222:2222:2222:2222", true),
        (r"1111:1111:1111:1111:1111::2222", true),
        (r"1111:1111:1111:1111::2222:2222", true),
        (r"1111:1111:1111::2222:2222:2222", true),
        (r"1111:1111::2222:2222:2222:2222", true),
        (r"1111::2222:2222:2222:2222:2222", true),
        (r"1111:1111:1111:1111:1111:1111::2222", true),
        (r"1111:1111:1111:1111:1111::2222:2222", true),
        (r"1111:1111:1111:1111::2222:2222:2222", true),
        (r"1111:1111:1111::2222:2222:2222:2222", true),
        (r"1111:1111::2222:2222:2222:2222:2222", true),
        (r"1111::2222:2222:2222:2222:2222:2222", true),
        (r"2001:db8:3:4::192.0.2.33", true),
        (r"64:ff9b::192.0.2.33", true),
        (r"::ffff:0:255.255.255.255", true),
        (r"::111.111.222.222", true),
        (r":", false),
        (r":::", false),
        (r"::5555:5555:5555:5555:5555:5555:5555:5555", false),
        (r"5555::5555:5555:5555:5555:5555:5555:5555", false),
        (r"5555:5555::5555:5555:5555:5555:5555:5555", false),
        (r"5555:5555:5555::5555:5555:5555:5555:5555", false),
        (r"5555:5555:5555:5555::5555:5555:5555:5555", false),
        (r"5555:5555:5555:5555:5555::5555:5555:5555", false),
        (r"5555:5555:5555:5555:5555:5555::5555:5555", false),
        (r"5555:5555:5555:5555:5555:5555:5555::5555", false),
        (r"5555:5555:5555:5555:5555:5555:5555:5555::", false),
    ];
    let schema = json!({"type": "string", "format": "ipv6"});
    let expected_grammar = format!(
        r#"{basic}string ::= "\"" ( ( [0-9a-fA-F]{{1,4}} ":" ){{7,7}} [0-9a-fA-F]{{1,4}} | ( [0-9a-fA-F]{{1,4}} ":" ){{1,7}} ":" | ( [0-9a-fA-F]{{1,4}} ":" ){{1,6}} ":" [0-9a-fA-F]{{1,4}} | ( [0-9a-fA-F]{{1,4}} ":" ){{1,5}} ( ":" [0-9a-fA-F]{{1,4}} ){{1,2}} | ( [0-9a-fA-F]{{1,4}} ":" ){{1,4}} ( ":" [0-9a-fA-F]{{1,4}} ){{1,3}} | ( [0-9a-fA-F]{{1,4}} ":" ){{1,3}} ( ":" [0-9a-fA-F]{{1,4}} ){{1,4}} | ( [0-9a-fA-F]{{1,4}} ":" ){{1,2}} ( ":" [0-9a-fA-F]{{1,4}} ){{1,5}} | [0-9a-fA-F]{{1,4}} ":" ( ( ":" [0-9a-fA-F]{{1,4}} ){{1,6}} ) | ":" ( ( ":" [0-9a-fA-F]{{1,4}} ){{1,7}} | ":" ) | ":" ":" ( "f" "f" "f" "f" ( ":" "0"{{1,4}} ){{0,1}} ":" ){{0,1}} ( ( "2" "5" [0-5] | ( "2" [0-4] | "1"{{0,1}} [0-9] ){{0,1}} [0-9] ) "." ){{3,3}} ( "2" "5" [0-5] | ( "2" [0-4] | "1"{{0,1}} [0-9] ){{0,1}} [0-9] ) | ( [0-9a-fA-F]{{1,4}} ":" ){{1,4}} ":" ( ( "2" "5" [0-5] | ( "2" [0-4] | "1"{{0,1}} [0-9] ){{0,1}} [0-9] ) "." ){{3,3}} ( "2" "5" [0-5] | ( "2" [0-4] | "1"{{0,1}} [0-9] ){{0,1}} [0-9] ) ) "\""
root ::= string
"#,
        basic = BASIC_JSON_RULES_EBNF
    );
    check_schema_with_grammar(&schema, &expected_grammar, true, None, None, true);

    for (instance, accepted) in instance_accepted {
        let value = format!("\"{}\"", instance);
        check_schema_with_instance(&schema, &value, accepted, true, None, None, true);
    }
}

/// Test IPv4 format validation
#[test]
#[serial]
fn test_ipv4_format() {
    let schema = r#"{"type": "string", "format": "ipv4"}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#""00.00.00.00""#));
    assert!(is_grammar_accept_string(&grammar, r#""000.000.000.000""#));
    assert!(is_grammar_accept_string(&grammar, r#""255.255.255.255""#));
    assert!(!is_grammar_accept_string(&grammar, r#""1""#));
    assert!(!is_grammar_accept_string(&grammar, r#""1.1""#));
    assert!(!is_grammar_accept_string(&grammar, r#""1.1.1""#));
    assert!(!is_grammar_accept_string(&grammar, r#""256.256.256.256""#));
}

/// Test hostname format validation
#[test]
#[serial]
fn test_hostname_format() {
    let schema = r#"{"type": "string", "format": "hostname"}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#""0""#));
    assert!(is_grammar_accept_string(&grammar, r#""a""#));
    assert!(is_grammar_accept_string(&grammar, r#""www.github.com""#));
    assert!(is_grammar_accept_string(&grammar, r#""w-w-w.g-i-t-h-u-b.c-o-m""#));
    assert!(!is_grammar_accept_string(&grammar, r#"".""#));
    assert!(!is_grammar_accept_string(&grammar, r#""-""#));
    assert!(!is_grammar_accept_string(&grammar, r#""_""#));
    assert!(!is_grammar_accept_string(&grammar, r#""a.""#));
    assert!(!is_grammar_accept_string(&grammar, r#""-b""#));
    assert!(!is_grammar_accept_string(&grammar, r#""c-""#));
}

/// Test UUID format validation
#[test]
#[serial]
fn test_uuid_format() {
    let schema = r#"{"type": "string", "format": "uuid"}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#""00000000-0000-0000-0000-000000000000""#));
    assert!(is_grammar_accept_string(&grammar, r#""FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF""#));
    assert!(is_grammar_accept_string(&grammar, r#""01234567-89AB-CDEF-abcd-ef0123456789""#));
    assert!(!is_grammar_accept_string(&grammar, r#""-""#));
    assert!(!is_grammar_accept_string(&grammar, r#""----""#));
    assert!(!is_grammar_accept_string(&grammar, r#""AAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA""#));
    assert!(!is_grammar_accept_string(&grammar, r#""AAAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA""#));
}

/// Test duration format validation
#[test]
#[serial]
fn test_duration_format() {
    let schema = r#"{"type": "string", "format": "duration"}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#""P0Y""#));
    assert!(is_grammar_accept_string(&grammar, r#""P12M""#));
    assert!(is_grammar_accept_string(&grammar, r#""P345D""#));
    assert!(is_grammar_accept_string(&grammar, r#""P6789W""#));
    assert!(is_grammar_accept_string(&grammar, r#""PT9H""#));
    assert!(is_grammar_accept_string(&grammar, r#""PT87M""#));
    assert!(is_grammar_accept_string(&grammar, r#""PT654S""#));
    assert!(is_grammar_accept_string(&grammar, r#""P1Y23M456D""#));
    assert!(is_grammar_accept_string(&grammar, r#""PT9H87M654S""#));
    assert!(is_grammar_accept_string(&grammar, r#""P1Y23M456DT9H87M654S""#));
    assert!(!is_grammar_accept_string(&grammar, r#""P""#));
    assert!(!is_grammar_accept_string(&grammar, r#""PD""#));
    assert!(!is_grammar_accept_string(&grammar, r#""P1""#));
    assert!(!is_grammar_accept_string(&grammar, r#""PT""#));
}

/// Test URI format validation
#[test]
#[serial]
fn test_uri_format() {
    let schema = r#"{"type": "string", "format": "uri"}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#""z+.-:""#));
    assert!(is_grammar_accept_string(&grammar, r#""abc:""#));
    assert!(is_grammar_accept_string(&grammar, r#""abc:a""#));
    assert!(is_grammar_accept_string(&grammar, r#""abc:/""#));
    assert!(is_grammar_accept_string(&grammar, r#""abc:/a""#));
    assert!(is_grammar_accept_string(&grammar, r#""abc://""#));
    assert!(!is_grammar_accept_string(&grammar, r#""abc://@@""#));
    assert!(!is_grammar_accept_string(&grammar, r#""abc://::""#));
}

#[test]
#[serial]
fn test_uri_reference_format() {
    let instance_accepted = [
        (r"?azAZ09-._~%Ff!$&'()*+,;=:@#azAZ09-._~%Aa!$&'()*+,;=:@", true),
        (r"", true),
        (r"a", true),
        (r"/", true),
        (r"/a", true),
        (r"//", true),
        (r"/////////", true),
        (r"//azAZ09-._~%Ff!$&'()*+,;=:@", true),
        (r"//:", true),
        (r"//:0123", true),
        (r"//azAZ09-._~%Ff!$&'()*+,;=", true),
        (r"/a", true),
        (r"/azAZ09-._~%Ff!$&'()*+,;=:@", true),
        (r"?[#]", false),
        (r"//@@", false),
        (r"//::", false),
        (r"/[]", false),
        (r":", false),
    ];
    let schema = json!({"type": "string", "format": "uri-reference"});
    let expected_grammar = [BASIC_JSON_RULES_EBNF, r##"string ::= "\"" ( "/" "/" ( ( [a-zA-Z0-9_.~!$&'()*+,;=:-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* "@" )? ( [a-zA-Z0-9_.~!$&'()*+,;=-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* ( ":" [0-9]* )? ( "/" ( [a-zA-Z0-9_.~!$&'()*+,;=:@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )* | "/" ( ( [a-zA-Z0-9_.~!$&'()*+,;=:@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )+ ( "/" ( [a-zA-Z0-9_.~!$&'()*+,;=:@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )* )? | ( [a-zA-Z0-9_.~!$&'()*+,;=@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )+ ( "/" ( [a-zA-Z0-9_.~!$&'()*+,;=:@-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )* )? ( "\?" ( [a-zA-Z0-9_.~!$&'()*+,;=:@/\?-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )? ( "#" ( [a-zA-Z0-9_.~!$&'()*+,;=:@/\?-] | "%" [0-9A-Fa-f] [0-9A-Fa-f] )* )? "\""
root ::= string
"##].concat();
    check_schema_with_grammar(&schema, &expected_grammar, true, None, None, true);

    for (instance, accepted) in instance_accepted {
        let value = format!("\"{}\"", instance);
        check_schema_with_instance(&schema, &value, accepted, true, None, None, true);
    }
}

#[test]
#[serial]
fn test_uri_template_format() {
    let instance_accepted = [
        (r"", true),
        (r"!#$&()*+,-./09:;=?@AZ[]_az~%Ff", true),
        (r"{+a}{#a}{.a}{/a}{;a}{?a}{&a}{=a}{,a}{!a}{@a}{|a}", true),
        (r"{%Ff}", true),
        (r"{i.j.k}", true),
        (r"{a_b_c:1234}", true),
        (r"{x_y_z*}", true),
        ("\"", false),
        ("'", false),
        (r"%", false),
        (r"<", false),
        (r">", false),
        (r"\\", false),
        (r"^", false),
        (r"`", false),
        (r"{", false),
        (r"|", false),
        (r"}", false),
        (r"{n.}", false),
        (r"{m:100001}", false),
        (r"%1", false),
        (r"%Gg", false),
    ];
    let schema = json!({"type": "string", "format": "uri-template"});
    let expected_grammar = [BASIC_JSON_RULES_EBNF, r#"string ::= "\"" ( ( [!#-$&(-;=\?-[\]_a-z~] | "%" [0-9A-Fa-f] [0-9A-Fa-f] ) | "{" ( [+#./;\?&=,!@|] )? ( [a-zA-Z0-9_] | "%" [0-9A-Fa-f] [0-9A-Fa-f] ) ( "."? ( [a-zA-Z0-9_] | "%" [0-9A-Fa-f] [0-9A-Fa-f] ) )* ( ":" [1-9] [0-9]? [0-9]? [0-9]? | "*" )? ( "," ( [a-zA-Z0-9_] | "%" [0-9A-Fa-f] [0-9A-Fa-f] ) ( "."? ( [a-zA-Z0-9_] | "%" [0-9A-Fa-f] [0-9A-Fa-f] ) )* ( ":" [1-9] [0-9]? [0-9]? [0-9]? | "*" )? )* "}" )* "\""
root ::= string
"#].concat();
    check_schema_with_grammar(&schema, &expected_grammar, true, None, None, true);

    for (instance, accepted) in instance_accepted {
        let value = format!("\"{}\"", instance);
        check_schema_with_instance(&schema, &value, accepted, true, None, None, true);
    }
}

#[test]
#[serial]
fn test_json_pointer_format() {
    let instance_accepted = [
        (r"/", true),
        (r"//", true),
        (r"/a/bc/def/ghij", true),
        (r"/~0/~1/", true),
        (r"abc", false),
        (r"/~", false),
        (r"/~2", false),
    ];
    let schema = json!({"type": "string", "format": "json-pointer"});
    let expected_grammar = [BASIC_JSON_RULES_EBNF, r#"string ::= "\"" ( "/" ( [\0-.] | [0-}] | [\x7f-\U0010ffff] | "~" [01] )* )* "\""
root ::= string
"#].concat();
    check_schema_with_grammar(&schema, &expected_grammar, true, None, None, true);

    for (instance, accepted) in instance_accepted {
        let value = format!("\"{}\"", instance);
        check_schema_with_instance(&schema, &value, accepted, true, None, None, true);
    }
}

#[test]
#[serial]
fn test_relative_json_pointer_format() {
    let instance_accepted = [
        (r"0/", true),
        (r"123/a/bc/def/ghij", true),
        (r"45/~0/~1/", true),
        (r"6789#", true),
        (r"#", false),
        (r"abc", false),
        (r"/", false),
        (r"9/~2", false),
    ];
    let schema = json!({"type": "string", "format": "relative-json-pointer"});
    let expected_grammar = [BASIC_JSON_RULES_EBNF, r##"string ::= "\"" ( "0" | [1-9] [0-9]* ) ( "#" | ( "/" ( [\0-.] | [0-}] | [\x7f-\U0010ffff] | "~" [01] )* )* ) "\""
root ::= string
"##].concat();
    check_schema_with_grammar(&schema, &expected_grammar, true, None, None, true);

    for (instance, accepted) in instance_accepted {
        let value = format!("\"{}\"", instance);
        check_schema_with_instance(&schema, &value, accepted, true, None, None, true);
    }
}
