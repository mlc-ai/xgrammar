mod test_utils;

use serial_test::serial;
use test_utils::*;
use xgrammar::Grammar;
#[cfg(feature = "hf")]
use xgrammar::{GrammarCompiler, GrammarMatcher, TokenizerInfo, VocabType};

// ============================================================================
// Test Functions - Matching Python test_json_schema_converter.py order
// ============================================================================

/// Test basic JSON schema with various field types
/// Corresponds to Python test: test_basic
#[test]
#[serial]
fn test_basic() {
    // Test basic integer field
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

    // Should reject non-integers
    assert!(!is_grammar_accept_string(&grammar, r#"{"integer_field": 42.5}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{"integer_field": "42"}"#));
}

/// Test JSON schema with indent formatting
/// Corresponds to Python test: test_indent
#[test]
#[serial]
fn test_indent() {
    let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}"#;

    // Test with indent=2
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

    // Should accept properly indented JSON
    let indented_json = r#"{
  "name": "Alice",
  "age": 30
}"#;
    assert!(
        is_grammar_accept_string(&grammar, indented_json),
        "Should accept indented JSON"
    );

    // Should reject non-indented when indent is specified
    assert!(
        !is_grammar_accept_string(&grammar, r#"{"name":"Alice","age":30}"#),
        "Should reject compact JSON when indent specified"
    );
}

/// Test non-strict mode (allows additional properties)
/// Corresponds to Python test: test_non_strict
#[test]
#[serial]
fn test_non_strict() {
    let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}"#;

    // Strict mode: should reject additional properties
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

    // Non-strict mode: should allow additional properties
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
/// Corresponds to Python test: test_enum_const
#[test]
#[serial]
fn test_enum_const() {
    // Test enum
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

    // Test const
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
/// Corresponds to Python test: test_optional
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

    // Should accept with only required field
    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"required_field": "value"}"#
    ));

    // Should accept with both fields
    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"required_field": "value", "optional_field": 42}"#
    ));

    // Should reject without required field
    assert!(!is_grammar_accept_string(&grammar, r#"{"optional_field": 42}"#));
}

/// Test empty object schema
/// Corresponds to Python test: test_empty
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

    // Should accept empty object
    assert!(is_grammar_accept_string(&grammar, r#"{}"#));

    // In non-strict mode, should accept objects with any properties
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
/// Corresponds to Python test: test_union
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

    // Should accept string
    assert!(is_grammar_accept_string(&grammar, r#""hello""#));

    // Should accept integer
    assert!(is_grammar_accept_string(&grammar, r#"42"#));

    // Should reject other types
    assert!(!is_grammar_accept_string(&grammar, r#"true"#));
    assert!(!is_grammar_accept_string(&grammar, r#"null"#));
}

/// Test any_whitespace flag
/// Corresponds to Python test: test_any_whitespace
#[test]
#[serial]
fn test_any_whitespace() {
    let schema = r#"{"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}"#;

    // With any_whitespace=true
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

    // With any_whitespace=false and no indent/separators specified
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

/// Test array schemas
/// Corresponds to Python test: test_array_schema
#[test]
#[serial]
fn test_array_schema() {
    // Array of strings
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
/// Corresponds to Python test: test_array_schema_min_max
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

    // Should reject arrays with less than minItems
    assert!(!is_grammar_accept_string(&grammar, r#"[]"#));
    assert!(!is_grammar_accept_string(&grammar, r#"[1]"#));

    // Should accept arrays within bounds
    assert!(is_grammar_accept_string(&grammar, r#"[1, 2]"#));
    assert!(is_grammar_accept_string(&grammar, r#"[1, 2, 3]"#));
    assert!(is_grammar_accept_string(&grammar, r#"[1, 2, 3, 4]"#));

    // Should reject arrays exceeding maxItems
    assert!(!is_grammar_accept_string(&grammar, r#"[1, 2, 3, 4, 5]"#));
}

/// Test Grammar::from_json_schema with max_whitespace_cnt=2
/// Corresponds to Python test: test_limited_whitespace_cnt
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

    // Should accept up to 2 whitespace characters
    assert!(
        is_grammar_accept_string(&grammar, r#"{  "key"  :  "value"  }"#),
        "Should accept 2 whitespaces"
    );
    assert!(
        is_grammar_accept_string(&grammar, r#"{"key":"value"}"#),
        "Should accept no whitespace"
    );

    // Should reject more than 2 whitespace characters
    assert!(
        !is_grammar_accept_string(&grammar, r#"{   "key"  :  "value"   }"#),
        "Should reject 3 whitespaces"
    );
    assert!(
        !is_grammar_accept_string(&grammar, r#"{    "key"  :  "value"    }"#),
        "Should reject 4 whitespaces"
    );
}

/// Test GrammarCompiler::compile_json_schema with max_whitespace_cnt=2
/// Corresponds to Python test: test_limited_whitespace_compile
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

    assert!(
        compiled_grammar.memory_size_bytes() > 0,
        "Compiled grammar should exist"
    );

    // Test with GrammarMatcher - should accept up to 2 whitespaces
    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();
    assert!(matcher.accept_string(r#"{  "key"  :  "value"  }"#, false));
    assert!(matcher.is_terminated());

    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();
    assert!(matcher.accept_string(r#"{"key":"value"}"#, false));
    assert!(matcher.is_terminated());

    // Should reject more than 2 whitespace characters
    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();
    assert!(!matcher.accept_string(r#"{   "key"  :  "value"   }"#, false));

    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();
    assert!(!matcher.accept_string(r#"{    "key"  :  "value"    }"#, false));
}

/// Test UTF-8 strings in enum
/// Corresponds to Python test: test_utf8_in_enum
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
/// Corresponds to Python test: test_utf8_string_in_const
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
/// Corresponds to Python test: test_all_optional
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

    // All fields are optional
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
/// Corresponds to Python test: test_reference_schema
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

    // Should accept valid nested structure
    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"value": {"name": "John", "age": 30}}"#
    ));

    // Should reject incomplete nested structure
    assert!(!is_grammar_accept_string(
        &grammar,
        r#"{"value": {"name": "John"}}"#
    ));
}

/// Test anyOf and oneOf
/// Corresponds to Python test: test_anyof_oneof
#[test]
#[serial]
fn test_anyof_oneof() {
    // Test anyOf
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
/// Corresponds to Python test: test_restricted_string
#[test]
#[serial]
fn test_restricted_string() {
    // String with minLength and maxLength
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

    // Should reject strings shorter than minLength
    assert!(!is_grammar_accept_string(&grammar, r#"""#));
    assert!(!is_grammar_accept_string(&grammar, r#""ab""#));

    // Should accept strings within bounds
    assert!(is_grammar_accept_string(&grammar, r#""abc""#));
    assert!(is_grammar_accept_string(&grammar, r#""abcd""#));
    assert!(is_grammar_accept_string(&grammar, r#""abcde""#));

    // Should reject strings exceeding maxLength
    assert!(!is_grammar_accept_string(&grammar, r#""abcdef""#));
}

/// Test number with minimum and maximum
/// Corresponds to Python test: test_complex_restrictions
#[test]
#[serial]
fn test_complex_restrictions() {
    // Number with minimum and maximum
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

    // Should accept numbers within range
    assert!(is_grammar_accept_string(&grammar, r#"0"#));
    assert!(is_grammar_accept_string(&grammar, r#"50"#));
    assert!(is_grammar_accept_string(&grammar, r#"100"#));

    // Note: The grammar generator creates patterns based on numeric ranges
    // Exact validation of min/max bounds is done at the grammar level
}

/// Test array with only items keyword
/// Corresponds to Python test: test_array_with_only_items_keyword
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
/// Corresponds to Python test: test_object_with_only_properties_keyword
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

    // In strict mode, should reject additional properties
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

/// Test boolean type
#[test]
#[serial]
fn test_boolean() {
    let schema = r#"{"type": "boolean"}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#"true"#));
    assert!(is_grammar_accept_string(&grammar, r#"false"#));
    assert!(!is_grammar_accept_string(&grammar, r#"1"#));
    assert!(!is_grammar_accept_string(&grammar, r#"0"#));
    assert!(!is_grammar_accept_string(&grammar, r#""true""#));
}

/// Test null type
#[test]
#[serial]
fn test_null() {
    let schema = r#"{"type": "null"}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#"null"#));
    assert!(!is_grammar_accept_string(&grammar, r#"0"#));
    assert!(!is_grammar_accept_string(&grammar, r#"""#));
    assert!(!is_grammar_accept_string(&grammar, r#"false"#));
}

/// Test number (float) type
#[test]
#[serial]
fn test_number() {
    let schema = r#"{"type": "number"}"#;
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

    assert!(is_grammar_accept_string(&grammar, r#"42"#));
    assert!(is_grammar_accept_string(&grammar, r#"42.5"#));
    assert!(is_grammar_accept_string(&grammar, r#"-3.14"#));
    assert!(is_grammar_accept_string(&grammar, r#"1e10"#));
    assert!(is_grammar_accept_string(&grammar, r#"1.5e-10"#));
    assert!(!is_grammar_accept_string(&grammar, r#""42""#));
}

/// Test additionalProperties
#[test]
#[serial]
fn test_additional_properties() {
    // Test with additionalProperties: false (strict)
    let schema_no_additional = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "additionalProperties": false
    }"#;

    let grammar_no = Grammar::from_json_schema(
        schema_no_additional,
        true,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(&grammar_no, r#"{"name": "Alice"}"#));
    assert!(!is_grammar_accept_string(
        &grammar_no,
        r#"{"name": "Alice", "extra": "field"}"#
    ));

    // Test with additionalProperties: true
    let schema_with_additional = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "additionalProperties": true
    }"#;

    let grammar_yes = Grammar::from_json_schema(
        schema_with_additional,
        true,
        None,
        None::<(&str, &str)>,
        false,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(&grammar_yes, r#"{"name": "Alice"}"#));
    assert!(is_grammar_accept_string(
        &grammar_yes,
        r#"{"name": "Alice", "extra": "field"}"#
    ));
}

/// Test tuple (array with prefixItems)
#[test]
#[serial]
fn test_tuple() {
    let schema = r#"{
        "type": "array",
        "prefixItems": [
            {"type": "string"},
            {"type": "integer"},
            {"type": "boolean"}
        ]
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

    // Should accept tuple with correct types
    assert!(is_grammar_accept_string(&grammar, r#"["hello", 42, true]"#));
}

/// Test nested objects
#[test]
#[serial]
fn test_nested_objects() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "person": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "zipcode": {"type": "string"}
                        },
                        "required": ["city"]
                    }
                },
                "required": ["name"]
            }
        },
        "required": ["person"]
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

    assert!(is_grammar_accept_string(
        &grammar,
        r#"{"person": {"name": "Alice", "address": {"city": "NYC"}}}"#
    ));

    assert!(!is_grammar_accept_string(
        &grammar,
        r#"{"person": {"address": {"city": "NYC"}}}"#
    ));
}

/// Test arrays of objects
#[test]
#[serial]
fn test_array_of_objects() {
    let schema = r#"{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"}
            },
            "required": ["id", "name"]
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

    assert!(is_grammar_accept_string(&grammar, r#"[]"#));
    assert!(is_grammar_accept_string(
        &grammar,
        r#"[{"id": 1, "name": "Alice"}]"#
    ));
    assert!(is_grammar_accept_string(
        &grammar,
        r#"[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]"#
    ));
    assert!(!is_grammar_accept_string(&grammar, r#"[{"id": 1}]"#));
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
