mod test_utils;

use serial_test::serial;
use test_utils::*;
use xgrammar::Grammar;

#[test]
#[serial]
fn test_const_string_format() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "const_string", "value": "Hello!"}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, "Hello!"));
    assert!(!is_grammar_accept_string(&grammar, "Hello"));
    assert!(!is_grammar_accept_string(&grammar, "Hello!!"));
    assert!(!is_grammar_accept_string(&grammar, "HELLO!"));
}

#[test]
#[serial]
fn test_json_schema_format() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "json_schema", "json_schema": {"type": "object", "properties": {"a": {"type": "string"}}}}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, r#"{"a": "hello"}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{"a": 123}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{"b": "hello"}"#));
    assert!(!is_grammar_accept_string(&grammar, "invalid json"));
}

#[test]
#[serial]
fn test_ebnf_grammar_format() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "grammar", "grammar": "root ::= \"Hello!\" number\nnumber ::= [0-9] | [0-9] number"}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, "Hello!12345"));
    assert!(is_grammar_accept_string(&grammar, "Hello!0"));
    assert!(!is_grammar_accept_string(&grammar, "Hello!"));
    assert!(!is_grammar_accept_string(&grammar, "Hello!123a"));
    assert!(!is_grammar_accept_string(&grammar, "Hi!123"));
}

#[test]
#[serial]
fn test_regex_format() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "regex", "pattern": "Hello![0-9]+"}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, "Hello!12345"));
    assert!(is_grammar_accept_string(&grammar, "Hello!0"));
    assert!(!is_grammar_accept_string(&grammar, "Hello!"));
    assert!(!is_grammar_accept_string(&grammar, "Hello!123a"));
    assert!(!is_grammar_accept_string(&grammar, "Hi!123"));
}

#[test]
#[serial]
fn test_sequence_format() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "sequence", "elements": [{"type": "const_string", "value": "Hello!"}, {"type": "json_schema", "json_schema": {"type": "number"}}, {"type": "grammar", "grammar": "root ::= \"\" | [-+*/]"}, {"type": "regex", "pattern": "[simple]?"}]}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, "Hello!123"));
    assert!(!is_grammar_accept_string(&grammar, "Hello!Hello!"));
    assert!(!is_grammar_accept_string(&grammar, "Hello!"));
    assert!(is_grammar_accept_string(&grammar, "Hello!123+"));
    assert!(is_grammar_accept_string(&grammar, "Hello!123s"));
}

#[test]
#[serial]
fn test_or_format() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "or", "elements": [{"type": "const_string", "value": "Hello!"}, {"type": "regex", "pattern": "[0-9]+"}]}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, "Hello!"));
    assert!(is_grammar_accept_string(&grammar, "123"));
    assert!(!is_grammar_accept_string(&grammar, "Hello!123"));
}

#[test]
#[serial]
fn test_tag_format() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "tag", "begin": "<tool>", "content": {"type": "json_schema", "json_schema": {"type": "string"}}, "end": "</tool>"}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, r#"<tool>"test"</tool>"#));
    assert!(!is_grammar_accept_string(&grammar, r#"<tool>123</tool>"#));
    assert!(!is_grammar_accept_string(&grammar, r#"<tool>"test""#));
}

#[test]
#[serial]
fn test_any_text_format() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "any_text"}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, "Hello world"));
    assert!(is_grammar_accept_string(&grammar, "123"));
    assert!(is_grammar_accept_string(&grammar, ""));
}

#[test]
#[serial]
fn test_triggered_tag_format() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["<tool>"], "tags": [{"type": "tag", "begin": "<tool>", "content": {"type": "json_schema", "json_schema": {"type": "string"}}, "end": "</tool>"}]}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, r#"prefix<tool>"test"</tool>"#));
    assert!(is_grammar_accept_string(&grammar, r#"<tool>"test"</tool>suffix"#));
}

#[test]
#[serial]
fn test_triggered_tags_corner_case() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["<"], "tags": [{"type": "tag", "begin": "<tool>", "content": {"type": "const_string", "value": "test"}, "end": "</tool>"}]}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, "<tool>test</tool>"));
    assert!(is_grammar_accept_string(&grammar, "prefix<tool>test</tool>"));
}

#[test]
#[serial]
fn test_triggered_tag_with_outside_tag() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["<tool>"], "tags": [{"type": "tag", "begin": "<tool>", "content": {"type": "json_schema", "json_schema": {"type": "string"}}, "end": "</tool>"}], "outside_tag": {"type": "any_text"}}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(
        &grammar,
        r#"Some text<tool>"value"</tool>more text"#
    ));
}

#[test]
#[serial]
fn test_tags_with_separator_format() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [{"type": "tag", "begin": "<a>", "content": {"type": "const_string", "value": "1"}, "end": "</a>"}, {"type": "tag", "begin": "<b>", "content": {"type": "const_string", "value": "2"}, "end": "</b>"}], "separator": ","}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, "<a>1</a>,<b>2</b>"));
    assert!(is_grammar_accept_string(&grammar, "<b>2</b>,<a>1</a>"));
}

#[test]
#[serial]
fn test_compound_format() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "sequence", "elements": [{"type": "const_string", "value": "Start: "}, {"type": "or", "elements": [{"type": "const_string", "value": "A"}, {"type": "const_string", "value": "B"}]}, {"type": "const_string", "value": " End"}]}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, "Start: A End"));
    assert!(is_grammar_accept_string(&grammar, "Start: B End"));
    assert!(!is_grammar_accept_string(&grammar, "Start: C End"));
}

#[test]
#[serial]
fn test_end_string_detector() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "tag", "begin": "<start>", "content": {"type": "any_text"}, "end": "<end>"}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, "<start>content<end>"));
    assert!(is_grammar_accept_string(
        &grammar,
        "<start>lots of text content here<end>"
    ));
    assert!(!is_grammar_accept_string(&grammar, "<start>no end tag"));
}

#[test]
#[serial]
fn test_basic_structural_tag_utf8() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "const_string", "value": "こんにちは"}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, "こんにちは"));
    assert!(!is_grammar_accept_string(&grammar, "hello"));
}

#[test]
#[serial]
fn test_from_structural_tag_with_structural_tag_instance() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "json_schema", "json_schema": {"type": "string"}}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, r#""test""#));
    assert!(!is_grammar_accept_string(&grammar, "test"));
}

