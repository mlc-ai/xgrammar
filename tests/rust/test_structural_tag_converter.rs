mod test_utils;

use serial_test::serial;
use serde_json::json;
use test_utils::*;
use xgrammar::Grammar;

fn check_stag_with_grammar(
    structural_tag_format: &serde_json::Value,
    expected_grammar_ebnf: &str,
) {
    let structural_tag =
        json!({"type": "structural_tag", "format": structural_tag_format});
    let grammar =
        Grammar::from_structural_tag(&structural_tag.to_string()).unwrap();
    assert_eq!(grammar.to_string(), expected_grammar_ebnf);
}

fn check_stag_with_instance(
    structural_tag_format: &serde_json::Value,
    instance: &str,
    is_accepted: bool,
) {
    let structural_tag =
        json!({"type": "structural_tag", "format": structural_tag_format});
    let grammar =
        Grammar::from_structural_tag(&structural_tag.to_string()).unwrap();
    assert_eq!(is_grammar_accept_string(&grammar, instance), is_accepted);
}

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
fn test_qwen_parameter_xml_format() {
    let stag_format = json!({
        "type": "qwen_xml_parameter",
        "json_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"]
        }
    });
    let expected_grammar = r#"basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
xml_string ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
xml_string_0 ::= ((xml_string))
xml_any ::= ((basic_number) | (xml_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_prop_0 ::= (("0") | (root_prop_0_1 [1-9] [0-9]*))
root_part_0 ::= (([ \n\t]* "<parameter=name>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>"))
root_0 ::= (([ \n\t]* "<parameter=age>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" root_part_0))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
root_prop_0_1 ::= ("" | ("-"))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
root ::= ((root_0))
"#;

    check_stag_with_grammar(&stag_format, expected_grammar);

    let instances = [
        ("<parameter=age>\t100\n</parameter><parameter=name>Bob</parameter>", true),
        ("<parameter=age>\t100\n</parameter>\t\n<parameter=name>Bob</parameter>", true),
        ("<parameter=age>100</parameter><parameter=name>Bob</parameter>", true),
        ("\n\t<parameter=age>100</parameter><parameter=name>Bob</parameter>", true),
        ("<parameter=age>100</parameter><parameter=name>\"Bob&lt;\"</parameter>", true),
        (
            "<parameter=age>100</parameter><parameter=name><!DOCTYPE html>\n<html lang=\"en\">\n  <body><h1>Hello</h1></body>\n</html></parameter>",
            true,
        ),
    ];
    for (instance, is_accepted) in instances {
        check_stag_with_instance(&stag_format, instance, is_accepted);
    }
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
fn test_any_text_only_format() {
    let stag_format = json!({"type": "any_text"});
    let expected_grammar = r#"any_text ::= (([\0-\U0010ffff]*))
root ::= ((any_text))
"#;
    check_stag_with_grammar(&stag_format, expected_grammar);

    let instances = [("ABCDEF", true), ("123456", true), ("", true)];
    for (instance, is_accepted) in instances {
        check_stag_with_instance(&stag_format, instance, is_accepted);
    }
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
fn test_tags_with_separator_format_with_outside_tag() {
    let stag_grammars = vec![
        (
            0,
            json!({
                "type": "tag",
                "begin": "begin",
                "content": {
                    "type": "tags_with_separator",
                    "tags": [
                        {"begin": "A1", "content": {"type": "const_string", "value": "L1"}, "end": "A"},
                        {"begin": "A2", "content": {"type": "const_string", "value": "L2"}, "end": "A"}
                    ],
                    "separator": "AA",
                    "at_least_one": false,
                    "stop_after_first": false
                },
                "end": "end"
            }),
            r#"const_string ::= (("L1"))
tag ::= (("A1" const_string "A"))
const_string_1 ::= (("L2"))
tag_1 ::= (("A2" const_string_1 "A"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= (("AA" tags_with_separator_tags tags_with_separator_sub) | ("end"))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub) | ("end"))
tag_2 ::= (("begin" tags_with_separator))
root ::= ((tag_2))
"#,
        ),
        (
            1,
            json!({
                "type": "tag",
                "begin": "begin",
                "content": {
                    "type": "tags_with_separator",
                    "tags": [
                        {"begin": "A1", "content": {"type": "const_string", "value": "L1"}, "end": "A"},
                        {"begin": "A2", "content": {"type": "const_string", "value": "L2"}, "end": "A"}
                    ],
                    "separator": "AA",
                    "at_least_one": true,
                    "stop_after_first": false
                },
                "end": "end"
            }),
            r#"const_string ::= (("L1"))
tag ::= (("A1" const_string "A"))
const_string_1 ::= (("L2"))
tag_1 ::= (("A2" const_string_1 "A"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= (("AA" tags_with_separator_tags tags_with_separator_sub) | ("end"))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
tag_2 ::= (("begin" tags_with_separator))
root ::= ((tag_2))
"#,
        ),
        (
            2,
            json!({
                "type": "tag",
                "begin": "begin",
                "content": {
                    "type": "tags_with_separator",
                    "tags": [
                        {"begin": "A1", "content": {"type": "const_string", "value": "L1"}, "end": "A"},
                        {"begin": "A2", "content": {"type": "const_string", "value": "L2"}, "end": "A"}
                    ],
                    "separator": "AA",
                    "at_least_one": false,
                    "stop_after_first": true
                },
                "end": "end"
            }),
            r#"const_string ::= (("L1"))
tag ::= (("A1" const_string "A"))
const_string_1 ::= (("L2"))
tag_1 ::= (("A2" const_string_1 "A"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator ::= ((tags_with_separator_tags "end") | ("end"))
tag_2 ::= (("begin" tags_with_separator))
root ::= ((tag_2))
"#,
        ),
        (
            3,
            json!({
                "type": "tag",
                "begin": "begin",
                "content": {
                    "type": "tags_with_separator",
                    "tags": [
                        {"begin": "A1", "content": {"type": "const_string", "value": "L1"}, "end": "A"},
                        {"begin": "A2", "content": {"type": "const_string", "value": "L2"}, "end": "A"}
                    ],
                    "separator": "AA",
                    "at_least_one": true,
                    "stop_after_first": true
                },
                "end": "end"
            }),
            r#"const_string ::= (("L1"))
tag ::= (("A1" const_string "A"))
const_string_1 ::= (("L2"))
tag_1 ::= (("A2" const_string_1 "A"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator ::= ((tags_with_separator_tags "end"))
tag_2 ::= (("begin" tags_with_separator))
root ::= ((tag_2))
"#,
        ),
    ];

    let instances = [
        ("beginend", vec![true, false, true, false]),
        ("beginA1L1Aend", vec![true, true, true, true]),
        ("beginA1L1AAAA2L2Aend", vec![true, true, false, false]),
        ("beginA1L1A", vec![false, false, false, false]),
        ("beginA1L1AA2L2Aend", vec![false, false, false, false]),
    ];

    for (stag_id, stag_format, expected_grammar) in &stag_grammars {
        check_stag_with_grammar(stag_format, expected_grammar);
        for (instance, accepted_results) in &instances {
            check_stag_with_instance(
                stag_format,
                instance,
                accepted_results[*stag_id],
            );
        }
    }
}

#[test]
#[serial]
fn test_tags_with_empty_separator_format() {
    let stag_grammars = vec![
        (
            0,
            json!({
                "type": "tags_with_separator",
                "tags": [
                    {"begin": "<a>", "content": {"type": "const_string", "value": "X"}, "end": "</a>"},
                    {"begin": "<b>", "content": {"type": "const_string", "value": "Y"}, "end": "</b>"}
                ],
                "separator": "",
                "at_least_one": false,
                "stop_after_first": false
            }),
            r#"const_string ::= (("X"))
tag ::= (("<a>" const_string "</a>"))
const_string_1 ::= (("Y"))
tag_1 ::= (("<b>" const_string_1 "</b>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
"#,
        ),
        (
            1,
            json!({
                "type": "tags_with_separator",
                "tags": [
                    {"begin": "<a>", "content": {"type": "const_string", "value": "X"}, "end": "</a>"},
                    {"begin": "<b>", "content": {"type": "const_string", "value": "Y"}, "end": "</b>"}
                ],
                "separator": "",
                "at_least_one": true,
                "stop_after_first": false
            }),
            r#"const_string ::= (("X"))
tag ::= (("<a>" const_string "</a>"))
const_string_1 ::= (("Y"))
tag_1 ::= (("<b>" const_string_1 "</b>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
"#,
        ),
        (
            2,
            json!({
                "type": "tags_with_separator",
                "tags": [
                    {"begin": "<a>", "content": {"type": "const_string", "value": "X"}, "end": "</a>"},
                    {"begin": "<b>", "content": {"type": "const_string", "value": "Y"}, "end": "</b>"}
                ],
                "separator": "",
                "at_least_one": false,
                "stop_after_first": true
            }),
            r#"const_string ::= (("X"))
tag ::= (("<a>" const_string "</a>"))
const_string_1 ::= (("Y"))
tag_1 ::= (("<b>" const_string_1 "</b>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator ::= ("" | (tags_with_separator_tags))
root ::= ((tags_with_separator))
"#,
        ),
        (
            3,
            json!({
                "type": "tags_with_separator",
                "tags": [
                    {"begin": "<a>", "content": {"type": "const_string", "value": "X"}, "end": "</a>"},
                    {"begin": "<b>", "content": {"type": "const_string", "value": "Y"}, "end": "</b>"}
                ],
                "separator": "",
                "at_least_one": true,
                "stop_after_first": true
            }),
            r#"const_string ::= (("X"))
tag ::= (("<a>" const_string "</a>"))
const_string_1 ::= (("Y"))
tag_1 ::= (("<b>" const_string_1 "</b>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator ::= ((tags_with_separator_tags))
root ::= ((tags_with_separator))
"#,
        ),
    ];

    let instances = [
        ("", vec![true, false, true, false]),
        ("<a>X</a>", vec![true, true, true, true]),
        ("<a>X</a><b>Y</b>", vec![true, true, false, false]),
        ("<b>Y</b><a>X</a><b>Y</b>", vec![true, true, false, false]),
        ("<a>X</a><a>X</a><a>X</a>", vec![true, true, false, false]),
        ("<a>X</a>,<b>Y</b>", vec![false, false, false, false]),
        ("<c>Z</c>", vec![false, false, false, false]),
    ];

    for (stag_id, stag_format, expected_grammar) in &stag_grammars {
        check_stag_with_grammar(stag_format, expected_grammar);
        for (instance, accepted_results) in &instances {
            check_stag_with_instance(
                stag_format,
                instance,
                accepted_results[*stag_id],
            );
        }
    }
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
fn test_structural_tag_json_format_errors() {
    // Test JSON format and parsing errors that occur during JSON parsing phase
    let cases = [
        // JSON Parsing Errors
        (
            r#"{"type": "structural_tag", "format": {"type": "const_string", "value": "hello""#,
            "Failed to parse JSON",
        ),
        (r#""not_an_object""#, "Structural tag must be an object"),
        (
            r#"{"type": "wrong_type", "format": {"type": "const_string", "value": "hello"}}"#,
            r#"Structural tag's type must be a string "structural_tag""#,
        ),
        (
            r#"{"type": "structural_tag"}"#,
            "Structural tag must have a format field",
        ),
        // Format Parsing Errors
        (
            r#"{"type": "structural_tag", "format": "not_an_object"}"#,
            "Format must be an object",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": 123, "value": "hello"}}"#,
            "Format's type must be a string",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "unknown_format"}}"#,
            "Format type not recognized: unknown_format",
        ),
        (
            r#"{"type": "structural_tag", "format": {"invalid_field": "value"}}"#,
            "Invalid format",
        ),
        // ConstStringFormat Errors
        (
            r#"{"type": "structural_tag", "format": {"type": "const_string"}}"#,
            "ConstString format must have a value field with a non-empty string",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "const_string", "value": 123}}"#,
            "ConstString format must have a value field with a non-empty string",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "const_string", "value": ""}}"#,
            "ConstString format must have a value field with a non-empty string",
        ),
        // JSONSchemaFormat Errors
        (
            r#"{"type": "structural_tag", "format": {"type": "json_schema"}}"#,
            "JSON schema format must have a json_schema field with a object or boolean value",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "json_schema", "json_schema": "invalid"}}"#,
            "JSON schema format must have a json_schema field with a object or boolean value",
        ),
        // SequenceFormat Errors
        (
            r#"{"type": "structural_tag", "format": {"type": "sequence"}}"#,
            "Sequence format must have an elements field with an array",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "sequence", "elements": "not_array"}}"#,
            "Sequence format must have an elements field with an array",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "sequence", "elements": []}}"#,
            "Sequence format must have at least one element",
        ),
        // OrFormat Errors
        (
            r#"{"type": "structural_tag", "format": {"type": "or"}}"#,
            "Or format must have an elements field with an array",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "or", "elements": "not_array"}}"#,
            "Or format must have an elements field with an array",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "or", "elements": []}}"#,
            "Or format must have at least one element",
        ),
        // TagFormat Errors
        (
            r#"{"type": "structural_tag", "format": {"type": "tag", "content": {"type": "const_string", "value": "hello"}, "end": "end"}}"#,
            "Tag format's begin field must be a string",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "tag", "begin": 123, "content": {"type": "const_string", "value": "hello"}, "end": "end"}}"#,
            "Tag format's begin field must be a string",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "tag", "begin": "start", "end": "end"}}"#,
            "Tag format must have a content field",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "tag", "begin": "start", "content": {"type": "const_string", "value": "hello"}}}"#,
            "Tag format must have an end field",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "tag", "begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": 123}}"#,
            "Tag format's end field must be a string or array of strings",
        ),
        // TriggeredTagsFormat Errors
        (
            r#"{"type": "structural_tag", "format": {"type": "triggered_tags", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}]}}"#,
            "Triggered tags format must have a triggers field with an array",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": "not_array", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}]}}"#,
            "Triggered tags format must have a triggers field with an array",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": [], "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}]}}"#,
            "Triggered tags format's triggers must be non-empty",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": [123], "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}]}}"#,
            "Triggered tags format's triggers must be non-empty strings",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": [""], "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}]}}"#,
            "Triggered tags format's triggers must be non-empty strings",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["trigger"]}}"#,
            "Triggered tags format must have a tags field with an array",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["trigger"], "tags": "not_array"}}"#,
            "Triggered tags format must have a tags field with an array",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["trigger"], "tags": []}}"#,
            "Triggered tags format's tags must be non-empty",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["trigger"], "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}], "at_least_one": "not_boolean"}}"#,
            "at_least_one must be a boolean",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["trigger"], "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}], "stop_after_first": "not_boolean"}}"#,
            "stop_after_first must be a boolean",
        ),
        // TagsWithSeparatorFormat Errors
        (
            r#"{"type": "structural_tag", "format": {"type": "tags_with_separator", "separator": "sep"}}"#,
            "Tags with separator format must have a tags field with an array",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": "not_array", "separator": "sep"}}"#,
            "Tags with separator format must have a tags field with an array",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [], "separator": "sep"}}"#,
            "Tags with separator format's tags must be non-empty",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}]}}"#,
            "Tags with separator format's separator field must be a string",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}], "separator": 123}}"#,
            "Tags with separator format's separator field must be a string",
        ),
        // Note: empty separator is now valid, so no error test for it
        (
            r#"{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}], "separator": "sep", "at_least_one": "not_boolean"}}"#,
            "at_least_one must be a boolean",
        ),
        (
            r#"{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}], "separator": "sep", "stop_after_first": "not_boolean"}}"#,
            "stop_after_first must be a boolean",
        ),
    ];

    for (json_input, expected_error) in cases {
        match Grammar::from_structural_tag(json_input) {
            Ok(_) => panic!("expected error for '{json_input}'"),
            Err(err) => assert!(
                err.contains(expected_error),
                "expected '{expected_error}' in '{err}'"
            ),
        }
    }
}

#[test]
#[serial]
fn test_structural_tag_error() {
    // Test analyzer and converter errors that occur after successful parsing
    let cases = vec![
        // Analyzer Errors - Only last element in sequence can be unlimited
        json!({
            "type": "sequence",
            "elements": [
                {"type": "const_string", "value": "start"},
                {"type": "any_text"},
                {"type": "const_string", "value": "end"}
            ]
        }),
        // Analyzer Errors - Or format with mixed unlimited and limited elements
        json!({
            "type": "or",
            "elements": [
                {"type": "const_string", "value": "limited"},
                {"type": "any_text"}
            ]
        }),
        // Analyzer Errors - Tag format with unlimited content but empty end
        json!({
            "type": "tag",
            "begin": "start",
            "content": {"type": "any_text"},
            "end": ""
        }),
        // Converter Errors - Tag matches multiple triggers
        json!({
            "type": "triggered_tags",
            "triggers": ["A", "AB"],
            "tags": [
                {"begin": "ABC", "content": {"type": "const_string", "value": "hello"}, "end": "end"}
            ]
        }),
        // Converter Errors - Tag matches no trigger
        json!({
            "type": "triggered_tags",
            "triggers": ["X", "Y"],
            "tags": [
                {"begin": "ABC", "content": {"type": "const_string", "value": "hello"}, "end": "end"}
            ]
        }),
        // Cannot detect end string of tags_with_separator in sequence
        json!({
            "type": "sequence",
            "elements": [
                {
                    "type": "tags_with_separator",
                    "tags": [
                        {"begin": "<start>", "content": {"type": "const_string", "value": "[TEXT]"}, "end": "<end>"}
                    ],
                    "separator": "<sep>"
                },
                {"type": "const_string", "value": "[TEXT]"}
            ]
        }),
        // Cannot detect end string of tags_with_separator in or
        json!({
            "type": "or",
            "elements": [
                {
                    "type": "tags_with_separator",
                    "tags": [
                        {"begin": "<start>", "content": {"type": "const_string", "value": "[TEXT]"}, "end": "<end>"}
                    ],
                    "separator": "<sep>"
                },
                {"type": "const_string", "value": "[TEXT]"}
            ]
        }),
        // Original test cases - Detected end string of tags_with_separator is empty
        json!({
            "type": "tag",
            "begin": "<start>",
            "content": {
                "type": "tags_with_separator",
                "tags": [
                    {"begin": "<start2>", "content": {"type": "const_string", "value": "[TEXT]"}, "end": "<end2>"}
                ],
                "separator": "<sep>"
            },
            "end": ""
        }),
    ];

    for stag_format in cases {
        let structural_tag =
            json!({"type": "structural_tag", "format": stag_format});
        match Grammar::from_structural_tag(&structural_tag.to_string()) {
            Ok(_) => panic!("expected error for structural tag"),
            Err(err) => assert!(
                err.contains("Invalid structural tag error"),
                "unexpected error: {err}"
            ),
        }
    }
}

#[test]
#[serial]
fn test_basic_structural_tag_utf8() {
    // Test structural tag with UTF-8 characters
    let cases = [
        (json!({"type": "const_string", "value": "你好"}), "你好", true),
        (json!({"type": "const_string", "value": "你好"}), "hello", false),
        (json!({"type": "any_text"}), "😊", true),
        (
            json!({
                "type": "sequence",
                "elements": [
                    {"type": "const_string", "value": "开始"},
                    {"type": "json_schema", "json_schema": {"type": "string"}},
                    {"type": "const_string", "value": "结束"}
                ]
            }),
            "开始\"中间\"结束",
            true,
        ),
        (
            json!({
                "type": "sequence",
                "elements": [
                    {"type": "const_string", "value": "开始"},
                    {"type": "json_schema", "json_schema": {"type": "string"}},
                    {"type": "const_string", "value": "结束"}
                ]
            }),
            "开始中间内容",
            false,
        ),
        (
            json!({
                "type": "tag",
                "begin": "标签开始",
                "content": {"type": "any_text"},
                "end": "标签结束"
            }),
            "标签开始一些内容标签结束",
            true,
        ),
        (
            json!({
                "type": "tag",
                "begin": "标签开始",
                "content": {"type": "any_text"},
                "end": "标签结束"
            }),
            "标签开始一些内容",
            false,
        ),
        (
            json!({
                "type": "or",
                "elements": [
                    {"type": "const_string", "value": "选项一"},
                    {"type": "const_string", "value": "选项二"}
                ]
            }),
            "选项一",
            true,
        ),
        (
            json!({
                "type": "or",
                "elements": [
                    {"type": "const_string", "value": "选项一"},
                    {"type": "const_string", "value": "选项二"}
                ]
            }),
            "选项三",
            false,
        ),
        (
            json!({
                "type": "tags_with_separator",
                "tags": [
                    {"begin": "项开始", "content": {"type": "any_text"}, "end": "项结束"}
                ],
                "separator": "分隔符"
            }),
            "项开始内容1项结束分隔符项开始内容2项结束",
            true,
        ),
        (
            json!({
                "type": "tags_with_separator",
                "tags": [
                    {"begin": "项开始", "content": {"type": "any_text"}, "end": "项结束"}
                ],
                "separator": "分隔符"
            }),
            "项开始内容1项结束项开始内容2项结束",
            false,
        ),
        (
            json!({
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {"字段": {"type": "string"}},
                    "required": ["字段"],
                    "additionalProperties": false
                }
            }),
            r#"{"字段": "值"}"#,
            true,
        ),
        (
            json!({
                "type": "qwen_xml_parameter",
                "json_schema": {
                    "type": "object",
                    "properties": {"参数": {"type": "string"}},
                    "required": ["参数"],
                    "additionalProperties": false
                }
            }),
            "<parameter=参数>值</parameter>",
            true,
        ),
    ];

    for (stag_format, instance, is_accepted) in cases {
        check_stag_with_instance(&stag_format, instance, is_accepted);
    }
}

#[test]
#[serial]
fn test_from_structural_tag_with_structural_tag_instance() {
    let structural_tag = r##"{"type": "structural_tag", "format": {"type": "json_schema", "json_schema": {"type": "string"}}}"##;
    let grammar = Grammar::from_structural_tag(structural_tag).unwrap();
    
    assert!(is_grammar_accept_string(&grammar, r#""test""#));
    assert!(!is_grammar_accept_string(&grammar, "test"));
}

#[test]
#[serial]
fn test_multiple_end_tokens_tag_grammar() {
    let cases = [
        (
            json!({
                "type": "tag",
                "begin": "BEG",
                "content": {"type": "const_string", "value": "CONTENT"},
                "end": ["END1", "END2"]
            }),
            r#"const_string ::= (("CONTENT"))
tag_end ::= (("END1") | ("END2"))
tag ::= (("BEG" const_string tag_end))
root ::= ((tag))
"#,
        ),
        (
            json!({
                "type": "tag",
                "begin": "<start>",
                "content": {"type": "const_string", "value": "X"},
                "end": ["</end>"]
            }),
            r#"const_string ::= (("X"))
tag ::= (("<start>" const_string "</end>"))
root ::= ((tag))
"#,
        ),
    ];

    for (stag_format, expected_grammar) in cases {
        check_stag_with_grammar(&stag_format, expected_grammar);
    }
}

#[test]
#[serial]
fn test_multiple_end_tokens_tag_instance() {
    let stag_format = json!({
        "type": "tag",
        "begin": "BEG",
        "content": {"type": "const_string", "value": "CONTENT"},
        "end": ["END1", "END2"]
    });
    let instances = [
        ("BEGCONTENTEND1", true),
        ("BEGCONTENTEND2", true),
        ("BEGCONTENTEND3", false),
        ("BEGCONTENTEND", false),
    ];
    for (instance, is_accepted) in instances {
        check_stag_with_instance(&stag_format, instance, is_accepted);
    }
}

#[test]
#[serial]
fn test_multiple_end_tokens_any_text_grammar() {
    let cases = [(
        json!({
            "type": "tag",
            "begin": "BEG",
            "content": {"type": "any_text"},
            "end": ["END1", "END2"]
        }),
        r#"any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("END1", "END2"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("BEG" any_text))
root ::= ((tag))
"#,
    )];
    for (stag_format, expected_grammar) in cases {
        check_stag_with_grammar(&stag_format, expected_grammar);
    }
}

#[test]
#[serial]
fn test_multiple_end_tokens_any_text_instance() {
    let stag_format = json!({
        "type": "tag",
        "begin": "BEG",
        "content": {"type": "any_text"},
        "end": ["END1", "END2"]
    });
    let instances = [
        ("BEGHello!END1", true),
        ("BEGHello!END2", true),
        ("BEGEND1", true),
        ("BEGEND2", true),
        ("BEGsome text hereEND1", true),
        ("BEGsome text hereEND2", true),
        ("BEGHello!END3", false),
        ("BEGHello!END", false),
    ];
    for (instance, is_accepted) in instances {
        check_stag_with_instance(&stag_format, instance, is_accepted);
    }
}

#[test]
#[serial]
fn test_multiple_end_tokens_with_empty_grammar() {
    let cases = [
        (
            json!({
                "type": "tag",
                "begin": "BEG",
                "content": {"type": "const_string", "value": "CONTENT"},
                "end": ["END1", ""]
            }),
            r#"const_string ::= (("CONTENT"))
tag_end ::= ("" | ("END1"))
tag ::= (("BEG" const_string tag_end))
root ::= ((tag))
"#,
        ),
        (
            json!({
                "type": "tag",
                "begin": "<start>",
                "content": {"type": "const_string", "value": "X"},
                "end": ["", "</end>"]
            }),
            r#"const_string ::= (("X"))
tag_end ::= ("" | ("</end>"))
tag ::= (("<start>" const_string tag_end))
root ::= ((tag))
"#,
        ),
    ];
    for (stag_format, expected_grammar) in cases {
        check_stag_with_grammar(&stag_format, expected_grammar);
    }
}

#[test]
#[serial]
fn test_multiple_end_tokens_with_empty_instance() {
    let stag_format = json!({
        "type": "tag",
        "begin": "BEG",
        "content": {"type": "const_string", "value": "CONTENT"},
        "end": ["END1", ""]
    });
    let instances = [
        ("BEGCONTENTEND1", true),
        ("BEGCONTENT", true),
        ("BEGCONTENTEND2", false),
        ("BEGCONTENTEND", false),
    ];
    for (instance, is_accepted) in instances {
        check_stag_with_instance(&stag_format, instance, is_accepted);
    }
}

#[test]
#[serial]
fn test_multiple_end_tokens_python_api() {
    // Test that TagFormat accepts both str and List[str] for end field
    let tag1 = json!({
        "type": "tag",
        "begin": "<start>",
        "content": {"type": "const_string", "value": "content"},
        "end": "</end>"
    });
    let tag2 = json!({
        "type": "tag",
        "begin": "<start>",
        "content": {"type": "const_string", "value": "content"},
        "end": ["</end1>", "</end2>"]
    });
    let structural_tag1 = json!({"type": "structural_tag", "format": tag1});
    let structural_tag2 = json!({"type": "structural_tag", "format": tag2});
    let grammar1 =
        Grammar::from_structural_tag(&structural_tag1.to_string()).unwrap();
    let grammar2 =
        Grammar::from_structural_tag(&structural_tag2.to_string()).unwrap();
    let _ = (grammar1, grammar2);
}

#[test]
#[serial]
fn test_multiple_end_tokens_empty_array_error() {
    // Test that empty end array raises an error
    let stag_format = json!({
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": "BEG",
            "content": {"type": "const_string", "value": "X"},
            "end": []
        }
    });
    match Grammar::from_structural_tag(&stag_format.to_string()) {
        Ok(_) => panic!("expected error for empty end array"),
        Err(err) => assert!(err.to_lowercase().contains("empty"), "{err}"),
    }
}

#[test]
#[serial]
fn test_multiple_end_tokens_unlimited_empty_error() {
    // Test that unlimited content with all empty end strings raises an error
    let stag_format = json!({
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": "BEG",
            "content": {"type": "any_text"},
            "end": ["", ""]
        }
    });
    match Grammar::from_structural_tag(&stag_format.to_string()) {
        Ok(_) => panic!("expected error for empty end strings"),
        Err(err) => {
            let err_lower = err.to_lowercase();
            assert!(
                err_lower.contains("non-empty") || err_lower.contains("empty"),
                "{err}"
            );
        }
    }
}

#[test]
#[serial]
fn test_excluded_strings_in_any_text() {
    let stag_format = json!({
        "type": "tag",
        "content": {"type": "any_text", "excludes": ["<end>", "</tag>"]},
        "begin": "",
        "end": "."
    });
    let expected_grammar = r#"any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("."),
  loop_after_dispatch=false,
  excludes=("<end>", "</tag>")
)
tag ::= (("" any_text))
root ::= ((tag))
"#;
    check_stag_with_grammar(&stag_format, expected_grammar);

    let instances = [
        ("This is a test string.", true),
        ("This string contains <end> which is excluded.", false),
        ("Another string with </tag> inside.", false),
        ("A clean string without excluded substrings.", true),
        ("<end> at the beginning.", false),
        ("At the end </tag>.", false),
    ];
    for (instance, is_accepted) in instances {
        check_stag_with_instance(&stag_format, instance, is_accepted);
    }
}

#[test]
#[serial]
fn test_excluded_strings_in_triggered_format() {
    let stag_format = json!({
        "type": "triggered_tags",
        "triggers": ["A"],
        "tags": [
            {"begin": "A1", "content": {"type": "const_string", "value": "L1"}, "end": "A"},
            {"begin": "A2", "content": {"type": "const_string", "value": "L2"}, "end": "A"}
        ],
        "at_least_one": true,
        "stop_after_first": false,
        "excludes": ["L1", "L2"]
    });
    let expected_grammar = r#"const_string ::= (("L1"))
const_string_1 ::= (("L2"))
triggered_tags_group ::= (("1" const_string "A") | ("2" const_string_1 "A"))
triggered_tags_first ::= (("A1" const_string "A") | ("A2" const_string_1 "A"))
triggered_tags_sub ::= TagDispatch(
  ("A", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("L1", "L2")
)
triggered_tags ::= ((triggered_tags_first triggered_tags_sub))
root ::= ((triggered_tags))
"#;
    check_stag_with_grammar(&stag_format, expected_grammar);

    let instances = [
        ("A", false),
        ("A1", false),
        ("A1L1AB", true),
        ("A1L2A", false),
        ("L1A1L1A", false),
        ("L2A2L2A", false),
        ("A1L1AL1", false),
        ("A1L1AA2L2A", true),
    ];
    for (instance, is_accepted) in instances {
        check_stag_with_instance(&stag_format, instance, is_accepted);
    }
}

#[test]
#[serial]
fn test_excluded_strings_in_single_any_text() {
    let format = json!({"type": "any_text", "excludes": ["ABC"]});
    let expected_grammar = r#"any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("ABC")
)
root ::= ((any_text))
"#;
    check_stag_with_grammar(&format, expected_grammar);

    let instances = [
        ("XYZ", true),
        ("Hello World", true),
        ("ABC", false),
        ("123ABC456", false),
        ("A quick brown fox", true),
        ("", true),
    ];
    for (instance, is_accepted) in instances {
        check_stag_with_instance(&format, instance, is_accepted);
    }
}

#[test]
#[serial]
fn test_excluded_any_text_within_sequence() {
    let format = json!({
        "type": "sequence",
        "elements": [
            {"type": "any_text", "excludes": ["ABC"]},
            {"type": "const_string", "value": "ABC"}
        ]
    });
    let expected_grammar = r#"any_text ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("ABC")
)
const_string ::= (("ABC"))
sequence ::= ((any_text const_string))
root ::= ((sequence))
"#;
    check_stag_with_grammar(&format, expected_grammar);

    let instances = [
        ("HelloABC", true),
        ("WorldABC", true),
        ("NoExclusionHere", false),
        ("JustSomeText", false),
        ("ABC", true),
        ("SomeTextBeforeABC", true),
    ];
    for (instance, is_accepted) in instances {
        check_stag_with_instance(&format, instance, is_accepted);
    }
}

#[test]
#[serial]
fn test_excludes_triggered_tags_without_end() {
    let stag = json!({
        "type": "sequence",
        "elements": [
            {
                "type": "triggered_tags",
                "triggers": ["1"],
                "tags": [{"begin": "1", "content": {"type": "any_text"}, "end": ["1"]}],
                "excludes": ["ABC"]
            },
            {"type": "const_string", "value": "ABC"}
        ]
    });
    let expected_grammar = r#"any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("1"),
  loop_after_dispatch=false,
  excludes=()
)
triggered_tags_group ::= (("" any_text))
triggered_tags ::= TagDispatch(
  ("1", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("ABC")
)
const_string ::= (("ABC"))
sequence ::= ((triggered_tags const_string))
root ::= ((sequence))
"#;
    check_stag_with_grammar(&stag, expected_grammar);

    let instances = [
        ("1ABC", false),
        ("11ABC", true),
        ("1HelloWorld", false),
        ("1ABC123", false),
        ("2ABC", true),
    ];
    for (instance, is_accepted) in instances {
        check_stag_with_instance(&stag, instance, is_accepted);
    }
}

