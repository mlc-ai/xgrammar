mod common;

use serde_json::json;
use xgrammar::structural_tag::{
    Format, JsonSchemaStyle, StructuralTag, StructuralTagItem, TagFormat,
};
use xgrammar::{Error, Grammar, GrammarCompiler, GrammarMatcher};

/// Serialized by Python: `StructuralTag(format=TriggeredTagsFormat(...))`.
const PYDANTIC_TRIGGERED: &str = r#"{"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<function="],"tags":[{"type":"tag","begin":"<function=get_weather>","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},"style":"json","any_order":false,"max_whitespace_cnt":null},"end":"</function>"}],"at_least_one":false,"stop_after_first":true,"excludes":[]}}"#;

/// Serialized by Python: a sequence with const strings, any_text, or.
const PYDANTIC_SEQUENCE: &str = r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"const_string","value":"<think>"},{"type":"any_text","excludes":["</think>"]},{"type":"const_string","value":"</think>"},{"type":"or","elements":[{"type":"regex","pattern":"[0-9]+"},{"type":"grammar","grammar":"root ::= \"a\""}]}]}}"#;

/// Serialized by Python: `DispatchFormat(rules=[("<f>", ConstStringFormat(value="x"))])`.
const PYDANTIC_DISPATCH: &str = r#"{"type":"structural_tag","format":{"type":"dispatch","rules":[["<f>",{"type":"const_string","value":"x"}]],"loop":true,"excludes":[]}}"#;

#[test]
fn wire_format_matches_pydantic_byte_for_byte() {
    for wire in [PYDANTIC_TRIGGERED, PYDANTIC_SEQUENCE, PYDANTIC_DISPATCH] {
        let tag = StructuralTag::from_json(wire).unwrap();
        assert_eq!(tag.to_json().unwrap(), wire);
    }
}

#[test]
fn natively_built_tag_serializes_like_pydantic() {
    let tag = StructuralTag::new(Format::TriggeredTags {
        triggers: vec!["<function=".to_string()],
        tags: vec![TagFormat::new(
            "<function=get_weather>",
            Format::JsonSchema {
                json_schema: json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }),
                style: JsonSchemaStyle::Json,
                any_order: false,
                max_whitespace_cnt: None,
            },
            "</function>",
        )],
        at_least_one: false,
        stop_after_first: true,
        excludes: vec![],
    });
    assert_eq!(tag.to_json().unwrap(), PYDANTIC_TRIGGERED);
}

#[test]
fn compile_and_match_structural_tag() {
    common::init();
    let tag = StructuralTag::from_json(PYDANTIC_TRIGGERED).unwrap();

    // Both entry points must accept the tag.
    Grammar::from_structural_tag(&tag).unwrap();
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    let compiled = compiler.compile_structural_tag(&tag).unwrap();

    let mut matcher = GrammarMatcher::new(&compiled).unwrap();
    assert!(matcher
        .accept_string(r#"hello <function=get_weather>{"city": "SF"}</function>"#)
        .unwrap());
    assert!(matcher.is_completed().unwrap());

    // A tag with the wrong function name is rejected.
    let mut matcher = GrammarMatcher::new(&compiled).unwrap();
    assert!(!matcher
        .accept_string(r#"<function=bad_name>{"city": "SF"}</function>"#)
        .unwrap());
}

#[test]
fn sequence_tag_end_to_end() {
    common::init();
    let tag = StructuralTag::from_json(PYDANTIC_SEQUENCE).unwrap();
    let ti = common::tiny_tokenizer_info();
    let compiler = GrammarCompiler::new(&ti).unwrap();
    let compiled = compiler.compile_structural_tag(&tag).unwrap();

    let mut matcher = GrammarMatcher::new(&compiled).unwrap();
    assert!(matcher
        .accept_string("<think>some thoughts</think>42")
        .unwrap());
    assert!(matcher.is_completed().unwrap());

    // The excluded string must not appear inside the think block.
    let mut matcher = GrammarMatcher::new(&compiled).unwrap();
    assert!(!matcher
        .accept_string("<think>a</think>b</think>42")
        .unwrap());
}

#[test]
fn invalid_structural_tag_error_kind() {
    common::init();
    // The trigger is not a prefix of the tag begin: rejected by C++ with the
    // InvalidStructuralTagError kind, surfaced as the typed variant.
    let tag = StructuralTag::new(Format::TriggeredTags {
        triggers: vec!["<tool>".to_string()],
        tags: vec![TagFormat::new(
            "<function=f>",
            Format::JsonSchema {
                json_schema: json!(true),
                style: JsonSchemaStyle::Json,
                any_order: false,
                max_whitespace_cnt: None,
            },
            "</function>",
        )],
        at_least_one: false,
        stop_after_first: false,
        excludes: vec![],
    });
    let err = Grammar::from_structural_tag(&tag).unwrap_err();
    assert!(
        matches!(err, Error::InvalidStructuralTag(_)),
        "unexpected error: {err:?}"
    );
}

#[test]
fn legacy_conversion() {
    let items = vec![StructuralTagItem {
        begin: "<function=get_weather>".to_string(),
        schema: json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }),
        end: "</function>".to_string(),
    }];
    let tag = StructuralTag::from_legacy(&items, &["<function=".to_string()]);
    match &tag.format {
        Format::TriggeredTags {
            triggers,
            tags,
            stop_after_first,
            ..
        } => {
            assert_eq!(triggers, &["<function=".to_string()]);
            assert_eq!(tags.len(), 1);
            assert!(!stop_after_first);
        }
        other => panic!("unexpected format: {other:?}"),
    }
    // And the converted tag compiles.
    common::init();
    Grammar::from_structural_tag(&tag).unwrap();
}

#[test]
fn token_formats_roundtrip() {
    use xgrammar::structural_tag::TokenRef;
    let tag = StructuralTag::new(Format::Sequence {
        elements: vec![
            Format::Token {
                token: TokenRef::Id(3),
            },
            Format::AnyTokens {
                exclude_tokens: vec![TokenRef::Text("</s>".to_string())],
            },
            Format::ExcludeToken {
                exclude_tokens: vec![TokenRef::Id(0)],
            },
            Format::Repeat {
                min: 1,
                max: -1,
                content: Box::new(Format::ConstString {
                    value: "x".to_string(),
                }),
            },
            Format::Optional {
                content: Box::new(Format::Star {
                    content: Box::new(Format::Plus {
                        content: Box::new(Format::ConstString {
                            value: "y".to_string(),
                        }),
                    }),
                }),
            },
        ],
    });
    let json = tag.to_json().unwrap();
    assert_eq!(StructuralTag::from_json(&json).unwrap(), tag);
    assert!(json.contains("\"type\":\"token\""));
    assert!(json.contains("\"exclude_tokens\":[\"</s>\"]"));
}
