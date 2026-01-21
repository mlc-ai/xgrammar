use serial_test::serial;
use xgrammar::{
    Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo, VocabType,
    testing::qwen_xml_tool_calling_to_ebnf,
};

fn matcher_from_grammar(grammar: &Grammar) -> GrammarMatcher {
    // Minimal tokenizer info is sufficient for string acceptance tests
    let empty_vocab: Vec<&str> = vec![];
    let stop_ids: Option<Box<[i32]>> = None;
    let tokenizer_info =
        TokenizerInfo::new(&empty_vocab, VocabType::RAW, &stop_ids, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(grammar).unwrap();
    GrammarMatcher::new(&compiled, None, true, -1).unwrap()
}

fn is_grammar_accept_string(
    grammar: &Grammar,
    input: &str,
) -> bool {
    let mut matcher = matcher_from_grammar(grammar);
    let accepted = matcher.accept_string(input, false);
    if !accepted {
        return false;
    }
    matcher.is_terminated()
}

#[test]
#[serial]
fn test_string_schema() {
    let test_cases: &[(&str, bool)] = &[
        (
            "<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter>",
            true,
        ),
        (
            "<parameter=name>Bob</parameter>\t\n<parameter=age>\t100\n</parameter>",
            true,
        ),
        ("<parameter=name>Bob</parameter><parameter=age>100</parameter>", true),
        (
            "\n\t<parameter=name>Bob</parameter><parameter=age>100</parameter>",
            true,
        ),
        (
            "<parameter=name>\"Bob&lt;\"</parameter><parameter=age>100</parameter>",
            true,
        ),
        (
            "<parameter=name>Bob</parameter><parameter=age>100</parameter>\t\t",
            false,
        ),
    ];

    let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}"#;

    let ebnf_grammar = qwen_xml_tool_calling_to_ebnf(schema);
    let grammar = Grammar::from_ebnf(&ebnf_grammar, "root").unwrap();
    for (input_str, accepted) in test_cases {
        assert_eq!(
            is_grammar_accept_string(&grammar, input_str),
            *accepted,
            "Failed for input: {}",
            input_str
        );
    }
}

#[test]
#[serial]
fn test_additional_properties_schema() {
    let test_cases: &[(&str, bool)] = &[
        (
            "<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter><parameter=location>New York</parameter>",
            true,
        ),
        (
            "<parameter=name>Bob</parameter><parameter=age>100</parameter><parameter=123invalid>A</parameter>",
            false,
        ),
    ];

    let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":true}"#;

    let ebnf_grammar = qwen_xml_tool_calling_to_ebnf(schema);
    let grammar = Grammar::from_ebnf(&ebnf_grammar, "root").unwrap();
    for (input_str, accepted) in test_cases {
        assert_eq!(
            is_grammar_accept_string(&grammar, input_str),
            *accepted,
            "Failed for input: {}",
            input_str
        );
    }
}

#[test]
#[serial]
fn test_not_required_properties_schema() {
    let test_cases: &[(&str, bool)] = &[
        (
            "<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter>",
            true,
        ),
        ("<parameter=name>Bob</parameter>", true),
        ("<parameter=age>100</parameter>", true),
        ("", true),
        ("<parameter=anything>It's a string.</parameter>", true),
    ];

    let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"additionalProperties":true}"#;

    let ebnf_grammar = qwen_xml_tool_calling_to_ebnf(schema);
    let grammar = Grammar::from_ebnf(&ebnf_grammar, "root").unwrap();
    for (input_str, accepted) in test_cases {
        assert_eq!(
            is_grammar_accept_string(&grammar, input_str),
            *accepted,
            "Failed for input: {}",
            input_str
        );
    }
}

#[test]
#[serial]
fn test_part_required_properties_schema() {
    let test_cases: &[(&str, bool)] = &[
        (
            "<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter>",
            true,
        ),
        ("<parameter=name>Bob</parameter>", true),
        ("<parameter=age>100</parameter>", false),
        (
            "<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter><parameter=anything>It's a string.</parameter>",
            true,
        ),
        (
            "<parameter=name>Bob</parameter><parameter=anything>It's a string.</parameter>",
            true,
        ),
        ("<parameter=anything>It's a string.</parameter>", false),
    ];

    let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name"],"additionalProperties":true}"#;

    let ebnf_grammar = qwen_xml_tool_calling_to_ebnf(schema);
    let grammar = Grammar::from_ebnf(&ebnf_grammar, "root").unwrap();
    for (input_str, accepted) in test_cases {
        assert_eq!(
            is_grammar_accept_string(&grammar, input_str),
            *accepted,
            "Failed for input: {}",
            input_str
        );
    }
}

#[test]
#[serial]
fn test_inner_object_schema() {
    let test_cases: &[(&str, bool)] = &[
        (
            "<parameter=address>{\"street\": \"Main St\", \"city\": \"New York\"}</parameter>",
            true,
        ),
        (
            "<parameter=address>{\"street\": \"Main St\", \"city\": \"No more xml escape&<>\"}</parameter>",
            true,
        ),
        (
            "<parameter=address>{\"street\": Main St, \"city\": New York}</parameter>",
            false,
        ),
        (
            "<parameter=address><parameter=street>Main St</parameter><parameter=city>New York</parameter></parameter>",
            false,
        ),
        ("<parameter=address>{\"street\": \"Main St\"}</parameter>", false),
        ("<parameter=address>{\"city\": \"New York\"}</parameter>", false),
    ];

    let schema = r#"{"type":"object","properties":{"address":{"type":"object","properties":{"street":{"type":"string"},"city":{"type":"string"}},"required":["street","city"]}},"required":["address"]}"#;

    let ebnf_grammar = qwen_xml_tool_calling_to_ebnf(schema);
    let grammar = Grammar::from_ebnf(&ebnf_grammar, "root").unwrap();
    for (input_str, accepted) in test_cases {
        assert_eq!(
            is_grammar_accept_string(&grammar, input_str),
            *accepted,
            "Failed for input: {}",
            input_str
        );
    }
}

#[test]
#[serial]
fn test_numbers_schema() {
    let test_cases: &[(&str, bool)] = &[
        ("<parameter=age>25</parameter>", false),
        (
            "<parameter=name>Bob</parameter>\n<parameter=age>25</parameter>",
            true,
        ),
        (
            "<parameter=name>Bob</parameter><parameter=ID>123456</parameter><parameter=is_student>true</parameter>",
            true,
        ),
        (
            "<parameter=name>John</parameter><parameter=age>1</parameter><parameter=ID>1</parameter><parameter=is_student>false</parameter>",
            false,
        ),
    ];

    let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"ID":{"type":"integer"},"is_student":{"type":"boolean"}},"maxProperties":3,"minProperties":2}"#;

    let ebnf_grammar = qwen_xml_tool_calling_to_ebnf(schema);
    let grammar = Grammar::from_ebnf(&ebnf_grammar, "root").unwrap();
    for (input_str, accepted) in test_cases {
        assert_eq!(
            is_grammar_accept_string(&grammar, input_str),
            *accepted,
            "Failed for input: {}",
            input_str
        );
    }
}

#[test]
#[serial]
fn test_string_format_length_schema() {
    let test_cases: &[(&str, bool)] = &[
        (
            "<parameter=name>ABC</parameter><parameter=contact_info>{\"phone\": \"12345\",   \"email\": \"test@test.com\"}</parameter>",
            true,
        ),
        (
            "<parameter=name>X</parameter><parameter=contact_info>{\"phone\": \"67890\", \"email\": \"a@b.com\"}</parameter>",
            true,
        ),
        (
            "<parameter=name></parameter><parameter=contact_info>{\"phone\": \"12345\", \"email\": \"test@test.com\"}</parameter>",
            false,
        ),
        (
            "<parameter=name>ABC</parameter><parameter=contact_info>{\"phone\": \"1234\", \"email\": \"test@test.com\"}</parameter>",
            false,
        ),
        (
            "<parameter=name>ABC</parameter><parameter=contact_info>{\"phone\": \"12345\", \"email\": \"not-an-email\"}</parameter>",
            false,
        ),
        (
            "<parameter=name>ABC</parameter><parameter=contact_info>{\"phone\": \"12345\"}</parameter>",
            false,
        ),
        (
            "<parameter=name>ABC</parameter><parameter=contact_info>{\"email\": \"test@test.com\"}</parameter>",
            false,
        ),
        ("<parameter=name>ABC</parameter>", false),
        (
            "<parameter=contact_info>{\"phone\": \"12345\", \"email\": \"test@test.com\"}</parameter>",
            false,
        ),
    ];

    let schema = r#"{"type":"object","properties":{"name":{"type":"string","minLength":1},"contact_info":{"type":"object","properties":{"phone":{"type":"string","pattern":"[0-9]{5}$"},"email":{"type":"string","format":"email"}},"required":["phone","email"]}},"required":["name","contact_info"]}"#;

    let ebnf_grammar = qwen_xml_tool_calling_to_ebnf(schema);
    let grammar = Grammar::from_ebnf(&ebnf_grammar, "root").unwrap();
    for (input_str, accepted) in test_cases {
        assert_eq!(
            is_grammar_accept_string(&grammar, input_str),
            *accepted,
            "Failed for input: {}",
            input_str
        );
    }
}

#[test]
#[serial]
fn test_invalid_function_calling_schema() {
    use std::panic;

    // Empty schema should fail
    let result = panic::catch_unwind(|| {
        qwen_xml_tool_calling_to_ebnf("{}")
    });
    assert!(result.is_err() || {
        // If it doesn't panic, we need to check the result produces invalid grammar
        let ebnf = qwen_xml_tool_calling_to_ebnf("{}");
        Grammar::from_ebnf(&ebnf, "root").is_err()
    });

    // Non-object schema (type: string) should fail
    let result = panic::catch_unwind(|| {
        qwen_xml_tool_calling_to_ebnf(r#"{"type":"string"}"#)
    });
    assert!(result.is_err() || {
        let ebnf = qwen_xml_tool_calling_to_ebnf(r#"{"type":"string"}"#);
        Grammar::from_ebnf(&ebnf, "root").is_err()
    });
}
