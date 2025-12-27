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
            "<parameter=name><>Bob</parameter><parameter=age>100</parameter>",
            false,
        ),
        (
            "<parameter=name>Bob</parameter><parameter=age>100</parameter>\t\t",
            false,
        ),
    ];

    let expected_grammar = r#"basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
xml_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
xml_entity ::=  "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
xml_string ::= ("" | [^<>&\0-\x1f\\\r\n] xml_string | "\\" xml_escape xml_string | xml_entity xml_string) (= [ \n\t]*)
xml_variable_name ::= [a-zA-Z_] [a-zA-Z0-9_]*
xml_string_0 ::= xml_string
xml_any ::= basic_number | xml_string | basic_boolean | basic_null | basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_0 ::= [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" ""
root ::=  [ \n\t]* (("<parameter=name>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>" root_part_0))"#;

    let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}"#;

    let ebnf_grammar = qwen_xml_tool_calling_to_ebnf(schema);
    assert_eq!(&ebnf_grammar[..ebnf_grammar.len() - 2], expected_grammar);

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

    let expected_grammar = r#"basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
xml_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
xml_entity ::=  "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
xml_string ::= ("" | [^<>&\0-\x1f\\\r\n] xml_string | "\\" xml_escape xml_string | xml_entity xml_string) (= [ \n\t]*)
xml_variable_name ::= [a-zA-Z_] [a-zA-Z0-9_]*
xml_string_0 ::= xml_string
xml_any ::= basic_number | xml_string | basic_boolean | basic_null | basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_1 ::= ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")*
root_part_0 ::= [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::=  [ \n\t]* (("<parameter=name>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>" root_part_0))"#;

    let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":true}"#;

    let ebnf_grammar = qwen_xml_tool_calling_to_ebnf(schema);
    assert_eq!(&ebnf_grammar[..ebnf_grammar.len() - 2], expected_grammar);

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

    let expected_grammar = r#"basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
xml_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
xml_entity ::=  "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
xml_string ::= ("" | [^<>&\0-\x1f\\\r\n] xml_string | "\\" xml_escape xml_string | xml_entity xml_string) (= [ \n\t]*)
xml_variable_name ::= [a-zA-Z_] [a-zA-Z0-9_]*
xml_string_0 ::= xml_string
xml_any ::= basic_number | xml_string | basic_boolean | basic_null | basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_part_1 ::= ([ \n\t]* "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>")*
root_part_0 ::= root_part_1 | [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1
root ::= "" |  [ \n\t]* (("<parameter=name>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>" root_part_0) | ("<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1) | "<parameter=" xml_variable_name ">" [ \n\t]* xml_any [ \n\t]* "</parameter>" root_part_1)"#;

    let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"additionalProperties":true}"#;

    let ebnf_grammar = qwen_xml_tool_calling_to_ebnf(schema);
    assert_eq!(&ebnf_grammar[..ebnf_grammar.len() - 2], expected_grammar);

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

    let expected_grammar = r#"basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
xml_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
xml_entity ::=  "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
xml_string ::= ("" | [^<>&\0-\x1f\\\r\n] xml_string | "\\" xml_escape xml_string | xml_entity xml_string) (= [ \n\t]*)
xml_variable_name ::= [a-zA-Z_] [a-zA-Z0-9_]*
xml_string_0 ::= xml_string
xml_any ::= basic_number | xml_string | basic_boolean | basic_null | basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_prop_0_part_0 ::= [ \n\t]* "," [ \n\t]* "\"city\"" [ \n\t]* ":" [ \n\t]* basic_string ""
root_prop_0 ::= "{" [ \n\t]* (("\"street\"" [ \n\t]* ":" [ \n\t]* basic_string root_prop_0_part_0)) [ \n\t]* "}"
root ::=  [ \n\t]* (("<parameter=address>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" ""))"#;

    let schema = r#"{"type":"object","properties":{"address":{"type":"object","properties":{"street":{"type":"string"},"city":{"type":"string"}},"required":["street","city"]}},"required":["address"]}"#;

    let ebnf_grammar = qwen_xml_tool_calling_to_ebnf(schema);
    assert_eq!(&ebnf_grammar[..ebnf_grammar.len() - 2], expected_grammar);

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

    let expected_grammar = r#"basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^\0-\x1f\"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
xml_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
xml_entity ::=  "&lt;" | "&gt;" | "&amp;" | "&quot;" | "&apos;"
xml_string ::= ("" | [^<>&\0-\x1f\\\r\n] xml_string | "\\" xml_escape xml_string | xml_entity xml_string) (= [ \n\t]*)
xml_variable_name ::= [a-zA-Z_] [a-zA-Z0-9_]*
xml_string_0 ::= xml_string
xml_any ::= basic_number | xml_string | basic_boolean | basic_null | basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_any)* [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= ("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any)* [ \n\t]* "}") | "{" [ \n\t]* "}"
root_prop_1 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_2 ::= ("0" | "-"? [1-9] [0-9]*)
root_prop_3 ::= "true" | "false"
root_part_2_1 ::= [ \n\t]* "<parameter=is_student>" [ \n\t]* root_prop_3 [ \n\t]* "</parameter>" ""
root_part_2_2 ::= "" | [ \n\t]* "<parameter=is_student>" [ \n\t]* root_prop_3 [ \n\t]* "</parameter>" ""
root_part_2_3 ::= ""
root_part_1_1 ::= root_part_2_1 | [ \n\t]* "<parameter=ID>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_2
root_part_1_2 ::= root_part_2_2 | [ \n\t]* "<parameter=ID>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_3
root_part_0_1 ::= root_part_1_1 | [ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1_2
root ::=  [ \n\t]* (("<parameter=name>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>" root_part_0_1) | ("<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1_1) | ("<parameter=ID>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>" root_part_2_1))"#;

    let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"ID":{"type":"integer"},"is_student":{"type":"boolean"}},"maxProperties":3,"minProperties":2}"#;

    let ebnf_grammar = qwen_xml_tool_calling_to_ebnf(schema);
    assert_eq!(&ebnf_grammar[..ebnf_grammar.len() - 2], expected_grammar);

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
