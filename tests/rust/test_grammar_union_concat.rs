mod test_utils;

use serial_test::serial;
use xgrammar::{Grammar, GrammarCompiler, StructuralTagItem, TokenizerInfo, VocabType};

#[test]
#[serial]
fn test_grammar_union() {
    let g1 = Grammar::from_ebnf(
        r#"root ::= r1 | r2
r1 ::= "true" | ""
r2 ::= "false" | ""
"#,
        "root",
    ).unwrap();

    let g2 = Grammar::from_ebnf(
        r#"root ::= "abc" | r1
r1 ::= "true" | r1
"#,
        "root",
    ).unwrap();

    let g3 = Grammar::from_ebnf(
        r#"root ::= r1 | r2 | r3
r1 ::= "true" | r3
r2 ::= "false" | r3
r3 ::= "abc" | ""
"#,
        "root",
    ).unwrap();

    let union = Grammar::union(&[g1, g2, g3]);
    let expected = r#"root ::= ((root_1) | (root_2) | (root_3))
root_1 ::= ((r1) | (r2))
r1 ::= ("" | ("true"))
r2 ::= ("" | ("false"))
root_2 ::= (("abc") | (r1_1))
r1_1 ::= (("true") | (r1_1))
root_3 ::= ((r1_2) | (r2_1) | (r3))
r1_2 ::= (("true") | (r3))
r2_1 ::= (("false") | (r3))
r3 ::= ("" | ("abc"))
"#;
    assert_eq!(union.to_string(), expected);
}

#[test]
#[serial]
fn test_grammar_concat() {
    let g1 = Grammar::from_ebnf(
        r#"root ::= r1 | r2
r1 ::= "true" | ""
r2 ::= "false" | ""
"#,
        "root",
    ).unwrap();

    let g2 = Grammar::from_ebnf(
        r#"root ::= "abc" | r1
r1 ::= "true" | r1
"#,
        "root",
    ).unwrap();

    let g3 = Grammar::from_ebnf(
        r#"root ::= r1 | r2 | r3
r1 ::= "true" | r3
r2 ::= "false" | r3
r3 ::= "abc" | ""
"#,
        "root",
    ).unwrap();

    let concat = Grammar::concat(&[g1, g2, g3]);
    let expected = r#"root ::= ((root_1 root_2 root_3))
root_1 ::= ((r1) | (r2))
r1 ::= ("" | ("true"))
r2 ::= ("" | ("false"))
root_2 ::= (("abc") | (r1_1))
r1_1 ::= (("true") | (r1_1))
root_3 ::= ((r1_2) | (r2_1) | (r3))
r1_2 ::= (("true") | (r3))
r2_1 ::= (("false") | (r3))
r3 ::= ("" | ("abc"))
"#;
    assert_eq!(concat.to_string(), expected);
}

#[test]
#[serial]
fn test_grammar_union_with_stag() {
    let _expected_grammar_union = r#"root ::= ((root_1) | (root_2))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"arg\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("" root_0 "end"))
triggered_tags ::= TagDispatch(
  ("start", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root_1 ::= ((triggered_tags))
root_2 ::= (([a-z] root_2) | ([a-z]))
"#;

    let _expected_grammar_concat = r#"root ::= ((root_1 root_2))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"arg\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("" root_0 "end"))
triggered_tags ::= TagDispatch(
  ("start", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root_1 ::= ((triggered_tags))
root_2 ::= (([a-z] root_2) | ([a-z]))
"#;

    let start = "start";
    let schema = r#"{"type":"object","properties":{"arg":{"type":"string"}}}"#;
    let end = "end";
    let tag = StructuralTagItem::new(start, schema, end);
    let triggers = vec![start];
    let empty_vocab: Vec<&str> = vec![];
    let tok = TokenizerInfo::new(&empty_vocab, VocabType::RAW, &None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tok, 1, false, -1).unwrap();
    let stag_compiled = compiler.compile_structural_tag(&[tag], &triggers).unwrap();
    let stag_grammar = stag_compiled.grammar();
    let start_grammar =
        Grammar::from_ebnf("root ::= [a-z] root | [a-z]", "root").unwrap();
    let grammar_union = Grammar::union(&[stag_grammar, start_grammar]);
    let union_str = grammar_union.to_string();
    assert!(union_str.contains("root ::= ((root_1) | (root_2))"));
    assert!(union_str.contains("root_1 ::= ((triggered_tags))"));
    assert!(union_str.contains("root_2 ::= (([a-z] root_2) | ([a-z]))"));

    let tag = StructuralTagItem::new(start, schema, end);
    let stag_compiled = compiler.compile_structural_tag(&[tag], &triggers).unwrap();
    let stag_grammar = stag_compiled.grammar();
    let start_grammar =
        Grammar::from_ebnf("root ::= [a-z] root | [a-z]", "root").unwrap();
    let grammar_concat = Grammar::concat(&[stag_grammar, start_grammar]);
    let concat_str = grammar_concat.to_string();
    assert!(concat_str.contains("root ::= ((root_1 root_2))"));
    assert!(concat_str.contains("root_1 ::= ((triggered_tags))"));
    assert!(concat_str.contains("root_2 ::= (([a-z] root_2) | ([a-z]))"));
}
