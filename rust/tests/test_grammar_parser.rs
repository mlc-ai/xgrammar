mod test_utils;

use serial_test::serial;
use xgrammar::{Grammar, testing::ebnf_to_grammar_no_normalization};

#[test]
#[serial]
fn test_basic_string_literal() {
    let before = r#"root ::= "hello"
"#;
    let expected = r#"root ::= (("hello"))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_empty_string() {
    let before = r#"root ::= ""
"#;
    let expected = r#"root ::= ((""))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_character_class() {
    let before = r#"root ::= [a-z]
"#;
    let expected = r#"root ::= (([a-z]))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_negated_character_class() {
    let before = r#"root ::= [^a-z]
"#;
    let expected = r#"root ::= (([^a-z]))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_complex_character_class() {
    let before = r#"root ::= [a-zA-Z0-9_-] [\r\n$\x10-o\]\--]
"#;
    let expected = r#"root ::= (([a-zA-Z0-9_\-] [\r\n$\x10-o\]\-\-]))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_sequence() {
    let before = r#"root ::= "a" "b" "c"
"#;
    let expected = r#"root ::= (("a" "b" "c"))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_choice() {
    let before = r#"root ::= "a" | "b" | "c"
"#;
    let expected = r#"root ::= (("a") | ("b") | ("c"))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_grouping() {
    let before = r#"root ::= ("a" "b") | ("c" "d")
"#;
    let expected = r#"root ::= (((("a" "b"))) | ((("c" "d"))))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_star_quantifier_simple() {
    let before = r#"root ::= "a"*
"#;
    let expected = r#"root ::= ((root_1))
root_1 ::= ("" | ("a" root_1))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_plus_quantifier() {
    let before = r#"root ::= "a"+
"#;
    let expected = r#"root ::= ((root_1))
root_1 ::= (("a" root_1) | "a")
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_question_quantifier() {
    let before = r#"root ::= "a"?
"#;
    let expected = r#"root ::= ((root_1))
root_1 ::= ("" | "a")
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_character_class_star() {
    let before = r#"root ::= [a-z]*
"#;
    let expected = r#"root ::= (([a-z]*))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_repetition_range_exact() {
    let before = r#"root ::= "a"{3}
"#;
    let expected = r#"root ::= ((("a" "a" "a")))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_repetition_range_min_max() {
    let before = r#"root ::= "a"{2,4}
"#;
    let expected = r#"root ::= ((("a" "a" root_1)))
root_1 ::= ("" | ("a" root_2))
root_2 ::= ("" | "a")
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_repetition_range_min_only() {
    let before = r#"root ::= "a"{2,}
"#;
    let expected = r#"root ::= ((("a" "a" root_1)))
root_1 ::= ("" | ("a" root_1))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_lookahead_assertion_simple() {
    let before = r#"root ::= "a" (="b")
"#;
    let expected = r#"root ::= (("a")) (=(("b")))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_complex_lookahead() {
    let before = r#"root ::= "a" (="b" "c" [0-9])
"#;
    let expected = r#"root ::= (("a")) (=(("b" "c" [0-9])))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_escape_sequences() {
    let before = r#"root ::= "\n" "\r" "\t" "\\" "\""
"#;
    let expected = r#"root ::= (("\n" "\r" "\t" "\\" "\""))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_unicode_escape() {
    let before = r#"root ::= "\u0041" "\u00E9" "\u4E2D"
"#;
    let expected = r#"root ::= (("A" "\xe9" "\u4e2d"))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_complex_grammar() {
    let before = r#"root ::= item+ ";"
item ::= "a" | "b" | "c"
"#;
    let expected = r#"root ::= ((root_1 ";"))
item ::= (("a") | ("b") | ("c"))
root_1 ::= ((item root_1) | item)
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_nested_quantifiers() {
    let before = r#"root ::= ("a" "b")+
"#;
    let expected = r#"root ::= ((root_1))
root_1 ::= (((("a" "b")) root_1) | (("a" "b")))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_combined_features() {
    let before = r#"root ::= [a-z]+ "-" [0-9]{3}
"#;
    let expected = r#"root ::= ((root_1 "-" ([0-9] [0-9] [0-9])))
root_1 ::= (([a-z] root_1) | [a-z])
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_bnf_comment() {
    let before = r#"root ::= "a" # this is a comment
"#;
    let expected = r#"root ::= (("a"))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_star_quantifier() {
    let before = r#"root ::= a b*
a ::= "a"
b ::= "b"
"#;
    let expected = r#"root ::= ((a root_1))
a ::= (("a"))
b ::= (("b"))
root_1 ::= ("" | (b root_1))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_repetition_range() {
    let before = r#"root ::= a{1,3}
a ::= "a"
"#;
    let expected = r#"root ::= (((a root_1)))
a ::= (("a"))
root_1 ::= ("" | (a root_2))
root_2 ::= ("" | a)
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_lookahead_assertion_with_normalizer() {
    let before = r#"root ::= "a" (=b)
b ::= "b"
"#;
    let expected = r#"root ::= (("a")) (=((b)))
b ::= (("b"))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_char() {
    let before = r#"root ::= [a]
"#;
    let expected = r#"root ::= (([a]))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_space() {
    let before = r#"root ::= " "
"#;
    let expected = r#"root ::= ((" "))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_nest() {
    let before = r#"root ::= (a)
a ::= "a"
"#;
    let expected = r#"root ::= ((((a))))
a ::= (("a"))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_empty_parentheses() {
    let before = r#"root ::= ()
"#;
    let expected = r#"root ::= ((""))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_lookahead_assertion_analyzer() {
    let before = r#"root ::= a (=b)
a ::= "a"
b ::= "b"
"#;
    let expected = r#"root ::= ((a)) (=((b)))
a ::= (("a"))
b ::= (("b"))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_flatten() {
    let before = r#"root ::= a
a ::= b
b ::= "c"
"#;
    let expected = r#"root ::= ((a))
a ::= ((b))
b ::= (("c"))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_rule_inliner_basic() {
    let before = r#"root ::= a
a ::= "a"
"#;
    let expected = r#"root ::= ((a))
a ::= (("a"))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_dead_code_eliminator_basic() {
    let before = r#"root ::= "a"
unused ::= "b"
"#;
    let expected = r#"root ::= (("a"))
unused ::= (("b"))
"#;
    let grammar = ebnf_to_grammar_no_normalization(before, "root");
    let after = grammar.to_string();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_e2e_json_grammar() {
    let json_grammar = Grammar::builtin_json_grammar();
    let json_str = json_grammar.to_string();
    let reparsed = Grammar::from_ebnf(&json_str, "root").unwrap();
    let json_str2 = reparsed.to_string();
    assert_eq!(json_str, json_str2);
}

#[test]
#[serial]
fn test_e2e_to_string_roundtrip() {
    let before = r#"root ::= ((b c) | (b root))
b ::= ((b_1 d))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ("b" b_1)) (=(d))
c_1 ::= (([acep-z] c_1) | ([acep-z])) (=("d"))
d_1 ::= ("" | ("d"))
"#;
    let g1 = Grammar::from_ebnf(before, "root").unwrap();
    let s1 = g1.to_string();
    let g2 = Grammar::from_ebnf(&s1, "root").unwrap();
    let s2 = g2.to_string();
    assert_eq!(s1, s2);
}
