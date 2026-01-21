use serial_test::serial;

use xgrammar::{Grammar, testing};

#[test]
#[serial]
fn test_tag_dispatch() {
    // Test TagDispatch functionality.
    let before = r#"root ::= TagDispatch(
    ("tag1", rule1),
    ("tag2", rule2),
    stop_eos = false,
    stop_str = ("abc", "def"),
    loop_after_dispatch = false
)
rule1 ::= "a"
rule2 ::= "b"
"#;

    let expected = r#"root ::= ((TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  stop_eos=false,
  stop_str=("abc", "def"),
  loop_after_dispatch=false,
  excludes=()
)))
rule1 ::= (("a"))
rule2 ::= (("b"))
"#;

    let grammar = testing::ebnf_to_grammar_no_normalization(before, "root");
    assert_eq!(grammar.to_string_ebnf(), expected);
}

#[test]
#[serial]
fn test_tag_dispatch_default_parameters() {
    // Test TagDispatch functionality.
    let before = r#"root ::= TagDispatch(("tag1", rule1), ("tag2", rule2))
rule1 ::= "a"
rule2 ::= "b"
"#;
    let expected = r#"root ::= ((TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)))
rule1 ::= (("a"))
rule2 ::= (("b"))
"#;

    let grammar = testing::ebnf_to_grammar_no_normalization(before, "root");
    assert_eq!(grammar.to_string_ebnf(), expected);
}

#[test]
#[serial]
fn test_lookahead_assertion_analyzer_tag_dispatch() {
    // tag dispatch disables lookahead assertion detection
    let before = r#"root ::= TagDispatch(("tag1", rule1), ("tag2", rule2), ("tag3", rule3), ("tag4", rule4), ("tag5", rule5))
rule1 ::= "b"
rule2 ::= "c"
rule3 ::= "" | "d" rule3
rule4 ::= "" | "e" rule4 "f"
rule5 ::= "" | "g" rule5 "h"
"#;
    let expected = r#"root ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  ("tag3", rule3),
  ("tag4", rule4),
  ("tag5", rule5),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
rule1 ::= (("b"))
rule2 ::= (("c"))
rule3 ::= ("" | ("d" rule3))
rule4 ::= ("" | ("e" rule4 "f"))
rule5 ::= ("" | ("g" rule5 "h"))
"#;

    let grammar = testing::ebnf_to_grammar_no_normalization(before, "root");
    let grammar = Grammar::from_ebnf(&grammar.to_string_ebnf(), "root").unwrap();
    let after = grammar.to_string_ebnf();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_tag_dispatch_end_to_end() {
    let before = r#"root ::= TagDispatch(("tag1", rule1), ("tag2", rule2), ("tag3", rule3))
rule1 ::= "a"
rule2 ::= "b"
rule3 ::= "c"
"#;
    let expected = r#"root ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  ("tag3", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
rule1 ::= (("a"))
rule2 ::= (("b"))
rule3 ::= (("c"))
"#;
    let grammar = Grammar::from_ebnf(before, "root").unwrap();
    let after = grammar.to_string_ebnf();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_tag_dispatch_end_to_end_complex() {
    let before = r#"root ::= TagDispatch(("tag1", rule1), ("tag2", rule2), ("tag3", rule3))
rule1 ::= ("a" TagDispatch(("tag1", rule2), ("tag2", rule3)) | "zzz")
rule2 ::= TagDispatch(("tag1", rule2), ("tag2", rule3)) | TagDispatch(("tag3", rule2), ("tag4", rule3))
rule3 ::= "c"
"#;
    let expected = r#"root ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  ("tag3", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
rule1 ::= (("a" rule1_1) | ("zzz"))
rule2 ::= ((rule2_1) | (rule2_2))
rule3 ::= (("c"))
rule1_1 ::= TagDispatch(
  ("tag1", rule2),
  ("tag2", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
rule2_1 ::= TagDispatch(
  ("tag1", rule2),
  ("tag2", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
rule2_2 ::= TagDispatch(
  ("tag3", rule2),
  ("tag4", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
"#;
    let grammar = Grammar::from_ebnf(before, "root").unwrap();
    let after = grammar.to_string_ebnf();
    assert_eq!(after, expected);
}

#[test]
#[serial]
fn test_e2e_tag_dispatch_roundtrip() {
    // Checks the printed result can be parsed, and the parsing-printing process is idempotent.
    let before = r#"root ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  ("tag3", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=()
)
rule1 ::= (("a"))
rule2 ::= (("b"))
rule3 ::= (("c"))
"#;

    let g1 = Grammar::from_ebnf(before, "root").unwrap();
    let s1 = g1.to_string_ebnf();
    let g2 = Grammar::from_ebnf(&s1, "root").unwrap();
    let s2 = g2.to_string_ebnf();
    assert_eq!(s1, before);
    assert_eq!(s2, s1);
}

#[test]
#[serial]
fn test_tag_dispatch_parser_errors() {
    let cases = [
        (
            r#"root ::= TagDispatch(("", rule1))
rule1 ::= "a""#,
            "Tag must be a non-empty string literal",
        ),
        (
            r#"root ::= TagDispatch(("tag1", undefined_rule))"#,
            r#"Rule "undefined_rule" is not defined"#,
        ),
        (
            r#"root ::= TagDispatch("tag1", rule1)"#,
            "Each tag dispatch element must be a tuple",
        ),
        (
            r#"root ::= TagDispatch(("tag1" rule1))"#,
            "Expect , or ) in tuple",
        ),
        (
            r#"root ::= TagDispatch(("tag1", rule1), stop_str=true)
rule1 ::= "a""#,
            "Stop strings must be a tuple",
        ),
        (
            r#"root ::= TagDispatch(("tag1", rule1), stop_eos=false)
rule1 ::= "a""#,
            "The TagDispatch must have stop_eos=true or stop_str is not empty",
        ),
    ];

    for (ebnf, expected_substring) in cases {
        let err = Grammar::from_ebnf(ebnf, "root")
            .err()
            .expect("expected Grammar::from_ebnf to return Err");
        assert!(
            err.contains(expected_substring),
            "unexpected error. want substring={expected_substring:?}, got={err:?}"
        );
    }
}


