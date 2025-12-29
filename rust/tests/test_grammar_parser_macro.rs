use serial_test::serial;

use xgrammar::{Grammar, testing};

#[test]
#[serial]
fn test_tag_dispatch_print() {
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
  loop_after_dispatch=false
)))
rule1 ::= (("a"))
rule2 ::= (("b"))
"#;

    let g = testing::ebnf_to_grammar_no_normalization(before, "root");
    assert_eq!(g.to_string_ebnf(), expected);
}

#[test]
#[serial]
fn test_tag_dispatch_default_parameters_print() {
    let before = r#"root ::= TagDispatch(("tag1", rule1), ("tag2", rule2))
rule1 ::= "a"
rule2 ::= "b"
"#;
    let expected = r#"root ::= ((TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)))
rule1 ::= (("a"))
rule2 ::= (("b"))
"#;

    let g = testing::ebnf_to_grammar_no_normalization(before, "root");
    assert_eq!(g.to_string_ebnf(), expected);
}

#[test]
#[serial]
fn test_tag_dispatch_roundtrip_idempotent() {
    let before = r#"root ::= TagDispatch(
  ("tag1", rule1),
  ("tag2", rule2),
  ("tag3", rule3),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false
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


