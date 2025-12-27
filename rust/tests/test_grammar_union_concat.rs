mod test_utils;

use serial_test::serial;
use xgrammar::Grammar;

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
