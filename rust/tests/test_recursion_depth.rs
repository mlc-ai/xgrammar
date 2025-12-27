mod test_utils;

use serial_test::serial;

#[test]
#[serial]
fn test_set_get_recursion_depth() {
    let default_depth = xgrammar::get_max_recursion_depth();
    assert_eq!(default_depth, 10_000);
    xgrammar::set_max_recursion_depth(1000);
    assert_eq!(xgrammar::get_max_recursion_depth(), 1000);
    xgrammar::set_max_recursion_depth(default_depth);
}

#[test]
#[serial]
fn test_recursion_exceed_does_not_crash() {
    // In Earley Parser, practical recursion depth isn't exceeded for typical grammars.
    // Set a small depth and parse a very long JSON string literal to ensure no crash and acceptance.
    let prev = xgrammar::get_max_recursion_depth();
    xgrammar::set_max_recursion_depth(1000);
    let ebnf = r#"
    root ::= "\"" basic_string "\""
    basic_string ::= "" | [^"\\\r\n] basic_string | "\\" escape basic_string
    escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
    "#;
    let g = xgrammar::Grammar::from_ebnf(ebnf, "root").unwrap();
    let mut m = test_utils::matcher_from_grammar(&g);
    let input = format!("\"{}\"", " ".repeat(10_000));
    assert!(m.accept_string(&input, false));
    assert!(m.is_terminated());
    xgrammar::set_max_recursion_depth(prev);
}

