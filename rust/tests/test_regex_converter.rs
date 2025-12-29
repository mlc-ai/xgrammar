mod test_utils;

use serial_test::serial;
use test_utils::*;

use xgrammar::{Grammar, testing};

#[test]
#[serial]
fn test_regex_to_ebnf_basic() {
    let regex = "123";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected = "root ::= \"1\" \"2\" \"3\"\n";
    assert_eq!(grammar_str, expected);

    let g = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&g, "123"));
    assert!(!is_grammar_accept_string(&g, "1234"));
}

#[test]
#[serial]
fn test_regex_to_ebnf_unicode() {
    let regex = "ww我😁";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected = "root ::= \"w\" \"w\" \"\\u6211\" \"\\U0001f601\"\n";
    assert_eq!(grammar_str, expected);

    let g = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&g, regex));
}

#[test]
#[serial]
fn test_regex_to_ebnf_escape_sequences() {
    let cases = [
        (
            r"\^\$\.\*\+\?\\\(\)\[\]\{\}\|\/",
            "root ::= \"^\" \"$\" \".\" \"*\" \"+\" \"\\?\" \"\\\\\" \"(\" \")\" \"[\" \"]\" \"{\" \"}\" \"|\" \"/\"\n",
            "^$.*+?\\()[]{}|/",
        ),
        (
            r#"\"\'\a\f\n\r\t\v\0\e"#,
            "root ::= \"\\\"\" \"\\'\" \"\\a\" \"\\f\" \"\\n\" \"\\r\" \"\\t\" \"\\v\" \"\\0\" \"\\e\"\n",
            "\"'\u{0007}\u{000C}\n\r\t\u{000B}\0\u{001B}",
        ),
        (
            r"\u{20BB7}\u0300\x1F\cJ",
            "root ::= \"\\U00020bb7\" \"\\u0300\" \"\\x1f\" \"\\n\"\n",
            "\u{20BB7}\u{0300}\u{001F}\n",
        ),
    ];

    for (regex, expected_grammar, instance) in cases {
        let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
        assert_eq!(grammar_str, expected_grammar);
        let g = Grammar::from_ebnf(&grammar_str, "root").unwrap();
        assert!(is_grammar_accept_string(&g, instance));
    }
}

#[test]
#[serial]
fn test_regex_to_ebnf_char_classes_and_quantifiers() {
    let cases = [
        (r"\w\w\W\d\D\s\S", r#"root ::= [a-zA-Z0-9_] [a-zA-Z0-9_] [^a-zA-Z0-9_] [0-9] [^0-9] [\f\n\r\t\v\u0020\u00a0] [^[\f\n\r\t\v\u0020\u00a0]
"#, "A_ 1b 0"),
        (r"[-a-zA-Z+--]+", "root ::= [-a-zA-Z+--]+\n", "a-+"),
        (r"^abc$", "root ::= \"a\" \"b\" \"c\"\n", "abc"),
        (
            r"abc|de(f|g)",
            "root ::= \"a\" \"b\" \"c\" | \"d\" \"e\" ( \"f\" | \"g\" )\n",
            "deg",
        ),
        (
            r" abc | df | g ",
            "root ::= \" \" \"a\" \"b\" \"c\" \" \" | \" \" \"d\" \"f\" \" \" | \" \" \"g\" \" \"\n",
            " df ",
        ),
        (
            r"(a|b)?[a-z]+(abc)*",
            "root ::= ( \"a\" | \"b\" )? [a-z]+ ( \"a\" \"b\" \"c\" )*\n",
            "adddabcabc",
        ),
        (
            r"(a|b)(c|d)",
            "root ::= ( \"a\" | \"b\" ) ( \"c\" | \"d\" )\n",
            "ac",
        ),
        (
            r".+a.+",
            "root ::= [\\u0000-\\U0010FFFF]+ \"a\" [\\u0000-\\U0010FFFF]+\n",
            "bbbabb",
        ),
    ];

    for (regex, expected_grammar, instance) in cases {
        let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
        assert_eq!(grammar_str, expected_grammar, "regex={regex}");
        let g = Grammar::from_ebnf(&grammar_str, "root").unwrap();
        assert!(is_grammar_accept_string(&g, instance), "regex={regex}, instance={instance}");
    }
}

#[test]
#[serial]
fn test_regex_to_ebnf_consecutive_quantifiers_error() {
    let bad = ["a{1,3}?{1,3}", "a???", "a++", "a+?{1,3}"];
    for regex in bad {
        let err = testing::regex_to_ebnf(regex, true).unwrap_err();
        assert!(
            err.contains("Two consecutive repetition modifiers")
                || err.contains("Check failed"),
            "unexpected error for {regex}: {err}"
        );
    }
}


