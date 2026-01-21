mod test_utils;

use serial_test::serial;
use test_utils::*;

use xgrammar::{Grammar, testing};
#[cfg(feature = "hf")]
use xgrammar::{GrammarCompiler, GrammarMatcher, TokenizerInfo, allocate_token_bitmask};

#[cfg(feature = "hf")]
fn get_stop_token_id(tokenizer_info: &TokenizerInfo) -> Option<i32> {
    tokenizer_info.stop_token_ids().first().copied()
}

#[test]
#[serial]
fn test_basic() {
    let regex = "123";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected = "root ::= \"1\" \"2\" \"3\"\n";
    assert_eq!(grammar_str, expected);

    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "123"));
    assert!(!is_grammar_accept_string(&grammar, "1234"));
}

#[test]
#[serial]
fn test_unicode() {
    let regex = "ww我😁";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected = "root ::= \"w\" \"w\" \"\\u6211\" \"\\U0001f601\"\n";
    assert_eq!(grammar_str, expected);

    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, regex));
}

#[test]
#[serial]
fn test_escape() {
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
        (
            r"[\r\n\$\u0010-\u006F\]\--]+",
            "root ::= [\\r\\n$\\x10-o\\]\\--]+\n",
            "\r\n$\u{0020}-",
        ),
    ];

    for (regex, expected_grammar, instance) in cases {
        let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
        assert_eq!(grammar_str, expected_grammar);
        let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
        assert!(is_grammar_accept_string(&grammar, instance));
    }
}

#[test]
#[serial]
fn test_escaped_char_class() {
    let regex = r"\w\w\W\d\D\s\S";
    let instance = "A_ 1b 0";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = r#"root ::= [a-zA-Z0-9_] [a-zA-Z0-9_] [^a-zA-Z0-9_] [0-9] [^0-9] [\f\n\r\t\v\u0020\u00a0] [^[\f\n\r\t\v\u0020\u00a0]
"#;
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, instance));
}

#[test]
#[serial]
fn test_char_class() {
    let regex = r"[-a-zA-Z+--]+";
    let instance = "a-+";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= [-a-zA-Z+--]+\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, instance));
}

#[test]
#[serial]
fn test_boundary() {
    let regex = r"^abc$";
    let instance = "abc";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= \"a\" \"b\" \"c\"\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, instance));
}

#[test]
#[serial]
fn test_disjunction() {
    let regex = r"abc|de(f|g)";
    let instance = "deg";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar =
        "root ::= \"a\" \"b\" \"c\" | \"d\" \"e\" ( \"f\" | \"g\" )\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, instance));
}

#[test]
#[serial]
fn test_space() {
    let regex = r" abc | df | g ";
    let instance = " df ";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= \" \" \"a\" \"b\" \"c\" \" \" | \" \" \"d\" \"f\" \" \" | \" \" \"g\" \" \"\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, instance));
}

#[test]
#[serial]
fn test_quantifier() {
    let regex = r"(a|b)?[a-z]+(abc)*";
    let instance = "adddabcabc";
    let instance1 = "z";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = r#"root ::= ( "a" | "b" )? [a-z]+ ( "a" "b" "c" )*
"#;
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, instance));
    assert!(is_grammar_accept_string(&grammar, instance1));
}

#[test]
#[serial]
fn test_consecutive_quantifiers() {
    let bad = ["a{1,3}?{1,3}", "a???", "a++", "a+?{1,3}"];
    for regex in bad {
        let err = testing::regex_to_ebnf(regex, true).unwrap_err();
        assert!(
            err.contains("Two consecutive repetition modifiers are not allowed."),
            "unexpected error for {regex}: {err}"
        );
    }
}

#[test]
#[serial]
fn test_group() {
    let regex = r"(a|b)(c|d)";
    let instance = "ac";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = r#"root ::= ( "a" | "b" ) ( "c" | "d" )
"#;
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, instance));
}

#[test]
#[serial]
fn test_any() {
    let regex = r".+a.+";
    let instance = "bbbabb";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = r#"root ::= [\u0000-\U0010FFFF]+ "a" [\u0000-\U0010FFFF]+
"#;
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, instance));
}

/// Test IPv4 regex pattern
#[test]
#[serial]
fn test_ipv4() {
    let regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)(25[0-5]|2[0-4]\d|[01]?\d\d?)";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = r#"root ::= ( ( "2" "5" [0-5] | "2" [0-4] [0-9] | [01]? [0-9] [0-9]? ) [\u0000-\U0010FFFF] ) ( ( "2" "5" [0-5] | "2" [0-4] [0-9] | [01]? [0-9] [0-9]? ) [\u0000-\U0010FFFF] ) ( ( "2" "5" [0-5] | "2" [0-4] [0-9] | [01]? [0-9] [0-9]? ) [\u0000-\U0010FFFF] ) ( "2" "5" [0-5] | "2" [0-4] [0-9] | [01]? [0-9] [0-9]? )
"#;
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "123.45.67.89"));
}

#[test]
#[serial]
fn test_date_time() {
    let regex = r"^\d\d\d\d-(0[1-9]|1[0-2])-([0-2]\d|3[01])T([01]\d|2[0123]):[0-5]\d:[0-5]\d(\.\d+)?(Z|[+-]([01]\d|2[0123]):[0-5]\d)$";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = r#"root ::= [0-9] [0-9] [0-9] [0-9] "-" ( "0" [1-9] | "1" [0-2] ) "-" ( [0-2] [0-9] | "3" [01] ) "T" ( [01] [0-9] | "2" [0123] ) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]+ )? ( "Z" | [+-] ( [01] [0-9] | "2" [0123] ) ":" [0-5] [0-9] )
"#;
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    let cases = [
        ("2024-05-19T14:23:45Z", true),
        ("2019-11-30T08:15:27+05:30", true),
        ("2030-02-01T22:59:59-07:00", true),
        ("2021-07-04T00:00:00.123456Z", true),
        ("2022-12-31T23:45:12-03:00", true),
        ("2024-13-15T14:30:00Z", false),
        ("2023-02-2010:59:59Z", false),
        ("2021-11-05T24:00:00+05:30", false),
        ("2022-08-20T12:61:10-03:00", false),
    ];
    for (instance, accepted) in cases {
        assert_eq!(is_grammar_accept_string(&grammar, instance), accepted);
    }
}

#[test]
#[serial]
fn test_date() {
    let regex = r"^\d\d\d\d-(0[1-9]|1[0-2])-([0-2]\d|3[01])$";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = r#"root ::= [0-9] [0-9] [0-9] [0-9] "-" ( "0" [1-9] | "1" [0-2] ) "-" ( [0-2] [0-9] | "3" [01] )
"#;
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    let cases = [
        ("0024-05-19", true),
        ("2019-11-30", true),
        ("2022-12-31", true),
        ("2024-13-15", false),
        ("2024-12-32", false),
    ];
    for (instance, accepted) in cases {
        assert_eq!(is_grammar_accept_string(&grammar, instance), accepted);
    }
}

#[test]
#[serial]
fn test_time() {
    let regex = r"^([01]\d|2[0123]):[0-5]\d:[0-5]\d(\.\d+)?(Z|[+-]([01]\d|2[0123]):[0-5]\d)$";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = r#"root ::= ( [01] [0-9] | "2" [0123] ) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]+ )? ( "Z" | [+-] ( [01] [0-9] | "2" [0123] ) ":" [0-5] [0-9] )
"#;
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    let cases = [
        ("14:23:45Z", true),
        ("08:15:27+05:30", true),
        ("22:59:59-07:00", true),
        ("00:00:00.123456Z", true),
        ("10:59:59ZA", false),
        ("24:00:00+05:30", false),
        ("12:15:10-03:60", false),
    ];
    for (instance, accepted) in cases {
        assert_eq!(is_grammar_accept_string(&grammar, instance), accepted);
    }
}

#[test]
#[serial]
fn test_email() {
    let regex = r#"^([\w!#$%&'*+/=?^_`{|}~-]+(\.[\w!#$%&'*+/=?^_`{|}~-]+)*|"([\w!#$%&'*+/=?^_`{|}~\-(),:;<>@[\].]|\\")+")@(([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z0-9]([a-z0-9-]*[a-z0-9])?)$"#;
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    let cases = [
        ("simple@example.com", true),
        ("very.common@example.com", true),
        ("user_name+123@example.co.uk", true),
        ("\"john.doe\"@example.org", true),
        ("mail-host@online-shop.biz", true),
        ("customer/department=shipping@example.com", true),
        ("$A12345@example.non-profit.org", true),
        ("\"!def!xyz%abc\"@example.com", true),
        ("support@192.168.1.1", true),
        ("plainaddress", false),
        ("@missingusername.com", false),
        ("user@.com.my", false),
        ("user@com", false),
        ("user@-example.com", false),
    ];
    for (instance, accepted) in cases {
        assert_eq!(is_grammar_accept_string(&grammar, instance), accepted);
    }
}

#[test]
#[serial]
fn test_empty_character_class() {
    let err = testing::regex_to_ebnf("[]", true).unwrap_err();
    assert!(
        err.contains("Empty character class is not allowed in regex."),
        "unexpected error: {err}"
    );
}

#[test]
#[serial]
fn test_group_modifiers() {
    // Test non-capturing group
    let regex = "(?:abc)";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= ( \"a\" \"b\" \"c\" )\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "abc"));

    // Test named capturing group
    let regex = "(?<name>abc)";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= ( \"a\" \"b\" \"c\" )\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "abc"));

    // Test unsupported group modifiers
    let unsupported = [
        ("(?=abc)", "Lookahead is not supported yet."),  // Positive lookahead
        ("(?!abc)", "Lookahead is not supported yet."),  // Negative lookahead
        ("(?<=abc)", "Lookbehind is not supported yet."), // Positive lookbehind
        ("(?<!abc)", "Lookbehind is not supported yet."), // Negative lookbehind
        ("(?i)abc", "Group modifier flag is not supported yet."), // Case-insensitive flag
    ];

    for (regex, expected) in unsupported {
        let err = testing::regex_to_ebnf(regex, true).unwrap_err();
        assert!(err.contains(expected), "regex={regex}, err={err}");
    }
}

/// Test unmatched parentheses errors
#[test]
#[serial]
fn test_unmatched_parentheses() {
    let err = testing::regex_to_ebnf("abc)", true).unwrap_err();
    assert!(err.contains("Unmatched ')'"), "unexpected error: {err}");

    let err = testing::regex_to_ebnf("abc((a)", true).unwrap_err();
    assert!(
        err.contains("The parenthesis is not closed."),
        "unexpected error: {err}"
    );
}

/// Test empty parentheses
#[test]
#[serial]
fn test_empty_parentheses() {
    let regex = "()";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= ( )\n";
    assert_eq!(grammar_str, expected_grammar);

    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, ""));

    let regex = "a()b";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= \"a\" ( ) \"b\"\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "ab"));
}

/// Test empty alternative
#[test]
#[serial]
fn test_empty_alternative() {
    let regex = "(a|)";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= ( \"a\" | \"\" )\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "a"));
    assert!(is_grammar_accept_string(&grammar, ""));
    assert!(!is_grammar_accept_string(&grammar, "b"));

    let regex = "ab(c|)";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= \"a\" \"b\" ( \"c\" | \"\" )\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "abc"));
    assert!(is_grammar_accept_string(&grammar, "ab"));
    assert!(!is_grammar_accept_string(&grammar, "abd"));
}

/// Test non-greedy quantifier
#[test]
#[serial]
fn test_non_greedy_quantifier() {
    let regex = "a{1,3}?";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= \"a\"{1,3}\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "a"));
    assert!(is_grammar_accept_string(&grammar, "aa"));
    assert!(is_grammar_accept_string(&grammar, "aaa"));
    assert!(!is_grammar_accept_string(&grammar, "aaaa"));

    let regex = "a+?";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= \"a\"+\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "a"));
    assert!(is_grammar_accept_string(&grammar, "aa"));
    assert!(is_grammar_accept_string(&grammar, "aaa"));
    assert!(!is_grammar_accept_string(&grammar, ""));

    let regex = "a*?";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= \"a\"*\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "a"));
    assert!(is_grammar_accept_string(&grammar, "aa"));
    assert!(is_grammar_accept_string(&grammar, "aaa"));
    assert!(is_grammar_accept_string(&grammar, ""));

    let regex = "a??";
    let grammar_str = testing::regex_to_ebnf(regex, true).unwrap();
    let expected_grammar = "root ::= \"a\"?\n";
    assert_eq!(grammar_str, expected_grammar);
    let grammar = Grammar::from_ebnf(&grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "a"));
    assert!(is_grammar_accept_string(&grammar, ""));
    assert!(!is_grammar_accept_string(&grammar, "aa"));
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_mask_generation() {
    let tokenizer_paths = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ];
    let regex_instances = [
        (r".+a.+", "bbbabb"),
        (
            r"((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)((25[0-5]|2[0-4]\d|[01]?\d\d?).)(25[0-5]|2[0-4]\d|[01]?\d\d?)",
            "123.45.67.89",
        ),
        (
            r"^\d\d\d\d-(0[1-9]|1[0-2])-([0-2]\d|3[01])T([01]\d|2[0123]):[0-5]\d:[0-5]\d(\.\d+)?(Z|[+-]([01]\d|2[0123]):[0-5]\d)$",
            "2024-05-19T14:23:45Z",
        ),
        (r"^\d\d\d\d-(0[1-9]|1[0-2])-([0-2]\d|3[01])$", "2024-05-19"),
        (
            r"^([01]\d|2[0123]):[0-5]\d:[0-5]\d(\.\d+)?(Z|[+-]([01]\d|2[0123]):[0-5]\d)$",
            "00:00:00.123456Z",
        ),
        (
            r#"^([\w!#$%&'*+/=?^_`{|}~-]+(\.[\w!#$%&'*+/=?^_`{|}~-]+)*|"([\w!#$%&'*+/=?^_`{|}~\-(),:;<>@[\].]|\\")+")@(([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z0-9]([a-z0-9-]*[a-z0-9])?)$"#,
            "customer/department=shipping@test.example.test-example.com",
        ),
    ];

    for tokenizer_path in tokenizer_paths {
        for (regex, instance) in regex_instances {
            println!(
                "Tokenizer: {}, regex: {}, instance: {}",
                tokenizer_path, regex, instance
            );

            let tokenizer_path = test_utils::download_tokenizer_json(tokenizer_path)
                .expect("download tokenizer.json");
            let tokenizer =
                tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();
            let tokenizer_info =
                TokenizerInfo::from_huggingface(&tokenizer, None, None).unwrap();
            let mut grammar_compiler =
                GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();

            let time_start = std::time::Instant::now();
            let ebnf = testing::regex_to_ebnf(regex, true).unwrap();
            let grammar = Grammar::from_ebnf(&ebnf, "root").unwrap();
            let compiled =
                grammar_compiler.compile_grammar(&grammar).unwrap();
            let time_end = time_start.elapsed();
            println!(
                "Time for preprocessing: {} us",
                time_end.as_micros()
            );
            let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
            let mut token_bitmask =
                allocate_token_bitmask(1, tokenizer_info.vocab_size());
            let (mut tensor, _shape, _strides) = create_bitmask_dltensor(
                &mut token_bitmask,
                1,
                tokenizer_info.vocab_size(),
            );

            for c in instance.as_bytes() {
                let time_start = std::time::Instant::now();
                matcher.fill_next_token_bitmask(&mut tensor, 0, false);
                let time_end = time_start.elapsed();
                println!(
                    "Time for fill_next_token_bitmask: {} us",
                    time_end.as_micros()
                );
                let s = unsafe {
                    std::str::from_utf8_unchecked(std::slice::from_ref(c))
                };
                assert!(matcher.accept_string(s, false));
                println!("Accepting {}", c);
            }

            let time_start = std::time::Instant::now();
            matcher.fill_next_token_bitmask(&mut tensor, 0, false);
            let time_end = time_start.elapsed();
            println!(
                "Time for fill_next_token_bitmask: {} us",
                time_end.as_micros()
            );

            if let Some(stop_id) = get_stop_token_id(&tokenizer_info)
                && matcher.accept_token(stop_id)
            {
                assert!(matcher.is_terminated());
            }
        }
    }
}

/// Test empty regex patterns
#[test]
#[serial]
fn test_empty() {
    let empty_regexes = ["", "^$", "(())", "()", "^", "$", "()|()"];

    for regex in empty_regexes {
        let grammar = Grammar::from_regex(regex, true).unwrap();
        let expected = "root ::= (\"\")\n";
        assert_eq!(grammar.to_string(), expected, "regex={regex}");
        assert!(is_grammar_accept_string(&grammar, ""), "regex={regex}");
        assert!(!is_grammar_accept_string(&grammar, "a"), "regex={regex}");
    }
}
