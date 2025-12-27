mod test_utils;

use serial_test::serial;
use test_utils::*;
use xgrammar::Grammar;
#[cfg(feature = "hf")]
use xgrammar::{GrammarCompiler, GrammarMatcher};

#[test]
#[serial]
fn test_simple() {
    let regex_str = "abc";
    let grammar = Grammar::from_regex(regex_str, false).unwrap();
    assert!(is_grammar_accept_string(&grammar, "abc"));
    assert!(!is_grammar_accept_string(&grammar, "ab"));
    assert!(!is_grammar_accept_string(&grammar, "abcd"));
}

#[test]
#[serial]
fn test_repetition() {
    let regex_str = "(a|[bc]{4,}){2,3}";
    let grammar = Grammar::from_regex(regex_str, false).unwrap();
    let cases = [
        ("aaa", true),
        ("abcbc", true),
        ("bcbcbcbcbc", true),
        ("bcbcbcbcbcbcbcb", true),
        ("d", false),
        ("aaaa", false),
    ];
    for (input, accepted) in cases {
        assert_eq!(
            is_grammar_accept_string(&grammar, input),
            accepted,
            "{}",
            input
        );
    }
}

#[test]
#[serial]
fn test_regex_accept() {
    let patterns = [
        r"abc",
        r"[abc]+",
        r"[a-z0-9]+",
        r"[^abc]+",
        r"a*b+c?",
        r"(abc|def)+",
        r"a{2,4}",
        r"\d+",
        r"\w+",
        r"[A-Z][a-z]*",
        r"[0-9]{3}-[0-9]{3}-[0-9]{4}",
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    ];
    for p in patterns {
        let grammar = Grammar::from_regex(p, false).unwrap();
        assert!(grammar.to_string_ebnf().len() > 0, "Pattern: {}", p);
    }
}

// Note: The invalid regex tests from Python cannot be properly ported because
// the C++ implementation uses XGRAMMAR_LOG(FATAL) which aborts the process.
// In Python, this is caught as a RuntimeError, but in Rust/C++, it terminates.
// Test cases for invalid patterns:
// - r"a{,3}" - Invalid range
// - r"a{3,2}" - Invalid range (max < min)
// - r"[z-a]" - Invalid range (max < min)
// - r"a++" - Invalid repetition
// - r"(?=a)" - Lookahead not supported
// - r"(?!a)" - Negative lookahead not supported

#[test]
#[serial]
fn test_advanced() {
    let cases = [
        // Basic patterns
        (r"abc", "abc", true),
        (r"abc", "def", false),
        // Character classes
        (r"[abc]+", "aabbcc", true),
        (r"[abc]+", "abcd", false),
        (r"[a-z0-9]+", "abc123", true),
        (r"[a-z0-9]+", "ABC", false),
        (r"[^abc]+", "def", true),
        (r"[^abc]+", "aaa", false),
        // Lazy character class
        (r"[abc]+?abc", "aabc", true),
        // Quantifiers
        (r"a*b+c?", "b", true),
        (r"a*b+c?", "aaabbc", true),
        (r"a*b+c?", "c", false),
        // Alternation
        (r"(abc|def)+", "abcdef", true),
        (r"(abc|def)+", "abcabc", true),
        (r"(abc|def)+", "ab", false),
        // Repetition ranges
        (r"a{2,4}", "aa", true),
        (r"a{2,4}", "aaaa", true),
        (r"a{2,4}", "a", false),
        (r"a{2,4}", "aaaaa", false),
        // Common patterns
        (r"\d+", "123", true),
        (r"\d+", "abc", false),
        (r"\w+", "abc123", true),
        (r"\w+", "!@#", false),
        (r"[A-Z][a-z]*", "Hello", true),
        (r"[A-Z][a-z]*", "hello", false),
        // Complex patterns
        (r"[0-9]{3}-[0-9]{3}-[0-9]{4}", "123-456-7890", true),
        (r"[0-9]{3}-[0-9]{3}-[0-9]{4}", "12-34-567", false),
        (
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}",
            "test@email.com",
            true,
        ),
        (
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}",
            "invalid.email",
            false,
        ),
    ];
    for (regex, instance, expected) in cases {
        let g = Grammar::from_regex(regex, false).unwrap();
        assert_eq!(
            is_grammar_accept_string(&g, instance),
            expected,
            "regex: {}, instance: {}",
            regex,
            instance
        );
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_fill_next_token_bitmask() {
    use xgrammar::{
        GrammarCompiler, GrammarMatcher, allocate_token_bitmask, testing,
    };

    let test_cases = [
        (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}", "test@email.com"),
        (r"[0-9]{3}-[0-9]{3}-[0-9]{4}", "123-456-7890"),
    ];

    for (regex, input_str) in test_cases {
        // Note: Using Llama-2 instead of Llama-3 due to authentication requirements
        let tokenizer_info =
            make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
        let mut compiler = GrammarCompiler::new(&tokenizer_info, 8, false, -1).unwrap();

        let compiled_grammar = compiler.compile_regex(regex).unwrap();
        let mut matcher =
            GrammarMatcher::new(&compiled_grammar, None, false, -1).unwrap();

        let vocab_size = tokenizer_info.vocab_size();
        let mut bitmask_data = allocate_token_bitmask(1, vocab_size);

        let input_bytes = input_str.as_bytes();

        for &c in input_bytes {
            let (mut tensor, _shape, _strides) =
                create_bitmask_dltensor(&mut bitmask_data, 1, vocab_size);

            assert!(matcher.fill_next_token_bitmask(&mut tensor, 0, false));

            let byte_array = [c];
            let byte_str = std::str::from_utf8(&byte_array).unwrap_or("");
            assert!(matcher.accept_string(byte_str, false));

            // Reset bitmask for next iteration
            bitmask_data.fill(-1);
        }

        // Final verification - check that EOS token is not rejected
        let (mut tensor, _shape, _strides) =
            create_bitmask_dltensor(&mut bitmask_data, 1, vocab_size);
        matcher.fill_next_token_bitmask(&mut tensor, 0, false);
        let rejected_token_ids = testing::get_masked_tokens_from_bitmask(
            &tensor,
            vocab_size as i32,
            0,
        );

        let eos_id = tokenizer_info.stop_token_ids()[0];
        assert!(
            !rejected_token_ids.contains(&eos_id),
            "EOS token should not be rejected"
        );
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_regex_with_large_range_compilation() {
    use xgrammar::GrammarCompiler;

    let regex_with_large_range = r"[a-z]{100,20000}";
    // Note: Using Llama-2 instead of Llama-3 due to authentication requirements
    let tokenizer_info =
        make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 8, false, -1).unwrap();

    let _ = compiler.compile_regex(regex_with_large_range);
    // Test passes if compilation succeeds without panic
}


#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_regression_lookahead_already_completed() {
    let tk = make_hf_tokenizer_info("Qwen/Qwen2.5-0.5B");
    let regex = r"[0-9]+";
    let mut compiler = GrammarCompiler::new(&tk, 1, false, -1).unwrap();
    let grammar = Grammar::from_regex(regex, false).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
    assert!(matcher.accept_string("123", false));
    assert!(matcher.is_terminated());
}
