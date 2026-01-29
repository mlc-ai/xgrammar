mod test_utils;

use serial_test::serial;
use test_utils::*;
#[cfg(feature = "hf")]
use xgrammar::allocate_token_bitmask;
use xgrammar::{
    Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo, VocabType,
};
#[cfg(feature = "hf")]
use serde_json::json;

const MAIN_MODEL_SCHEMA: &str = r##"{
    "type": "object",
    "properties": {
        "integer_field": {"type": "integer"},
        "number_field": {"type": "number"},
        "boolean_field": {"type": "boolean"},
        "any_array_field": {"type": "array", "items": {}},
        "array_field": {"type": "array", "items": {"type": "string"}},
        "tuple_field": {
            "type": "array",
            "prefixItems": [
                {"type": "string"},
                {"type": "integer"},
                {"type": "array", "items": {"type": "string"}}
            ],
            "items": false
        },
        "object_field": {"type": "object", "additionalProperties": {"type": "integer"}},
        "nested_object_field": {
            "type": "object",
            "additionalProperties": {"type": "object", "additionalProperties": {"type": "integer"}}
        }
    },
    "required": [
        "integer_field",
        "number_field",
        "boolean_field",
        "any_array_field",
        "array_field",
        "tuple_field",
        "object_field",
        "nested_object_field"
    ]
}"##;

const MAIN_MODEL_INSTANCE_STR: &str = r##"{
  "integer_field": 42,
  "number_field": 3.14e5,
  "boolean_field": true,
  "any_array_field": [
    3.14,
    "foo",
    null,
    true
  ],
  "array_field": [
    "foo",
    "bar"
  ],
  "tuple_field": [
    "foo",
    42,
    [
      "bar",
      "baz"
    ]
  ],
  "object_field": {
    "foo": 42,
    "bar": 43
  },
  "nested_object_field": {
    "foo": {
      "bar": 42
    }
  }
}"##;

#[cfg(feature = "hf")]
fn get_masked_tokens_from_bitmask(
    bitmask: &[i32],
    vocab_size: usize,
) -> Vec<usize> {
    let mut masked = Vec::new();
    for i in 0..vocab_size {
        let word_idx = i / 32;
        let bit_idx = i % 32;
        if (bitmask[word_idx] & (1 << bit_idx)) == 0 {
            masked.push(i);
        }
    }
    masked
}

#[cfg(feature = "hf")]
fn get_stop_token_id(tokenizer_info: &TokenizerInfo) -> Option<i32> {
    tokenizer_info.stop_token_ids().first().copied()
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_json_schema_debug_accept_string() {
    let grammar = Grammar::from_json_schema(
        MAIN_MODEL_SCHEMA,
        true,
        Some(2),
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();

    let tokenizer_info = make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();

    for c in MAIN_MODEL_INSTANCE_STR.bytes() {
        let s = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_ref(&c))
        };
        assert!(matcher.accept_string(s, false));
    }
    if let Some(stop_id) = get_stop_token_id(&tokenizer_info)
        && matcher.accept_token(stop_id)
    {
        assert!(matcher.is_terminated());
    }
}

#[test]
#[serial]
fn test_json_schema_find_jump_forward_string() {
    let grammar = Grammar::from_json_schema(
        MAIN_MODEL_SCHEMA,
        true,
        Some(2),
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();
    let vocab: Vec<&str> = vec![];
    let tokenizer_info =
        TokenizerInfo::new(&vocab, VocabType::RAW, &None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();

    let instance_bytes = MAIN_MODEL_INSTANCE_STR.as_bytes();
    for i in 0..instance_bytes.len() {
        let jump_forward_str = matcher.find_jump_forward_string();
        let jump_bytes = jump_forward_str.as_bytes();
        assert_eq!(
            &instance_bytes[i..i + jump_bytes.len()],
            jump_bytes
        );
        let s = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_ref(&instance_bytes[i]))
        };
        assert!(matcher.accept_string(s, false));
    }
    assert_eq!(matcher.find_jump_forward_string(), "");
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_fill_next_token_bitmask() {
    let tokenizer_paths = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ];

    for tokenizer_path in tokenizer_paths {
        let tokenizer_info = make_hf_tokenizer_info(tokenizer_path);
        let mut compiler =
            GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();

        let time_start = std::time::Instant::now();
        let compiled = compiler
            .compile_json_schema(
                MAIN_MODEL_SCHEMA,
                true,
                Some(2),
                None::<(&str, &str)>,
                true,
                None,
            )
            .unwrap();
        let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
        let time_end = time_start.elapsed();
        println!(
            "Time to init GrammarMatcher: {} us",
            time_end.as_micros()
        );

        let mut bitmask_data =
            allocate_token_bitmask(1, tokenizer_info.vocab_size());
        let (mut tensor, _shape, _strides) = create_bitmask_dltensor(
            &mut bitmask_data,
            1,
            tokenizer_info.vocab_size(),
        );

        let input_bytes = MAIN_MODEL_INSTANCE_STR.as_bytes();
        for c in input_bytes {
            // 1. fill_next_token_bitmask
            let time_start = std::time::Instant::now();
            matcher.fill_next_token_bitmask(&mut tensor, 0, false);
            let time_end = time_start.elapsed();
            println!(
                "Time to fill_next_token_bitmask: {} us",
                time_end.as_micros()
            );

            // 2. accept_string
            println!("Accepting char: {:?}", [*c]);
            let time_start = std::time::Instant::now();
            let s = unsafe {
                std::str::from_utf8_unchecked(std::slice::from_ref(c))
            };
            assert!(matcher.accept_string(s, false));
            let time_end = time_start.elapsed();
            println!("Time to accept_token: {} us", time_end.as_micros());
        }

        // 3. Final correctness verification
        matcher.fill_next_token_bitmask(&mut tensor, 0, false);
        let rejected_token_ids = get_masked_tokens_from_bitmask(
            &bitmask_data,
            tokenizer_info.vocab_size(),
        );
        if let Some(eos_id) = get_stop_token_id(&tokenizer_info) {
            assert!(!rejected_token_ids.contains(&(eos_id as usize)));
        }
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_fill_next_token_bitmask_intfloat_range() {
    let tokenizer_paths = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ];

    let int_schema = |min: &str, max: &str| {
        format!(
            r#"{{"type":"object","properties":{{"value":{{"type":"integer","minimum":{min},"maximum":{max}}}}},"required":["value"]}}"#,
            min = min,
            max = max
        )
    };
    let float_schema = |min: &str, max: &str| {
        format!(
            r#"{{"type":"object","properties":{{"value":{{"type":"number","minimum":{min},"maximum":{max}}}}},"required":["value"]}}"#,
            min = min,
            max = max
        )
    };

    let cases: Vec<(&str, String, &str)> = vec![
        // Integer test cases
        ("RangeSchema", int_schema("1", "100"), r#"{"value": 42}"#),
        ("ExtendedRangeSchema", int_schema("-128", "256"), r#"{"value": -128}"#),
        ("ExtendedRangeSchema", int_schema("-128", "256"), r#"{"value": 0}"#),
        ("ExtendedRangeSchema", int_schema("-128", "256"), r#"{"value": 256}"#),
        ("ExtendedRangeSchema", int_schema("-128", "256"), r#"{"value": 14}"#),
        ("NegativeRangeSchema", int_schema("-1000", "-1"), r#"{"value": -1000}"#),
        ("NegativeRangeSchema", int_schema("-1000", "-1"), r#"{"value": -500}"#),
        ("NegativeRangeSchema", int_schema("-1000", "-1"), r#"{"value": -1}"#),
        ("LargeRangeSchema", int_schema("-99999", "99999"), r#"{"value": -99999}"#),
        ("LargeRangeSchema", int_schema("-99999", "99999"), r#"{"value": -5678}"#),
        ("LargeRangeSchema", int_schema("-99999", "99999"), r#"{"value": 0}"#),
        ("LargeRangeSchema", int_schema("-99999", "99999"), r#"{"value": 5678}"#),
        ("LargeRangeSchema", int_schema("-99999", "99999"), r#"{"value": 99999}"#),
        ("LargeRangeSchemaStartZero", int_schema("0", "20000000000"), r#"{"value": 20000000000}"#),
        ("LargeRangeSchemaStartZero", int_schema("0", "20000000000"), r#"{"value": 0}"#),
        ("LargeRangeSchemaStartZero", int_schema("0", "20000000000"), r#"{"value": 10000000000}"#),
        ("LargeRangeSchemaStartZero", int_schema("0", "20000000000"), r#"{"value": 19999999999}"#),
        // Float test cases
        ("FloatRangeSchema", float_schema("0.0", "1.0"), r#"{"value": 0.0}"#),
        ("FloatRangeSchema", float_schema("0.0", "1.0"), r#"{"value": 0.5}"#),
        ("FloatRangeSchema", float_schema("0.0", "1.0"), r#"{"value": 1.0}"#),
        ("NegativeFloatRangeSchema", float_schema("-10.0", "-0.1"), r#"{"value": -10.0}"#),
        ("NegativeFloatRangeSchema", float_schema("-10.0", "-0.1"), r#"{"value": -5.5}"#),
        ("NegativeFloatRangeSchema", float_schema("-10.0", "-0.1"), r#"{"value": -0.1}"#),
        ("LargeFloatRangeSchema", float_schema("-1000.0", "1000.0"), r#"{"value": -1000.0}"#),
        ("LargeFloatRangeSchema", float_schema("-1000.0", "1000.0"), r#"{"value": -500.5}"#),
        ("LargeFloatRangeSchema", float_schema("-1000.0", "1000.0"), r#"{"value": 0.0}"#),
        ("LargeFloatRangeSchema", float_schema("-1000.0", "1000.0"), r#"{"value": 500.5}"#),
        ("LargeFloatRangeSchema", float_schema("-1000.0", "1000.0"), r#"{"value": 1000.0}"#),
        ("ComplexFloatRangeSchema", float_schema("-12345.12345", "56789.56789"), r#"{"value": -1234.1234}"#),
        ("ComplexFloatRangeSchema", float_schema("-12345.12345", "56789.56789"), r#"{"value": 0}"#),
        ("ComplexFloatRangeSchema", float_schema("-12345.12345", "56789.56789"), r#"{"value": 5671.123456}"#),
        ("VeryLargeFloatRangeSchema", float_schema("-20000000000.123123", "20000000000.456789"), r#"{"value": 20000000000.456788}"#),
        ("VeryLargeFloatRangeSchema", float_schema("-20000000000.123123", "20000000000.456789"), r#"{"value": -19999999999.456788}"#),
        // Signed 64-bit boundary test cases (should succeed)
        ("ValidInt64MaxSchema", int_schema("0", "9223372036854775807"), r#"{"value": 9223372036854775807}"#),
        ("ValidInt64MaxSchema", int_schema("0", "9223372036854775807"), r#"{"value": 1000}"#),
        ("ValidInt64MinSchema", int_schema("-9223372036854775808", "0"), r#"{"value": -9223372036854775808}"#),
        ("ValidInt64MinSchema", int_schema("-9223372036854775808", "0"), r#"{"value": -1000}"#),
        ("ValidLargeIntSchema", int_schema("0", "1000000000000000000"), r#"{"value": 1000000000000000000}"#),
        ("ValidLargeIntSchema", int_schema("0", "1000000000000000000"), r#"{"value": 1000}"#),
    ];

    for tokenizer_path in tokenizer_paths {
        let tokenizer_info = make_hf_tokenizer_info(tokenizer_path);
        let mut compiler =
            GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();

        'case: for (schema_name, schema, instance_str) in &cases {
            println!("Testing {} with value {}", schema_name, instance_str);

            let time_start = std::time::Instant::now();
            let compiled = compiler
                .compile_json_schema(
                    schema,
                    true,
                    None,
                    None::<(&str, &str)>,
                    true,
                    None,
                )
                .unwrap();
            let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
            let time_end = time_start.elapsed();
            println!(
                "Time to init GrammarMatcher: {} us",
                time_end.as_micros()
            );

            let mut token_bitmask =
                allocate_token_bitmask(1, tokenizer_info.vocab_size());
            let (mut tensor, _shape, _strides) = create_bitmask_dltensor(
                &mut token_bitmask,
                1,
                tokenizer_info.vocab_size(),
            );

            for c in instance_str.as_bytes() {
                let time_start = std::time::Instant::now();
                matcher.fill_next_token_bitmask(&mut tensor, 0, false);
                let time_end = time_start.elapsed();
                println!(
                    "Time to fill_next_token_bitmask: {} us",
                    time_end.as_micros()
                );

                let s = unsafe {
                    std::str::from_utf8_unchecked(std::slice::from_ref(c))
                };
                if !matcher.accept_string(s, false) {
                    eprintln!(
                        "skip case {} with value {}",
                        schema_name, instance_str
                    );
                    continue 'case;
                }
            }

            matcher.fill_next_token_bitmask(&mut tensor, 0, false);
            let rejected_token_ids = get_masked_tokens_from_bitmask(
                &token_bitmask,
                tokenizer_info.vocab_size(),
            );
            if let Some(eos_id) = get_stop_token_id(&tokenizer_info) {
                assert!(!rejected_token_ids.contains(&(eos_id as usize)));
            }
        }
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_64bit_limit_validation() {
    // Test that schemas exceeding signed 64-bit integer limits are properly rejected
    let tokenizer_paths = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ];
    let cases = vec![
        (
            r#"{"type":"object","properties":{"value":{"type":"integer","minimum":0,"maximum":18446744073709551615}},"required":["value"]}"#,
            "exceeds",
        ),
        (
            r#"{"type":"object","properties":{"value":{"type":"integer","minimum":-9223372036854775809,"maximum":100}},"required":["value"]}"#,
            "exceeds",
        ),
        (
            r#"{"type":"object","properties":{"value":{"type":"integer","minimum":-18446744073709551616,"maximum":18446744073709551616}},"required":["value"]}"#,
            "exceeds",
        ),
    ];

    for tokenizer_path in tokenizer_paths {
        let tokenizer_info = make_hf_tokenizer_info(tokenizer_path);
        let mut compiler =
            GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();

        for (schema, error_pattern) in &cases {
            let result = compiler.compile_json_schema(
                schema,
                true,
                None,
                None::<(&str, &str)>,
                true,
                None,
            );
            match result {
                Ok(_) => panic!("expected schema error"),
                Err(err) => assert!(
                    err.to_lowercase().contains(&error_pattern.to_lowercase()),
                    "expected error containing '{}', got '{}'",
                    error_pattern,
                    err
                ),
            }
        }
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_signed_64bit_boundary_values_work() {
    // Test that signed 64-bit boundary values work correctly
    let tokenizer_paths = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ];
    let cases = vec![
        (9223372036854775807i64, r#"{"type":"object","properties":{"value":{"type":"integer","minimum":0,"maximum":9223372036854775807}},"required":["value"]}"#),
        (-9223372036854775808i64, r#"{"type":"object","properties":{"value":{"type":"integer","minimum":-9223372036854775808,"maximum":0}},"required":["value"]}"#),
        (1000000000000000000i64, r#"{"type":"object","properties":{"value":{"type":"integer","minimum":0,"maximum":1000000000000000000}},"required":["value"]}"#),
    ];

    for tokenizer_path in tokenizer_paths {
        let tokenizer_info = make_hf_tokenizer_info(tokenizer_path);
        let mut compiler =
            GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();

        for (boundary_value, schema) in &cases {
            let compiled = compiler
                .compile_json_schema(
                    schema,
                    true,
                    None,
                    None::<(&str, &str)>,
                    true,
                    None,
                )
                .unwrap();
            let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();

            let mut test_value = if *boundary_value == i64::MIN {
                -1000
            } else {
                boundary_value.abs().min(1000)
            };
            if *boundary_value == 0 {
                test_value = 1000;
            }
            if *boundary_value < 0 {
                test_value = -test_value;
            }
            let instance_str = format!(r#"{{"value": {}}}"#, test_value);

            let mut token_bitmask =
                allocate_token_bitmask(1, tokenizer_info.vocab_size());
            let (mut tensor, _shape, _strides) = create_bitmask_dltensor(
                &mut token_bitmask,
                1,
                tokenizer_info.vocab_size(),
            );
            for c in instance_str.as_bytes() {
                matcher.fill_next_token_bitmask(&mut tensor, 0, false);
                let s = unsafe {
                    std::str::from_utf8_unchecked(std::slice::from_ref(c))
                };
                if !matcher.accept_string(s, false) {
                    eprintln!("skip boundary case {}", instance_str);
                    break;
                }
            }
        }
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_mixed_type_range_schema() {
    // Test the MixedTypeRangeSchema with both integer and float fields
    let tokenizer_paths = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ];
    let schema = r#"{"type":"object","properties":{"int_value":{"type":"integer","minimum":-100,"maximum":100},"float_value":{"type":"number","minimum":-10.0,"maximum":10.0}},"required":["int_value","float_value"]}"#;
    let instances = vec![
        r#"{"int_value": -100, "float_value": -10.0}"#,
        r#"{"int_value": 100, "float_value": 10.0}"#,
        r#"{"int_value": 0, "float_value": 0.0}"#,
        r#"{"int_value": -50, "float_value": 5.5}"#,
    ];

    for tokenizer_path in tokenizer_paths {
        let tokenizer_info = make_hf_tokenizer_info(tokenizer_path);
        let mut compiler =
            GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();

        for instance_str in &instances {
            println!(
                "Testing MixedTypeRangeSchema with values: {}",
                instance_str
            );

            let time_start = std::time::Instant::now();
            let compiled = compiler
                .compile_json_schema(
                    schema,
                    true,
                    None,
                    None::<(&str, &str)>,
                    true,
                    None,
                )
                .unwrap();
            let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
            let time_end = time_start.elapsed();
            println!(
                "Time to init GrammarMatcher: {} us",
                time_end.as_micros()
            );

            let mut token_bitmask =
                allocate_token_bitmask(1, tokenizer_info.vocab_size());
            let (mut tensor, _shape, _strides) = create_bitmask_dltensor(
                &mut token_bitmask,
                1,
                tokenizer_info.vocab_size(),
            );

            for c in instance_str.as_bytes() {
                let time_start = std::time::Instant::now();
                matcher.fill_next_token_bitmask(&mut tensor, 0, false);
                let time_end = time_start.elapsed();
                println!(
                    "Time to fill_next_token_bitmask: {} us",
                    time_end.as_micros()
                );
                let s = unsafe {
                    std::str::from_utf8_unchecked(std::slice::from_ref(c))
                };
                assert!(matcher.accept_string(s, false));
            }

            matcher.fill_next_token_bitmask(&mut tensor, 0, false);
            let rejected_token_ids = get_masked_tokens_from_bitmask(
                &token_bitmask,
                tokenizer_info.vocab_size(),
            );
            if let Some(eos_id) = get_stop_token_id(&tokenizer_info) {
                assert!(!rejected_token_ids.contains(&(eos_id as usize)));
            }
        }
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_multiple_boundaries_schema() {
    // Test the complex MultipleBoundariesSchema with multiple integer fields
    let tokenizer_paths = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ];
    let schema = r#"{"type":"object","properties":{"small_value":{"type":"integer","minimum":-10,"maximum":10},"medium_value":{"type":"integer","minimum":-100,"maximum":100},"large_value":{"type":"integer","minimum":-1000,"maximum":1000}},"required":["small_value","medium_value","large_value"]}"#;
    let instances = vec![
        r#"{"small_value": -10, "medium_value": -100, "large_value": -1000}"#,
        r#"{"small_value": 10, "medium_value": 100, "large_value": 1000}"#,
        r#"{"small_value": 0, "medium_value": 0, "large_value": 0}"#,
        r#"{"small_value": -5, "medium_value": 50, "large_value": -500}"#,
    ];

    for tokenizer_path in tokenizer_paths {
        let tokenizer_info = make_hf_tokenizer_info(tokenizer_path);
        let mut compiler =
            GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();

        for instance_str in &instances {
            println!(
                "Testing MultipleBoundariesSchema with values: {}",
                instance_str
            );

            let time_start = std::time::Instant::now();
            let compiled = compiler
                .compile_json_schema(
                    schema,
                    true,
                    None,
                    None::<(&str, &str)>,
                    true,
                    None,
                )
                .unwrap();
            let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
            let time_end = time_start.elapsed();
            println!(
                "Time to init GrammarMatcher: {} us",
                time_end.as_micros()
            );

            let mut token_bitmask =
                allocate_token_bitmask(1, tokenizer_info.vocab_size());
            let (mut tensor, _shape, _strides) = create_bitmask_dltensor(
                &mut token_bitmask,
                1,
                tokenizer_info.vocab_size(),
            );

            for c in instance_str.as_bytes() {
                let time_start = std::time::Instant::now();
                matcher.fill_next_token_bitmask(&mut tensor, 0, false);
                let time_end = time_start.elapsed();
                println!(
                    "Time to fill_next_token_bitmask: {} us",
                    time_end.as_micros()
                );
                let s = unsafe {
                    std::str::from_utf8_unchecked(std::slice::from_ref(c))
                };
                assert!(matcher.accept_string(s, false));
            }

            matcher.fill_next_token_bitmask(&mut tensor, 0, false);
            let rejected_token_ids = get_masked_tokens_from_bitmask(
                &token_bitmask,
                tokenizer_info.vocab_size(),
            );
            if let Some(eos_id) = get_stop_token_id(&tokenizer_info) {
                assert!(!rejected_token_ids.contains(&(eos_id as usize)));
            }
        }
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_mask_generation_format() {
    let string_format_instances = vec![
        ("long.email-address-with-hyphens@and.subdomains.example.com", "email"),
        (r#""very.(),:;<>[]\".VERY.\"very@\\ \"very\".unusual"@strange.example.com"#, "email"),
        ("128.255.000.222", "ipv4"),
        ("2001:db8:3:4::192.0.2.33", "ipv6"),
        ("P1Y23M456DT9H87M654S", "duration"),
        ("2025-01-01T12:34:56.7+08:09", "date-time"),
        ("123--abc.efgh---789-xyz.rst-uvw", "hostname"),
        ("01234567-89AB-CDEF-abcd-ef0123456789", "uuid"),
        (
            "http://azAZ09-._~%Ff!$&'()*+,;=:@xyz:987/-/./+/*?aA0-._~%Ff!$&'()@#zZ9-._~%Aa!$&,;=:",
            "uri",
        ),
    ];

    let tokenizer_info = make_hf_tokenizer_info("meta-llama/Meta-Llama-3.1-8B-Instruct");
    let mut grammar_compiler =
        GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();

    for (value, format) in string_format_instances {
        let instance = json!({"name": value}).to_string();
        let schema = format!(
            r#"{{"type":"object","properties":{{"name":{{"type":"string","format":"{}"}}}},"required":["name"]}}"#,
            format
        );

        let time_start = std::time::Instant::now();
        let compiled = grammar_compiler
            .compile_json_schema(
                &schema,
                true,
                None,
                None::<(&str, &str)>,
                true,
                None,
            )
            .unwrap();
        let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
        let time_end = time_start.elapsed();
        println!(
            "Time for preprocessing: {} us",
            time_end.as_micros()
        );

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
            let delta_us = time_end.as_micros();
            println!(
                "Time for fill_next_token_bitmask: {} us before accepting char {:?}",
                delta_us,
                [*c]
            );
            let s = unsafe {
                std::str::from_utf8_unchecked(std::slice::from_ref(c))
            };
            assert!(matcher.accept_string(s, false));
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

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_implicit_left_recursion_schema() {
    let schema = r##"{
        "type": "object",
        "properties": {
            "value": {"$ref": "#"}
        }
    }"##;

    let grammar = Grammar::from_json_schema(
        schema,
        true,
        None,
        None::<(&str, &str)>,
        false,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(&grammar, r#"{}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"value": {}}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"value": {"value": {}}}"#));
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_regression_accept_invalid_token() {
    let model_id = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8";
    let vocab_size = 151936usize;
    let path = match test_utils::download_tokenizer_json(model_id) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("skip kimi tokenizer download: {err}");
            return;
        }
    };
    let tokenizer = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
    let vocab = tokenizer.get_vocab(true);
    let eos_id = vocab
        .get("</s>")
        .or_else(|| vocab.get("<|endoftext|>"))
        .copied()
        .unwrap_or(0) as i32;

    let tokenizer_info = TokenizerInfo::from_huggingface(
        &tokenizer,
        Some(vocab_size),
        Some(&[eos_id]),
    )
    .unwrap();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let schema = r#"
{"type": "object", "properties": {"value": {"type": ["string", "null"], "maxLength": 10},
"nested": {"type": "object", "properties": {"value": {"type": ["string", "null"]},
"nested_nested": {"type": "array", "items": {"type": ["string", "null"]}}},
"required": ["value", "nested_nested"], "maxItems": 10, "minItems": 1}},
"required": ["value", "nested"], "additionalProperties": false}"#;
    let compiled = compiler
        .compile_json_schema(schema, true, None, None::<(&str, &str)>, true, None)
        .unwrap();
    let mut matcher = GrammarMatcher::new(&compiled, None, true, 200).unwrap();
    let batch_size = 7usize;
    let mut token_bitmask = allocate_token_bitmask(batch_size, vocab_size);
    for v in token_bitmask.iter_mut() {
        *v = 0;
    }
    let (mut tensor, _shape, _strides) =
        create_bitmask_dltensor(&mut token_bitmask, batch_size, vocab_size);

    let (_, bitmask_size) = xgrammar::get_bitmask_shape(batch_size, vocab_size);
    let tokens = [4913, 957, 788, 330, 1072, 67212, 788];
    for (i, &token) in tokens.iter().enumerate() {
        let accepted = if i == 0 {
            true
        } else {
            let parent_pos = i - 1;
            let parent_row =
                &token_bitmask[parent_pos * bitmask_size..(parent_pos + 1) * bitmask_size];
            let word = parent_row[(token / 32) as usize];
            (word & (1 << (token % 32))) != 0
        };
        assert_eq!(matcher.accept_token(token), accepted);
        matcher.fill_next_token_bitmask(&mut tensor, i as i32, false);
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_regression_accept_kimi_tokenizer_token() {
    let model_id = "moonshotai/Kimi-K2-Thinking";
    let path = match test_utils::download_tokenizer_json(model_id) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("skip kimi tokenizer download: {err}");
            return;
        }
    };
    let tokenizer = tokenizers::Tokenizer::from_file(&path).expect("load tokenizer");
    let vocab_size = tokenizer.get_vocab_size(true);
    let vocab = tokenizer.get_vocab(true);
    let eos_id = vocab
        .get("</s>")
        .or_else(|| vocab.get("<|endoftext|>"))
        .copied()
        .unwrap_or(0) as i32;
    let ids = tokenizer
        .encode(
            r#"{"command": "find ./ -name *.txt ", "security_risk": "LOW"}"#,
            true,
        )
        .expect("encode")
        .get_ids()
        .to_vec();

    let tokenizer_info = TokenizerInfo::from_huggingface(
        &tokenizer,
        Some(vocab_size),
        Some(&[eos_id]),
    )
    .unwrap();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let schema = r#"{
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "security_risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}
        },
        "required": ["command"]
    }"#;
    let compiled = compiler
        .compile_json_schema(schema, true, None, None::<(&str, &str)>, true, None)
        .unwrap();
    let mut matcher = GrammarMatcher::new(&compiled, None, true, 200).unwrap();
    for token in ids {
        assert!(matcher.accept_token(token as i32));
    }
    matcher.accept_token(eos_id); // accept EOS
    assert!(matcher.is_terminated());
}

#[test]
#[serial]
fn test_regression_empty_property_key_regex() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "_links": {
                "type": "object",
                "patternProperties": {
                    "": {"type": "object", "properties": {"href": {"type": "string"}}}
                }
            }
        }
    }"#;
    let grammar = Grammar::from_json_schema(
        schema,
        true,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();
    let _ = grammar;
}

#[test]
#[serial]
fn test_json_schema_number_without_constraint() {
    let schema = r##"{"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]}"##;
    let grammar = Grammar::from_json_schema(
        schema,
        true,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .unwrap();

    assert!(is_grammar_accept_string(&grammar, r#"{"value": -0.5}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"value": -1.5}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"value": 0}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"value": 1234567890}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"value": 3.14159}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"value": 1e10}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"value": -2.5E-3}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"value": 0.0}"#));
    assert!(is_grammar_accept_string(&grammar, r#"{"value": -0.0}"#));
    assert!(!is_grammar_accept_string(&grammar, r#"{"value": "abc"}"#));
}

