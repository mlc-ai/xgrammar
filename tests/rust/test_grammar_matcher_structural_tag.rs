mod test_utils;

use serial_test::serial;
use test_utils::*;

use xgrammar::{Grammar, GrammarCompiler, GrammarMatcher, StructuralTagItem, TokenizerInfo, VocabType};
use serde_json::json;

const EXPECTED_GRAMMAR_TEST_STRUCTURAL_TAG_AFTER_OPTIMIZATION: &str = r#"basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(basic_string_sub))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_string ::= (("\"" basic_string_sub)) (=(root_part_0 [ \n\t]* "}"))
root_part_0 ::= (([ \n\t]* "," [ \n\t]* "\"arg2\"" [ \n\t]* ":" [ \n\t]* basic_integer)) (=([ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"arg1\"" [ \n\t]* ":" [ \n\t]* basic_string root_part_0 [ \n\t]* "}")) (=("</function>"))
basic_integer_1 ::= ("" | ("-")) (=([1-9] [0-9]*))
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(basic_string_sub_1))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_string_1 ::= (("\"" basic_string_sub_1)) (=(root_part_0_1 [ \n\t]* "}"))
root_part_0_1 ::= (([ \n\t]* "," [ \n\t]* "\"arg2\"" [ \n\t]* ":" [ \n\t]* basic_integer_2)) (=([ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"arg1\"" [ \n\t]* ":" [ \n\t]* basic_string_1 root_part_0_1 [ \n\t]* "}")) (=("</function>"))
basic_integer_1_1 ::= ("" | ("-")) (=([1-9] [0-9]*))
basic_escape_2 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(basic_string_sub_2))
basic_string_sub_2 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_2) | ("\\" basic_escape_2 basic_string_sub_2)) (=([ \n\t]* [,}\]:]))
basic_number_9 ::= ((basic_number_1_2 basic_number_7_2 basic_number_3_2 basic_number_6_2)) (=(root_part_0_2 [ \n\t]* "}"))
basic_string_2 ::= (("\"" basic_string_sub_2))
root_prop_1 ::= (("[" [ \n\t]* basic_string_2 root_prop_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
root_part_0_2 ::= (([ \n\t]* "," [ \n\t]* "\"arg4\"" [ \n\t]* ":" [ \n\t]* root_prop_1)) (=([ \n\t]* "}"))
root_2 ::= (("{" [ \n\t]* "\"arg3\"" [ \n\t]* ":" [ \n\t]* basic_number_9 root_part_0_2 [ \n\t]* "}")) (=("</function>"))
basic_number_1_2 ::= ("" | ("-")) (=(basic_number_7_2 basic_number_3_2 basic_number_6_2))
basic_number_2_2 ::= (([0-9] basic_number_2_2) | ([0-9]))
basic_number_3_2 ::= ("" | ("." basic_number_2_2)) (=(basic_number_6_2))
basic_number_4_2 ::= ("" | ([+\-])) (=(basic_number_5_2))
basic_number_5_2 ::= (([0-9] basic_number_5_2) | ([0-9]))
basic_number_6_2 ::= ("" | ([eE] basic_number_4_2 basic_number_5_2))
root_prop_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_2 root_prop_1_1)) (=([ \n\t]* "]"))
basic_number_7_2 ::= (("0") | ([1-9] [0-9]*)) (=(basic_number_3_2 basic_number_6_2))
triggered_tags_group ::= (("1>" root_0 "</function>") | ("2>" root_1 "</function>"))
triggered_tags_group_1 ::= ((">" root_2 "</function>"))
triggered_tags ::= TagDispatch(
  ("<function=f", triggered_tags_group),
  ("<function=g", triggered_tags_group_1),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
"#;

const EXPECTED_GRAMMAR_TEST_STRUCTURAL_TAG_BEFORE_OPTIMIZATION: &str = r#"basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_part_0 ::= (([ \n\t]* "," [ \n\t]* "\"arg2\"" [ \n\t]* ":" [ \n\t]* basic_integer))
root_0 ::= (("{" [ \n\t]* "\"arg1\"" [ \n\t]* ":" [ \n\t]* basic_string root_part_0 [ \n\t]* "}"))
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
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
basic_any_1 ::= ((basic_number_8) | (basic_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_number_8 ::= ((basic_number_1_1 basic_number_7_1 basic_number_3_1 basic_number_6_1))
basic_string_1 ::= (("\"" basic_string_sub_1))
basic_boolean_1 ::= (("true") | ("false"))
basic_null_1 ::= (("null"))
basic_array_2 ::= (("[" [ \n\t]* basic_any_1 basic_array_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_2 ::= (("{" [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_part_0_1 ::= (([ \n\t]* "," [ \n\t]* "\"arg2\"" [ \n\t]* ":" [ \n\t]* basic_integer_2))
root_1 ::= (("{" [ \n\t]* "\"arg1\"" [ \n\t]* ":" [ \n\t]* basic_string_1 root_part_0_1 [ \n\t]* "}"))
basic_integer_1_1 ::= ("" | ("-"))
basic_number_1_1 ::= ("" | ("-"))
basic_number_2_1 ::= (([0-9] basic_number_2_1) | ([0-9]))
basic_number_3_1 ::= ("" | ("." basic_number_2_1))
basic_number_4_1 ::= ("" | ([+\-]))
basic_number_5_1 ::= (([0-9] basic_number_5_1) | ([0-9]))
basic_number_6_1 ::= ("" | ([eE] basic_number_4_1 basic_number_5_1))
basic_array_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_1 basic_array_1_1))
basic_object_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1))
basic_number_7_1 ::= (("0") | ([1-9] [0-9]*))
basic_escape_2 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_2 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_2) | ("\\" basic_escape_2 basic_string_sub_2)) (=([ \n\t]* [,}\]:]))
basic_any_2 ::= ((basic_number_9) | (basic_string_2) | (basic_boolean_2) | (basic_null_2) | (basic_array_3) | (basic_object_3))
basic_integer_3 ::= (("0") | (basic_integer_1_2 [1-9] [0-9]*))
basic_number_9 ::= ((basic_number_1_2 basic_number_7_2 basic_number_3_2 basic_number_6_2))
basic_string_2 ::= (("\"" basic_string_sub_2))
basic_boolean_2 ::= (("true") | ("false"))
basic_null_2 ::= (("null"))
basic_array_3 ::= (("[" [ \n\t]* basic_any_2 basic_array_1_2 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_3 ::= (("{" [ \n\t]* basic_string_2 [ \n\t]* ":" [ \n\t]* basic_any_2 basic_object_1_2 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_prop_1 ::= (("[" [ \n\t]* basic_string_2 root_prop_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
root_part_0_2 ::= (([ \n\t]* "," [ \n\t]* "\"arg4\"" [ \n\t]* ":" [ \n\t]* root_prop_1))
root_2 ::= (("{" [ \n\t]* "\"arg3\"" [ \n\t]* ":" [ \n\t]* basic_number_9 root_part_0_2 [ \n\t]* "}"))
basic_integer_1_2 ::= ("" | ("-"))
basic_number_1_2 ::= ("" | ("-"))
basic_number_2_2 ::= (([0-9] basic_number_2_2) | ([0-9]))
basic_number_3_2 ::= ("" | ("." basic_number_2_2))
basic_number_4_2 ::= ("" | ([+\-]))
basic_number_5_2 ::= (([0-9] basic_number_5_2) | ([0-9]))
basic_number_6_2 ::= ("" | ([eE] basic_number_4_2 basic_number_5_2))
basic_array_1_2 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_2 basic_array_1_2))
basic_object_1_2 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_2 [ \n\t]* ":" [ \n\t]* basic_any_2 basic_object_1_2))
root_prop_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_2 root_prop_1_1))
basic_number_7_2 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("1>" root_0 "</function>") | ("2>" root_1 "</function>"))
triggered_tags_group_1 ::= ((">" root_2 "</function>"))
triggered_tags ::= TagDispatch(
  ("<function=f", triggered_tags_group),
  ("<function=g", triggered_tags_group_1),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
"#;

#[cfg(feature = "hf")]
fn get_masked_tokens_from_bitmask(bitmask: &[i32], vocab_size: usize) -> Vec<usize> {
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
fn get_stop_token_id(tokenizer_info: &TokenizerInfo) -> i32 {
    tokenizer_info.stop_token_ids().first().copied().unwrap_or(-1)
}

#[test]
#[serial]
fn test_utf8() {
    // Test utf8-encoded string with structural tags
    let schema = r#"{"type":"object","properties":{"arg1":{"type":"string"},"arg2":{"type":"integer"}},"required":["arg1","arg2"]}"#;
    let tags = vec![
        StructuralTagItem::new("，，", schema, "。"),
        StructuralTagItem::new("，！", schema, "。。"),
        StructuralTagItem::new("，，？", schema, "。。。"),
        StructuralTagItem::new("｜｜？", schema, "｜？｜"),
    ];
    let triggers = vec!["，", "｜｜"];

    let empty_vocab: Vec<&str> = vec![];
    let tok = TokenizerInfo::new(&empty_vocab, VocabType::RAW, &None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tok, 1, false, -1).unwrap();
    let compiled_grammar = compiler.compile_structural_tag(&tags, &triggers).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();

    let accepted_inputs = [
        r#"这是无用的内容，，{"arg1": "你好，世界！", "arg2": 0}。这是无用的内容"#,
        r#"这是无用的内容，！{"arg1": "こんにちは！", "arg2": 1}。。这是无用的内容"#,
        r#"这是无用的内容，，？{"arg1": "안녕하세요！", "arg2": 2}。。。这是无用的内容，！{"arg1": "안녕하세요！", "arg2": 3}。。"#,
        r#"这是无用的内容｜｜？{"arg1": "။စ်န, ်ပြ！", "arg2": 0}｜？｜｜｜？{"arg1": "။စ်န, ်ပြ", "arg2": 0}｜？｜"#,
    ];

    for input_str in accepted_inputs {
        matcher.reset();
        assert!(matcher.accept_string(input_str, false), "failed to accept: {input_str}");
        assert!(matcher.is_terminated(), "not terminated for: {input_str}");
    }
}

#[test]
#[serial]
fn test_structural_tag() {
    let schema1 = json!({
        "type": "object",
        "properties": {"arg1": {"type": "string"}, "arg2": {"type": "integer"}},
        "required": ["arg1", "arg2"]
    });
    let schema2 = json!({
        "type": "object",
        "properties": {"arg3": {"type": "number"}, "arg4": {"type": "array", "items": {"type": "string"}}},
        "required": ["arg3", "arg4"]
    });
    let structural_tag = json!({
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<function=f", "<function=g"],
            "tags": [
                {"begin": "<function=f1>", "content": {"type": "json_schema", "json_schema": schema1}, "end": "</function>"},
                {"begin": "<function=f2>", "content": {"type": "json_schema", "json_schema": schema1}, "end": "</function>"},
                {"begin": "<function=g>", "content": {"type": "json_schema", "json_schema": schema2}, "end": "</function>"}
            ]
        }
    });

    let grammar =
        Grammar::from_structural_tag(&structural_tag.to_string()).unwrap();
    assert_eq!(
        grammar.to_string_ebnf(),
        EXPECTED_GRAMMAR_TEST_STRUCTURAL_TAG_BEFORE_OPTIMIZATION
    );

    let accepted_inputs = [
        r#"<function=f1>{"arg1": "abc", "arg2": 1}</function>"#,
        r#"<function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>"#,
        r#"<function=f2>{"arg1": "abc", "arg2": 1}</function><function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>"#,
        r#"hhhh<function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>haha<function=f1>{"arg1": "abc", "arg2": 1}</function>123"#,
    ];
    for input_str in accepted_inputs {
        assert!(is_grammar_accept_string(&grammar, input_str));
    }
}

#[test]
#[serial]
fn test_structural_tag_compiler() {
    let schema1 = r#"{"type":"object","properties":{"arg1":{"type":"string"},"arg2":{"type":"integer"}},"required":["arg1","arg2"]}"#;
    let schema2 = r#"{"type":"object","properties":{"arg3":{"type":"number"},"arg4":{"type":"array","items":{"type":"string"}}},"required":["arg3","arg4"]}"#;
    let tags = vec![
        StructuralTagItem::new("<function=f1>", schema1, "</function>"),
        StructuralTagItem::new("<function=f2>", schema1, "</function>"),
        StructuralTagItem::new("<function=g>", schema2, "</function>"),
    ];
    // in real cases, we should use one trigger: "<function=" and dispatch to two tags
    // but here we use two triggers for testing such cases
    let triggers = vec!["<function=f", "<function=g"];

    let empty_vocab: Vec<&str> = vec![];
    let tok = TokenizerInfo::new(&empty_vocab, VocabType::RAW, &None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tok, 1, false, -1).unwrap();
    let compiled_grammar = compiler.compile_structural_tag(&tags, &triggers).unwrap();
    assert_eq!(
        compiled_grammar.grammar().to_string_ebnf(),
        EXPECTED_GRAMMAR_TEST_STRUCTURAL_TAG_AFTER_OPTIMIZATION
    );
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_structural_tag_mask_gen() {
    // Define schemas for the test
    let schema1 = r#"{"type":"object","properties":{"arg1":{"type":"string"},"arg2":{"type":"integer"}},"required":["arg1","arg2"]}"#;
    let schema2 = r#"{"type":"object","properties":{"arg3":{"type":"number"},"arg4":{"type":"array","items":{"type":"string"}}},"required":["arg3","arg4"]}"#;

    // Set up grammar from schemas
    let tags = vec![
        StructuralTagItem::new("<function=f>", schema1, "</function>"),
        StructuralTagItem::new("<function=g>", schema2, "</function>"),
    ];
    let triggers = vec!["<function=f", "<function=g"];

    // Set up tokenizer
    let tokenizer_id = "meta-llama/Llama-3.1-8B-Instruct";
    let tokenizer_path =
        test_utils::download_tokenizer_json(tokenizer_id).unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();
    let tokenizer_info = TokenizerInfo::from_huggingface(&tokenizer, None, None)
        .unwrap();

    // Compile grammar and create matcher
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let time_start = std::time::Instant::now();
    let compiled = compiler.compile_structural_tag(&tags, &triggers).unwrap();
    let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
    let time_end = time_start.elapsed();
    println!(
        "Time to compile grammar and init GrammarMatcher: {} us",
        time_end.as_micros()
    );

    // Test input string
    let accepted_input = concat!(
        r#"hhhh<function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>"#,
        r#"haha<function=f>{"arg1": "abc", "arg2": 1}</function>123"#
    );
    let dont_apply_mask_indices = vec![
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 66, 67, 68, 69, 70,
        71, 72, 73, 74, 75, 76, 77, 78, 119, 120, 121, 122,
    ];
    let input_bytes = accepted_input.as_bytes();

    // Set up token bitmask for validation
    let mut token_bitmask =
        xgrammar::allocate_token_bitmask(1, tokenizer_info.vocab_size());
    let (mut tensor, _shape, _strides) = create_bitmask_dltensor(
        &mut token_bitmask,
        1,
        tokenizer_info.vocab_size(),
    );

    // Process input character by character
    for (i, c) in input_bytes.iter().enumerate() {
        // 1. Test token bitmask generation
        let time_start = std::time::Instant::now();
        let need_apply = matcher.fill_next_token_bitmask(&mut tensor, 0, false);
        let time_end = time_start.elapsed();
        println!(
            "Time to fill_next_token_bitmask: {} us",
            time_end.as_micros()
        );
        assert_eq!(need_apply, !dont_apply_mask_indices.contains(&i));

        // 2. Verify token bitmask correctness
        let rejected_token_ids = get_masked_tokens_from_bitmask(
            &token_bitmask,
            tokenizer_info.vocab_size(),
        );
        // This checking does not support non-ascii characters for now
        let token_id_for_next_char =
            tokenizer.token_to_id(&(*c as char).to_string());
        if let Some(token_id) = token_id_for_next_char {
            assert!(!rejected_token_ids.contains(&(token_id as usize)));
        }

        // 3. Test character acceptance
        println!("Accepting char: {:?}", [*c]);
        let time_start = std::time::Instant::now();
        let s = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_ref(c))
        };
        assert!(matcher.accept_string(s, false));
        let time_end = time_start.elapsed();
        println!("Time to accept_token: {} us", time_end.as_micros());
    }

    // Final verification - check that EOS token is allowed
    let time_start = std::time::Instant::now();
    let need_apply = matcher.fill_next_token_bitmask(&mut tensor, 0, false);
    let time_end = time_start.elapsed();
    assert_eq!(need_apply, !dont_apply_mask_indices.contains(&input_bytes.len()));
    println!(
        "Time to fill_next_token_bitmask: {} us",
        time_end.as_micros()
    );
    let rejected_token_ids = get_masked_tokens_from_bitmask(
        &token_bitmask,
        tokenizer_info.vocab_size(),
    );
    let eos_id = get_stop_token_id(&tokenizer_info) as usize;
    assert!(!rejected_token_ids.contains(&eos_id));
}

#[test]
#[serial]
fn test_empty_tag_dispatch() {
    let grammar_str = r#"root ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)
"#;
    let grammar = Grammar::from_ebnf(grammar_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar, "any string"));
    assert!(is_grammar_accept_string(&grammar, ""));
    assert!(is_grammar_accept_string(&grammar, "好"));

    let grammar_with_stop_str_str = r#"root ::= TagDispatch(
  stop_eos=false,
  stop_str=("end"),
  loop_after_dispatch=true
)
"#;

    let grammar_with_stop_str =
        Grammar::from_ebnf(grammar_with_stop_str_str, "root").unwrap();
    assert!(is_grammar_accept_string(&grammar_with_stop_str, "any stringend"));
    assert!(is_grammar_accept_string(&grammar_with_stop_str, "end"));
    assert!(is_grammar_accept_string(&grammar_with_stop_str, "好end"));

    assert!(!is_grammar_accept_string(&grammar_with_stop_str, "aaa"));
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_utf8_structural_tag_begin_end() {
    let model = "deepseek-ai/DeepSeek-V3-0324";
    let tokenizer_path =
        test_utils::download_tokenizer_json(model).unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();
    let tokenizer_info = TokenizerInfo::from_huggingface(&tokenizer, None, None)
        .unwrap();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
    let structures = vec![StructuralTagItem::new(
        "<｜tool▁calls▁begin｜>",
        "{}",
        "<｜tool▁calls▁end｜>",
    )];
    let triggers = vec!["<｜tool▁calls▁begin｜>"];
    let _ = compiler.compile_structural_tag(&structures, &triggers).unwrap();
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_pressure_structural_tag() {
    let model = "meta-llama/Llama-3.1-8B-Instruct";
    let tokenizer_path =
        test_utils::download_tokenizer_json(model).unwrap();
    let tokenizer =
        tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();
    let _tokenizer_info = TokenizerInfo::from_huggingface(&tokenizer, None, None)
        .unwrap();
    let start = "start";
    let schema = r#"{"type":"object","properties":{"arg":{"type":"string"}}}"#;
    let end = "end";

    let mut handles = Vec::new();
    for i in 0..128usize {
        let start = start.to_string();
        let schema = schema.to_string();
        let end = end.to_string();
        let tokenizer_path = tokenizer_path.clone();
        handles.push(std::thread::spawn(move || {
            let tokenizer =
                tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();
            let tokenizer_info =
                TokenizerInfo::from_huggingface(&tokenizer, None, None).unwrap();
            let mut compiler =
                GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();
            let tag = StructuralTagItem::new(&start, &schema, &end);
            let triggers = vec![start.as_str()];
            let stag_compiled = compiler.compile_structural_tag(&[tag], &triggers).unwrap();
            let stag_grammar = stag_compiled.grammar();
            let start_grammar =
                Grammar::from_ebnf("root ::= [a-z] root | [a-z]", "root").unwrap();
            let mut grammar = start_grammar;
            for _ in 0..i {
                let start_grammar =
                    Grammar::from_ebnf("root ::= [a-z] root | [a-z]", "root").unwrap();
                grammar = Grammar::concat(&[grammar, start_grammar]);
            }
            let final_grammar = Grammar::concat(&[grammar, stag_grammar]);
            let _ = compiler.compile_grammar(&final_grammar).unwrap();
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}


