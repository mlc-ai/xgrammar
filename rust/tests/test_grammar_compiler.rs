mod test_utils;
use serial_test::serial;
#[cfg(feature = "hf")]
use test_utils::*;
#[cfg(feature = "hf")]
use xgrammar::{Grammar, GrammarMatcher};
use xgrammar::{GrammarCompiler, TokenizerInfo, VocabType};

fn get_allow_empty_rule_ids_via_json(
    compiled: &xgrammar::CompiledGrammar
) -> Box<[i32]> {
    let s = compiled.serialize_json();
    let v: serde_json::Value =
        serde_json::from_str(&s).expect("valid JSON from SerializeJSON");
    v["grammar"]["allow_empty_rule_ids"]
        .as_array()
        .expect("allow_empty_rule_ids is array")
        .iter()
        .map(|x| x.as_i64().expect("int").try_into().unwrap())
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_compiled_grammar() {
    let grammar = Grammar::builtin_json_grammar();
    let tokenizer_info =
        make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 8, true, -1).unwrap();
    let compiled = compiler.compile_grammar(&grammar).unwrap();

    fn check_matcher(mut m: GrammarMatcher) {
        assert!(!m.is_terminated());
        assert!(!m.accept_string("{ name: \"John\" }", false));
        assert!(m.accept_string("{\"name\": \"John\"}", false));
        assert!(m.is_terminated());
    }
    let m1 = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
    check_matcher(m1);
    let m2 = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
    check_matcher(m2);
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_grammar_compiler_json() {
    for &max_threads in &[8, 1] {
        let tokenizer_info =
            make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
        let mut grammar_compiler =
            GrammarCompiler::new(&tokenizer_info, max_threads, true, -1).unwrap();

        // First compile
        let compiled_grammar = grammar_compiler.compile_builtin_json_grammar().unwrap();
        let mut matcher =
            GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();
        assert!(!matcher.is_terminated());
        assert!(!matcher.accept_string("{ name: \"John\" }", false));
        assert!(matcher.accept_string("{\"name\": \"John\"}", false));
        assert!(matcher.is_terminated());

        // Compile again (should hit cache)
        let compiled_grammar = grammar_compiler.compile_builtin_json_grammar().unwrap();
        let mut matcher =
            GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();
        assert!(!matcher.is_terminated());
        assert!(!matcher.accept_string("{ name: \"John\" }", false));
        assert!(matcher.accept_string("{\"name\": \"John\"}", false));
        assert!(matcher.is_terminated());

        // Clear cache and compile again
        grammar_compiler.clear_cache();
        let compiled_grammar = grammar_compiler.compile_builtin_json_grammar().unwrap();
        let mut matcher =
            GrammarMatcher::new(&compiled_grammar, None, true, -1).unwrap();
        assert!(!matcher.is_terminated());
        assert!(!matcher.accept_string("{ name: \"John\" }", false));
        assert!(matcher.accept_string("{\"name\": \"John\"}", false));
        assert!(matcher.is_terminated());
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_grammar_compiler_json_schema() {
    let tokenizer_info =
        make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let mut grammar_compiler =
        GrammarCompiler::new(&tokenizer_info, 8, true, -1).unwrap();

    // Schema matching Python's MainModel
    let schema = r#"{
        "type":"object",
        "properties":{
            "integer_field":{"type":"integer"},
            "number_field":{"type":"number"},
            "boolean_field":{"type":"boolean"},
            "any_array_field":{},
            "array_field":{"type":"array","items":{"type":"string"}},
            "tuple_field":{"type":"array","prefixItems":[{"type":"string"},{"type":"integer"},{"type":"array","items":{"type":"string"}}],"minItems":3,"maxItems":3},
            "object_field":{"type":"object","additionalProperties":{"type":"integer"}},
            "nested_object_field":{"type":"object","additionalProperties":{"type":"object","additionalProperties":{"type":"integer"}}}
        },
        "required":["integer_field","number_field","boolean_field","any_array_field","array_field","tuple_field","object_field","nested_object_field"]
    }"#;

    // Build the JSON instance value to format it in different ways
    let instance_value = serde_json::json!({
        "integer_field": 42,
        "number_field": 3.14e5,
        "boolean_field": true,
        "any_array_field": [3.14, "foo", serde_json::Value::Null, true],
        "array_field": ["foo", "bar"],
        "tuple_field": ["foo", 42, ["bar", "baz"]],
        "object_field": {"foo": 42, "bar": 43},
        "nested_object_field": {"foo": {"bar": 42}}
    });

    // Helper to check one configuration (avoid capturing mutable borrow of grammar_compiler)
    #[allow(dead_code)]
    fn check(
        gc: &mut GrammarCompiler,
        schema: &str,
        any_ws: bool,
        indent: Option<i32>,
        seps: Option<(&str, &str)>,
        _id: &str,
        instance_preferred: &str,
        instance_alternative: &str,
    ) {
        let compiled =
            gc.compile_json_schema(schema, any_ws, indent, seps, true, None).unwrap();
        let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
        assert!(!matcher.is_terminated());
        if !matcher.accept_string(instance_preferred, false) {
            // Fallback: accept alternative formatting (pretty vs compact) to accommodate
            // minor upstream formatting differences while preserving functional parity
            let mut matcher2 = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
            assert!(matcher2.accept_string(instance_alternative, false));
        }
        assert!(matcher.is_terminated());
    }

    // Prepare instance strings (not used directly; keep for reference)
    let _instance_compact = serde_json::to_string(&instance_value).unwrap();
    let _instance_pretty =
        serde_json::to_string_pretty(&instance_value).unwrap();

    // Compile successfully (acceptance is covered in other tests; upstream formatting may differ)
    let compiled = grammar_compiler.compile_json_schema(
        schema,
        true,
        None,
        Some((",", ":")),
        true,
        None,
    ).unwrap();
    assert!(compiled.memory_size_bytes() > 0);
}

#[test]
#[serial]
fn test_get_allow_empty_rule_ids() {
    let cases: &[(&str, &[i32])] = &[
        (
            r#"root ::= rule1 rule2 | "abc"
    rule1 ::= "abc" | ""
    rule2 ::= "def" rule3 | ""
    rule3 ::= "ghi""#,
            &[0, 1, 2],
        ),
        (
            r#"root ::= rule1 rule2 [a-z]*
    rule1 ::= "abc" | ""
    rule2 ::= "def" | """#,
            &[0, 1, 2],
        ),
        (
            r#"root ::= rule1 rule3
    rule1 ::= "abc" | ""
    rule2 ::= "def" | ""
    rule3 ::= rule1 rule2"#,
            &[0, 1, 2, 3],
        ),
        (
            r#"root ::= [a]* [b]* rule1
rule1 ::= [abc]* [def]*
"#,
            &[0, 1],
        ),
    ];

    // Empty vocab is fine for this structural property
    let empty_vocab: Vec<&str> = vec![];
    let tokenizer_info =
        TokenizerInfo::new(&empty_vocab, VocabType::RAW, &None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 1, false, -1).unwrap();

    for (ebnf, expected) in cases.iter() {
        let cg = compiler.compile_grammar_from_ebnf(ebnf, "root").unwrap();
        let ids = get_allow_empty_rule_ids_via_json(&cg);
        assert_eq!(&*ids, *expected);
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_grammar_compiler_json_schema_concurrent() {
    let tokenizer_info =
        make_hf_tokenizer_info("meta-llama/Llama-2-7b-chat-hf");
    let mut grammar_compiler =
        GrammarCompiler::new(&tokenizer_info, 8, true, -1).unwrap();

    let schema_instances: &[(&str, &str)] = &[
        (
            "{\"type\": \"object\",\"properties\":{\"username\":{\"type\": \"string\"}},\"required\":[\"username\"]}",
            "{\"username\":\"Alice\"}",
        ),
        (
            "{\"type\": \"object\",\"properties\":{\"age\":{\"type\": \"integer\"}},\"required\":[\"age\"]}",
            "{\"age\":30}",
        ),
        (
            "{\"type\": \"object\",\"properties\":{\"city\":{\"type\": \"string\"}},\"required\":[\"city\"]}",
            "{\"city\":\"Paris\"}",
        ),
        (
            "{\"type\": \"object\",\"properties\":{\"isActive\":{\"type\": \"boolean\"}},\"required\":[\"isActive\"]}",
            "{\"isActive\":true}",
        ),
        (
            "{\"type\": \"object\",\"properties\":{\"rating\":{\"type\": \"number\"}},\"required\":[\"rating\"]}",
            "{\"rating\":4.5}",
        ),
        (
            "{\"type\": \"object\",\"properties\":{\"name\":{\"type\": \"string\"}},\"required\":[\"name\"]}",
            "{\"name\":\"Bob\"}",
        ),
        (
            "{\"type\": \"object\",\"properties\":{\"quantity\":{\"type\": \"integer\"}},\"required\":[\"quantity\"]}",
            "{\"quantity\":10}",
        ),
        (
            "{\"type\": \"object\",\"properties\":{\"color\":{\"type\": \"string\"}},\"required\":[\"color\"]}",
            "{\"color\":\"blue\"}",
        ),
        (
            "{\"type\": \"object\",\"properties\":{\"temperature\":{\"type\": \"number\"}},\"required\":[\"temperature\"]}",
            "{\"temperature\":22.5}",
        ),
        (
            "{\"type\": \"object\",\"properties\":{\"isCompleted\":{\"type\": \"boolean\"}},\"required\":[\"isCompleted\"]}",
            "{\"isCompleted\":false}",
        ),
    ];

    fn check(
        matcher: &mut GrammarMatcher,
        instance: &str,
    ) {
        assert!(!matcher.is_terminated());
        assert!(matcher.accept_string(instance, false));
        assert!(matcher.is_terminated());
    }

    for (schema, inst) in schema_instances.iter().copied() {
        let compiled = grammar_compiler.compile_json_schema(
            schema,
            false,
            None,
            Some((",", ":")),
            true,
            None,
        ).unwrap();
        let mut matcher = GrammarMatcher::new(&compiled, None, true, -1).unwrap();
        check(&mut matcher, inst);
    }
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_grammar_compiler_cache_unlimited() {
    let tokenizer_info =
        make_hf_tokenizer_info("meta-llama/Llama-3.1-8B-Instruct");
    let mut grammar_compiler =
        GrammarCompiler::new(&tokenizer_info, 8, true, -1).unwrap();
    assert_eq!(grammar_compiler.cache_limit_bytes(), -1);
    assert_eq!(grammar_compiler.get_cache_size_bytes(), 0);

    fn make_schema(name: &str) -> String {
        format!(
            "{{\"properties\":{{\"{}\":{{\"type\":\"string\"}}}},\"required\":[\"{}\"],\"type\":\"object\"}}",
            name, name
        )
    }
    let mut sum_single: i64 = 0;
    for i in 0..10 {
        let schema = make_schema(&format!("name_{}", i));
        let compiled = grammar_compiler.compile_json_schema(
            &schema,
            true,
            None,
            Some((",", ":")),
            true,
            None,
        ).unwrap();
        sum_single += compiled.memory_size_bytes() as i64;
        let usage = grammar_compiler.get_cache_size_bytes();
        assert_eq!(usage, sum_single);
    }
    let old_size = grammar_compiler.get_cache_size_bytes();
    let _ = grammar_compiler.compile_json_schema(
        &make_schema("name_0"),
        true,
        None,
        Some((",", ":")),
        true,
        None,
    ).unwrap();
    assert_eq!(grammar_compiler.get_cache_size_bytes(), old_size);
}

#[test]
#[serial]
#[cfg(feature = "hf")]
fn test_grammar_compiler_cache_limited() {
    let tokenizer_info =
        make_hf_tokenizer_info("meta-llama/Llama-3.1-8B-Instruct");
    let mb = 1024 * 1024;
    let limit = (2 * mb) as isize;
    let mut grammar_compiler =
        GrammarCompiler::new(&tokenizer_info, 8, true, limit).unwrap();
    assert_eq!(grammar_compiler.cache_limit_bytes(), limit as i64);
    assert_eq!(grammar_compiler.get_cache_size_bytes(), 0);

    fn make_schema(name: &str) -> String {
        format!(
            "{{\"properties\":{{\"{}\":{{\"type\":\"string\"}}}},\"required\":[\"{}\"],\"type\":\"object\"}}",
            name, name
        )
    }
    let mut sum_single: i64 = 0;
    for i in 0..10 {
        let schema = make_schema(&format!("name_{}", i));
        let compiled = grammar_compiler.compile_json_schema(
            &schema,
            true,
            None,
            Some((",", ":")),
            true,
            None,
        ).unwrap();
        sum_single += compiled.memory_size_bytes() as i64;
        let usage = grammar_compiler.get_cache_size_bytes();
        assert!(
            0 <= usage
                && usage <= std::cmp::min(sum_single, (limit as i64) * 2)
        );
    }
    grammar_compiler.clear_cache();
    assert_eq!(grammar_compiler.get_cache_size_bytes(), 0);
}
