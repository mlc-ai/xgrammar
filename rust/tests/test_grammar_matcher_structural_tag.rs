mod test_utils;

use serial_test::serial;
use test_utils::*;

use xgrammar::{GrammarCompiler, GrammarMatcher, StructuralTagItem, TokenizerInfo, VocabType};

#[test]
#[serial]
fn test_structural_tag_utf8_accepts() {
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
    let cg = compiler.compile_structural_tag(&tags, &triggers).unwrap();
    let mut matcher = GrammarMatcher::new(&cg, None, true, -1).unwrap();

    let accepted_inputs = [
        r#"这是无用的内容，，{"arg1": "你好，世界！", "arg2": 0}。这是无用的内容"#,
        r#"这是无用的内容，！{"arg1": "こんにちは！", "arg2": 1}。。这是无用的内容"#,
        r#"这是无用的内容，，？{"arg1": "안녕하세요！", "arg2": 2}。。。这是无用的内容，！{"arg1": "안녕하세요！", "arg2": 3}。。"#,
        r#"这是无用的内容｜｜？{"arg1": "။စ်န, ်ပြ！", "arg2": 0}｜？｜｜｜？{"arg1": "။စ်န, ်ပြ", "arg2": 0}｜？｜"#,
    ];

    for s in accepted_inputs {
        matcher.reset();
        assert!(matcher.accept_string(s, false), "failed to accept: {s}");
        assert!(matcher.is_terminated(), "not terminated for: {s}");
    }
}

#[test]
#[serial]
fn test_structural_tag_compiled_accepts() {
    let schema1 = r#"{"type":"object","properties":{"arg1":{"type":"string"},"arg2":{"type":"integer"}},"required":["arg1","arg2"]}"#;
    let schema2 = r#"{"type":"object","properties":{"arg3":{"type":"number"},"arg4":{"type":"array","items":{"type":"string"}}},"required":["arg3","arg4"]}"#;
    let tags = vec![
        StructuralTagItem::new("<function=f1>", schema1, "</function>"),
        StructuralTagItem::new("<function=f2>", schema1, "</function>"),
        StructuralTagItem::new("<function=g>", schema2, "</function>"),
    ];
    let triggers = vec!["<function=f", "<function=g"];

    let empty_vocab: Vec<&str> = vec![];
    let tok = TokenizerInfo::new(&empty_vocab, VocabType::RAW, &None, false).unwrap();
    let mut compiler = GrammarCompiler::new(&tok, 1, false, -1).unwrap();
    let cg = compiler.compile_structural_tag(&tags, &triggers).unwrap();

    let mut matcher = GrammarMatcher::new(&cg, None, true, -1).unwrap();
    let accepted_inputs = [
        r#"<function=f1>{"arg1": "abc", "arg2": 1}</function>"#,
        r#"<function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>"#,
        r#"<function=f2>{"arg1": "abc", "arg2": 1}</function><function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>"#,
        r#"hhhh<function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>haha<function=f1>{"arg1": "abc", "arg2": 1}</function>123"#,
    ];

    for s in accepted_inputs {
        matcher.reset();
        assert!(matcher.accept_string(s, false), "failed to accept: {s}");
        assert!(matcher.is_terminated(), "not terminated for: {s}");
    }

    // Also validate the same behavior through the Grammar API.
    // (The compiled grammar holds an optimized Grammar instance.)
    let g = cg.grammar();
    for s in accepted_inputs {
        assert!(is_grammar_accept_string(&g, s), "grammar didn't accept: {s}");
    }
}


