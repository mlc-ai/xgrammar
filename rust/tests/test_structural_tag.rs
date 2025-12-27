mod test_utils;

use serial_test::serial;
use xgrammar::{Grammar, StructuralTagItem};

#[test]
#[serial]
fn test_structural_tag_grammar_print_and_accept() {
    // Schema1: {arg1: string, arg2: integer}
    let schema1 = r#"{"type":"object","properties":{"arg1":{"type":"string"},"arg2":{"type":"integer"}},"required":["arg1","arg2"]}"#;
    // Schema2: {arg3: number, arg4: array of string}
    let schema2 = r#"{"type":"object","properties":{"arg3":{"type":"number"},"arg4":{"type":"array","items":{"type":"string"}}},"required":["arg3","arg4"]}"#;

    let tags = vec![
        StructuralTagItem {
            begin: "<function=f".into(),
            schema: schema1.into(),
            end: "</function>".into(),
        },
        StructuralTagItem {
            begin: "<function=g".into(),
            schema: schema2.into(),
            end: "</function>".into(),
        },
    ];
    let triggers = vec!["<function=f", "<function=g"];

    // Note: from_structural_tag has been removed in xgrammar 0.1.26
    // Use GrammarCompiler::compile_structural_tag instead
    let tok = xgrammar::TokenizerInfo::new(
        &[""],
        xgrammar::VocabType::RAW,
        &None,
        false,
    ).unwrap();
    let mut compiler = xgrammar::GrammarCompiler::new(&tok, 1, false, -1).unwrap();
    let compiled_grammar = compiler.compile_structural_tag(&tags, &triggers).unwrap();
    // Basic smoke check: ensure it compiled successfully
    assert!(compiled_grammar.memory_size_bytes() > 0);
}

#[test]
#[serial]
fn test_empty_tag_dispatch_accepts_any() {
    let ebnf = r#"root ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)
"#;
    let g = Grammar::from_ebnf(ebnf, "root").unwrap();
    let empty_vocab: Vec<&str> = vec![];
    let tok = xgrammar::TokenizerInfo::new(
        &empty_vocab,
        xgrammar::VocabType::RAW,
        &None,
        false,
    ).unwrap();
    let mut compiler = xgrammar::GrammarCompiler::new(&tok, 1, false, -1).unwrap();
    let cg = compiler.compile_grammar(&g).unwrap();
    let mut m = xgrammar::GrammarMatcher::new(&cg, None, true, -1).unwrap();
    assert!(m.accept_string("any string", false));
    assert!(m.is_terminated());
}
