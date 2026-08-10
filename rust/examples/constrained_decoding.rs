//! End-to-end constrained decoding against a JSON schema, using a small
//! hand-written vocabulary and a mock sampler.
//!
//! Run from the repository checkout (after building `xgrammar_bindings`):
//!
//! ```text
//! cargo run --example constrained_decoding
//! ```

use xgrammar::{
    GrammarCompiler, GrammarMatcher, JsonSchemaOptions, TokenBitmask, TokenizerInfo,
    TokenizerInfoOptions, VocabType,
};

/// Greedy "sampler": picks the allowed token the mock model likes most.
fn sample(logits: &[f32]) -> i64 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i64)
        .expect("non-empty logits")
}

fn main() -> xgrammar::Result<()> {
    // In this repository checkout the freshly built library is next to the
    // Python package; outside of it, set XGRAMMAR_BINDINGS_LIB instead.
    let _ = xgrammar::load_library(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../python/xgrammar/libxgrammar_bindings.so"
    ));

    // A byte-level toy vocabulary: token 0 is EOS, the rest are ASCII bytes.
    let mut vocab: Vec<Vec<u8>> = vec![b"</s>".to_vec()];
    vocab.extend((0x20u8..0x7f).map(|b| vec![b]));
    let tokenizer_info = TokenizerInfo::new(
        &vocab,
        VocabType::Raw,
        &TokenizerInfoOptions {
            stop_token_ids: Some(vec![0]),
            ..Default::default()
        },
    )?;

    let schema = r#"{
        "type": "object",
        "properties": {"temperature": {"type": "integer"}},
        "required": ["temperature"]
    }"#;
    let compiler = GrammarCompiler::new(&tokenizer_info)?;
    let compiled = compiler.compile_json_schema(schema, &JsonSchemaOptions::default())?;

    let mut matcher = GrammarMatcher::new(&compiled)?;
    let vocab_size = tokenizer_info.vocab_size()?;
    let mut bitmask = TokenBitmask::new(vocab_size);
    let mut output = Vec::new();

    let digit_token = |d: u8| (d - 0x20) as usize + 1;
    let mut wrote_digit = false;
    while !matcher.is_terminated()? {
        // The mock model wants to write a single "7" and then stop; the
        // grammar forces the JSON structure around it.
        let mut logits = vec![0.0f32; vocab_size];
        logits[0] = 1.0; // EOS, when the grammar allows it
        for d in b'0'..=b'9' {
            logits[digit_token(d)] = if wrote_digit { -1.0 } else { -2.0 };
        }
        if !wrote_digit {
            logits[digit_token(b'7')] = 2.0;
        }

        bitmask.reset();
        if matcher.fill_next_token_bitmask(&mut bitmask, 0)? {
            bitmask.apply_to_logits(&mut logits, 0);
        }
        let token_id = sample(&logits);
        assert!(matcher.accept_token(token_id)?);
        if token_id == digit_token(b'7') as i64 {
            wrote_digit = true;
        }
        output.push(token_id);
        assert!(output.len() < 200, "runaway generation");
    }

    let text: Vec<u8> = output
        .iter()
        .filter(|&&t| t != 0)
        .map(|&t| (t as u8 - 1) + 0x20)
        .collect();
    println!("generated: {}", String::from_utf8_lossy(&text));
    Ok(())
}
