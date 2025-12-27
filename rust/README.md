<div align="center" id="top">

<img src="https://raw.githubusercontent.com/mlc-ai/xgrammar/main/assets/logo.svg" alt="logo" width="400" margin="10px"></img>

[![License](https://img.shields.io/badge/license-apache_2-blue)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/xgrammar-rs)](https://crates.io/crates/xgrammar-rs)
[![Documentation](https://docs.rs/xgrammar-rs/badge.svg)](https://docs.rs/xgrammar-rs)

**Efficient, Flexible and Portable Structured Generation for Rust**

Rust bindings for [XGrammar](https://github.com/mlc-ai/xgrammar)

</div>

## Overview

XGrammar is an open-source library for efficient, flexible, and portable structured generation.

It leverages constrained decoding to ensure **100% structural correctness** of the output. It supports general context-free grammar to enable a broad range of structures, including **JSON**, **regex**, **custom context-free grammar**, etc.

XGrammar uses careful optimizations to achieve extremely low overhead in structured generation. It has achieved **near-zero overhead** in JSON generation, making it one of the fastest structured generation engines available.

XGrammar features **universal deployment**. It supports:
* **Platforms**: Linux, macOS, Windows
* **Hardware**: CPU, NVIDIA GPU, AMD GPU, Apple Silicon, TPU, etc.
* **Models**: Qwen, Llama, DeepSeek, Phi, Gemma, etc.

## Features

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
xgrammar-rs = "0.1"
```

For HuggingFace tokenizer support:

```toml
[dependencies]
xgrammar-rs = { version = "0.1", features = ["hf"] } 
```

## Quick Start

### JSON Schema Generation

```rust
use xgrammar::{Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo, VocabType};

fn main() -> Result<(), String> {
    // Define your JSON schema
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }"#;

    // Create grammar from JSON schema
    let grammar = Grammar::from_json_schema(
        schema,
        true,  // any_whitespace
        None,  // indent
        Some((",", ":")),  // separators
        true,  // strict_mode
        None,  // max_whitespace_cnt
        false, // print_converted_ebnf
    )?;

    // Create tokenizer info (example with empty vocab)
    let vocab: Vec<&str> = vec![];
    let tokenizer_info = TokenizerInfo::new(&vocab, VocabType::RAW, &None, false)?;

    // Compile grammar
    let mut compiler = GrammarCompiler::new(&tokenizer_info, 8, true, -1)?;
    let compiled_grammar = compiler.compile_grammar(&grammar)?;

    // Create matcher
    let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1)?;

    // Use the matcher to validate strings
    assert!(matcher.accept_string(r#"{"name":"John","age":30}"#, false));
    assert!(matcher.is_terminated());
    
    Ok(())
}
```

### EBNF Grammar

```rust
use xgrammar::Grammar;

let ebnf = r#"
root ::= expression
expression ::= term ("+" term | "-" term)*
term ::= factor ("*" factor | "/" factor)*
factor ::= number | "(" expression ")"
number ::= [0-9]+
"#;

let grammar = Grammar::from_ebnf(ebnf, "root")?;
```

### Regular Expression

```rust
use xgrammar::Grammar;

let regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}";
let grammar = Grammar::from_regex(regex, false)?;
```

### With HuggingFace Tokenizers (requires `hf` feature)

```rust
use xgrammar::{Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo, allocate_token_bitmask};

// Load tokenizer from HuggingFace
let tokenizer = tokenizers::Tokenizer::from_file("tokenizer.json")?;
let tokenizer_info = TokenizerInfo::from_huggingface(&tokenizer, None, None)?;

// Create and compile grammar
let grammar = Grammar::builtin_json_grammar();
let mut compiler = GrammarCompiler::new(&tokenizer_info, 8, true, -1)?;
let compiled_grammar = compiler.compile_grammar(&grammar)?;

// Create matcher and use for token-level generation
let mut matcher = GrammarMatcher::new(&compiled_grammar, None, true, -1)?;

// Allocate token bitmask for batch generation
let mut bitmask_data = allocate_token_bitmask(1, tokenizer_info.vocab_size());

// For string-based generation (simpler approach)
assert!(matcher.accept_string(r#"{"key":"value"}"#, false));
assert!(matcher.is_terminated());
```

## API Documentation

For detailed API documentation, visit [docs.rs/xgrammar-rs](https://docs.rs/xgrammar-rs).

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
