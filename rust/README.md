# xgrammar-rs

Rust bindings for [XGrammar](https://github.com/mlc-ai/xgrammar) - Efficient, Flexible and Portable Structured Generation.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
xgrammar = "0.1.31"
```

To enable Hugging Face tokenizers integration:

```toml
[dependencies]
xgrammar = { version = "0.1.31", features = ["hf"] }
```

## Usage

```rust
use xgrammar::{GrammarCompiler, GrammarMatcher, TokenizerInfo};

// Initialize tokenizer info (requires "hf" feature)
let tokenizer_info = TokenizerInfo::from_huggingface("meta-llama/Llama-2-7b-chat-hf", None).unwrap();

// Compile a grammar (e.g., from JSON Schema)
let compiler = GrammarCompiler::new(tokenizer_info);
let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
let grammar = compiler.compile_json_schema(schema).unwrap();

// Create a matcher
let mut matcher = GrammarMatcher::new(grammar);

// During generation loop...
// matcher.accept_token(token_id);
// matcher.fill_next_token_bitmask(&mut bitmask);
```

## Features

- **Efficient**: Near-zero overhead structured generation.
- **Flexible**: Supports JSON Schema, EBNF, Regex, and custom grammars.
- **Portable**: Works on CPU, GPU, and Apple Silicon.

## License

Apache-2.0
