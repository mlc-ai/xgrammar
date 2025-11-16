# Quick Start

This guide introduces how to use XGrammar with HuggingFace `transformers` in Python to generate
structured outputs. It focuses on JSON generation -- the most important use case of structured
generation. You should have already [installed XGrammar](installation).

## Preparation

Instantiate a model, a tokenizer, and inputs to the LLM.

```python
import xgrammar as xgr

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

device = "cuda"  # Or "cpu" if you don't have a GPU
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Introduce yourself in JSON briefly."},
]
texts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(texts, return_tensors="pt").to(model.device)
```

## Compile Grammar

Construct a `GrammarCompiler` and compile the grammar.

The grammar can be a built-in JSON grammar, a JSON schema string, or an EBNF string. EBNF provides
more flexibility for customization. See
[GBNF documentation](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) for
specification.

```python
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
compiled_grammar = grammar_compiler.compile_builtin_json_grammar()
# Other ways: provide a json schema string
# compiled_grammar = grammar_compiler.compile_json_schema(json_schema_string)
# Or provide an EBNF string
# compiled_grammar = grammar_compiler.compile_grammar(ebnf_string)
```

## Generate with grammar

Use logits_processor to generate with grammar.

```python
xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
generated_ids = model.generate(
    **model_inputs, max_new_tokens=512, logits_processor=[xgr_logits_processor]
)
generated_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
print(tokenizer.decode(generated_ids, skip_special_tokens=True))
```

## Notice: Generation Quality

When applying constrained decoding, it is recommended to **clearly describe the expected output structure in the prompt**. This is because constrained decoding only affects the sampling stage, but we want the LLM’s underlying probability distribution to already align with the structure as much as possible.

## What to Do Next

- Check out [JSON Generation Guide](../tutorials/ebnf_guided_generation.md) and other How-To guides for the detailed usage guide of XGrammar.
- Report any problem or ask any question: open new issues in our [GitHub repo](https://github.com/mlc-ai/xgrammar/issues).
