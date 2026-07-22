# Quick Start

XGrammar guarantees that the LLM output follows a specified structure through
[constrained decoding](constrained_decoding). It supports several ways to describe the structure:

- **[Structural Tag](../structural_tag/structural_tag)**: describe outputs that mix reasoning,
  free-form text, and tool calls. It supports tool calling for all common models (Llama, Qwen,
  DeepSeek, Kimi, OpenAI Harmony, etc.) via
  [built-in model styles](../structural_tag/tool_calling_and_reasoning).
- **[JSON Schema](../defining_structures/json_generation)**: constrain the output to be a valid
  JSON, or a JSON that conforms to a given schema.
- **[EBNF](../defining_structures/ebnf_grammar)**: describe arbitrary structures with a
  context-free grammar in the extended BNF format.
- **[Lark](../defining_structures/lark_grammar)**: describe arbitrary structures in the compact and
  readable Lark grammar language.

This guide walks through the most common use case: generating a valid JSON with HuggingFace
`transformers` in Python. You should have already [installed XGrammar](installation).

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
more flexibility for customization. See [EBNF Grammar](../defining_structures/ebnf_grammar) for
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

- To understand how constrained decoding works, see [Constrained Decoding](constrained_decoding).
- To learn the core concepts and APIs of XGrammar, see
  [Workflow of XGrammar](../using_xgrammar/workflow_of_xgrammar).
- To integrate XGrammar into an LLM engine, see
  [Integration with LLM Engine](../using_xgrammar/engine_integration).
- To constrain tool calling and reasoning outputs, see
  [Structural Tag](../structural_tag/structural_tag).
- Report any problem or ask any question: open new issues in our
  [GitHub repo](https://github.com/mlc-ai/xgrammar/issues).
