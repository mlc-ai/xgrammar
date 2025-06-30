# JSON Generation

XGrammar enables efficient structured generation. One example structure is JSON and JSON Schema.
In this tutorial, we go over how to use XGrammar to ensure that an LLM's output is a
valid JSON, or adheres to a customized JSON schema.

We first go over how to use XGrammar in an LLM engine to achieve this in
[JSON Generation in LLM Engines](#json-generation-in-llm-engines), we then provide
an end-to-end JSON generation using XGrammar with HF `transformers` in
[Try out via HF Transformers](#try-out-via-hf-transformers).

## Install XGrammar

[XGrammar](../start/installation) is available via pip.
It is always recommended to install it in an isolated conda virtual environment.

## JSON Generation in LLM Engines

In this section, we see how to use XGrammar in an LLM engine to ensure that the output is
always a valid JSON.

All code snippets below are actual runnable code as we simulate the LLM generation.

First, import necessary libraries for the tutorial.

```python
import xgrammar as xgr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
```

Then, we extract tokenizer info from the LLM we are using with `xgr.TokenizerInfo`. With
the `tokenizer_info`, instantiate `xgr.GrammarCompiler` that will compiler a grammar of
your choice.

```python
# Get tokenizer info
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
# This can be larger than tokenizer.vocab_size due to paddings
full_vocab_size = config.vocab_size
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)

compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
```

For JSON generation, there are generally three options for compiling the grammar: using a built-in
JSON grammar, specify JSON schema with a Pydantic model, or from a JSON schema string. Pick one
one of the three below to run.

```python
# Option 1: Compile with a built-in JSON grammar
compiled_grammar: xgr.CompiledGrammar = compiler.compile_builtin_json_grammar()
```

```python
# Option 2: Compile with JSON schema from a pydantic model
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

compiled_grammar = compiler.compile_json_schema(Person)
```

```python
# Option 3: Compile with JSON schema from a JSON schema string
import json

person_schema = {
  "title": "Person",
  "type": "object",
  "properties": {
    "name": {
      "type": "string"
    },
    "age": {
      "type": "integer",
    }
  },
  "required": ["name", "age"]
}
compiled_grammar = compiler.compile_json_schema(json.dumps(person_schema))
```

With the compiled grammar, we can instantiate a `xgr.GrammarMatcher`, the main construct
we interact with that maintains the state of the structured generation. We also allocate a
bitmask that will be used to mask logits.

```python
# Instantiate grammar matcher and allocate the bitmask
matcher = xgr.GrammarMatcher(compiled_grammar)
token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
```

Now we simulate a single-request auto-regressive generation. See [Integration with LLM Engine](engine_integration.md)
for batched inference.

```python
# Here we simulate a valid sampled response
sim_sampled_response = '{"name": "xgrammar", "age": 0}<|end_of_text|>'
sim_sampled_token_ids = tokenizer.encode(sim_sampled_response, add_special_tokens=False)

# Each loop iteration is a simulated auto-regressive step
for i, sim_token_id in enumerate(sim_sampled_token_ids):
    # LLM inference to get logits, here we use randn to simulate.
    # logits is a tensor of shape (full_vocab_size,) on GPU
    # logits = LLM.inference()
    logits = torch.randn(full_vocab_size).cuda()

    # Apply bitmask to logits to mask invalid tokens
    matcher.fill_next_token_bitmask(token_bitmask)
    xgr.apply_token_bitmask_inplace(logits, token_bitmask.to(logits.device))

    # Sample next token
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    next_token_id = np.random.choice(list(range(full_vocab_size)), p=probs)

    # Accept token from matcher to update its state, so that the next bitmask
    # generated will enforce the next token to be generated. Assert to make
    # sure the token is indeed valid. Here we accept the simulated response
    # assert matcher.accept_token(next_token_id)
    assert matcher.accept_token(sim_token_id)

# Since we accepted a stop token `<|end_of_text|>`, we have terminated
assert matcher.is_terminated()

# Reset to be ready for the next auto-regressive generation
matcher.reset()
```

## Try out via HF Transformers

XGrammar can be easily integrate with HF transformers using a `LogitsProcessor`. Note that
this integration mainly aims for accessibility and may contain extra overhead.

First, instantiate a model, a tokenizer, and inputs.

```python
import xgrammar as xgr

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

device = "cuda"  # Or "cpu", etc.
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Introduce yourself in JSON with two fields: name and age."},
]
texts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(texts, return_tensors="pt").to(model.device)
```

Then construct a `GrammarCompiler` and compile the grammar.

```python
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
# Option 1: Compile with a built-in JSON grammar
# compiled_grammar = grammar_compiler.compile_builtin_json_grammar()
# Option 2: Compile with JSON schema from a pydantic model
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

compiled_grammar = grammar_compiler.compile_json_schema(Person)
```

Finally, use `LogitsProcessor` to generate with grammar.

```python
xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
generated_ids = model.generate(
    **model_inputs, max_new_tokens=512, logits_processor=[xgr_logits_processor]
)
generated_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
print(tokenizer.decode(generated_ids, skip_special_tokens=True))
```
