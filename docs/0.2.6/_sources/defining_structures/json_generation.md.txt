# JSON Generation

One example structure XGrammar supports is JSON and JSON Schema. In this tutorial, we go over how
to use XGrammar to ensure that an LLM's output is a valid JSON, or adheres to a customized JSON
schema.

First, construct a [`xgr.GrammarCompiler`](xgrammar.GrammarCompiler) from the tokenizer info of
the model (see [Workflow of XGrammar](../start/workflow_of_xgrammar.md) for details).

```python
import xgrammar as xgr
from transformers import AutoTokenizer, AutoConfig

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
JSON grammar, specify JSON schema with a Pydantic model, or from a JSON schema string. Pick
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

With the compiled grammar, we can instantiate a [`xgr.GrammarMatcher`](xgrammar.GrammarMatcher)
and generate the token masks in the generation loop.

```python
matcher = xgr.GrammarMatcher(compiled_grammar)
token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
```

## Next Steps

- For the complete generation loop and batched inference, see
  [Integration with LLM Engine](../using_xgrammar/engine_integration.md).
- For an end-to-end runnable example with HF `transformers`, see
  [Quick Start](../start/quick_start.md).
