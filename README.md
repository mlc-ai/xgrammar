## XGrammar

Cross-platform Near-zero Overhead Grammar-guided Generation for LLMs

- G1: Universal: support any common tokenizer, and common grammar
- G2: Efficient: Grammar should not cause additional burden for generation
- G3: Cross-platform: pure C++ impl, portable for every platform, construct E2E pipeline on every platform
- G4: Easy to understand and maintain

This project is under active development.

### Compile and Install

```bash
# install requirements
sudo apt install cmake
python3 -m pip install ninja pybind11 torch

# build XGrammar core and Python bindings
# see scripts/config.cmake for configuration options
mkdir build
cd build
# specify your own CUDA architecture
cmake .. -G Ninja -DXGRAMMAR_CUDA_ARCHITECTURES=89
ninja

# install Python package
cd ../python
python3 -m pip install .

# optional: add the python directory to PATH
echo "export PATH=\$PATH:$(pwd)" >> ~/.bashrc
```

### Python Usage Guide

#### Step 1:Construction of grammar

```python
from xgrammar import BNFGrammar, BuiltinGrammar
from pydantic import BaseModel

# Method 1: provide a GBNF grammar string
# For specification, see https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
gbnf_grammar = """
root  ::= (expr "=" term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= num | "(" expr ")"
num   ::= [0-9]+
"""

gbnf_grammar = BNFGrammar(gbnf_grammar)

# Method 2: unconstrained JSON
json_grammar = BuiltinGrammar.json()

# Method 3: provide a Pydantic model
class Person(BaseModel):
    name: str
    age: int
json_schema_pydantic = BuiltinGrammar.json_schema(Person)

# Method 4: provide a JSON schema string
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
json_schema_str = BuiltinGrammar.json_schema(json.dumps(person_schema))
```

#### Step 2: Compiling grammars

```python
from xgrammar import TokenizerInfo, CachedGrammarCompiler, CompiledGrammar, GrammarMatcher
from transformers import AutoTokenizer

# 1. Convert huggingface tokenizer to TokenizerInfo (once per model)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
```

Method 1: Use CachedGrammarCompiler to avoid compile grammar multiple times
```python
# 2. Construct CachedGrammarCompiler (once per model)
compiler = CachedGrammarCompiler(tokenizer_info)

# 3. Fetch CompiledGrammar and construct GrammarMatcher (once per request)
compiled_grammar = compiler.compile_json_schema(json_schema_str)
matcher = GrammarMatcher(compiled_grammar)
```

Method 2: Compile grammar directly
```python
# 2. Construct CompiledGrammar directly (once per grammar)
compiled_grammar = CompiledGrammar(grammar, tokenizer_info)

# 3. Construct GrammarMatcher (once per request)
matcher = GrammarMatcher(compiled_grammar)
```

#### Step 3: Grammar-guided generation

For single-batch generation:
```python
import torch

token_bitmask = GrammarMatcher.allocate_token_bitmask(matcher.vocab_size)
while True:
    logits = LLM.inference() # logits is a tensor of shape (vocab_size,) on GPU
    matcher.fill_next_token_bitmask(logits, token_bitmask)
    GrammarMatcher.apply_token_bitmask_inplace(logits, token_bitmask)

    prob = torch.softmax(logits, dim=-1) # get probability from logits
    next_token_id = Sampler.sample(logits) # use your own sampler

    matcher.accept_token(next_token_id)
    if matcher.is_terminated(): # or your own termination condition
        break
```

For multi-batch generation:
```python
import torch

matchers: List[GrammarMatcher] # The grammar matcher for every request
token_bitmasks = GrammarMatcher.allocate_token_bitmask(matchers[0].vocab_size, batch_size)
while True:
    logits = LLM.inference() # logits is a tensor of shape (batch_size, vocab_size) on GPU
    # This for loop is parallelizable using threading.Thread. But estimate the overhead in your
    # engine.
    for i in range(len(matchers)):
        matchers[i].fill_next_token_bitmask(token_bitmasks, i)
    GrammarMatcher.apply_token_bitmask_inplace(logits, token_bitmasks)

    prob = torch.softmax(logits, dim=-1) # get probability from logits
    next_token_ids = Sampler.sample(logits) # use your own sampler

    for i in range(len(matchers)):
        matchers[i].accept_token(next_token_ids[i])
        if matchers[i].is_terminated(): # or your own termination condition
            requests[i].terminate()
```
