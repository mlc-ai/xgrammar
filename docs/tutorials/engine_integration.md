# Integration with LLM Engine

XGrammar enables efficient structured generation. In this tutorial, we go over the key components
of XGrammar and how to integrate XGrammar into an LLM engine.

We first lay out the concepts in [High-Level Flow](#high-level-flow).
We then demonstrate how XGrammar enables
[Structured Generation for Batched Inference](#structured-generation-for-batched-inference).

The code snippets below are actual runnable code as we simulate the LLM generation.

## Install XGrammar

[XGrammar](../start/installation) is available via pip.
It is always recommended to install it in an isolated conda virtual environment.

## High-Level Flow

In this section, we go over the key components of XGrammar when integrating it into an LLM engine
for structured generation.

First, import necessary libraries for the tutorial.

```python
import xgrammar as xgr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
```

### xgr.TokenizerInfo

`xgr.TokenizerInfo` is a per-model construct that encapsulates tokenizer information, including
all its vocabulary. There are several ways of instantiating it, and the most convenient way
is using an `AutoTokenizer`. Note that for some models, `AutoConfig.vocab_size` can be larger
than `AutoTokenizer.vocab_size` due to paddings, with the former being the shape of the model's
logits. To be safe, always pass in the former when instantiating `xgr.TokenizerInfo`.

```python
# Get tokenizer info
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
# This can be larger than tokenizer.vocab_size due to paddings
full_vocab_size = config.vocab_size
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)
```

### xgr.GrammarCompiler

With an `xgr.TokenizerInfo`, we can instantiate an `xgr.GrammarCompiler`. This is a construct
that compiles a grammar according to the model's tokenizer info. Therefore, for each model, you
can use the same `xgr.GrammarCompiler` persistently, as it can compile different grammars for
the same `xgr.TokenizerInfo`. Note that the `compiler` behavior can be configured with
`max_threads` for multithreading, and `enable_cache` (defaults to true) for caching
compiled grammars.

```python
compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
```

### xgr.CompiledGrammar

Then, using the `xgr.GrammarCompiler`, we can compile a grammar, with the result being an
`xgr.CompiledGrammar`. Here we use a built-in JSON grammar. For other grammars, see
[JSON Generation](json_generation.md) and [EBNF-Guided Generation](ebnf_guided_generation.md).
Every thing we have seen up to now are per-model (rather than per-generation).

```python
compiled_grammar: xgr.CompiledGrammar = compiler.compile_builtin_json_grammar()
```

### xgr.GrammarMatcher

With the compiled grammar, we can instantiate a `xgr.GrammarMatcher`. It is the main construct
an LLM engine interacts with that maintains the state of the structured generation. Note that
each request should have its own `xgr.GrammarMatcher` since each has a different generation state,
as we will see in [Structured Generation for Batched Inference](#structured-generation-for-batched-inference).

```python
# Instantiate grammar matcher with the compiled grammar
matcher = xgr.GrammarMatcher(compiled_grammar)
```

### Bitmasking Logits in Auto-regressive Generation

Now we simulate a single-request auto-regressive generation. See later section for
[Structured Generation for Batched Inference](#structured-generation-for-batched-inference).

First, we pre-allocate a token bitmask with `xgr.allocate_token_bitmask()`,
which is essentially a `torch.Tensor` of shape `(batch_size, vocab_size)`. You can also
use your own implementation for allocating a bitmask.

In each auto-regressive step, we fill the token bitmask according to the current state
of the matcher with `xgr.GrammarMatcher.fill_next_token_bitmask()`. Then, we apply the bitmask
into the model's logits with `xgr.apply_token_bitmask_inplace()`, which calls a CUDA kernel
if `logits` is on CUDA (recommended), otherwise a CPU implementation.

After masking, the logits for illegal tokens are set to negative infinity, so that
we will never sample them. After sampling the token, update the `xgr.GrammarMatcher`'s state with
`xgr.GrammarMatcher.accept_token()`. Finally, use  `xgr.GrammarMatcher.reset()` to prepare
for the next generation.

```python
# Here we simulate a valid sampled response
sim_sampled_response = '{ "library": "xgrammar" }<|end_of_text|>'
sim_sampled_token_ids = tokenizer.encode(sim_sampled_response, add_special_tokens=False)

# Allocate a token bitmask
token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

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

## Structured Generation for Batched Inference

The code snippets above assume a single request generation.
This section demonstrates how the same concept works with batched generation.

First, follow the exact same steps above for the per-model constructs
`xgr.TokenizerInfo` and `xgr.GrammarCompiler`. Say each request needs
to generate a valid JSON.

```python
import xgrammar as xgr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# Get tokenizer info
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
# This can be larger than tokenizer.vocab_size due to paddings
full_vocab_size = config.vocab_size
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)

# Compile a JSON grammar
compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
compiled_grammar: xgr.CompiledGrammar = compiler.compile_builtin_json_grammar()
```

Now, we need to maintain an `xgr.GrammarMatcher` for each request in the batch, since
each has a different generation state. Note that each request in the batch can follow a different
`xgr.CompiledGrammar`, but here for simplicity, they are all just following the general
JSON grammar.

```python
batch_size = 2
matchers = [
    xgr.GrammarMatcher(compiled_grammar)
    for i in range(batch_size)
]
token_bitmask = xgr.allocate_token_bitmask(batch_size, tokenizer_info.vocab_size)
```

We simulate an auto-regressive generation of batched inference. Note that here we
assume the generation lengths of the two requests are the same for simplicity. But
it should be easy to generalize based on how your engine supports batched inference.
The key difference from single-request generation is that, in batched-request generation,
each request has its own `xgr.GrammarMatcher` to maintain.

```python
sim_sampled_responses = ['{"name": "a"}<|end_of_text|>', '{"name": "b"}<|end_of_text|>']
sim_sampled_token_ids = [
  tokenizer.encode(response, add_special_tokens=False)
  for response in sim_sampled_responses
]

# Each loop iteration is a simulated auto-regressive step
for loop_iter in range(len(sim_sampled_token_ids[0])):
    # LLM batched inference to get logits, here we use randn to simulate
    # Now, logits is a tensor of shape (batch_size, full_vocab_size) on GPU
    # logits = LLM.inference()
    logits = torch.randn(batch_size, full_vocab_size).cuda()

    # This for loop is parallelizable using threading.Thread. But estimate
    # the overhead in your engine.
    for i in range(batch_size):
        matchers[i].fill_next_token_bitmask(token_bitmask, i)
    xgr.apply_token_bitmask_inplace(logits, token_bitmask.to(logits.device))

    # Sample next token
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    next_token_ids = [
        np.random.choice(list(range(full_vocab_size)), p=probs[i])
        for i in range(batch_size)
    ]

    # Update the matcher for each request
    for i in range(batch_size):
        # Here we accept the simulated response
        # assert matchers[i].accept_token(next_token_ids[i])
        matchers[i].accept_token(sim_sampled_token_ids[i][loop_iter])

# In our simulated case, all requests should have terminated since we accepted
# a stop token `<|end_of_text|>`
for i in range(batch_size):
    assert matchers[i].is_terminated()
    # Reset to be ready for the next generation
    matchers[i].reset()
```
