.. _tutorial-structured-generation:

Structured Generation
======================

XGrammar enables efficient structured generation. In this tutorial, we go over how to
use XGrammar to ensure that an LLM's output adheres to the structure of a valid JSON, a
customized JSON schema, and a customized GBNF grammar string.

We first lay out the concepts by going over :ref:`JSON generation <tutorial-json-generation>`
in detail. Then we go over how to generate with :ref:`customized JSON schemas <tutorial-json-schema-generation>`
and :ref:`customized GBNF grammar strings <tutorial-gbnf-generation>`. Finally, we demonstrate
how xgrammar works with :ref:`batched inference <tutorial-batched-inference>`.

Therefore, we encourage you to start with :ref:`JSON Generation <tutorial-json-generation>`.

The code snippets below are actual runnable code as we simulate the LLM generation.


Install XGrammar
~~~~~~~~~~~~~~~~

:ref:`XGrammar <installation_prebuilt_package>` is available via pip.
It is always recommended to install it in an isolated conda virtual environment.


.. _tutorial-json-generation:

JSON Generation
~~~~~~~~~~~~~~~

In this section, we see how to use XGrammar to ensure that an LLM's output is
always a valid JSON.

First, import necessary libraries for the tutorial.

.. code:: python

  import xgrammar as xgr
  import torch
  import numpy as np
  from transformers import AutoTokenizer, AutoConfig

Then, we extract tokenizer info from the LLM we are using with ``xgr.TokenizerInfo``

.. code:: python

  # Get tokenizer info
  model_id = "Qwen/Qwen2.5-1.5B-Instruct"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  config = AutoConfig.from_pretrained(model_id)
  tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
  # This can be larger than tokenizer.vocab_size due to paddings
  full_vocab_size = config.vocab_size

With the ``tokenizer_info``, instantiate ``xgr.GrammarCompiler`` that compiles a
grammar of your choice. Here we use a JSON grammar. Note that the ``compiler`` behavior
can be configured with ``max_threads`` for multithreading, and ``enable_cache`` (defaults to
true) for caching compiled grammars. Note that every thing we have seen up to now are per-model (rather
than per-generation).

.. code:: python

  compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
  compiled_grammar: xgr.CompiledGrammar = compiler.compile_builtin_json_grammar()

With the compiled grammar, we can instantiate a ``xgr.GrammarMatcher``, the main construct
we interact with that maintains the state of the structured generation. We also allocate a
bitmask that will be used to mask logits.

.. code:: python

  # Instantiate grammar matcher and allocate the bitmask
  matcher = xgr.GrammarMatcher(compiled_grammar, tokenizer_info, vocab_size=full_vocab_size)
  token_bitmask = xgr.GrammarMatcher.allocate_token_bitmask(matcher.vocab_size)

Now we simulate a single-request auto-regressive generation. See later section for :ref:`batched generation <tutorial-batched-inference>`.

.. code:: python

  # Here we simulate a valid sampled response
  sim_sampled_response = '{ "library": "xgrammar" }<|im_end|>'
  sim_sampled_token_ids = tokenizer.encode(sim_sampled_response)

  # Each loop iteration is a simulated auto-regressive step
  for i, sim_token_id in enumerate(sim_sampled_token_ids):
      # LLM inference to get logits, here we use randn to simulate
      # logits is a tensor of shape (full_vocab_size,) on GPU
      # logits = LLM.inference()
      logits = torch.randn(full_vocab_size).cuda()

      # Apply bitmask to logits to mask invalid tokens
      matcher.fill_next_token_bitmask(token_bitmask)

      # Sample next token
      probs = torch.softmax(logits, dim=-1)
      next_token_id = np.random.choice(list(range(full_vocab_size)), p=probs)

      # Accept token from matcher to update its state, so that the next bitmask
      # generated will enforce the next token to be generated. Assert to make
      # sure the token is indeed valid. Here we accept the simulated response
      # assert matcher.accept_token(next_token_id)
      assert matcher.accept(sim_token_id)

  # Since we accepted a stop token `<|im_end|>`, we have terminated
  assert matcher.is_terminated()

  # Reset to be ready for the next auto-regressive generation
  matcher.reset()


.. _tutorial-json-schema-generation:

JSON Schema Guided Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, we see how to use XGrammar to generate an output that adheres
to a customized JSON schema.

The flow is almost identical to the one above, except that the ``CompiledGrammar``
is compiled based on the JSON schema, rather than being compiled with a generic JSON grammar.

First, set up the tokenizer info and the grammar compiler as above.

.. code:: python

  import xgrammar as xgr
  import torch
  import numpy as np
  from transformers import AutoTokenizer, AutoConfig

  # Get tokenizer info
  model_id = "Qwen/Qwen2.5-1.5B-Instruct"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  config = AutoConfig.from_pretrained(model_id)
  tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
  # This can be larger than tokenizer.vocab_size due to paddings
  full_vocab_size = config.vocab_size

  compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)

Now, to compile a grammar from a JSON schema, there are generically two methods: from a Pydantic model,
or from a JSON schema string. The two code snippets below are functionally identical, pick one to run.

.. code:: python

  # Method 1. Compile with a pydantic model
  from pydantic import BaseModel

  class Person(BaseModel):
      name: str
      age: int

  compiled_grammar = compiler.compile_json_schema(Person)

.. code:: python

  # Method 2. Compile with a JSON schema string
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


Then, the remaining steps are identical to before, except that we now use a different
``xgr.CompiledGrammar`` and have a different simulated valid generation.

.. code:: python

  # Instantiate grammar matcher and allocate the bitmask
  matcher = xgr.GrammarMatcher(compiled_grammar, tokenizer_info, vocab_size=full_vocab_size)
  token_bitmask = xgr.GrammarMatcher.allocate_token_bitmask(matcher.vocab_size)

  # Here we simulate a valid sampled response
  sim_sampled_response = '{ "name": "xgrammar", "age": 0 }<|im_end|>'
  sim_sampled_token_ids = tokenizer.encode(sim_sampled_response)

  # Each loop iteration is a simulated auto-regressive step
  for i, sim_token_id in enumerate(sim_sampled_token_ids):
      # LLM inference to get logits, here we use randn to simulate
      # logits is a tensor of shape (vocab_size,) on GPU
      # logits = LLM.inference()
      logits = torch.randn(full_vocab_size).cuda()

      # Apply bitmask to logits to mask invalid tokens
      matcher.fill_next_token_bitmask(token_bitmask)

      # Sample next token
      probs = torch.softmax(logits, dim=-1)
      next_token_id = np.random.choice(list(range(full_vocab_size)), p=probs)

      # Accept token from matcher to update its state, so that the next bitmask
      # generated will enforce the next token to be generated. Assert to make
      # sure the token is indeed valid. Here we accept the simulated response
      # assert matcher.accept_token(next_token_id)
      assert matcher.accept(sim_token_id)

  # Since we accepted a stop token `<|im_end|>`, we have terminated
  assert matcher.is_terminated()

  # Reset to be ready for the next auto-regressive generation
  matcher.reset()


.. _tutorial-gbnf-generation:

GBNF Guided Generation
~~~~~~~~~~~~~~~~~~~~~~~

XGrammar also enables generation that adheres to a customized GBNF (GGML BNF) grammar string, with
the specification `here <https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md>`__.

The code is largely identical to above, except that the ``CompiledGrammar`` is now compiled with
the provided GBNF grammar string.

First, set up the tokenizer info and the grammar compiler as above.

.. code:: python

  import xgrammar as xgr
  import torch
  import numpy as np
  from transformers import AutoTokenizer, AutoConfig

  # Get tokenizer info
  model_id = "Qwen/Qwen2.5-1.5B-Instruct"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  config = AutoConfig.from_pretrained(model_id)
  tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
  # This can be larger than tokenizer.vocab_size due to paddings
  full_vocab_size = config.vocab_size

  compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)

Now, compile ``CompiledGrammar`` with your GBNF grammar string.

.. code:: python

  gbnf_grammar = """
  root  ::= (expr "=" term "\n")+
  expr  ::= term ([-+*/] term)*
  term  ::= num | "(" expr ")"
  num   ::= [0-9]+
  """

  compiled_grammar = compiler.compile_json_schema(Person)

Then, the remaining steps are identical to before, except that we now use a different
``xgr.CompiledGrammar`` and have a different simulated valid generation.

.. code:: python

  # Instantiate grammar matcher and allocate the bitmask
  matcher = xgr.GrammarMatcher(compiled_grammar, tokenizer_info, vocab_size=full_vocab_size)
  token_bitmask = xgr.GrammarMatcher.allocate_token_bitmask(matcher.vocab_size)

  # Here we simulate a valid sampled response
  sim_sampled_response = '(5+3)*2=16'
  sim_sampled_token_ids = tokenizer.encode(sim_sampled_response)

  # Each loop iteration is a simulated auto-regressive step
  for i, sim_token_id in enumerate(sim_sampled_token_ids):
      # LLM inference to get logits, here we use randn to simulate
      # logits is a tensor of shape (full_vocab_size,) on GPU
      # logits = LLM.inference()
      logits = torch.randn(full_vocab_size).cuda()

      # Apply bitmask to logits to mask invalid tokens
      matcher.fill_next_token_bitmask(token_bitmask)

      # Sample next token
      probs = torch.softmax(logits, dim=-1)
      next_token_id = np.random.choice(list(range(full_vocab_size)), p=probs)

      # Accept token from matcher to update its state, so that the next bitmask
      # generated will enforce the next token to be generated. Assert to make
      # sure the token is indeed valid. Here we accept the simulated response
      # assert matcher.accept_token(next_token_id)
      assert matcher.accept(sim_token_id)

  # Since we accepted a stop token `<|im_end|>`, we have terminated
  assert matcher.is_terminated()

  # Reset to be ready for the next auto-regressive generation
  matcher.reset()


.. _tutorial-batched-inference:

Structured Generation for Batched Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the above code snippets above assume a single request generation.
This section demonstrates how the same concept works with batched generation.

First, follow the exact same steps above for the per-model constructs
``xgr.TokenizerInfo`` and ``xgr.GrammarCompiler``. Say each request needs
to generate a valid JSON.

.. code:: python

  import xgrammar as xgr
  import torch
  import numpy as np
  from transformers import AutoTokenizer, AutoConfig

  # Get tokenizer info
  model_id = "Qwen/Qwen2.5-1.5B-Instruct"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  config = AutoConfig.from_pretrained(model_id)
  tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
  # This can be larger than tokenizer.vocab_size due to paddings
  full_vocab_size = config.vocab_size

  # Compile a JSON grammar
  compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
  compiled_grammar: xgr.CompiledGrammar = compiler.compile_builtin_json_grammar()

Now, we need to maintain an ``xgr.GrammarMatcher`` for each request in the batch, since
each has a different generation state. Note that each request in the batch can follow a different
``xgr.CompiledGrammar``, but here for simplicity, they are all just following the general
JSON grammar.

.. code:: python

  batch_size = 2
  matchers = [
      xgr.GrammarMatcher(compiled_grammar, tokenizer_info, vocab_size=full_vocab_size)
      for i in range(batch_size)
  ]
  token_bitmasks = xgr.GrammarMatcher.allocate_token_bitmask(matchers[0].vocab_size, batch_size)

We simulate an auto-regressive generation of batched inference. Note that here we
assume the generation lengths of the two requests are the same for simplicity. But
it should be easy to generalize based on how your engine supports batched inference.
The key difference from single-request generation is that, in batched-request generation,
each request has its own ``xgr.GrammarMatcher`` to maintain.

.. code:: python

  sim_sampled_responses = ['{ "name": "a" }<|im_end|>', '{ "name": "b" }<|im_end|>']
  sim_sampled_token_ids = [tokenizer.encode(response) for response in sim_sampled_responses]

  # Each loop iteration is a simulated auto-regressive step
  for loop_iter in range(len(sim_sampled_token_ids[0])):
      # LLM batched inference to get logits, here we use randn to simulate
      # Now, logits is a tensor of shape (batch_size, full_vocab_size) on GPU
      # logits = LLM.inference()
      logits = torch.randn(batch_size, full_vocab_size).cuda()

      # This for loop is parallelizable using threading.Thread. But estimate
      # the overhead in your engine.
      for i in range(batch_size):
          matchers[i].fill_next_token_bitmask(token_bitmasks, i)
      xgr.GrammarMatcher.apply_token_bitmask_inplace(logits, token_bitmasks)

      # Sample next token
      probs = torch.softmax(logits, dim=-1)
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
  # a stop token `<|im_end|>`
  for i in range(batch_size):
      assert matchers[i].is_terminated()
      # Reset to be ready for the next generation
      matcher.reset()
