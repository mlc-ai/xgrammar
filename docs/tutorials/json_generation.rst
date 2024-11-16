.. _tutorial-json-generation:

JSON Generation
====================


Install XGrammar
~~~~~~~~~~~~~~~~

:ref:`XGrammar <installation_prebuilt_package>` is available via pip.
It is always recommended to install it in an isolated conda virtual environment.


.. _tutorial-json-generation-construct-grammar:

Step 1: Construct a grammar
~~~~~~~~~~~~~~~~~~~~~~~~~~~

XGrammar provides the following methods to flexibly construct a grammar.
You can choose from any of the following ways to construct grammar from different sources.

**Method 1: Construct with a GBNF string.**
The GBNF (GGML BNF) specification is available
`here <https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md>`__.


.. code:: python

  from xgrammar import BNFGrammar

  # Method 1: Construct with a GBNF string.
  gbnf_grammar = """
  root  ::= (expr "=" term "\n")+
  expr  ::= term ([-+*/] term)*
  term  ::= num | "(" expr ")"
  num   ::= [0-9]+
  """
  gbnf_grammar = BNFGrammar(gbnf_grammar)


**Method 2: Use the builtin JSON grammar.**

.. code:: python

  from xgrammar import BuiltinGrammar

  # Method 2: Use the builtin JSON grammar.
  json_grammar = BuiltinGrammar.json()


**Method 3: Construct from a Pydantic model.**

.. code:: python

  from xgrammar import BuiltinGrammar
  from pydantic import BaseModel

  # Method 3: Construct from a Pydantic model.
  class Person(BaseModel):
      name: str
      age: int
  json_schema_pydantic = BuiltinGrammar.json_schema(Person)

**Method 4: Construct from a JSON schema string.**

.. code:: python

  import json
  from xgrammar import BuiltinGrammar

  # Method 4: Construct from a JSON schema string.
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


.. _tutorial-json-generation-compile-grammar:

Step 2: Compile grammars
~~~~~~~~~~~~~~~~~~~~~~~~

XGrammar supports multi-threaded grammar compilation.
In addition, we provide a cache in the grammar compiler to avoid
repetitive compilation for a same grammar.

To initialize a grammar compiler, we first need to obtain
information from the target tokenizer.
As an example, here we use the Llama-3 model tokenizer.

.. code:: python

  from xgrammar import TokenizerInfo
  from transformers import AutoTokenizer

  # Obtain XGrammar TokenizerInfo from HuggingFace tokenizer (once per model).
  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
  tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)


Now we can create a grammar compiler :class:`xgrammar.CachedGrammarCompiler`
and compile the constructed grammar.
Notably, we cache all the compiled grammars, so each grammar will be compiled
at most once.

.. code:: python

  from xgrammar import CachedGrammarCompiler

  # Construct CachedGrammarCompiler.
  compiler = CachedGrammarCompiler(tokenizer_info, max_threads=8)
  # Compiler the grammar.
  compiled_grammar = compiler.compile_json_schema(json_schema_str)


Alternatively, we also provide the no-cache compiler, which does not
cache grammars after compilation.

.. code:: python

  from xgrammar import CompiledGrammar

  # Construct CompiledGrammar (no cache).
  compiler = CompiledGrammar(tokenizer_info, max_threads=8)
  # Compiler the grammar.
  compiled_grammar = compiler.compile_json_schema(json_schema_str)



.. _tutorial-json-generation-grammar-guided-generation:

Step 3: Grammar-guided generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can now use the compiled grammar in structured generation.
Below are two pseudo Python code examples for
single-request generation and batch-request generation respectively.

**Single-request generation.**

.. code:: python

  from xgrammar import GrammarMatcher
  import torch

  # Create a grammar matcher from the compiled grammar.
  matcher = GrammarMatcher(compiled_grammar)

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


**Batch-request generation.**

.. code:: python

  from xgrammar import GrammarMatcher
  import torch

  batch_size = 10
  # Create a grammar matcher for each request.
  matchers = [GrammarMatcher(compiled_grammar) for i in range(batch_size)]
  token_bitmasks = GrammarMatcher.allocate_token_bitmask(matchers[0].vocab_size, batch_size)
  while True:
      logits = LLM.inference() # logits is a tensor of shape (batch_size, vocab_size) on GPU
      # This for loop is parallelizable using threading.Thread. But estimate the overhead in your
      # engine.
      for i in range(batch_size):
          matchers[i].fill_next_token_bitmask(token_bitmasks, i)
      GrammarMatcher.apply_token_bitmask_inplace(logits, token_bitmasks)

      prob = torch.softmax(logits, dim=-1) # get probability from logits
      next_token_ids = Sampler.sample(logits) # use your own sampler

      for i in range(batch_size):
          matchers[i].accept_token(next_token_ids[i])
          if matchers[i].is_terminated(): # or your own termination condition
              requests[i].terminate()

