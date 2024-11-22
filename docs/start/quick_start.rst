.. _quick-start:

Quick Start
===========

Example
-------

The easiest way of trying out XGrammar is to use the ``transformers`` library in Python.
After :ref:`installing XGrammar <installation>`, run the following example to see how XGrammar enables
structured generation -- a JSON in this case.


Perparation
^^^^^^^^^^^
Instantiate a model, a tokenizer, and inputs.

.. code:: python

    import xgrammar as xgr

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    device = "cuda"  # Or "cpu", etc.
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
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

Compile Grammar
^^^^^^^^^^^^^^^

Construct a ``GrammarCompiler`` and compile the grammar.

The grammar can be a built-in JSON grammar, a JSON schema string, or an EBNF string. EBNF provides
more flexibility for customization. See
`llama.cpp grammars <https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md>`_.

.. code:: python

    # 1. Compile grammar (NOTE: you can substitute this with other grammars like EBNF, JSON Schema)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = grammar_compiler.compile_builtin_json_grammar()
    # Other ways: provide a json schema string
    # compiled_grammar = grammar_compiler.compile_json_schema(json_schema_string)
    # Or provide an EBNF string
    # compiled_grammar = grammar_compiler.compile_grammar(ebnf_string)

    # 2. Prepare inputs

    # 3. Instantiate logits_processor per each generate, generate, and print response
    xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
    generated_ids = model.generate(
        **model_inputs, max_new_tokens=512, logits_processor=[xgr_logits_processor]
    )
    generated_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))


What to Do Next
---------------

- Check out :ref:`tutorial-structured-generation` for the detailed usage guide of XGrammar.
- Report any problem or ask any question: open new issues in our `GitHub repo <https://github.com/mlc-ai/xgrammar/issues>`_.
