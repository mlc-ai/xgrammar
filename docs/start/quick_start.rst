.. _quick-start:

Quick Start
===========

Example
-------

After :ref:`installing XGrammar <installation>`, run the following example to see how XGrammar enables
structured generation -- a JSON in this case.

.. code:: python

    import xgrammar as xgr
    import torch
    from transformers import AutoTokenizer, AutoConfig

    # Get tokenizer info
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    # This can be larger than tokenizer.vocab_size due to paddings
    full_vocab_size = config.vocab_size
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)

    # Compile a JSON grammar
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
    compiled_grammar: xgr.CompiledGrammar = compiler.compile_builtin_json_grammar()

    # Instantiate grammar matcher and allocate the bitmask
    matcher = xgr.GrammarMatcher(compiled_grammar)
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    # Each loop iteration is a simulated auto-regressive step. Here we use
    # simulated logits and sampled tokens. In real application, use XGrammar
    # in a LLM generation loop and sample with the masked logits.
    sim_sampled_response = '{ "library": "xgrammar" }<|endoftext|>'
    sim_sampled_token_ids = tokenizer.encode(sim_sampled_response)
    for i, sim_token_id in enumerate(sim_sampled_token_ids):
        logits = torch.randn(full_vocab_size).cuda()
        matcher.fill_next_token_bitmask(token_bitmask)
        xgr.apply_token_bitmask_inplace(logits, token_bitmask.to(logits.device))
        assert matcher.accept_token(sim_token_id)

    assert matcher.is_terminated()
    matcher.reset()


What to Do Next
---------------

- Check out :ref:`tutorial-structured-generation` for the detailed usage guide of XGrammar.
- Report any problem or ask any question: open new issues in our `GitHub repo <https://github.com/mlc-ai/xgrammar/issues>`_.
