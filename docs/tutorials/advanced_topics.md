# Advanced Topics

This section covers advanced topics about XGrammar.

## Multi-threaded Grammar Compilation and Cache

To accelerate computation, [`xgr.GrammarCompiler`](xgrammar.GrammarCompiler) is multithreaded. It uses multiple threads to process a single grammar and can also compile multiple grammars in parallel. `xgr.GrammarCompiler.compile_*` functions releases the GIL, so you can use asyncio to compile multiple grammars in parallel.

The `max_threads` parameter controls the maximum number of threads used. We recommend setting it to half the number of your CPU’s virtual cores for optimal performance.

```python
grammar_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)

# Use asyncio to compile multiple grammars in parallel
async def compile_grammars():
    # Submit two grammars in sequence
    future1 = asyncio.to_thread(grammar_compiler.compile_grammar, grammar1)
    future2 = asyncio.to_thread(grammar_compiler.compile_grammar, grammar2)

    # Wait for both futures to complete
    compiled_grammar1 = await future1
    compiled_grammar2 = await future2

    return compiled_grammar1, compiled_grammar2

compiled_grammar1, compiled_grammar2 = asyncio.run(compile_grammars())
```

[`xgr.GrammarCompiler`](xgrammar.GrammarCompiler) also includes a cache. If the same grammar is compiled again, the cached result is returned directly. Set `cache_enabled` to `True` to enable the cache, and `cache_limit_bytes` to control the maximum memory usage for the cache. The cache uses LRU (Least Recently Used) eviction policy.

The EBNF string, JSON Schema string, regex pattern are used as the cache key for [`compile_grammar`](xgrammar.GrammarCompiler.compile_grammar), [`compile_json_schema`](xgrammar.GrammarCompiler.compile_json_schema), [`compile_regex`](xgrammar.GrammarCompiler.compile_regex), respectively. By caching the input string directly, we further reduce the time spent constructing the grammar.

```python
grammar_compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=True, cache_limit_bytes=128 * 1024 * 1024)
compiled_grammar1 = grammar_compiler.compile_grammar(grammar)
# return immidiately
compiled_grammar2 = grammar_compiler.compile_grammar(grammar)
grammar_compiler.clear_cache()
```

## Handle Padding to the LLM Output Logits

Sometimes the shape of the LLM output logits can be larger than the size of the LLM tokenizer’s vocabulary. This is because the LLM pads the output tensor. For example, the tokenizer of DeepSeek-V3 only defines 128,815 tokens, but its output probability distribution has a dimension of 129,280.

Note that XGrammar always treat **the size of the model’s output logits** as the vocabulary size, because the bitmask operates on the LLM output logits. This is used in [`xgr.TokenizerInfo`](xgrammar.TokenizerInfo) and [`xgr.allocate_token_bitmask`](xgrammar.allocate_token_bitmask):

```python
tokenizer_info = xgr.TokenizerInfo(tokenizer, vocab_size=129280)
token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
```

For most models, the logits' vocabulary size can be found in the model config.

```python
config = AutoConfig.from_pretrained(model_path)
vocab_size = config.vocab_size
```

## Generate Token Masks in a Batch

XGrammar provides a new class [`xgr.BatchGrammarMatcher`](xgrammar.BatchGrammarMatcher) for users to generate token masks in a batch.

```python
batch_grammar_matcher = xgr.BatchGrammarMatcher(max_threads=8)
```

`BatchGrammarMatcher` needs a parameter `max_threads` to initialize. It represents the maximum threads in `fill_next_token_mask`. If not set, it will use std::thread::hardware_concurrency() / 2 as the default value.

`BatchGrammarMatcher` has three methods: `batch_fill_next_token_bitmask`, `batch_accept_token`, and `batch_accept_string` to handle the mask generation tasks. Here is an example to use `batch_fill_next_token_bitmask`:

```python
matchers = [grammar_matcher_1, grammar_matcher_2, grammar_matcher_3, ...]
batch_size = len(matchers)
token_bitmask = xgr.allocate_token_bitmask(batch_size, tokenizer_info.vocab_size)
batch_grammar_matcher = xgr.BatchGrammarMatcher(max_threads=8)
batch_grammar_matcher.batch_fill_next_token_bitmask(matchers, token_bitmask)
```

Each matcher will store its token mask in the corresponding tensor. For `batch_accept_token` and `batch_accept_string`, each matcher will try to accept the corresponding token_id/str.

```python
    inputs = [token_id_1, token_id_2, token_id_3, ...]
    results = xgr.BatchGrammarMatcher.batch_accept_token(matchers, inputs) # List[Bool]
```

```python
    inputs = [str_1, str_2, str_3, ...]
    results = xgr.BatchGrammarMatcher.batch_accept_string(matchers, inputs) # List[Bool]
```

Each boolean value in `results` represents whether the input is accepted by the corresponding matcher.
