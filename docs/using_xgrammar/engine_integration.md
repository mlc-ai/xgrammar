# Integration with LLM Engine

This tutorial covers the key information for integrating XGrammar into an LLM serving engine:
the lifecycle of XGrammar objects, the integration points in the engine's decoding step,
batched inference, and speculative decoding. Read
[Workflow of XGrammar](../start/workflow_of_xgrammar.md) first for the basic usage of each
component.

## Object Lifecycle in an Engine

XGrammar objects fall into two lifecycle categories in a serving engine.

**Per-model objects** are constructed once and reused for all requests:

- [`xgr.TokenizerInfo`](xgrammar.TokenizerInfo) encapsulates the vocabulary information of the
  model.
- [`xgr.GrammarCompiler`](xgrammar.GrammarCompiler) compiles grammars for this model. Keep one
  persistent compiler per model, so that its compilation cache is shared across requests.
- [`xgr.CompiledGrammar`](xgrammar.CompiledGrammar) is the compilation result. Requests using
  the same grammar can share one compiled grammar.

**Per-request objects:**

- [`xgr.GrammarMatcher`](xgrammar.GrammarMatcher) maintains the matching state of one request,
  so each request needs its own matcher. When the request finishes, discard the matcher or
  [`reset()`](xgrammar.GrammarMatcher.reset) it for reuse.

Compiling a complex grammar can take non-negligible time, so it should not block the engine's
main loop. Compile the grammar asynchronously when the request arrives, so that compilation
overlaps with the prefill phase of the request, as shown in the figure in the next section. See
[Multi-threaded Compilation](../start/workflow_of_xgrammar.md#multi-threaded-compilation) for
the asynchronous compilation API.

## Integrating into the Engine Step

![Constrained Decoding Pipeline](https://blog.mlc.ai/img/xgrammar/constrained-decoding-pipeline-overlap.png)

The figure shows how grammar processing fits into the engine's timeline. In the top pipeline,
grammar processing runs serially with LLM inference. In the bottom pipeline, grammar
compilation overlaps with prefilling, and mask generation overlaps with the GPU forward pass.
XGrammar's mask generation runs on the CPU, so a serving engine can hide almost all of its
latency behind GPU computation.

There are three integration points in each decoding step:

1. **Fill the token bitmask.**
   [`xgr.GrammarMatcher.fill_next_token_bitmask`](xgrammar.GrammarMatcher.fill_next_token_bitmask)
   computes the mask for the next token on the CPU. It only depends on the matcher state, so it
   can start as soon as the last token is accepted, running in parallel with the GPU forward
   pass of the current step. It returns a boolean `need_apply`; if `False`, the mask allows all
   tokens and the applying step below can be skipped.

2. **Apply the bitmask to the logits.**
   [`xgr.apply_token_bitmask_inplace`](xgrammar.apply_token_bitmask_inplace) sets the logits of
   invalid tokens to `-inf` before sampling. Copy the bitmask to the logits' device first; on
   CUDA devices it launches an optimized kernel. If structured and unstructured requests are
   mixed in one batch, pass the `indices` parameter to mask only the rows of the structured
   requests.

3. **Accept the sampled token.**
   [`xgr.GrammarMatcher.accept_token`](xgrammar.GrammarMatcher.accept_token) advances the
   matcher state after sampling. It returns `False` if the token does not conform to the
   grammar, which can only happen when the mask was not applied to the logits.

To decide when the structured generation ends, check
[`xgr.GrammarMatcher.is_terminated`](xgrammar.GrammarMatcher.is_terminated) in the engine's
stopping condition.

## Batched Inference

In a batch, each request has its own matcher, and all requests share one bitmask tensor:
request `i` fills row `i` of the bitmask, and a single
[`xgr.apply_token_bitmask_inplace`](xgrammar.apply_token_bitmask_inplace) call masks the whole
logits batch.

[`xgr.BatchGrammarMatcher`](xgrammar.BatchGrammarMatcher) performs the per-request operations
in one pass on the C++ side with multiple threads, avoiding Python-side loops:

```python
matchers = [xgr.GrammarMatcher(compiled_grammar) for _ in range(batch_size)]
token_bitmask = xgr.allocate_token_bitmask(batch_size, tokenizer_info.vocab_size)
batch_matcher = xgr.BatchGrammarMatcher(max_threads=8)

# In each decoding step:
batch_matcher.batch_fill_next_token_bitmask(matchers, token_bitmask)
xgr.apply_token_bitmask_inplace(logits, token_bitmask.to(logits.device))
next_token_ids = sample(logits)  # engine-specific sampling
xgr.BatchGrammarMatcher.batch_accept_token(matchers, next_token_ids)
```

[`batch_fill_next_token_bitmask`](xgrammar.BatchGrammarMatcher.batch_fill_next_token_bitmask)
also accepts an `indices` parameter to fill only a subset of the bitmask rows, matching the
`indices` parameter of `apply_token_bitmask_inplace`.
[`batch_accept_string`](xgrammar.BatchGrammarMatcher.batch_accept_string) and
[`batch_rollback`](xgrammar.BatchGrammarMatcher.batch_rollback) are the batch versions of the
corresponding [`xgr.GrammarMatcher`](xgrammar.GrammarMatcher) methods.

## Speculative Decoding

In speculative decoding, a draft model proposes several draft tokens organized as a chain or a
tree, and the target model verifies them in one forward pass. For structured generation, the
engine also needs a token bitmask at every draft position to verify that the draft tokens
conform to the grammar.

![Overlap of Constrained Decoding and Speculative Decoding](https://blog.mlc.ai/img/xgrammar2/image3.png)

**Draft tree traversal.**
[`xgr.GrammarMatcher.traverse_draft_tree`](xgrammar.GrammarMatcher.traverse_draft_tree)
traverses a draft token tree once and fills the bitmask for every node. The tree is encoded
with three int64 tensors: `draft_tokens[i]` is the token id of node `i`,
`retrieve_next_token[i]` is the first child of node `i`, and `retrieve_next_sibling[i]` is the
next sibling of node `i` (`-1` means none). The `time_threshold` parameter bounds the traversal
time: if exceeded, the method stops and returns `False`.

As the figure shows, the traversal runs on the CPU while the target model verifies the same
tree on the GPU, so mask generation overlaps with the verification. This draft tree traversal
API was first proposed by the SGLang team.

**Manual state control.** For chain-shaped drafts, or for finer-grained control over the tree,
the engine can walk the draft manually: fill the bitmask and accept each draft token in turn,
then roll back the rejected suffix with
[`xgr.GrammarMatcher.rollback`](xgrammar.GrammarMatcher.rollback) after the verification
(batched version:
[`batch_rollback`](xgrammar.BatchGrammarMatcher.batch_rollback)). To explore multiple branches
with independent states, duplicate the matcher with
[`xgr.GrammarMatcher.fork`](xgrammar.GrammarMatcher.fork).

**Jump-forward decoding.**
[`xgr.GrammarMatcher.find_jump_forward_string`](xgrammar.GrammarMatcher.find_jump_forward_string)
returns the longest string that certainly follows the current state under the grammar. The
engine can append this string to the output directly without LLM decoding, saving decoding
steps.

## Real-World Integrations

XGrammar has been integrated into mainstream LLM serving engines. Their integration code can
serve as a reference: [SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), and
[MLC-LLM](https://github.com/mlc-ai/mlc-llm).
