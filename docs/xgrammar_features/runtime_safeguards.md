# Runtime Safeguards

XGrammar has a set of mechanisms to safeguard the runtime and avoid the LLM server from
crashing.

## Recursion Limit

The {py:class}`xgrammar.GrammarMatcher` class uses a pushdown automata parser to parse the grammar.
It may involve very deep recursion, which may cause stack overflow. XGrammar provides
{py:meth}`xgrammar.set_max_recursion_depth` to set the maximum recursion depth and
{py:meth}`xgrammar.get_max_recursion_depth` to get the current maximum recursion
depth. The maximum recursion depth is set per process.The default maximum recursion depth is 10000.

If the recursion depth exceeds the limit,
the matcher operations (including {py:meth}`xgrammar.GrammarMatcher.accept_token`,
{py:meth}`xgrammar.GrammarMatcher.accept_string`, {py:meth}`xgrammar.GrammarMatcher.fill_next_token_bitmask`,
{py:meth}`xgrammar.GrammarMatcher.find_jump_forward_string`) will raise
{py:exc}`RuntimeError`.

You can also use the {py:func}`xgrammar.max_recursion_depth` context manager to set the maximum
recursion depth for a code block.

```python
from xgrammar import max_recursion_depth

with max_recursion_depth(10000):
    matcher.accept_token(token_id)
```

## Cache Size Limit

The {py:class}`xgrammar.GrammarCompiler` class uses a cache to store the compiled grammars.
The cache size can be limited to avoid the cache from growing too large. The cache uses an LRU
algorithm to evict the least recently used items. The cache size limit is -1 by default, which means
no limit.

```python
from xgrammar import GrammarCompiler

compiler = GrammarCompiler(tokenizer_info, cache_limit_bytes=10000)
```
