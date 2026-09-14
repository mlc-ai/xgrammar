"""Load independent test tokenizers without repeatedly querying the Hub."""

from copy import deepcopy
from functools import lru_cache

from transformers import AutoTokenizer


@lru_cache(maxsize=8)
def _load_tokenizer(model_id, **kwargs):
    # The Hub's file cache does not cache metadata requests made by from_pretrained.
    # Bound the cache because tokenizers with large vocabularies use substantial memory.
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def load_tokenizer(model_id, **kwargs):
    """Reuse tokenizer loads while isolating mutable state between tests."""
    return deepcopy(_load_tokenizer(model_id, **kwargs))
