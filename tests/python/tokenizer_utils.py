"""Load independent test tokenizers without repeatedly querying the Hub."""

import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def local_model_path(model_id, revision=None):
    """Resolve a prepared snapshot in CI; keep normal developer loads unchanged."""
    assets = os.environ.get("XGRAMMAR_TOKENIZER_ASSETS")
    if not assets or Path(model_id).is_dir():
        return model_id
    root = Path(assets)
    index = json.loads((root / "index.json").read_text())
    if model_id not in index:
        raise RuntimeError("Tokenizer asset missing from CI bundle: " + str(model_id))
    entry = index[model_id]
    if revision is not None and revision not in (entry["revision"], entry["commit"]):
        raise RuntimeError("Tokenizer revision missing from CI bundle: " + str(model_id))
    return str(root / entry["path"])


def _local_load_args(model_id, kwargs):
    path = local_model_path(model_id, kwargs.get("revision"))
    if path != model_id:
        kwargs.pop("revision", None)
        kwargs["local_files_only"] = True
    return path, kwargs


@lru_cache(maxsize=None)
def _load_tokenizer(model_id, **kwargs):
    # The Hub's file cache does not cache metadata requests made by from_pretrained.
    path, kwargs = _local_load_args(model_id, kwargs)
    return AutoTokenizer.from_pretrained(path, **kwargs)


def load_tokenizer(model_id, **kwargs):
    """Reuse tokenizer loads while isolating mutable state between tests."""
    return deepcopy(_load_tokenizer(model_id, **kwargs))


def load_config(model_id, **kwargs):
    """Load model metadata from the same snapshot as the tokenizer."""
    path, kwargs = _local_load_args(model_id, kwargs)
    return AutoConfig.from_pretrained(path, **kwargs)


def load_model_file(model_id, filename, revision=None):
    """Resolve custom encoder files without accessing the Hub in CI."""
    path = local_model_path(model_id, revision)
    if path != model_id:
        file = Path(path) / filename
        if not file.is_file():
            raise RuntimeError("Missing prepared model file: " + str(file))
        return str(file)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_id, filename, revision=revision)
