import json
import os
import socket
from pathlib import Path

import pytest
from tokenizer_utils import load_config, load_model_file, load_tokenizer, local_model_path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast


@pytest.mark.thread_unsafe
def test_cached_tokenizers_are_independent(tmp_path, monkeypatch):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(tmp_path)
    original_load = AutoTokenizer.from_pretrained
    calls = []

    def tracked_load(*args, **kwargs):
        calls.append((args, kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tracked_load)
    first = load_tokenizer(tmp_path)
    second = load_tokenizer(tmp_path)
    assert len(calls) == 1
    assert first is not second
    assert first.encode("hello") == second.encode("hello") == [1]

    first.add_special_tokens({"pad_token": "[PAD]"})
    first.chat_template = "changed"
    third = load_tokenizer(tmp_path)
    assert len(calls) == 1
    for other in (second, third):
        assert "[PAD]" not in other.get_vocab()
        assert other.pad_token is None
        assert other.chat_template is None

    left_padding = load_tokenizer(tmp_path, padding_side="left")
    assert left_padding.padding_side == "left"
    assert third.padding_side == "right"
    assert len(calls) == 2


@pytest.mark.thread_unsafe
def test_failed_tokenizer_load_is_not_cached(tmp_path, monkeypatch):
    calls = []

    def failing_load(*args, **kwargs):
        calls.append((args, kwargs))
        raise OSError("Tokenizer download failed")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", failing_load)
    for _ in range(2):
        with pytest.raises(OSError, match="Tokenizer download failed"):
            load_tokenizer(tmp_path)
    assert len(calls) == 2


@pytest.mark.thread_unsafe
def test_prepared_bundle_loading(tmp_path, monkeypatch):
    model_dir = tmp_path / "example--tokenizer"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(model_dir)
    (model_dir / "config.json").write_text(
        '{"model_type": "gpt2", "vocab_size": 2, "bos_token_id": null, "eos_token_id": null}'
    )
    (model_dir / "encoder.py").write_text("VALUE = 1\n")
    (tmp_path / "index.json").write_text(
        json.dumps(
            {"example/tokenizer": {"path": model_dir.name, "revision": "main", "commit": "abc"}}
        )
    )
    monkeypatch.setenv("XGRAMMAR_TOKENIZER_ASSETS", str(tmp_path))
    assert load_tokenizer("example/tokenizer", revision="abc").encode("hello") == [1]
    assert load_config("example/tokenizer").vocab_size == 2
    assert load_model_file("example/tokenizer", "encoder.py") == str(model_dir / "encoder.py")
    with pytest.raises(RuntimeError, match="asset missing"):
        local_model_path("example/missing")
    with pytest.raises(RuntimeError, match="revision missing"):
        local_model_path("example/tokenizer", "wrong")
    with pytest.raises(RuntimeError, match="Missing prepared model file"):
        load_model_file("example/tokenizer", "missing.py")


def test_public_tokenizer_bundle_smoke():
    """Exercise a real tokenizer even in PRs without gated-model credentials."""
    if not os.environ.get("XGRAMMAR_TOKENIZER_ASSETS"):
        pytest.skip("Prepared tokenizer bundle is only required in CI")
    tokenizer = load_tokenizer("Qwen/Qwen2.5-0.5B", use_fast=True)
    text = "Hello, tokenizer!"
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text


def test_prepared_tiktoken_encoding():
    """Phi-3-small requires tiktoken's external vocabulary cache as well."""
    assets = os.environ.get("XGRAMMAR_TOKENIZER_ASSETS")
    if not assets or not (Path(assets) / "tiktoken").is_dir():
        pytest.skip("The public-only bundle does not need tiktoken's vocabulary cache")
    import tiktoken

    assert Path(os.environ["TIKTOKEN_CACHE_DIR"]) == Path(assets) / "tiktoken"
    encoding = tiktoken.get_encoding("cl100k_base")
    text = "Hello, tokenizer!"
    assert encoding.decode(encoding.encode(text)) == text


def test_ci_network_guard():
    if os.environ.get("XGRAMMAR_TEST_OFFLINE") != "1":
        pytest.skip("Network guard is only required in offline CI")
    with socket.socket() as connection:
        with pytest.raises(RuntimeError, match="Network access attempted"):
            connection.connect(("huggingface.co", 443))
