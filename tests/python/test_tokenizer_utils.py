import pytest
from tokenizer_utils import load_tokenizer
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
