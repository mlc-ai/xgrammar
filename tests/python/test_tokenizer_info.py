import logging
import sys
from typing import Dict, List, Tuple

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import xgrammar as xgr


@pytest.fixture(scope="module")
def tokenizer_info_storage() -> Dict[str, Tuple[PreTrainedTokenizerBase, xgr.TokenizerInfo]]:
    """Mapping from the tokenizer path to the huggingface tokenizer and XGrammar tokenizer info."""
    return {}


tokenizer_path__vocab_type__prepend_space = [
    ("luodian/llama-7b-hf", xgr.VocabType.BYTE_FALLBACK, True),
    ("meta-llama/Llama-2-7b-chat-hf", xgr.VocabType.BYTE_FALLBACK, True),
    ("meta-llama/Meta-Llama-3-8B-Instruct", xgr.VocabType.BYTE_LEVEL, False),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", xgr.VocabType.BYTE_LEVEL, False),
    ("lmsys/vicuna-7b-v1.5", xgr.VocabType.BYTE_FALLBACK, True),
    ("NousResearch/Hermes-2-Theta-Llama-3-70B", xgr.VocabType.BYTE_LEVEL, False),
    ("NousResearch/Hermes-3-Llama-3.1-8B", xgr.VocabType.BYTE_LEVEL, False),
    ("google/gemma-2b-it", xgr.VocabType.BYTE_FALLBACK, False),
    ("CohereForAI/aya-23-8B", xgr.VocabType.BYTE_LEVEL, False),
    ("deepseek-ai/DeepSeek-Coder-V2-Instruct", xgr.VocabType.BYTE_LEVEL, False),
    ("deepseek-ai/DeepSeek-V2-Chat-0628", xgr.VocabType.BYTE_LEVEL, False),
    ("deepseek-ai/deepseek-coder-7b-instruct-v1.5", xgr.VocabType.BYTE_LEVEL, False),
    ("microsoft/phi-2", xgr.VocabType.BYTE_LEVEL, False),
    ("microsoft/Phi-3-mini-4k-instruct", xgr.VocabType.BYTE_FALLBACK, True),
    ("microsoft/Phi-3.5-mini-instruct", xgr.VocabType.BYTE_FALLBACK, True),
    ("Qwen/Qwen1.5-4B-Chat", xgr.VocabType.BYTE_LEVEL, False),
    ("Qwen/Qwen2-7B-Instruct", xgr.VocabType.BYTE_LEVEL, False),
    ("microsoft/Phi-3-small-8k-instruct", xgr.VocabType.RAW, False),
    ("Qwen/Qwen-7B-Chat", xgr.VocabType.RAW, False),
    ("meta-llama/Llama-3.2-1B", xgr.VocabType.BYTE_LEVEL, False),
    ("google/gemma-2-2b-it", xgr.VocabType.BYTE_FALLBACK, False),
    ("deepseek-ai/DeepSeek-V2.5", xgr.VocabType.BYTE_LEVEL, False),
    ("Qwen/Qwen2.5-1.5B", xgr.VocabType.BYTE_LEVEL, False),
    ("internlm/internlm2_5-7b-chat", xgr.VocabType.BYTE_FALLBACK, False),
    ("mistralai/Mixtral-8x22B-Instruct-v0.1", xgr.VocabType.BYTE_FALLBACK, True),
    ("THUDM/glm-4-9b-chat", xgr.VocabType.RAW, False),
    ("THUDM/chatglm3-6b", xgr.VocabType.BYTE_FALLBACK, True),
    ("deepseek-ai/DeepSeek-R1", xgr.VocabType.BYTE_LEVEL, False),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", xgr.VocabType.BYTE_LEVEL, False),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", xgr.VocabType.BYTE_LEVEL, False),
]

tokenizer_paths = [path for path, *_ in tokenizer_path__vocab_type__prepend_space]


@pytest.mark.hf_token_required
@pytest.mark.parametrize("tokenizer_path", tokenizer_paths)
def test_build_tokenizer_info(
    tokenizer_path: str,
    tokenizer_info_storage: Dict[str, Tuple[PreTrainedTokenizerBase, xgr.TokenizerInfo]],
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    tokenizer_info_storage[tokenizer_path] = (tokenizer, tokenizer_info)


@pytest.mark.hf_token_required
@pytest.mark.parametrize(
    "tokenizer_path, vocab_type, add_prefix_space", tokenizer_path__vocab_type__prepend_space
)
def test_properties(
    tokenizer_path: str,
    vocab_type: xgr.VocabType,
    add_prefix_space: bool,
    tokenizer_info_storage: Dict[str, Tuple[PreTrainedTokenizerBase, xgr.TokenizerInfo]],
):
    tokenizer, tokenizer_info = tokenizer_info_storage[tokenizer_path]
    vocab_dict = tokenizer.get_vocab()
    max_id = max(vocab_dict.values()) if vocab_dict else -1
    assert tokenizer_info.vocab_size == max(len(vocab_dict), max_id + 1)
    assert tokenizer_info.vocab_type == vocab_type
    assert tokenizer_info.add_prefix_space == add_prefix_space


@pytest.mark.hf_token_required
@pytest.mark.parametrize("tokenizer_path", tokenizer_paths)
def test_decoded_vocab(
    tokenizer_path: str,
    tokenizer_info_storage: Dict[str, Tuple[PreTrainedTokenizerBase, xgr.TokenizerInfo]],
):
    tokenizer, tokenizer_info = tokenizer_info_storage[tokenizer_path]
    decoded_vocab = tokenizer_info.decoded_vocab
    vocab_dict = tokenizer.get_vocab()
    max_id = max(vocab_dict.values()) if vocab_dict else -1
    assert isinstance(decoded_vocab, list)
    assert all(isinstance(token, bytes) for token in decoded_vocab)
    assert len(decoded_vocab) == max(len(vocab_dict), max_id + 1)
    assert len(decoded_vocab) == tokenizer_info.vocab_size


@pytest.mark.hf_token_required
@pytest.mark.parametrize("tokenizer_path", tokenizer_paths)
def test_stop_token_ids(
    tokenizer_path: str,
    tokenizer_info_storage: Dict[str, Tuple[PreTrainedTokenizerBase, xgr.TokenizerInfo]],
):
    tokenizer, tokenizer_info = tokenizer_info_storage[tokenizer_path]
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        assert tokenizer_info.stop_token_ids == [tokenizer.eos_token_id]
    else:
        logging.warning(f"EOS token id is not defined for tokenizer {tokenizer_path}")


@pytest.mark.hf_token_required
@pytest.mark.parametrize("tokenizer_path", tokenizer_paths)
def test_decode_text(
    tokenizer_path: str,
    tokenizer_info_storage: Dict[str, Tuple[PreTrainedTokenizerBase, xgr.TokenizerInfo]],
):
    text = (
        "Hello 你好 こんにちは 안녕하세요! 🌎🌍🌏 \u0300\u0301\u0302 \U0001f600\U0001f601\U0001f602 "
        + "αβγδ АБВГД عربي עברית"
        + "\n\t\r Special chars: &*()_+-=[]{}|;:'\",.<>?/\\~`!@#$%^<think>haha</think>"
    )
    tokenizer, tokenizer_info = tokenizer_info_storage[tokenizer_path]
    decoded_vocab = tokenizer_info.decoded_vocab
    tokenized_text = tokenizer.encode(text)

    recovered_text = b"".join(decoded_vocab[token_id] for token_id in tokenized_text).decode(
        "utf-8"
    )

    trial_text = "a"
    trial_text_roundtrip = b"".join(
        decoded_vocab[token_id] for token_id in tokenizer.encode(trial_text)
    ).decode("utf-8")
    assert trial_text_roundtrip[-1] == "a"
    detected_prefix = trial_text_roundtrip[:-1]

    assert tokenizer_info.add_prefix_space == (
        len(detected_prefix) > 0 and detected_prefix[-1] == " "
    )
    assert detected_prefix + text == recovered_text


tokenizer_path__token_ids__raw_tokens = [
    # raw
    ("microsoft/Phi-3-small-8k-instruct", [10, 94, 37046], [b"+", b"\xa1", b"\xe6\x88\x91"]),
    # byte_fallback
    (
        "meta-llama/Llama-2-7b-chat-hf",
        [4, 259, 261, 20565],
        [b"\x01", b"  ", b"er", " исследова".encode("utf-8")],
    ),
    # byte_level
    (
        "meta-llama/Meta-Llama-3-8B-Instruct",
        [1, 37046, 40508],
        [b'"', "我".encode("utf-8"), b" automotive"],
    ),
]


@pytest.mark.hf_token_required
@pytest.mark.parametrize(
    "tokenizer_path, token_ids, raw_tokens", tokenizer_path__token_ids__raw_tokens
)
def test_vocab_conversion(tokenizer_path: str, token_ids: List[int], raw_tokens: List[bytes]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    vocab = tokenizer_info.decoded_vocab
    for token_id, raw_token in zip(token_ids, raw_tokens):
        assert vocab[token_id] == raw_token


tokenizer_path__metadata_str = [
    (
        "microsoft/Phi-3-small-8k-instruct",
        '{"vocab_type":0,"vocab_size":100352,"add_prefix_space":false,"stop_token_ids":[100257]}',
    ),
    (
        "meta-llama/Llama-2-7b-chat-hf",
        '{"vocab_type":1,"vocab_size":32000,"add_prefix_space":true,"stop_token_ids":[2]}',
    ),
    (
        "meta-llama/Meta-Llama-3-8B-Instruct",
        '{"vocab_type":2,"vocab_size":128256,"add_prefix_space":false,"stop_token_ids":[128009]}',
    ),
]


@pytest.mark.hf_token_required
@pytest.mark.parametrize("tokenizer_path, metadata_str", tokenizer_path__metadata_str)
def test_dump_metadata_load(tokenizer_path: str, metadata_str: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    assert tokenizer_info.dump_metadata() == metadata_str

    encoded_vocab = tokenizer.get_vocab()
    encoded_vocab = [token for token, _ in sorted(encoded_vocab.items(), key=lambda x: x[1])]

    loaded = xgr.TokenizerInfo.from_vocab_and_metadata(encoded_vocab, metadata_str)
    assert loaded.decoded_vocab == tokenizer_info.decoded_vocab

    loaded_new = xgr.TokenizerInfo(tokenizer_info.decoded_vocab)
    assert loaded_new.decoded_vocab == tokenizer_info.decoded_vocab


def test_special_token_detection():
    # Now only empty string "" is treated as a special token.
    vocab_dict = ["", "<s>", "</s>", "[@BOS@]", "regular", "<>", "<think>", "</think>"]
    tokenizer_info = xgr.TokenizerInfo.from_vocab_and_metadata(
        vocab_dict, '{"vocab_type":1,"vocab_size":8,"add_prefix_space":true,"stop_token_ids":[2]}'
    )
    expected_special_tokens = {0}
    assert set(tokenizer_info.special_token_ids) == expected_special_tokens


@pytest.mark.hf_token_required
@pytest.mark.parametrize(
    "tokenizer_path", ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct"]
)
def test_customize_stop_token_ids(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, stop_token_ids=[1, 2, 3])
    assert tokenizer_info.stop_token_ids == [1, 2, 3]


@pytest.mark.hf_token_required
@pytest.mark.parametrize(
    "tokenizer_path", ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct"]
)
def test_padding_vocab_size(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    original_vocab_size = len(tokenizer.get_vocab())
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=original_vocab_size + 5
    )
    assert tokenizer_info.vocab_size == original_vocab_size + 5
    assert tokenizer_info.special_token_ids[-5:] == [original_vocab_size + i for i in range(5)]


tokenizer_path__model_vocab_size = [
    ("meta-llama/Llama-3.2-11B-Vision-Instruct", 128256),
    ("meta-llama/Llama-Guard-3-11B-Vision", 128256),
    ("allenai/Molmo-72B-0924", 152064),
]


@pytest.mark.hf_token_required
@pytest.mark.parametrize("tokenizer_path, model_vocab_size", tokenizer_path__model_vocab_size)
def test_model_vocab_size_smaller_than_tokenizer(tokenizer_path: str, model_vocab_size: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    original_vocab_size = len(tokenizer.get_vocab())
    assert original_vocab_size > model_vocab_size
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model_vocab_size)
    assert tokenizer_info.vocab_size == model_vocab_size
    assert len(tokenizer_info.decoded_vocab) == model_vocab_size
    print(tokenizer_info.special_token_ids)
    print(len(tokenizer_info.decoded_vocab))


if __name__ == "__main__":
    pytest.main(sys.argv)
