import sys
from typing import List

import pytest
from transformers import AutoTokenizer

from xgrammar import TokenizerInfo, VocabType

tokenizer_paths = [
    "luodian/llama-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "lmsys/vicuna-7b-v1.5",
    "NousResearch/Hermes-2-Theta-Llama-3-70B",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "google/gemma-2b-it",
    "CohereForAI/aya-23-8B",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "deepseek-ai/DeepSeek-V2-Chat-0628",
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen2-7B-Instruct",
    "microsoft/Phi-3-small-8k-instruct",
    "Qwen/Qwen-7B-Chat",
]

# [vocab_type, prepend_space_in_tokenization]
tokenizer_metadata = [
    [VocabType.BYTE_FALLBACK, True],
    [VocabType.BYTE_FALLBACK, True],
    [VocabType.BYTE_LEVEL, False],
    [VocabType.BYTE_LEVEL, False],
    [VocabType.BYTE_FALLBACK, False],
    [VocabType.BYTE_LEVEL, False],
    [VocabType.BYTE_LEVEL, False],
    [VocabType.BYTE_FALLBACK, False],
    [VocabType.BYTE_LEVEL, False],
    [VocabType.BYTE_LEVEL, False],
    [VocabType.BYTE_LEVEL, False],
    [VocabType.BYTE_LEVEL, False],
    [VocabType.BYTE_LEVEL, False],
    [VocabType.BYTE_FALLBACK, True],
    [VocabType.BYTE_FALLBACK, True],
    [VocabType.BYTE_LEVEL, False],
    [VocabType.BYTE_LEVEL, False],
    [VocabType.RAW, False],
    [VocabType.RAW, False],
]


@pytest.mark.parametrize(
    "tokenizer_path, vocab_type, prepend_space_in_tokenization",
    list(zip(tokenizer_paths, *zip(*tokenizer_metadata))),
)
def test_properties(
    tokenizer_path: str, vocab_type: VocabType, prepend_space_in_tokenization: bool
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    assert tokenizer_info.vocab_size == len(tokenizer.get_vocab())
    assert tokenizer_info.vocab_type == vocab_type
    assert tokenizer_info.prepend_space_in_tokenization == prepend_space_in_tokenization


@pytest.mark.parametrize("tokenizer_path", tokenizer_paths)
def test_decoded_vocab(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    decoded_vocab = tokenizer_info.decoded_vocab
    assert isinstance(decoded_vocab, list)
    assert all(isinstance(token, bytes) for token in decoded_vocab)
    assert len(decoded_vocab) == len(tokenizer.get_vocab())
    assert len(decoded_vocab) == tokenizer_info.vocab_size


tokenizer_paths_token_ids_raw_tokens = [
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


@pytest.mark.parametrize(
    "tokenizer_path, token_ids, raw_tokens",
    tokenizer_paths_token_ids_raw_tokens,
)
def test_vocab_conversion(tokenizer_path: str, token_ids: List[int], raw_tokens: List[bytes]):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    vocab = tokenizer_info.decoded_vocab
    for token_id, raw_token in zip(token_ids, raw_tokens):
        assert vocab[token_id] == raw_token


tokenizer_path_metadata_str = [
    (
        "microsoft/Phi-3-small-8k-instruct",
        '{"vocab_type":"RAW","prepend_space_in_tokenization":false}',
    ),
    (
        "meta-llama/Llama-2-7b-chat-hf",
        '{"vocab_type":"BYTE_FALLBACK","prepend_space_in_tokenization":true}',
    ),
    (
        "meta-llama/Meta-Llama-3-8B-Instruct",
        '{"vocab_type":"BYTE_LEVEL","prepend_space_in_tokenization":false}',
    ),
]


@pytest.mark.parametrize("tokenizer_path, metadata_str", tokenizer_path_metadata_str)
def test_dump_metadata_load(tokenizer_path: str, metadata_str: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    assert tokenizer_info.dump_metadata() == metadata_str

    encoded_vocab = tokenizer.get_vocab()
    encoded_vocab = [token for token, _ in sorted(encoded_vocab.items(), key=lambda x: x[1])]

    loaded = TokenizerInfo.from_vocab_and_metadata(encoded_vocab, metadata_str)
    assert loaded.decoded_vocab == tokenizer_info.decoded_vocab

    loaded_new = TokenizerInfo(tokenizer_info.decoded_vocab)
    assert loaded_new.decoded_vocab == tokenizer_info.decoded_vocab


if __name__ == "__main__":
    pytest.main(sys.argv)
