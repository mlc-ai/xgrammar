# pylint: disable=missing-module-docstring,missing-function-docstring
import pytest
from transformers import AutoTokenizer
from xgrammar import TokenizerInfo

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
]

tokenizer_path_expected = list(
    zip(
        tokenizer_paths,
        [
            '{"token_postproc_method":"byte_fallback","prepend_space_in_encode":true}',
            '{"token_postproc_method":"byte_fallback","prepend_space_in_encode":true}',
            '{"token_postproc_method":"byte_level","prepend_space_in_encode":false}',
            '{"token_postproc_method":"byte_level","prepend_space_in_encode":false}',
            '{"token_postproc_method":"byte_fallback","prepend_space_in_encode":false}',
            '{"token_postproc_method":"byte_level","prepend_space_in_encode":false}',
            '{"token_postproc_method":"byte_level","prepend_space_in_encode":false}',
            '{"token_postproc_method":"byte_fallback","prepend_space_in_encode":false}',
            '{"token_postproc_method":"byte_level","prepend_space_in_encode":false}',
            '{"token_postproc_method":"byte_level","prepend_space_in_encode":false}',
            '{"token_postproc_method":"byte_level","prepend_space_in_encode":false}',
            '{"token_postproc_method":"byte_level","prepend_space_in_encode":false}',
            '{"token_postproc_method":"byte_level","prepend_space_in_encode":false}',
            '{"token_postproc_method":"byte_fallback","prepend_space_in_encode":true}',
            '{"token_postproc_method":"byte_fallback","prepend_space_in_encode":true}',
            '{"token_postproc_method":"byte_level","prepend_space_in_encode":false}',
            '{"token_postproc_method":"byte_level","prepend_space_in_encode":false}',
        ],
    )
)


@pytest.mark.parametrize("tokenizer_path,expected", tokenizer_path_expected)
def test_tokenizer_info(tokenizer_path: str, expected: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer_info = TokenizerInfo(tokenizer)
    assert str(tokenizer_info) == expected


@pytest.mark.parametrize("tokenizer_path", tokenizer_paths)
def test_get_decoded_token_table(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer_info = TokenizerInfo(tokenizer)
    raw_vocab = tokenizer.get_vocab()
    decoded_token_table = tokenizer_info.get_decoded_token_table(raw_vocab)
    assert isinstance(decoded_token_table, list)
    assert all(isinstance(token, bytes) for token in decoded_token_table)
    assert len(decoded_token_table) == len(raw_vocab)


if __name__ == "__main__":
    pytest.main([__file__])
