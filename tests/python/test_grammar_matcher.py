"""This test uses the optimized JSON grammar provided by the grammar library."""

import sys
import time
from typing import List, Optional

import pytest
import torch
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.testing import (
    _allocate_token_bitmask,
    _get_masked_tokens_from_bitmask,
    _get_matcher_from_grammar_and_tokenizer_info,
    _match_grammar_with_string,
)

json_grammar = xgr.BNFGrammar.builtin_json_grammar()


input_accepted = [
    '{"name": "John"}',
    '{ "name" : "John" }',
]


@pytest.mark.parametrize("input_accepted", input_accepted)
def test_accept(input_accepted: str):
    assert _match_grammar_with_string(json_grammar, input_accepted)


input_refused = (
    '{ name: "John" }',
    '{ "name": "John" } ',
)


@pytest.mark.parametrize("input_refused", input_refused)
def test_refuse(input_refused: str):
    assert not _match_grammar_with_string(json_grammar, input_refused)


tokenizer_path__input_str__expected_rejected_sizes = [
    (
        "meta-llama/Llama-2-7b-chat-hf",
        '{"id": 1,"name": "Example"}',
        [
            # fmt: off
            31989, 31912, 272, 272, 272, 31973, 31846, 31846, 31948, 31915, 272, 272, 272, 272,
            272, 31973, 31846, 31846, 265, 265, 265, 265, 265, 265, 265, 265, 31974, 31999,
            # fmt: on
        ],
    ),
    (
        # test for llama 3
        "meta-llama/Meta-Llama-3-8B-Instruct",
        '{"id": 1,"name": "Example哈哈"}',
        [
            # fmt: off
            128235, 127497, 5002, 5002, 5002, 127849, 126399, 126399, 126760, 127499, 5002, 5002,
            5002, 5002, 5002, 127849, 126399, 126399, 4952, 4952, 4952, 4952, 4952, 4952, 4952,
            4952, 128066, 128111, 4952, 128066, 128111, 4952, 127873, 128254,
            # fmt: on
        ],
    ),
]


@pytest.mark.parametrize(
    "tokenizer_path, input_str, expected_rejected_sizes",
    tokenizer_path__input_str__expected_rejected_sizes,
)
def test_fill_next_token_bitmask(
    tokenizer_path: str,
    input_str: str,
    expected_rejected_sizes: Optional[List[int]],
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(json_grammar, tokenizer_info)

    token_bitmask = _allocate_token_bitmask(1, tokenizer_info.vocab_size)

    input_bytes = input_str.encode("utf-8")
    rejected_sizes = []

    for i, c in enumerate(input_bytes):
        matcher.fill_next_token_bitmask(token_bitmask)
        rejected_token_ids = _get_masked_tokens_from_bitmask(
            token_bitmask, tokenizer_info.vocab_size
        )
        rejected_sizes.append(len(rejected_token_ids))
        if expected_rejected_sizes is not None:
            assert rejected_sizes[-1] == expected_rejected_sizes[i], (
                rejected_sizes[-1],
                expected_rejected_sizes[i],
            )
        assert matcher._debug_accept_string(bytes([c]))

    matcher.fill_next_token_bitmask(token_bitmask)
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    rejected_sizes.append(len(rejected_token_ids))
    if expected_rejected_sizes is not None:
        assert rejected_sizes[-1] == expected_rejected_sizes[-1]


def test_token_operations():
    """Test accepting token and finding the next token mask."""
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"]
    input_ids = [vocab.index(t) for t in input_splitted]

    tokenizer_info = xgr.TokenizerInfo(vocab)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(json_grammar, tokenizer_info)
    token_bitmask = _allocate_token_bitmask(1, tokenizer_info.vocab_size)

    expected = [
        ["{"],
        ['"', "}", "\n", " ", '"a":true'],
        ["a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
        ["a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
        [":", "\n", " ", ':"'],
        ['"', "{", "6", "\n", " "],
        ["}", ", ", "6", "\n", " "],
        [" ", "\n", '"', '"a":true'],
        [" ", "\n", '"', '"a":true'],
        ["}", ", ", "\n", " "],
        ["</s>"],
    ]

    result = []

    for id in input_ids:
        matcher.fill_next_token_bitmask(token_bitmask)
        rejected_token_ids = _get_masked_tokens_from_bitmask(
            token_bitmask, tokenizer_info.vocab_size
        )
        accepted = list(set(range(len(vocab))) - set(rejected_token_ids))
        accepted_tokens = [vocab[i] for i in accepted]
        result.append(accepted_tokens)
        assert id in accepted, vocab[id]
        assert matcher.accept_token(id)

    matcher.fill_next_token_bitmask(token_bitmask)
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    accepted = list(set(range(len(vocab))) - set(rejected_token_ids))
    accepted_tokens = [vocab[i] for i in accepted]
    result.append(accepted_tokens)

    assert result == expected


def test_apply_token_bitmask_inplace():
    neginf = float("-inf")
    bool_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32)
    expected = torch.where(bool_mask, logits, neginf)

    logits_gpu = logits.to("cuda")
    bitmask = torch.tensor([0b1010101010], dtype=torch.int32).to("cuda")
    xgr.apply_token_bitmask_inplace(logits_gpu, bitmask)
    torch.cuda.synchronize()
    assert torch.all(logits_gpu == expected.to("cuda"))


def test_apply_token_bitmask_inplace_large():
    batch_size = 64
    vocab_size = 128000
    masked_cnt = 64000
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    masked_positions = torch.randint(0, vocab_size, (batch_size, masked_cnt))
    bool_mask = torch.ones((batch_size, vocab_size), dtype=torch.bool)
    bool_mask.scatter_(1, masked_positions, False)

    neginf = float("-inf")
    logits_expected = torch.where(bool_mask, logits, neginf)

    def bool_mask_to_bitmask(bool_mask: torch.Tensor) -> torch.Tensor:
        bool_mask_int32 = bool_mask.to(torch.int32)
        bool_mask_view = bool_mask_int32.view(batch_size, -1, 32)
        weights = torch.tensor([1 << i for i in range(32)], dtype=torch.int64)
        weights = weights.to(torch.int32)
        bitmask = (bool_mask_view * weights).sum(dim=2)
        return bitmask.to(torch.int32)

    bitmask = bool_mask_to_bitmask(bool_mask)
    logits_gpu = logits.to("cuda")
    torch.cuda.synchronize()
    time_start = time.monotonic_ns()
    xgr.apply_token_bitmask_inplace(logits_gpu, bitmask)
    torch.cuda.synchronize()
    time_end = time.monotonic_ns()
    print(f"Time taken: {(time_end - time_start) / 1e3} us")
    assert torch.all(logits_gpu == logits_expected.to("cuda"))


def test_rollback():
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"]
    input_ids = [vocab.index(t) for t in input_splitted]

    tokenizer_info = xgr.TokenizerInfo(vocab)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(
        json_grammar, tokenizer_info, max_rollback_tokens=5
    )

    assert matcher.max_rollback_tokens == 5

    input_ids_splitted = [input_ids[i : i + 2] for i in range(0, len(input_ids), 2)]

    for i_1, i_2 in input_ids_splitted:
        orig_result = []
        token_bitmask1 = _allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(token_bitmask1)
        orig_result.append(token_bitmask1)
        assert matcher.accept_token(i_1)
        token_bitmask2 = _allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(token_bitmask2)
        orig_result.append(token_bitmask2)
        assert matcher.accept_token(i_2)

        matcher.rollback(2)
        result_after_rollback = []
        new_token_bitmask1 = _allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(new_token_bitmask1)
        result_after_rollback.append(new_token_bitmask1)
        assert matcher.accept_token(i_1)
        new_token_bitmask2 = _allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(new_token_bitmask2)
        result_after_rollback.append(new_token_bitmask2)
        assert matcher.accept_token(i_2)
        assert all(torch.all(l == r) for l, r in zip(orig_result, result_after_rollback))


def test_reset():
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"]
    input_ids = [vocab.index(t) for t in input_splitted]

    tokenizer_info = xgr.TokenizerInfo(vocab)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(json_grammar, tokenizer_info)

    orig_result = []

    for i in input_ids:
        token_bitmask = _allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(token_bitmask)
        orig_result.append(token_bitmask)
        assert matcher.accept_token(i)

    matcher.reset()

    result_after_reset = []

    for i in input_ids:
        token_bitmask = _allocate_token_bitmask(1, tokenizer_info.vocab_size)
        matcher.fill_next_token_bitmask(token_bitmask)
        result_after_reset.append(token_bitmask)
        assert matcher.accept_token(i)

    assert all(torch.all(l == r) for l, r in zip(orig_result, result_after_reset))


def test_termination():
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = [
        "{",
        '"',
        "abc",
        'b"',
        ":",
        "6",
        ", ",
        " ",
        '"a":true',
        "}",
        "</s>",
    ]
    input_ids = [vocab.index(t) for t in input_splitted]
    tokenizer_info = xgr.TokenizerInfo(vocab)

    matcher = _get_matcher_from_grammar_and_tokenizer_info(
        json_grammar, tokenizer_info, max_rollback_tokens=5
    )
    token_bitmask = _allocate_token_bitmask(1, tokenizer_info.vocab_size)

    for i in input_ids:
        matcher.fill_next_token_bitmask(token_bitmask)
        assert matcher.accept_token(i)

    assert matcher.is_terminated()

    assert matcher.accept_token(0) is False

    with pytest.raises(RuntimeError):
        matcher.fill_next_token_bitmask(token_bitmask)

    matcher.rollback(2)

    assert not matcher.is_terminated()
    assert matcher.accept_token(input_ids[-2])


def test_get_jump_forward_string():
    grammar_ebnf = r"""root ::= "abb" | "abbd" | other_rule
other_rule ::= "a" sub_rule "b"
sub_rule ::= "b"
"""
    grammar = xgr.BNFGrammar.from_ebnf(grammar_ebnf)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(grammar)
    assert matcher._debug_accept_string("a")
    assert matcher.find_jump_forward_string() == "bb"


def test_vocab_size():
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    tokenizer_info = xgr.TokenizerInfo(vocab, vocab_size=64)
    matcher = _get_matcher_from_grammar_and_tokenizer_info(json_grammar, tokenizer_info)

    token_bitmask = _allocate_token_bitmask(1, tokenizer_info.vocab_size)
    matcher.fill_next_token_bitmask(token_bitmask)
    assert token_bitmask.shape == (2,)

    rejected_tokens = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert rejected_tokens == [i for i in range(64) if i != 7]


tokenizer_path_override_stop_tokens = [
    ("meta-llama/Llama-2-7b-chat-hf", [2]),
    ("meta-llama/Meta-Llama-3-8B-Instruct", [128001, 128009]),
    ("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", [100001]),
]


@pytest.mark.parametrize(
    "tokenizer_path, override_stop_tokens", tokenizer_path_override_stop_tokens
)
def test_override_stop_tokens(tokenizer_path: str, override_stop_tokens: List[int]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info_1 = xgr.TokenizerInfo.from_huggingface(
        tokenizer, stop_token_ids=override_stop_tokens
    )
    matcher_1 = _get_matcher_from_grammar_and_tokenizer_info(json_grammar, tokenizer_info_1)
    assert tokenizer_info_1.stop_token_ids == override_stop_tokens
    assert matcher_1.stop_token_ids == override_stop_tokens

    tokenizer_info_2 = xgr.TokenizerInfo.from_huggingface(tokenizer)
    matcher_2 = _get_matcher_from_grammar_and_tokenizer_info(
        json_grammar, tokenizer_info_2, override_stop_tokens=override_stop_tokens
    )
    assert matcher_2.stop_token_ids == override_stop_tokens


if __name__ == "__main__":
    pytest.main(sys.argv)
