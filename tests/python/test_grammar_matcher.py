"""This test uses the optimized JSON grammar provided by the grammar library."""

import sys
from typing import List, Optional

import pytest
import torch
from transformers import AutoTokenizer

from xgrammar import BNFGrammar, BuiltinGrammar, GrammarMatcher, TokenizerInfo

json_grammar = BuiltinGrammar.json()


def match_complete_string(grammar: BNFGrammar, input_str: str) -> bool:
    matcher = GrammarMatcher(grammar, terminate_without_stop_token=True)
    can_accept = matcher.accept_string(input_str)
    can_terminate = matcher.is_terminated()
    return can_accept and can_terminate


input_accepted = [
    '{"name": "John"}',
    '{ "name" : "John" }',
]


@pytest.mark.parametrize("input_accepted", input_accepted)
def test_accept(input_accepted: str):
    assert match_complete_string(json_grammar, input_accepted)


input_refused = (
    '{ name: "John" }',
    '{ "name": "John" } ',
)


@pytest.mark.parametrize("input_refused", input_refused)
def test_refuse(input_refused: str):
    assert not match_complete_string(json_grammar, input_refused)


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
def test_get_next_rejected_tokens(
    tokenizer_path: str,
    input_str: str,
    expected_rejected_sizes: Optional[List[int]],
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    matcher = GrammarMatcher(json_grammar, tokenizer)
    input_bytes = input_str.encode("utf-8")
    rejected_sizes = []

    for i, c in enumerate(input_bytes):
        bitmask = matcher.get_next_token_bitmask()
        rejected_token_ids = GrammarMatcher.get_rejected_tokens_from_bitmask(
            bitmask, matcher.mask_vocab_size
        )
        rejected_sizes.append(len(rejected_token_ids))
        if expected_rejected_sizes is not None:
            assert rejected_sizes[-1] == expected_rejected_sizes[i], (
                rejected_sizes[-1],
                expected_rejected_sizes[i],
            )
        assert matcher.accept_string(bytes([c]))

    bitmask = matcher.get_next_token_bitmask()
    rejected_token_ids = GrammarMatcher.get_rejected_tokens_from_bitmask(
        bitmask, matcher.mask_vocab_size
    )
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

    matcher = GrammarMatcher(json_grammar, vocab)

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
        bitmask = matcher.get_next_token_bitmask()
        rejected_token_ids = GrammarMatcher.get_rejected_tokens_from_bitmask(
            bitmask, matcher.mask_vocab_size
        )
        accepted = list(set(range(len(vocab))) - set(rejected_token_ids))
        accepted_tokens = [vocab[i] for i in accepted]
        result.append(accepted_tokens)
        assert id in accepted, vocab[id]
        assert matcher.accept_token(id)

    bitmask = matcher.get_next_token_bitmask()
    rejected_token_ids = GrammarMatcher.get_rejected_tokens_from_bitmask(
        bitmask, matcher.mask_vocab_size
    )
    accepted = list(set(range(len(vocab))) - set(rejected_token_ids))
    accepted_tokens = [vocab[i] for i in accepted]
    result.append(accepted_tokens)

    assert result == expected


def test_apply_token_bitmask_inplace():
    neginf = float("-inf")
    bool_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    expected = torch.where(bool_mask, logits, neginf)
    bitmask = torch.tensor([0b1010101010], dtype=torch.int32)

    GrammarMatcher.apply_token_bitmask_inplace(logits, bitmask)
    assert torch.all(logits == expected)


def test_rollback():
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"]
    input_ids = [vocab.index(t) for t in input_splitted]

    matcher = GrammarMatcher(json_grammar, vocab, max_rollback_tokens=5)

    assert matcher.max_rollback_tokens == 5

    input_ids_splitted = [input_ids[i : i + 2] for i in range(0, len(input_ids), 2)]

    for i_1, i_2 in input_ids_splitted:
        orig_result = []
        orig_result.append(matcher.get_next_token_bitmask())
        assert matcher.accept_token(i_1)
        orig_result.append(matcher.get_next_token_bitmask())
        assert matcher.accept_token(i_2)
        matcher.rollback(2)
        result_after_rollback = []
        result_after_rollback.append(matcher.get_next_token_bitmask())
        assert matcher.accept_token(i_1)
        result_after_rollback.append(matcher.get_next_token_bitmask())
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

    matcher = GrammarMatcher(json_grammar, vocab)

    orig_result = []

    for i in input_ids:
        orig_result.append(matcher.get_next_token_bitmask())
        assert matcher.accept_token(i)

    matcher.reset()

    result_after_reset = []

    for i in input_ids:
        result_after_reset.append(matcher.get_next_token_bitmask())
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

    matcher = GrammarMatcher(json_grammar, vocab, max_rollback_tokens=5)

    for i in input_ids:
        matcher.get_next_token_bitmask()
        assert matcher.accept_token(i)

    assert matcher.is_terminated()

    assert matcher.accept_token(0) is False

    with pytest.raises(RuntimeError):
        matcher.get_next_token_bitmask()

    matcher.rollback(2)

    assert not matcher.is_terminated()
    assert matcher.accept_token(input_ids[-2])


def test_get_jump_forward_string():
    grammar_ebnf = r"""root ::= "abb" | "abbd" | other_rule
other_rule ::= "a" sub_rule "b"
sub_rule ::= "b"
"""
    grammar = BNFGrammar(grammar_ebnf)
    matcher = GrammarMatcher(grammar)
    assert matcher.accept_string("a")
    assert matcher.find_jump_forward_string() == "bb"


def test_mask_vocab_size():
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    matcher = GrammarMatcher(json_grammar, vocab, mask_vocab_size=64)
    assert matcher.mask_vocab_size == 64

    mask = matcher.get_next_token_bitmask()
    assert mask.shape == (2,)

    rejected_tokens = GrammarMatcher.get_rejected_tokens_from_bitmask(mask, matcher.mask_vocab_size)
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
    tokenizer_info = TokenizerInfo.from_huggingface(tokenizer)
    matcher = GrammarMatcher(
        json_grammar, tokenizer_info, override_stop_tokens=override_stop_tokens
    )
    assert matcher.stop_token_ids == override_stop_tokens


if __name__ == "__main__":
    pytest.main(sys.argv)
