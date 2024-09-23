"""This test uses the optimized JSON grammar provided by the grammar library."""

from typing import List, Optional

import pytest
import torch
from transformers import AutoTokenizer
from xgrammar import BNFGrammar, BuiltinGrammar, GrammarStateMatcher

json_grammar = BuiltinGrammar.json()


def match_complete_string(grammar: BNFGrammar, input_str: str) -> bool:
    matcher = GrammarStateMatcher(grammar, terminate_without_stop_token=True)
    can_accept = matcher._accept_string(input_str)
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
    ("tokenizer_path", "input_str", "expected_rejected_sizes"),
    tokenizer_path__input_str__expected_rejected_sizes,
)
def test_find_next_rejected_tokens(
    tokenizer_path: str,
    input_str: str,
    expected_rejected_sizes: Optional[List[int]],
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )
    matcher = GrammarStateMatcher(json_grammar, tokenizer)
    input_bytes = input_str.encode("utf-8")
    rejected_sizes = []

    for i, c in enumerate(input_bytes):
        bitmask = matcher.find_next_token_bitmask()
        rejected_token_ids = GrammarStateMatcher.get_rejected_tokens_from_bitmask(
            bitmask, matcher.vocab_size
        )
        rejected_sizes.append(len(rejected_token_ids))
        if expected_rejected_sizes is not None:
            assert rejected_sizes[-1] == expected_rejected_sizes[i], (
                rejected_sizes[-1],
                expected_rejected_sizes[i],
            )
        assert matcher._accept_string(bytes([c]))

    bitmask = matcher.find_next_token_bitmask()
    rejected_token_ids = GrammarStateMatcher.get_rejected_tokens_from_bitmask(
        bitmask, matcher.vocab_size
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

    matcher = GrammarStateMatcher(json_grammar, vocab)

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
        bitmask = matcher.find_next_token_bitmask()
        rejected_token_ids = GrammarStateMatcher.get_rejected_tokens_from_bitmask(
            bitmask, matcher.vocab_size
        )
        accepted = list(set(range(len(vocab))) - set(rejected_token_ids))
        accepted_tokens = [vocab[i] for i in accepted]
        result.append(accepted_tokens)
        assert id in accepted, vocab[id]
        assert matcher.accept_token(id)

    bitmask = matcher.find_next_token_bitmask()
    rejected_token_ids = GrammarStateMatcher.get_rejected_tokens_from_bitmask(
        bitmask, matcher.vocab_size
    )
    accepted = list(set(range(len(vocab))) - set(rejected_token_ids))
    accepted_tokens = [vocab[i] for i in accepted]
    result.append(accepted_tokens)

    assert result == expected


def test_rollback():
    vocab = [
        # fmt: off
        "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
        # fmt: on
    ]
    input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"]
    input_ids = [vocab.index(t) for t in input_splitted]

    matcher = GrammarStateMatcher(json_grammar, vocab, max_rollback_steps=5)

    assert matcher.max_rollback_steps == 5

    input_ids_splitted = [input_ids[i : i + 2] for i in range(0, len(input_ids), 2)]

    for i_1, i_2 in input_ids_splitted:
        orig_result = []
        orig_result.append(matcher.find_next_token_bitmask())
        assert matcher.accept_token(i_1)
        orig_result.append(matcher.find_next_token_bitmask())
        assert matcher.accept_token(i_2)
        matcher.rollback(2)
        result_after_rollback = []
        result_after_rollback.append(matcher.find_next_token_bitmask())
        assert matcher.accept_token(i_1)
        result_after_rollback.append(matcher.find_next_token_bitmask())
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

    matcher = GrammarStateMatcher(json_grammar, vocab)

    orig_result = []

    for i in input_ids:
        orig_result.append(matcher.find_next_token_bitmask())
        assert matcher.accept_token(i)

    matcher.reset()

    result_after_reset = []

    for i in input_ids:
        result_after_reset.append(matcher.find_next_token_bitmask())
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

    matcher = GrammarStateMatcher(json_grammar, vocab, max_rollback_steps=5)

    for i in input_ids:
        matcher.find_next_token_bitmask()
        assert matcher.accept_token(i)

    assert matcher.is_terminated()

    assert matcher.accept_token(0) is False

    with pytest.raises(RuntimeError):
        matcher.find_next_token_bitmask()

    matcher.rollback(2)

    assert not matcher.is_terminated()
    assert matcher.accept_token(input_ids[-2])


def test_get_jump_forward_string():
    grammar_ebnf = r"""main ::= "abb" | "abbd" | other_rule
other_rule ::= "a" sub_rule "b"
sub_rule ::= "b"
"""
    grammar = BNFGrammar(grammar_ebnf)
    matcher = GrammarStateMatcher(grammar)
    assert matcher._accept_string("a")
    assert matcher.find_jump_forward_string() == "bb"


if __name__ == "__main__":
    pytest.main([__file__])
