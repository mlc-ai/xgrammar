"""Token jump-forward keeps accepted boundaries and withholds only a prefix tail token."""

import itertools
import json
import random

import pytest
import torch

import xgrammar as xgr


def make_matcher(vocab, grammar, *, vocab_type=xgr.VocabType.RAW, stop_token_ids=None, **kwargs):
    info = xgr.TokenizerInfo(vocab, vocab_type=vocab_type, stop_token_ids=stop_token_ids)
    compiled = xgr.GrammarCompiler(info).compile_grammar(xgr.Grammar.from_ebnf(grammar))
    return xgr.GrammarMatcher(compiled, **kwargs), info


@pytest.mark.parametrize(
    "vocab, grammar, expected",
    [
        (["ab", "c"], 'root ::= "abc" [xy]', [0, 1]),
        # The prefix test is vocabulary-wide, even when "cd" is not grammar-valid.
        (["ab", "c", "cd"], 'root ::= "abc" [xy]', [0]),
        (["abc", "abcd"], 'root ::= "abc"', []),
        (["abc", "abc"], 'root ::= "abc"', [0]),
        (["abc", "abc", "abcd"], 'root ::= "abc"', []),
        (["a", "ab", "abc", "d"], 'root ::= "abcd"', [2, 3]),
        (["a", "ab", "c", "cd"], 'root ::= "ac"', [0]),
        (["ab", "c", "bcx"], 'root ::= "abc" [xy]', [0, 1]),
        (["abc", "zabcq"], 'root ::= "abc"', [0]),
        # Missing bytes stop tokenization; they must never be skipped.
        (["ab", "d"], 'root ::= "abcd"', [0]),
        (["ab", "abcd", "e"], 'root ::= "abce"', []),
        (["ab", "abc", "cd"], 'root ::= "abcd"', [1]),
        (["b"], 'root ::= "ab"', []),
        ([], 'root ::= "a"', []),
        (["", "a"], 'root ::= "a"', [1]),
        (["a", "b"], "root ::= [ab]", []),
        (["a"], 'root ::= ""', []),
        (["a"], 'root ::= "a"*', []),
        (["ab", "c"], 'root ::= "ab" | "abc"', [0]),
        (["a", "b"], 'root ::= Token(0) | "b"', []),
        (["a", "b"], 'root ::= ExcludeToken(0) | "a"', []),
        (["ab", "c", "d"], 'root ::= "ab" (Token(2) | "c")', [0]),
        ([b"\xe4\xbd\xa0", b"\xe5\xa5\xbd"], 'root ::= "你好"', [0, 1]),
        ([b"\xe4", b"\xbd", b"\xa0"], 'root ::= "你"', [0, 1, 2]),
        # A forced byte prefix need not itself be valid UTF-8.
        ([b"\xe4", b"\xbd"], 'root ::= "你" | "佡"', [0, 1]),
        ([b"\xe4", b"\xbd", b"\xbd\xa0"], 'root ::= "你" | "佡"', [0]),
        ([b"\x00", b"a"], 'root ::= "a"', [1]),
        ([b"a", b"a\x00"], 'root ::= "a"', []),
    ],
)
def test_tail_rule(vocab, grammar, expected):
    matcher, info = make_matcher(vocab, grammar)
    before = matcher._debug_print_internal_state()
    assert matcher.find_jump_forward_tokens() == expected
    assert matcher.find_jump_forward_tokens() == expected
    assert matcher._debug_print_internal_state() == before
    for token_id in expected:
        assert matcher.accept_token(token_id)
        assert info.decoded_vocab[token_id]


@pytest.mark.parametrize(
    "vocab, vocab_type, expected",
    [
        (["Ġhello", "!"], xgr.VocabType.BYTE_LEVEL, [0, 1]),
        (["▁hello", "!"], xgr.VocabType.BYTE_FALLBACK, [0, 1]),
        (["<0x20>", "hello", "!"], xgr.VocabType.BYTE_FALLBACK, [0, 1, 2]),
    ],
)
def test_decoded_vocabulary(vocab, vocab_type, expected):
    matcher, _ = make_matcher(vocab, 'root ::= " hello!"', vocab_type=vocab_type)
    assert matcher.find_jump_forward_tokens() == expected


def test_preserves_accepted_boundary():
    matcher, _ = make_matcher(["a", "bc", "abc", "x", "y"], 'root ::= "abc" [xy]')
    assert matcher.accept_token(0)
    assert matcher.find_jump_forward_tokens() == [1]
    assert matcher.accept_token(1)
    assert matcher.find_jump_forward_tokens() == []
    matcher.rollback()
    assert matcher.find_jump_forward_tokens() == [1]
    matcher.rollback()
    assert matcher.find_jump_forward_tokens() == [2]


@pytest.mark.parametrize("terminate_without_stop_token", [False, True])
def test_completion_stop_tokens_and_reset(terminate_without_stop_token):
    matcher, _ = make_matcher(
        ["a", "ab", ""],
        'root ::= "a"',
        stop_token_ids=[1],
        terminate_without_stop_token=terminate_without_stop_token,
    )
    # A stop token's spelling is not an ordinary extension of "a".
    assert matcher.find_jump_forward_tokens() == [0]
    assert matcher.accept_token(0)
    assert matcher.is_completed()
    assert matcher.find_jump_forward_tokens() == []
    if not terminate_without_stop_token:
        assert not matcher.is_terminated()
        assert matcher.accept_token(1)
    assert matcher.is_terminated()
    assert matcher.find_jump_forward_tokens() == []
    matcher.reset()
    assert matcher.find_jump_forward_tokens() == [0]


def test_override_stop_and_duplicate_tokens():
    matcher, _ = make_matcher(["a", "a", "b"], 'root ::= "ab"', override_stop_tokens=[0])
    assert matcher.find_jump_forward_tokens() == [1, 2]
    for token_id in [1, 2]:
        assert matcher.accept_token(token_id)
    assert matcher.accept_token(0)
    assert matcher.find_jump_forward_tokens() == []


def test_override_stop_is_not_a_text_extension():
    matcher, _ = make_matcher(["a", "ab"], 'root ::= "a"', override_stop_tokens=[1])
    assert matcher.find_jump_forward_tokens() == [0]


@pytest.mark.parametrize("roundtrip", [False, True])
def test_captures_masks_fork_and_serialization(roundtrip):
    info = xgr.TokenizerInfo(["a", "b", "c", "cd", "x", "y", ""], vocab_size=32)
    grammar = xgr.Grammar.from_ebnf('root ::= item [xy]\nitem[capture="value"] ::= "abc"')
    compiled = xgr.GrammarCompiler(info).compile_grammar(grammar)
    if roundtrip:
        info = xgr.TokenizerInfo.deserialize_json(info.serialize_json())
        compiled = xgr.CompiledGrammar.deserialize_json(compiled.serialize_json(), info)
    matcher = xgr.GrammarMatcher(compiled)
    control = matcher.fork()
    before = xgr.allocate_token_bitmask(1, info.vocab_size)
    after = xgr.allocate_token_bitmask(1, info.vocab_size)
    control.fill_next_token_bitmask(before)
    assert matcher.find_jump_forward_tokens() == [0, 1]
    assert matcher.get_captures() == []
    matcher.fill_next_token_bitmask(after)
    assert torch.equal(before, after)
    for token_id in [0, 1, 2]:
        assert matcher.accept_token(token_id)
        assert control.accept_token(token_id)
    assert matcher.get_captures() == control.get_captures() == [("value", b"abc")]
    matcher.rollback(3)
    assert matcher.get_captures() == []
    assert matcher.find_jump_forward_tokens() == [0, 1]


def test_budget_enforcement_state_is_preserved():
    matcher, info = make_matcher(["a", "x", "!"], 'root ::= body "!"\nbody[max_tokens=1] ::= [ax]*')
    assert matcher.accept_token(0)
    bitmask = xgr.allocate_token_bitmask(1, info.vocab_size)
    matcher.fill_next_token_bitmask(bitmask)
    control = matcher.fork()
    assert matcher.find_jump_forward_tokens() == [2]
    assert matcher.find_jump_forward_tokens() == [2]
    assert not matcher.accept_token(1)
    assert not control.accept_token(1)
    assert matcher.accept_token(2)
    assert control.accept_token(2)
    assert matcher.is_completed() and control.is_completed()


def test_token_edges_do_not_force_a_byte_alternative():
    matcher, _ = make_matcher(["a", "b"], 'root ::= Token(0) | "b"')
    assert matcher.find_jump_forward_string() == ""
    assert matcher.find_jump_forward_tokens() == []
    assert matcher.accept_token(0)


def test_sequences_remain_grammar_valid():
    """Exercise many overlapping vocabularies against finite grammar languages."""
    rng = random.Random(0)
    words = [
        "".join(chars) for length in range(1, 5) for chars in itertools.product("ab", repeat=length)
    ]
    for _ in range(100):
        vocab = ["a", "b"] + rng.sample(words, 10)
        prefix = rng.choice(words)
        alternatives = [prefix + word for word in rng.sample(words, 5)]
        grammar = "root ::= " + " | ".join(json.dumps(word) for word in alternatives)
        matcher, info = make_matcher(vocab, grammar)
        accepted = b""
        while not matcher.is_completed():
            candidates = matcher.find_jump_forward_tokens()
            for token_id in candidates:
                viable = [
                    word.encode() for word in alternatives if word.encode().startswith(accepted)
                ]
                assert all(
                    word.startswith(accepted + info.decoded_vocab[token_id]) for word in viable
                )
                assert matcher.accept_token(token_id)
                accepted += info.decoded_vocab[token_id]
                assert any(word.encode().startswith(accepted) for word in alternatives)
            if matcher.is_completed():
                break
            # Choose a legal single-byte continuation when the query has no more forced tokens.
            token_id = next(
                i
                for i in [0, 1]
                if any(
                    word.encode().startswith(accepted + info.decoded_vocab[i])
                    for word in alternatives
                )
            )
            assert matcher.accept_token(token_id)
            accepted += info.decoded_vocab[token_id]
        assert accepted.decode() in alternatives
