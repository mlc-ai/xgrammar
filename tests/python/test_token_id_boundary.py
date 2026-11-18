"""Offline contract tests for token ids at and beyond the matcher's vocabulary boundary.

These tests only need the CPU build of xgrammar. They encode the contract that the
matcher's public API must satisfy for callers that sample from a logits tensor whose last
dimension can be wider than the token table the grammar was compiled against (padded
vocabularies, mixed real/padding ids, hostile or corrupted sampler output):

1. an id outside the matcher's vocabulary is rejected and leaves the matcher untouched;
2. for every id in range, "the mask allows it" and "accept_token accepts it" agree, when
   both are observed from a fresh replay of the same prefix;
3. the bitmask buffer width contract is explicit, and the tail of the logits that the
   buffer does not cover is not silently reported as masked;
4. an id that does not fit in int32 is never silently reinterpreted as a different,
   in-range id.

Failure of these tests is a contract failure, not a reproduction of any specific
downstream report: nothing here runs a language model.
"""

import math
from typing import List, Optional, Tuple

import pytest
import torch

import xgrammar as xgr
from xgrammar.testing import _get_masked_tokens_from_bitmask

SIMPLE_GRAMMAR = 'root ::= "t0" "t1" "t2"'
PREFIX_IDS = [0]
VALID_NEXT_IDS = [1]

JSON_PREFIX = b'{"a"'
JSON_PREFIX_IDS = [0, 1, 2]


def make_tokenizer_info(vocab_size: int, *, padded_to: Optional[int] = None) -> xgr.TokenizerInfo:
    """Build a deterministic synthetic vocabulary.

    Ids ``0 .. vocab_size - 2`` are ordinary tokens ``t<i>``; the last real id is ``<eos>``
    and is registered as the only stop token. Padding ids added by ``padded_to`` carry the
    empty token ``""`` and are classified as special tokens by xgrammar.
    """
    vocab = [f"t{i}" for i in range(vocab_size - 1)] + ["<eos>"]
    return xgr.TokenizerInfo(vocab, vocab_size=padded_to, stop_token_ids=[vocab_size - 1])


def make_matcher(grammar: str = SIMPLE_GRAMMAR, tokenizer_info: Optional[xgr.TokenizerInfo] = None):
    if tokenizer_info is None:
        tokenizer_info = make_tokenizer_info(32)
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(
        xgr.Grammar.from_ebnf(grammar)
    )
    return xgr.GrammarMatcher(compiled)


def allowed_ids(bitmask: torch.Tensor, vocab_size: int, index: int = 0) -> List[int]:
    """Ids the mask allows, read straight out of the int32 words."""
    return [
        i
        for i in range(vocab_size)
        if (int(bitmask[index, i // 32].item()) >> (i % 32)) & 1
    ]


def fresh_with_prefix(prefix_ids: List[int], tokenizer_info: Optional[xgr.TokenizerInfo] = None):
    matcher = make_matcher(tokenizer_info=tokenizer_info)
    for token_id in prefix_ids:
        assert matcher.accept_token(token_id), f"prefix token {token_id} unexpectedly rejected"
    return matcher


# --------------------------------------------------------------------------------------
# 1. invalid ids: rejection plus state invariance
# --------------------------------------------------------------------------------------


def out_of_range_ids(vocab_size: int) -> List[int]:
    """Out-of-vocabulary ids that still fit in int32, so they must be plain rejections."""
    return [-1, -(2**31), vocab_size, vocab_size + 1, 2**31 - 1]


def beyond_int32_ids() -> List[int]:
    """Ids that do not fit in int32 at all: never a token id, so never a plain rejection."""
    return [-(2**31) - 1, -(2**40), 2**31, 2**32, 2**32 + 1, 2**40]


@pytest.mark.parametrize("vocab_size", [31, 32, 33, 63, 64, 65])
def test_out_of_range_ids_are_rejected(vocab_size: int):
    tokenizer_info = make_tokenizer_info(vocab_size)
    matcher = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
    for bad_id in out_of_range_ids(vocab_size):
        assert matcher.accept_token(bad_id) is False, f"{bad_id} should be rejected"


@pytest.mark.parametrize("vocab_size", [31, 32, 33, 63, 64, 65])
def test_beyond_int32_ids_raise_and_do_not_disturb_state(vocab_size: int):
    """Ids outside int32 raise, and the raise leaves the matcher exactly as it was.

    A raise is not silent truncation, but it must still be a no-op on the matcher: a caller
    that catches the error must be able to keep decoding from the same prefix.
    """
    tokenizer_info = make_tokenizer_info(vocab_size)

    control = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
    control_mask = xgr.allocate_token_bitmask(1, vocab_size)
    control_need_apply = control.fill_next_token_bitmask(control_mask, 0)
    control_state = (
        control.is_terminated(),
        control.is_completed(),
        control.get_captures(),
        control.find_jump_forward_string(),
    )

    for bad_id in beyond_int32_ids():
        matcher = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
        with pytest.raises(RuntimeError, match="int32 range"):
            matcher.accept_token(bad_id)
        mask = xgr.allocate_token_bitmask(1, vocab_size)
        need_apply = matcher.fill_next_token_bitmask(mask, 0)
        assert need_apply == control_need_apply
        assert torch.equal(mask, control_mask), f"mask changed after rejecting {bad_id}"
        assert (
            matcher.is_terminated(),
            matcher.is_completed(),
            matcher.get_captures(),
            matcher.find_jump_forward_string(),
        ) == control_state, f"state changed after rejecting {bad_id}"


@pytest.mark.parametrize("vocab_size", [31, 32, 33, 63, 64, 65])
def test_rejection_does_not_change_mask_or_state(vocab_size: int):
    """A rejected id must be a pure no-op on the matcher's mask and lifecycle state."""
    tokenizer_info = make_tokenizer_info(vocab_size)

    control = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
    control_mask = xgr.allocate_token_bitmask(1, vocab_size)
    control_need_apply = control.fill_next_token_bitmask(control_mask, 0)
    control_state = (
        control.is_terminated(),
        control.is_completed(),
        control.get_captures(),
        control.find_jump_forward_string(),
    )

    for bad_id in out_of_range_ids(vocab_size):
        matcher = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
        assert matcher.accept_token(bad_id) is False
        mask = xgr.allocate_token_bitmask(1, vocab_size)
        need_apply = matcher.fill_next_token_bitmask(mask, 0)
        assert need_apply == control_need_apply
        assert torch.equal(mask, control_mask), f"mask changed after rejecting {bad_id}"
        assert (
            matcher.is_terminated(),
            matcher.is_completed(),
            matcher.get_captures(),
            matcher.find_jump_forward_string(),
        ) == control_state, f"state changed after rejecting {bad_id}"


@pytest.mark.parametrize("vocab_size", [31, 33, 65])
def test_repeated_rejection_and_recovery(vocab_size: int):
    """Many rejects in a row, then a legal token, must behave like never rejecting."""
    tokenizer_info = make_tokenizer_info(vocab_size)

    control = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
    assert control.accept_token(VALID_NEXT_IDS[0]) is True

    matcher = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
    for _ in range(8):
        for bad_id in out_of_range_ids(vocab_size):
            assert matcher.accept_token(bad_id) is False
    assert matcher.accept_token(VALID_NEXT_IDS[0]) is True

    control_mask = xgr.allocate_token_bitmask(1, vocab_size)
    matcher_mask = xgr.allocate_token_bitmask(1, vocab_size)
    control.fill_next_token_bitmask(control_mask, 0)
    matcher.fill_next_token_bitmask(matcher_mask, 0)
    assert torch.equal(control_mask, matcher_mask)


@pytest.mark.parametrize("vocab_size", [31, 33, 65])
def test_rejection_does_not_enter_token_history(vocab_size: int):
    """Rejected ids must not consume a rollback step.

    After a rejection, rolling back one token removes the last *accepted* token, and a
    reset matcher and an equal fork stay in the same state as an untreated control.
    """
    tokenizer_info = make_tokenizer_info(vocab_size)

    matcher = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
    assert matcher.accept_token(VALID_NEXT_IDS[0]) is True
    assert matcher.accept_token(vocab_size) is False
    assert matcher.accept_token(-1) is False
    matcher.rollback(1)

    rolled_back = xgr.allocate_token_bitmask(1, vocab_size)
    matcher.fill_next_token_bitmask(rolled_back, 0)
    control = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
    control_mask = xgr.allocate_token_bitmask(1, vocab_size)
    control.fill_next_token_bitmask(control_mask, 0)
    assert torch.equal(rolled_back, control_mask), "rejected ids leaked into token history"


@pytest.mark.parametrize("vocab_size", [31, 33])
def test_reset_and_fork_after_rejection(vocab_size: int):
    tokenizer_info = make_tokenizer_info(vocab_size)

    matcher = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
    assert matcher.accept_token(vocab_size) is False

    forked = matcher.fork()
    forked_mask = xgr.allocate_token_bitmask(1, vocab_size)
    matcher_mask = xgr.allocate_token_bitmask(1, vocab_size)
    matcher.fill_next_token_bitmask(matcher_mask, 0)
    forked.fill_next_token_bitmask(forked_mask, 0)
    assert torch.equal(matcher_mask, forked_mask)

    matcher.reset()
    reset_mask = xgr.allocate_token_bitmask(1, vocab_size)
    matcher.fill_next_token_bitmask(reset_mask, 0)
    pristine = make_matcher(tokenizer_info=tokenizer_info)
    pristine_mask = xgr.allocate_token_bitmask(1, vocab_size)
    pristine.fill_next_token_bitmask(pristine_mask, 0)
    assert torch.equal(reset_mask, pristine_mask)


def test_rejection_on_terminated_matcher(vocab_size: int = 31):
    """A terminated matcher rejects every further id, including out-of-range ones."""
    tokenizer_info = make_tokenizer_info(vocab_size)
    # This grammar completes after a single token, so the stop token becomes admissible.
    matcher = make_matcher('root ::= "t0"', tokenizer_info=tokenizer_info)
    assert matcher.accept_token(0) is True
    assert matcher.is_completed()
    assert matcher.accept_token(vocab_size - 1) is True  # <eos>
    assert matcher.is_terminated()
    for bad_id in [0, vocab_size, -1, 2**31 - 1]:
        assert matcher.accept_token(bad_id) is False


def test_completed_without_stop_token_is_not_terminated(vocab_size: int = 31):
    """IsCompleted and IsTerminated are different states, and both check out-of-range ids."""
    tokenizer_info = make_tokenizer_info(vocab_size)
    matcher = make_matcher('root ::= "t0"', tokenizer_info=tokenizer_info)
    assert matcher.accept_token(0) is True
    assert matcher.is_completed()
    assert not matcher.is_terminated()
    # Only the stop token is left admissible.
    mask = xgr.allocate_token_bitmask(1, vocab_size)
    matcher.fill_next_token_bitmask(mask, 0)
    assert allowed_ids(mask, vocab_size) == [vocab_size - 1]
    assert matcher.accept_token(vocab_size) is False


# --------------------------------------------------------------------------------------
# 2. mask <-> accept consistency, replayed from a fresh matcher for every id
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("vocab_size", [31, 32, 33, 63, 64, 65])
def test_mask_agrees_with_accept_for_every_id(vocab_size: int):
    tokenizer_info = make_tokenizer_info(vocab_size)

    probe = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
    mask = xgr.allocate_token_bitmask(1, vocab_size)
    need_apply = probe.fill_next_token_bitmask(mask, 0)
    mask_allows = set(allowed_ids(mask, vocab_size))

    disagreements = []
    for token_id in range(vocab_size):
        matcher = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
        accepted = bool(matcher.accept_token(token_id))
        in_mask = token_id in mask_allows
        if accepted != in_mask:
            disagreements.append((token_id, accepted, in_mask))

    assert not disagreements, (
        f"mask/accept disagreement (id, accepted, mask_allows): {disagreements}; "
        f"need_apply={need_apply}"
    )


@pytest.mark.parametrize("vocab_size", [33, 65])
def test_mask_agrees_with_accept_for_builtin_json(vocab_size: int):
    """Same invariant for the builtin JSON grammar, which is what JSON mode compiles to."""
    vocab = [f"t{i}" for i in range(vocab_size - 3)] + ['{', '}', '<eos>']
    tokenizer_info = xgr.TokenizerInfo(vocab, stop_token_ids=[vocab_size - 1])
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(
        xgr.Grammar.builtin_json_grammar()
    )

    probe = xgr.GrammarMatcher(compiled)
    mask = xgr.allocate_token_bitmask(1, vocab_size)
    probe.fill_next_token_bitmask(mask, 0)
    mask_allows = set(allowed_ids(mask, vocab_size))

    disagreements = []
    for token_id in range(vocab_size):
        matcher = xgr.GrammarMatcher(compiled)
        accepted = bool(matcher.accept_token(token_id))
        if accepted != (token_id in mask_allows):
            disagreements.append((token_id, accepted, token_id in mask_allows))

    assert not disagreements, f"mask/accept disagreement: {disagreements}"
    # The stop token must not be allowed while the grammar is incomplete.
    assert vocab_size - 1 not in mask_allows


def test_fill_return_value_matches_all_true_mask():
    """``fill_next_token_bitmask`` returns False only when the mask allowed everything."""
    tokenizer_info = make_tokenizer_info(32)
    # A grammar that can start with any ordinary token: use a character class over the
    # single-character tokens of a purpose-built vocabulary.
    vocab = ["a", "b", "c", "<eos>"]
    tokenizer_info = xgr.TokenizerInfo(vocab, stop_token_ids=[3])
    compiled = xgr.GrammarCompiler(tokenizer_info).compile_grammar(
        xgr.Grammar.from_ebnf('root ::= [abc]')
    )
    matcher = xgr.GrammarMatcher(compiled)
    mask = xgr.allocate_token_bitmask(1, 4)
    need_apply = matcher.fill_next_token_bitmask(mask, 0)
    assert need_apply is bool(allowed_ids(mask, 4) != [0, 1, 2, 3])


# --------------------------------------------------------------------------------------
# 3. vocabulary and padding matrix
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("vocab_size", [31, 32, 33, 63, 64, 65])
def test_vocab_size_and_mask_width(vocab_size: int):
    tokenizer_info = make_tokenizer_info(vocab_size)
    assert tokenizer_info.vocab_size == vocab_size
    expected_words = math.ceil(vocab_size / 32)
    assert xgr.get_bitmask_shape(1, vocab_size) == (1, expected_words)

    matcher = make_matcher(tokenizer_info=tokenizer_info)
    mask = xgr.allocate_token_bitmask(1, vocab_size)
    matcher.fill_next_token_bitmask(mask, 0)
    assert mask.shape == (1, expected_words)
    # Ids outside the vocabulary are storage padding, not real tokens.
    assert all(i < vocab_size for i in allowed_ids(mask, vocab_size))


@pytest.mark.parametrize("real_vocab", [17, 33])
@pytest.mark.parametrize("padded_vocab", [48, 64, 65, 96])
def test_padded_vocab_ids_are_masked_or_rejected(real_vocab: int, padded_vocab: int):
    """Padding ids carry no token text and must never be sample-able *and* acceptable."""
    if padded_vocab < real_vocab:
        pytest.skip("padding must not shrink the vocabulary")
    tokenizer_info = make_tokenizer_info(real_vocab, padded_to=padded_vocab)
    assert tokenizer_info.vocab_size == padded_vocab

    matcher = make_matcher(tokenizer_info=tokenizer_info)
    mask = xgr.allocate_token_bitmask(1, padded_vocab)
    matcher.fill_next_token_bitmask(mask, 0)
    allows = set(allowed_ids(mask, padded_vocab))

    for padding_id in range(real_vocab - 1, padded_vocab):
        accepted = bool(fresh_with_prefix(PREFIX_IDS, tokenizer_info).accept_token(padding_id))
        assert not accepted, f"padding id {padding_id} was accepted"
        # A special/padding id may be masked (normal case); it must never be both
        # "allowed by the mask" and "rejected on accept".
        assert padding_id not in allows or padding_id == real_vocab - 1, (
            f"padding id {padding_id} is allowed by the mask but rejected on accept"
        )


def test_unknown_padding_id_is_not_representable_as_a_token():
    """The stop token sits at the top of the real vocab; above it there is no text.

    Accepting an id in the padding region is a no-op rejection, and the decoded token for
    such an id does not exist, so callers must not treat the region as generated text.
    """
    real_vocab, padded_vocab = 20, 40
    tokenizer_info = make_tokenizer_info(real_vocab, padded_to=padded_vocab)
    decoded = tokenizer_info.decoded_vocab
    assert len(decoded) == real_vocab
    assert real_vocab - 1 not in tokenizer_info.special_token_ids  # <eos>

    matcher = make_matcher(tokenizer_info=tokenizer_info)
    for padding_id in range(real_vocab, padded_vocab):
        assert matcher.accept_token(padding_id) is False


# --------------------------------------------------------------------------------------
# 4. synthetic all-masked / malformed distributions (no grammar involved)
# --------------------------------------------------------------------------------------


def _apply(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = logits.clone()
    xgr.apply_token_bitmask_inplace(out, mask, vocab_size=logits.shape[-1])
    return out


def test_all_masked_leaves_no_finite_logit():
    vocab_size = 40
    logits = torch.zeros(1, vocab_size)
    mask = torch.zeros(1, math.ceil(vocab_size / 32), dtype=torch.int32)
    out = _apply(logits, mask)
    assert not torch.isfinite(out).any(), "an all-masked mask must leave no sample-able logit"


def test_masked_tail_is_reported_by_the_debug_helper():
    vocab_size = 40
    mask = torch.zeros(1, math.ceil(vocab_size / 32), dtype=torch.int32)
    mask[0, 0] = 1  # only id 0 allowed
    rejected = _get_masked_tokens_from_bitmask(mask, vocab_size, 0)
    assert rejected == list(range(1, vocab_size))


def test_malformed_logits_are_passed_through_unchanged():
    """NaN / +inf / -inf are caller-side data; the mask only forces masked lanes to -inf."""
    vocab_size = 40
    logits = torch.full((1, vocab_size), float("nan"))
    logits[0, 1] = float("inf")
    logits[0, 2] = float("-inf")
    mask = torch.zeros(1, math.ceil(vocab_size / 32), dtype=torch.int32)
    mask[0, 0] = (1 << 0) | (1 << 1) | (1 << 2)
    out = _apply(logits, mask)
    assert math.isnan(out[0, 0].item())
    assert math.isinf(out[0, 1].item()) and out[0, 1].item() > 0
    assert math.isinf(out[0, 2].item()) and out[0, 2].item() < 0
    assert not torch.isfinite(out[0, 3:]).any()


def test_storage_padding_bits_are_consistently_unmasked():
    """The word-alignment tail of the bitmask carries no information and is left unmasked.

    A bitmask for a 40-token vocabulary has two int32 words and can express 64 bits. Only
    the low 40 bits are real tokens. ``fill_next_token_bitmask`` stores the matcher's result
    into those words with int32 bitwise operations, so the value of bits 40..63 depends on
    how the caller obtained the buffer unless the library normalizes it. xgrammar
    normalizes them to 1 ("not decided here") so that callers applying the bitmask to a
    wider padded logits tensor do not have logits lanes killed by a mask that was never
    about them.

    The important half of this contract is the other direction: normalizing the tail must
    not change any of the 40 real bits.
    """
    vocab_size = 40
    tokenizer_info = make_tokenizer_info(vocab_size)
    matcher = make_matcher(tokenizer_info=tokenizer_info)
    mask = xgr.allocate_token_bitmask(1, vocab_size)
    matcher.fill_next_token_bitmask(mask, 0)

    capacity = mask.shape[-1] * 32
    assert capacity > vocab_size
    real = set(allowed_ids(mask, vocab_size))
    padding = [i for i in range(vocab_size, capacity) if (int(mask[0, i // 32].item()) >> (i % 32)) & 1]
    assert len(padding) == capacity - vocab_size, (
        f"padding ids {sorted(set(range(vocab_size, capacity)) - set(padding))} are masked; "
        "the tail must stay unmasked"
    )

    # Starting from a different buffer pattern gives the same real bits and the same tail.
    mask2 = torch.zeros(1, math.ceil(vocab_size / 32), dtype=torch.int32)
    matcher2 = make_matcher(tokenizer_info=tokenizer_info)
    matcher2.fill_next_token_bitmask(mask2, 0)
    assert allowed_ids(mask2, vocab_size) == sorted(real)
    assert torch.equal(mask, mask2)


def test_storage_padding_normalization_does_not_change_the_real_bits():
    """The real bits must agree with the matcher for every id, before and after the tail."""
    for vocab_size in [31, 33, 40, 63, 65]:
        tokenizer_info = make_tokenizer_info(vocab_size)
        matcher = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
        mask = xgr.allocate_token_bitmask(1, vocab_size)
        matcher.fill_next_token_bitmask(mask, 0)
        allows = set(allowed_ids(mask, vocab_size))
        capacity = mask.shape[-1] * 32
        for padding_id in range(vocab_size, capacity):
            bit = (int(mask[0, padding_id // 32].item()) >> (padding_id % 32)) & 1
            assert bit == 1, f"padding bit {padding_id} is masked"
        # Cross-check the real bits against the matcher itself.
        for token_id in range(vocab_size):
            agrees = bool(fresh_with_prefix(PREFIX_IDS, tokenizer_info).accept_token(token_id))
            assert agrees == (token_id in allows), f"real bit {token_id} disagrees"


def test_bitmask_narrower_than_logits_leaves_tail_unmasked():
    """This is the polarity the Python API documents, pinned as a contract.

    ``apply_token_bitmask_inplace`` only covers ``min(logits.shape[-1],
    bitmask.shape[-1] * 32)`` when ``vocab_size`` is not given, so a logits tail wider than
    the bitmask stays finite and therefore sample-able. Callers whose logits are wider than
    the matcher's vocabulary must handle that tail themselves.
    """
    vocab_size, mask_vocab = 100, 40
    capacity = math.ceil(mask_vocab / 32) * 32
    logits = torch.zeros(1, vocab_size)
    mask = torch.zeros(1, math.ceil(mask_vocab / 32), dtype=torch.int32)
    mask[0, 0] = 1
    with pytest.warns(UserWarning, match="left unmasked"):
        out = logits.clone()
        xgr.apply_token_bitmask_inplace(out, mask)
    assert out[0, 0].item() == 0.0
    assert not torch.isfinite(out[0, 1:capacity]).any()
    # Everything the bitmask's words do not cover stays sample-able.
    assert torch.isfinite(out[0, capacity:]).all()


def test_explicit_vocab_size_wider_than_the_bitmask_is_rejected():
    """An explicit ``vocab_size`` cannot exceed what the bitmask's words can express.

    Otherwise the apply would read past the end of the bitmask buffer.
    """
    logits = torch.zeros(1, 100)
    mask = torch.zeros(1, 2, dtype=torch.int32)  # capacity 64
    xgr.apply_token_bitmask_inplace(logits, mask, vocab_size=64)  # boundary is fine
    with pytest.raises(RuntimeError):
        xgr.apply_token_bitmask_inplace(logits.clone(), mask, vocab_size=65)


def test_bitmask_width_must_match_the_matcher_vocabulary():
    """A matcher fills only a bitmask whose word count matches its own vocabulary.

    The buffer width is ``ceil(vocab_size / 32)`` int32 words, so any vocabulary in
    ``(32, 64]`` uses the same buffer shape. A wider buffer is rejected rather than filled;
    the caller has to size the bitmask from ``TokenizerInfo.vocab_size`` and mask off any
    extra logits lanes itself.
    """
    tokenizer_info = make_tokenizer_info(32)
    assert tokenizer_info.vocab_size == 32
    matcher = make_matcher(tokenizer_info=tokenizer_info)

    exact = xgr.allocate_token_bitmask(1, 32)
    assert exact.shape == (1, 1)
    matcher.fill_next_token_bitmask(exact, 0)  # must not raise

    # Crossing the 32-bit word boundary changes the required buffer shape.
    assert xgr.allocate_token_bitmask(1, 33).shape == (1, 2)
    assert xgr.allocate_token_bitmask(1, 31).shape == (1, 1)

    too_wide = xgr.allocate_token_bitmask(1, 64)
    assert too_wide.shape == (1, 2)
    with pytest.raises(RuntimeError, match="not valid"):
        matcher.fill_next_token_bitmask(too_wide, 0)

    # A vocabulary that needs three words is also invalid for this one-word matcher.
    three_words = xgr.allocate_token_bitmask(1, 96)
    assert three_words.shape == (1, 3)
    with pytest.raises(RuntimeError, match="not valid"):
        matcher.fill_next_token_bitmask(three_words, 0)

    exact = xgr.allocate_token_bitmask(1, 32)
    matcher.fill_next_token_bitmask(exact, 0)  # must not raise


# --------------------------------------------------------------------------------------
# 5. ids that do not fit in int32 must never be silently reinterpreted
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "oversized_id",
    [
        2**31,
        2**32,
        2**32 + 1,
        2**32 + 2,
        2**33 + 1,
        2**40,
        2**40 + 1,
        -(2**31) - 1,
        -(2**32),
    ],
)
def test_oversized_id_is_never_reinterpreted_as_an_in_range_id(oversized_id: int):
    """An id outside the int32 domain must be rejected, not wrapped into a valid one.

    ``t1`` after the prefix is accepted, and it is exactly the id that ``2**32 + 1``
    truncates to. Before the boundary fix this test observed ``accept_token(2**32 + 1) is
    True``, i.e. a corrupted 64-bit id was silently reinterpreted as a valid token. The
    matcher's vocabulary is 32 ids, so no in-range value should ever be produced from an
    id that large.
    """
    tokenizer_info = make_tokenizer_info(32)
    control = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
    assert control.accept_token(1) is True  # the id that truncation would produce

    matcher = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
    with pytest.raises(RuntimeError, match="int32 range"):
        matcher.accept_token(oversized_id)

    # Rejection must be total: the matcher is still usable and unchanged.
    assert matcher.accept_token(1) is True


def test_oversized_id_in_batch_accept_token_is_never_reinterpreted():
    """``BatchGrammarMatcher.batch_accept_token`` goes through the same boundary."""
    tokenizer_info = make_tokenizer_info(32)
    matchers = [fresh_with_prefix(PREFIX_IDS, tokenizer_info) for _ in range(2)]
    with pytest.raises(RuntimeError, match="int32 range"):
        xgr.BatchGrammarMatcher.batch_accept_token(matchers, [2**32 + 1, 2**40])


def test_in_range_ids_are_unaffected_by_the_boundary_fix():
    """The two int32 endpoints are ordinary out-of-vocabulary ids, not cast errors."""
    tokenizer_info = make_tokenizer_info(32)
    for token_id in [2**31 - 1, -(2**31)]:
        matcher = fresh_with_prefix(PREFIX_IDS, tokenizer_info)
        assert matcher.accept_token(token_id) is False
