"""Test the speculative decoding utilities."""

import sys

import pytest
import torch

import xgrammar as xgr
from xgrammar.matcher import allocate_token_bitmask
from xgrammar.testing import _traverse_draft_tree

VOCAB = ["a", "b", "c", "{", "}", '"', ":", ",", " ", "true", "false", "null"]
VOCAB_SIZE = len(VOCAB)

# ── Tree definitions ──────────────────────────────────────────────────────────

# Linear tree: 0 -> 1 -> 2
LINEAR_TREE = (
    torch.tensor([1, 2, -1], dtype=torch.int64),  # next_token
    torch.tensor([-1, -1, -1], dtype=torch.int64),  # next_sibling
    torch.tensor([3, 6, 4], dtype=torch.int64),  # draft tokens: {, :, }
)

# Tree with siblings:
#       0
#      / \
#     1   2
SIBLING_TREE = (
    torch.tensor([1, -1, -1], dtype=torch.int64),  # next_token
    torch.tensor([-1, 2, -1], dtype=torch.int64),  # next_sibling
    torch.tensor([3, 5, 4], dtype=torch.int64),  # draft tokens: {, ", }
)


@pytest.fixture(scope="module")
def compiled_grammar():
    """Compile the built-in JSON grammar with our small test vocab."""
    grammar = xgr.Grammar.builtin_json_grammar()
    tokenizer_info = xgr.TokenizerInfo(VOCAB, vocab_size=VOCAB_SIZE, stop_token_ids=[])
    compiler = xgr.GrammarCompiler(tokenizer_info)
    return compiler.compile_grammar(grammar)


@pytest.fixture(scope="module")
def literal_compiled_grammar():
    """Compile a small deterministic grammar for batch traversal parity tests."""
    tokenizer_info = xgr.TokenizerInfo(["a", "b", "c"], vocab_size=3, stop_token_ids=[])
    return xgr.GrammarCompiler(tokenizer_info).compile_grammar('root ::= "ab"')


def _run_traverse(compiled_grammar, tree, **traverse_kwargs):
    """Run traverse_draft_tree on a tree and return (result, bitmask)."""
    retrieve_next_token, retrieve_next_sibling, draft_tokens = tree
    num_nodes = retrieve_next_token.shape[0]
    matcher = xgr.GrammarMatcher(compiled_grammar)
    bitmask = allocate_token_bitmask(num_nodes, VOCAB_SIZE)
    result = matcher.traverse_draft_tree(
        retrieve_next_token, retrieve_next_sibling, draft_tokens, bitmask, **traverse_kwargs
    )
    return result, bitmask


# ── Basic traversal tests ────────────────────────────────────────────────────


def test_traverse_draft_tree_linear(compiled_grammar):
    """Test traverse_draft_tree with a simple linear tree structure."""
    result, bitmask = _run_traverse(compiled_grammar, LINEAR_TREE)
    assert result is True
    assert bitmask[0].any(), "First position bitmask should be non-zero"


def test_traverse_draft_tree_with_siblings(compiled_grammar):
    """Test traverse_draft_tree with a tree that has sibling nodes."""
    result, bitmask = _run_traverse(compiled_grammar, SIBLING_TREE)
    assert result is True
    assert bitmask[0].any(), "Root position bitmask should be non-zero"


def test_traverse_draft_tree_rejected_node(compiled_grammar):
    """Rejected nodes should have zero bitmasks."""
    rejected_tree = (
        torch.tensor([1, 2, -1], dtype=torch.int64),
        torch.tensor([-1, -1, -1], dtype=torch.int64),
        torch.tensor([3, 0, 3], dtype=torch.int64),
    )

    result, bitmask = _run_traverse(compiled_grammar, rejected_tree)

    assert result is True
    assert bitmask[0].any(), "Root position bitmask should be non-zero"
    assert not bitmask[1].any(), "Rejected node bitmask should be zero"


def test_traverse_draft_tree_invalid_token_with_sibling(compiled_grammar):
    """Invalid draft tokens should not break traversal of valid sibling branches."""
    invalid_token_tree = (
        torch.tensor([1, 2, -1, -1], dtype=torch.int64),
        torch.tensor([-1, 3, -1, -1], dtype=torch.int64),
        torch.tensor([3, 100, 3, 3], dtype=torch.int64),
    )

    result, bitmask = _run_traverse(compiled_grammar, invalid_token_tree)

    assert result is True
    assert bitmask[0].any(), "Root position bitmask should be non-zero"
    assert not bitmask[1].any(), "Invalid token node bitmask should be zero"
    assert bitmask[3].any(), "Valid sibling bitmask should be computed"


def test_traverse_draft_tree_terminated_node():
    """Terminated nodes should have zero bitmasks."""
    grammar = xgr.Grammar.from_ebnf('root ::= "a"')
    tokenizer_info = xgr.TokenizerInfo(["a", "b"], vocab_size=2, stop_token_ids=[])
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_grammar(grammar)
    matcher = xgr.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
    bitmask = allocate_token_bitmask(3, 2)
    terminated_tree = (
        torch.tensor([1, 2, -1], dtype=torch.int64),
        torch.tensor([-1, -1, -1], dtype=torch.int64),
        torch.tensor([0, 0, 1], dtype=torch.int64),
    )

    result = matcher.traverse_draft_tree(*terminated_tree, bitmask)

    assert result is True
    assert bitmask[0].any(), "Root position bitmask should be non-zero"
    assert not bitmask[1].any(), "Terminated node bitmask should be zero"


def test_old_traverse_draft_tree(compiled_grammar):
    """Test the backward-compatible testing wrapper."""
    retrieve_next_token, retrieve_next_sibling, draft_tokens = LINEAR_TREE
    matcher = xgr.GrammarMatcher(compiled_grammar)
    bitmask = allocate_token_bitmask(retrieve_next_token.shape[0], VOCAB_SIZE)

    result = _traverse_draft_tree(
        retrieve_next_token, retrieve_next_sibling, draft_tokens, matcher, bitmask
    )

    assert result is True
    assert bitmask[0].any(), "First position bitmask should be non-zero"


# ── Shape / dtype validation ─────────────────────────────────────────────────


def test_traverse_draft_tree_shape_assertion(compiled_grammar):
    """Test that traverse_draft_tree raises RuntimeError for mismatched shapes/dtypes."""
    matcher = xgr.GrammarMatcher(compiled_grammar)
    retrieve_next_token = torch.tensor([1, 2, -1], dtype=torch.int64)
    draft_tokens = torch.tensor([3, 6, 4], dtype=torch.int64)
    bitmask = allocate_token_bitmask(3, VOCAB_SIZE)

    # Wrong shape for retrieve_next_sibling
    with pytest.raises(RuntimeError):
        matcher.traverse_draft_tree(
            retrieve_next_token, torch.tensor([-1, -1], dtype=torch.int64), draft_tokens, bitmask
        )

    # Wrong dtype for retrieve_next_sibling
    with pytest.raises(RuntimeError):
        matcher.traverse_draft_tree(
            retrieve_next_token,
            torch.tensor([-1, -1, -1], dtype=torch.int32),
            draft_tokens,
            bitmask,
        )

    # Wrong rank for token_bitmask
    with pytest.raises(RuntimeError):
        matcher.traverse_draft_tree(
            retrieve_next_token,
            torch.tensor([-1, -1, -1], dtype=torch.int64),
            draft_tokens,
            torch.full((bitmask.shape[1],), -1, dtype=torch.int32),
        )

    # Wrong batch size for token_bitmask
    with pytest.raises(RuntimeError):
        matcher.traverse_draft_tree(
            retrieve_next_token,
            torch.tensor([-1, -1, -1], dtype=torch.int64),
            draft_tokens,
            allocate_token_bitmask(2, VOCAB_SIZE),
        )

    # Root should not have siblings
    with pytest.raises(RuntimeError):
        matcher.traverse_draft_tree(
            retrieve_next_token, torch.tensor([1, -1, -1], dtype=torch.int64), draft_tokens, bitmask
        )


# ── Batched traversal tests ──────────────────────────────────────────────────


def test_batch_traverse_draft_tree_matches_scalar_and_reuses_pool(literal_compiled_grammar):
    batch_size = 3
    num_nodes = 3
    retrieve_next_token = torch.tensor([1, 2, -1], dtype=torch.int64)
    retrieve_next_sibling = torch.full((num_nodes,), -1, dtype=torch.int64)
    draft_tokens = torch.tensor([[2, 0, 1], [2, 0, 2], [2, 0, 1]], dtype=torch.int64)
    batch_mask = torch.zeros((batch_size * num_nodes, 1), dtype=torch.int32)
    scalar_mask = torch.full_like(batch_mask, -1)
    indices = [0, 2]
    batch_matchers = [xgr.GrammarMatcher(literal_compiled_grammar) for _ in indices]
    scalar_matchers = [xgr.GrammarMatcher(literal_compiled_grammar) for _ in indices]
    batch = xgr.BatchGrammarMatcher(max_threads=2)

    completed = batch.batch_traverse_draft_tree(
        batch_matchers,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
        batch_mask,
        indices,
    )
    assert completed == [True, True]

    for matcher, index in zip(scalar_matchers, indices):
        start = index * num_nodes
        assert matcher.traverse_draft_tree(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens[index],
            scalar_mask[start : start + num_nodes],
        )

    torch.testing.assert_close(batch_mask, scalar_mask)
    assert torch.equal(
        batch_mask[num_nodes : 2 * num_nodes], torch.full((num_nodes, 1), -1, dtype=torch.int32)
    )

    # A BatchGrammarMatcher owns one persistent native pool. Reusing it must
    # produce the same result without reconstructing joined worker threads.
    batch_mask.zero_()
    assert batch.batch_traverse_draft_tree(
        batch_matchers,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
        batch_mask,
        indices,
    ) == [True, True]
    torch.testing.assert_close(batch_mask, scalar_mask)


def test_batch_traverse_draft_tree_batched_trees_and_roots(literal_compiled_grammar):
    retrieve_next_token = torch.tensor([[1, 2, 3, -1], [-1, -1, 3, -1]], dtype=torch.int64)
    retrieve_next_sibling = torch.full((2, 4), -1, dtype=torch.int64)
    draft_tokens = torch.tensor([[2, 0, 1, 2], [2, 2, 2, 0]], dtype=torch.int64)
    root_positions = [0, 2]
    batch_mask = torch.empty((8, 1), dtype=torch.int32)
    scalar_mask = torch.full_like(batch_mask, -1)
    batch_matchers = [xgr.GrammarMatcher(literal_compiled_grammar) for _ in range(2)]
    scalar_matchers = [xgr.GrammarMatcher(literal_compiled_grammar) for _ in range(2)]

    completed = xgr.BatchGrammarMatcher(max_threads=2).batch_traverse_draft_tree(
        batch_matchers,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
        batch_mask,
        root_positions=root_positions,
    )
    assert completed == [True, True]

    for index, matcher in enumerate(scalar_matchers):
        start = index * 4
        assert matcher.traverse_draft_tree(
            retrieve_next_token[index],
            retrieve_next_sibling[index],
            draft_tokens[index],
            scalar_mask[start : start + 4],
            root_position=root_positions[index],
        )
    torch.testing.assert_close(batch_mask, scalar_mask)


def test_batch_traverse_draft_tree_honors_tensor_storage_offsets(literal_compiled_grammar):
    retrieve_next_token = torch.tensor([99, 1, 2, -1], dtype=torch.int64)[1:]
    retrieve_next_sibling = torch.tensor([99, -1, -1, -1], dtype=torch.int64)[1:]
    draft_tokens = torch.tensor([99, 2, 0, 1, 2, 0, 1], dtype=torch.int64)[1:].view(2, 3)
    offset_mask = torch.empty((7, 1), dtype=torch.int32)[1:]
    expected_mask = torch.empty_like(offset_mask)
    assert retrieve_next_token.is_contiguous() and retrieve_next_token.storage_offset() > 0
    assert retrieve_next_sibling.is_contiguous() and retrieve_next_sibling.storage_offset() > 0
    assert draft_tokens.is_contiguous() and draft_tokens.storage_offset() > 0
    assert offset_mask.is_contiguous() and offset_mask.storage_offset() > 0

    batch = xgr.BatchGrammarMatcher(max_threads=2)
    completed = batch.batch_traverse_draft_tree(
        [xgr.GrammarMatcher(literal_compiled_grammar) for _ in range(2)],
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
        offset_mask,
    )
    assert completed == [True, True]

    completed = batch.batch_traverse_draft_tree(
        [xgr.GrammarMatcher(literal_compiled_grammar) for _ in range(2)],
        retrieve_next_token.clone(),
        retrieve_next_sibling.clone(),
        draft_tokens.clone(),
        expected_mask,
    )
    assert completed == [True, True]
    torch.testing.assert_close(offset_mask, expected_mask, rtol=0, atol=0)


def test_batch_traverse_draft_tree_rejects_racy_or_noncontiguous_input(literal_compiled_grammar):
    batch = xgr.BatchGrammarMatcher(max_threads=2)
    matchers = [xgr.GrammarMatcher(literal_compiled_grammar) for _ in range(2)]
    retrieve_next_token = torch.tensor([1, 2, -1], dtype=torch.int64)
    retrieve_next_sibling = torch.full((3,), -1, dtype=torch.int64)
    draft_tokens = torch.tensor([[2, 0, 1], [2, 0, 1]], dtype=torch.int64)
    bitmask = torch.empty((6, 1), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="assigned to more than one matcher"):
        batch.batch_traverse_draft_tree(
            matchers,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            bitmask,
            indices=[0, 0],
        )

    noncontiguous_tokens = torch.zeros((2, 6), dtype=torch.int64)[:, ::2]
    assert not noncontiguous_tokens.is_contiguous()
    with pytest.raises(RuntimeError, match="requires contiguous tensors"):
        batch.batch_traverse_draft_tree(
            matchers, retrieve_next_token, retrieve_next_sibling, noncontiguous_tokens, bitmask
        )

    with pytest.raises(RuntimeError, match="root position"):
        batch.batch_traverse_draft_tree(
            matchers,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            bitmask,
            root_positions=[0, 3],
        )


# ── Timeout tests ────────────────────────────────────────────────────────────


def test_traverse_draft_tree_timeout_no_change(compiled_grammar):
    """Results should be identical whether time_threshold is omitted or set generously."""
    for tree in [LINEAR_TREE, SIBLING_TREE]:
        # Baseline with timeout explicitly disabled
        _, bitmask_baseline = _run_traverse(compiled_grammar, tree, time_threshold=-1.0)

        # Omitting time_threshold entirely (exercises the Python default)
        result_default, bitmask_default = _run_traverse(compiled_grammar, tree)
        assert result_default is True
        assert torch.equal(bitmask_baseline, bitmask_default)

        # Large timeout should also produce identical results
        result_large, bitmask_large = _run_traverse(compiled_grammar, tree, time_threshold=100.0)
        assert result_large is True
        assert torch.equal(bitmask_baseline, bitmask_large)


def test_traverse_draft_tree_timeout_triggers():
    """A near-zero time_threshold should cause the traversal to time out.

    Timeout is only checked for non-root nodes, so the root bitmask is always
    computed.  We build a deep linear chain so the timeout check is exercised.
    """
    grammar = xgr.Grammar.from_ebnf('root ::= "a" root | "a"')
    tokenizer_info = xgr.TokenizerInfo(["a"], vocab_size=1, stop_token_ids=[])
    compiler = xgr.GrammarCompiler(tokenizer_info)
    chain_grammar = compiler.compile_grammar(grammar)
    num_nodes = 10000
    deep_chain = (
        torch.tensor([i + 1 for i in range(num_nodes - 1)] + [-1], dtype=torch.int64),
        torch.full((num_nodes,), -1, dtype=torch.int64),
        torch.zeros(num_nodes, dtype=torch.int64),
    )

    matcher = xgr.GrammarMatcher(chain_grammar)
    bitmask = allocate_token_bitmask(num_nodes, 1)
    result = matcher.traverse_draft_tree(*deep_chain, bitmask, time_threshold=1e-7)

    assert result is False, "Traversal should time out with near-zero threshold"
    # Root bitmask should still be filled since timeout is only checked for non-root nodes
    assert bitmask[0].any(), "Root bitmask should still be computed even when timed out"


if __name__ == "__main__":
    pytest.main(sys.argv)
