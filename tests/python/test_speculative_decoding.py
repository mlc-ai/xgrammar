"""Test the speculative decoding utilities."""

import sys

import pytest
import torch

import xgrammar as xgr
from xgrammar.matcher import allocate_token_bitmask
from xgrammar.testing import _traverse_draft_tree


def test_traverse_draft_tree_linear():
    """Test _traverse_draft_tree with a simple linear tree structure."""
    # Create a simple grammar for JSON
    grammar = xgr.Grammar.builtin_json_grammar()

    # Create a simple tokenizer info with some tokens
    vocab = ["a", "b", "c", "{", "}", '"', ":", ",", " ", "true", "false", "null"]
    tokenizer_info = xgr.TokenizerInfo(vocab, vocab_size=len(vocab), stop_token_ids=[])

    # Compile the grammar
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_grammar(grammar)

    # Create a matcher
    matcher = xgr.GrammarMatcher(compiled_grammar)

    # Create test tree structure (simple linear tree: 0 -> 1 -> 2)
    num_nodes = 3
    retrieve_next_token = torch.tensor([1, 2, -1], dtype=torch.int64)
    retrieve_next_sibling = torch.tensor([-1, -1, -1], dtype=torch.int64)
    draft_tokens = torch.tensor([3, 6, 4], dtype=torch.int64)  # {, :, }

    # Allocate bitmask
    bitmask = allocate_token_bitmask(num_nodes, len(vocab))

    # Call the function
    _traverse_draft_tree(retrieve_next_token, retrieve_next_sibling, draft_tokens, matcher, bitmask)

    # Verify that the bitmask was filled (at least the first position should be non-zero)
    assert bitmask[0].any(), "First position bitmask should be non-zero"


def test_traverse_draft_tree_with_siblings():
    """Test _traverse_draft_tree with a tree that has sibling nodes."""
    grammar = xgr.Grammar.builtin_json_grammar()

    vocab = ["a", "b", "c", "{", "}", '"', ":", ",", " ", "true", "false", "null"]
    tokenizer_info = xgr.TokenizerInfo(vocab, vocab_size=len(vocab), stop_token_ids=[])

    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_grammar(grammar)

    matcher = xgr.GrammarMatcher(compiled_grammar)

    # Tree structure:
    #       0
    #      / \
    #     1   2
    num_nodes = 3
    retrieve_next_token = torch.tensor([1, -1, -1], dtype=torch.int64)
    retrieve_next_sibling = torch.tensor([-1, 2, -1], dtype=torch.int64)
    draft_tokens = torch.tensor([3, 5, 4], dtype=torch.int64)  # {, ", }

    bitmask = allocate_token_bitmask(num_nodes, len(vocab))

    _traverse_draft_tree(retrieve_next_token, retrieve_next_sibling, draft_tokens, matcher, bitmask)

    # Verify that the bitmask was filled
    assert bitmask[0].any(), "Root position bitmask should be non-zero"


def test_traverse_draft_tree_shape_assertion():
    """Test that _traverse_draft_tree raises assertion error for mismatched shapes."""
    grammar = xgr.Grammar.builtin_json_grammar()

    vocab = ["a", "b", "c", "{", "}", '"', ":", ",", " ", "true", "false", "null"]
    tokenizer_info = xgr.TokenizerInfo(vocab, vocab_size=len(vocab), stop_token_ids=[])

    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_grammar(grammar)

    matcher = xgr.GrammarMatcher(compiled_grammar)

    # Mismatched shapes
    retrieve_next_token = torch.tensor([1, 2, -1], dtype=torch.int64)
    retrieve_next_sibling_wrong_shape = torch.tensor([-1, -1], dtype=torch.int64)  # Wrong shape
    retrieve_next_sibling_wrong_dtype = torch.tensor([-1, -1, -1], dtype=torch.int32)  # Wrong dtype
    draft_tokens = torch.tensor([3, 6, 4], dtype=torch.int32)

    bitmask = allocate_token_bitmask(3, len(vocab))

    with pytest.raises(RuntimeError):
        _traverse_draft_tree(
            retrieve_next_token, retrieve_next_sibling_wrong_shape, draft_tokens, matcher, bitmask
        )

    with pytest.raises(RuntimeError):
        _traverse_draft_tree(
            retrieve_next_token, retrieve_next_sibling_wrong_dtype, draft_tokens, matcher, bitmask
        )


if __name__ == "__main__":
    pytest.main(sys.argv)
