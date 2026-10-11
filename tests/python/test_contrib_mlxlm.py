"""Test the mlx-lm integration in xgrammar.contrib.mlxlm."""

import numpy as np
import pytest
import torch

import xgrammar as xgr

# mlx-lm is optional; CI installs it only on macOS arm64.
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402


@pytest.mark.thread_unsafe
def test_logits_processor_matches_cpu_kernel():
    from xgrammar.contrib.mlxlm import XGrammarLogitsProcessor

    # 257 tokens so that the vocab size is not a multiple of 32.
    vocab = [bytes([b]) for b in range(256)] + [b"</s>"]
    tokenizer_info = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, stop_token_ids=[256])
    grammar = xgr.GrammarCompiler(tokenizer_info).compile_builtin_json_grammar()
    processor = XGrammarLogitsProcessor(grammar)

    # The first call receives the prompt; its last token is not part of the JSON output.
    logits = np.random.default_rng(0).standard_normal((1, len(vocab)), dtype=np.float32)
    masked = processor(mx.array([ord("x")]), mx.array(logits))

    expected = torch.from_numpy(logits.copy())
    xgr.apply_token_bitmask_inplace(expected, processor.bitmask)
    expected = expected.numpy()
    assert np.array_equal(np.array(masked), expected)
    assert np.isfinite(expected[0, ord("{")]) and not np.isfinite(expected[0, ord("}")])
