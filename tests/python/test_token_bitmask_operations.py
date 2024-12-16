"""This test uses the optimized JSON grammar provided by the grammar library."""

import sys
import time

import pytest
import torch
from triton.testing import do_bench

import xgrammar as xgr
from xgrammar.testing import _get_masked_tokens_from_bitmask


def _bool_mask_to_bitmask(bool_mask: torch.Tensor) -> torch.Tensor:
    bool_mask_int32 = bool_mask.to(torch.int32)
    # Pad to multiple of 32
    pad_size = (32 - bool_mask.shape[1] % 32) % 32
    if pad_size > 0:
        bool_mask_int32 = torch.nn.functional.pad(bool_mask_int32, (0, pad_size))
    bool_mask_view = bool_mask_int32.view(bool_mask.shape[0], -1, 32)
    # To avoid error for overflow, we construct int64 weights and convert to int32
    weights = torch.tensor([1 << i for i in range(32)], dtype=torch.int64).to(torch.int32)
    bitmask = (bool_mask_view * weights).sum(dim=2)
    return bitmask.to(torch.int32)


def test_allocate_reset_token_bitmask():
    batch_size = 10
    vocab_size = 128005
    bitmask = xgr.allocate_token_bitmask(batch_size, vocab_size)
    assert bitmask.shape == (batch_size, (vocab_size + 31) // 32)
    assert bitmask.device.type == "cpu"
    assert (bitmask == 0xFFFFFFFF).all()
    bitmask.fill_(0)
    xgr.reset_token_bitmask(bitmask)
    assert (bitmask == 0xFFFFFFFF).all()


token_mask_sizes = (1024, 32000, 32001, 32011)


@pytest.mark.parametrize("token_mask_size", token_mask_sizes)
@pytest.mark.parametrize("index", (0, 1))
def test_get_masked_tokens_from_bitmask(token_mask_size: int, index: int):
    bool_mask = torch.randint(0, 2, (2, token_mask_size), dtype=torch.bool)
    bitmask = _bool_mask_to_bitmask(bool_mask)
    expected = torch.where(~bool_mask[index])[0].tolist()
    assert _get_masked_tokens_from_bitmask(bitmask, token_mask_size, index) == expected


@pytest.mark.parametrize("is_cuda", (True, False))
def test_apply_token_bitmask_inplace(is_cuda: bool):
    neginf = float("-inf")
    bool_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32)
    expected = torch.where(bool_mask, logits, neginf)

    if is_cuda:
        logits_gpu = logits.to("cuda")
        bitmask = torch.tensor([0b1010101010], dtype=torch.int32).to("cuda")
        xgr.apply_token_bitmask_inplace(logits_gpu, bitmask)
        torch.cuda.synchronize()
        torch.testing.assert_close(logits_gpu, expected.to("cuda"))
    else:
        bitmask = torch.tensor([0b1010101010], dtype=torch.int32)
        xgr.apply_token_bitmask_inplace(logits, bitmask)
        torch.testing.assert_close(logits, expected)


batch_size_vocab_size_masked_cnt_stride = [
    (1, 128000, 1024, 1),
    (1, 128000, 120000, 1),
    (1, 128001, 120000, 1),
    (1, 128010, 120000, 1),
    (64, 128000, 1024, 1),
    (64, 128000, 120000, 1),
    (64, 128000, 1024, 4),
    (64, 128000, 120000, 4),
]


@pytest.mark.parametrize(
    "batch_size, vocab_size, masked_cnt, stride",
    batch_size_vocab_size_masked_cnt_stride,
)
@pytest.mark.parametrize("is_cuda", (False, True))
def test_apply_token_bitmask_inplace_large(
    batch_size: int, vocab_size: int, masked_cnt: int, stride: int, is_cuda: bool
):

    masked_batch_ids = list(range(0, batch_size, stride))
    masked_positions = torch.randint(0, vocab_size, (batch_size, masked_cnt))
    bool_mask = torch.ones((batch_size, vocab_size), dtype=torch.bool)
    bool_mask.scatter_(1, masked_positions, False)

    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    logits_expected = logits.clone()
    logits_expected[masked_batch_ids] = torch.masked_fill(
        logits_expected[masked_batch_ids], ~bool_mask[masked_batch_ids], float("-inf")
    )

    bitmask = _bool_mask_to_bitmask(bool_mask)
    if is_cuda:
        logits_gpu = logits.to("cuda")
        bitmask_gpu = bitmask.to("cuda")
        torch.cuda.synchronize()
        if stride == 1:
            # Test logic without indices
            f = lambda: xgr.apply_token_bitmask_inplace(logits_gpu, bitmask_gpu)
        else:
            f = lambda: xgr.apply_token_bitmask_inplace(
                logits_gpu, bitmask_gpu, indices=masked_batch_ids
            )
        f()
        torch.testing.assert_close(logits_gpu, logits_expected.to("cuda"))

        dur = do_bench(f, warmup=100, rep=1000)
        print(f"apply_token_bitmask_inplace_cuda time: {(dur) * 1e3} us")
    else:
        time_start = time.monotonic_ns()
        if stride == 1:
            # Test logic without indices
            xgr.apply_token_bitmask_inplace(logits, bitmask)

        else:
            xgr.apply_token_bitmask_inplace(logits, bitmask, indices=masked_batch_ids)
        time_end = time.monotonic_ns()
        print(f"apply_token_bitmask_inplace_cpu time: {(time_end - time_start) / 1e3} us")
        torch.testing.assert_close(logits, logits_expected)


if __name__ == "__main__":
    pytest.main(sys.argv)
