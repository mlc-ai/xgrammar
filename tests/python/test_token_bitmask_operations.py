"""This test uses the optimized JSON grammar provided by the grammar library."""

import sys
import time

import pytest
import torch

import xgrammar as xgr
from xgrammar.testing import _bool_mask_to_bitmask, _get_masked_tokens_from_bitmask


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


@pytest.mark.parametrize("impl", ("cpu", "cuda", "triton"))
def test_apply_token_bitmask_inplace(impl: str):
    if impl == "cuda" and "cuda" not in xgr.kernels.apply_token_bitmask_inplace:
        pytest.skip(reason="CUDA is not installed")
    if impl == "triton" and "triton" not in xgr.kernels.apply_token_bitmask_inplace:
        pytest.skip(reason="Triton is not installed")

    neginf = float("-inf")
    bool_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
    logits = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32
    )
    expected = torch.where(bool_mask, logits, neginf)

    if impl in ["cuda", "triton"]:
        if not torch.cuda.is_available():
            pytest.skip(reason="CUDA is not installed")

        logits_gpu = logits.to("cuda")
        bitmask = torch.tensor([0b1010101010], dtype=torch.int32).to("cuda")
        if impl == "cuda":
            xgr.kernels.apply_token_bitmask_inplace["cuda"](logits_gpu, bitmask)
        else:
            xgr.kernels.apply_token_bitmask_inplace["triton"](logits_gpu, bitmask)
        torch.cuda.synchronize()
        torch.testing.assert_close(logits_gpu, expected.to("cuda"))
    else:
        bitmask = torch.tensor([0b1010101010], dtype=torch.int32)
        xgr.apply_token_bitmask_inplace(logits, bitmask)
        torch.testing.assert_close(logits, expected)


batch_size_vocab_size_masked_cnt_stride_logits_dtype = [
    (1, 128000, 1024, 1, "float32"),
    (1, 128000, 120000, 1, "float32"),
    (1, 128001, 120000, 1, "float32"),
    (1, 128010, 120000, 1, "float32"),
    (64, 128000, 1024, 1, "float32"),
    (64, 128000, 120000, 1, "float32"),
    (64, 128000, 1024, 4, "float32"),
    (64, 128000, 120000, 4, "float32"),
    (64, 128001, 120000, 1, "float32"),
    (64, 128010, 120000, 1, "float32"),
    (64, 128000, 1024, 1, "float16"),
    (64, 128000, 1024, 1, "bfloat16"),
]


@pytest.mark.parametrize(
    "batch_size, vocab_size, masked_cnt, stride, logits_dtype",
    batch_size_vocab_size_masked_cnt_stride_logits_dtype,
)
@pytest.mark.parametrize("impl", ("cpu", "cuda", "triton"))
def test_apply_token_bitmask_inplace_large(
    batch_size: int,
    vocab_size: int,
    masked_cnt: int,
    stride: int,
    logits_dtype: str,
    impl: str,
):
    if impl == "cpu" and logits_dtype != "float32":
        pytest.skip(reason="cpu implementation supports float32 only")

    if impl == "cuda" and "cuda" not in xgr.kernels.apply_token_bitmask_inplace:
        pytest.skip(reason="CUDA is not installed")
    if impl == "triton" and "triton" not in xgr.kernels.apply_token_bitmask_inplace:
        pytest.skip(reason="Triton is not installed")

    logits_dtype = getattr(torch, logits_dtype)
    logits = torch.randn(batch_size, vocab_size, dtype=logits_dtype)

    if masked_cnt >= vocab_size:
        bool_mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool)
    else:
        bool_mask = torch.ones(batch_size, vocab_size, dtype=torch.bool)
        if masked_cnt > 0:
            masked_positions = torch.stack(
                [torch.randperm(vocab_size)[:masked_cnt] for _ in range(batch_size)]
            )
            bool_mask.scatter_(1, masked_positions, False)
            assert (bool_mask.sum(dim=-1) + masked_cnt == vocab_size).all().item()

    bitmask = _bool_mask_to_bitmask(bool_mask)

    masked_batch_ids = torch.arange(0, batch_size, stride, dtype=torch.int32)

    logits_expected = logits.clone()
    logits_expected[masked_batch_ids] = torch.masked_fill(
        logits_expected[masked_batch_ids], ~bool_mask[masked_batch_ids], float("-inf")
    )

    bitmask = _bool_mask_to_bitmask(bool_mask)
    if impl in ["cuda", "triton"]:
        if not torch.cuda.is_available():
            pytest.skip(reason="CUDA is not installed")

        logits_gpu = logits.to("cuda")
        bitmask_gpu = bitmask.to("cuda")
        masked_batch_ids_gpu = masked_batch_ids.to("cuda")
        torch.cuda.synchronize()
        kwargs = {} if stride == 1 else {"indices": masked_batch_ids_gpu}
        if impl == "cuda":
            f = lambda: xgr.kernels.apply_token_bitmask_inplace["cuda"](
                logits_gpu, bitmask_gpu, **kwargs
            )
        else:
            f = lambda: xgr.kernels.apply_token_bitmask_inplace["triton"](
                logits_gpu, bitmask_gpu, **kwargs
            )
        f()
        torch.testing.assert_close(logits_gpu, logits_expected.to("cuda"))

        try:
            from triton.testing import do_bench

            exec_time = do_bench(f, warmup=100, rep=1000)
            exec_time *= 1e3
        except ImportError:
            pytest.skip(reason="Triton is not installed")
    else:
        kwargs = {} if stride == 1 else {"indices": masked_batch_ids.tolist()}
        time_start = time.monotonic_ns()
        xgr.apply_token_bitmask_inplace(logits, bitmask, **kwargs)
        time_end = time.monotonic_ns()
        exec_time = (time_end - time_start) / 1e3
        torch.testing.assert_close(logits, logits_expected)

    print(f"Implementation: {impl}\t| Execution time (μs): {exec_time:.4f}")


if __name__ == "__main__":
    pytest.main(sys.argv)
