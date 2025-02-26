"""This test uses the optimized JSON grammar provided by the grammar library."""

import sys
import time
from typing import List, Optional, Tuple

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
    if impl == "cuda" and "cuda" not in xgr.kernels.apply_token_bitmask_inplace_kernels:
        pytest.skip(reason="CUDA is not installed")
    if impl == "triton" and "triton" not in xgr.kernels.apply_token_bitmask_inplace_kernels:
        pytest.skip(reason="Triton is not installed")

    neginf = float("-inf")
    bool_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32)
    expected = torch.where(bool_mask, logits, neginf)

    if impl in ["cuda", "triton"]:
        if not torch.cuda.is_available():
            pytest.skip(reason="CUDA is not installed")

        logits_gpu = logits.to("cuda")
        bitmask = torch.tensor([0b1010101010], dtype=torch.int32).to("cuda")
        if impl == "cuda":
            xgr.kernels.apply_token_bitmask_inplace_kernels["cuda"](logits_gpu, bitmask)
        else:
            xgr.kernels.apply_token_bitmask_inplace_kernels["triton"](logits_gpu, bitmask)
        torch.cuda.synchronize()
        torch.testing.assert_close(logits_gpu, expected.to("cuda"))
    else:
        bitmask = torch.tensor([0b1010101010], dtype=torch.int32)
        xgr.apply_token_bitmask_inplace(logits, bitmask)
        torch.testing.assert_close(logits, expected)


batch_size__vocab_size__masked_cnt__stride__logits_dtype = [
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
    batch_size__vocab_size__masked_cnt__stride__logits_dtype,
)
@pytest.mark.parametrize("impl", ("cpu", "cuda", "triton"))
def test_apply_token_bitmask_inplace_large(
    batch_size: int, vocab_size: int, masked_cnt: int, stride: int, logits_dtype: str, impl: str
):
    if impl == "cpu" and logits_dtype != "float32":
        pytest.skip(reason="cpu implementation supports float32 only")
    if impl == "cuda" and "cuda" not in xgr.kernels.apply_token_bitmask_inplace_kernels:
        pytest.skip(reason="CUDA is not installed")
    if impl == "triton" and "triton" not in xgr.kernels.apply_token_bitmask_inplace_kernels:
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

    batch_indices = torch.arange(0, batch_size, stride, dtype=torch.int32)

    logits_expected = logits.clone()
    logits_expected[batch_indices] = torch.masked_fill(
        logits_expected[batch_indices], ~bool_mask[batch_indices], float("-inf")
    )

    bitmask = _bool_mask_to_bitmask(bool_mask)
    if impl in ["cuda", "triton"]:
        if not torch.cuda.is_available():
            pytest.skip(reason="CUDA is not installed")

        logits_gpu = logits.to("cuda")
        bitmask_gpu = bitmask.to("cuda")
        indices = batch_indices.to("cuda") if stride != 1 else None
        if impl == "cuda":

            def f():
                return xgr.kernels.apply_token_bitmask_inplace_kernels["cuda"](
                    logits_gpu, bitmask_gpu, indices=indices
                )

        else:

            def f():
                return xgr.kernels.apply_token_bitmask_inplace_kernels["triton"](
                    logits_gpu, bitmask_gpu, indices=indices
                )

        torch.cuda.synchronize()
        f()
        torch.cuda.synchronize()
        torch.testing.assert_close(logits_gpu, logits_expected.to("cuda"))

        try:
            from triton.testing import do_bench

            exec_time = do_bench(f, warmup=100, rep=1000)
            exec_time *= 1e3
        except ImportError:
            pytest.skip(reason="Triton is not installed")
    else:
        indices = batch_indices.tolist() if stride == 1 else None
        time_start = time.monotonic_ns()
        xgr.apply_token_bitmask_inplace(logits, bitmask, indices=indices)
        time_end = time.monotonic_ns()
        exec_time = (time_end - time_start) / 1e3
        torch.testing.assert_close(logits, logits_expected)

    print(
        f"Batch: {batch_size:2} | Vocab: {vocab_size:6} | Masked: {masked_cnt:6} | "
        f"Stride: {stride:1} | DType: {str(logits_dtype):15} | Impl: {impl:6} | "
        f"Execution time (μs): {exec_time:.4f}"
    )


batch_size__logits_strides__vocab_size = [
    # logits's vocab has extra paddings (vLLM)
    (2, 161, 128),
    (2, 161, 120),
    # bitmask's vocab sizes do not align with the 32-bit block size
    (2, 130, 130),
]


@pytest.mark.parametrize(
    "batch_size, logits_strides, vocab_size", batch_size__logits_strides__vocab_size
)
@pytest.mark.parametrize("logits_dtype", ("float32", "float16", "bfloat16"))
@pytest.mark.parametrize("impl", ("cpu", "cuda", "triton"))
def test_apply_token_bitmask_inplace_special_shape(
    batch_size: int, logits_strides: int, vocab_size: int, logits_dtype: str, impl: str
):
    if impl == "cpu" and logits_dtype != "float32":
        pytest.skip(reason="cpu implementation supports float32 only")

    logits_dtype = getattr(torch, logits_dtype)
    logits = torch.ones(batch_size, logits_strides, dtype=logits_dtype)

    bool_mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool)
    bitmask = _bool_mask_to_bitmask(bool_mask)

    bool_mask_padded = torch.nn.functional.pad(bool_mask, (0, logits_strides - vocab_size), value=1)
    logits_expected = logits.clone()
    logits_expected[~bool_mask_padded] = float("-inf")

    if impl in ["cuda", "triton"]:
        logits_gpu = logits.to("cuda")
        bitmask_gpu = bitmask.to("cuda")
        if impl == "cuda":
            xgr.kernels.apply_token_bitmask_inplace_kernels["cuda"](logits_gpu, bitmask_gpu)
        else:
            xgr.kernels.apply_token_bitmask_inplace_kernels["triton"](logits_gpu, bitmask_gpu)
        torch.testing.assert_close(logits_gpu, logits_expected.to("cuda"))
    else:
        xgr.apply_token_bitmask_inplace(logits, bitmask)
        torch.testing.assert_close(logits, logits_expected)


logits_batch_size__bitmask_batch_size__vocab_size__indices = [
    (3, 3, 128, [0, 1]),
    (2, 3, 128, [0]),
    (3, 2, 130, [0]),
    (3, 2, 130, [1, 1, 0, 0]),
]


@pytest.mark.parametrize(
    "logits_batch_size, bitmask_batch_size, vocab_size, indices",
    logits_batch_size__bitmask_batch_size__vocab_size__indices,
)
@pytest.mark.parametrize("impl", ("cpu", "cuda", "triton"))
def test_apply_token_bitmask_inplace_select_indices(
    logits_batch_size: int, bitmask_batch_size: int, vocab_size: int, indices: List[int], impl: str
):
    logits = torch.ones(logits_batch_size, vocab_size, dtype=torch.float32)
    bool_mask = torch.zeros(bitmask_batch_size, vocab_size, dtype=torch.bool)
    bitmask = _bool_mask_to_bitmask(bool_mask)

    logits_expected = logits.clone()
    logits_expected[indices] = torch.masked_fill(
        logits_expected[indices], ~bool_mask[indices], float("-inf")
    )

    if impl in ["cuda", "triton"]:
        logits_gpu = logits.to("cuda")
        bitmask_gpu = bitmask.to("cuda")
        if impl == "cuda":
            xgr.kernels.apply_token_bitmask_inplace_kernels["cuda"](
                logits_gpu, bitmask_gpu, indices
            )
        else:
            xgr.kernels.apply_token_bitmask_inplace_kernels["triton"](
                logits_gpu, bitmask_gpu, indices
            )
        torch.testing.assert_close(logits_gpu, logits_expected.to("cuda"))
    else:
        xgr.apply_token_bitmask_inplace(logits, bitmask, indices=indices)
        torch.testing.assert_close(logits, logits_expected)


logits_shape__bitmask_shape__indices = [((2, 128), (1, 4), None), ((2, 128), (2, 5), None)]


@pytest.mark.parametrize(
    "logits_shape, bitmask_shape, indices", logits_shape__bitmask_shape__indices
)
@pytest.mark.parametrize("impl", ("cpu", "cuda", "triton"))
def test_apply_token_bitmask_inplace_invalid_shape(
    logits_shape: Tuple[int], bitmask_shape: Tuple[int], indices: Optional[List[int]], impl: str
):
    device = "cpu" if impl == "cpu" else "cuda"
    logits = torch.ones(logits_shape, dtype=torch.float32, device=device)
    bitmask = torch.zeros(bitmask_shape, dtype=torch.int32, device=device)

    with pytest.raises((RuntimeError, AssertionError)):
        if impl == "cuda":
            xgr.kernels.apply_token_bitmask_inplace_kernels["cuda"](logits, bitmask, indices)
        elif impl == "triton":
            xgr.kernels.apply_token_bitmask_inplace_kernels["triton"](logits, bitmask, indices)
        else:
            xgr.apply_token_bitmask_inplace(logits, bitmask, indices=indices)


if __name__ == "__main__":
    pytest.main(sys.argv)
