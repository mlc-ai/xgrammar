"""This test uses the optimized JSON grammar provided by the grammar library."""

import sys
import time
from typing import Callable, List, Optional, Tuple

import pytest
import torch

import xgrammar as xgr
from xgrammar.testing import (
    _bool_mask_to_bitmask,
    _get_masked_tokens_from_bitmask,
    _is_single_token_bitmask,
)

_is_cuda_available = torch.cuda.is_available()
_is_mps_available = torch.backends.mps.is_available()


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


def test_is_single_token_bitmask():
    batch = 2
    batch_index = 1
    vocab_size = 1024
    token_id = 100

    bool_mask = torch.zeros(batch, vocab_size, dtype=torch.bool)
    bitmask = _bool_mask_to_bitmask(bool_mask)
    assert _is_single_token_bitmask(bitmask, vocab_size, batch_index) == (False, -1)
    bool_mask[batch_index, token_id] = True
    bitmask = _bool_mask_to_bitmask(bool_mask)
    assert _is_single_token_bitmask(bitmask, vocab_size, batch_index) == (True, token_id)
    bool_mask[batch_index, token_id + 1] = True
    bitmask = _bool_mask_to_bitmask(bool_mask)
    assert _is_single_token_bitmask(bitmask, vocab_size, batch_index) == (False, -1)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_apply_token_bitmask_inplace(device: str):
    if device == "cuda" and not _is_cuda_available:
        pytest.skip(reason="CUDA is not installed")

    neginf = float("-inf")
    bool_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool, device=device)
    bitmask = torch.tensor([0b1010101010], dtype=torch.int32, device=device)
    logits = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32, device=device
    )
    expected = torch.where(bool_mask, logits, neginf)

    xgr.apply_token_bitmask_inplace(logits, bitmask)
    torch.testing.assert_close(logits, expected)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_apply_token_bitmask_inplace_shape_stride_mismatch(device: str):
    if device == "cuda" and not _is_cuda_available:
        pytest.skip(reason="CUDA is not installed")

    col = 100
    compacted_col = (col + 31) // 32
    neginf = float("-inf")
    # Mask even positions (0-indexed) in the first row, and
    # mask odd positions in the second row.
    bool_mask = torch.tensor(
        [[i % 2 == 0 for i in range(col)], [i % 2 == 1 for i in range(col)]],
        dtype=torch.bool,
        device=device,
    )
    # In int32 binary representation,
    # 0x55555555 = 1431655765
    # 0xAAAAAAAA = -1431655766
    bitmask = torch.tensor(
        [[1431655765] * compacted_col, [-1431655766] * compacted_col],
        dtype=torch.int32,
        device=device,
    )
    master_logits = torch.tensor(
        [[i + 0.1 for i in range(col + 1)], [i + 0.2 for i in range(col + 1)]],
        dtype=torch.float32,
        device=device,
    )
    logits = master_logits[:, :col]

    # Ensure the test environment setup is accurate (i.e. shape[-1] != stride[0])
    assert logits.size() == (2, col)
    assert logits.stride() == (col + 1, 1)

    expected = torch.where(bool_mask, logits, neginf)

    xgr.apply_token_bitmask_inplace(logits, bitmask)
    torch.testing.assert_close(logits, expected)


def get_apply_token_bitmask_kernel(impl: str) -> Callable:
    if impl == "cpu":
        from xgrammar.kernels.apply_token_bitmask_inplace_cpu import apply_token_bitmask_inplace_cpu

        return apply_token_bitmask_inplace_cpu
    elif impl == "cuda":
        from xgrammar.kernels.apply_token_bitmask_inplace_cuda import (
            apply_token_bitmask_inplace_cuda,
        )

        return apply_token_bitmask_inplace_cuda
    elif impl == "triton":
        from xgrammar.kernels.apply_token_bitmask_inplace_triton import (
            apply_token_bitmask_inplace_triton,
        )

        return apply_token_bitmask_inplace_triton
    elif impl == "metal":
        from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

        return apply_token_bitmask_mlx
    elif impl == "torch_compile":
        from xgrammar.kernels.apply_token_bitmask_inplace_torch_compile import (
            apply_token_bitmask_inplace_torch_compile,
        )

        return apply_token_bitmask_inplace_torch_compile
    else:
        raise ValueError(f"Invalid implementation: {impl}")


@pytest.mark.parametrize("impl", ("cpu", "cuda", "triton", "metal", "torch_compile"))
def test_apply_token_bitmask_inplace_kernel(impl: str):
    if impl in ["cuda", "triton", "torch_compile"] and not _is_cuda_available:
        pytest.skip(reason="CUDA is not installed")
    elif impl == "metal" and not _is_mps_available:
        pytest.skip(reason="MLX is not installed")

    kernel = get_apply_token_bitmask_kernel(impl)

    neginf = float("-inf")
    bool_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32)
    expected = torch.where(bool_mask, logits, neginf)

    if impl in ["cuda", "triton", "torch_compile"]:
        logits_gpu = logits.to("cuda")
        bitmask = torch.tensor([0b1010101010], dtype=torch.int32).to("cuda")
        kernel(logits_gpu, bitmask)
        torch.cuda.synchronize()
        torch.testing.assert_close(logits_gpu, expected.to("cuda"))
    elif impl == "metal":
        # Import MLX only when needed for the Metal test
        import mlx.core as mx

        bitmask = mx.array([0b1010101010], dtype=mx.int32)
        logits = mx.array(logits.numpy())
        result = kernel(bitmask, logits, vocab_size=10)
        expected = mx.array(expected.numpy())
        assert mx.allclose(result, expected)
    else:
        assert impl == "cpu"
        bitmask = torch.tensor([0b1010101010], dtype=torch.int32)
        kernel(logits, bitmask)
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
@pytest.mark.parametrize("impl", ("cpu", "cuda", "triton", "torch_compile"))
def test_apply_token_bitmask_inplace_kernel_large(
    batch_size: int, vocab_size: int, masked_cnt: int, stride: int, logits_dtype: str, impl: str
):
    if impl == "cpu" and logits_dtype != "float32":
        pytest.skip(reason="CPU implementation supports float32 only")
    elif impl in ["cuda", "triton", "torch_compile"] and not _is_cuda_available:
        pytest.skip(reason="CUDA is not installed")

    kernel = get_apply_token_bitmask_kernel(impl)

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
    if impl in ["cuda", "triton", "torch_compile"]:
        logits_gpu = logits.to("cuda")
        bitmask_gpu = bitmask.to("cuda")
        indices = batch_indices.to("cuda") if stride != 1 else None
        f = lambda: kernel(logits_gpu, bitmask_gpu, indices=indices)

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
        assert impl == "cpu"
        indices = batch_indices.tolist() if stride != 1 else None
        time_start = time.monotonic_ns()
        kernel(logits, bitmask, indices=indices)
        time_end = time.monotonic_ns()
        exec_time = (time_end - time_start) / 1e3
        torch.testing.assert_close(logits, logits_expected)

    print(
        f"Batch: {batch_size:2} | Vocab: {vocab_size:6} | Masked: {masked_cnt:6} | "
        f"Stride: {stride:1} | DType: {str(logits_dtype):15} | Impl: {impl:6} | "
        f"Execution time (μs): {exec_time:.4f}"
    )


logits_shape__bitmask_shape__vocab_size = [
    # logits is larger
    ((2, 130), (2, 4), None),
    # bitmask is larger
    ((2, 120), (2, 4), None),
    # vocab size is specified
    ((2, 130), (2, 4), 120),
]


@pytest.mark.parametrize(
    "logits_shape, bitmask_shape, vocab_size", logits_shape__bitmask_shape__vocab_size
)
@pytest.mark.parametrize("impl", ("cpu", "triton", "torch_compile"))
def test_apply_token_bitmask_inplace_vocab_size(
    logits_shape: Tuple[int, int],
    bitmask_shape: Tuple[int, int],
    vocab_size: Optional[int],
    impl: str,
):
    if impl in ["triton", "torch_compile"] and not _is_cuda_available:
        pytest.skip(reason="CUDA is not installed")

    kernel = get_apply_token_bitmask_kernel(impl)

    logits_dtype = torch.float32
    logits = torch.ones(logits_shape, dtype=logits_dtype)

    bitmask = torch.zeros(bitmask_shape, dtype=torch.int32)

    vocab_size = min(logits_shape[1], bitmask_shape[1] * 32) if vocab_size is None else vocab_size
    logits_expected = logits.clone()
    logits_expected[..., :vocab_size] = float("-inf")

    if impl in ["triton", "torch_compile"]:
        logits_gpu = logits.to("cuda")
        bitmask_gpu = bitmask.to("cuda")
        kernel(logits_gpu, bitmask_gpu, vocab_size=vocab_size)
        torch.testing.assert_close(logits_gpu, logits_expected.to("cuda"))
    else:
        assert impl == "cpu"
        kernel(logits, bitmask, vocab_size=vocab_size)
        torch.testing.assert_close(logits, logits_expected)


logits_batch_size__bitmask_batch_size__vocab_size__indices = [
    (3, 3, 128, [0, 1]),
    (2, 3, 128, [0]),
    (3, 2, 130, [0]),
]


@pytest.mark.parametrize(
    "logits_batch_size, bitmask_batch_size, vocab_size, indices",
    logits_batch_size__bitmask_batch_size__vocab_size__indices,
)
@pytest.mark.parametrize("impl", ("cpu", "cuda", "triton", "torch_compile"))
def test_apply_token_bitmask_inplace_indices(
    logits_batch_size: int, bitmask_batch_size: int, vocab_size: int, indices: List[int], impl: str
):
    if impl in ["cuda", "triton", "torch_compile"] and not _is_cuda_available:
        pytest.skip(reason="CUDA is not installed")

    kernel = get_apply_token_bitmask_kernel(impl)

    logits = torch.ones(logits_batch_size, vocab_size, dtype=torch.float32)
    bool_mask = torch.zeros(bitmask_batch_size, vocab_size, dtype=torch.bool)
    bitmask = _bool_mask_to_bitmask(bool_mask)

    logits_expected = logits.clone()
    logits_expected[indices] = torch.masked_fill(
        logits_expected[indices], ~bool_mask[indices], float("-inf")
    )

    if impl in ["cuda", "triton", "torch_compile"]:
        logits_gpu = logits.to("cuda")
        bitmask_gpu = bitmask.to("cuda")
        kernel(logits_gpu, bitmask_gpu, indices=indices)
        torch.testing.assert_close(logits_gpu, logits_expected.to("cuda"))
    else:
        assert impl == "cpu"
        kernel(logits, bitmask, indices=indices)
        torch.testing.assert_close(logits, logits_expected)


if __name__ == "__main__":
    pytest.main(sys.argv)
