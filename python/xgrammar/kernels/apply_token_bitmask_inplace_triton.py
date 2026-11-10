from typing import List, NamedTuple, Optional, Union

import torch

try:
    import triton
    import triton.language as tl
except ImportError as err:
    raise ImportError("Triton is not installed") from err


@triton.jit
def apply_token_bitmask_inplace_kernel(
    logits_ptr,
    bitmask_ptr,
    indices_ptr,
    num_rows,
    vocab_size,
    logits_strides,
    bitmask_strides,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply a bitmask to logits in-place using Triton. The bitmask is a 01 bitwise compressed tensor,
    where 0 means the token is masked and 1 means the token is not masked. After applying the bitmask,
    the masked logits will be set to -inf.

    Parameters
    ----------
    logits_ptr : tl.tensor
        Pointer to the logits tensor to apply the bitmask to.

    bitmask_ptr : tl.tensor
        Pointer to the bitmask tensor to apply.

    indices_ptr : Optional[tl.tensor]
        Optional pointer to indices tensor specifying which rows to apply the mask to.

    num_rows : int
        Number of rows to process. If indices_ptr is provided, this is the number of unique indices.

    vocab_size : int
        Size of the vocabulary dimension. If the logits does not have a vocab padding, this is the
        same as the logits's second dimension. Otherwise, this is the actual size of the vocabulary.

    logits_strides : int
        Stride between rows in the logits tensor.

    bitmask_strides : int
        Stride between rows in the bitmask tensor.

    NUM_SMS : int
        Number of streaming multiprocessors to use.

    BLOCK_SIZE : int
        Size of processing blocks.
    """

    pid = tl.program_id(0)
    num_blocks = tl.cdiv(vocab_size, BLOCK_SIZE)
    for work_id in tl.range(pid, num_rows * num_blocks, NUM_SMS):
        row_id = work_id // num_blocks
        block_offset = (work_id % num_blocks) * BLOCK_SIZE
        batch_id = row_id if indices_ptr is None else tl.load(indices_ptr + row_id)
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        bitmask_offsets = block_offset // 32 + tl.arange(0, BLOCK_SIZE // 32)
        vocab_mask = offsets < vocab_size
        packed_bitmask_mask = bitmask_offsets < bitmask_strides
        packed_bitmask = tl.load(
            bitmask_ptr + batch_id * bitmask_strides + bitmask_offsets, packed_bitmask_mask
        )
        bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
        bitmask = bitmask.reshape(BLOCK_SIZE)

        tl.store(
            logits_ptr + batch_id * logits_strides + offsets, -float("inf"), vocab_mask & bitmask
        )


class _LaunchConfig(NamedTuple):
    num_programs: int
    """Number of persistent programs to launch, one per SM (CUDA) or Xe-core (XPU)."""
    warp_size: int
    """Number of threads in a warp (CUDA) or sub-group (XPU)."""
    max_num_warps: Optional[int]
    """Upper bound on warps per program imposed by the device, or None if no bound applies."""


def _get_launch_config(device: torch.device) -> _LaunchConfig:
    """Return the launch configuration of the persistent kernel for the given device.
    Supports CUDA (NVIDIA / AMD ROCm) and Intel XPU devices."""
    if device.type == "xpu":
        props = torch.xpu.get_device_properties(device)
        # Xe-cores are the closest analogue of CUDA SMs. Fall back to the EU count on older
        # PyTorch versions that do not expose the sub-slice count.
        num_programs = getattr(props, "gpu_subslice_count", 0) or props.max_compute_units
        # Triton on Intel GPUs maps a warp to a sub-group, 32 lanes by default.
        warp_size = 32
        # A program runs as one work-group, which must not exceed the device limit. Some
        # integrated GPUs allow fewer threads than the 1024 that fp32 logits would use.
        max_work_group_size = getattr(props, "max_work_group_size", 1024)
        return _LaunchConfig(num_programs, warp_size, max(1, max_work_group_size // warp_size))

    props = torch.cuda.get_device_properties(device)
    if torch.version.hip is not None and "gfx1" not in props.gcnArchName:
        # For AMD GPUs (non-Navi)
        warp_size = 64
    else:
        warp_size = 32
    return _LaunchConfig(props.multi_processor_count, warp_size, max_num_warps=None)


def apply_token_bitmask_inplace_triton(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    vocab_size: Optional[int] = None,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
):
    BLOCK_SIZE = 4096
    launch_config = _get_launch_config(logits.device)
    NUM_SMS = launch_config.num_programs
    num_warps = BLOCK_SIZE // launch_config.warp_size // (16 // logits.element_size())
    if launch_config.max_num_warps is not None:
        num_warps = min(num_warps, launch_config.max_num_warps)

    assert bitmask.dtype == torch.int32, "bitmask must be of type int32"

    detected_vocab_size = min(logits.shape[-1], bitmask.shape[-1] * 32)
    if vocab_size is None:
        vocab_size = detected_vocab_size
    else:
        assert (
            vocab_size <= detected_vocab_size
        ), f"vocab_size {vocab_size} is larger than the detected vocab_size {detected_vocab_size}"

    num_rows = len(indices) if indices is not None else logits.shape[0] if logits.ndim == 2 else 1

    if indices is not None:
        if isinstance(indices, torch.Tensor):
            indices = indices.to(dtype=torch.int32, device=logits.device, non_blocking=True)
        else:
            indices_cpu = torch.tensor(indices, dtype=torch.int32)
            indices = indices_cpu.to(device=logits.device, non_blocking=True)

    grid = (NUM_SMS,)

    apply_token_bitmask_inplace_kernel[grid](
        logits,
        bitmask,
        indices,
        num_rows,
        vocab_size,
        logits.stride()[0],
        bitmask.stride()[0],
        NUM_SMS,
        BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=3,
    )
