from typing import List, Optional, Union

import torch
import torch.utils.cpp_extension


def _load_module():
    src = __file__.replace(".py", ".cu")
    cflags = ["-O3", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        "--threads",
        "4",
        "-use_fast_math",
    ]
    return torch.utils.cpp_extension.load(
        name="bitmask",
        sources=[src],
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        with_cuda=True,
    )


# @torch.library.custom_op("xgrammar::apply_token_bitmask_inplace_cuda", mutates_args=("logits",))
def apply_token_bitmask_inplace_cuda(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
) -> None:
    if isinstance(indices, list):
        indices = torch.tensor(indices, dtype=torch.int32, device=logits.device)
    _load_module().apply_token_bitmask_inplace(logits, bitmask, indices)


# @apply_token_bitmask_inplace_cuda.register_fake
# def _(logits: torch.Tensor, bitmask: torch.Tensor) -> None:
#     pass

if __name__ == "__main__":
    # export TORCH_CUDA_ARCH_LIST=9.0

    neginf = float("-inf")
    bool_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32)
    expected = torch.where(bool_mask, logits, neginf)

    logits_gpu = logits.to("cuda")
    bitmask = torch.tensor([0b1010101010], dtype=torch.int32).to("cuda")
    # xgr.apply_token_bitmask_inplace(logits_gpu, bitmask)
    apply_token_bitmask_inplace_cuda(logits_gpu, bitmask)
    torch.cuda.synchronize()
    torch.testing.assert_close(logits_gpu, expected.to("cuda"))
    breakpoint()
