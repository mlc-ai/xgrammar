from typing import List, Optional, Union

import torch
import torch.utils.cpp_extension


def _load_torch_ops():
    src = __file__.replace(".py", ".cu")
    cflags = ["-O3", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        "--threads",
        "4",
        "-use_fast_math",
    ]
    torch.utils.cpp_extension.load(
        name="xgrammar",
        sources=[src],
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        with_cuda=True,
        is_python_module=False,
        is_standalone=False,
    )


_load_torch_ops()


@torch.library.register_fake("xgrammar::apply_token_bitmask_inplace_cuda")
def _(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
) -> None:
    pass


def apply_token_bitmask_inplace_cuda(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
) -> None:
    if isinstance(indices, list):
        indices = torch.tensor(indices, dtype=torch.int32, device=logits.device)
    torch.ops.xgrammar.apply_token_bitmask_inplace_cuda(logits, bitmask, indices)
