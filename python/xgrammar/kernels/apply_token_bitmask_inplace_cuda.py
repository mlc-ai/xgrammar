# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
from contextlib import suppress
from typing import List, Optional, Union

import torch
import torch.utils.cpp_extension


def _check_cuda_toolchain() -> None:
    """check if nvcc is available and if pytorch will likely find it"""
    import glob
    import os
    import shutil
    from pathlib import Path

    # First check if CUDA is available in PyTorch
    if not torch.cuda.is_available():
        raise ImportError("CUDA is not available in PyTorch")

    # This is similar logic to what pytorch does to find the nvcc compiler
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", None))
        if cuda_home is None:
            if os.name == "nt":
                # This is a very hardcoded asumption about install directories but pytorch does this.
                cuda_homes = glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*")

                if len(cuda_homes) == 0:
                    cuda_home = ""
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = "/usr/local/cuda"

        if cuda_home is None:
            raise ImportError("No CUDA toolchain found")

        nvcc_path = str(Path(cuda_home) / "bin" / "nvcc")

    if not os.path.exists(nvcc_path):
        raise ImportError(f"nvcc compiler not found at {nvcc_path}")


def _remove_torch_nvcc_flags() -> None:
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        with suppress(ValueError):
            torch.utils.cpp_extension.COMMON_NVCC_FLAGS.remove(flag)


def _load_torch_ops() -> None:
    from pathlib import Path

    torch_op_file_path = Path(__file__).with_suffix(".cu")
    with open(torch_op_file_path) as f:
        source = f.read()
    cflags = ["-O3"]
    if platform.system() != "Windows":
        cflags.append("-Wno-switch-bool")
    cuda_cflags = ["-O3", "-std=c++17", "--threads", "4", "-use_fast_math"]
    # Use the safer cpp_extension.load_inline instead of cpp_extension.load
    torch.utils.cpp_extension.load_inline(
        name="xgrammar",
        cpp_sources=[],  # No C++ sources
        cuda_sources=[source],
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        with_cuda=True,
        is_python_module=False,
    )


_check_cuda_toolchain()
_remove_torch_nvcc_flags()
_load_torch_ops()


_is_register_fake_available = hasattr(torch, "library") and hasattr(torch.library, "register_fake")

if _is_register_fake_available:
    # To support torch.compile with fullgraph=True, a fake kernel is needed.
    @torch.library.register_fake("xgrammar::apply_token_bitmask_inplace_cuda")
    def _(
        logits: torch.Tensor, bitmask: torch.Tensor, indices: Optional[torch.Tensor] = None
    ) -> None:
        pass


def apply_token_bitmask_inplace_cuda(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: Optional[Union[List[int], torch.Tensor]] = None,
) -> None:
    if isinstance(indices, list):
        indices = torch.tensor(indices, dtype=torch.int32, device=logits.device)
    if indices is not None:
        indices = indices.to(logits.device)
    torch.ops.xgrammar.apply_token_bitmask_inplace_cuda(logits, bitmask, indices)
