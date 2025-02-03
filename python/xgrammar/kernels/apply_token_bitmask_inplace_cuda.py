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
