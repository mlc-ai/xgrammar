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
import argparse
from itertools import product
from typing import Any

import torch
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from xgrammar.kernels.apply_token_bitmask_inplace_cuda import apply_token_bitmask_inplace_cuda
from xgrammar.kernels.apply_token_bitmask_inplace_torch_compile import (
    apply_token_bitmask_inplace_torch_compile,
)
from xgrammar.kernels.apply_token_bitmask_inplace_triton import apply_token_bitmask_inplace_triton
from xgrammar.testing import _bool_mask_to_bitmask

IMPL_TORCH_COMPILE: str = "Torch Compile"
IMPL_TRITON: str = "Triton"
IMPL_CUDA: str = "CUDA"

ALL_IMPLS: list[str] = [IMPL_TORCH_COMPILE, IMPL_TRITON, IMPL_CUDA]


def bench_single_impl(
    impl: str,
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    logits_expected: torch.Tensor,
    kwargs: dict[str, Any],
    args: argparse.Namespace,
) -> float:
    if impl == IMPL_TORCH_COMPILE:
        f = lambda: apply_token_bitmask_inplace_torch_compile(logits, bitmask, **kwargs)
    elif impl == IMPL_TRITON:
        f = lambda: apply_token_bitmask_inplace_triton(logits, bitmask, **kwargs)
    else:
        f = lambda: apply_token_bitmask_inplace_cuda(logits, bitmask, **kwargs)

    f()
    torch.testing.assert_close(logits, logits_expected.to("cuda"))

    torch.cuda.synchronize()
    exec_time = do_bench(f, warmup=args.warmup, rep=args.rep)
    return exec_time * 1000


def bench_single_setup(batch_size: int, masked_cnt: int, args: argparse.Namespace) -> list[float]:
    vocab_size = args.vocab_size
    stride = args.stride
    logits_dtype = getattr(torch, args.logits_dtype)
    logits = torch.randn(batch_size, vocab_size, dtype=logits_dtype, device="cuda")
    if masked_cnt >= vocab_size:
        bool_mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device="cuda")
    else:
        bool_mask = torch.ones(batch_size, vocab_size, dtype=torch.bool, device="cuda")
        if masked_cnt > 0:
            masked_positions = torch.stack(
                [torch.randperm(vocab_size, device="cuda")[:masked_cnt] for _ in range(batch_size)]
            )
            bool_mask.scatter_(1, masked_positions, False)
            assert (bool_mask.sum(dim=-1) + masked_cnt == vocab_size).all().item()
    bitmask = _bool_mask_to_bitmask(bool_mask)
    masked_batch_ids = torch.arange(0, batch_size, stride, dtype=torch.int32, device="cuda")
    kwargs = {} if stride == 1 else {"indices": masked_batch_ids}

    logits_copies = [logits.clone() for _ in range(len(args.impl))]
    logits[masked_batch_ids] = torch.masked_fill(
        logits[masked_batch_ids], ~bool_mask[masked_batch_ids], float("-inf")
    )
    return [
        bench_single_impl(impl, logits_copy, bitmask, logits, kwargs, args)
        for impl, logits_copy in zip(args.impl, logits_copies)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impl",
        type=str,
        nargs="*",
        choices=ALL_IMPLS,
        default=[IMPL_TORCH_COMPILE, IMPL_TRITON, IMPL_CUDA],
    )
    parser.add_argument("--batch-size", type=int, nargs="*", default=[1, 8, 64, 512, 4096])
    parser.add_argument("--vocab-size", type=int, default=128000)
    parser.add_argument("--masked-cnt", type=int, nargs="*", default=[1, 64000, 127000])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--logits_dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--rep", type=int, default=2000)
    args = parser.parse_args()

    data_rows = []
    for batch_size, masked_cnt in tqdm(list(product(args.batch_size, args.masked_cnt))):
        all_us = bench_single_setup(batch_size, masked_cnt, args)
        data_rows.append(
            [
                batch_size,
                args.vocab_size,
                masked_cnt,
                f"{all_us[0]:.2f}",
                *[f"{us:.2f} ({all_us[0]/us:>4.2f}x)" for us in all_us[1:]],
            ]
        )

    print(
        tabulate(
            data_rows,
            headers=[
                "Batch\nsize",
                "Vocab\nsize",
                "Masked cnt",
                f"{args.impl[0]}\nBaseline us",
                *[f"{impl} \nus (speedup)" for impl in args.impl[1:]],
            ],
            tablefmt="pipe",
            floatfmt=".2f",
            colalign=["right"] * len(data_rows[0]),
        )
    )
