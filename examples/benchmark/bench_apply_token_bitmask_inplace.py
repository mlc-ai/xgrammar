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

import torch
from triton.testing import do_bench

from xgrammar.kernels import apply_token_bitmask_inplace_kernels
from xgrammar.testing import _bool_mask_to_bitmask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", type=str, choices=["cuda", "triton"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--vocab_size", type=int, default=128000)
    parser.add_argument("--masked_cnt", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--logits_dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--rep", type=int, default=2000)
    args = parser.parse_args()

    vocab_size = args.vocab_size
    batch_size = args.batch_size
    bitmask_size = (vocab_size + 32 - 1) // 32
    masked_cnt = args.masked_cnt
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

    logits_expected = logits.clone()
    logits_expected[masked_batch_ids] = torch.masked_fill(
        logits_expected[masked_batch_ids], ~bool_mask[masked_batch_ids], float("-inf")
    )

    if args.impl == "cuda":
        if "cuda" not in apply_token_bitmask_inplace_kernels:
            raise ImportError("CUDA is not installed")
        f = lambda: apply_token_bitmask_inplace_kernels["cuda"](logits, bitmask, **kwargs)
    elif args.impl == "triton":
        if "triton" not in apply_token_bitmask_inplace_kernels:
            raise ImportError("Triton is not installed")
        f = lambda: apply_token_bitmask_inplace_kernels["triton"](logits, bitmask, **kwargs)

    f()
    torch.testing.assert_close(logits, logits_expected.to("cuda"))

    torch.cuda.synchronize()
    exec_time = do_bench(f, warmup=args.warmup, rep=args.rep)
    exec_time *= 10**3

    print(f"Implementation: {args.impl}\t| Execution time (μs): {exec_time:.4f}")
