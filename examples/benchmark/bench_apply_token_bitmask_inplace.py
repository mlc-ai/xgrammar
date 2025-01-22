import argparse
import time

import torch

from xgrammar.kernels import (
    apply_token_bitmask_inplace_cuda,
    apply_token_bitmask_inplace_triton,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, choices=["cuda", "triton"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=128000)
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_iters", type=int, default=50)
    args = parser.parse_args()

    vocab_size = args.vocab_size
    batch_size = args.batch_size
    bitmask_size = (vocab_size + 32 - 1) // 32

    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device="cuda")
    bitmask = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (batch_size, bitmask_size),
        dtype=torch.int32,
        device="cuda",
    )

    def run():
        if args.kernel == "cuda":
            apply_token_bitmask_inplace_cuda(logits, bitmask)
        elif args.kernel == "triton":
            apply_token_bitmask_inplace_triton(logits, bitmask)

    for i in range(args.num_warmup):
        run()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for i in range(args.num_iters):
        run()
    torch.cuda.synchronize()
    exec_time = time.perf_counter() - start
    exec_time = (exec_time / args.num_iters) * 10**6

    print(f"Kernel: {args.kernel}")
    print(f"Kernel execution time (us): {exec_time:.4f}")
