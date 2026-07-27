"""Benchmark native batched draft-tree traversal against scalar Python dispatch."""

import argparse
import statistics
import string
import time
from typing import Callable

import torch

import xgrammar as xgr


def benchmark(operation: Callable[[], None], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare scalar and native batched XGrammar draft-tree traversal."
    )
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-nodes", type=int, default=6)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_nodes < 2:
        parser.error("--num-nodes must be at least 2")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be nonnegative and --iterations must be at least 1")
    if any(thread_count < 1 for thread_count in args.threads):
        parser.error("all --threads values must be at least 1")

    tokens = list(string.printable) + ["<eos>"]
    tokenizer_info = xgr.TokenizerInfo(
        tokens, vocab_size=len(tokens), stop_token_ids=[len(tokens) - 1]
    )
    literal = "".join("abcde"[index % 5] for index in range(args.num_nodes - 1))
    compiled_grammar = xgr.GrammarCompiler(tokenizer_info).compile_grammar(f'root ::= "{literal}"')

    next_token = torch.arange(1, args.num_nodes + 1, dtype=torch.int64)
    next_token[-1] = -1
    next_sibling = torch.full((args.num_nodes,), -1, dtype=torch.int64)
    draft_row = [0] + [tokens.index(character) for character in literal]
    draft_tokens = torch.tensor([draft_row] * args.batch_size, dtype=torch.int64)
    mask_rows = args.batch_size * args.num_nodes

    scalar_matchers = [xgr.GrammarMatcher(compiled_grammar) for _ in range(args.batch_size)]
    scalar_mask = xgr.allocate_token_bitmask(mask_rows, tokenizer_info.vocab_size)

    def run_scalar() -> None:
        scalar_mask.fill_(-1)
        for index, matcher in enumerate(scalar_matchers):
            begin = index * args.num_nodes
            completed = matcher.traverse_draft_tree(
                next_token,
                next_sibling,
                draft_tokens[index],
                scalar_mask[begin : begin + args.num_nodes],
            )
            if not completed:
                raise RuntimeError("scalar traversal timed out")

    scalar_seconds = benchmark(run_scalar, args.warmup, args.iterations)
    results = [("scalar", 1, scalar_seconds)]

    for thread_count in args.threads:
        batch_matchers = [xgr.GrammarMatcher(compiled_grammar) for _ in range(args.batch_size)]
        batch_mask = xgr.allocate_token_bitmask(mask_rows, tokenizer_info.vocab_size)
        batch_matcher = xgr.BatchGrammarMatcher(max_threads=thread_count)

        def run_batch() -> None:
            completed = batch_matcher.batch_traverse_draft_tree(
                batch_matchers, next_token, next_sibling, draft_tokens, batch_mask
            )
            if not all(completed):
                raise RuntimeError("batched traversal timed out")

        run_scalar()
        run_batch()
        torch.testing.assert_close(batch_mask, scalar_mask, rtol=0, atol=0)
        results.append(("batch", thread_count, benchmark(run_batch, args.warmup, args.iterations)))

    print(
        f"batch_size={args.batch_size}, nodes={args.num_nodes}, "
        f"vocab_size={tokenizer_info.vocab_size}"
    )
    print("| mode | threads | median ms | requests/s | speedup |")
    print("|---|---:|---:|---:|---:|")
    for mode, thread_count, seconds in results:
        requests_per_second = args.batch_size / seconds
        print(
            f"| {mode} | {thread_count} | {seconds * 1000:.3f} | "
            f"{requests_per_second:,.0f} | {scalar_seconds / seconds:.2f}x |"
        )


if __name__ == "__main__":
    main()
