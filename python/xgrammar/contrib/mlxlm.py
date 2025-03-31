"""
Usage:
    python mlxlm.py --model mlx-community/Qwen2.5-Coder-32B-Instruct-3bit
"""

import argparse

import mlx.core as mx
import torch
from mlx_lm.generate import generate as mlx_generate
from mlx_lm.utils import load as mlx_load
from transformers import AutoTokenizer

import xgrammar


def apply_token_bitmask_inplace_mlx(bitmask: torch.Tensor, logits: mx.array) -> mx.array:
    """This is an easy mimic of the apply_token_bitmask_inplace function.

    NOTE: This function works with only the case of batch size is 1, which is common for on-device
          applications.

    Args:
        bitmask (torch.Tensor): Created by calling xgrammar.allocate_token_bitmask.
        logits (mx.array): Passed in from mlx_generate.

    Returns:
        mx.array: The masked logits where invalid tokens have their logits set to -inf
    """
    # If Metal implementation is available, use it
    if "metal" in xgrammar.kernels.apply_token_bitmask_inplace_kernels:
        return xgrammar.kernels.apply_token_bitmask_inplace_kernels["metal"](bitmask, logits)

    # Otherwise, fall back to the naive implementation
    vocab_size = logits.shape[-1]
    # Create mask as torch tensor for mutability
    logits_mask = torch.zeros(logits.shape, dtype=torch.float32)
    for i in range(bitmask.size(1)):
        mask_int = bitmask[0][i].item()  # batch size is 1
        for bit in range(32):
            token_idx = i * 32 + bit
            if token_idx >= vocab_size:
                break
            is_valid = (mask_int >> bit) & 1
            logits_mask[0][token_idx] = 0.0 if is_valid else float("-inf")
    # Convert to mx.array only so we can add it to logits
    return logits + mx.array(logits_mask.numpy())


class XGrammarLogitsProcessor:
    def __init__(self, grammar: xgrammar.CompiledGrammar, max_rollback_tokens: int = 16):
        self.matcher = xgrammar.GrammarMatcher(grammar, max_rollback_tokens=max_rollback_tokens)
        self.vocab_size = grammar.tokenizer_info.vocab_size
        self.bitmask = xgrammar.allocate_token_bitmask(1, self.vocab_size)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        last_token = tokens[-1].item()
        acc = self.matcher.accept_token(last_token) if not self.matcher.is_terminated() else False
        if not acc:
            self.matcher.reset()
            self.matcher.accept_token(last_token)
        if not self.matcher.is_terminated():
            self.matcher.fill_next_token_bitmask(self.bitmask)
            return apply_token_bitmask_inplace_mlx(self.bitmask, logits)
        return logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--prompt", type=str, default="Generate a simple example JSON. No text. Only the JSON"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    model, _ = mlx_load(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    mx.random.seed(args.seed)
    with_logits_processor = mlx_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True
        ),
        verbose=False,
        logits_processors=[
            XGrammarLogitsProcessor(
                grammar=xgrammar.GrammarCompiler(
                    tokenizer_info=xgrammar.TokenizerInfo.from_huggingface(tokenizer)
                ).compile_builtin_json_grammar()
            )
        ],
    )
    without_logits_processor = mlx_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True
        ),
        verbose=False,
    )
    assert without_logits_processor == with_logits_processor


if __name__ == "__main__":
    main()
