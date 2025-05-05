"""
Usage:
    python mlxlm.py --model mlx-community/Qwen2.5-Coder-32B-Instruct-3bit
"""

import argparse

import mlx.core as mx
from mlx_lm.generate import generate as mlx_generate
from mlx_lm.utils import load as mlx_load
from transformers import AutoTokenizer

import xgrammar
from xgrammar.kernels import apply_token_bitmask_inplace_kernels


class XGrammarLogitsProcessor:
    def __init__(self, grammar: xgrammar.CompiledGrammar, max_rollback_tokens: int = 16):
        self.matcher = xgrammar.GrammarMatcher(grammar, max_rollback_tokens=max_rollback_tokens)
        self.vocab_size = grammar.tokenizer_info.vocab_size
        self.bitmask = xgrammar.allocate_token_bitmask(1, self.vocab_size)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        assert tokens.size > 0  # In the first call, tokens.size == #tokens in prompt
        last_token = tokens[-1].item()
        acc = self.matcher.accept_token(last_token) if not self.matcher.is_terminated() else False
        if not acc:
            self.matcher.reset()
            self.matcher.accept_token(last_token)
        if not self.matcher.is_terminated():
            self.matcher.fill_next_token_bitmask(self.bitmask)
            return apply_token_bitmask_inplace_kernels["metal"](
                mx.array(self.bitmask.numpy()), logits, self.vocab_size
            )
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
    print(without_logits_processor)


if __name__ == "__main__":
    main()
