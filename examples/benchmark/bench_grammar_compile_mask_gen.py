"""This script benchmarks the time for grammar compilation and mask generation."""

import argparse
import json
import time

import datasets
import torch
from lmformatenforcer import JsonSchemaParser, TokenEnforcer
from lmformatenforcer.integrations.transformers import (
    TokenEnforcerTokenizerData,
    build_token_enforcer_tokenizer_data,
)
from outlines.fsm.guide import Guide, RegexGuide
from outlines.fsm.json_schema import convert_json_schema_to_str
from outlines.generate.generator import bias_logits
from outlines.generate.json import build_regex_from_schema
from outlines.models import TransformerTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer

import xgrammar as xgr

wrong_data_indices = [1]


def xgrammar_build(schema: str, grammar_compiler: xgr.GrammarCompiler):
    grammar = grammar_compiler.compile_json_schema(schema)
    matcher = xgr.GrammarMatcher(grammar)
    return matcher


def xgrammar_exec(
    matcher: xgr.GrammarMatcher, logits: torch.Tensor, bitmask: torch.Tensor, token_id: int
):
    # Logits processing
    matcher.fill_next_token_bitmask(bitmask)
    xgr.apply_token_bitmask_inplace(logits, bitmask)
    # Update state
    assert matcher.accept_token(token_id)
    return


def outlines_build(schema: str, tokenizer: TransformerTokenizer):
    schema_str = convert_json_schema_to_str(json_schema=schema)
    regex_string = build_regex_from_schema(schema_str, whitespace_pattern=None)
    guide = RegexGuide.from_regex(regex_string, tokenizer)
    return guide


def outlines_exec(guide: Guide, logits: torch.Tensor, token_id: int, state=None):
    if state is None:
        state = guide.initial_state
    # Logits processing
    allowed_tokens = guide.get_next_instruction(state).tokens
    biased_logits = bias_logits(logits.view(1, -1), [allowed_tokens])
    # Update state
    next_state = guide.get_next_state(state, token_id)
    return next_state


def lmformatenforcer_build(schema: str, tokenizer: TokenEnforcerTokenizerData):
    parser = JsonSchemaParser(json.loads(schema))
    token_enforcer = TokenEnforcer(tokenizer, parser)
    return token_enforcer


def lmformatenforcer_exec(token_enforcer: TokenEnforcer, logits: torch.Tensor, token_ids):
    # Logits processing
    allowed_tokens = token_enforcer.get_allowed_tokens(token_ids)
    logits[allowed_tokens] = float("-inf")
    # Update state
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        choices=["xgrammar", "outlines", "lmformatenforcer"],
        default="xgrammar",
    )
    parser.add_argument("--num_iters", type=int, default=5)
    parser.add_argument("--num_warmup", type=int, default=-1)
    args = parser.parse_args()

    backend = args.backend
    num_iters = args.num_iters
    num_warmup = args.num_warmup if args.num_warmup != -1 else 5 if num_iters >= 40 else 1

    dataset = datasets.load_dataset("NousResearch/json-mode-eval", split="train")

    hf_model_path = "meta-llama/Llama-3.1-8B-Instruct"

    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    xgrammar_tokenizer_info = xgr.TokenizerInfo.from_huggingface(hf_tokenizer)
    xgrammar_grammar_compiler = xgr.GrammarCompiler(xgrammar_tokenizer_info)
    outlines_tokenizer = TransformerTokenizer(hf_tokenizer)
    lmformatenforcer_tokenizer = build_token_enforcer_tokenizer_data(hf_tokenizer)

    vocab_size = len(hf_tokenizer)

    build_time = 0
    exec_time = 0
    total_data_points = 0
    total_tokens = 0
    fail_cnt = 0

    tqdm_iter = tqdm(range(-num_warmup, num_iters))
    for iter in tqdm_iter:
        if iter < 0:
            tqdm_iter.set_description(f"Backend: {backend}, Warmup Iter: {iter + num_warmup}")
        else:
            tqdm_iter.set_description(f"Backend: {backend}, Iter: {iter}")

        if iter == 0:
            # Reset time
            build_time = 0
            exec_time = 0

        tqdm_data_point_iter = tqdm(range(len(dataset)))
        for data_point_idx in tqdm_data_point_iter:
            tqdm_data_point_iter.set_description(
                f"Backend: {backend}, Data Point: {data_point_idx}"
            )
            if data_point_idx in wrong_data_indices:
                continue

            schema = dataset["schema"][data_point_idx]
            completion = dataset["completion"][data_point_idx]
            token_ids = hf_tokenizer.encode(completion, add_special_tokens=False)
            prompt = hf_tokenizer.apply_chat_template(
                dataset["prompt"][data_point_idx], tokenize=False
            )
            prompt_token_ids = hf_tokenizer.encode(prompt)
            print(f"Prompt: {prompt}, Schema: {schema}")

            start = time.perf_counter()
            try:
                if backend == "xgrammar":
                    worker = xgrammar_build(schema, xgrammar_grammar_compiler)
                    bitmask = xgr.allocate_token_bitmask(worker.vocab_size)
                elif backend == "outlines":
                    worker = outlines_build(schema, outlines_tokenizer)
                elif backend == "lmformatenforcer":
                    worker = lmformatenforcer_build(schema, lmformatenforcer_tokenizer)
            except Exception as e:
                if iter >= 0:
                    fail_cnt += 1
                continue

            build_time += time.perf_counter() - start

            # use different logits for each mask generation process
            # to avoid caching effects between different tokens
            logits = [torch.randn(vocab_size).cuda() for _ in range(len(token_ids))]

            torch.cuda.synchronize()
            start = time.perf_counter()
            fail_flag = False
            for idx, token_id in enumerate(token_ids):
                # Logits processing
                try:
                    if backend == "xgrammar":
                        xgrammar_exec(worker, logits[idx], bitmask, token_id)
                    elif backend == "outlines":
                        if idx == 0:
                            state = None
                        state = outlines_exec(worker, logits[idx], token_id, state)
                    elif backend == "lmformatenforcer":
                        lmformatenforcer_exec(
                            worker, logits[idx], prompt_token_ids + token_ids[:idx]
                        )
                except Exception as e:
                    if iter >= 0:
                        fail_cnt += 1
                    fail_flag = True
                    break

            if fail_flag:
                continue

            torch.cuda.synchronize()
            exec_time += time.perf_counter() - start

            if iter >= 0:
                total_data_points += 1
                total_tokens += len(token_ids)

    print(f"Backend: {backend}")
    print(f"Fail count: {fail_cnt / num_iters:.0f} / {len(dataset) - len(wrong_data_indices)}")
    print(f"Grammar preprocessing time (ms): {build_time / total_data_points * 1e3:.4f}")
    print(f"Mask generation time (us/token): {exec_time / total_tokens * 1e6:.4f}")
