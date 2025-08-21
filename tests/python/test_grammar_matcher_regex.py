import sys
import time

import pytest
import torch
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.testing import _get_masked_tokens_from_bitmask, _is_grammar_accept_string


def test_simple():
    regex_str = "abc"
    grammar = xgr.Grammar.from_regex(regex_str)
    assert _is_grammar_accept_string(grammar, "abc")
    assert not _is_grammar_accept_string(grammar, "ab")
    assert not _is_grammar_accept_string(grammar, "abcd")


test_repetition_input_accepted_test_repetition = (
    ("aaa", True),
    ("abcbc", True),
    ("bcbcbcbcbc", True),
    ("bcbcbcbcbcbcbcb", True),
    ("d", False),
    ("aaaa", False),
)


@pytest.mark.parametrize("input, accepted", test_repetition_input_accepted_test_repetition)
def test_repetition(input: str, accepted: bool):
    regex_str = "(a|[bc]{4,}){2,3}"
    grammar = xgr.Grammar.from_regex(regex_str)
    assert _is_grammar_accept_string(grammar, input) == accepted


test_regex_accept_regex_input_accepted = [
    r"abc",
    r"[abc]+",
    r"[a-z0-9]+",
    r"[^abc]+",
    r"a*b+c?",
    r"(abc|def)+",
    r"a{2,4}",
    r"\d+",
    r"\w+",
    r"[A-Z][a-z]*",
    r"[0-9]{3}-[0-9]{3}-[0-9]{4}",
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
]


@pytest.mark.parametrize("regex_input_accepted", test_regex_accept_regex_input_accepted)
def test_regex_accept(regex_input_accepted: str):
    grammar = xgr.Grammar.from_regex(regex_input_accepted)
    assert grammar is not None


test_regex_refuse_regex_input_refused = (
    r"a{,3}",  # Invalid range
    r"a{3,2}",  # Invalid range (max < min)
    r"[z-a]",  # Invalid range (max < min)
    r"a++",  # Invalid repetition
    r"(?=a)",  # Lookahead not supported
    r"(?!a)",  # Negative lookahead not supported
)


@pytest.mark.parametrize("regex_input_refused", test_regex_refuse_regex_input_refused)
def test_regex_refuse(regex_input_refused: str):
    with pytest.raises(RuntimeError):
        xgr.Grammar.from_regex(regex_input_refused)


test_advanced_regex_string_instance_is_accepted = [
    # Basic patterns
    (r"abc", "abc", True),
    (r"abc", "def", False),
    # Character classes
    (r"[abc]+", "aabbcc", True),
    (r"[abc]+", "abcd", False),
    (r"[a-z0-9]+", "abc123", True),
    (r"[a-z0-9]+", "ABC", False),
    (r"[^abc]+", "def", True),
    (r"[^abc]+", "aaa", False),
    # Lazy character class
    (r"[abc]+?abc", "aabc", True),
    # Quantifiers
    (r"a*b+c?", "b", True),
    (r"a*b+c?", "aaabbc", True),
    (r"a*b+c?", "c", False),
    # Alternation
    (r"(abc|def)+", "abcdef", True),
    (r"(abc|def)+", "abcabc", True),
    (r"(abc|def)+", "ab", False),
    # Repetition ranges
    (r"a{2,4}", "aa", True),
    (r"a{2,4}", "aaaa", True),
    (r"a{2,4}", "a", False),
    (r"a{2,4}", "aaaaa", False),
    # Common patterns
    (r"\d+", "123", True),
    (r"\d+", "abc", False),
    (r"\w+", "abc123", True),
    (r"\w+", "!@#", False),
    (r"[A-Z][a-z]*", "Hello", True),
    (r"[A-Z][a-z]*", "hello", False),
    # Complex patterns
    (r"[0-9]{3}-[0-9]{3}-[0-9]{4}", "123-456-7890", True),
    (r"[0-9]{3}-[0-9]{3}-[0-9]{4}", "12-34-567", False),
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}", "test@email.com", True),
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}", "invalid.email", False),
]


@pytest.mark.parametrize(
    "regex_string, instance, is_accepted", test_advanced_regex_string_instance_is_accepted
)
def test_advanced(regex_string: str, instance: str, is_accepted: bool):
    grammar = xgr.Grammar.from_regex(regex_string)
    assert _is_grammar_accept_string(grammar, instance) == is_accepted


regex_input_str_test_fill_next_token_bitmask = [
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}", "test@email.com"),
    (r"[0-9]{3}-[0-9]{3}-[0-9]{4}", "123-456-7890"),
]


@pytest.mark.hf_token_required
@pytest.mark.parametrize("regex, input_str", regex_input_str_test_fill_next_token_bitmask)
def test_fill_next_token_bitmask(regex: str, input_str: str):
    tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)

    time_start = time.monotonic_ns()
    compiled_grammar = compiler.compile_regex(regex)
    matcher = xgr.GrammarMatcher(compiled_grammar)
    time_end = time.monotonic_ns()
    print(f"Time to init GrammarMatcher: {(time_end - time_start) / 1e3} us")

    input_bytes = input_str.encode("utf-8")
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    for c in input_bytes:
        time_start = time.monotonic_ns()
        assert matcher.fill_next_token_bitmask(token_bitmask)
        time_end = time.monotonic_ns()
        print(f"Time to fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")

        time_start = time.monotonic_ns()
        assert matcher.accept_string(bytes([c]))
        time_end = time.monotonic_ns()
        print(f"Time to accept char {chr(c)}: {(time_end - time_start) / 1e3} us")

    matcher.fill_next_token_bitmask(token_bitmask)
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert tokenizer.eos_token_id not in rejected_token_ids


@pytest.mark.hf_token_required
def test_regex_with_large_range_compilation():
    regex_with_large_range = r"[a-z]{100,20000}"
    tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)

    time_start = time.monotonic_ns()
    _ = compiler.compile_regex(regex_with_large_range)
    time_end = time.monotonic_ns()
    print(f"Time to compile regex with large range: {(time_end - time_start) / 1e3} us")


@pytest.mark.hf_token_required
def test_regression_lookahead_already_completed():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    xgr_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1)
    compiled_grammar = xgr_compiler.compile_regex(r"\/\*(\*+[^*\/]|[^*])*\*+\/")
    matcher = xgr.GrammarMatcher(compiled_grammar)

    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    def process_logit(input_ids: list, logit: torch.Tensor) -> torch.Tensor:
        if input_ids:
            last_token = input_ids[-1]
            assert matcher.accept_token(last_token)
        matcher.fill_next_token_bitmask(token_bitmask)
        xgr.apply_token_bitmask_inplace(logit, token_bitmask)
        return logit

    def process_tokens(tokens: list):
        for i in range(len(tokens)):
            logit = torch.zeros((tokenizer_info.vocab_size,), dtype=torch.float)
            visible_tokens = tokens[:i]
            masked_logit = process_logit(visible_tokens, logit)
            assert masked_logit[tokens[i]] != float(
                "-inf"
            ), f"token {i} ({tokens[i]}, {tokenizer.decode(tokens[i])!r}) is masked"

    text = "/*  */"
    tokens = tokenizer.encode(text)
    process_tokens(tokens)


if __name__ == "__main__":
    pytest.main(sys.argv)
