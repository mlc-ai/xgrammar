"""Local pressure tests for representative large-grammar workloads."""

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, List

import pytest
from transformers import AutoTokenizer

import xgrammar as xgr


def _running_in_ci() -> bool:
    true_values = {"1", "true", "yes", "on"}
    return any(
        os.environ.get(name, "").strip().lower() in true_values for name in ("CI", "GITHUB_ACTIONS")
    )


pytestmark = [
    pytest.mark.skipif(_running_in_ci(), reason="Pressure tests are intended for local runs"),
    pytest.mark.hf_token_required,
]


@dataclass
class Workload:
    name: str
    compile_kind: str
    source: Any
    valid_input: str


def _make_flat_json_workload(size: int) -> Workload:
    names = [f"field_{i:05d}" for i in range(size)]
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in names},
        "required": names,
        "additionalProperties": False,
    }
    instance = dict.fromkeys(names, "x")
    return Workload(
        name=f"flat_json_{size}",
        compile_kind="json",
        source=json.dumps(schema, separators=(",", ":")),
        valid_input=json.dumps(instance, separators=(",", ":")),
    )


def _make_dense_patterns(size: int) -> List[str]:
    return ["@" + hashlib.sha1(f"dispatch-{i}".encode()).hexdigest()[:15] for i in range(size)]


def _make_shared_prefix_patterns(size: int) -> List[str]:
    return [f"<function=namespace.deep.tool_{i:05d}>" for i in range(size)]


def _make_dispatch_workload(
    name: str, patterns: List[str], *, unique_content: bool = False
) -> Workload:
    rules = [
        [pattern, {"type": "const_string", "value": f"X{i:05d}" if unique_content else "X"}]
        for i, pattern in enumerate(patterns)
    ]
    selected = len(patterns) - 269
    content = f"X{selected:05d}" if unique_content else "X"
    valid_input = ("free" + patterns[selected] + content) * 16 + "tail"
    return Workload(
        name=name,
        compile_kind="structural_tag",
        source={
            "type": "structural_tag",
            "format": {"type": "dispatch", "rules": rules, "loop": True, "excludes": []},
        },
        valid_input=valid_input,
    )


def _make_ring_workload(size: int) -> Workload:
    lines = ["root ::= rule_00000"]
    for index in range(size):
        next_index = (index + 1) % size
        lines.append(f'rule_{index:05d} ::= "a{index:05d}" rule_{next_index:05d} | "z{index:05d}"')
    return Workload(
        name=f"ring_ebnf_{size}",
        compile_kind="grammar",
        source="\n".join(lines) + "\n",
        valid_input="z00000",
    )


def _get_workload(name: str) -> Workload:
    if name == "flat_json_20000":
        return _make_flat_json_workload(20_000)
    if name == "flat_json_50000":
        return _make_flat_json_workload(50_000)
    if name == "ac_dense_4000":
        return _make_dispatch_workload(name, _make_dense_patterns(4_000))
    if name == "ac_dense_unique_4000":
        return _make_dispatch_workload(name, _make_dense_patterns(4_000), unique_content=True)
    if name == "ac_shared_prefix_4000":
        return _make_dispatch_workload(name, _make_shared_prefix_patterns(4_000))
    if name == "ring_ebnf_4000":
        return _make_ring_workload(4_000)
    raise ValueError(f"Unknown pressure workload: {name}")


def _compile_workload(compiler: xgr.GrammarCompiler, workload: Workload) -> xgr.CompiledGrammar:
    if workload.compile_kind == "json":
        return compiler.compile_json_schema(
            workload.source, any_whitespace=True, max_whitespace_cnt=4
        )
    if workload.compile_kind == "grammar":
        return compiler.compile_grammar(workload.source)
    if workload.compile_kind == "structural_tag":
        return compiler.compile_structural_tag(workload.source)
    raise ValueError(f"Unknown compile kind: {workload.compile_kind}")


def _nearest_rank_percentile(samples: List[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _token_is_allowed(bitmask, token_id: int) -> bool:
    word = int(bitmask[0, token_id // 32])
    return ((word >> (token_id % 32)) & 1) == 1


def _measure_mask_calls(
    compiled: xgr.CompiledGrammar, tokenizer: Any, valid_input: str
) -> List[float]:
    token_ids = tokenizer.encode(valid_input, add_special_tokens=False)[:96]
    assert token_ids
    measured_repeats = max(5, math.ceil(512 / len(token_ids)))
    samples = []
    for repeat in range(measured_repeats + 1):
        matcher = xgr.GrammarMatcher(compiled)
        bitmask = xgr.allocate_token_bitmask(1, compiled.tokenizer_info.vocab_size)
        for token_id in token_ids:
            xgr.reset_token_bitmask(bitmask)
            start = time.perf_counter_ns()
            matcher.fill_next_token_bitmask(bitmask)
            elapsed_us = (time.perf_counter_ns() - start) / 1_000
            assert _token_is_allowed(bitmask, token_id)
            assert matcher.accept_token(token_id)
            if repeat > 0:
                samples.append(elapsed_us)
    return samples


@pytest.fixture(scope="module")
def glm_tokenizer():
    return AutoTokenizer.from_pretrained("zai-org/GLM-5.2", trust_remote_code=True)


@pytest.fixture(scope="module")
def glm_tokenizer_info(glm_tokenizer):
    return xgr.TokenizerInfo.from_huggingface(glm_tokenizer)


@pytest.mark.parametrize(
    "workload_name",
    [
        "flat_json_20000",
        "flat_json_50000",
        "ac_dense_4000",
        "ac_dense_unique_4000",
        "ac_shared_prefix_4000",
        "ring_ebnf_4000",
    ],
)
def test_pressure_workload(workload_name: str, glm_tokenizer, glm_tokenizer_info):
    workload = _get_workload(workload_name)
    compiler = xgr.GrammarCompiler(
        glm_tokenizer_info, max_threads=8, cache_enabled=True, cache_limit_bytes=512 * 1024 * 1024
    )

    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    compiled = _compile_workload(compiler, workload)
    compile_wall_ms = (time.perf_counter() - wall_start) * 1_000
    compile_cpu_ms = (time.process_time() - cpu_start) * 1_000

    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string(workload.valid_input)
    assert matcher.is_terminated()

    mask_samples = _measure_mask_calls(compiled, glm_tokenizer, workload.valid_input)
    result = {
        "workload": workload.name,
        "compile_wall_ms": compile_wall_ms,
        "compile_cpu_ms": compile_cpu_ms,
        "mask_p50_us": _nearest_rank_percentile(mask_samples, 0.50),
        "mask_p99_us": _nearest_rank_percentile(mask_samples, 0.99),
        "mask_samples": len(mask_samples),
    }
    print("PRESSURE_RESULT " + json.dumps(result, sort_keys=True))
