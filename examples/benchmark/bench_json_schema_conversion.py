"""Benchmark direct and legacy JSON Schema grammar construction.

The legacy mode reports conversion-to-EBNF and EBNF-parser time separately. Run each mode in a
separate process when comparing two builds so only one xgrammar binding is loaded at a time.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

JSONSCHEMABENCH_CASES = (
    "JsonSchemaStore/cloudformation.schema.json",
    "JsonSchemaStore/config.json",
    "Kubernetes/kb_1161_Normalized.json",
    "JsonSchemaStore/service-schema.json",
)


@dataclass
class Case:
    name: str
    schema: str


@dataclass
class Result:
    name: str
    schema_bytes: int
    mode: str
    conversion_ms: float
    parser_ms: float
    total_ms: float


def bootstrap_xgrammar(xgrammar_root: Path):
    """Load the binding built in xgrammar_root, even if another wheel is installed."""
    import tvm_ffi.libinfo

    python_dir = xgrammar_root / "python"
    local_binding = python_dir / "xgrammar" / "libxgrammar_bindings.so"
    if not local_binding.is_file():
        raise FileNotFoundError(f"Build the Python binding first; missing {local_binding}")

    original_find = tvm_ffi.libinfo._find_library_by_basename

    def find_local_binding(package, target_name, extra_lib_paths=None):
        if package == "xgrammar" and target_name == "xgrammar_bindings":
            return local_binding
        return original_find(package, target_name, extra_lib_paths)

    tvm_ffi.libinfo._find_library_by_basename = find_local_binding
    sys.path.insert(0, str(python_dir))

    import xgrammar as xgr
    from xgrammar.testing import _json_schema_to_ebnf

    return xgr, _json_schema_to_ebnf


def make_wide_object(property_count: int) -> Case:
    properties = {f"field_{index:05d}": {"type": "string"} for index in range(property_count)}
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return Case(f"synthetic/wide_object_{property_count}", compact_json(schema))


def make_ref_object(definition_count: int) -> Case:
    definitions = {
        f"type_{index:05d}": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        for index in range(definition_count)
    }
    properties = {
        f"field_{index:05d}": {"$ref": f"#/$defs/type_{index:05d}"}
        for index in range(definition_count)
    }
    schema = {
        "type": "object",
        "$defs": definitions,
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return Case(f"synthetic/ref_object_{definition_count}", compact_json(schema))


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_cases(args: argparse.Namespace) -> list[Case]:
    cases = []
    for schema_path_str in args.schema:
        schema_path = Path(schema_path_str)
        cases.append(Case(str(schema_path), schema_path.read_text()))
    if args.jsonschemabench_root is not None:
        data_root = args.jsonschemabench_root / "data"
        for relative_path in JSONSCHEMABENCH_CASES:
            schema_path = data_root / relative_path
            cases.append(Case(f"jsonschemabench/{relative_path}", schema_path.read_text()))
    cases.extend(make_wide_object(size) for size in args.synthetic_wide)
    cases.extend(make_ref_object(size) for size in args.synthetic_refs)
    if not cases:
        raise ValueError("Provide --schema, --jsonschemabench-root, or a synthetic case")
    return cases


def median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1_000


def benchmark_direct(
    case: Case, from_json_schema: Callable, repeats: int, warmups: int, strict_mode: bool
) -> Result:
    for _ in range(warmups):
        from_json_schema(case.schema, strict_mode=strict_mode)
    samples = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        grammar = from_json_schema(case.schema, strict_mode=strict_mode)
        samples.append(time.perf_counter() - start)
        del grammar
    total_ms = median_ms(samples)
    return Result(case.name, len(case.schema.encode()), "direct", total_ms, 0.0, total_ms)


def benchmark_legacy(
    case: Case,
    from_ebnf: Callable,
    json_schema_to_ebnf: Callable,
    repeats: int,
    warmups: int,
    strict_mode: bool,
) -> Result:
    for _ in range(warmups):
        from_ebnf(json_schema_to_ebnf(case.schema, strict_mode=strict_mode))
    conversion_samples = []
    parser_samples = []
    total_samples = []
    for _ in range(repeats):
        gc.collect()
        total_start = time.perf_counter()
        conversion_start = time.perf_counter()
        ebnf = json_schema_to_ebnf(case.schema, strict_mode=strict_mode)
        parser_start = time.perf_counter()
        grammar = from_ebnf(ebnf)
        end = time.perf_counter()
        conversion_samples.append(parser_start - conversion_start)
        parser_samples.append(end - parser_start)
        total_samples.append(end - total_start)
        del grammar
    return Result(
        case.name,
        len(case.schema.encode()),
        "legacy",
        median_ms(conversion_samples),
        median_ms(parser_samples),
        median_ms(total_samples),
    )


def print_results(results: list[Result]) -> None:
    print("case\tschema_bytes\tmode\tconversion_ms\tparser_ms\ttotal_ms")
    for result in results:
        print(
            f"{result.name}\t{result.schema_bytes}\t{result.mode}\t"
            f"{result.conversion_ms:.3f}\t{result.parser_ms:.3f}\t{result.total_ms:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "legacy"), required=True)
    parser.add_argument(
        "--xgrammar-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing the binding to benchmark",
    )
    parser.add_argument("--schema", action="append", default=[], help="Additional schema file")
    parser.add_argument(
        "--jsonschemabench-root",
        type=Path,
        help="JSONSchemaBench checkout; four representative large schemas are selected",
    )
    parser.add_argument("--synthetic-wide", type=int, action="append", default=[])
    parser.add_argument("--synthetic-refs", type=int, action="append", default=[])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--strict-mode", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.warmups < 0:
        raise ValueError("--repeats must be positive and --warmups must be non-negative")
    xgr, json_schema_to_ebnf = bootstrap_xgrammar(args.xgrammar_root.resolve())
    cases = load_cases(args)
    if args.mode == "direct":
        results = [
            benchmark_direct(
                case, xgr.Grammar.from_json_schema, args.repeats, args.warmups, args.strict_mode
            )
            for case in cases
        ]
    else:
        results = [
            benchmark_legacy(
                case,
                xgr.Grammar.from_ebnf,
                json_schema_to_ebnf,
                args.repeats,
                args.warmups,
                args.strict_mode,
            )
            for case in cases
        ]
    print_results(results)


if __name__ == "__main__":
    main()
