"""This script benchmarks the time for grammar compilation and mask generation using XGrammar."""

import argparse
import json
import time
from typing import Any, Dict, List, Tuple

import datasets
import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import xgrammar as xgr

wrong_data_indices = [1]


def xgrammar_build(schema: str, grammar_compiler: xgr.GrammarCompiler):
    grammar = grammar_compiler.compile_json_schema(schema)
    matcher = xgr.GrammarMatcher(grammar)
    return matcher


def download_gorilla_file(filename: str) -> Tuple[List, List]:
    base_url = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/data"
    function_url = f"{base_url}/{filename}"
    answer_url = f"{base_url}/possible_answer/{filename}"

    print(f"Downloading {filename} from GitHub...")

    try:
        function_response = requests.get(function_url)
        function_response.raise_for_status()
        function_text = function_response.text

        functions_data = []
        for line in function_text.strip().split("\n"):
            if line.strip():
                try:
                    functions_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing function line in {filename}: {e}")

        answer_response = requests.get(answer_url)
        answer_response.raise_for_status()
        answer_text = answer_response.text

        answers_data = []
        for line in answer_text.strip().split("\n"):
            if line.strip():
                try:
                    answers_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing answer line in {filename}: {e}")

        print(
            f"Successfully downloaded {filename}: {len(functions_data)} functions, {len(answers_data)} answers"
        )
        return functions_data, answers_data
    except requests.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return [], []


def load_gorilla_data() -> List[Dict[str, Any]]:
    gorilla_data = []

    # excluding live test cases part of BFCL v2/v3
    file_patterns = [
        "BFCL_v3_java.json",
        "BFCL_v3_javascript.json",
        "BFCL_v3_multiple.json",
        "BFCL_v3_parallel.json",
        "BFCL_v3_parallel_multiple.json",
        "BFCL_v3_simple.json",
        "BFCL_v3_sql.json",
    ]

    filtered_count = 0

    for filename in file_patterns:
        functions_data, answers_data = download_gorilla_file(filename)

        if not functions_data or not answers_data:
            print(f"Skipping {filename} - failed to download data")
            continue

        print(f"Processing {filename}...")

        answers_by_id = {item["id"]: item for item in answers_data}

        for item in functions_data:
            item_id = item["id"]

            if item_id not in answers_by_id:
                print(f"Warning: No answer found for item {item_id}")
                continue

            if "function" not in item or not item["function"]:
                print(f"Warning: No function definition for item {item_id}")
                filtered_count += 1
                continue

            if len(item["function"]) > 1:
                # print(f"Skipping item {item_id} - contains multiple functions ({len(item['function'])})")
                filtered_count += 1
                continue

            function_def = item["function"][0]  # Use the first function

            schema = convert_function_to_schema(function_def)

            answer = answers_by_id[item_id]
            if "ground_truth" not in answer or not answer["ground_truth"]:
                print(f"Warning: No ground truth for item {item_id}")
                filtered_count += 1
                continue

            ground_truth = answer["ground_truth"][0]  # Use the first ground truth

            completion = convert_ground_truth_to_completion(ground_truth)

            gorilla_data.append(
                {"schema": schema, "completion": completion, "id": item_id, "source": filename}
            )

    print(
        f"Loaded {len(gorilla_data)} examples from Gorilla BFCL dataset (filtered out {filtered_count} examples)"
    )
    return gorilla_data


def convert_function_to_schema(function_def: Dict) -> str:
    """Convert a Gorilla function definition to a JSON schema string with improved type handling."""
    function_name = function_def["name"]
    parameters = function_def["parameters"]

    schema = {
        "type": "object",
        "properties": {function_name: {"type": "object", "properties": {}, "required": []}},
        "required": [function_name],
    }

    for key, value in parameters.get("properties", {}).items():
        param_type = value.get("type", "string").lower()

        if param_type == "integer":
            schema_def = {"type": "integer"}
        elif param_type in ("float", "number", "double"):
            schema_def = {"type": "number"}
        elif param_type == "boolean":
            schema_def = {"type": "boolean"}
        elif param_type in ("hashmap", "map", "dict", "dictionary"):
            schema_def = {"type": "object", "additionalProperties": True}
        elif param_type in ("array", "list"):
            schema_def = {"type": "array", "items": {"type": "string"}}
        elif param_type == "any":
            schema_def = {}  # No type restriction
        else:
            schema_def = {"type": "string"}

        schema["properties"][function_name]["properties"][key] = schema_def

    required_fields = parameters.get("required", [])
    if required_fields:
        schema["properties"][function_name]["required"] = required_fields

    return json.dumps(schema)


def convert_ground_truth_to_completion(ground_truth: Dict) -> str:
    """Convert a Gorilla ground truth to a completion string with improved handling of nested structures."""
    function_name = list(ground_truth.keys())[0]
    params = ground_truth[function_name]

    transformed_params = {}
    for key, values in params.items():
        if isinstance(values, list) and len(values) == 1 and isinstance(values[0], dict):
            nested_obj = {}

            for nested_key, nested_values in values[0].items():
                if isinstance(nested_values, list) and nested_values:
                    nested_obj[nested_key] = nested_values[0]
                else:
                    nested_obj[nested_key] = nested_values

            transformed_params[key] = nested_obj
        elif isinstance(values, list) and values:
            transformed_params[key] = values[0]
        else:
            transformed_params[key] = None

    completion = {function_name: transformed_params}

    return json.dumps(completion)


def run_benchmark(
    dataset_name: str, dataset_data, tokenizer_info, hf_tokenizer, num_iters, num_warmup
):
    vocab_size = len(hf_tokenizer)

    build_time = 0
    exec_time = 0
    total_data_points = 0
    total_tokens = 0
    fail_cnt = 0
    schema_mismatch_cnt = 0

    tqdm_iter = tqdm(range(-num_warmup, num_iters), disable=True)
    for iter in tqdm_iter:
        if iter < 0:
            tqdm_iter.set_description(f"{dataset_name} Warmup Iter: {iter + num_warmup}")
        else:
            tqdm_iter.set_description(f"{dataset_name} Iter: {iter}")

        if iter == 0:
            build_time = 0
            exec_time = 0

        tqdm_data_point_iter = tqdm(range(len(dataset_data)), disable=True)
        for data_point_idx in tqdm_data_point_iter:
            tqdm_data_point_iter.set_description(f"{dataset_name} Data Point: {data_point_idx}")

            if dataset_name == "json-mode-eval" and data_point_idx in wrong_data_indices:
                continue

            schema = dataset_data[data_point_idx]["schema"]
            completion = dataset_data[data_point_idx]["completion"]

            if dataset_name == "gorilla-bfcl":
                try:
                    schema_obj = json.loads(schema)
                    completion_obj = (
                        json.loads(completion) if isinstance(completion, str) else completion
                    )

                    schema_function_name = schema_obj.get("required", [""])[0]

                    completion_function_name = (
                        list(completion_obj.keys())[0] if completion_obj else ""
                    )

                    if (
                        schema_function_name
                        and completion_function_name
                        and schema_function_name != completion_function_name
                    ):
                        if iter >= 0:
                            schema_mismatch_cnt += 1
                            if iter == 0:
                                print(
                                    f"Schema-completion function name mismatch for data point {data_point_idx}:"
                                )
                                print(f"  Schema expects: {schema_function_name}")
                                print(f"  Completion has: {completion_function_name}")
                        continue
                except Exception as e:
                    # If there's an issue parsing the JSON, proceed anyway
                    pass

            if isinstance(completion, dict):
                completion = json.dumps(completion)

            token_ids = hf_tokenizer.encode(completion, add_special_tokens=False)
            grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

            start = time.perf_counter()
            try:
                worker = xgrammar_build(schema, grammar_compiler)
                bitmask = xgr.allocate_token_bitmask(1, vocab_size)
            except Exception as e:
                if iter >= 0:
                    fail_cnt += 1
                    if iter == 0:
                        print(f"Failed to build grammar for data point {data_point_idx}: {e}")
                continue

            build_time += time.perf_counter() - start

            # Use different logits for each mask generation process
            # to avoid caching effects between different tokens
            logits = [torch.randn(vocab_size).cuda() for _ in range(len(token_ids))]

            torch.cuda.synchronize()
            start = time.perf_counter()
            fail_flag = False
            token_rejection_count = 0  # give some leniency, can remove
            for idx, token_id in enumerate(token_ids):
                try:
                    worker.fill_next_token_bitmask(bitmask)

                    cuda_bitmask = bitmask.cuda()

                    xgr.apply_token_bitmask_inplace(logits[idx], cuda_bitmask)

                    # Update state
                    if not worker.accept_token(token_id):
                        token_rejection_count += 1
                        if token_rejection_count > 5:
                            fail_flag = True
                            break
                except Exception as e:
                    if iter >= 0:
                        if iter == 0:  # Only print once to avoid spam
                            print(
                                f"Failed to process token {idx} for data point {data_point_idx}: {e}"
                            )
                    fail_flag = True
                    break

            if fail_flag:
                if iter >= 0:
                    fail_cnt += 1
                continue

            torch.cuda.synchronize()
            exec_time += time.perf_counter() - start

            if iter >= 0:
                total_data_points += 1
                total_tokens += len(token_ids)

    results = {
        "dataset": dataset_name,
        "successful_data_points": total_data_points / num_iters if num_iters > 0 else 0,
        "failed_data_points": fail_cnt / num_iters if num_iters > 0 else 0,
        "schema_mismatch_count": schema_mismatch_cnt / num_iters if num_iters > 0 else 0,
        "total_possible_data_points": len(dataset_data)
        - (len(wrong_data_indices) if dataset_name == "json-mode-eval" else 0),
        "grammar_compilation_time_ms": (
            build_time / total_data_points * 1e3 if total_data_points > 0 else float("inf")
        ),
        "per_token_overhead_us_per_token": (
            exec_time / total_tokens * 1e6 if total_tokens > 0 else float("inf")
        ),
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=5)
    parser.add_argument("--num_warmup", type=int, default=-1)
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Datasets to benchmark: json-mode-eval, gorilla, or all",
    )
    args = parser.parse_args()

    num_iters = args.num_iters
    num_warmup = args.num_warmup if args.num_warmup != -1 else 5 if num_iters >= 40 else 1
    selected_datasets = args.datasets.lower()

    hf_model_path = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading tokenizer from {hf_model_path}...")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    xgrammar_tokenizer_info = xgr.TokenizerInfo.from_huggingface(hf_tokenizer)

    # Try to get GPU info
    try:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "No GPU"
        print(f"Running benchmark with: {device_name} (Device count: {device_count})")
    except:
        print("Could not detect GPU information")

    results = []

    if selected_datasets in ["json-mode-eval", "all"]:
        print("Loading json-mode-eval dataset...")
        json_mode_eval_dataset = datasets.load_dataset("NousResearch/json-mode-eval", split="train")
        json_mode_eval_data = [
            {"schema": item["schema"], "completion": item["completion"]}
            for item in json_mode_eval_dataset
        ]

        print(f"Running benchmark on json-mode-eval ({len(json_mode_eval_data)} examples)...")
        json_mode_eval_results = run_benchmark(
            "json-mode-eval",
            json_mode_eval_data,
            xgrammar_tokenizer_info,
            hf_tokenizer,
            num_iters,
            num_warmup,
        )
        results.append(json_mode_eval_results)

    if selected_datasets in ["gorilla", "all"]:
        print("Loading Gorilla BFCL dataset directly from GitHub...")
        gorilla_data = load_gorilla_data()

        if gorilla_data:
            print(f"Running benchmark on Gorilla BFCL ({len(gorilla_data)} examples)...")
            gorilla_results = run_benchmark(
                "gorilla-bfcl",
                gorilla_data,
                xgrammar_tokenizer_info,
                hf_tokenizer,
                num_iters,
                num_warmup,
            )
            results.append(gorilla_results)
        else:
            print("No Gorilla data loaded, skipping benchmark")

    print("\n===== XGrammar Benchmark Results =====")
    print(f"Model: {hf_model_path}")
    print(f"Iterations: {num_iters}")
    print(f"Warmup Iterations: {num_warmup}")

    for result in results:
        print(f"\nDataset: {result['dataset']}")
        print(
            f"Successful data points: {result['successful_data_points']:.0f} / {result['total_possible_data_points']}"
        )
        print(
            f"Failed data points: {result['failed_data_points']:.0f} / {result['total_possible_data_points']}"
        )
        print(f"Grammar compilation time (ms): {result['grammar_compilation_time_ms']:.4f}")
        print(f"Per token overhead (us/token): {result['per_token_overhead_us_per_token']:.4f}")
