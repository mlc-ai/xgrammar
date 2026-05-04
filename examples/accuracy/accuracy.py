"""Tool-calling accuracy benchmark entrypoint."""

import argparse
import functools
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from api_endpoint import SUPPORTED_BACKENDS, create_api_endpoint
from dataset import SUPPORTED_DATASET, Dataset, create_dataset
from request_processor import MetricAnalyzer, RequestProcessor, create_pipelines
from request_record import (
    RequestRecord,
    convert_reports_to_df,
    generate_metrics_summary,
    pretty_print_report,
)
from transformers import AutoTokenizer  # pylint: disable=import-error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:  # pragma: no cover - fallback for older sglang
    FunctionCallParser = None


def _parse_xml_parameter_value(raw_value: str) -> Any:
    raw_value = raw_value.strip()
    if not raw_value:
        return ""
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _parse_abc_calls_with_qwen_xml(
    stringified_calls: str, tools: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    if FunctionCallParser is not None and tools is not None:
        try:
            parser = FunctionCallParser(tools=tools, tool_call_parser="qwen_xml")
            _, calls = parser.parse_non_stream(stringified_calls)
            parsed = []
            for call in calls:
                name = getattr(call, "name", None)
                if name is None and isinstance(call, dict):
                    name = call.get("name")
                arguments = getattr(call, "parameters", None)
                if arguments is None and isinstance(call, dict):
                    arguments = call.get("parameters") or call.get("arguments")
                if name is None or arguments is None:
                    continue
                parsed.append({"function": {"name": name, "arguments": arguments}})
            if parsed:
                return parsed
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    pattern = re.compile(
        r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>", re.DOTALL
    )
    parameter_pattern = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
    parsed = []
    for match in pattern.finditer(stringified_calls):
        func_name = match.group(1).strip()
        func_body = match.group(2)
        arguments = {}
        for parameter_match in parameter_pattern.finditer(func_body):
            key = parameter_match.group(1).strip()
            value = _parse_xml_parameter_value(parameter_match.group(2))
            arguments[key] = value
        parsed.append({"function": {"name": func_name, "arguments": arguments}})
    return parsed


def _parse_num_concurrent_requests(num_str: Optional[str]) -> Optional[List[int]]:
    if num_str is None:
        return None
    numbers = num_str.split(",")
    if any(not number.isdigit() for number in numbers):
        raise ValueError(f"Unrecognized num_concurrent_requests list: {numbers}")
    return list(int(number) for number in numbers)


def _parse_request_rate(request_rate_str: Optional[str]) -> Optional[List[np.float32]]:
    if request_rate_str is None:
        return None
    request_rates = request_rate_str.split(",")
    results = []
    for rate_str in request_rates:
        request_rate = float(rate_str)
        if request_rate <= 0:
            raise ValueError(f"Invalid request rate {request_rate}")
        results.append(np.float32(request_rate))
    return results


def convert_calls_to_json(
    stringified_calls: str, model: str, tools: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Convert stringified tool calls to a list of dicts."""
    if "Qwen3.6" in model:
        return _parse_abc_calls_with_qwen_xml(stringified_calls, tools)

    function_calls_json = []
    if "Llama-3" in model:
        start = 0
        while True:
            index = stringified_calls.find('{"name":', start)
            if index == -1:
                break
            try:
                decoder = json.JSONDecoder()
                result, end_index = decoder.raw_decode(stringified_calls, index)
            except:  # pylint: disable=bare-except
                start = index + 1
                continue
            start = end_index
            if not isinstance(result, dict) or "name" not in result or "parameters" not in result:
                continue
            function_calls_json.append(
                {"function": {"name": result["name"], "arguments": result["parameters"]}}
            )
    elif "Qwen2" in model:
        start = 0
        while True:
            index = stringified_calls.find('<tool_call>\n{"name":', start)
            if index == -1:
                break
            try:
                decoder = json.JSONDecoder()
                result, end_index = decoder.raw_decode(
                    stringified_calls, index + len("<tool_call>\n")
                )
            except:  # pylint: disable=bare-except
                start = index + 1
                continue
            start = end_index
            if not isinstance(result, dict) or "name" not in result or "arguments" not in result:
                continue
            function_calls_json.append(
                {"function": {"name": result["name"], "arguments": result["arguments"]}}
            )
    return function_calls_json


def run_pipeline(
    pipeline: RequestProcessor, dataset: Dataset, tokenizer: AutoTokenizer, args: argparse.Namespace
) -> Tuple[Dict[str, Any], List[RequestRecord]]:
    """Run the pipeline with the given dataset and args. Return the benchmark report dict."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    request_records = dataset.generate_request_records(
        args.input_len, args.output_len, args.input_len_std, args.output_len_std
    )
    request_records = pipeline(request_records)
    num_total_requests = (
        args.num_requests if not args.per_gpu_workload else args.num_requests * args.num_gpus
    )
    assert len(request_records) == num_total_requests
    sorted_requests: List[RequestRecord] = [None] * num_total_requests
    for request_record in request_records:
        assert request_record.request_id is not None
        assert sorted_requests[request_record.request_id] is None
        sorted_requests[request_record.request_id] = request_record

    request_records = MetricAnalyzer(tokenizer)(request_records)
    report = generate_metrics_summary(request_records, num_total_requests, args.num_gpus)
    return report, sorted_requests


def main(args: argparse.Namespace):
    """Main benchmark entrance."""
    if args.num_requests <= 0:
        raise ValueError("Number of requests to benchmark must be positive.")

    def _main():
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        dataset = create_dataset(args, tokenizer)
        dataset.require_fake_warmup = True
        f_create_api_endpoint = functools.partial(create_api_endpoint, args)
        pipelines = create_pipelines(args, f_create_api_endpoint, dataset)
        store_record = []
        model_part_name = args.model.split("/")[-1]
        output_dir = f"{args.output}/{model_part_name}/{args.dataset}/{'use_stag' if args.use_stag else 'no_stag'}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, pipeline in enumerate(pipelines):
            report, request_records = run_pipeline(pipeline, dataset, tokenizer, args)
            for request in request_records:
                store_record.append({"id": request.request_id})
                store_record[-1]["output"] = request.output_str
                store_record[-1]["call"] = convert_calls_to_json(
                    request.output_str, args.model, dataset.gorilla_data[request.request_id]["tool"]
                )
            with open(f"{output_dir}/result.json", "w") as f:
                json.dump(store_record, f, indent=4)

    _main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SGLang tool-calling accuracy benchmark")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASET,
        help=f"The benchmark dataset kind. Supporting {SUPPORTED_DATASET}",
    )
    parser.add_argument("--dataset-path", type=str, help="The dataset file path.")
    parser.add_argument(
        "--api-endpoint",
        type=str,
        choices=SUPPORTED_BACKENDS,
        default="sglang",
        help="The API endpoint API for benchmarking.",
    )
    parser.add_argument("--model", type=str, required=True, help="The name of the model.")
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="The path of the tokenizer directory."
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="The number of GPUs used by the server. "
        "We need this to better analyze the throughput per GPU.",
    )
    parser.add_argument(
        "--num-requests", type=int, required=True, help="The number of requests for benchmark."
    )
    parser.add_argument(
        "--num-warmup-requests",
        type=int,
        help="The number of requests for warmup. "
        "It is optional when fixing the number of concurrent requests, and is required otherwise.",
    )
    parser.add_argument(
        "--per-gpu-workload",
        default=False,
        action="store_true",
        help='When set to True, the specified "num_concurrent_requests"/"request_rate" '
        "denote the workload **per GPU**, which means that the real values of "
        '"num_concurrent_requests"/"request_rate" used in benchmark'
        'will be multiplied by "num_gpus".',
    )
    parser.add_argument(
        "--num-concurrent-requests",
        type=_parse_num_concurrent_requests,
        help="The number(s) of concurrent requests to benchmark. "
        'It can be either one integer or a list of integer separated by commas(","). '
        "When specified, for each integer, the benchmark keeps these many consistent "
        "number of concurrently running requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=_parse_request_rate,
        help="The request rate(s) denoting the number of new requests each second. "
        'It can be either one float number (or "inf") or a list of numbers separated '
        'by commas(","). '
        "When specified, the benchmark sends these many new requests each second. "
        'If it is "inf", all requests will be sent together at once.',
    )
    parser.add_argument(
        "--replay-timestamp-scale",
        type=float,
        help="The timestamp scale when replaying the timestamps in a dataset. "
        'The dataset replay mode is enabled when neither "--num-concurrent-requests" and '
        '"--request-rate" is specified. '
        "The scale is 1 by default in the replay mode.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        help="The benchmark request average input length. Default to None, "
        "which means the request input length depends on the dataset being used.",
    )
    parser.add_argument(
        "--input-len-std",
        type=float,
        default=0,
        help="The benchmark request input length standard deviation. Default to 0.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        help="The benchmark request average output length. Default to None, "
        "which means the request output length depends on the dataset being used.",
    )
    parser.add_argument(
        "--output-len-std",
        type=float,
        default=0,
        help="The benchmark request output length standard deviation. Default to 0.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Whether to benchmark stream responses. "
        "When not enabled, metrics such as time-to-first-token (TTFT) will not be available. "
        "Default to False.",
    )
    parser.add_argument(
        "--include-server-metrics",
        action="store_true",
        help="Whether to include server-side request metrics when the endpoint provides them.",
    )
    parser.add_argument(
        "--host", type=str, required=True, help="The host address of the backend API."
    )
    parser.add_argument("--port", type=int, required=True, help="The port of the backend API.")
    parser.add_argument(
        "--timeout", type=float, default=3 * 60 * 60, help="The timeout limit of each request."
    )
    parser.add_argument("--seed", type=int, default=0, help="The random number seed. Default to 0.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature value for logit adjustment. Default to 1.",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="The top-p value for sampling. Default to 1."
    )
    parser.add_argument(
        "--ignore-eos",
        default=False,
        action="store_true",
        help='Whether to set the "ignore_eos" field.',
    )
    parser.add_argument(
        "--apply-chat-template",
        default=False,
        action="store_true",
        help="Whether to apply chat template to the request input text. "
        'It is not supported when "--input-len" is specified.',
    )
    parser.add_argument(
        "--num-process-workers",
        type=int,
        help="The number of parallel process workers to send the requests.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Whether to disable showing progress bar with tqdm during benchmarking.",
    )
    parser.add_argument(
        "--max-schedule-gap",
        type=float,
        default=0.5,
        help="The maximum allowed delay between the scheduled time in seconds.",
    )
    parser.add_argument(
        "--cuda-profile",
        default=False,
        action="store_true",
        help="Whether to enable CUDA profiling on the SGLang server debug endpoint.",
    )
    parser.add_argument(
        "--debug-dump",
        default=False,
        action="store_true",
        help="Whether to dump all request record raw data to file.",
    )
    parser.add_argument(
        "--multi-round",
        default=False,
        action="store_true",
        help="Whether to chat like multi round conversion with history log each request. "
        "Only enabled when benchmarked with fixed concurrent request mode."
        "The --num-concurrent-requests should be provided when enabling this option.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="sglang_accuracy",
        help="The path of the output file where to dump the benchmark results.",
    )
    parser.add_argument("--use-stag", action="store_true", help="Whether to set structural tag.")
    parser.add_argument(
        "--use-jf", action="store_true", help="Whether to use jump-forward-decoding."
    )

    main(parser.parse_args())
