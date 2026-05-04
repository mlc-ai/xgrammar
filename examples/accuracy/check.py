"""Tool-calling accuracy result checker."""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

SUPPORTED_DATASET = [
    "BFCL_v3_simple",
    "BFCL_v3_multiple",
    "BFCL_v3_parallel",
    "BFCL_v3_live_simple",
    "BFCL_v3_live_multiple",
    "BFCL_v3_live_parallel",
    "ALL",
]

SUPPORTED_MODEL = [
    "Llama-3.2-1B-Instruct-q0f16-MLC",
    "Llama-3.2-3B-Instruct-q0f16-MLC",
    "Llama-3.1-8B-Instruct-q0f16-MLC",
    "Llama-3.1-70B-Instruct-q0f16-MLC",
    "Qwen2.5-72B-Instruct-q0f16-MLC",
    "Qwen3.6-27B",
    "Qwen3.6-35B-A3B",
    "ALL",
]

from enum import IntEnum

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:  # pragma: no cover - fallback for older sglang
    FunctionCallParser = None


class Err_type(IntEnum):
    CALL_NUMBER_ERROR = 0
    FUNC_SELECT_ERROR = 1
    FUNC_NAME_ERROR = 2
    PARA_KEY_ERROR = 3
    TYPE_ERROR = 4
    ENUM_ERROR = 5
    PARA_VALUE_ERROR = 6
    NONE = 7


class Error:
    def __init__(self, message: str = "", err_type: Err_type = Err_type.NONE):
        self.message = message
        self.error_type = err_type


def _parse_xml_parameter_value(raw_value: str) -> Any:
    raw_value = raw_value.strip()
    if not raw_value:
        return ""
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _parse_Qwen_calls_with_qwen_xml(
    output: str, tools: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    if FunctionCallParser is not None and tools is not None:
        try:
            parser = FunctionCallParser(tools=tools, tool_call_parser="qwen_xml")
            _, calls = parser.parse_non_stream(output)
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
    for match in pattern.finditer(output):
        func_name = match.group(1).strip()
        func_body = match.group(2)
        arguments = {}
        for parameter_match in parameter_pattern.finditer(func_body):
            key = parameter_match.group(1).strip()
            value = _parse_xml_parameter_value(parameter_match.group(2))
            arguments[key] = value
        parsed.append({"function": {"name": func_name, "arguments": arguments}})
    return parsed


def _parse_calls_for_model(
    model: str, output: str, tools: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    if "Qwen3.6" in model:
        return _parse_Qwen_calls_with_qwen_xml(output, tools)

    parsed_calls = []
    start = 0
    while True:
        index = output.find('{"name":', start)
        if index == -1:
            break
        try:
            decoder = json.JSONDecoder()
            result, end_index = decoder.raw_decode(output, index)
        except json.JSONDecodeError:
            start = index + 1
            continue
        start = end_index + 1
        if "Llama-3" in model:
            if "name" not in result or "parameters" not in result:
                continue
            parsed_calls.append(
                {"function": {"name": result["name"], "arguments": result["parameters"]}}
            )
        elif "Qwen2" in model:
            if "name" not in result or "arguments" not in result:
                continue
            parsed_calls.append(
                {"function": {"name": result["name"], "arguments": result["arguments"]}}
            )
    return parsed_calls


def valid_data_point(tools: List[Dict], expected: List[Dict]) -> bool:
    # check expected call-function name
    valid_func_name = set()
    for tool in tools:
        valid_func_name.add(tool["function"]["name"])
    for call in expected:
        if call["name"] not in valid_func_name:
            return False
    # check the enum schema
    results = []

    def _find(obj, target_type: str):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    if "type" in value and value["type"] == target_type:
                        results.append(value)
                if isinstance(value, (dict, list)):
                    _find(value, target_type)
        elif isinstance(obj, list):
            for item in obj:
                _find(item, target_type)

    for tool in tools:
        results = []
        _find(tool, "array")
        for item in results:
            if "enum" in item:
                for entry in item["enum"]:
                    if not isinstance(entry, list):
                        return False
    for tool in tools:
        results = []
        _find(tool, "integer")
        for item in results:
            if "enum" in item:
                for entry in item["enum"]:
                    if not isinstance(entry, int):
                        return False
    return True


# Modified by https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/eval_checker/ast_eval/ast_checker.py
def check_simple(
    gorilla, tool_call: Dict[str, Any], tool: Dict[str, Any], ideal: Dict[str, Any]
) -> Tuple[bool, Error]:
    # check func name
    if ideal["name"] != tool_call["function"]["name"]:
        return False, Error("wrong function name.", Err_type.FUNC_NAME_ERROR)
    func = tool["function"]
    # check func args
    if not isinstance(tool_call["function"]["arguments"], dict):
        return False, Error("wrong format", Err_type.PARA_KEY_ERROR)

    for arg in func["parameters"]["required"]:
        if arg not in tool_call["function"]["arguments"]:
            return False, Error(f"missing arg: {arg}", Err_type.PARA_KEY_ERROR)

        for arg in tool_call["function"]["arguments"].keys():
            ideal_arg: List = ideal["arguments"][arg] if arg in ideal["arguments"] else None
            real_arg = tool_call["function"]["arguments"][arg]
            if arg not in func["parameters"]["properties"]:
                return False, Error(f"unknown arg: {arg}", Err_type.PARA_KEY_ERROR)
            info_arg = func["parameters"]["properties"][arg]
            if info_arg["type"] == "integer":
                acc, err = check_integer(gorilla, real_arg, ideal_arg)
                if not acc:
                    return False, err
            elif info_arg["type"] == "number":
                acc, err = check_number(gorilla, real_arg, ideal_arg)
                if not acc:
                    return False, err
            elif info_arg["type"] == "boolean":
                acc, err = check_boolean(gorilla, real_arg, ideal_arg)
                if not acc:
                    return False, err
            elif info_arg["type"] == "string":
                # XML tool calls parse parameter values with json.loads; string slots may
                # arrive as non-str, and will be converted to str automatically.
                pass
            elif info_arg["type"] == "array":
                acc, err = check_list(gorilla, real_arg, ideal_arg, info_arg["items"])
                if not acc:
                    return False, err
            elif info_arg["type"] == "dict":
                acc, err = check_dict(gorilla, real_arg, ideal_arg, info_arg["properties"])
                if not acc:
                    return False, err
    return True, Error()


def check_simple_schema(
    gorilla, tool_call: Dict[str, Any], tool: Dict[str, Any]
) -> Tuple[bool, Error]:
    # check func name
    func = tool["function"]
    if func["name"] != tool_call["function"]["name"]:
        return False, Error("wrong function name.", Err_type.FUNC_NAME_ERROR)
    # check func args
    if not isinstance(tool_call["function"]["arguments"], dict):
        return False, Error("wrong format", Err_type.PARA_KEY_ERROR)
    for arg in func["parameters"]["required"]:
        if arg not in tool_call["function"]["arguments"]:
            return False, Error(f"missing arg: {arg}", Err_type.PARA_KEY_ERROR)
    for arg in tool_call["function"]["arguments"].keys():
        real_arg = tool_call["function"]["arguments"][arg]
        if arg not in func["parameters"]["properties"]:
            return False, Error(f"unknown arg: {arg}", Err_type.PARA_KEY_ERROR)
        info_arg = func["parameters"]["properties"][arg]
        if info_arg["type"] == "integer":
            acc, err = check_integer(gorilla, real_arg, None)
            if not acc:
                return False, err
        elif info_arg["type"] == "number":
            acc, err = check_number(gorilla, real_arg, None)
            if not acc:
                return False, err
        elif info_arg["type"] == "boolean":
            acc, err = check_boolean(gorilla, real_arg, None)
            if not acc:
                return False, err
        elif info_arg["type"] == "string":
            pass
        elif info_arg["type"] == "array":
            acc, err = check_list(gorilla, real_arg, None, info_arg["items"])
            if not acc:
                return False, err
        elif info_arg["type"] == "dict":
            acc, err = check_dict(gorilla, real_arg, None, info_arg["properties"])
            if not acc:
                return False, err
    return True, Error()


def check_integer(gorilla, real_arg: Any, ideal_arg: Optional[List[Any]]) -> Tuple[bool, Error]:
    if type(real_arg) != int:
        return False, Error(f"wrong type {real_arg}: not int", Err_type.TYPE_ERROR)
    if ideal_arg is None:
        return True, Error()
    match = False
    err = Error(f"value not match: {real_arg}, ideal-opt: {ideal_arg}", Err_type.PARA_VALUE_ERROR)
    for ideal in ideal_arg:
        if real_arg == ideal:
            match = True
            err = Error()
            break
    return match, err


def check_number(gorilla, real_arg: Any, ideal_arg: Optional[List[Any]]) -> Tuple[bool, Error]:
    if type(real_arg) != float and type(real_arg) != int:
        return False, Error(f"wrong type {real_arg}: not number", Err_type.TYPE_ERROR)
    if ideal_arg is None:
        return True, Error()
    match = False
    err = Error(f"value not match: {real_arg}, ideal-opt: {ideal_arg}", Err_type.PARA_VALUE_ERROR)
    for ideal in ideal_arg:
        if real_arg == ideal:
            match = True
            err = Error()
            break
    return match, err


def check_string(
    gorilla, real_arg: Any, ideal_arg: Optional[List[Any]], enum: Optional[List[str]]
) -> Tuple[bool, Error]:
    def standardize_string(string: Any) -> str:
        if not isinstance(string, str):
            return "-----Error------"
        regex_string = r"[ \,\.\/\-\_\*\^]"
        return re.sub(regex_string, "", string).lower().replace("'", '"')

    if type(real_arg) != str:
        return False, Error(f"wrong type {real_arg}: not string", Err_type.TYPE_ERROR)
    match = False
    err = Error(f"value not match: {real_arg}, ideal-opt: {ideal_arg}", Err_type.PARA_VALUE_ERROR)
    real_arg = standardize_string(real_arg)
    if ideal_arg is None:
        if enum is None:
            return True, Error()
        else:
            err.error_type = Err_type.ENUM_ERROR
            for ideal in enum:
                if real_arg == standardize_string(ideal):
                    match = True
                    err = Error()
                    break
    else:
        for ideal in ideal_arg:
            if real_arg == standardize_string(ideal):
                match = True
                err = Error()
                break
    return match, err


def check_boolean(gorilla, real_arg: bool, ideal_arg: Optional[List[bool]]) -> Tuple[bool, Error]:
    if type(real_arg) != bool:
        return False, Error(f"wrong type {real_arg}: not bool", Err_type.TYPE_ERROR)
    if ideal_arg is None:
        return True, Error()
    match = False
    err = Error(f"value not match: {real_arg}, ideal-opt: {ideal_arg}", Err_type.PARA_VALUE_ERROR)
    for ideal in ideal_arg:
        if real_arg == ideal:
            match = True
            err = Error()
            break
    return match, err


def check_list(
    gorilla, real_arg: List, ideal_arg: Optional[List[List]], item: Dict[str, Any]
) -> Tuple[bool, Error]:
    if type(real_arg) != list:
        return False, Error(f"wrong type of {real_arg}: not list.", Err_type.TYPE_ERROR)
    item_type = item["type"]
    if ideal_arg is None:
        if item_type == "integer":
            for i, integer in enumerate(real_arg):
                acc, err = check_integer(gorilla, integer, None)
                if not acc:
                    return False, err
        elif item_type == "number":
            for i, integer in enumerate(real_arg):
                acc, err = check_number(gorilla, integer, None)
                if not acc:
                    return False, err
        elif item_type == "boolean":
            for i, boolean in enumerate(real_arg):
                acc, err = check_boolean(gorilla, boolean, None)
                if not acc:
                    return False, err
        elif item_type == "string":
            pass
        elif item_type == "array":
            for i, array in enumerate(real_arg):
                acc, err = check_list(gorilla, array, None, item["items"])
                if not acc:
                    return False, err
        elif item_type == "dict":
            for i, dictionary in enumerate(real_arg):
                acc, err = check_dict(gorilla, dictionary, None, item["properties"])
                if not acc:
                    return False, err
        return True, Error()
    else:
        final_err = ""
        err_type = Err_type.NONE
        for j, ideal in enumerate(ideal_arg):
            if len(ideal) != len(real_arg):
                final_err += f"[ideal {j}] wrong length of {real_arg}."
                err_type = min(err_type, Err_type.PARA_VALUE_ERROR)
                continue
            match = True
            if item_type == "integer":
                for i, integer in enumerate(real_arg):
                    acc, err = check_integer(gorilla, integer, [ideal[i]])
                    if not acc:
                        match = False
                        final_err += f"[ideal {j}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            elif item_type == "number":
                for i, integer in enumerate(real_arg):
                    acc, err = check_number(gorilla, integer, [ideal[i]])
                    if not acc:
                        match = False
                        final_err += f"[ideal {j}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            elif item_type == "boolean":
                for i, boolean in enumerate(real_arg):
                    acc, err = check_boolean(gorilla, boolean, [ideal[i]])
                    if not acc:
                        match = False
                        final_err += f"[ideal {j}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            elif item_type == "string":
                pass
            elif item_type == "array":
                for i, array in enumerate(real_arg):
                    acc, err = check_list(gorilla, array, [ideal[i]], item["items"])
                    if not acc:
                        match = False
                        final_err += f"[ideal {j}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            elif item_type == "dict":
                for i, dictionary in enumerate(real_arg):
                    acc, err = check_dict(gorilla, dictionary, [ideal[i]], item["properties"])
                    if not acc:
                        match = False
                        final_err += f"[ideal {j}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            if match:
                return True, Error()
        return err_type == Err_type.NONE, Error(final_err, err_type)


def check_dict(
    gorilla,
    real_arg: Dict[str, Any],
    ideal_arg: Optional[Dict[str, Any]],
    properties: Dict[str, Any],
) -> Tuple[bool, Error]:
    if type(real_arg) != dict:
        return False, Error(f"wrong type of {real_arg}: not dict.", Err_type.TYPE_ERROR)
    if ideal_arg is None:
        for key in properties.keys():
            if key not in real_arg:
                return False, Error(f"missing key: {key}.", Err_type.PARA_KEY_ERROR)
            item_type = properties[key]["type"]
            if item_type == "integer":
                acc, err = check_integer(gorilla, real_arg[key], None)
                if not acc:
                    return False, err
            elif item_type == "number":
                acc, err = check_number(gorilla, real_arg[key], None)
                if not acc:
                    return False, err
            elif item_type == "boolean":
                acc, err = check_boolean(gorilla, real_arg[key], None)
                if not acc:
                    return False, err
            elif item_type == "string":
                pass
            elif item_type == "array":
                acc, err = check_list(gorilla, real_arg[key], None, properties[key]["items"])
                if not acc:
                    return False, err
            elif item_type == "dict":
                acc, err = check_dict(gorilla, real_arg[key], None, properties[key]["properties"])
                if not acc:
                    return False, err
        return True, Error()
    else:
        final_err = ""
        err_type = Err_type.NONE
        for i, ideal in enumerate(ideal_arg):
            match = True
            for key in properties.keys():
                if key not in real_arg:
                    match = False
                    final_err += f"[ideal {i}] missing key: {key}."
                    err_type = min(err_type, Err_type.PARA_KEY_ERROR)
                    break
                item_type = properties[key]["type"]
                if item_type == "integer":
                    acc, err = check_integer(gorilla, real_arg[key], [ideal[key]])
                    if not acc:
                        match = False
                        final_err += f"[ideal {i}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
                elif item_type == "number":
                    acc, err = check_number(gorilla, real_arg[key], [ideal[key]])
                    if not acc:
                        match = False
                        final_err += f"[ideal {i}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
                elif item_type == "boolean":
                    acc, err = check_boolean(gorilla, real_arg[key], [ideal[key]])
                    if not acc:
                        match = False
                        final_err += f"[ideal {i}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
                elif item_type == "string":
                    pass
                elif item_type == "array":
                    acc, err = check_list(
                        gorilla, real_arg[key], [ideal[key]], properties[key]["items"]
                    )
                    if not acc:
                        match = False
                        final_err += f"[ideal {i}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
                elif item_type == "dict":
                    acc, err = check_dict(
                        gorilla, real_arg[key], [ideal[key]], properties[key]["properties"]
                    )
                    if not acc:
                        match = False
                        final_err += f"[ideal {i}] {err}"
                        err_type = min(err_type, err.error_type)
                        break
            if match:
                return True, Error()
        return err_type == Err_type.NONE, Error(final_err, err_type)


def check_acc(
    model: str, dataset: str, gorilla: Dict, summary: Dict, totol_summary: Dict, use_stag: bool
):
    """Check the accuracy of the generated requests."""
    if model not in totol_summary:
        totol_summary[model] = {}
    if dataset not in totol_summary[model]:
        totol_summary[model][dataset] = {"use_stag": {}, "no_stag": {}}
    err_types = [0] * (len(Err_type) - 1)
    stag_cate = "use_stag" if use_stag else "no_stag"
    valid_data_point = 0
    if dataset == "BFCL_v3_simple" or dataset == "BFCL_v3_live_simple":
        for item in summary:
            if not item["valid_datapoint"]:
                continue
            else:
                valid_data_point += 1
            id = item["id"]
            info = gorilla[id]
            if len(item[stag_cate]["call"]) == 0:
                summary[id][stag_cate]["success"] = False
                summary[id][stag_cate]["err_type"] = "CALL_NUMBER_ERROR"
                summary[id][stag_cate]["err_msg"] = "missing calling."
                err_types[Err_type.CALL_NUMBER_ERROR] += 1
                continue
            if len(item[stag_cate]["call"]) != 1:
                acc, err = (False, Error("wrong calling numbers.", Err_type.CALL_NUMBER_ERROR))
            else:
                acc, err = check_simple(
                    gorilla, item[stag_cate]["call"][0], info["tool"][0], info["ideal_call"][0]
                )
            if not acc:
                summary[id][stag_cate]["success"] = False
                summary[id][stag_cate]["err_type"] = Err_type(err.error_type).name
                summary[id][stag_cate]["err_msg"] = err.message
                err_types[err.error_type] += 1
            else:
                summary[id][stag_cate]["success"] = True
                summary[id][stag_cate]["err_type"] = None
                summary[id][stag_cate]["err_msg"] = None
    elif dataset == "BFCL_v3_multiple" or dataset == "BFCL_v3_live_multiple":
        for item in summary:
            if not item["valid_datapoint"]:
                continue
            else:
                valid_data_point += 1
            id = item["id"]
            info = gorilla[id]
            if len(item[stag_cate]["call"]) == 0:
                summary[id][stag_cate]["success"] = False
                summary[id][stag_cate]["err_type"] = "CALL_NUMBER_ERROR"
                summary[id][stag_cate]["err_msg"] = "missing calling."
                err_types[Err_type.CALL_NUMBER_ERROR] += 1
                continue
            if len(item[stag_cate]["call"]) != 1:
                acc, err = (False, Error("wrong calling numbers.", Err_type.CALL_NUMBER_ERROR))
            else:
                expected_tool = None
                match_tool = None
                for tool in info["tool"]:
                    if tool["function"]["name"] == info["ideal_call"][0]["name"]:
                        expected_tool = tool
                    if item[stag_cate]["call"][0]["function"]["name"] == tool["function"]["name"]:
                        match_tool = tool
                if match_tool != None and expected_tool != match_tool:
                    acc, err = (
                        False,
                        Error("wrong function selection.", Err_type.FUNC_SELECT_ERROR),
                    )
                else:
                    acc, err = check_simple(
                        gorilla, item[stag_cate]["call"][0], expected_tool, info["ideal_call"][0]
                    )
            if not acc:
                summary[id][stag_cate]["success"] = False
                summary[id][stag_cate]["err_type"] = Err_type(err.error_type).name
                summary[id][stag_cate]["err_msg"] = err.message
                err_types[err.error_type] += 1
            else:
                summary[id][stag_cate]["success"] = True
                summary[id][stag_cate]["err_type"] = None
                summary[id][stag_cate]["err_msg"] = None
    elif dataset == "BFCL_v3_parallel" or dataset == "BFCL_v3_live_parallel":
        for item in summary:
            if not item["valid_datapoint"]:
                continue
            else:
                valid_data_point += 1
            id = item["id"]
            info = gorilla[id]
            if len(item[stag_cate]["call"]) == 0:
                summary[id][stag_cate]["success"] = False
                summary[id][stag_cate]["err_type"] = "CALL_NUMBER_ERROR"
                summary[id][stag_cate]["err_msg"] = "missing calling."
                err_types[Err_type.CALL_NUMBER_ERROR] += 1
                continue
            if len(item[stag_cate]["call"]) != len(info["ideal_call"]):
                # print("__________________________________")
                # print(item[stag_cate]["call"])
                # print(info["ideal_call"][0])
                acc, err = (False, Error("wrong calling numbers.", Err_type.CALL_NUMBER_ERROR))
            else:
                matched = set()
                for ideal in info["ideal_call"]:
                    expected_tool = None
                    for tool in info["tool"]:
                        if tool["function"]["name"] == ideal["name"]:
                            expected_tool = tool
                            break

                    err = Error("", err_type=Err_type.CALL_NUMBER_ERROR)
                    acc = False
                    for index, singal_call in enumerate(item[stag_cate]["call"]):
                        if index in matched:
                            continue
                        tmp_acc, tmp_err = check_simple(gorilla, singal_call, expected_tool, ideal)
                        if tmp_acc:
                            acc = True
                        if err.error_type < tmp_err.error_type:
                            err = tmp_err
                        if tmp_acc:
                            matched.add(index)
                            break
                    if not acc:
                        break
            if not acc:
                summary[id][stag_cate]["success"] = False
                summary[id][stag_cate]["err_type"] = Err_type(err.error_type).name
                summary[id][stag_cate]["err_msg"] = err.message
                err_types[err.error_type] += 1
            else:
                summary[id][stag_cate]["success"] = True
                summary[id][stag_cate]["err_type"] = None
                summary[id][stag_cate]["err_msg"] = None
    total_acc = 1
    for i in range(len(Err_type) - 1):
        totol_summary[model][dataset][stag_cate][Err_type(i).name] = err_types[i] / valid_data_point
        total_acc -= err_types[i] / valid_data_point
    totol_summary[model][dataset][stag_cate]["CORRECT_CALL"] = total_acc


def get_correct_schema_rate(
    model: str, dataset: str, gorilla: Dict, summary: Dict, totol_summary: Dict, use_stag: bool
) -> float:
    """Get the correct schema rate of the generated requests."""
    stag_cate = "use_stag" if use_stag else "no_stag"
    call_number = 0
    correct_schema_number = 0
    for entry in summary:
        if not entry["valid_datapoint"]:
            continue
        output = entry[stag_cate]["output"]
        parsed_calls = _parse_calls_for_model(model, output, entry["tools"])
        for call in parsed_calls:
            call_number += 1
            err_list = []
            for tool in entry["tools"]:
                acc, err = check_simple_schema(gorilla, call, tool)
                err_list.append(err)
                if acc or err.error_type == Err_type.PARA_VALUE_ERROR:
                    correct_schema_number += 1
                    break
    if "correct_schema_rate" not in totol_summary[model][dataset]:
        totol_summary[model][dataset]["correct_schema_rate"] = {}
    totol_summary[model][dataset]["correct_schema_rate"][stag_cate] = (
        correct_schema_number / call_number
    )


def main(args: argparse.Namespace):
    """Main benchmark entrance."""
    models = []
    datasets = []
    if args.dataset == "ALL":
        datasets = SUPPORTED_DATASET
        datasets.pop(-1)
    else:
        datasets.append(args.dataset)
    if args.model == "ALL":
        models = SUPPORTED_MODEL
        models.pop(-1)
    else:
        models.append(args.model)
    total_summary = {}
    for model in models:
        for dataset in datasets:
            result_dir = f"{args.output_root}/{model}/{dataset}"
            if not os.path.exists(f"{result_dir}/use_stag/result.json") or not os.path.exists(
                f"{result_dir}/no_stag/result.json"
            ):
                continue
            with open(f"{result_dir}/use_stag/result.json", mode="r", encoding="utf-8") as file:
                use_stag_result = json.load(file)
            with open(f"{result_dir}/no_stag/result.json", mode="r", encoding="utf-8") as file:
                no_stag_result = json.load(file)
            with open(f"{args.dataset_path}/{dataset}.json", mode="r", encoding="utf-8") as file:
                gorilla = json.load(file)
            print(f"Begin checking {model} on {dataset}...")
            summary = []
            for i in range(len(use_stag_result)):
                assert i == use_stag_result[i]["id"]
                assert i == no_stag_result[i]["id"]
                assert i == gorilla[i]["id"]
                if not valid_data_point(gorilla[i]["tool"], gorilla[i]["ideal_call"]):
                    summary.append(
                        {
                            "id": i,
                            "valid_datapoint": False,
                            "no_stag": None,
                            "use_stag": None,
                            "input": None,
                            "tools": None,
                            "expected": None,
                        }
                    )
                    continue
                summary.append(
                    {
                        "id": i,
                        "valid_datapoint": True,
                        "no_stag": {
                            "output": no_stag_result[i]["output"],
                            "call": (
                                no_stag_result[i]["call"] if ("call" in no_stag_result[i]) else []
                            ),
                        },
                        "use_stag": {
                            "output": use_stag_result[i]["output"],
                            "call": (
                                use_stag_result[i]["call"] if ("call" in use_stag_result[i]) else []
                            ),
                        },
                        "input": gorilla[i]["question"],
                        "tools": gorilla[i]["tool"],
                        "expected": gorilla[i]["ideal_call"],
                    }
                )
            check_acc(model, dataset, gorilla, summary, total_summary, False)
            check_acc(model, dataset, gorilla, summary, total_summary, True)
            get_correct_schema_rate(model, dataset, gorilla, summary, total_summary, False)
            get_correct_schema_rate(model, dataset, gorilla, summary, total_summary, True)

            if not os.path.exists(f"{args.final_root}/{model}/{dataset}"):
                os.makedirs(f"{args.final_root}/{model}/{dataset}")
            with open(f"{args.final_root}/{model}/{dataset}/summary.json", "w") as f:
                json.dump(summary, f, indent=4)
    if not os.path.exists(args.final_root):
        os.makedirs(args.final_root)
    with open(f"{args.final_root}/summary.json", "w") as f:
        json.dump(total_summary, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tool-calling accuracy result checker")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASET,
        help=f"The benchmark dataset kind. Supporting {SUPPORTED_DATASET}",
    )
    parser.add_argument(
        "--model",
        type=str,
        help=f'The benchmark model kind, or "ALL" (supported defaults: {SUPPORTED_MODEL}).',
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="The dataset file path.")
    parser.add_argument(
        "--output-root", type=str, required=True, help="The root of the raw output file."
    )
    parser.add_argument(
        "--final-root", type=str, required=True, help="The root of the summary file."
    )
    args = parser.parse_args()
    main(args)
