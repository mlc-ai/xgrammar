"""Validate builtin structural tags against official chat templates.

Uses tokenizer.apply_chat_template (or encoding scripts for DeepSeek V3.2/V4)
to render model outputs, then checks that xgrammar structural tag grammars
accept them. Requires encoding_dsv32.py and encoding_dsv4.py in the same
directory.
"""

import json
import os
import sys
from functools import lru_cache

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from xgrammar import Grammar
from xgrammar.builtin_structural_tag import get_model_structural_tag
from xgrammar.testing import _is_grammar_accept_string

TOOL_A = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}
TOOL_B = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time in a timezone.",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        },
    },
}

USER_MSG = {"role": "user", "content": "What is the weather in Beijing?"}
REASONING_CONTENT = "Let me think about this step by step."

TOOL_SCENARIOS = [
    (0, "auto"),
    (1, "auto"),
    (1, "forced"),
    (1, "required"),
    (2, "auto"),
    (2, "required"),
]

# (stag_key, model_id, reasoning, template_kwargs)
# Excluded:
#   - Llama-4: pythonic tool call format, needs separate structural tag
#   - gemma_4: tool calls use <|"|> quoting, not JSON
#   - deepseek_r1 thinking=True: template drops <think> in history rendering,
#     prompt diff extraction doesn't work
#   - Kimi-K2-Thinking thinking=False: model always outputs <think></think>,
#     but grammar excludes these tokens in non-reasoning mode
MODEL_CONFIGS = [
    ("llama", "meta-llama/Llama-3.1-8B-Instruct", False, {}),
    ("kimi", "moonshotai/Kimi-K2-Thinking", True, {"thinking": True}),
    ("kimi", "moonshotai/Kimi-K2-Instruct", False, {}),
    ("deepseek_r1", "deepseek-ai/DeepSeek-R1", True, {}),
    ("deepseek_v3_1", "deepseek-ai/DeepSeek-V3.1", True, {"thinking": True}),
    ("deepseek_v3_1", "deepseek-ai/DeepSeek-V3.1", False, {"thinking": False}),
    ("qwen_3_coder", "Qwen/Qwen3-Coder-30B-A3B-Instruct", False, {}),
    ("qwen_3_coder", "Qwen/Qwen3-Coder-Next", False, {}),
    ("qwen_3_5", "Qwen/Qwen3.5-35B-A3B", True, {"enable_thinking": True}),
    ("qwen_3_5", "Qwen/Qwen3.5-35B-A3B", False, {"enable_thinking": False}),
    ("qwen_3_5", "Qwen/Qwen3.6-35B-A3B", True, {"enable_thinking": True}),
    ("qwen_3_5", "Qwen/Qwen3.6-35B-A3B", False, {"enable_thinking": False}),
    ("qwen_3", "Qwen/Qwen3-4B-Thinking-2507", True, {}),
    ("qwen_3", "Qwen/Qwen3-4B-Instruct-2507", False, {}),
    ("qwen_3", "Qwen/Qwen3-Next-80B-A3B-Thinking", True, {}),
    ("qwen_3", "Qwen/Qwen3-Next-80B-A3B-Instruct", False, {}),
    ("harmony", "openai/gpt-oss-20b", True, {}),
    ("harmony", "openai/gpt-oss-20b", False, {}),
    ("deepseek_v3_2", "ENCODER:dsv32", True, {"thinking_mode": "thinking"}),
    ("deepseek_v3_2", "ENCODER:dsv32", False, {"thinking_mode": "chat"}),
    ("minimax", "MiniMaxAI/MiniMax-M2.5", True, {}),
    ("glm_4_7", "zai-org/GLM-4.7-Flash", True, {"enable_thinking": True}),
    ("glm_4_7", "zai-org/GLM-4.7-Flash", False, {"enable_thinking": False}),
    ("deepseek_v4", "ENCODER:dsv4", True, {"thinking_mode": "thinking"}),
    ("deepseek_v4", "ENCODER:dsv4", False, {"thinking_mode": "chat"}),
]

# DeepSeek V3.2 encoder rejects empty reasoning + no tool calls.
SKIP_EMPTY_REASONING = {"ENCODER:dsv32", "MiniMaxAI/MiniMax-M2.5"}

# Models where tool call format in template doesn't match structural tag.
SKIP_TOOLS = set()

# Models whose template strips or skips <think> for empty reasoning_content,
# requiring "strip base + prepend reasoning_content" to reconstruct model output.
# Includes: R1/V3.1 (content.split('</think>')), GLM/MiniMax/Qwen3.5 (falsy branch).
STRIP_THINK_MODELS = {
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-R1",
    "zai-org/GLM-4.7-Flash",
    "MiniMaxAI/MiniMax-M2.5",
    "Qwen/Qwen3.5-35B-A3B",
}

EOS_SUFFIXES = {
    "llama": ["<|eot_id|>"],
    "kimi": ["<|im_end|>"],
    "deepseek_r1": ["<｜end▁of▁sentence｜>"],
    "deepseek_v3_1": ["<｜end▁of▁sentence｜>"],
    "qwen_3": ["<|im_end|>"],
    "qwen_3_5": ["<|im_end|>"],
    "qwen_3_coder": ["<|im_end|>"],
    "harmony": None,
    "minimax": ["[e~["],
    "glm_4_7": [],
    "deepseek_v3_2": ["<｜end▁of▁sentence｜>"],
    "deepseek_v4": ["<｜end▁of▁sentence｜>"],
}


@lru_cache(maxsize=None)
def load_tokenizer(model_id):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def make_tools(num_tools):
    if num_tools == 0:
        return None
    if num_tools == 1:
        return [TOOL_A]
    return [TOOL_A, TOOL_B]


def make_tool_choice(choice_str, tools):
    if choice_str == "forced":
        return {"type": "function", "function": {"name": tools[0]["function"]["name"]}}
    return choice_str


def make_assistant_msg(stag_key, reasoning_content, num_tool_calls):
    msg = {"role": "assistant"}
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    if num_tool_calls == 0:
        msg["content"] = "The answer is 42."
        return msg

    msg["content"] = ""
    tool_specs = [("get_weather", {"location": "Beijing"}), ("get_time", {"timezone": "UTC"})]
    calls = []
    for i in range(num_tool_calls):
        name, args = tool_specs[i]
        tc = {"type": "function", "function": {"name": name}}
        if stag_key in ("deepseek_r1", "deepseek_v3_1"):
            tc["function"]["arguments"] = json.dumps(args)
        else:
            tc["function"]["arguments"] = args
        if stag_key == "kimi":
            tc["id"] = f"functions.{name}:{i}"
        else:
            tc["id"] = f"call_{i}"
        calls.append(tc)
    msg["tool_calls"] = calls
    return msg


def strip_eos(output, stag_key, tokenizer=None):
    eos_list = EOS_SUFFIXES.get(stag_key)
    if eos_list is None:
        return output
    if not eos_list and tokenizer:
        eos_list = [tokenizer.eos_token] if tokenizer.eos_token else []
    for eos in eos_list:
        if output.endswith("\n" + eos + "\n"):
            output = output[: -(len(eos) + 2)]
            break
        if output.endswith(eos + "\n"):
            output = output[: -(len(eos) + 1)]
            break
        if output.endswith("\n" + eos):
            output = output[: -(len(eos) + 1)]
            break
        if output.endswith(eos):
            output = output[: -len(eos)]
            break
    return output


def extract_output_tokenizer(model_id, stag_key, assistant_msg, tools, template_kwargs):
    tokenizer = load_tokenizer(model_id)
    kwargs = dict(tokenize=False, **template_kwargs)
    if tools:
        kwargs["tools"] = tools
    prompt = tokenizer.apply_chat_template([USER_MSG], add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(
        [USER_MSG, assistant_msg], add_generation_prompt=False, **kwargs
    )

    if model_id in STRIP_THINK_MODELS and assistant_msg.get("reasoning_content") is not None:
        if not full.startswith(prompt):
            base = prompt.removesuffix("<think>\n").removesuffix("<think>")
            assert full.startswith(base), (
                f"Base mismatch.\nbase[-200:]={repr(base[-200:])}\n"
                f"full[:len(base)+200]={repr(full[: len(base) + 200])}"
            )
            raw = full[len(base) :]
            raw = strip_eos(raw, stag_key, tokenizer)
            reasoning = assistant_msg["reasoning_content"]
            if not raw.startswith("</think>"):
                raw = "</think>" + raw
            return reasoning + raw

    assert full.startswith(prompt), (
        f"Full does not start with prompt.\nprompt[-200:]={repr(prompt[-200:])}\n"
        f"full[:len(prompt)+200]={repr(full[: len(prompt) + 200])}"
    )
    output = full[len(prompt) :]
    return strip_eos(output, stag_key, tokenizer)


def extract_output_encoder(encoder_name, stag_key, assistant_msg, tools, template_kwargs):
    if encoder_name == "dsv32":
        from encoding_dsv32 import encode_messages, eos_token
    else:
        from encoding_dsv4 import encode_messages, eos_token

    thinking_mode = template_kwargs["thinking_mode"]
    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    if tools:
        system_msg["tools"] = [{"type": "function", "function": t["function"]} for t in tools]

    ds_assistant = dict(assistant_msg)
    if "tool_calls" in ds_assistant:
        ds_calls = []
        for tc in ds_assistant["tool_calls"]:
            ds_tc = dict(tc)
            func = dict(tc["function"])
            if not isinstance(func["arguments"], str):
                func["arguments"] = json.dumps(func["arguments"])
            ds_tc["function"] = func
            ds_calls.append(ds_tc)
        ds_assistant["tool_calls"] = ds_calls

    prompt = encode_messages([system_msg, USER_MSG], thinking_mode=thinking_mode)
    full = encode_messages([system_msg, USER_MSG, ds_assistant], thinking_mode=thinking_mode)
    assert full.startswith(prompt)
    output = full[len(prompt) :]
    if output.endswith(eos_token):
        output = output[: -len(eos_token)]
    return output


def extract_model_output(stag_key, model_id, assistant_msg, tools, template_kwargs):
    if model_id.startswith("ENCODER:"):
        encoder_name = model_id.split(":")[1]
        return extract_output_encoder(encoder_name, stag_key, assistant_msg, tools, template_kwargs)
    return extract_output_tokenizer(model_id, stag_key, assistant_msg, tools, template_kwargs)


def validate_output(stag_key, tools, tool_choice, reasoning, model_output):
    structural_tag = get_model_structural_tag(
        stag_key, tools=tools or [], tool_choice=tool_choice, reasoning=reasoning
    )
    grammar = Grammar.from_structural_tag(structural_tag)
    accepted = _is_grammar_accept_string(grammar, model_output)
    assert accepted, f"Grammar rejected output:\n{repr(model_output[:500])}"


def generate_test_cases():
    cases = []
    for stag_key, model_id, reasoning, template_kwargs in MODEL_CONFIGS:
        for num_tools, tool_choice_str in TOOL_SCENARIOS:
            if num_tools > 0 and model_id in SKIP_TOOLS:
                continue
            if reasoning:
                cases.append(
                    (
                        stag_key,
                        model_id,
                        True,
                        REASONING_CONTENT,
                        num_tools,
                        tool_choice_str,
                        template_kwargs,
                    )
                )
                if model_id not in SKIP_EMPTY_REASONING:
                    cases.append(
                        (stag_key, model_id, True, "", num_tools, tool_choice_str, template_kwargs)
                    )
            else:
                cases.append(
                    (stag_key, model_id, False, None, num_tools, tool_choice_str, template_kwargs)
                )
    return cases


def case_id(case):
    stag_key, model_id, reasoning, reasoning_content, num_tools, tool_choice_str, _ = case
    model_short = model_id.split("/")[-1] if "/" in model_id else model_id.replace("ENCODER:", "")
    if not reasoning:
        r_tag = "off"
    elif reasoning_content:
        r_tag = "on"
    else:
        r_tag = "empty"
    return f"{stag_key}-{model_short}-r{r_tag}-{num_tools}t-{tool_choice_str}"


TEST_CASES = generate_test_cases()


@pytest.mark.parametrize("case", TEST_CASES, ids=[case_id(c) for c in TEST_CASES])
def test_reasoning_stag(case):
    (
        stag_key,
        model_id,
        reasoning,
        reasoning_content,
        num_tools,
        tool_choice_str,
        template_kwargs,
    ) = case
    tools = make_tools(num_tools)
    tool_choice = make_tool_choice(tool_choice_str, tools or [])
    num_tool_calls = 1 if num_tools > 0 else 0
    assistant_msg = make_assistant_msg(stag_key, reasoning_content, num_tool_calls)
    model_output = extract_model_output(stag_key, model_id, assistant_msg, tools, template_kwargs)
    validate_output(stag_key, tools, tool_choice, reasoning, model_output)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
