import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.structural_tag import StructuralTag
from xgrammar.testing import _is_grammar_accept_string


class Profiler:
    def __init__(self, tokenizer_id: str):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, use_fast=True, trust_remote_code=True
        )
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
        self.compiler = xgr.GrammarCompiler(
            self.tokenizer_info, max_threads=16, cache_enabled=False
        )

    def profile_stag(
        self, structural_tag_format: Union[Dict[str, Any], StructuralTag], instance: str
    ):
        if isinstance(structural_tag_format, StructuralTag):
            structural_tag = structural_tag_format
        else:
            structural_tag = {"type": "structural_tag", "format": structural_tag_format}
        time_begin = time.monotonic_ns()
        compiled_grammar = self.compiler.compile_structural_tag(structural_tag)
        time_end = time.monotonic_ns()
        compiler_duration = time_end - time_begin
        print(f"Compiling structural tag {structural_tag_format}")
        print(f"Compile time: {compiler_duration / 1000 / 1000} ms")
        matcher = xgr.GrammarMatcher(compiled_grammar)
        token_bitmask = xgr.allocate_token_bitmask(1, self.tokenizer_info.vocab_size)

        print(f"Matching instance: {instance}")

        for char in instance:
            matcher.accept_string(char)
            time_begin = time.monotonic_ns()
            matcher.fill_next_token_bitmask(token_bitmask)
            time_end = time.monotonic_ns()

            duration = time_end - time_begin
            print(f"Time to generate mask: {duration / 1000} us, Character: '{char}'")


profiler: Optional[Profiler] = None
PROFILER_ON = True
tokenizer_id = "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture(autouse=True, scope="module")
def disable_profiler(request):
    global PROFILER_ON
    global profiler
    markexpr = getattr(request.config.option, "markexpr", "") or request.config.getoption(
        "markexpr", ""
    )
    hf_token_not_provided = "not hf_token_required" in (markexpr or "")
    if hf_token_not_provided:
        PROFILER_ON = False
    else:
        profiler = Profiler(tokenizer_id)


def check_template_stag_with_grammar(
    structural_tag_format: Union[Dict[str, Any], StructuralTag],
    expected_grammar_ebnf: str,
    **kwargs: List[Dict[str, str]],
):
    if isinstance(structural_tag_format, StructuralTag) or (
        "type" in structural_tag_format and structural_tag_format["type"] == "structural_tag"
    ):
        grammar = xgr.Grammar.from_structural_tag_template(structural_tag_format, **kwargs)
    else:
        structural_tag = {"type": "structural_tag", "format": structural_tag_format}
        grammar = xgr.Grammar.from_structural_tag_template(structural_tag, **kwargs)
    assert str(grammar) == expected_grammar_ebnf


def check_template_stag_with_instance(
    structural_tag_format: Union[Dict[str, Any], StructuralTag],
    instance: str,
    is_accepted: bool = True,
    debug_print: bool = False,
    **kwargs: List[Dict[str, str]],
):
    if isinstance(structural_tag_format, StructuralTag) or (
        "type" in structural_tag_format and structural_tag_format["type"] == "structural_tag"
    ):
        stag_grammar = xgr.Grammar.from_structural_tag_template(structural_tag_format, **kwargs)
    else:
        structural_tag = {"type": "structural_tag", "format": structural_tag_format}
        stag_grammar = xgr.Grammar.from_structural_tag_template(structural_tag, **kwargs)
    accepted = _is_grammar_accept_string(stag_grammar, instance, debug_print=debug_print)
    assert accepted == is_accepted
    if PROFILER_ON:
        profiler.profile_stag(structural_tag_format, instance)


nested_array_template_value_grammar_instance_accepted = [
    (
        {
            "type": "structural_tag",
            "format": {
                "type": "const_string",
                "value": "Hello, {{nested.nested[].nested.nested[]}}!",
            },
        },
        {
            "nested": {
                "nested": [
                    {"nested": {"nested": ["A", "B", "C"]}},
                    {"nested": {"nested": ["D", "E", "F"]}},
                ]
            }
        },
        r"""const_string ::= (("Hello, A!"))
const_string_1 ::= (("Hello, B!"))
const_string_2 ::= (("Hello, C!"))
const_string_3 ::= (("Hello, D!"))
const_string_4 ::= (("Hello, E!"))
const_string_5 ::= (("Hello, F!"))
or ::= ((const_string) | (const_string_1) | (const_string_2) | (const_string_3) | (const_string_4) | (const_string_5))
root ::= ((or))
""",
        [
            ("Hello, A!", True),
            ("Hello, B!", True),
            ("Hello, C!", True),
            ("Hello, D!", True),
            ("Hello, E!", True),
            ("Hello, F!", True),
            ("Hello, G!", False),
        ],
    ),
    (
        {
            "type": "structural_tag",
            "format": {
                "type": "sequence",
                "elements": [
                    {"type": "const_string", "value": "{{nested1.nested2[].first}}"},
                    {
                        "type": "const_string",
                        "value": "Hello, {{nested1.nested2[].nested3.nested4[]}}!",
                    },
                    {"type": "const_string", "value": "{{nested1.nested2[].nested3.third}}"},
                    {"type": "const_string", "value": "{{nested1.last}}"},
                ],
            },
        },
        {
            "nested1": {
                "nested2": [
                    {
                        "nested3": {"nested4": ["A", "B", "C"], "third": "Farewell."},
                        "first": "Greetings",
                    },
                    {
                        "nested3": {"nested4": ["D", "E", "F"], "third": "See you later."},
                        "first": "Salutations",
                    },
                ],
                "last": "Goodbye.",
            }
        },
        r"""const_string ::= (("Greetings"))
const_string_1 ::= (("Hello, A!"))
const_string_2 ::= (("Hello, B!"))
const_string_3 ::= (("Hello, C!"))
or ::= ((const_string_1) | (const_string_2) | (const_string_3))
const_string_4 ::= (("Farewell."))
const_string_5 ::= (("Goodbye."))
sequence ::= ((const_string or const_string_4 const_string_5))
const_string_6 ::= (("Salutations"))
const_string_7 ::= (("Hello, D!"))
const_string_8 ::= (("Hello, E!"))
const_string_9 ::= (("Hello, F!"))
or_1 ::= ((const_string_7) | (const_string_8) | (const_string_9))
const_string_10 ::= (("See you later."))
const_string_11 ::= (("Goodbye."))
sequence_1 ::= ((const_string_6 or_1 const_string_10 const_string_11))
or_2 ::= ((sequence) | (sequence_1))
root ::= ((or_2))
""",
        [
            ("GreetingsHello, A!Farewell.Goodbye.", True),
            ("GreetingsHello, B!Farewell.Goodbye.", True),
            ("GreetingsHello, C!Farewell.Goodbye.", True),
            ("SalutationsHello, D!See you later.Goodbye.", True),
            ("SalutationsHello, E!See you later.Goodbye.", True),
            ("SalutationsHello, F!See you later.Goodbye.", True),
            ("GreetingsHello, D!Farewell.Goodbye.", False),
            ("SalutationsHello, A!See you later.Goodbye.", False),
        ],
    ),
    (
        {
            "type": "structural_tag",
            "format": {
                "type": "sequence",
                "elements": [
                    {
                        "type": "tag",
                        "begin": "{{nested1.nested2[].first}}",
                        "content": {
                            "type": "const_string",
                            "value": "Hello, {{nested1.nested2[].nested3.nested4[]}}!",
                        },
                        "end": "{{nested1.nested2[].nested3.third}}",
                    },
                    {"type": "const_string", "value": "{{nested1.last}}"},
                ],
            },
        },
        {
            "nested1": {
                "nested2": [
                    {
                        "nested3": {"nested4": ["A", "B", "C"], "third": "Farewell."},
                        "first": "Greetings",
                    },
                    {
                        "nested3": {"nested4": ["D", "E", "F"], "third": "See you later."},
                        "first": "Salutations",
                    },
                ],
                "last": "Goodbye.",
            }
        },
        r"""const_string ::= (("Hello, A!"))
const_string_1 ::= (("Hello, B!"))
const_string_2 ::= (("Hello, C!"))
or ::= ((const_string) | (const_string_1) | (const_string_2))
tag ::= (("Greetings" or "Farewell."))
const_string_3 ::= (("Goodbye."))
sequence ::= ((tag const_string_3))
const_string_4 ::= (("Hello, D!"))
const_string_5 ::= (("Hello, E!"))
const_string_6 ::= (("Hello, F!"))
or_1 ::= ((const_string_4) | (const_string_5) | (const_string_6))
tag_1 ::= (("Salutations" or_1 "See you later."))
const_string_7 ::= (("Goodbye."))
sequence_1 ::= ((tag_1 const_string_7))
or_2 ::= ((sequence) | (sequence_1))
root ::= ((or_2))
""",
        [
            ("GreetingsHello, A!Farewell.Goodbye.", True),
            ("GreetingsHello, B!Farewell.Goodbye.", True),
            ("GreetingsHello, C!Farewell.Goodbye.", True),
            ("SalutationsHello, D!See you later.Goodbye.", True),
            ("SalutationsHello, E!See you later.Goodbye.", True),
            ("SalutationsHello, F!See you later.Goodbye.", True),
            ("GreetingsHello, D!Farewell.Goodbye.", False),
            ("SalutationsHello, A!See you later.Goodbye.", False),
        ],
    ),
]


@pytest.mark.parametrize(
    "template_stag_format, template_values, expected_stag, instance_is_accepted_tuples",
    nested_array_template_value_grammar_instance_accepted,
)
def test_nested_array_template(
    template_stag_format: Dict[str, Any],
    template_values: Dict[str, Any],
    expected_stag: Dict,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    check_template_stag_with_grammar(template_stag_format, expected_stag, **template_values)
    for instance, is_accepted in instance_is_accepted_tuples:
        check_template_stag_with_instance(
            template_stag_format, instance, is_accepted, **template_values
        )


builtin_template_values_stag_grammar_instance_accepted = [
    (
        "llama",
        {
            "tools": [
                {
                    "name": "Calculator",
                    "description": "A calculator that can perform basic arithmetic operations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                            },
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["operation", "a", "b"],
                    },
                },
                {
                    "name": "Weather",
                    "description": "A tool to get the current weather in a specified location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The name of the city or location to get the weather for.",
                            }
                        },
                        "required": ["location"],
                    },
                },
            ]
        },
        r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_prop_0 ::= (("\"add\"") | ("\"subtract\"") | ("\"multiply\"") | ("\"divide\""))
root_part_1 ::= (([ \n\t]* "," [ \n\t]* "\"b\"" [ \n\t]* ":" [ \n\t]* basic_number))
root_part_0 ::= (([ \n\t]* "," [ \n\t]* "\"a\"" [ \n\t]* ":" [ \n\t]* basic_number root_part_1))
root_0 ::= (("{" [ \n\t]* "\"operation\"" [ \n\t]* ":" [ \n\t]* root_prop_0 root_part_0 [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
basic_any_1 ::= ((basic_number_8) | (basic_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_number_8 ::= ((basic_number_1_1 basic_number_7_1 basic_number_3_1 basic_number_6_1))
basic_string_1 ::= (("\"" basic_string_sub_1))
basic_boolean_1 ::= (("true") | ("false"))
basic_null_1 ::= (("null"))
basic_array_2 ::= (("[" [ \n\t]* basic_any_1 basic_array_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_2 ::= (("{" [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"location\"" [ \n\t]* ":" [ \n\t]* basic_string_1 [ \n\t]* "}"))
basic_integer_1_1 ::= ("" | ("-"))
basic_number_1_1 ::= ("" | ("-"))
basic_number_2_1 ::= (([0-9] basic_number_2_1) | ([0-9]))
basic_number_3_1 ::= ("" | ("." basic_number_2_1))
basic_number_4_1 ::= ("" | ([+\-]))
basic_number_5_1 ::= (([0-9] basic_number_5_1) | ([0-9]))
basic_number_6_1 ::= ("" | ([eE] basic_number_4_1 basic_number_5_1))
basic_array_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_1 basic_array_1_1))
basic_object_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1))
basic_number_7_1 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("\"Calculator\", \"parameters\": " root_0 "}") | ("\"Weather\", \"parameters\": " root_1 "}"))
triggered_tags ::= TagDispatch(
  ("{\"name\": ", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
        [
            (
                'OK, I will use the Calculator tool to perform the operation. {"name": "Calculator", "parameters": {"operation": "add", "a": 5, "b": 3}}',
                True,
            ),
            (
                'I need to know the weather in Paris. {"name": "Weather", "parameters": {"location": "Paris"}}',
                True,
            ),
            (
                'Can you calculate Paris add 5? {"name": "Calculator", "parameters": {"operation": "add", "a": 5, "b": "Paris"}}',
                False,
            ),
            (
                'I want to know the weather in 1. {"name": "Weather", "parameters": {"location": 1}}',
                False,
            ),
            ("Some random text", True),
        ],
    ),
    (
        "kimi",
        {
            "tools": [
                {
                    "name": "Calculator",
                    "description": "A calculator that can perform basic arithmetic operations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                            },
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["operation", "a", "b"],
                    },
                },
                {
                    "name": "Weather",
                    "description": "A tool to get the current weather in a specified location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The name of the city or location to get the weather for.",
                            }
                        },
                        "required": ["location"],
                    },
                },
            ]
        },
        r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_prop_0 ::= (("\"add\"") | ("\"subtract\"") | ("\"multiply\"") | ("\"divide\""))
root_part_1 ::= (([ \n\t]* "," [ \n\t]* "\"b\"" [ \n\t]* ":" [ \n\t]* basic_number))
root_part_0 ::= (([ \n\t]* "," [ \n\t]* "\"a\"" [ \n\t]* ":" [ \n\t]* basic_number root_part_1))
root_0 ::= (("{" [ \n\t]* "\"operation\"" [ \n\t]* ":" [ \n\t]* root_prop_0 root_part_0 [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
basic_any_1 ::= ((basic_number_8) | (basic_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_number_8 ::= ((basic_number_1_1 basic_number_7_1 basic_number_3_1 basic_number_6_1))
basic_string_1 ::= (("\"" basic_string_sub_1))
basic_boolean_1 ::= (("true") | ("false"))
basic_null_1 ::= (("null"))
basic_array_2 ::= (("[" [ \n\t]* basic_any_1 basic_array_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_2 ::= (("{" [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"location\"" [ \n\t]* ":" [ \n\t]* basic_string_1 [ \n\t]* "}"))
basic_integer_1_1 ::= ("" | ("-"))
basic_number_1_1 ::= ("" | ("-"))
basic_number_2_1 ::= (([0-9] basic_number_2_1) | ([0-9]))
basic_number_3_1 ::= ("" | ("." basic_number_2_1))
basic_number_4_1 ::= ("" | ([+\-]))
basic_number_5_1 ::= (([0-9] basic_number_5_1) | ([0-9]))
basic_number_6_1 ::= ("" | ([eE] basic_number_4_1 basic_number_5_1))
basic_array_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_1 basic_array_1_1))
basic_object_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1))
basic_number_7_1 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("Calculator<|tool_call_argument_begin|>" root_0 "<|tool_call_end|>") | ("Weather<|tool_call_argument_begin|>" root_1 "<|tool_call_end|>"))
triggered_tags ::= TagDispatch(
  ("<|tool_call_begin|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
        [
            (
                'OK, I will use the Calculator tool to perform the operation. <|tool_call_begin|>Calculator<|tool_call_argument_begin|>{"operation": "add", "a": 5, "b": 3}<|tool_call_end|>',
                True,
            ),
            (
                'I need to know the weather in Paris. <|tool_call_begin|>Weather<|tool_call_argument_begin|>{"location": "Paris"}<|tool_call_end|>',
                True,
            ),
            (
                'Can you calculate Paris add 5? <|tool_call_begin|>Calculator<|tool_call_argument_begin|>{"operation": "add", "a": 5, "b": "Paris"}<|tool_call_end|>',
                False,
            ),
            (
                'I want to know the weather in 1. <|tool_call_begin|>Weather<|tool_call_argument_begin|>{"location": 1}<|tool_call_end|>',
                False,
            ),
            ("Some random text", True),
        ],
    ),
    (
        "deepseek",
        {
            "tools": [
                {
                    "name": "Calculator",
                    "description": "A calculator that can perform basic arithmetic operations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                            },
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["operation", "a", "b"],
                    },
                },
                {
                    "name": "Weather",
                    "description": "A tool to get the current weather in a specified location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The name of the city or location to get the weather for.",
                            }
                        },
                        "required": ["location"],
                    },
                },
            ]
        },
        r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_prop_0 ::= (("\"add\"") | ("\"subtract\"") | ("\"multiply\"") | ("\"divide\""))
root_part_1 ::= (([ \n\t]* "," [ \n\t]* "\"b\"" [ \n\t]* ":" [ \n\t]* basic_number))
root_part_0 ::= (([ \n\t]* "," [ \n\t]* "\"a\"" [ \n\t]* ":" [ \n\t]* basic_number root_part_1))
root_0 ::= (("{" [ \n\t]* "\"operation\"" [ \n\t]* ":" [ \n\t]* root_prop_0 root_part_0 [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
basic_any_1 ::= ((basic_number_8) | (basic_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_number_8 ::= ((basic_number_1_1 basic_number_7_1 basic_number_3_1 basic_number_6_1))
basic_string_1 ::= (("\"" basic_string_sub_1))
basic_boolean_1 ::= (("true") | ("false"))
basic_null_1 ::= (("null"))
basic_array_2 ::= (("[" [ \n\t]* basic_any_1 basic_array_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_2 ::= (("{" [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"location\"" [ \n\t]* ":" [ \n\t]* basic_string_1 [ \n\t]* "}"))
basic_integer_1_1 ::= ("" | ("-"))
basic_number_1_1 ::= ("" | ("-"))
basic_number_2_1 ::= (([0-9] basic_number_2_1) | ([0-9]))
basic_number_3_1 ::= ("" | ("." basic_number_2_1))
basic_number_4_1 ::= ("" | ([+\-]))
basic_number_5_1 ::= (([0-9] basic_number_5_1) | ([0-9]))
basic_number_6_1 ::= ("" | ([eE] basic_number_4_1 basic_number_5_1))
basic_array_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_1 basic_array_1_1))
basic_object_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1))
basic_number_7_1 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("Calculator<\uff5ctool\u2581sep\uff5c>" root_0 "<\uff5ctool\u2581call\u2581end\uff5c>") | ("Weather<\uff5ctool\u2581sep\uff5c>" root_1 "<\uff5ctool\u2581call\u2581end\uff5c>"))
triggered_tags ::= TagDispatch(
  ("<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
        [
            (
                'OK, I will use the Calculator tool to perform the operation. <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>Calculator<｜tool▁sep｜>{"operation": "add", "a": 5, "b": 3}<｜tool▁call▁end｜>',
                True,
            ),
            (
                'I need to know the weather in Paris. <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>Weather<｜tool▁sep｜>{"location": "Paris"}<｜tool▁call▁end｜>',
                True,
            ),
            (
                'Can you calculate Paris add 5? <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>Calculator<｜tool▁sep｜>{"operation": "add", "a": 5, "b": "Paris"}<｜tool▁call▁end｜>',
                False,
            ),
            (
                'I want to know the weather in 1. <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>Weather<｜tool▁sep｜>{"location": 1}<<｜tool▁call▁end｜>',
                False,
            ),
            ("Some random text", True),
        ],
    ),
    (
        "qwen_coder",
        {
            "tools": [
                {
                    "name": "Calculator",
                    "description": "A calculator that can perform basic arithmetic operations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                            },
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["operation", "a", "b"],
                    },
                },
                {
                    "name": "Weather",
                    "description": "A tool to get the current weather in a specified location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The name of the city or location to get the weather for.",
                            }
                        },
                        "required": ["location"],
                    },
                },
            ]
        },
        r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
xml_string ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
xml_string_0 ::= ((xml_string))
xml_any ::= ((basic_number) | (xml_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_prop_0 ::= (("add") | ("subtract") | ("multiply") | ("divide"))
root_prop_1 ::= ((root_prop_1_1 root_prop_1_7 root_prop_1_3 root_prop_1_6))
root_prop_2 ::= ((root_prop_2_1 root_prop_2_7 root_prop_2_3 root_prop_2_6))
root_part_1 ::= (([ \n\t]* "<parameter=b>" [ \n\t]* root_prop_2 [ \n\t]* "</parameter>"))
root_part_0 ::= (([ \n\t]* "<parameter=a>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>" root_part_1))
root_0 ::= (([ \n\t]* "<parameter=operation>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" root_part_0))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
root_prop_1_1 ::= ("" | ("-"))
root_prop_1_2 ::= (([0-9] root_prop_1_2) | ([0-9]))
root_prop_1_3 ::= ("" | ("." root_prop_1_2))
root_prop_1_4 ::= ("" | ([+\-]))
root_prop_1_5 ::= (([0-9] root_prop_1_5) | ([0-9]))
root_prop_1_6 ::= ("" | ([eE] root_prop_1_4 root_prop_1_5))
root_prop_2_1 ::= ("" | ("-"))
root_prop_2_2 ::= (([0-9] root_prop_2_2) | ([0-9]))
root_prop_2_3 ::= ("" | ("." root_prop_2_2))
root_prop_2_4 ::= ("" | ([+\-]))
root_prop_2_5 ::= (([0-9] root_prop_2_5) | ([0-9]))
root_prop_2_6 ::= ("" | ([eE] root_prop_2_4 root_prop_2_5))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
root_prop_1_7 ::= (("0") | ([1-9] [0-9]*))
root_prop_2_7 ::= (("0") | ([1-9] [0-9]*))
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
xml_string_1 ::= TagDispatch(
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=("</parameter>")
)
xml_variable_name_1 ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
xml_string_0_1 ::= ((xml_string_1))
xml_any_1 ::= ((basic_number_8) | (xml_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_any_1 ::= ((basic_number_8) | (basic_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_number_8 ::= ((basic_number_1_1 basic_number_7_1 basic_number_3_1 basic_number_6_1))
basic_string_1 ::= (("\"" basic_string_sub_1))
basic_boolean_1 ::= (("true") | ("false"))
basic_null_1 ::= (("null"))
basic_array_2 ::= (("[" [ \n\t]* basic_any_1 basic_array_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_2 ::= (("{" [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_1 ::= (([ \n\t]* "<parameter=location>" [ \n\t]* xml_string_0_1 [ \n\t]* "</parameter>"))
basic_integer_1_1 ::= ("" | ("-"))
basic_number_1_1 ::= ("" | ("-"))
basic_number_2_1 ::= (([0-9] basic_number_2_1) | ([0-9]))
basic_number_3_1 ::= ("" | ("." basic_number_2_1))
basic_number_4_1 ::= ("" | ([+\-]))
basic_number_5_1 ::= (([0-9] basic_number_5_1) | ([0-9]))
basic_number_6_1 ::= ("" | ([eE] basic_number_4_1 basic_number_5_1))
basic_array_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_1 basic_array_1_1))
basic_object_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1))
basic_number_7_1 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("Calculator>" root_0 "</function>") | ("Weather>" root_1 "</function>"))
triggered_tags ::= TagDispatch(
  ("<function=", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
        [
            (
                "OK, I will use the Calculator tool to perform the operation. <function=Calculator><parameter=operation>add</parameter><parameter=a>5</parameter><parameter=b>3</parameter></function>",
                True,
            ),
            (
                "I need to know the weather in Paris. <function=Weather><parameter=location>Paris</parameter></function>",
                True,
            ),
            (
                "Can you calculate Paris add 5? <function=Calculator><parameter=operation>add</parameter><parameter=a>5</parameter><parameter=b>Paris</parameter></function>",
                False,
            ),
            ("Some random text", True),
        ],
    ),
    (
        "qwen",
        {
            "tools": [
                {
                    "name": "Calculator",
                    "description": "A calculator that can perform basic arithmetic operations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                            },
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["operation", "a", "b"],
                    },
                },
                {
                    "name": "Weather",
                    "description": "A tool to get the current weather in a specified location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The name of the city or location to get the weather for.",
                            }
                        },
                        "required": ["location"],
                    },
                },
            ]
        },
        r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_prop_0 ::= (("\"add\"") | ("\"subtract\"") | ("\"multiply\"") | ("\"divide\""))
root_part_1 ::= (([ \n\t]* "," [ \n\t]* "\"b\"" [ \n\t]* ":" [ \n\t]* basic_number))
root_part_0 ::= (([ \n\t]* "," [ \n\t]* "\"a\"" [ \n\t]* ":" [ \n\t]* basic_number root_part_1))
root_0 ::= (("{" [ \n\t]* "\"operation\"" [ \n\t]* ":" [ \n\t]* root_prop_0 root_part_0 [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
basic_any_1 ::= ((basic_number_8) | (basic_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_number_8 ::= ((basic_number_1_1 basic_number_7_1 basic_number_3_1 basic_number_6_1))
basic_string_1 ::= (("\"" basic_string_sub_1))
basic_boolean_1 ::= (("true") | ("false"))
basic_null_1 ::= (("null"))
basic_array_2 ::= (("[" [ \n\t]* basic_any_1 basic_array_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_2 ::= (("{" [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"location\"" [ \n\t]* ":" [ \n\t]* basic_string_1 [ \n\t]* "}"))
basic_integer_1_1 ::= ("" | ("-"))
basic_number_1_1 ::= ("" | ("-"))
basic_number_2_1 ::= (([0-9] basic_number_2_1) | ([0-9]))
basic_number_3_1 ::= ("" | ("." basic_number_2_1))
basic_number_4_1 ::= ("" | ([+\-]))
basic_number_5_1 ::= (([0-9] basic_number_5_1) | ([0-9]))
basic_number_6_1 ::= ("" | ([eE] basic_number_4_1 basic_number_5_1))
basic_array_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_1 basic_array_1_1))
basic_object_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1))
basic_number_7_1 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("{\"name\": \"Calculator\", \"arguments\": " root_0 "}</tool_call>") | ("{\"name\": \"Weather\", \"arguments\": " root_1 "}</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
        [
            (
                'OK, I will use the Calculator tool to perform the operation. <tool_call>{"name": "Calculator", "arguments": {"operation": "add", "a": 5, "b": 3}}</tool_call>',
                True,
            ),
            (
                'I need to know the weather in Paris. <tool_call>{"name": "Weather", "arguments": {"location": "Paris"}}</tool_call>',
                True,
            ),
            (
                'Can you calculate Paris add 5? <tool_call>{"name": "Calculator", "arguments": {"operation": "add", "a": 5, "b": "Paris"}}</tool_call>',
                False,
            ),
            (
                'I want to know the weather in 1. <tool_call>{"name": "Weather", "arguments": {"location": 1}}</tool_call>',
                False,
            ),
            ("Some random text", True),
        ],
    ),
    (
        "harmony",
        {
            "tools": [
                {
                    "name": "Calculator",
                    "description": "A calculator that can perform basic arithmetic operations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                            },
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["operation", "a", "b"],
                    },
                },
                {
                    "name": "Weather",
                    "description": "A tool to get the current weather in a specified location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The name of the city or location to get the weather for.",
                            }
                        },
                        "required": ["location"],
                    },
                },
            ],
            "builtin_tools": [
                {
                    "name": "Python",
                    "description": "A Python interpreter that can execute Python code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "The Python code to execute."}
                        },
                        "required": ["code"],
                    },
                }
            ],
        },
        r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|end|>"),
  loop_after_dispatch=false,
  excludes=()
)
any_text_1 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|return|>"),
  loop_after_dispatch=false,
  excludes=()
)
any_text_2 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<|call|>"),
  loop_after_dispatch=false,
  excludes=()
)
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_prop_0 ::= (("\"add\"") | ("\"subtract\"") | ("\"multiply\"") | ("\"divide\""))
root_part_1 ::= (([ \n\t]* "," [ \n\t]* "\"b\"" [ \n\t]* ":" [ \n\t]* basic_number))
root_part_0 ::= (([ \n\t]* "," [ \n\t]* "\"a\"" [ \n\t]* ":" [ \n\t]* basic_number root_part_1))
root_0 ::= (("{" [ \n\t]* "\"operation\"" [ \n\t]* ":" [ \n\t]* root_prop_0 root_part_0 [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
basic_any_1 ::= ((basic_number_8) | (basic_string_1) | (basic_boolean_1) | (basic_null_1) | (basic_array_2) | (basic_object_2))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_number_8 ::= ((basic_number_1_1 basic_number_7_1 basic_number_3_1 basic_number_6_1))
basic_string_1 ::= (("\"" basic_string_sub_1))
basic_boolean_1 ::= (("true") | ("false"))
basic_null_1 ::= (("null"))
basic_array_2 ::= (("[" [ \n\t]* basic_any_1 basic_array_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_2 ::= (("{" [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"location\"" [ \n\t]* ":" [ \n\t]* basic_string_1 [ \n\t]* "}"))
basic_integer_1_1 ::= ("" | ("-"))
basic_number_1_1 ::= ("" | ("-"))
basic_number_2_1 ::= (([0-9] basic_number_2_1) | ([0-9]))
basic_number_3_1 ::= ("" | ("." basic_number_2_1))
basic_number_4_1 ::= ("" | ([+\-]))
basic_number_5_1 ::= (([0-9] basic_number_5_1) | ([0-9]))
basic_number_6_1 ::= ("" | ([eE] basic_number_4_1 basic_number_5_1))
basic_array_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_1 basic_array_1_1))
basic_object_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1))
basic_number_7_1 ::= (("0") | ([1-9] [0-9]*))
basic_escape_2 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub_2 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_2) | ("\\" basic_escape_2 basic_string_sub_2)) (=([ \n\t]* [,}\]:]))
basic_any_2 ::= ((basic_number_9) | (basic_string_2) | (basic_boolean_2) | (basic_null_2) | (basic_array_3) | (basic_object_3))
basic_integer_3 ::= (("0") | (basic_integer_1_2 [1-9] [0-9]*))
basic_number_9 ::= ((basic_number_1_2 basic_number_7_2 basic_number_3_2 basic_number_6_2))
basic_string_2 ::= (("\"" basic_string_sub_2))
basic_boolean_2 ::= (("true") | ("false"))
basic_null_2 ::= (("null"))
basic_array_3 ::= (("[" [ \n\t]* basic_any_2 basic_array_1_2 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object_3 ::= (("{" [ \n\t]* basic_string_2 [ \n\t]* ":" [ \n\t]* basic_any_2 basic_object_1_2 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_2 ::= (("{" [ \n\t]* "\"code\"" [ \n\t]* ":" [ \n\t]* basic_string_2 [ \n\t]* "}"))
basic_integer_1_2 ::= ("" | ("-"))
basic_number_1_2 ::= ("" | ("-"))
basic_number_2_2 ::= (([0-9] basic_number_2_2) | ([0-9]))
basic_number_3_2 ::= ("" | ("." basic_number_2_2))
basic_number_4_2 ::= ("" | ([+\-]))
basic_number_5_2 ::= (([0-9] basic_number_5_2) | ([0-9]))
basic_number_6_2 ::= ("" | ([eE] basic_number_4_2 basic_number_5_2))
basic_array_1_2 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_2 basic_array_1_2))
basic_object_1_2 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_2 [ \n\t]* ":" [ \n\t]* basic_any_2 basic_object_1_2))
basic_number_7_2 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("assistant<|channel|>analysis<|message|>" any_text) | ("assistant<|channel|>final<|message|>" any_text_1) | ("assistant<|channel|>final<|message|>" any_text_2) | ("assistant<|channel|>commentary to=Calculator<|constrain|>json<|message|>" root_0 "<|end|>") | ("assistant<|channel|>commentary to=Weather<|constrain|>json<|message|>" root_1 "<|end|>") | ("assistant<|channel|>analysis to=Python<|message|>" root_2 "<|end|>"))
triggered_tags ::= TagDispatch(
  ("<|start|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
        [
            (
                '<|start|>assistant<|channel|>analysis<|message|>OK, I will use the Calculator tool to perform the operation.<|end|><|start|>assistant<|channel|>commentary to=Calculator<|constrain|>json<|message|>{"operation": "add", "a": 5, "b": 3}<|end|>',
                True,
            ),
            (
                '<|start|>assistant<|channel|>analysis<|message|>I need to know the weather in Paris.<|end|><|start|>assistant<|channel|>commentary to=Weather<|constrain|>json<|message|>{"location": "Paris"}<|end|>',
                True,
            ),
            (
                '<|start|>assistant<|channel|>analysis<|message|>Can you calculate Paris add 5?<|end|><|start|>assistant<|channel|>commentary to=Calculator<|constrain|>json<|message|>{"operation": "add", "a": 5, "b": "Paris"}<|end|>',
                False,
            ),
            (
                '<|start|>assistant<|channel|>analysis<|message|>I want to know the weather in 1.<|end|><|start|>assistant<|channel|>commentary to=Weather<|constrain|>json<|message|>{"location": 1}<|end|>',
                False,
            ),
            ("<|start|>assistant<|channel|>analysis<|message|>Some random text<|end|>", True),
            (
                '<|start|>assistant<|channel|>analysis to=Python<|message|>{"code": "print(\\"Hello, World!\\")"}<|end|>',
                True,
            ),
            (
                "<|start|>assistant<|channel|>final<|message|>The function should be called.<|call|>",
                True,
            ),
            ("<|start|>assistant<|channel|>final<|message|>All tasks done.<|return|>", True),
        ],
    ),
    (
        "llama",
        {"tools": []},
        r"""any_text ::= (([\0-\U0010ffff]*))
root ::= ((any_text))
""",
        [
            (
                'OK, I will use the Calculator tool to perform the operation. {"name": "Calculator", "parameters": {"operation": "add", "a": 5, "b": 3}}',
                True,
            ),
            (
                'I need to know the weather in Paris. {"name": "Weather", "parameters": {"location": "Paris"}}',
                True,
            ),
            (
                'Can you calculate Paris add 5? {"name": "Calculator", "parameters": {"operation": "add", "a": 5, "b": "Paris"}}',
                True,
            ),
            (
                'I want to know the weather in 1. {"name": "Weather", "parameters": {"location": 1}}',
                True,
            ),
            ("Some random text", True),
        ],
    ),
]


@pytest.mark.parametrize(
    "builtin_format_type, template_values, expected_grammar, instance_is_accepted_tuples",
    builtin_template_values_stag_grammar_instance_accepted,
)
def test_builtin_template_values(
    builtin_format_type: str,
    template_values: Dict[str, List[Dict[str, Any]]],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    """Test builtin format with template values"""
    template_stag_format = xgr.builtin_structural_tag_template.get_builtin_structural_tag_template(
        builtin_format_type
    )

    check_template_stag_with_grammar(template_stag_format, expected_grammar, **template_values)

    for instance, is_accepted in instance_is_accepted_tuples:
        check_template_stag_with_instance(
            template_stag_format, instance, is_accepted, **template_values
        )


example_tools = [
    {
        "name": "search_web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords or a full query string; must not be empty",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of results to return",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "summarize_text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Original text to be summarized"},
                "max_length": {
                    "type": "integer",
                    "minimum": 10,
                    "description": "Maximum length of the summary in tokens or characters",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "translate_text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to be translated"},
                "target_language": {
                    "type": "string",
                    "description": "Target language code, e.g. en, zh, ja",
                },
                "source_language": {
                    "type": "string",
                    "description": "Source language code; optional, auto-detected if omitted",
                },
            },
            "required": ["text", "target_language"],
        },
    },
    {
        "name": "extract_keywords",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Input text"},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Number of keywords to extract",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "classify_text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to be classified"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Candidate label list",
                },
                "multi_label": {
                    "type": "boolean",
                    "description": "Whether multiple labels are allowed",
                },
            },
            "required": ["text", "labels"],
        },
    },
    {
        "name": "sentiment_analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text used for sentiment analysis"},
                "granularity": {
                    "type": "string",
                    "enum": ["document", "sentence"],
                    "description": "Granularity level of analysis",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "generate_code",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Description of the code generation requirement",
                },
                "language": {
                    "type": "string",
                    "description": "Target programming language, e.g. python, cpp",
                },
                "style": {"type": "string", "description": "Desired code style or conventions"},
            },
            "required": ["prompt", "language"],
        },
    },
    {
        "name": "explain_code",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code snippet to be explained"},
                "language": {"type": "string", "description": "Programming language of the code"},
                "detail_level": {
                    "type": "string",
                    "enum": ["brief", "detailed"],
                    "description": "Level of explanation detail",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "refactor_code",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Original code"},
                "language": {"type": "string", "description": "Programming language"},
                "style": {
                    "type": "string",
                    "description": "Refactoring goal, e.g. performance or readability",
                },
            },
            "required": ["code", "language"],
        },
    },
    {
        "name": "answer_question",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "User question"},
                "context": {
                    "type": "string",
                    "description": "Optional background or reference information",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "generate_sql",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of the SQL query",
                },
                "dialect": {"type": "string", "description": "SQL dialect, e.g. mysql, postgres"},
                "tables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Available table names",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "analyze_data",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Input dataset, typically a structured array",
                },
                "operation": {
                    "type": "string",
                    "description": "Analysis operation, e.g. statistics or aggregation",
                },
                "group_by": {"type": "string", "description": "Field used for grouping"},
            },
            "required": ["data", "operation"],
        },
    },
    {
        "name": "format_text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Original text"},
                "format": {
                    "type": "string",
                    "enum": ["markdown", "html", "latex"],
                    "description": "Target output format",
                },
            },
            "required": ["text", "format"],
        },
    },
    {
        "name": "detect_language",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text whose language is to be detected"}
            },
            "required": ["text"],
        },
    },
    {
        "name": "generate_test_cases",
        "parameters": {
            "type": "object",
            "properties": {
                "function_description": {
                    "type": "string",
                    "description": "Description of function behavior and constraints",
                },
                "language": {"type": "string", "description": "Target language for the test cases"},
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of test cases to generate",
                },
            },
            "required": ["function_description", "language"],
        },
    },
    {
        "name": "rewrite_text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Original text"},
                "tone": {
                    "type": "string",
                    "description": "Desired tone, e.g. formal or conversational",
                },
                "length_constraint": {
                    "type": "string",
                    "description": "Length requirement, e.g. shorter or longer",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "extract_entities",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Input text"},
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity types to extract, e.g. person or location",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "generate_outline",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic or title"},
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Depth level of the outline",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "chat_completion",
        "parameters": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "List of conversation messages, each with role and content",
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 2,
                    "description": "Controls randomness of generation",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum number of tokens to generate",
                },
            },
            "required": ["messages"],
        },
    },
    {
        "name": "evaluate_answer",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Original question"},
                "answer": {"type": "string", "description": "Answer to be evaluated"},
                "criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Evaluation criteria, e.g. correctness or completeness",
                },
            },
            "required": ["question", "answer"],
        },
    },
]


builtin_model_names = ["llama", "qwen", "qwen_coder", "deepseek", "kimi"]
function_numbers = [5, 20]


@pytest.mark.parametrize("model_name", builtin_model_names)
@pytest.mark.parametrize("function_number", function_numbers)
def test_builtin_template_efficiency(model_name: str, function_number: int):
    tools = example_tools[:function_number]
    builtin_template = xgr.builtin_structural_tag_template.get_builtin_structural_tag_template(
        model_name
    )

    start_time = time.monotonic_ns()
    grammar = xgr.Grammar.from_structural_tag_template(builtin_template, tools=tools)
    end_time = time.monotonic_ns()
    print(
        f"Model: {model_name}, Function number: {function_number}, Conversion Time: {(end_time - start_time) / 1e6} ms."
    )

    if profiler is not None:
        start_time = time.monotonic_ns()
        _ = profiler.compiler.compile_grammar(grammar)
        end_time = time.monotonic_ns()
        print(
            f"Model: {model_name}, Function number: {function_number}, Compilation Time: {(end_time - start_time) / 1e6} ms."
        )


@pytest.mark.parametrize("function_number", function_numbers)
def test_harmony_builtin_template_efficiency(function_number: int):
    tools = example_tools[:function_number]
    builtin_template = xgr.builtin_structural_tag_template.get_builtin_structural_tag_template(
        "harmony"
    )

    start_time = time.monotonic_ns()
    grammar = xgr.Grammar.from_structural_tag_template(
        builtin_template, tools=tools, builtin_tools=tools
    )
    end_time = time.monotonic_ns()
    print(
        f"Model: harmony, Function number: {function_number}, Conversion Time: {(end_time - start_time) / 1e6} ms."
    )

    if profiler is not None:
        start_time = time.monotonic_ns()
        _ = profiler.compiler.compile_grammar(grammar)
        end_time = time.monotonic_ns()
        print(
            f"Model: harmony, Function number: {function_number}, Compilation Time: {(end_time - start_time) / 1e6} ms."
        )
