import sys
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


def check_stag_with_grammar(structural_tag_format: Dict[str, Any], expected_grammar_ebnf: str):
    structural_tag = {"type": "structural_tag", "format": structural_tag_format}
    stag_ebnf = xgr.Grammar.from_structural_tag(structural_tag)
    assert str(stag_ebnf) == expected_grammar_ebnf


def check_stag_with_instance(
    structural_tag_format: Union[Dict[str, Any], StructuralTag],
    instance: str,
    is_accepted: bool = True,
    debug_print: bool = False,
):
    if isinstance(structural_tag_format, StructuralTag):
        stag_grammar = xgr.Grammar.from_structural_tag(structural_tag_format)
    else:
        structural_tag = {"type": "structural_tag", "format": structural_tag_format}
        stag_grammar = xgr.Grammar.from_structural_tag(structural_tag)
    accepted = _is_grammar_accept_string(stag_grammar, instance, debug_print=debug_print)
    assert accepted == is_accepted
    if PROFILER_ON:
        profiler.profile_stag(structural_tag_format, instance)


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
    print(grammar)
    print(expected_grammar_ebnf)
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


const_string_stag_grammar = [
    (
        {"type": "const_string", "value": "Hello!"},
        r"""const_string ::= (("Hello!"))
root ::= ((const_string))
""",
    )
]

const_string_instance_is_accepted = [
    ("Hello!", True),
    ("Hello", False),
    ("Hello!!", False),
    ("HELLO!", False),
]


@pytest.mark.parametrize("stag_format, expected_grammar", const_string_stag_grammar)
@pytest.mark.parametrize("instance, is_accepted", const_string_instance_is_accepted)
def test_const_string_format(
    stag_format: Dict[str, Any], expected_grammar: str, instance: str, is_accepted: bool
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, is_accepted, debug_print=True)


json_schema_stag_grammar = [
    (
        {
            "type": "json_schema",
            "json_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
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
root_0 ::= (("{" [ \n\t]* "\"a\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
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
root ::= ((root_0))
""",
    )
]


json_schema_instance_is_accepted = [
    ('{"a": "hello"}', True),
    ('{"a": 123}', False),
    ('{"b": "hello"}', False),
    ("invalid json", False),
]


@pytest.mark.parametrize("stag_format, expected_grammar", json_schema_stag_grammar)
@pytest.mark.parametrize("instance, is_accepted", json_schema_instance_is_accepted)
def test_json_schema_format(
    stag_format: Dict[str, Any], expected_grammar: str, instance: str, is_accepted: bool
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, is_accepted)


qwen_parameter_xml_stag_grammar = [
    (
        {
            "type": "qwen_xml_parameter",
            "json_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
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
root_prop_1 ::= (("0") | (root_prop_1_1 [1-9] [0-9]*))
root_part_0 ::= (([ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>"))
root_0 ::= (([ \n\t]* "<parameter=name>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>" root_part_0))
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
basic_number_7 ::= (("0") | ([1-9] [0-9]*))
root ::= ((root_0))
""",
    )
]
qwen_parameter_xml_instance_is_accepted = [
    ("<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter>", True),
    ("<parameter=name>Bob</parameter>\t\n<parameter=age>\t100\n</parameter>", True),
    ("<parameter=name>Bob</parameter><parameter=age>100</parameter>", True),
    ("\n\t<parameter=name>Bob</parameter><parameter=age>100</parameter>", True),
    ('<parameter=name>"Bob&lt;"</parameter><parameter=age>100</parameter>', True),
    (
        """<parameter=name><!DOCTYPE html>
<html lang="en">
  <body><h1>Hello</h1></body>
</html></parameter><parameter=age>100</parameter>""",
        True,
    ),
]


@pytest.mark.parametrize("stag_format, expected_grammar", qwen_parameter_xml_stag_grammar)
@pytest.mark.parametrize("instance, is_accepted", qwen_parameter_xml_instance_is_accepted)
def test_qwen_parameter_xml_format(
    stag_format: Dict[str, Any], expected_grammar: str, instance: str, is_accepted: bool
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, is_accepted)


ebnf_grammar_stag_grammar = [
    (
        {
            "type": "grammar",
            "grammar": r"""root ::= "Hello!" number
            number ::= [0-9] | [0-9] number""",
        },
        r"""root_0 ::= (("Hello!" number))
number ::= (([0-9]) | ([0-9] number))
root ::= ((root_0))
""",
    )
]
ebnf_grammar_instance_is_accepted = [
    ("Hello!12345", True),
    ("Hello!0", True),
    ("Hello!", False),
    ("Hello!123a", False),
    ("Hi!123", False),
]


@pytest.mark.parametrize("stag_format, expected_grammar", ebnf_grammar_stag_grammar)
@pytest.mark.parametrize("instance, is_accepted", ebnf_grammar_instance_is_accepted)
def test_ebnf_grammar_format(
    stag_format: Dict[str, Any], expected_grammar: str, instance: str, is_accepted: bool
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, is_accepted)


regex_stag_grammar = [
    (
        {"type": "regex", "pattern": "Hello![0-9]+"},
        r"""root_0 ::= (("H" "e" "l" "l" "o" "!" root_1))
root_1 ::= (([0-9] root_1) | ([0-9]))
root ::= ((root_0))
""",
    )
]
regex_instance_is_accepted = [
    ("Hello!12345", True),
    ("Hello!0", True),
    ("Hello!", False),
    ("Hello!123a", False),
    ("Hi!123", False),
]


@pytest.mark.parametrize("stag_format, expected_grammar", regex_stag_grammar)
@pytest.mark.parametrize("instance, is_accepted", regex_instance_is_accepted)
def test_regex_format(
    stag_format: Dict[str, Any], expected_grammar: str, instance: str, is_accepted: bool
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, is_accepted)


sequence_stag_grammar = [
    (
        {
            "type": "sequence",
            "elements": [
                {"type": "const_string", "value": "Hello!"},
                {"type": "json_schema", "json_schema": {"type": "number"}},
                {"type": "grammar", "grammar": 'root ::= "" | [-+*/]'},
                {"type": "regex", "pattern": "[simple]?"},
            ],
        },
        r"""const_string ::= (("Hello!"))
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
root_0 ::= ((basic_number))
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
root_1 ::= ("" | ([\-+*/]))
root_2 ::= ((root_1_1))
root_1_1 ::= ("" | ([simple]))
sequence ::= ((const_string root_0 root_1 root_2))
root ::= ((sequence))
""",
    )
]


sequence_instance_is_accepted = [
    ("Hello!123", True),
    ("Hello!Hello!", False),
    ("Hello!", False),
    ("123Hello!", False),
    ("???", False),
    ("Hello!123+", True),
    ("Hello!123-", True),
    ("Hello!123!", False),
    ("Hello!123s", True),
    ("Hello!123+s", True),
    ("Hello!123q", False),
]


@pytest.mark.parametrize("stag_format, expected_grammar", sequence_stag_grammar)
@pytest.mark.parametrize("instance, is_accepted", sequence_instance_is_accepted)
def test_sequence_format(
    stag_format: Dict[str, Any], expected_grammar: str, instance: str, is_accepted: bool
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, is_accepted)


or_stag_grammar = [
    (
        {
            "type": "or",
            "elements": [
                {"type": "const_string", "value": "Hello!"},
                {"type": "json_schema", "json_schema": {"type": "number"}},
            ],
        },
        r"""const_string ::= (("Hello!"))
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
root_0 ::= ((basic_number))
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
or ::= ((const_string) | (root_0))
root ::= ((or))
""",
    )
]


or_instance_is_accepted = [
    ("Hello!", True),
    ("123", True),
    ("Hello!Hello!", False),
    ("123Hello!", False),
    ("???", False),
]


@pytest.mark.parametrize("stag_format, expected_grammar", or_stag_grammar)
@pytest.mark.parametrize("instance, is_accepted", or_instance_is_accepted)
def test_or_format(
    stag_format: Dict[str, Any], expected_grammar: str, instance: str, is_accepted: bool
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, is_accepted)


tag_stag_grammar = [
    (
        {
            "type": "tag",
            "begin": "BEG",
            "content": {"type": "json_schema", "json_schema": {"type": "number"}},
            "end": "END",
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
root_0 ::= ((basic_number))
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
tag ::= (("BEG" root_0 "END"))
root ::= ((tag))
""",
    ),
    (
        {
            "type": "tag",
            "begin": "BEG",
            "content": {"type": "grammar", "grammar": "root ::= [+\\-]?[1-9][0-9]*"},
            "end": "END",
        },
        r"""root_0 ::= ((root_1 [1-9] [0-9]*))
root_1 ::= ("" | ([+\-]))
tag ::= (("BEG" root_0 "END"))
root ::= ((tag))
""",
    ),
    (
        {
            "type": "tag",
            "begin": "BEG",
            "content": {"type": "regex", "pattern": "[+\\-]?[1-9][0-9]*"},
            "end": "END",
        },
        r"""root_0 ::= ((root_1 [1-9] [0-9]*))
root_1 ::= ("" | ([+\-]))
tag ::= (("BEG" root_0 "END"))
root ::= ((tag))
""",
    ),
]


tag_instance_is_accepted = [
    ("BEG12345END", True),
    ("BEG123456END", True),
    ("BEG1234567END", True),
    ("BEG???END", False),
    ("BEG12345ENDEND", False),
]


@pytest.mark.parametrize("stag_format, expected_grammar", tag_stag_grammar)
@pytest.mark.parametrize("instance, is_accepted", tag_instance_is_accepted)
def test_tag_format(
    stag_format: Dict[str, Any], expected_grammar: str, instance: str, is_accepted: bool
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, is_accepted)


any_text_stag_grammar = [
    (
        {"type": "tag", "begin": "BEG", "content": {"type": "any_text"}, "end": "END"},
        r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("END"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("BEG" any_text))
root ::= ((tag))
""",
    )
]


any_text_instance_is_accepted = [
    ("BEGHello!END", True),
    ("BEGENENNDENEND", True),
    ("BEGENENDEN", False),
    ("BEGBEGENDEND", False),
]


@pytest.mark.parametrize("stag_format, expected_grammar", any_text_stag_grammar)
@pytest.mark.parametrize("instance, is_accepted", any_text_instance_is_accepted)
def test_any_text_format(
    stag_format: Dict[str, Any], expected_grammar: str, instance: str, is_accepted: bool
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, is_accepted)


any_text_only_stag_grammar = [
    (
        {"type": "any_text"},
        r"""any_text ::= (([\0-\U0010ffff]*))
root ::= ((any_text))
""",
    )
]


any_text_only_instance_is_accepted = [("ABCDEF", True), ("123456", True), ("", True)]


@pytest.mark.parametrize("stag_format, expected_grammar", any_text_only_stag_grammar)
@pytest.mark.parametrize("instance, is_accepted", any_text_only_instance_is_accepted)
def test_any_text_only_format(
    stag_format: Dict[str, Any], expected_grammar: str, instance: str, is_accepted: bool
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, is_accepted)


def _get_triggered_tag_format(at_least_one: bool, stop_after_first: bool):
    return {
        "type": "triggered_tags",
        "triggers": ["A"],
        "tags": [
            {"begin": "A1", "content": {"type": "const_string", "value": "L1"}, "end": "A"},
            {"begin": "A2", "content": {"type": "const_string", "value": "L2"}, "end": "A"},
        ],
        "at_least_one": at_least_one,
        "stop_after_first": stop_after_first,
    }


triggered_tag_stag_grammar = [
    (
        0,
        _get_triggered_tag_format(at_least_one=False, stop_after_first=False),
        r"""const_string ::= (("L1"))
const_string_1 ::= (("L2"))
triggered_tags_group ::= (("1" const_string "A") | ("2" const_string_1 "A"))
triggered_tags ::= TagDispatch(
  ("A", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
    ),
    (
        1,
        _get_triggered_tag_format(at_least_one=True, stop_after_first=False),
        r"""const_string ::= (("L1"))
const_string_1 ::= (("L2"))
triggered_tags_group ::= (("1" const_string "A") | ("2" const_string_1 "A"))
triggered_tags_first ::= (("A1" const_string "A") | ("A2" const_string_1 "A"))
triggered_tags_sub ::= TagDispatch(
  ("A", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
triggered_tags ::= ((triggered_tags_first triggered_tags_sub))
root ::= ((triggered_tags))
""",
    ),
    (
        2,
        _get_triggered_tag_format(at_least_one=False, stop_after_first=True),
        r"""const_string ::= (("L1"))
const_string_1 ::= (("L2"))
triggered_tags_group ::= (("1" const_string "A") | ("2" const_string_1 "A"))
triggered_tags ::= TagDispatch(
  ("A", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=false,
  excludes=()
)
root ::= ((triggered_tags))
""",
    ),
    (
        3,
        _get_triggered_tag_format(at_least_one=True, stop_after_first=True),
        r"""const_string ::= (("L1"))
const_string_1 ::= (("L2"))
triggered_tags ::= (("A1" const_string "A") | ("A2" const_string_1 "A"))
root ::= ((triggered_tags))
""",
    ),
]


triggered_tag_instance_accepted_results = [
    ("textA1L1AtextA2L2AText", [True, False, False, False]),
    ("textA1L1AtextA2L2A", [True, False, False, False]),
    ("A1L1Atext", [True, True, False, False]),
    ("A1L1AtextA2L2A", [True, True, False, False]),
    ("A1L1A", [True, True, True, True]),
    ("text", [True, False, True, False]),
    ("", [True, False, True, False]),
    ("AA", [False, False, False, False]),
    ("A1L2A", [False, False, False, False]),
    ("A1L1A2L2A", [False, False, False, False]),
]


@pytest.mark.parametrize("stag_id, stag_format, expected_grammar", triggered_tag_stag_grammar)
@pytest.mark.parametrize("instance, accepted_results", triggered_tag_instance_accepted_results)
def test_triggered_tag_format(
    stag_id: int,
    stag_format: Dict[str, Any],
    expected_grammar: str,
    instance: str,
    accepted_results: List[bool],
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, accepted_results[stag_id])


test_triggered_tags_corner_case_data = [
    (
        {
            "type": "triggered_tags",
            "triggers": ["<start>"],
            "tags": [
                {
                    "begin": "<start>",
                    "content": {"type": "const_string", "value": "[TEXT]"},
                    "end": "<end>",
                }
            ],
        },
        r"""const_string ::= (("[TEXT]"))
triggered_tags_group ::= (("" const_string "<end>"))
triggered_tags ::= TagDispatch(
  ("<start>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
        [("<start>[TEXT]<end>[TEXT]<start>[TEXT]<end>[TEXT]", True)],
    )
]


@pytest.mark.parametrize(
    "stag_format, expected_grammar, instance_is_accepted_tuples",
    test_triggered_tags_corner_case_data,
)
def test_triggered_tags_corner_case(
    stag_format: Dict[str, Any],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    check_stag_with_grammar(stag_format, expected_grammar)
    for instance, is_accepted in instance_is_accepted_tuples:
        check_stag_with_instance(stag_format, instance, is_accepted)


triggered_tag_format = {
    "type": "triggered_tags",
    "triggers": ["A"],
    "tags": [
        {"begin": "A1", "content": {"type": "const_string", "value": "L1"}, "end": "A"},
        {"begin": "A2", "content": {"type": "const_string", "value": "L2"}, "end": "A"},
    ],
}


def _get_triggered_tag_with_outside_tag(at_least_one: bool, stop_after_first: bool):
    return {
        "type": "tag",
        "begin": "begin",
        "content": {
            "type": "triggered_tags",
            "triggers": ["A"],
            "tags": [
                {"begin": "A1", "content": {"type": "const_string", "value": "L1"}, "end": "A"},
                {"begin": "A2", "content": {"type": "const_string", "value": "L2"}, "end": "A"},
            ],
            "at_least_one": at_least_one,
            "stop_after_first": stop_after_first,
        },
        "end": "end",
    }


triggered_tag_with_outside_tag_stag_grammar = [
    (
        0,
        _get_triggered_tag_with_outside_tag(at_least_one=False, stop_after_first=False),
        r"""const_string ::= (("L1"))
const_string_1 ::= (("L2"))
triggered_tags_group ::= (("1" const_string "A") | ("2" const_string_1 "A"))
triggered_tags ::= TagDispatch(
  ("A", triggered_tags_group),
  stop_eos=false,
  stop_str=("end"),
  loop_after_dispatch=true,
  excludes=()
)
tag ::= (("begin" triggered_tags))
root ::= ((tag))
""",
    ),
    (
        1,
        _get_triggered_tag_with_outside_tag(at_least_one=True, stop_after_first=False),
        r"""const_string ::= (("L1"))
const_string_1 ::= (("L2"))
triggered_tags_group ::= (("1" const_string "A") | ("2" const_string_1 "A"))
triggered_tags_first ::= (("A1" const_string "A") | ("A2" const_string_1 "A"))
triggered_tags_sub ::= TagDispatch(
  ("A", triggered_tags_group),
  stop_eos=false,
  stop_str=("end"),
  loop_after_dispatch=true,
  excludes=()
)
triggered_tags ::= ((triggered_tags_first triggered_tags_sub))
tag ::= (("begin" triggered_tags))
root ::= ((tag))
""",
    ),
    (
        2,
        _get_triggered_tag_with_outside_tag(at_least_one=False, stop_after_first=True),
        r"""const_string ::= (("L1"))
const_string_1 ::= (("L2"))
triggered_tags_group ::= (("1" const_string "A") | ("2" const_string_1 "A"))
triggered_tags ::= TagDispatch(
  ("A", triggered_tags_group),
  stop_eos=false,
  stop_str=("end"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("begin" triggered_tags))
root ::= ((tag))
""",
    ),
    (
        3,
        _get_triggered_tag_with_outside_tag(at_least_one=True, stop_after_first=True),
        r"""const_string ::= (("L1"))
const_string_1 ::= (("L2"))
triggered_tags_sub ::= (("A1" const_string "A") | ("A2" const_string_1 "A"))
triggered_tags ::= ((triggered_tags_sub "end"))
tag ::= (("begin" triggered_tags))
root ::= ((tag))
""",
    ),
]


triggered_tag_with_outside_tag_instance_accepted_results = [
    ("beginabcA1L1Atextend", [True, False, False, False]),
    ("beginA1L1AtextA2L2Aend", [True, True, False, False]),
    ("beginA1L1Aend", [True, True, True, True]),
    ("beginend", [True, False, True, False]),
    ("beginA1L1Aendabc", [False, False, False, False]),
    ("beginA1L2end", [False, False, False, False]),
]


@pytest.mark.parametrize(
    "stag_id, stag_format, expected_grammar", triggered_tag_with_outside_tag_stag_grammar
)
@pytest.mark.parametrize(
    "instance, accepted_results", triggered_tag_with_outside_tag_instance_accepted_results
)
def test_triggered_tag_with_outside_tag(
    stag_id: int,
    stag_format: Dict[str, Any],
    expected_grammar: str,
    instance: str,
    accepted_results: List[bool],
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, accepted_results[stag_id])


def _get_tags_with_separator_format(at_least_one: bool, stop_after_first: bool):
    return {
        "type": "tags_with_separator",
        "tags": [
            {"begin": "A1", "content": {"type": "const_string", "value": "L1"}, "end": "A"},
            {"begin": "A2", "content": {"type": "const_string", "value": "L2"}, "end": "A"},
        ],
        "separator": "AA",
        "at_least_one": at_least_one,
        "stop_after_first": stop_after_first,
    }


tags_with_separator_stag_grammar = [
    (
        0,
        _get_tags_with_separator_format(at_least_one=False, stop_after_first=False),
        r"""const_string ::= (("L1"))
tag ::= (("A1" const_string "A"))
const_string_1 ::= (("L2"))
tag_1 ::= (("A2" const_string_1 "A"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | ("AA" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
    ),
    (
        1,
        _get_tags_with_separator_format(at_least_one=True, stop_after_first=False),
        r"""const_string ::= (("L1"))
tag ::= (("A1" const_string "A"))
const_string_1 ::= (("L2"))
tag_1 ::= (("A2" const_string_1 "A"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | ("AA" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
    ),
    (
        2,
        _get_tags_with_separator_format(at_least_one=False, stop_after_first=True),
        r"""const_string ::= (("L1"))
tag ::= (("A1" const_string "A"))
const_string_1 ::= (("L2"))
tag_1 ::= (("A2" const_string_1 "A"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator ::= ("" | (tags_with_separator_tags))
root ::= ((tags_with_separator))
""",
    ),
    (
        3,
        _get_tags_with_separator_format(at_least_one=True, stop_after_first=True),
        r"""const_string ::= (("L1"))
tag ::= (("A1" const_string "A"))
const_string_1 ::= (("L2"))
tag_1 ::= (("A2" const_string_1 "A"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator ::= ((tags_with_separator_tags))
root ::= ((tags_with_separator))
""",
    ),
]


tags_with_separator_instance_accepted_results = [
    ("", [True, False, True, False]),
    ("A1L1A", [True, True, True, True]),
    ("A1L1AAAA2L2A", [True, True, False, False]),
    ("A1L1AA2L2A", [False, False, False, False]),
]


@pytest.mark.parametrize("stag_id, stag_format, expected_grammar", tags_with_separator_stag_grammar)
@pytest.mark.parametrize(
    "instance, accepted_results", tags_with_separator_instance_accepted_results
)
def test_tags_with_separator_format(
    stag_id: int,
    stag_format: Dict[str, Any],
    expected_grammar: str,
    instance: str,
    accepted_results: List[bool],
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, accepted_results[stag_id])


def _get_tags_with_separator_format_with_outside_tag(at_least_one: bool, stop_after_first: bool):
    return {
        "type": "tag",
        "begin": "begin",
        "content": {
            "type": "tags_with_separator",
            "tags": [
                {"begin": "A1", "content": {"type": "const_string", "value": "L1"}, "end": "A"},
                {"begin": "A2", "content": {"type": "const_string", "value": "L2"}, "end": "A"},
            ],
            "separator": "AA",
            "at_least_one": at_least_one,
            "stop_after_first": stop_after_first,
        },
        "end": "end",
    }


tags_with_separator_with_outside_tag_stag_grammar = [
    (
        0,
        _get_tags_with_separator_format_with_outside_tag(
            at_least_one=False, stop_after_first=False
        ),
        r"""const_string ::= (("L1"))
tag ::= (("A1" const_string "A"))
const_string_1 ::= (("L2"))
tag_1 ::= (("A2" const_string_1 "A"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= (("AA" tags_with_separator_tags tags_with_separator_sub) | ("end"))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub) | ("end"))
tag_2 ::= (("begin" tags_with_separator))
root ::= ((tag_2))
""",
    ),
    (
        1,
        _get_tags_with_separator_format_with_outside_tag(at_least_one=True, stop_after_first=False),
        r"""const_string ::= (("L1"))
tag ::= (("A1" const_string "A"))
const_string_1 ::= (("L2"))
tag_1 ::= (("A2" const_string_1 "A"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= (("AA" tags_with_separator_tags tags_with_separator_sub) | ("end"))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
tag_2 ::= (("begin" tags_with_separator))
root ::= ((tag_2))
""",
    ),
    (
        2,
        _get_tags_with_separator_format_with_outside_tag(at_least_one=False, stop_after_first=True),
        r"""const_string ::= (("L1"))
tag ::= (("A1" const_string "A"))
const_string_1 ::= (("L2"))
tag_1 ::= (("A2" const_string_1 "A"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator ::= ((tags_with_separator_tags "end") | ("end"))
tag_2 ::= (("begin" tags_with_separator))
root ::= ((tag_2))
""",
    ),
    (
        3,
        _get_tags_with_separator_format_with_outside_tag(at_least_one=True, stop_after_first=True),
        r"""const_string ::= (("L1"))
tag ::= (("A1" const_string "A"))
const_string_1 ::= (("L2"))
tag_1 ::= (("A2" const_string_1 "A"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator ::= ((tags_with_separator_tags "end"))
tag_2 ::= (("begin" tags_with_separator))
root ::= ((tag_2))
""",
    ),
]


tags_with_separator_with_outside_tag_instance_accepted_results = [
    ("beginend", [True, False, True, False]),
    ("beginA1L1Aend", [True, True, True, True]),
    ("beginA1L1AAAA2L2Aend", [True, True, False, False]),
    ("beginA1L1A", [False, False, False, False]),
    ("beginA1L1AA2L2Aend", [False, False, False, False]),
]


@pytest.mark.parametrize(
    "stag_id, stag_format, expected_grammar", tags_with_separator_with_outside_tag_stag_grammar
)
@pytest.mark.parametrize(
    "instance, accepted_results", tags_with_separator_with_outside_tag_instance_accepted_results
)
def test_tags_with_separator_format_with_outside_tag(
    stag_id: int,
    stag_format: Dict[str, Any],
    expected_grammar: str,
    instance: str,
    accepted_results: List[bool],
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, accepted_results[stag_id])


# Test for empty separator in tags_with_separator
def _get_tags_with_empty_separator_format(at_least_one: bool, stop_after_first: bool):
    return {
        "type": "tags_with_separator",
        "tags": [
            {"begin": "<a>", "content": {"type": "const_string", "value": "X"}, "end": "</a>"},
            {"begin": "<b>", "content": {"type": "const_string", "value": "Y"}, "end": "</b>"},
        ],
        "separator": "",
        "at_least_one": at_least_one,
        "stop_after_first": stop_after_first,
    }


tags_with_empty_separator_stag_grammar = [
    (
        0,
        _get_tags_with_empty_separator_format(at_least_one=False, stop_after_first=False),
        r"""const_string ::= (("X"))
tag ::= (("<a>" const_string "</a>"))
const_string_1 ::= (("Y"))
tag_1 ::= (("<b>" const_string_1 "</b>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
    ),
    (
        1,
        _get_tags_with_empty_separator_format(at_least_one=True, stop_after_first=False),
        r"""const_string ::= (("X"))
tag ::= (("<a>" const_string "</a>"))
const_string_1 ::= (("Y"))
tag_1 ::= (("<b>" const_string_1 "</b>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
    ),
    (
        2,
        _get_tags_with_empty_separator_format(at_least_one=False, stop_after_first=True),
        r"""const_string ::= (("X"))
tag ::= (("<a>" const_string "</a>"))
const_string_1 ::= (("Y"))
tag_1 ::= (("<b>" const_string_1 "</b>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator ::= ("" | (tags_with_separator_tags))
root ::= ((tags_with_separator))
""",
    ),
    (
        3,
        _get_tags_with_empty_separator_format(at_least_one=True, stop_after_first=True),
        r"""const_string ::= (("X"))
tag ::= (("<a>" const_string "</a>"))
const_string_1 ::= (("Y"))
tag_1 ::= (("<b>" const_string_1 "</b>"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator ::= ((tags_with_separator_tags))
root ::= ((tags_with_separator))
""",
    ),
]


tags_with_empty_separator_instance_accepted_results = [
    ("", [True, False, True, False]),
    ("<a>X</a>", [True, True, True, True]),
    ("<a>X</a><b>Y</b>", [True, True, False, False]),
    ("<b>Y</b><a>X</a><b>Y</b>", [True, True, False, False]),
    ("<a>X</a><a>X</a><a>X</a>", [True, True, False, False]),
    # Invalid cases
    ("<a>X</a>,<b>Y</b>", [False, False, False, False]),  # Has separator when none expected
    ("<c>Z</c>", [False, False, False, False]),  # Unknown tag
]


@pytest.mark.parametrize(
    "stag_id, stag_format, expected_grammar", tags_with_empty_separator_stag_grammar
)
@pytest.mark.parametrize(
    "instance, accepted_results", tags_with_empty_separator_instance_accepted_results
)
def test_tags_with_empty_separator_format(
    stag_id: int,
    stag_format: Dict[str, Any],
    expected_grammar: str,
    instance: str,
    accepted_results: List[bool],
):
    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, accepted_results[stag_id])


compound_stag_instance_is_accepted = [
    # Llama JSON-based tool calling
    (
        {
            "type": "triggered_tags",
            "triggers": ['{"name":'],
            "tags": [
                {
                    "begin": '{"name": "func1", "parameters": ',
                    "content": {"type": "json_schema", "json_schema": {"type": "object"}},
                    "end": "}",
                },
                {
                    "begin": '{"name": "func2", "parameters": ',
                    "content": {"type": "json_schema", "json_schema": {"type": "object"}},
                    "end": "}",
                },
            ],
        },
        [
            (
                '<text>{"name": "func2", "parameters": {"arg": 10}}<text>{"name": "func1", "parameters": {"arg": "123"}}<text>',
                True,
            ),
            ('<text>{"name": "func3", "parameters": {"arg": 10}}', False),
        ],
    ),
    # Force think
    (
        {
            "type": "sequence",
            "elements": [
                {
                    "type": "tag",
                    "begin": "<think>",
                    "content": {"type": "any_text"},
                    "end": "</think>",
                },
                {
                    "type": "triggered_tags",
                    "triggers": ["<function="],
                    "tags": [
                        {
                            "begin": "<function=func1>",
                            "content": {"type": "json_schema", "json_schema": {"type": "object"}},
                            "end": "</function>",
                        },
                        {
                            "begin": "<function=func2>",
                            "content": {"type": "json_schema", "json_schema": {"type": "object"}},
                            "end": "</function>",
                        },
                    ],
                },
            ],
        },
        [
            (
                '<think>[any_text]</think>[any_text]<function=func2>{"arg": 10}</function>[any_text]<function=func1>{"arg": 10}</function>[any_text]',
                True,
            ),
            (
                '[any_text]<function=func2>{"arg": 10}</function>[any_text]<function=func1>{"arg": 10}</function>[any_text]',
                False,
            ),
            ('<think>[any_text]</think>[any_text]<function=func3>{"arg": 10}', False),
        ],
    ),
    # Think & Force tool calling (Llama style)
    (
        {
            "type": "sequence",
            "elements": [
                {
                    "type": "tag",
                    "begin": "<think>",
                    "content": {"type": "any_text"},
                    "end": "</think>",
                },
                {
                    "type": "triggered_tags",
                    "triggers": ["<function="],
                    "tags": [
                        {
                            "begin": "<function=func1>",
                            "content": {"type": "json_schema", "json_schema": {"type": "object"}},
                            "end": "</function>",
                        },
                        {
                            "begin": "<function=func2>",
                            "content": {"type": "json_schema", "json_schema": {"type": "object"}},
                            "end": "</function>",
                        },
                    ],
                    "stop_after_first": True,
                    "at_least_one": True,
                },
            ],
        },
        [
            ('<think>[any_text]</think><function=func2>{"arg": 10}</function>', True),
            ('<think>[any_text]</think>[any_text]<function=func2>{"arg": 10}</function>', False),
            ('<think>[any_text]</think><function=func2>{"arg": 10}</function>[any_text]', False),
        ],
    ),
    # Think & force tool calling (DeepSeek style)
    (
        {
            "type": "sequence",
            "elements": [
                {
                    "type": "tag",
                    "begin": "<think>",
                    "content": {"type": "any_text"},
                    "end": "</think>",
                },
                {
                    "type": "triggered_tags",
                    "triggers": ["<｜tool▁calls▁begin｜>"],
                    "tags": [
                        {
                            "begin": "<｜tool▁calls▁begin｜>",
                            "end": "<｜tool▁calls▁end｜>",
                            "content": {
                                "type": "tags_with_separator",
                                "separator": "\n",
                                "tags": [
                                    {
                                        "begin": "<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```json\n",
                                        "content": {
                                            "type": "json_schema",
                                            "json_schema": {"type": "object"},
                                        },
                                        "end": "\n```<｜tool▁call▁end｜>",
                                    },
                                    {
                                        "begin": "<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2\n```json\n",
                                        "content": {
                                            "type": "json_schema",
                                            "json_schema": {"type": "object"},
                                        },
                                        "end": "\n```<｜tool▁call▁end｜>",
                                    },
                                ],
                            },
                        }
                    ],
                    "stop_after_first": True,
                },
            ],
        },
        [
            ("<think>[any_text]</think>[any_text]", True),
            ("<think>[any_text]</think>[any_text]<｜tool▁calls▁begin｜><｜tool▁calls▁end｜>", True),
            (
                """<think>[any_text]</think>[any_text]<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1
```json
{"arg": 10}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2
```json
{"arg": 10}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>""",
                True,
            ),
            (
                """<think>[any_text]</think>[any_text]<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_3
```json
{"arg": 10}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>""",
                False,
            ),
            (
                """<think>[any_text]</think>[any_text]<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2
```json
{"arg": 10}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>[any_text]""",
                False,
            ),
        ],
    ),
    # Force non-think mode
    (
        {
            "type": "sequence",
            "elements": [
                {"type": "const_string", "value": "<think></think>"},
                {
                    "type": "triggered_tags",
                    "triggers": ["<tool_call>"],
                    "tags": [
                        {
                            "begin": '<tool_call>\n{"name": "func1", "arguments": ',
                            "content": {"type": "json_schema", "json_schema": {"type": "object"}},
                            "end": "}\n</tool_call>",
                        },
                        {
                            "begin": '<tool_call>\n{"name": "func2", "arguments": ',
                            "content": {"type": "json_schema", "json_schema": {"type": "object"}},
                            "end": "}\n</tool_call>",
                        },
                    ],
                },
            ],
        },
        [
            (
                '<think></think>[any_text]<tool_call>\n{"name": "func1", "arguments": {"arg": 10}}\n</tool_call>[any_text]',
                True,
            ),
            (
                '<think>abcd</think>[any_text]<tool_call>\n{"name": "func1", "arguments": {"arg": 10}}\n</tool_call>[any_text]',
                False,
            ),
        ],
    ),
]


@pytest.mark.parametrize(
    "stag_format, instance_is_accepted_tuples", compound_stag_instance_is_accepted
)
def test_compound_format(
    stag_format: Dict[str, Any], instance_is_accepted_tuples: List[Tuple[str, bool]]
):
    for instance, is_accepted in instance_is_accepted_tuples:
        check_stag_with_instance(stag_format, instance, is_accepted)


end_string_detector_test_data = [
    (
        {
            "type": "tag",
            "begin": "<start>",
            "content": {
                "type": "sequence",
                "elements": [{"type": "const_string", "value": "[TEXT]"}, {"type": "any_text"}],
            },
            "end": "<end>",
        },
        r"""const_string ::= (("[TEXT]"))
any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<end>"),
  loop_after_dispatch=false,
  excludes=()
)
sequence ::= ((const_string any_text))
tag ::= (("<start>" sequence))
root ::= ((tag))
""",
        [
            ("<start>[TEXT]<end>", True),
            ("<start>[TEXT]abcde<end>", True),
            ("<start>[TEXT]abcde", False),
            ("<start><end>", False),
        ],
    ),
    (
        # Detect the end string for nested structures
        {
            "type": "tag",
            "begin": "<start>",
            "content": {
                "type": "or",
                "elements": [
                    {
                        "type": "triggered_tags",
                        "triggers": ["<start2"],
                        "tags": [
                            {"begin": "<start2>", "content": {"type": "any_text"}, "end": "<end2>"}
                        ],
                        "at_least_one": True,
                    },
                    {
                        "type": "sequence",
                        "elements": [
                            {"type": "const_string", "value": "[TEXT2]"},
                            {"type": "any_text"},
                        ],
                    },
                    {
                        "type": "tags_with_separator",
                        "tags": [
                            {"begin": "<start3>", "content": {"type": "any_text"}, "end": "<end3>"}
                        ],
                        "separator": "<sep>",
                    },
                ],
            },
            "end": "<end>",
        },
        r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<end2>"),
  loop_after_dispatch=false,
  excludes=()
)
triggered_tags_group ::= ((">" any_text))
triggered_tags_first ::= (("<start2>" any_text))
triggered_tags_sub ::= TagDispatch(
  ("<start2", triggered_tags_group),
  stop_eos=false,
  stop_str=("<end>"),
  loop_after_dispatch=true,
  excludes=()
)
triggered_tags ::= ((triggered_tags_first triggered_tags_sub))
const_string ::= (("[TEXT2]"))
any_text_1 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<end>"),
  loop_after_dispatch=false,
  excludes=()
)
sequence ::= ((const_string any_text_1))
any_text_2 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<end3>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<start3>" any_text_2))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= (("<sep>" tags_with_separator_tags tags_with_separator_sub) | ("<end>"))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub) | ("<end>"))
or ::= ((triggered_tags) | (sequence) | (tags_with_separator))
tag_1 ::= (("<start>" or))
root ::= ((tag_1))
""",
        [
            ("<start><start2>[TEXT]<end2><end>", True),
            ("<start><start2><end2><end>", True),
            ("<start>[TEXT2]abc<end>", True),
            ("<start><start3>abc<end3><end>", True),
            ("<start><start3><end3><end>", True),
            ("<start><end>", True),
            ("<start>[TEXT2]", False),
        ],
    ),
    (
        # Also in nested structures, but none end string can be detected
        {
            "type": "or",
            "elements": [
                {
                    "type": "triggered_tags",
                    "triggers": ["<start2"],
                    "tags": [
                        {"begin": "<start2>", "content": {"type": "any_text"}, "end": "<end2>"}
                    ],
                    "at_least_one": True,
                },
                {
                    "type": "sequence",
                    "elements": [{"type": "const_string", "value": "[TEXT]"}, {"type": "any_text"}],
                },
                {
                    "type": "or",
                    "elements": [
                        {
                            "type": "tags_with_separator",
                            "tags": [
                                {
                                    "begin": "<start3>",
                                    "content": {"type": "any_text"},
                                    "end": "<end3>",
                                }
                            ],
                            "separator": "<sep>",
                            "at_least_one": True,
                        },
                        {
                            "type": "sequence",
                            "elements": [
                                {"type": "const_string", "value": "[TEXT2]"},
                                {"type": "any_text"},
                            ],
                        },
                    ],
                },
            ],
        },
        r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("<end2>"),
  loop_after_dispatch=false,
  excludes=()
)
triggered_tags_group ::= ((">" any_text))
triggered_tags_first ::= (("<start2>" any_text))
triggered_tags_sub ::= TagDispatch(
  ("<start2", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
triggered_tags ::= ((triggered_tags_first triggered_tags_sub))
const_string ::= (("[TEXT]"))
any_text_1 ::= (([\0-\U0010ffff]*))
sequence ::= ((const_string any_text_1))
any_text_2 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<end3>"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("<start3>" any_text_2))
tags_with_separator_tags ::= ((tag))
tags_with_separator_sub ::= ("" | ("<sep>" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ((tags_with_separator_tags tags_with_separator_sub))
const_string_1 ::= (("[TEXT2]"))
any_text_3 ::= (([\0-\U0010ffff]*))
sequence_1 ::= ((const_string_1 any_text_3))
or ::= ((tags_with_separator) | (sequence_1))
or_1 ::= ((triggered_tags) | (sequence) | (or))
root ::= ((or_1))
""",
        [
            ("<start2>abc<end2>abcdef", True),
            ("[TEXT]abc", True),
            ("[TEXT]", True),
            ("<start3>abc<end3>", True),
            ("<start3>abc<end3><sep><start3>def<end3>", True),
            ("[TEXT2]def", True),
            ("[TEXT2]", True),
            ("<start>abc<end>", False),
            ("<start2>abc", False),
            ("abc<end2>", False),
            ("<start3>abc", False),
            ("<start3>abc<end3><start3>def<end3>", False),
            ("random text", False),
        ],
    ),
]


@pytest.mark.parametrize(
    "stag_format, expected_grammar, instance_is_accepted_tuples", end_string_detector_test_data
)
def test_end_string_detector(
    stag_format: Dict[str, Any],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    check_stag_with_grammar(stag_format, expected_grammar)
    for instance, is_accepted in instance_is_accepted_tuples:
        check_stag_with_instance(stag_format, instance, is_accepted)


# Test cases for JSON format and parsing errors (need string input)
json_format_error_test_data = [
    # JSON Parsing Errors
    (
        '{"type": "structural_tag", "format": {"type": "const_string", "value": "hello"',
        "Failed to parse JSON",
    ),
    ('"not_an_object"', "Structural tag must be an object"),
    (
        '{"type": "wrong_type", "format": {"type": "const_string", "value": "hello"}}',
        'Structural tag\'s type must be a string "structural_tag"',
    ),
    ('{"type": "structural_tag"}', "Structural tag must have a format field"),
    # Format Parsing Errors
    ('{"type": "structural_tag", "format": "not_an_object"}', "Format must be an object"),
    (
        '{"type": "structural_tag", "format": {"type": 123, "value": "hello"}}',
        "Format's type must be a string",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "unknown_format"}}',
        "Format type not recognized: unknown_format",
    ),
    ('{"type": "structural_tag", "format": {"invalid_field": "value"}}', "Invalid format"),
    # ConstStringFormat Errors
    (
        '{"type": "structural_tag", "format": {"type": "const_string"}}',
        "ConstString format must have a value field with a non-empty string",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "const_string", "value": 123}}',
        "ConstString format must have a value field with a non-empty string",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "const_string", "value": ""}}',
        "ConstString format must have a value field with a non-empty string",
    ),
    # JSONSchemaFormat Errors
    (
        '{"type": "structural_tag", "format": {"type": "json_schema"}}',
        "JSON schema format must have a json_schema field with a object or boolean value",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "json_schema", "json_schema": "invalid"}}',
        "JSON schema format must have a json_schema field with a object or boolean value",
    ),
    # SequenceFormat Errors
    (
        '{"type": "structural_tag", "format": {"type": "sequence"}}',
        "Sequence format must have an elements field with an array",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "sequence", "elements": "not_array"}}',
        "Sequence format must have an elements field with an array",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "sequence", "elements": []}}',
        "Sequence format must have at least one element",
    ),
    # OrFormat Errors
    (
        '{"type": "structural_tag", "format": {"type": "or"}}',
        "Or format must have an elements field with an array",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "or", "elements": "not_array"}}',
        "Or format must have an elements field with an array",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "or", "elements": []}}',
        "Or format must have at least one element",
    ),
    # TagFormat Errors
    (
        '{"type": "structural_tag", "format": {"type": "tag", "content": {"type": "const_string", "value": "hello"}, "end": "end"}}',
        "Tag format's begin field must be a string",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "tag", "begin": 123, "content": {"type": "const_string", "value": "hello"}, "end": "end"}}',
        "Tag format's begin field must be a string",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "tag", "begin": "start", "end": "end"}}',
        "Tag format must have a content field",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "tag", "begin": "start", "content": {"type": "const_string", "value": "hello"}}}',
        "Tag format must have an end field",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "tag", "begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": 123}}',
        "Tag format's end field must be a string or array of strings",
    ),
    # TriggeredTagsFormat Errors
    (
        '{"type": "structural_tag", "format": {"type": "triggered_tags", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}]}}',
        "Triggered tags format must have a triggers field with an array",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": "not_array", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}]}}',
        "Triggered tags format must have a triggers field with an array",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": [], "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}]}}',
        "Triggered tags format's triggers must be non-empty",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": [123], "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}]}}',
        "Triggered tags format's triggers must be non-empty strings",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": [""], "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}]}}',
        "Triggered tags format's triggers must be non-empty strings",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["trigger"]}}',
        "Triggered tags format must have a tags field with an array",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["trigger"], "tags": "not_array"}}',
        "Triggered tags format must have a tags field with an array",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["trigger"], "tags": []}}',
        "Triggered tags format's tags must be non-empty",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["trigger"], "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}], "at_least_one": "not_boolean"}}',
        "at_least_one must be a boolean",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["trigger"], "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}], "stop_after_first": "not_boolean"}}',
        "stop_after_first must be a boolean",
    ),
    # TagsWithSeparatorFormat Errors
    (
        '{"type": "structural_tag", "format": {"type": "tags_with_separator", "separator": "sep"}}',
        "Tags with separator format must have a tags field with an array",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": "not_array", "separator": "sep"}}',
        "Tags with separator format must have a tags field with an array",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [], "separator": "sep"}}',
        "Tags with separator format's tags must be non-empty",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}]}}',
        "Tags with separator format's separator field must be a string",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}], "separator": 123}}',
        "Tags with separator format's separator field must be a string",
    ),
    # Note: empty separator is now valid, so no error test for it
    (
        '{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}], "separator": "sep", "at_least_one": "not_boolean"}}',
        "at_least_one must be a boolean",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}], "separator": "sep", "stop_after_first": "not_boolean"}}',
        "stop_after_first must be a boolean",
    ),
]


@pytest.mark.parametrize("json_input, expected_error", json_format_error_test_data)
def test_structural_tag_json_format_errors(json_input: str, expected_error: str):
    """Test JSON format and parsing errors that occur during JSON parsing phase"""
    with pytest.raises(Exception) as exc_info:
        xgr.Grammar.from_structural_tag(json_input)
    assert expected_error in str(exc_info.value)


structural_tag_error_test_data = [
    # Analyzer Errors - Only last element in sequence can be unlimited
    {
        "type": "sequence",
        "elements": [
            {"type": "const_string", "value": "start"},
            {"type": "any_text"},  # This unlimited element in middle will cause error
            {"type": "const_string", "value": "end"},
        ],
    },
    # Analyzer Errors - Or format with mixed unlimited and limited elements
    {
        "type": "or",
        "elements": [
            {"type": "const_string", "value": "limited"},  # Limited element
            {"type": "any_text"},  # Unlimited element - mix not allowed
        ],
    },
    # Analyzer Errors - Tag format with unlimited content but empty end
    {
        "type": "tag",
        "begin": "start",
        "content": {"type": "any_text"},  # Unlimited content
        "end": "",  # Empty end with unlimited content causes error
    },
    # Converter Errors - Tag matches multiple triggers
    {
        "type": "triggered_tags",
        "triggers": ["A", "AB"],  # Both will match tag beginning with "ABC"
        "tags": [
            {"begin": "ABC", "content": {"type": "const_string", "value": "hello"}, "end": "end"}
        ],
    },
    # Converter Errors - Tag matches no trigger
    {
        "type": "triggered_tags",
        "triggers": ["X", "Y"],  # Neither matches "ABC" begin
        "tags": [
            {"begin": "ABC", "content": {"type": "const_string", "value": "hello"}, "end": "end"}
        ],
    },
    # Cannot detect end string of tags_with_separator in sequence
    {
        "type": "sequence",
        "elements": [
            {
                "type": "tags_with_separator",
                "tags": [
                    {
                        "begin": "<start>",
                        "content": {"type": "const_string", "value": "[TEXT]"},
                        "end": "<end>",
                    }
                ],
                "separator": "<sep>",
            },
            {"type": "const_string", "value": "[TEXT]"},
        ],
    },
    # Cannot detect end string of tags_with_separator in or
    {
        "type": "or",
        "elements": [
            {
                "type": "tags_with_separator",
                "tags": [
                    {
                        "begin": "<start>",
                        "content": {"type": "const_string", "value": "[TEXT]"},
                        "end": "<end>",
                    }
                ],
                "separator": "<sep>",
            },
            {"type": "const_string", "value": "[TEXT]"},
        ],
    },
    # Original test cases - Detected end string of tags_with_separator is empty
    {
        "type": "tag",
        "begin": "<start>",
        "content": {
            "type": "tags_with_separator",
            "tags": [
                {
                    "begin": "<start2>",
                    "content": {"type": "const_string", "value": "[TEXT]"},
                    "end": "<end2>",
                }
            ],
            "separator": "<sep>",
        },
        "end": "",
    },
]


@pytest.mark.parametrize("stag_format", structural_tag_error_test_data)
def test_structural_tag_error(stag_format: Dict[str, Any]):
    """Test analyzer and converter errors that occur after successful parsing"""
    structural_tag = {"type": "structural_tag", "format": stag_format}
    with pytest.raises(Exception, match="Invalid structural tag error"):
        xgr.Grammar.from_structural_tag(structural_tag)


utf8_stag_format_and_instance_accepted = [
    ({"type": "const_string", "value": "你好"}, "你好", True),
    ({"type": "const_string", "value": "你好"}, "hello", False),
    ({"type": "any_text"}, "😊", True),
    (
        {
            "type": "sequence",
            "elements": [
                {"type": "const_string", "value": "开始"},
                {"type": "json_schema", "json_schema": {"type": "string"}},
                {"type": "const_string", "value": "结束"},
            ],
        },
        '开始"中间"结束',
        True,
    ),
    (
        {
            "type": "sequence",
            "elements": [
                {"type": "const_string", "value": "开始"},
                {"type": "json_schema", "json_schema": {"type": "string"}},
                {"type": "const_string", "value": "结束"},
            ],
        },
        "开始中间内容",
        False,
    ),
    (
        {"type": "tag", "begin": "标签开始", "content": {"type": "any_text"}, "end": "标签结束"},
        "标签开始一些内容标签结束",
        True,
    ),
    (
        {"type": "tag", "begin": "标签开始", "content": {"type": "any_text"}, "end": "标签结束"},
        "标签开始一些内容",
        False,
    ),
    (
        {
            "type": "or",
            "elements": [
                {"type": "const_string", "value": "选项一"},
                {"type": "const_string", "value": "选项二"},
            ],
        },
        "选项一",
        True,
    ),
    (
        {
            "type": "or",
            "elements": [
                {"type": "const_string", "value": "选项一"},
                {"type": "const_string", "value": "选项二"},
            ],
        },
        "选项三",
        False,
    ),
    (
        {
            "type": "tags_with_separator",
            "tags": [{"begin": "项开始", "content": {"type": "any_text"}, "end": "项结束"}],
            "separator": "分隔符",
        },
        "项开始内容1项结束分隔符项开始内容2项结束",
        True,
    ),
    (
        {
            "type": "tags_with_separator",
            "tags": [{"begin": "项开始", "content": {"type": "any_text"}, "end": "项结束"}],
            "separator": "分隔符",
        },
        "项开始内容1项结束项开始内容2项结束",
        False,
    ),
    (
        {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {"字段": {"type": "string"}},
                "required": ["字段"],
                "additionalProperties": False,
            },
        },
        '{"字段": "值"}',
        True,
    ),
    (
        {
            "type": "qwen_xml_parameter",
            "json_schema": {
                "type": "object",
                "properties": {"参数": {"type": "string"}},
                "required": ["参数"],
                "additionalProperties": False,
            },
        },
        "<parameter=参数>值</parameter>",
        True,
    ),
]


@pytest.mark.parametrize(
    "stag_format, instance, is_accepted", utf8_stag_format_and_instance_accepted
)
def test_basic_structural_tag_utf8(stag_format: Dict[str, Any], instance: str, is_accepted: bool):
    """Test structural tag with UTF-8 characters"""
    check_stag_with_instance(stag_format, instance, is_accepted)


basic_structural_tags_instance_is_accepted = [
    # ConstStringFormat
    (xgr.structural_tag.ConstStringFormat(value="hello"), "hello", True),
    (xgr.structural_tag.ConstStringFormat(value="hello"), "hello world", False),
    # JSONSchemaFormat
    (xgr.structural_tag.JSONSchemaFormat(json_schema={"type": "object"}), '{"key": "value"}', True),
    (xgr.structural_tag.JSONSchemaFormat(json_schema={"type": "string"}), '"abc"', True),
    (xgr.structural_tag.JSONSchemaFormat(json_schema={"type": "integer"}), "123", True),
    (xgr.structural_tag.JSONSchemaFormat(json_schema={"type": "integer"}), "abc", False),
    # AnyTextFormat
    (xgr.structural_tag.AnyTextFormat(), "", True),
    (xgr.structural_tag.AnyTextFormat(), "any text here", True),
    # SequenceFormat
    (
        xgr.structural_tag.SequenceFormat(
            elements=[
                xgr.structural_tag.ConstStringFormat(value="A"),
                xgr.structural_tag.ConstStringFormat(value="B"),
            ]
        ),
        "AB",
        True,
    ),
    (
        xgr.structural_tag.SequenceFormat(
            elements=[
                xgr.structural_tag.ConstStringFormat(value="A"),
                xgr.structural_tag.ConstStringFormat(value="B"),
            ]
        ),
        "A",
        False,
    ),
    # OrFormat
    (
        xgr.structural_tag.OrFormat(
            elements=[
                xgr.structural_tag.ConstStringFormat(value="A"),
                xgr.structural_tag.ConstStringFormat(value="B"),
            ]
        ),
        "A",
        True,
    ),
    (
        xgr.structural_tag.OrFormat(
            elements=[
                xgr.structural_tag.ConstStringFormat(value="A"),
                xgr.structural_tag.ConstStringFormat(value="B"),
            ]
        ),
        "B",
        True,
    ),
    (
        xgr.structural_tag.OrFormat(
            elements=[
                xgr.structural_tag.ConstStringFormat(value="A"),
                xgr.structural_tag.ConstStringFormat(value="B"),
            ]
        ),
        "C",
        False,
    ),
    # TagFormat
    (
        xgr.structural_tag.TagFormat(
            begin="<b>", content=xgr.structural_tag.AnyTextFormat(), end="</b>"
        ),
        "<b>text</b>",
        True,
    ),
    (
        xgr.structural_tag.TagFormat(
            begin="<b>", content=xgr.structural_tag.AnyTextFormat(), end="</b>"
        ),
        "<b>text</b",
        False,
    ),
    # TagsWithSeparatorFormat
    (
        xgr.structural_tag.TagsWithSeparatorFormat(
            tags=[
                xgr.structural_tag.TagFormat(
                    begin="<b>", content=xgr.structural_tag.AnyTextFormat(), end="</b>"
                )
            ],
            separator=",",
        ),
        '<b>"1"</b>,<b>"2"</b>',
        True,
    ),
    (
        xgr.structural_tag.TagsWithSeparatorFormat(
            tags=[
                xgr.structural_tag.TagFormat(
                    begin="<b>", content=xgr.structural_tag.AnyTextFormat(), end="</b>"
                )
            ],
            separator=",",
        ),
        '<b>"1"</b><b>"2"</b>',
        False,
    ),
    # QwenXMLParameterFormat
    (
        xgr.structural_tag.QwenXMLParameterFormat(
            json_schema={"type": "object", "properties": {"name": {"type": "string"}}}
        ),
        "<parameter=name>value</parameter>",
        True,
    ),
    (
        xgr.structural_tag.QwenXMLParameterFormat(
            json_schema={"type": "object", "properties": {"name": {"type": "string"}}}
        ),
        "<parameter=name>value</param>",
        False,
    ),
]


@pytest.mark.parametrize(
    "stag_format, instance, is_accepted", basic_structural_tags_instance_is_accepted
)
def test_from_structural_tag_with_structural_tag_instance(
    stag_format: xgr.structural_tag.Format, instance: str, is_accepted: bool
):
    stag = xgr.StructuralTag(format=stag_format)
    check_stag_with_instance(stag, instance, is_accepted)


# ---------- Multiple End Tokens Tests ----------


multiple_end_tokens_tag_stag_grammar = [
    # Test tag with multiple end tokens (limited content)
    (
        {
            "type": "tag",
            "begin": "BEG",
            "content": {"type": "const_string", "value": "CONTENT"},
            "end": ["END1", "END2"],
        },
        r"""const_string ::= (("CONTENT"))
tag_end ::= (("END1") | ("END2"))
tag ::= (("BEG" const_string tag_end))
root ::= ((tag))
""",
    ),
    # Test tag with single end token in array (should work the same as string)
    (
        {
            "type": "tag",
            "begin": "<start>",
            "content": {"type": "const_string", "value": "X"},
            "end": ["</end>"],
        },
        r"""const_string ::= (("X"))
tag ::= (("<start>" const_string "</end>"))
root ::= ((tag))
""",
    ),
]


multiple_end_tokens_instance_is_accepted = [
    ("BEGCONTENTEND1", True),
    ("BEGCONTENTEND2", True),
    ("BEGCONTENTEND3", False),
    ("BEGCONTENTEND", False),
]


@pytest.mark.parametrize("stag_format, expected_grammar", multiple_end_tokens_tag_stag_grammar)
def test_multiple_end_tokens_tag_grammar(stag_format: Dict[str, Any], expected_grammar: str):
    check_stag_with_grammar(stag_format, expected_grammar)


@pytest.mark.parametrize("instance, is_accepted", multiple_end_tokens_instance_is_accepted)
def test_multiple_end_tokens_tag_instance(instance: str, is_accepted: bool):
    stag_format = {
        "type": "tag",
        "begin": "BEG",
        "content": {"type": "const_string", "value": "CONTENT"},
        "end": ["END1", "END2"],
    }
    check_stag_with_instance(stag_format, instance, is_accepted)


# Test multiple end tokens with any_text (unlimited content)
multiple_end_tokens_any_text_stag_grammar = [
    (
        {"type": "tag", "begin": "BEG", "content": {"type": "any_text"}, "end": ["END1", "END2"]},
        r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("END1", "END2"),
  loop_after_dispatch=false,
  excludes=()
)
tag ::= (("BEG" any_text))
root ::= ((tag))
""",
    )
]


multiple_end_tokens_any_text_instance_is_accepted = [
    ("BEGHello!END1", True),
    ("BEGHello!END2", True),
    ("BEGEND1", True),
    ("BEGEND2", True),
    ("BEGsome text hereEND1", True),
    ("BEGsome text hereEND2", True),
    ("BEGHello!END3", False),
    ("BEGHello!END", False),
]


@pytest.mark.parametrize("stag_format, expected_grammar", multiple_end_tokens_any_text_stag_grammar)
def test_multiple_end_tokens_any_text_grammar(stag_format: Dict[str, Any], expected_grammar: str):
    check_stag_with_grammar(stag_format, expected_grammar)


@pytest.mark.parametrize("instance, is_accepted", multiple_end_tokens_any_text_instance_is_accepted)
def test_multiple_end_tokens_any_text_instance(instance: str, is_accepted: bool):
    stag_format = {
        "type": "tag",
        "begin": "BEG",
        "content": {"type": "any_text"},
        "end": ["END1", "END2"],
    }
    check_stag_with_instance(stag_format, instance, is_accepted)


# Test multiple end tokens with one empty string
multiple_end_tokens_with_empty_stag_grammar = [
    # Test tag with one actual end token and one empty string
    (
        {
            "type": "tag",
            "begin": "BEG",
            "content": {"type": "const_string", "value": "CONTENT"},
            "end": ["END1", ""],
        },
        r"""const_string ::= (("CONTENT"))
tag_end ::= ("" | ("END1"))
tag ::= (("BEG" const_string tag_end))
root ::= ((tag))
""",
    ),
    # Test with empty string first
    (
        {
            "type": "tag",
            "begin": "<start>",
            "content": {"type": "const_string", "value": "X"},
            "end": ["", "</end>"],
        },
        r"""const_string ::= (("X"))
tag_end ::= ("" | ("</end>"))
tag ::= (("<start>" const_string tag_end))
root ::= ((tag))
""",
    ),
]


multiple_end_tokens_with_empty_instance_is_accepted = [
    ("BEGCONTENTEND1", True),  # Ends with END1
    ("BEGCONTENT", True),  # Ends with empty string
    ("BEGCONTENTEND2", False),  # Wrong end token
    ("BEGCONTENTEND", False),  # Partial match of END1
]


@pytest.mark.parametrize(
    "stag_format, expected_grammar", multiple_end_tokens_with_empty_stag_grammar
)
def test_multiple_end_tokens_with_empty_grammar(stag_format: Dict[str, Any], expected_grammar: str):
    check_stag_with_grammar(stag_format, expected_grammar)


@pytest.mark.parametrize(
    "instance, is_accepted", multiple_end_tokens_with_empty_instance_is_accepted
)
def test_multiple_end_tokens_with_empty_instance(instance: str, is_accepted: bool):
    stag_format = {
        "type": "tag",
        "begin": "BEG",
        "content": {"type": "const_string", "value": "CONTENT"},
        "end": ["END1", ""],
    }
    check_stag_with_instance(stag_format, instance, is_accepted)


# Test multiple end tokens with Python API
def test_multiple_end_tokens_python_api():
    """Test that TagFormat accepts both str and List[str] for end field"""
    # Test with single string (backward compatible)
    tag1 = xgr.structural_tag.TagFormat(
        begin="<start>", content=xgr.structural_tag.ConstStringFormat(value="content"), end="</end>"
    )
    assert tag1.end == "</end>"

    # Test with list of strings
    tag2 = xgr.structural_tag.TagFormat(
        begin="<start>",
        content=xgr.structural_tag.ConstStringFormat(value="content"),
        end=["</end1>", "</end2>"],
    )
    assert tag2.end == ["</end1>", "</end2>"]

    # Test that both work in StructuralTag
    stag1 = xgr.StructuralTag(format=tag1)
    stag2 = xgr.StructuralTag(format=tag2)

    # Test that the grammars can be created
    grammar1 = xgr.Grammar.from_structural_tag(stag1)
    grammar2 = xgr.Grammar.from_structural_tag(stag2)

    assert grammar1 is not None
    assert grammar2 is not None


# Test error case: empty end array
def test_multiple_end_tokens_empty_array_error():
    """Test that empty end array raises an error"""
    stag_format = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": "BEG",
            "content": {"type": "const_string", "value": "X"},
            "end": [],
        },
    }
    with pytest.raises(Exception) as exc_info:
        xgr.Grammar.from_structural_tag(stag_format)
    assert "empty" in str(exc_info.value).lower()


# Test error case: unlimited content with all empty end strings
def test_multiple_end_tokens_unlimited_empty_error():
    """Test that unlimited content with all empty end strings raises an error"""
    stag_format = {
        "type": "structural_tag",
        "format": {"type": "tag", "begin": "BEG", "content": {"type": "any_text"}, "end": ["", ""]},
    }
    with pytest.raises(Exception) as exc_info:
        xgr.Grammar.from_structural_tag(stag_format)
    assert "non-empty" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()


# ---------- Excludes Tests ----------


test_strings_is_accepted_any_text_excludes = [
    ("This is a test string.", True),
    ("This string contains <end> which is excluded.", False),
    ("Another string with </tag> inside.", False),
    ("A clean string without excluded substrings.", True),
    ("<end> at the beginning.", False),
    ("At the end </tag>.", False),
]


@pytest.mark.parametrize("instance, is_accepted", test_strings_is_accepted_any_text_excludes)
def test_excluded_strings_in_any_text(instance: str, is_accepted: bool):

    stag_format = {
        "type": "tag",
        "content": {"type": "any_text", "excludes": ["<end>", "</tag>"]},
        "begin": "",
        "end": ".",
    }

    expected_grammar = r"""any_text ::= TagDispatch(
  stop_eos=false,
  stop_str=("."),
  loop_after_dispatch=false,
  excludes=("<end>", "</tag>")
)
tag ::= (("" any_text))
root ::= ((tag))
"""

    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, is_accepted)


test_strings_is_accepted_triggered_excludes = [
    ("A", False),
    ("A1", False),
    ("A1L1AB", True),
    ("A1L2A", False),
    ("L1A1L1A", False),
    ("L2A2L2A", False),
    ("A1L1AL1", False),
    ("A1L1AA2L2A", True),
]


@pytest.mark.parametrize("instance, is_accepted", test_strings_is_accepted_triggered_excludes)
def test_excluded_strings_in_triggered_format(instance: str, is_accepted: bool):

    stag_format = {
        "type": "triggered_tags",
        "triggers": ["A"],
        "tags": [
            {"begin": "A1", "content": {"type": "const_string", "value": "L1"}, "end": "A"},
            {"begin": "A2", "content": {"type": "const_string", "value": "L2"}, "end": "A"},
        ],
        "at_least_one": True,
        "stop_after_first": False,
        "excludes": ["L1", "L2"],
    }

    expected_grammar = r"""const_string ::= (("L1"))
const_string_1 ::= (("L2"))
triggered_tags_group ::= (("1" const_string "A") | ("2" const_string_1 "A"))
triggered_tags_first ::= (("A1" const_string "A") | ("A2" const_string_1 "A"))
triggered_tags_sub ::= TagDispatch(
  ("A", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=("L1", "L2")
)
triggered_tags ::= ((triggered_tags_first triggered_tags_sub))
root ::= ((triggered_tags))
"""

    check_stag_with_grammar(stag_format, expected_grammar)
    check_stag_with_instance(stag_format, instance, is_accepted)


const_string_template_values_stag_grammar_instance_accepted = [
    (
        {"type": "const_string", "value": "The value is: {{strings[].value}}."},
        [{"value": "a"}, {"value": "b"}, {"value": "c"}],
        r"""const_string ::= (("The value is: a."))
const_string_1 ::= (("The value is: b."))
const_string_2 ::= (("The value is: c."))
or ::= ((const_string) | (const_string_1) | (const_string_2))
root ::= ((or))
""",
        [
            ("The value is: a.", True),
            ("The value is: b.", True),
            ("The value is: c.", True),
            ("The value is: d.", False),
        ],
    ),
    (
        {"type": "const_string", "value": "{{strings[].value}}"},
        [{"value": "a"}, {"value": "b"}, {"value": "c"}],
        r"""const_string ::= (("a"))
const_string_1 ::= (("b"))
const_string_2 ::= (("c"))
or ::= ((const_string) | (const_string_1) | (const_string_2))
root ::= ((or))
""",
        [("a", True), ("b", True), ("c", True), ("d", False)],
    ),
    (
        {"type": "const_string", "value": "The value is: {{strings[].value}}"},
        [{"value": "a"}, {"value": "b"}, {"value": "c"}],
        r"""const_string ::= (("The value is: a"))
const_string_1 ::= (("The value is: b"))
const_string_2 ::= (("The value is: c"))
or ::= ((const_string) | (const_string_1) | (const_string_2))
root ::= ((or))
""",
        [
            ("The value is: a", True),
            ("The value is: b", True),
            ("The value is: c", True),
            ("The value is: d", False),
        ],
    ),
    (
        {"type": "const_string", "value": "{{strings[].value}}是"},
        [{"value": "a"}, {"value": "b"}, {"value": "c"}],
        r"""const_string ::= (("a\u662f"))
const_string_1 ::= (("b\u662f"))
const_string_2 ::= (("c\u662f"))
or ::= ((const_string) | (const_string_1) | (const_string_2))
root ::= ((or))
""",
        [("a是", True), ("b是", True), ("c是", True), ("d是", False)],
    ),
    (
        {"type": "const_string", "value": "{{strings[].begin}} a dog{{strings[].end}}"},
        [{"begin": "It is", "end": "."}, {"begin": "Is it", "end": "?"}],
        r"""const_string ::= (("It is a dog."))
const_string_1 ::= (("Is it a dog\?"))
or ::= ((const_string) | (const_string_1))
root ::= ((or))
""",
        [
            ("It is a dog.", True),
            ("Is it a dog?", True),
            ("It is a dog?", False),
            ("Is it a dog.", False),
        ],
    ),
]


@pytest.mark.parametrize(
    "template_stag_format, template_values, expected_grammar, instance_is_accepted_tuples",
    const_string_template_values_stag_grammar_instance_accepted,
)
def test_const_string_template_values(
    template_stag_format: Dict[str, Any],
    template_values: List[Dict[str, Any]],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    """Test const_string format with template values"""
    check_template_stag_with_grammar(
        template_stag_format, expected_grammar, strings=template_values
    )

    for instance, is_accepted in instance_is_accepted_tuples:
        check_template_stag_with_instance(
            template_stag_format, instance, is_accepted, strings=template_values
        )


def test_const_string_template_values_with_mingled_templates():
    mingled_format = {
        "type": "const_string",
        "value": "{{string_a[].value}} and {{string_b[].value}} are mingled!",
    }
    structural_tag = {"type": "structural_tag", "format": mingled_format}
    with pytest.raises(Exception) as exc_info:
        xgr.Grammar.from_structural_tag_template(
            structural_tag, string_a=[{"value": "1"}], string_b=[{"value": "2"}]
        )
    expected_info = (
        "Invalid structural tag error: Multiple different placeholder names "
        "found in the same string: '{{string_a[].value}} and {{string_b[].value}} "
        "are mingled!'"
    )
    assert str(exc_info.value) == expected_info


json_schema_template_values_stag_grammar_instance_accepted = [
    (
        {"type": "json_schema", "json_schema": "{{schemas[].value}}"},
        [
            {
                "value": r"""{"type":"object", "properties": {"arg": {"type": "string"}}, "required": ["arg"]}"""
            },
            {"value": r"""{"type":"string"}"""},
        ],
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
root ::= (("{" [ \n\t]* "\"arg\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}"))
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
root_1 ::= ((basic_string_1))
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
or ::= ((root) | (root_1))
root_2 ::= ((or))
""",
        [
            ('{"arg": "value"}', True),
            ('{"arg": "another value"}', True),
            ('{"arg": 123}', False),
            ('{"arg": "value", "extra": "field"}', False),
            ('{"arg": "value"', False),
            ('"just a string"', True),
        ],
    )
]


@pytest.mark.parametrize(
    "template_stag_format, template_values, expected_grammar, instance_is_accepted_tuples",
    json_schema_template_values_stag_grammar_instance_accepted,
)
def test_json_schema_template_values(
    template_stag_format: Dict[str, Any],
    template_values: List[Dict[str, Any]],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    """Test json_schema format with template values"""
    check_template_stag_with_grammar(
        template_stag_format, expected_grammar, schemas=template_values
    )

    for instance, is_accepted in instance_is_accepted_tuples:
        check_template_stag_with_instance(
            template_stag_format, instance, is_accepted, schemas=template_values
        )


def test_part_json_schema_template_failure():
    template_format = {"type": "json_schema", "json_schema": r"""{"type": {{types[].value}}}"""}
    structural_tag = {"type": "structural_tag", "format": template_format}
    types = [{"value": "object"}, {"value": "string"}, {"value": "integer"}]
    with pytest.raises(Exception) as exc_info:
        xgr.Grammar.from_structural_tag_template(structural_tag, types=types)
    expected_info = (
        "Invalid structural tag error: JSON schema format must have a json_schema field with "
        "a object or boolean value"
    )
    assert str(exc_info.value) == expected_info


qwen_template_values_stag_grammar_instance_accepted = [
    (
        {"type": "qwen_xml_parameter", "json_schema": "{{schemas[].value}}"},
        [
            {
                "value": r"""{"type":"object", "properties": {"name": {"type": "string"}}, "required": ["name"]}"""
            },
            {
                "value": r"""{"type":"object", "properties": {"age": {"type": "integer"}}, "required": ["age"]}"""
            },
        ],
        r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
xml_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
xml_entity ::= (("&lt;") | ("&gt;") | ("&amp;") | ("&quot;") | ("&apos;"))
xml_string ::= ("" | ([^<>&\0-\x1f\\\r\n] xml_string) | ("\\" xml_escape xml_string) | (xml_entity xml_string)) (=([ \n\t]*))
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
root ::= (([ \n\t]* "<parameter=name>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>"))
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
xml_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
xml_entity_1 ::= (("&lt;") | ("&gt;") | ("&amp;") | ("&quot;") | ("&apos;"))
xml_string_1 ::= ("" | ([^<>&\0-\x1f\\\r\n] xml_string_1) | ("\\" xml_escape_1 xml_string_1) | (xml_entity_1 xml_string_1)) (=([ \n\t]*))
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
root_prop_0 ::= (("0") | (root_prop_0_1 [1-9] [0-9]*))
root_1 ::= (([ \n\t]* "<parameter=age>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>"))
basic_integer_1_1 ::= ("" | ("-"))
basic_number_1_1 ::= ("" | ("-"))
basic_number_2_1 ::= (([0-9] basic_number_2_1) | ([0-9]))
basic_number_3_1 ::= ("" | ("." basic_number_2_1))
basic_number_4_1 ::= ("" | ([+\-]))
basic_number_5_1 ::= (([0-9] basic_number_5_1) | ([0-9]))
basic_number_6_1 ::= ("" | ([eE] basic_number_4_1 basic_number_5_1))
basic_array_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_1 basic_array_1_1))
basic_object_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_1 [ \n\t]* ":" [ \n\t]* basic_any_1 basic_object_1_1))
root_prop_0_1 ::= ("" | ("-"))
basic_number_7_1 ::= (("0") | ([1-9] [0-9]*))
or ::= ((root) | (root_1))
root_2 ::= ((or))
""",
        [
            ("<parameter=name>value</parameter>", True),
            ("<parameter=name>another value</parameter>", True),
            ("<parameter=name>123</parameter>", True),
            ("<parameter=name>value</parameter>", True),
            ("just a string", False),
            ("<parameter=age>25</parameter>", True),
            ("<parameter=age>-5</parameter>", True),
            ("<parameter=age>abc</parameter>", False),
        ],
    )
]


@pytest.mark.parametrize(
    "template_stag_format, template_values, expected_grammar, instance_is_accepted_tuples",
    qwen_template_values_stag_grammar_instance_accepted,
)
def test_qwen_template_values(
    template_stag_format: Dict[str, Any],
    template_values: List[Dict[str, Any]],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    """Test qwen_xml_parameter format with template values"""
    check_template_stag_with_grammar(
        template_stag_format, expected_grammar, schemas=template_values
    )

    for instance, is_accepted in instance_is_accepted_tuples:
        check_template_stag_with_instance(
            template_stag_format, instance, is_accepted, schemas=template_values
        )


def test_part_qwen_template_failure():
    mingled_format = {
        "type": "qwen_xml_parameter",
        "json_schema": r"""{"type": {{types[].value}}}""",
    }
    structural_tag = {"type": "structural_tag", "format": mingled_format}
    types = [{"value": "object"}, {"value": "string"}]
    with pytest.raises(Exception) as exc_info:
        xgr.Grammar.from_structural_tag_template(structural_tag, types=types)
    expected_info = (
        "Invalid structural tag error: Qwen XML Parameter format must have a json_schema field "
        "with a object or boolean value"
    )

    assert str(exc_info.value) == expected_info


regex_template_values_stag_grammar_instance_accepted = [
    (
        {"type": "regex", "pattern": r"{{patterns[].value}}"},
        [{"value": r"123"}, {"value": r"[a-zA-Z]+"}],
        r"""root ::= (("1" "2" "3"))
root_1 ::= ((root_1_1))
root_1_1 ::= (([a-zA-Z] root_1_1) | ([a-zA-Z]))
or ::= ((root) | (root_1))
root_2 ::= ((or))
""",
        [("123", True), ("abc", True), ("123abc", False), ("123abc456", False), ("", False)],
    )
]


@pytest.mark.parametrize(
    "template_stag_format, template_values, expected_grammar, instance_is_accepted_tuples",
    regex_template_values_stag_grammar_instance_accepted,
)
def test_regex_template_values(
    template_stag_format: Dict[str, Any],
    template_values: List[Dict[str, Any]],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    """Test regex format with template values"""
    check_template_stag_with_grammar(
        template_stag_format, expected_grammar, patterns=template_values
    )

    for instance, is_accepted in instance_is_accepted_tuples:
        check_template_stag_with_instance(
            template_stag_format, instance, is_accepted, patterns=template_values
        )


def test_part_regex_template_failure():
    mingled_format = {"type": "grammar", "pattern": r"{{patterns[].value}}!!!"}
    structural_tag = {"type": "structural_tag", "format": mingled_format}
    patterns = [{"value": r"123"}, {"value2": r"[a-zA-Z]+"}]
    with pytest.raises(RuntimeError) as exc_info:
        xgr.Grammar.from_structural_tag_template(structural_tag, patterns=patterns)
    assert exc_info is not None


grammar_template_values_stag_grammar_instance_accepted = [
    (
        {"type": "grammar", "grammar": "{{grammars[].value}}"},
        [{"value": 'root::= "a" | "b"'}, {"value": 'root ::= a+\na::= "c" | "d"'}],
        r"""root ::= (("a") | ("b"))
root_1 ::= ((root_1_1))
a ::= (("c") | ("d"))
root_1_1 ::= ((a root_1_1) | (a))
or ::= ((root) | (root_1))
root_2 ::= ((or))
""",
        [
            ("a", True),
            ("b", True),
            ("c", True),
            ("d", True),
            ("aa", False),
            ("ab", False),
            ("cc", True),
        ],
    )
]


@pytest.mark.parametrize(
    "template_stag_format, template_values, expected_grammar, instance_is_accepted_tuples",
    grammar_template_values_stag_grammar_instance_accepted,
)
def test_grammar_template_values(
    template_stag_format: Dict[str, Any],
    template_values: List[Dict[str, Any]],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    """Test grammar format with template values"""
    check_template_stag_with_grammar(
        template_stag_format, expected_grammar, grammars=template_values
    )

    for instance, is_accepted in instance_is_accepted_tuples:
        check_template_stag_with_instance(
            template_stag_format, instance, is_accepted, grammars=template_values
        )


def test_part_grammar_template_failure():
    format = {"type": "grammar", "grammar": "root ::= {{grammars[].value}}!!!"}
    structural_tag = {"type": "structural_tag", "format": format}
    grammars = [{"value": 'root ::= "a" | "b"'}, {"value": 'root ::= "c" | "d"'}]
    with pytest.raises(RuntimeError) as exc_info:
        xgr.Grammar.from_structural_tag_template(structural_tag, grammars=grammars)
    assert exc_info is not None


any_text_instance_is_accepted_template = [("abc", True), ("好", True)]


@pytest.mark.parametrize("instance, is_accepted", any_text_instance_is_accepted_template)
def test_any_text_compatible(instance: str, is_accepted: bool):
    """Test that AnyTextFormat is compatible with all structural tag formats"""
    any_text_format = {"type": "any_text"}
    dummy_grammars = [{"value": 'root ::= "a" | "b"'}, {"value": 'root ::= "c" | "d"'}]

    expected_grammar = r"""any_text ::= (([\0-\U0010ffff]*))
root ::= ((any_text))
"""

    check_template_stag_with_grammar(any_text_format, expected_grammar, grammars=dummy_grammars)

    check_template_stag_with_instance(
        any_text_format, instance, is_accepted, grammars=dummy_grammars
    )


def test_no_parameter_error():
    format = {"type": "regex", "pattern": "{{patterns[].value}}"}
    structural_tag = {"type": "structural_tag", "format": format}
    grammars = [{"value": 'root ::= "a" | "b"'}, {"value": 'root ::= "c" | "d"'}]
    expected_error_info = "Invalid structural tag error: Placeholder name 'patterns' not found in values, which is required for the template: '{{patterns[].value}}'"

    with pytest.raises(RuntimeError) as exc_info:
        xgr.Grammar.from_structural_tag_template(structural_tag, grammars=grammars)
    assert str(exc_info.value) == expected_error_info


or_template_values_stag_grammar_instance_accepted = [
    (
        {
            "type": "or",
            "elements": [
                {"type": "const_string", "value": "{{strings[].value}}"},
                {"type": "const_string", "value": "{{numbers[].value}}"},
            ],
        },
        {
            "strings": [{"value": "hello"}, {"value": "world"}],
            "numbers": [{"value": "1"}, {"value": "2"}],
        },
        r"""const_string ::= (("hello"))
const_string_1 ::= (("world"))
const_string_2 ::= (("1"))
const_string_3 ::= (("2"))
or ::= ((const_string) | (const_string_1) | (const_string_2) | (const_string_3))
root ::= ((or))
""",
        [
            ("hello", True),
            ("world", True),
            ("1", True),
            ("2", True),
            ("3", False),
            ("hello world", False),
        ],
    )
]


@pytest.mark.parametrize(
    "template_stag_format, template_values, expected_grammar, instance_is_accepted_tuples",
    or_template_values_stag_grammar_instance_accepted,
)
def test_or_template_values(
    template_stag_format: Dict[str, Any],
    template_values: Dict[str, List[Dict[str, Any]]],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    """Test grammar format with template values"""
    check_template_stag_with_grammar(template_stag_format, expected_grammar, **template_values)

    for instance, is_accepted in instance_is_accepted_tuples:
        check_template_stag_with_instance(
            template_stag_format, instance, is_accepted, **template_values
        )


sequence_template_values_stag_grammar_instance_accepted = [
    (
        {
            "type": "sequence",
            "elements": [
                {"type": "const_string", "value": "{{first[].value}}"},
                {"type": "const_string", "value": "{{second[].value}}"},
            ],
        },
        {
            "first": [{"value": "I'm "}, {"value": "You're "}],
            "second": [{"value": "Alice"}, {"value": "Bob"}],
        },
        r"""const_string ::= (("I\'m "))
const_string_1 ::= (("You\'re "))
or ::= ((const_string) | (const_string_1))
const_string_2 ::= (("Alice"))
const_string_3 ::= (("Bob"))
or_1 ::= ((const_string_2) | (const_string_3))
sequence ::= ((or or_1))
root ::= ((sequence))
""",
        [
            ("I'm Alice", True),
            ("You're Bob", True),
            ("I'm Bob", True),
            ("You're Alice", True),
            ("Alice I'm", False),
            ("Bob You're", False),
        ],
    )
]


@pytest.mark.parametrize(
    "template_stag_format, template_values, expected_grammar, instance_is_accepted_tuples",
    sequence_template_values_stag_grammar_instance_accepted,
)
def test_sequence_template_values(
    template_stag_format: Dict[str, Any],
    template_values: Dict[str, List[Dict[str, Any]]],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    """Test sequence format with template values"""
    check_template_stag_with_grammar(template_stag_format, expected_grammar, **template_values)

    for instance, is_accepted in instance_is_accepted_tuples:
        check_template_stag_with_instance(
            template_stag_format, instance, is_accepted, **template_values
        )


tag_template_values_stag_grammar_instance_accepted = [
    (
        {
            "type": "tag",
            "begin": "{{outter[].first}}",
            "content": {
                "type": "or",
                "elements": [
                    {"type": "const_string", "value": "dog"},
                    {"type": "const_string", "value": "cat"},
                ],
            },
            "end": "{{outter[].symbol}}",
        },
        {"outter": [{"first": "It is a ", "symbol": "!"}, {"first": "Is it a ", "symbol": "?"}]},
        r"""const_string ::= (("dog"))
const_string_1 ::= (("cat"))
or ::= ((const_string) | (const_string_1))
tag ::= (("It is a " or "!"))
const_string_2 ::= (("dog"))
const_string_3 ::= (("cat"))
or_1 ::= ((const_string_2) | (const_string_3))
tag_1 ::= (("Is it a " or_1 "\?"))
or_2 ::= ((tag) | (tag_1))
root ::= ((or_2))
""",
        [
            ("It is a dog!", True),
            ("Is it a cat?", True),
            ("It is a cat!", True),
            ("Is it a dog?", True),
            ("It is a dog?", False),
            ("It is a cat?", False),
            ("Is it a dog!", False),
            ("Is it a cat!", False),
        ],
    )
]


@pytest.mark.parametrize(
    "template_stag_format, template_values, expected_grammar, instance_is_accepted_tuples",
    tag_template_values_stag_grammar_instance_accepted,
)
def test_tag_template_values(
    template_stag_format: Dict[str, Any],
    template_values: Dict[str, List[Dict[str, Any]]],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    """Test tag format with template values"""
    check_template_stag_with_grammar(template_stag_format, expected_grammar, **template_values)

    for instance, is_accepted in instance_is_accepted_tuples:
        check_template_stag_with_instance(
            template_stag_format, instance, is_accepted, **template_values
        )


def test_mingled_tag_template():
    format = {
        "type": "tag",
        "begin": "{{outter[].first}}",
        "content": {
            "type": "or",
            "elements": [{"type": "const_string", "value": "{{inner[].animal}}"}],
        },
        "end": "{{inner[].animal}}",
    }
    structural_tag = {"type": "structural_tag", "format": format}
    outter = [{"first": "It is a "}, {"first": "Is it a "}]
    inner = [{"animal": "dog"}, {"animal": "cat"}]
    with pytest.raises(Exception) as exc_info:
        xgr.Grammar.from_structural_tag_template(structural_tag, outter=outter, inner=inner)
    expected_error_info = "Invalid structural tag error: Mingled placeholder names found, which indicates that there is a product of placeholders, which is ambiguous to expand."
    assert str(exc_info.value) == expected_error_info


triggered_tag_template_values_stag_grammar_instance_accepted = [
    (
        {
            "type": "triggered_tags",
            "triggers": ["I"],
            "tags": [
                {
                    "type": "tag",
                    "begin": "{{outter[].first}}",
                    "content": {
                        "type": "or",
                        "elements": [
                            {"type": "const_string", "value": "dog"},
                            {"type": "const_string", "value": "cat"},
                        ],
                    },
                    "end": "{{outter[].symbol}}",
                }
            ],
            "at_least_one": False,
            "stop_after_first": False,
        },
        {"outter": [{"first": "It is a ", "symbol": "!"}, {"first": "Is it a ", "symbol": "?"}]},
        r"""const_string ::= (("dog"))
const_string_1 ::= (("cat"))
or ::= ((const_string) | (const_string_1))
const_string_2 ::= (("dog"))
const_string_3 ::= (("cat"))
or_1 ::= ((const_string_2) | (const_string_3))
triggered_tags_group ::= (("t is a " or "!") | ("s it a " or_1 "\?"))
triggered_tags ::= TagDispatch(
  ("I", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
""",
        [
            ("It is a dog!", True),
            ("Is it a cat?", True),
            ("It is a cat!", True),
            ("Is it a dog?", True),
            ("It is a dog?", False),
            ("It is a cat?", False),
            ("Is it a dog!", False),
            ("Is it a cat!", False),
            ("Hello world", True),
            ("I am happy", False),
        ],
    )
]


@pytest.mark.parametrize(
    "template_stag_format, template_values, expected_grammar, instance_is_accepted_tuples",
    triggered_tag_template_values_stag_grammar_instance_accepted,
)
def test_triggered_tag_template_values(
    template_stag_format: Dict[str, Any],
    template_values: Dict[str, List[Dict[str, Any]]],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    """Test triggered_tags format with template values"""
    check_template_stag_with_grammar(template_stag_format, expected_grammar, **template_values)

    for instance, is_accepted in instance_is_accepted_tuples:
        check_template_stag_with_instance(
            template_stag_format, instance, is_accepted, **template_values
        )


tag_with_separator_template_values_stag_grammar_instance_accepted = [
    (
        {
            "type": "tags_with_separator",
            "tags": [
                {
                    "type": "tag",
                    "begin": "{{outter[].first}}",
                    "content": {
                        "type": "or",
                        "elements": [
                            {"type": "const_string", "value": "dog"},
                            {"type": "const_string", "value": "cat"},
                        ],
                    },
                    "end": "{{outter[].symbol}}",
                }
            ],
            "separator": "\n",
            "at_least_one": False,
            "stop_after_first": False,
        },
        {"outter": [{"first": "It is a ", "symbol": "!"}, {"first": "Is it a ", "symbol": "?"}]},
        r"""const_string ::= (("dog"))
const_string_1 ::= (("cat"))
or ::= ((const_string) | (const_string_1))
tag ::= (("It is a " or "!"))
const_string_2 ::= (("dog"))
const_string_3 ::= (("cat"))
or_1 ::= ((const_string_2) | (const_string_3))
tag_1 ::= (("Is it a " or_1 "\?"))
tags_with_separator_tags ::= ((tag) | (tag_1))
tags_with_separator_sub ::= ("" | ("\n" tags_with_separator_tags tags_with_separator_sub))
tags_with_separator ::= ("" | (tags_with_separator_tags tags_with_separator_sub))
root ::= ((tags_with_separator))
""",
        [
            ("It is a dog!", True),
            ("Is it a cat?", True),
            ("It is a cat!", True),
            ("Is it a dog?", True),
            ("It is a dog!\nIs it a cat?", True),
            ("It is a cat!\nIt is a dog!", True),
            ("Is it a dog?\nIt is a cat!", True),
            ("Is it a cat?\nIs it a dog?\nIs it a cat?", True),
            ("It is a dog?", False),
            ("It is a cat!\nIt is a dog?", False),
            ("Is it a dog!", False),
        ],
    )
]


@pytest.mark.parametrize(
    "template_stag_format, template_values, expected_grammar, instance_is_accepted_tuples",
    tag_with_separator_template_values_stag_grammar_instance_accepted,
)
def test_tag_with_separator_template_values(
    template_stag_format: Dict[str, Any],
    template_values: Dict[str, List[Dict[str, Any]]],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    """Test tags_with_separator format with template values"""
    check_template_stag_with_grammar(template_stag_format, expected_grammar, **template_values)

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
root ::= (("{" [ \n\t]* "\"operation\"" [ \n\t]* ":" [ \n\t]* root_prop_0 root_part_0 [ \n\t]* "}"))
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
triggered_tags_group ::= (("\"Calculator\", \"parameters\": " root "}") | ("\"Weather\", \"parameters\": " root_1 "}"))
triggered_tags ::= TagDispatch(
  ("{\"name\": ", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root_2 ::= ((triggered_tags))
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
root ::= (("{" [ \n\t]* "\"operation\"" [ \n\t]* ":" [ \n\t]* root_prop_0 root_part_0 [ \n\t]* "}"))
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
triggered_tags_group ::= (("Calculator<|tool_call_argument_begin|>" root "<|tool_call_end|>") | ("Weather<|tool_call_argument_begin|>" root_1 "<|tool_call_end|>"))
triggered_tags ::= TagDispatch(
  ("<|tool_call_begin|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root_2 ::= ((triggered_tags))
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
root ::= (("{" [ \n\t]* "\"operation\"" [ \n\t]* ":" [ \n\t]* root_prop_0 root_part_0 [ \n\t]* "}"))
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
triggered_tags_group ::= (("Calculator<\uff5ctool\u2581sep\uff5c>" root "<\uff5ctool\u2581call\u2581end\uff5c>") | ("Weather<\uff5ctool\u2581sep\uff5c>" root_1 "<\uff5ctool\u2581call\u2581end\uff5c>"))
triggered_tags ::= TagDispatch(
  ("<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root_2 ::= ((triggered_tags))
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
xml_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
xml_entity ::= (("&lt;") | ("&gt;") | ("&amp;") | ("&quot;") | ("&apos;"))
xml_string ::= ("" | ([^<>&\0-\x1f\\\r\n] xml_string) | ("\\" xml_escape xml_string) | (xml_entity xml_string)) (=([ \n\t]*))
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
root ::= (([ \n\t]* "<parameter=operation>" [ \n\t]* root_prop_0 [ \n\t]* "</parameter>" root_part_0))
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
xml_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
xml_entity_1 ::= (("&lt;") | ("&gt;") | ("&amp;") | ("&quot;") | ("&apos;"))
xml_string_1 ::= ("" | ([^<>&\0-\x1f\\\r\n] xml_string_1) | ("\\" xml_escape_1 xml_string_1) | (xml_entity_1 xml_string_1)) (=([ \n\t]*))
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
triggered_tags_group ::= (("Calculator>" root "</function>") | ("Weather>" root_1 "</function>"))
triggered_tags ::= TagDispatch(
  ("<function=", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root_2 ::= ((triggered_tags))
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
root ::= (("{" [ \n\t]* "\"operation\"" [ \n\t]* ":" [ \n\t]* root_prop_0 root_part_0 [ \n\t]* "}"))
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
triggered_tags_group ::= (("{\"name\": \"Calculator\", \"arguments\": " root "}</tool_call>") | ("{\"name\": \"Weather\", \"arguments\": " root_1 "}</tool_call>"))
triggered_tags ::= TagDispatch(
  ("<tool_call>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root_2 ::= ((triggered_tags))
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
root ::= (("{" [ \n\t]* "\"operation\"" [ \n\t]* ":" [ \n\t]* root_prop_0 root_part_0 [ \n\t]* "}"))
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
triggered_tags_group ::= (("assistant<|channel|>analysis<|message|>" any_text "") | ("assistant<|channel|>final<|message|>" any_text_1 "") | ("assistant<|channel|>final<|message|>" any_text_2 "") | ("assistant<|channel|>commentary to=Calculator<|constrain|>json<|message|>" root "<|end|>") | ("assistant<|channel|>commentary to=Weather<|constrain|>json<|message|>" root_1 "<|end|>") | ("assistant<|channel|>analysis to=Python<|message|>" root_2 "<|end|>"))
triggered_tags ::= TagDispatch(
  ("<|start|>", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true,
  excludes=()
)
root_3 ::= ((triggered_tags))
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


array_and_none_array_elements_template_value_grammar_instance_accepted = [
    (
        {
            "type": "structural_tag",
            "format": {
                "type": "tag",
                "begin": "{{calling.begin}}",
                "end": "{{calling.end}}",
                "content": {"type": "json_schema", "json_schema": "{{calling.functions[]}}"},
            },
        },
        {
            "calling": {
                "begin": "<calling>",
                "end": "</calling>",
                "functions": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}],
            }
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
root ::= ((basic_string))
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
root_1 ::= ((basic_number_8))
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
root_2 ::= ((basic_boolean_2))
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
or ::= ((root) | (root_1) | (root_2))
tag ::= (("<calling>" or "</calling>"))
root_3 ::= ((tag))
""",
        [
            ('<calling>"Hello, World!"</calling>', True),
            ("<calling>42</calling>", True),
            ("<calling>true</calling>", True),
            ("<calling>[1, 2, 3]</calling>", False),
            ('<calling>{"key": "value"}</calling>', False),
        ],
    )
]


@pytest.mark.parametrize(
    "template_stag_format, template_values, expected_grammar, instance_is_accepted_tuples",
    array_and_none_array_elements_template_value_grammar_instance_accepted,
)
def test_array_and_none_array_elements(
    template_stag_format: Dict[str, Any],
    template_values: Dict[str, Any],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
    """Test array and none-array elements in template values"""
    check_template_stag_with_grammar(template_stag_format, expected_grammar, **template_values)
    for instance, is_accepted in instance_is_accepted_tuples:
        check_template_stag_with_instance(
            template_stag_format, instance, is_accepted, **template_values
        )


def test_invalid_multiple_array():
    structural_tag_format = {
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": "{{calling.begin[]}}",
            "end": "{{calling.end}}",
            "content": {"type": "json_schema", "json_schema": "{{calling.functions[]}}"},
        },
    }

    template_values = {
        "calling": {
            "begin": ["<calling>", "<calling1>"],
            "end": "</calling>",
            "functions": [{"type": "string"}, {"type": "number"}],
        }
    }

    with pytest.raises(Exception) as exc_info:
        check_template_stag_with_grammar(structural_tag_format, "", **template_values)
    expected_error_info = (
        "Invalid structural tag error: Invalid placeholder structure found in tag format"
    )
    assert str(exc_info.value) == expected_error_info


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
    "template_stag_format, template_values, expected_grammar, instance_is_accepted_tuples",
    nested_array_template_value_grammar_instance_accepted,
)
def test_nested_array_template(
    template_stag_format: Dict[str, Any],
    template_values: Dict[str, Any],
    expected_grammar: str,
    instance_is_accepted_tuples: List[Tuple[str, bool]],
):
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


if __name__ == "__main__":
    pytest.main(sys.argv)
