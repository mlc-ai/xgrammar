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


@pytest.fixture(autouse=True)
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
basic_number ::= ((basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root ::= (("{" [ \n\t]* "\"a\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "}") | ("{" [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | (basic_number_1 [1-9] [0-9]*))
root_1 ::= ((root))
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
xml_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
xml_entity ::= (("&lt;") | ("&gt;") | ("&amp;") | ("&quot;") | ("&apos;"))
xml_string ::= ("" | ([^<>&\0-\x1f\\\r\n] xml_string) | ("\\" xml_escape xml_string) | (xml_entity xml_string)) (=([ \n\t]*))
xml_variable_name ::= (([a-zA-Z_] [a-zA-Z0-9_]*))
xml_string_0 ::= ((xml_string))
xml_any ::= ((basic_number) | (xml_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_prop_1 ::= (("0") | (root_prop_1_1 [1-9] [0-9]*))
root_part_0 ::= (([ \n\t]* "<parameter=age>" [ \n\t]* root_prop_1 [ \n\t]* "</parameter>"))
root ::= (([ \n\t]* "<parameter=name>" [ \n\t]* xml_string_0 [ \n\t]* "</parameter>" root_part_0))
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
basic_number_7 ::= (("0") | (basic_number_1 [1-9] [0-9]*))
root_1 ::= ((root))
""",
    )
]
qwen_parameter_xml_instance_is_accepted = [
    ("<parameter=name>Bob</parameter><parameter=age>\t100\n</parameter>", True),
    ("<parameter=name>Bob</parameter>\t\n<parameter=age>\t100\n</parameter>", True),
    ("<parameter=name>Bob</parameter><parameter=age>100</parameter>", True),
    ("\n\t<parameter=name>Bob</parameter><parameter=age>100</parameter>", True),
    ('<parameter=name>"Bob&lt;"</parameter><parameter=age>100</parameter>', True),
    ("<parameter=name><>Bob</parameter><parameter=age>100</parameter>", False),
    ("<parameter=name>Bob</parameter><parameter=age>100</parameter>\t\t", False),
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
        r"""root ::= (("Hello!" number))
number ::= (([0-9]) | ([0-9] number))
root_1 ::= ((root))
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
        r"""root ::= (("H" "e" "l" "l" "o" "!" root_1))
root_1 ::= (([0-9] root_1) | ([0-9]))
root_2 ::= ((root))
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
basic_number ::= ((basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root ::= ((basic_number))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | (basic_number_1 [1-9] [0-9]*))
root_1 ::= ("" | ([\-+*/]))
root_2 ::= ((root_1_1))
root_1_1 ::= ("" | ([simple]))
sequence ::= ((const_string root root_1 root_2))
root_3 ::= ((sequence))
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
basic_number ::= ((basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root ::= ((basic_number))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | (basic_number_1 [1-9] [0-9]*))
or ::= ((const_string) | (root))
root_1 ::= ((or))
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
basic_number ::= ((basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root ::= ((basic_number))
basic_integer_1 ::= ("" | ("-"))
basic_number_1 ::= ("" | ("-"))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2))
basic_number_4 ::= ("" | ([+\-]))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
basic_array_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any basic_array_1))
basic_object_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1))
basic_number_7 ::= (("0") | (basic_number_1 [1-9] [0-9]*))
tag ::= (("BEG" root "END"))
root_1 ::= ((tag))
""",
    ),
    (
        {
            "type": "tag",
            "begin": "BEG",
            "content": {"type": "grammar", "grammar": "root ::= [+\\-]?[1-9][0-9]*"},
            "end": "END",
        },
        r"""root ::= ((root_1 [1-9] [0-9]*))
root_1 ::= ("" | ([+\-]))
tag ::= (("BEG" root "END"))
root_2 ::= ((tag))
""",
    ),
    (
        {
            "type": "tag",
            "begin": "BEG",
            "content": {"type": "regex", "pattern": "[+\\-]?[1-9][0-9]*"},
            "end": "END",
        },
        r"""root ::= ((root_1 [1-9] [0-9]*))
root_1 ::= ("" | ([+\-]))
tag ::= (("BEG" root "END"))
root_2 ::= ((tag))
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
  loop_after_dispatch=false
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
  loop_after_dispatch=true
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
  loop_after_dispatch=true
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
  loop_after_dispatch=false
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
  loop_after_dispatch=true
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
  loop_after_dispatch=true
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
  loop_after_dispatch=true
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
  loop_after_dispatch=false
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
  loop_after_dispatch=false
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
  loop_after_dispatch=false
)
triggered_tags_group ::= ((">" any_text ""))
triggered_tags_first ::= (("<start2>" any_text ""))
triggered_tags_sub ::= TagDispatch(
  ("<start2", triggered_tags_group),
  stop_eos=false,
  stop_str=("<end>"),
  loop_after_dispatch=true
)
triggered_tags ::= ((triggered_tags_first triggered_tags_sub))
const_string ::= (("[TEXT2]"))
any_text_1 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<end>"),
  loop_after_dispatch=false
)
sequence ::= ((const_string any_text_1))
any_text_2 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<end3>"),
  loop_after_dispatch=false
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
  loop_after_dispatch=false
)
triggered_tags_group ::= ((">" any_text ""))
triggered_tags_first ::= (("<start2>" any_text ""))
triggered_tags_sub ::= TagDispatch(
  ("<start2", triggered_tags_group),
  stop_eos=true,
  stop_str=(),
  loop_after_dispatch=true
)
triggered_tags ::= ((triggered_tags_first triggered_tags_sub))
const_string ::= (("[TEXT]"))
any_text_1 ::= (([\0-\U0010ffff]*))
sequence ::= ((const_string any_text_1))
any_text_2 ::= TagDispatch(
  stop_eos=false,
  stop_str=("<end3>"),
  loop_after_dispatch=false
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
    # AnyTextFormat Errors
    (
        '{"type": "structural_tag", "format": {"type": "any_text", "extra_field": "value"}}',
        "Any text format should not have any fields other than type",
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
        "Tag format's end field must be a string",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "tag", "begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": 123}}',
        "Tag format's end field must be a string",
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
        "Tags with separator format's separator field must be a non-empty string",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}], "separator": 123}}',
        "Tags with separator format's separator field must be a non-empty string",
    ),
    (
        '{"type": "structural_tag", "format": {"type": "tags_with_separator", "tags": [{"begin": "start", "content": {"type": "const_string", "value": "hello"}, "end": "end"}], "separator": ""}}',
        "Tags with separator format's separator field must be a non-empty string",
    ),
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


if __name__ == "__main__":
    pytest.main(sys.argv)
