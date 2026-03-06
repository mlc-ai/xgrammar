import json
import sys
import threading
import time
from typing import List

import pytest
from pydantic import BaseModel
from transformers import AutoTokenizer

import xgrammar as xgr
from xgrammar.testing import _get_masked_tokens_from_bitmask, _is_grammar_accept_string


def test_utf8():
    # Test utf8-encoded string with structural tags
    class Schema(BaseModel):
        arg1: str
        arg2: int

    tags = [
        xgr.StructuralTagItem(begin="，，", schema=Schema, end="。"),
        xgr.StructuralTagItem(begin="，！", schema=Schema, end="。。"),
        xgr.StructuralTagItem(begin="，，？", schema=Schema, end="。。。"),
        xgr.StructuralTagItem(begin="｜｜？", schema=Schema, end="｜？｜"),
    ]
    triggers = ["，", "｜｜"]

    grammar = xgr.Grammar.from_structural_tag(tags, triggers)

    accepted_inputs = [
        '这是无用的内容，，{"arg1": "你好，世界！", "arg2": 0}。这是无用的内容',
        '这是无用的内容，！{"arg1": "こんにちは！", "arg2": 1}。。这是无用的内容',
        '这是无用的内容，，？{"arg1": "안녕하세요！", "arg2": 2}。。。这是无用的内容，！{"arg1": "안녕하세요！", "arg2": 3}。。',
        '这是无用的内容｜｜？{"arg1": "။စ်န, ်ပြ！", "arg2": 0}｜？｜｜｜？{"arg1": "။စ်န, ်ပြ", "arg2": 0}｜？｜',
    ]
    for input_str in accepted_inputs:
        assert _is_grammar_accept_string(grammar, input_str, print_time=True)


expected_grammar_test_structural_tag_after_optimization = r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(basic_string_sub))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_string ::= (("\"" basic_string_sub)) (=(root_part_0 [ \n\t]* "}"))
root_part_0 ::= (([ \n\t]* "," [ \n\t]* "\"arg2\"" [ \n\t]* ":" [ \n\t]* basic_integer)) (=([ \n\t]* "}"))
root_0 ::= (("{" [ \n\t]* "\"arg1\"" [ \n\t]* ":" [ \n\t]* basic_string root_part_0 [ \n\t]* "}")) (=("</function>"))
basic_integer_1 ::= ("" | ("-")) (=([1-9] [0-9]*))
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(basic_string_sub_1))
basic_string_sub_1 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*))
basic_string_1 ::= (("\"" basic_string_sub_1)) (=(root_part_0_1 [ \n\t]* "}"))
root_part_0_1 ::= (([ \n\t]* "," [ \n\t]* "\"arg2\"" [ \n\t]* ":" [ \n\t]* basic_integer_2)) (=([ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"arg1\"" [ \n\t]* ":" [ \n\t]* basic_string_1 root_part_0_1 [ \n\t]* "}")) (=("</function>"))
basic_integer_1_1 ::= ("" | ("-")) (=([1-9] [0-9]*))
basic_escape_2 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(basic_string_sub_2))
basic_string_sub_2 ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub_2) | ("\\" basic_escape_2 basic_string_sub_2)) (=([ \n\t]* [,}\]:]))
basic_number_9 ::= ((basic_number_1_2 basic_number_7_2 basic_number_3_2 basic_number_6_2)) (=(root_part_0_2 [ \n\t]* "}"))
basic_string_2 ::= (("\"" basic_string_sub_2))
root_prop_1 ::= (("[" [ \n\t]* basic_string_2 root_prop_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
root_part_0_2 ::= (([ \n\t]* "," [ \n\t]* "\"arg4\"" [ \n\t]* ":" [ \n\t]* root_prop_1)) (=([ \n\t]* "}"))
root_2 ::= (("{" [ \n\t]* "\"arg3\"" [ \n\t]* ":" [ \n\t]* basic_number_9 root_part_0_2 [ \n\t]* "}")) (=("</function>"))
basic_number_1_2 ::= ("" | ("-")) (=(basic_number_7_2 basic_number_3_2 basic_number_6_2))
basic_number_2_2 ::= (([0-9] basic_number_2_2) | ([0-9]))
basic_number_3_2 ::= ("" | ("." basic_number_2_2)) (=(basic_number_6_2))
basic_number_4_2 ::= ("" | ([+\-])) (=(basic_number_5_2))
basic_number_5_2 ::= (([0-9] basic_number_5_2) | ([0-9]))
basic_number_6_2 ::= ("" | ([eE] basic_number_4_2 basic_number_5_2))
root_prop_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_2 root_prop_1_1)) (=([ \n\t]* "]"))
basic_number_7_2 ::= (("0") | ([1-9] [0-9]*)) (=(basic_number_3_2 basic_number_6_2))
triggered_tags_group ::= (("1>" root_0 "</function>") | ("2>" root_1 "</function>"))
triggered_tags_group_1 ::= ((">" root_2 "</function>"))
triggered_tags ::= TagDispatch(
  ("<function=f", triggered_tags_group),
  ("<function=g", triggered_tags_group_1),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
"""

expected_grammar_test_structural_tag_before_optimization = r"""basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_any ::= ((basic_number) | (basic_string) | (basic_boolean) | (basic_null) | (basic_array) | (basic_object))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*))
basic_number ::= ((basic_number_1 basic_number_7 basic_number_3 basic_number_6))
basic_string ::= (("\"" basic_string_sub))
basic_boolean ::= (("true") | ("false"))
basic_null ::= (("null"))
basic_array ::= (("[" [ \n\t]* basic_any basic_array_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
basic_object ::= (("{" [ \n\t]* basic_string [ \n\t]* ":" [ \n\t]* basic_any basic_object_1 [ \n\t]* "}") | ("{" [ \n\t]* "}"))
root_part_0 ::= (([ \n\t]* "," [ \n\t]* "\"arg2\"" [ \n\t]* ":" [ \n\t]* basic_integer))
root_0 ::= (("{" [ \n\t]* "\"arg1\"" [ \n\t]* ":" [ \n\t]* basic_string root_part_0 [ \n\t]* "}"))
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
root_part_0_1 ::= (([ \n\t]* "," [ \n\t]* "\"arg2\"" [ \n\t]* ":" [ \n\t]* basic_integer_2))
root_1 ::= (("{" [ \n\t]* "\"arg1\"" [ \n\t]* ":" [ \n\t]* basic_string_1 root_part_0_1 [ \n\t]* "}"))
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
root_prop_1 ::= (("[" [ \n\t]* basic_string_2 root_prop_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]"))
root_part_0_2 ::= (([ \n\t]* "," [ \n\t]* "\"arg4\"" [ \n\t]* ":" [ \n\t]* root_prop_1))
root_2 ::= (("{" [ \n\t]* "\"arg3\"" [ \n\t]* ":" [ \n\t]* basic_number_9 root_part_0_2 [ \n\t]* "}"))
basic_integer_1_2 ::= ("" | ("-"))
basic_number_1_2 ::= ("" | ("-"))
basic_number_2_2 ::= (([0-9] basic_number_2_2) | ([0-9]))
basic_number_3_2 ::= ("" | ("." basic_number_2_2))
basic_number_4_2 ::= ("" | ([+\-]))
basic_number_5_2 ::= (([0-9] basic_number_5_2) | ([0-9]))
basic_number_6_2 ::= ("" | ([eE] basic_number_4_2 basic_number_5_2))
basic_array_1_2 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_any_2 basic_array_1_2))
basic_object_1_2 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_2 [ \n\t]* ":" [ \n\t]* basic_any_2 basic_object_1_2))
root_prop_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_2 root_prop_1_1))
basic_number_7_2 ::= (("0") | ([1-9] [0-9]*))
triggered_tags_group ::= (("1>" root_0 "</function>") | ("2>" root_1 "</function>"))
triggered_tags_group_1 ::= ((">" root_2 "</function>"))
triggered_tags ::= TagDispatch(
  ("<function=f", triggered_tags_group),
  ("<function=g", triggered_tags_group_1),
  loop_after_dispatch=true,
  excludes=()
)
root ::= ((triggered_tags))
"""


def test_structural_tag():
    class Schema1(BaseModel):
        arg1: str
        arg2: int

    class Schema2(BaseModel):
        arg3: float
        arg4: List[str]

    tags = [
        xgr.StructuralTagItem(begin="<function=f1>", schema=Schema1, end="</function>"),
        xgr.StructuralTagItem(begin="<function=f2>", schema=Schema1, end="</function>"),
        xgr.StructuralTagItem(begin="<function=g>", schema=Schema2, end="</function>"),
    ]
    # in real cases, we should use one trigger: "<function=" and dispatch to two tags
    # but here we use two triggers for testing such cases
    triggers = ["<function=f", "<function=g"]

    grammar = xgr.Grammar.from_structural_tag(tags, triggers)
    assert str(grammar) == expected_grammar_test_structural_tag_before_optimization

    accepted_inputs = [
        '<function=f1>{"arg1": "abc", "arg2": 1}</function>',
        '<function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>',
        '<function=f2>{"arg1": "abc", "arg2": 1}</function><function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>',
        'hhhh<function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>haha<function=f1>{"arg1": "abc", "arg2": 1}</function>123',
    ]
    for input in accepted_inputs:
        assert _is_grammar_accept_string(grammar, input, print_time=True)


def test_structural_tag_compiler():
    class Schema1(BaseModel):
        arg1: str
        arg2: int

    class Schema2(BaseModel):
        arg3: float
        arg4: List[str]

    tags = [
        xgr.StructuralTagItem(begin="<function=f1>", schema=Schema1, end="</function>"),
        xgr.StructuralTagItem(begin="<function=f2>", schema=Schema1, end="</function>"),
        xgr.StructuralTagItem(begin="<function=g>", schema=Schema2, end="</function>"),
    ]

    # in real cases, we should use one trigger: "<function=" and dispatch to two tags
    # but here we use two triggers for testing such cases
    triggers = ["<function=f", "<function=g"]

    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo([]))
    compiled_grammar = compiler.compile_structural_tag(tags, triggers)
    assert str(compiled_grammar.grammar) == expected_grammar_test_structural_tag_after_optimization


@pytest.mark.hf_token_required
def test_structural_tag_mask_gen():
    # Define schemas for the test
    class Schema1(BaseModel):
        arg1: str
        arg2: int

    class Schema2(BaseModel):
        arg3: float
        arg4: List[str]

    # Set up grammar from schemas
    tags = [
        xgr.StructuralTagItem(
            begin="<function=f>", schema=json.dumps(Schema1.model_json_schema()), end="</function>"
        ),
        xgr.StructuralTagItem(
            begin="<function=g>", schema=json.dumps(Schema2.model_json_schema()), end="</function>"
        ),
    ]
    triggers = ["<function=f", "<function=g"]

    # Set up tokenizer
    tokenizer_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)

    # Compile grammar and create matcher
    compiler = xgr.GrammarCompiler(tokenizer_info)
    time_start = time.monotonic_ns()
    compiled_grammar = compiler.compile_structural_tag(tags, triggers)
    matcher = xgr.GrammarMatcher(compiled_grammar)
    time_end = time.monotonic_ns()
    print(f"Time to compile grammar and init GrammarMatcher: {(time_end - time_start) / 1e3} us")

    # Test input string
    accepted_input = (
        'hhhh<function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>'
        'haha<function=f>{"arg1": "abc", "arg2": 1}</function>123'
    )
    dont_apply_mask_indices = [
        # fmt: off
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
        77, 78, 119, 120, 121, 122
        # fmt: on
    ]
    input_bytes = accepted_input.encode("utf-8")

    # Set up token bitmask for validation
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    # Process input character by character
    for i, c in enumerate(input_bytes):
        # 1. Test token bitmask generation
        time_start = time.monotonic_ns()
        need_apply = matcher.fill_next_token_bitmask(token_bitmask)
        time_end = time.monotonic_ns()
        print(f"Time to fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")
        assert need_apply == (i not in dont_apply_mask_indices)

        # 2. Verify token bitmask correctness
        rejected_token_ids = _get_masked_tokens_from_bitmask(
            token_bitmask, tokenizer_info.vocab_size
        )
        # This checking does not support non-ascii characters for now
        token_id_for_next_char = tokenizer.convert_tokens_to_ids(chr(c))
        assert token_id_for_next_char not in rejected_token_ids

        # 3. Test character acceptance
        print("Accepting char:", bytes([c]))
        time_start = time.monotonic_ns()
        assert matcher.accept_string(bytes([c]))
        time_end = time.monotonic_ns()
        print(f"Time to accept_token: {(time_end - time_start) / 1e3} us")

    # Final verification - check that EOS token is allowed
    time_start = time.monotonic_ns()
    need_apply = matcher.fill_next_token_bitmask(token_bitmask)
    time_end = time.monotonic_ns()
    assert need_apply == (len(input_bytes) not in dont_apply_mask_indices)
    print(f"Time to fill_next_token_bitmask: {(time_end - time_start) / 1e3} us")
    rejected_token_ids = _get_masked_tokens_from_bitmask(token_bitmask, tokenizer_info.vocab_size)
    assert tokenizer.eos_token_id not in rejected_token_ids


def test_empty_tag_dispatch():
    grammar_str = """root ::= TagDispatch(
  loop_after_dispatch=true
)
"""
    grammar = xgr.Grammar.from_ebnf(grammar_str)
    assert _is_grammar_accept_string(grammar, "any string")
    assert _is_grammar_accept_string(grammar, "")
    assert _is_grammar_accept_string(grammar, "好")

    grammar_with_excludes_str = """root ::= TagDispatch(
  excludes=("end"),
  loop_after_dispatch=true
)
"""

    grammar_with_excludes = xgr.Grammar.from_ebnf(grammar_with_excludes_str)

    assert _is_grammar_accept_string(grammar_with_excludes, "any string")
    assert _is_grammar_accept_string(grammar_with_excludes, "好")
    assert not _is_grammar_accept_string(grammar_with_excludes, "any stringend")
    assert not _is_grammar_accept_string(grammar_with_excludes, "endaaa")


@pytest.mark.hf_token_required
def test_utf8_structural_tag_begin_end():
    model = "deepseek-ai/DeepSeek-V3-0324"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    structures = [
        xgr.StructuralTagItem(begin="<｜tool▁calls▁begin｜>", schema={}, end="<｜tool▁calls▁end｜>")
    ]
    triggers = ["<｜tool▁calls▁begin｜>"]
    _ = compiler.compile_structural_tag(structures, triggers)


@pytest.mark.hf_token_required
def test_pressure_structural_tag():
    model = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1)
    threads = []
    start = "start"
    schema = {"type": "object", "properties": {"arg": {"type": "string"}}}
    end = "end"

    def worker(idx: int):
        tag = xgr.StructuralTagItem(begin=start, schema=schema, end=end)
        triggers = [start]
        stag_grammar = xgr.Grammar.from_structural_tag([tag], triggers)
        start_grammar = xgr.Grammar.from_ebnf("root ::= [a-z] root | [a-z]")
        grammar = start_grammar
        for _ in range(idx):
            grammar = grammar.concat(grammar, start_grammar)
        final_grammar = xgr.Grammar.concat(grammar, stag_grammar)
        _ = compiler.compile_grammar(final_grammar)

    for i in range(128):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


_JSON_BODY_RULES = """\
json_body ::= "{" ws kvs ws "}"
kvs ::= kv ("," ws kv)*
kv ::= ["] key_chars ["] ws ":" ws val
key_chars ::= [a-zA-Z_] [a-zA-Z0-9_]*
val ::= ["] val_chars ["] | [0-9]+ | "true" | "false"
val_chars ::= [a-zA-Z0-9 _.+/=]*
ws ::= [ ]*
"""

_tag_dispatch_perf_scenarios = [
    # S1: Long exclude, normal text (baseline - most tokens hit fast path)
    (
        f"""root ::= TagDispatch(("<tool>", json_body), loop_after_dispatch=true, excludes=("</response>"))
{_JSON_BODY_RULES}""",
        (
            "The quick brown fox jumps over the lazy dog. "
            "Machine learning models have revolutionized natural language processing. "
            "Transformers use self-attention mechanisms to capture long-range dependencies. "
            '<tool>{"action": "search", "query": "test"}'
            "After retrieving results we can summarize the findings effectively. "
            "The experiment showed significant improvements across all benchmarks measured."
        ),
    ),
    # S2: Short exclude "\n\n" (many tokens contain \n -> second_slicing_bitset fails often)
    (
        f"""root ::= TagDispatch(("<tool>", json_body), loop_after_dispatch=true, excludes=("\\n\\n"))
{_JSON_BODY_RULES}""",
        (
            "def hello():\n    print('hello')\n"
            "def world():\n    return 42\n"
            "class Foo:\n    def bar(self):\n        pass\n"
            "x = [i for i in range(10)]\n"
            "result = sum(x)\n"
            '<tool>{"action": "run"}'
            "for item in collection:\n    process(item)\n"
            "logger.info('done')\n"
        ),
    ),
    # S3: Single char exclude "|" (extreme short exclude)
    (
        f"""root ::= TagDispatch(("<A>", json_body), loop_after_dispatch=true, excludes=("|"))
{_JSON_BODY_RULES}""",
        (
            "This is a simple text without any pipe characters in the content. "
            "We keep writing more content to have enough tokens for measurement. "
            "The test validates single character exclude performance overhead. "
            '<A>{"tag": "content", "value": "here"}'
            "More text after the tag to continue the sequence to the end."
        ),
    ),
    # S4: Dense partial match (FSM repeatedly enters exclude prefix -> slow path)
    (
        f"""root ::= TagDispatch(("<tool>", json_body), loop_after_dispatch=true, excludes=("</response>"))
{_JSON_BODY_RULES}""",
        (
            "Check </r value. The </re field is important. See </res tag here. "
            "Use </resp element. Read </respo data carefully. Got </respon value. "
            "The </respons info shows. Almost </response but not quite yet here. "
            "Back to normal text. Check </r again. And </re once more. "
            "Field </res appears. Element </resp shows. Data </respo found."
        ),
    ),
    # S5: 8 tags with shared prefix "<tool_" (deep trie, many tokens fail bitset)
    (
        f"""root ::= TagDispatch(
  ("<tool_calc>", tc), ("<tool_search>", ts), ("<tool_code>", tcode),
  ("<tool_file>", tf), ("<tool_web>", tw), ("<tool_db>", tdb),
  ("<tool_api>", tapi), ("<tool_shell>", tsh),
  loop_after_dispatch=true, excludes=("</end>")
)
tc ::= json_body
ts ::= json_body
tcode ::= json_body
tf ::= json_body
tw ::= json_body
tdb ::= json_body
tapi ::= json_body
tsh ::= json_body
{_JSON_BODY_RULES}""",
        (
            "Here is some text before any tags appear in the output. "
            '<tool_calc>{"expr": "2+3"}'
            "The answer is 5. Now let me search for more information. "
            '<tool_search>{"query": "xgrammar benchmarks"}'
            "Found some results. Let me write code to process them. "
            '<tool_code>{"code": "print hello"}'
            "Code executed successfully. Now checking files on disk. "
            '<tool_file>{"path": "data.csv"}'
            "Data loaded. Processing complete with all results verified."
        ),
    ),
    # S6: 8 exclude strings (more excludes -> more tokens hit substring check)
    (
        f"""root ::= TagDispatch(
  ("<tool>", json_body), loop_after_dispatch=true,
  excludes=("</end>", "STOP", "HALT", "QUIT", "EXIT", "ABORT", "CANCEL", "TERMINATE")
)
{_JSON_BODY_RULES}""",
        (
            "Starting the process now. The system is running fine and stable. "
            "Several steps are being executed in proper sequence order. "
            "Have you seen the strategy that was described in the handbook? "
            '<tool>{"action": "compute", "value": 42}'
            "The tool returned a value. Continuing execution of the pipeline. "
            "After checking everything looks correct and verified properly."
        ),
    ),
    # S7: 10 tag dispatch loop cycles (tests loop overhead)
    (
        f"""root ::= TagDispatch(("<t>", json_body), loop_after_dispatch=true, excludes=("</done>"))
{_JSON_BODY_RULES}""",
        (
            'A<t>{"k": "v1"}B<t>{"k": "v2"}C<t>{"k": "v3"}D<t>{"k": "v4"}E<t>{"k": "v5"}'
            'F<t>{"k": "v6"}G<t>{"k": "v7"}H<t>{"k": "v8"}I<t>{"k": "v9"}J<t>{"k": "v10"}K'
        ),
    ),
    # S8: No tags, pure AnyText (lightest TagDispatch, reference baseline)
    (
        """root ::= TagDispatch(loop_after_dispatch=false, excludes=("</end>"))
""",
        (
            "This is purely text without any tags at all in the output. "
            "We are testing the lightest possible TagDispatch configuration. "
            "No tags are defined, only a single exclude string is present. "
            "This serves as the baseline reference measurement for comparison."
        ),
    ),
    # S9: Sustained exclude boundary (FSM stays in exclude prefix -> all tokens slow path)
    (
        f"""root ::= TagDispatch(("<tool>", json_body), loop_after_dispatch=true, excludes=("</response>"))
{_JSON_BODY_RULES}""",
        (
            "Normal text. </respons</respons</respons</respons</respons"
            "</respons</respons</respons</respons</respons</respons"
            "</respons</respons</respons</respons</respons</respons"
            "Back to normal text after the sustained boundary stress test."
        ),
    ),
]


@pytest.mark.hf_token_required
@pytest.mark.parametrize("ebnf, input_str", _tag_dispatch_perf_scenarios)
def test_tag_dispatch_perf(ebnf, input_str):
    import statistics

    tokenizer_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True, trust_remote_code=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar = xgr.Grammar.from_ebnf(ebnf)
    input_tokens = tokenizer.encode(input_str, add_special_tokens=False)

    warmup, rounds = 2, 10
    compile_times = []
    mask_times_all = []
    accept_times_all = []

    for round_idx in range(warmup + rounds):
        is_measure = round_idx >= warmup
        compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)

        t0 = time.monotonic_ns()
        compiled = compiler.compile_grammar(grammar)
        t1 = time.monotonic_ns()

        if is_measure:
            compile_times.append((t1 - t0) / 1e6)

        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
        round_mask = []
        round_accept = []

        for tok in input_tokens:
            t_m0 = time.monotonic_ns()
            matcher.fill_next_token_bitmask(token_bitmask)
            t_m1 = time.monotonic_ns()

            t_a0 = time.monotonic_ns()
            ok = matcher.accept_token(tok)
            t_a1 = time.monotonic_ns()

            if is_measure:
                round_mask.append((t_m1 - t_m0) / 1e3)
                round_accept.append((t_a1 - t_a0) / 1e3)

            assert ok, f"token {tok} ({repr(tokenizer.decode([tok]))}) rejected"

        if is_measure:
            mask_times_all.append(round_mask)
            accept_times_all.append(round_accept)

    flat_mask = [t for rnd in mask_times_all for t in rnd]
    flat_accept = [t for rnd in accept_times_all for t in rnd]

    print(f"Tokens: {len(input_tokens)}")
    print(
        f"Compile: {statistics.mean(compile_times):.2f} +/- "
        f"{statistics.stdev(compile_times) if len(compile_times) > 1 else 0:.2f} ms"
    )
    print(f"Mask gen: avg={statistics.mean(flat_mask):.2f} us, max={max(flat_mask):.2f} us")
    print(f"Accept token: avg={statistics.mean(flat_accept):.2f} us, max={max(flat_accept):.2f} us")


if __name__ == "__main__":
    pytest.main(sys.argv)
