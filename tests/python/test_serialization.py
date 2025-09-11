# -*- coding: utf-8 -*-
import json
import sys
from typing import Any, List, Tuple

import pytest
from pydantic import BaseModel, RootModel
from transformers import AutoTokenizer  # type: ignore

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string


def construct_grammar():
    """Construct a Grammar object for testing."""
    return xgr.Grammar.from_ebnf(
        """rule1 ::= ([^0-9] rule1) | ""
root_rule ::= rule1 "a"
""",
        root_rule_name="root_rule",
    )


def construct_tokenizer_info():
    """Construct a TokenizerInfo object for testing."""
    return xgr.TokenizerInfo(
        ["1", "212", "a", "A", "b", "一", "-", "aBc", "abc"],
        vocab_type=xgr.VocabType.BYTE_FALLBACK,
        vocab_size=10,
        stop_token_ids=[0, 1],
        add_prefix_space=True,
    )


def construct_compiled_grammar():
    """Construct a CompiledGrammar object for testing."""
    tokenizer_info = construct_tokenizer_info()
    grammar = construct_grammar()
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    return grammar_compiler.compile_grammar(grammar), tokenizer_info


def test_get_serialization_version():
    """Test the version of the serialized JSON string."""
    assert xgr.get_serialization_version() == "v7"


def test_serialize_grammar():
    """Test Grammar serialization produces expected JSON string."""
    grammar = construct_grammar()
    serialized = grammar.serialize_json()
    expected_json = {
        "rules": [["rule1", 4, -1, False], ["root_rule", 8, -1, False]],
        "grammar_expr_data": [0, 5, 8, 12, 14, 18, 21, 24, 28],
        "grammar_expr_indptr": [
            # fmt: off
            1,3,1,48,57,4,1,0,5,2,0,1,3,0,6,2,3,2,4,1,0,0,1,97,5,2,5,6,6,1,7
            # fmt: on
        ],
        "root_rule_id": 1,
        "complete_fsm": None,
        "per_rule_fsms": [],
        "allow_empty_rule_ids": [],
        "optimized": False,
        "__VERSION__": "v7",
    }
    # The fsms are the same one, but the start state and end states are different.
    assert json.loads(serialized) == expected_json


def test_serialize_grammar_exception():
    """Test Grammar serialization produces expected JSON string."""
    expected_json = {
        "rules": [["rule1", 4, 9, True], ["root_rule", 8, -1, False]],
        "grammar_expr_data": [0, 2, 7, 10, 14, 18, 21, 24, 28, 31],
        "grammar_expr_indptr": [
            # fmt: off
            3,0,1,3,1,48,57,4,1,0,5,2,1,2,6,2,0,3,4,1,0,0,1,97,5,2,5,6,6,1,7,5,1,6
            # fmt: on
        ],
        "root_rule_id": 1,
        "allow_empty_rule_ids": [],
        "complete_fsm": None,
        "per_rule_fsms": [],
        "__VERSION__": "v7",
    }

    expected_json["__VERSION__"] = "v1"  # Change version to trigger error
    with pytest.raises(xgr.DeserializeVersionError):
        xgr.Grammar.deserialize_json(json.dumps(expected_json))

    expected_json["__VERSION__"] = "v7"
    expected_json.pop("rules")  # Remove required field to trigger error
    with pytest.raises(xgr.DeserializeFormatError):
        xgr.Grammar.deserialize_json(json.dumps(expected_json))

    with pytest.raises(xgr.InvalidJSONError):
        xgr.Grammar.deserialize_json("not a valid json string")


def test_serialize_grammar_roundtrip():
    """Test Grammar serialization and deserialization roundtrip."""
    original_grammar = construct_grammar()
    serialized = original_grammar.serialize_json()
    recovered_grammar = xgr.Grammar.deserialize_json(serialized)
    serialized_new = recovered_grammar.serialize_json()
    assert serialized == serialized_new


def test_serialize_grammar_functional():
    """Test that deserialized Grammar object functions correctly."""
    original_grammar = construct_grammar()
    serialized = original_grammar.serialize_json()
    recovered_grammar = xgr.Grammar.deserialize_json(serialized)

    # Test functional equivalence by checking string representation
    assert str(original_grammar) == str(recovered_grammar)

    # Test with GrammarMatcher functionality
    tokenizer_info = construct_tokenizer_info()
    compiler = xgr.GrammarCompiler(tokenizer_info)

    compiled_original = compiler.compile_grammar(original_grammar)
    compiled_recovered = compiler.compile_grammar(recovered_grammar)

    matcher_original = xgr.GrammarMatcher(compiled_original)
    matcher_recovered = xgr.GrammarMatcher(compiled_recovered)

    # Test that both matchers accept the same input
    test_input = "aaa"
    assert matcher_original.accept_string(test_input) == matcher_recovered.accept_string(test_input)


def test_serialize_tokenizer_info():
    """Test TokenizerInfo serialization produces expected JSON string."""
    tokenizer_info = construct_tokenizer_info()
    serialized = tokenizer_info.serialize_json()
    expected_json = (
        '{"vocab_type":1,"vocab_size":10,"add_prefix_space":true,'
        '"stop_token_ids":[0,1],"special_token_ids":[9],'
        '"decoded_vocab":["1","212","a","A","b","\\u00e4\\u00b8\\u0080","-","aBc","abc"],'
        '"sorted_decoded_vocab":[[6,"-"],[3,"A"],[2,"a"],[7,"aBc"],[8,"abc"],[4,"b"],[5,"\\u00e4\\u00b8\\u0080"]],'
        '"trie_subtree_nodes_range":[1,2,5,4,5,6,7],'
        '"__VERSION__":"v7"}'
    )
    assert json.loads(serialized) == json.loads(expected_json)


def test_serialize_tokenizer_info_roundtrip():
    """Test TokenizerInfo serialization and deserialization roundtrip."""
    original_tokenizer_info = construct_tokenizer_info()
    serialized = original_tokenizer_info.serialize_json()
    recovered_tokenizer_info = xgr.TokenizerInfo.deserialize_json(serialized)
    serialized_new = recovered_tokenizer_info.serialize_json()
    assert serialized == serialized_new


def test_serialize_tokenizer_info_functional():
    """Test that deserialized TokenizerInfo object functions correctly."""
    original_tokenizer_info = construct_tokenizer_info()
    serialized = original_tokenizer_info.serialize_json()
    recovered_tokenizer_info = xgr.TokenizerInfo.deserialize_json(serialized)

    # Test property equivalence
    assert original_tokenizer_info.vocab_type == recovered_tokenizer_info.vocab_type
    assert original_tokenizer_info.vocab_size == recovered_tokenizer_info.vocab_size
    assert original_tokenizer_info.add_prefix_space == recovered_tokenizer_info.add_prefix_space
    assert original_tokenizer_info.stop_token_ids == recovered_tokenizer_info.stop_token_ids
    assert original_tokenizer_info.special_token_ids == recovered_tokenizer_info.special_token_ids
    assert original_tokenizer_info.decoded_vocab == recovered_tokenizer_info.decoded_vocab

    # Test functional equivalence with GrammarCompiler
    grammar = construct_grammar()

    compiler_original = xgr.GrammarCompiler(original_tokenizer_info)
    compiler_recovered = xgr.GrammarCompiler(recovered_tokenizer_info)

    compiled_original = compiler_original.compile_grammar(grammar)
    compiled_recovered = compiler_recovered.compile_grammar(grammar)

    # Both should produce functional matchers
    matcher_original = xgr.GrammarMatcher(compiled_original)
    matcher_recovered = xgr.GrammarMatcher(compiled_recovered)

    test_input = "aaa"
    assert matcher_original.accept_string(test_input) == matcher_recovered.accept_string(test_input)


def test_serialize_compiled_grammar():
    """Test CompiledGrammar serialization produces expected JSON string. We verify the adaptive
    token mask part separately.
    """
    compiled_grammar, tokenizer_info = construct_compiled_grammar()
    serialized = compiled_grammar.serialize_json()

    expected_json = {
        "grammar": {
            "rules": [["rule1", 4, 9, True], ["root_rule", 8, -1, False]],
            "grammar_expr_data": [0, 2, 7, 10, 14, 18, 21, 24, 28, 31],
            "grammar_expr_indptr": [
                # fmt: off
                3,0,1,3,1,48,57,4,1,0,5,2,1,2,6,2,0,3,4,1,0,0,1,97,5,2,5,6,6,1,7,5,1,6
                # fmt: on
            ],
            "root_rule_id": 1,
            "allow_empty_rule_ids": [0],
            # fmt: off
            "complete_fsm": {
                'data_': [[0, 47, 3], [58, 127, 3], [192, 223, 1], [224, 239, 4], [240, 247, 5], [128, 191, 3], [-2, 0, 2], [128, 191, 1], [128, 191, 4], [-2, 0, 8], [97, 97, 6]],
                'indptr_': [0, 5, 6, 6, 7, 8, 9, 9, 10, 11]
                },
            "per_rule_fsms": [
                [{'data_': [[0, 47, 3], [58, 127, 3], [192, 223, 1], [224, 239, 4], [240, 247, 5], [128, 191, 3], [-2, 0, 2], [128, 191, 1], [128, 191, 4], [-2, 0, 8], [97, 97, 6]],
                'indptr_': [0, 5, 6, 6, 7, 8, 9, 9, 10, 11]}, 0, [0, 2], False],
                [{'data_': [[0, 47, 3], [58, 127, 3], [192, 223, 1], [224, 239, 4], [240, 247, 5], [128, 191, 3], [-2, 0, 2], [128, 191, 1], [128, 191, 4], [-2, 0, 8], [97, 97, 6]],
                'indptr_': [0, 5, 6, 6, 7, 8, 9, 9, 10, 11]}, 7, [6], False]],
            # fmt: on
            "optimized": True,
        },
        "tokenizer_metadata": {
            "vocab_type": 1,
            "vocab_size": 10,
            "add_prefix_space": True,
            "stop_token_ids": [0, 1],
        },
        "__VERSION__": "v7",
    }

    class AdaptiveTokenMask(BaseModel):
        store_type: int
        accepted_indices: List[int]
        rejected_indices: List[int]
        accepted_bitset: Any
        uncertain_indices: List[int]

    class AdaptiveTokenMaskCache(RootModel):
        root: List[Tuple[List[int], AdaptiveTokenMask]]

    recovered_obj = json.loads(serialized)
    adaptive_token_mask_cache = recovered_obj.pop("adaptive_token_mask_cache", None)
    assert recovered_obj == expected_json
    AdaptiveTokenMaskCache.model_validate(adaptive_token_mask_cache)


def test_serialize_compiled_grammar_roundtrip():
    """Test CompiledGrammar serialization and deserialization roundtrip."""
    original_compiled_grammar, tokenizer_info = construct_compiled_grammar()
    serialized = original_compiled_grammar.serialize_json()
    recovered_compiled_grammar = xgr.CompiledGrammar.deserialize_json(serialized, tokenizer_info)
    serialized_new = recovered_compiled_grammar.serialize_json()
    assert serialized == serialized_new


def test_serialize_compiled_grammar_functional():
    """Test that deserialized CompiledGrammar object functions correctly."""
    original_compiled_grammar, tokenizer_info = construct_compiled_grammar()
    serialized = original_compiled_grammar.serialize_json()
    recovered_compiled_grammar = xgr.CompiledGrammar.deserialize_json(serialized, tokenizer_info)

    # Test that both create functional matchers
    matcher_original = xgr.GrammarMatcher(original_compiled_grammar)
    matcher_recovered = xgr.GrammarMatcher(recovered_compiled_grammar)

    # Test token mask generation
    token_bitmask_original = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    token_bitmask_recovered = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    # Both should generate the same masks
    assert matcher_original.fill_next_token_bitmask(
        token_bitmask_original
    ) == matcher_recovered.fill_next_token_bitmask(token_bitmask_recovered)

    # Import torch for tensor comparison
    import torch

    torch.testing.assert_close(token_bitmask_original, token_bitmask_recovered)

    # Test input acceptance
    test_input = "aaa"
    assert matcher_original.accept_string(test_input) == matcher_recovered.accept_string(test_input)
    assert matcher_original.is_terminated() == matcher_recovered.is_terminated()


@pytest.mark.hf_token_required
def test_serialize_compiled_grammar_with_hf_tokenizer():
    """Test CompiledGrammar serialization with a real HuggingFace tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", use_fast=True, trust_remote_code=True
    )
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

    # Test with JSON schema
    class TestModel(BaseModel):
        name: str
        age: int

    # Compile grammar
    compiled_grammar = grammar_compiler.compile_json_schema(TestModel)

    # Serialize and deserialize
    tokenizer_info_json = tokenizer_info.serialize_json()
    tokenizer_info_recovered = xgr.TokenizerInfo.deserialize_json(tokenizer_info_json)
    serialized = compiled_grammar.serialize_json()
    recovered_compiled_grammar = xgr.CompiledGrammar.deserialize_json(
        serialized, tokenizer_info_recovered
    )

    # Test functional equivalence
    test_json = '{"name": "John", "age": 30}'
    token_ids = tokenizer.encode(test_json)[1:]  # skip the initial BOS token
    matcher = xgr.GrammarMatcher(recovered_compiled_grammar)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    for token_id in token_ids:
        matcher.fill_next_token_bitmask(bitmask)
        masked_token_ids = xgr.testing._get_masked_tokens_from_bitmask(
            bitmask, tokenizer_info.vocab_size
        )
        assert token_id not in masked_token_ids
        assert matcher.accept_token(token_id)

    assert matcher.accept_token(tokenizer.eos_token_id)
    assert matcher.is_terminated()


def test_serialize_grammar_utf8():
    """Test Grammar serialization with UTF-8 characters."""
    grammar = xgr.Grammar.from_ebnf('root ::= "こんにちは" | "😊" | "你好" | "hello" | "\\n"')
    serialized = grammar.serialize_json()
    recovered_grammar = xgr.Grammar.deserialize_json(serialized)
    assert _is_grammar_accept_string(recovered_grammar, "こんにちは")
    assert _is_grammar_accept_string(recovered_grammar, "😊")
    assert _is_grammar_accept_string(recovered_grammar, "你好")
    assert _is_grammar_accept_string(recovered_grammar, "hello")
    assert _is_grammar_accept_string(recovered_grammar, "\n")


if __name__ == "__main__":
    pytest.main(sys.argv)
