import sys

import pytest
from pydantic import BaseModel
from transformers import AutoTokenizer  # type: ignore

import xgrammar as xgr


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


def test_serialize_grammar():
    """Test Grammar serialization produces expected JSON string."""
    grammar = construct_grammar()
    serialized = grammar.serialize_json()
    expected_json = (
        '{"rules_":[["rule1",3,8],["root_rule",7,-1]],'
        '"grammar_expr_data_":{"data_":[1,1,48,57,4,0,5,0,1,6,2,4,0,0,97,5,4,5,6,6,5,5],'
        '"indptr_":[0,4,6,9,11,13,15,18,20,22]},"root_rule_id_":1,"root_tag_dispatch_fsm":null,'
        '"tag_dispatch_end_node_to_rule_id":[],"allow_empty_rule_ids":[],"__VERSION__":"v2"}'
    )
    assert serialized == expected_json


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
    test_input = "a"
    assert matcher_original.accept_string(test_input) == matcher_recovered.accept_string(test_input)


def test_serialize_tokenizer_info():
    """Test TokenizerInfo serialization produces expected JSON string."""
    tokenizer_info = construct_tokenizer_info()
    serialized = tokenizer_info.serialize_json()
    expected_json = (
        '{"vocab_type":1,"vocab_size":10,"add_prefix_space":true,'
        '"stop_token_ids":[0,1],"special_token_ids":[9],"__VERSION__":"v2"}'
    )
    assert serialized == expected_json


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

    test_input = "a"
    assert matcher_original.accept_string(test_input) == matcher_recovered.accept_string(test_input)


def test_serialize_compiled_grammar():
    """Test CompiledGrammar serialization produces expected JSON string."""
    compiled_grammar, tokenizer_info = construct_compiled_grammar()
    serialized = compiled_grammar.serialize_json()

    expected_json = "TODO"
    assert serialized == expected_json


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
    test_input = "a"
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
    token_ids = tokenizer.encode(test_json)
    matcher = xgr.GrammarMatcher(recovered_compiled_grammar)
    bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

    for token_id in token_ids:
        matcher.fill_next_token_bitmask(bitmask)
        masked_token_ids = xgr.get_masked_tokens_from_bitmask(bitmask, tokenizer_info.vocab_size)
        assert token_id not in masked_token_ids
        assert matcher.accept_token(token_id)

    assert matcher.accept_token(tokenizer.eos_token_id)
    assert matcher.is_terminated()


if __name__ == "__main__":
    pytest.main(sys.argv)
