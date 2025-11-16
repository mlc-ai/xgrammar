import xgrammar as xgr
from transformers import AutoTokenizer
import time

string_test_schemas = [
    {
        "type":"string",
        "minLength": 10  
    },
    {
        "type":"string",
        "maxLength": 2048
    },
    {
        "type":"string",
        "minLength": 10,
        "maxLength": 2048
    },
    {
        "type":"string"
    }
]

array_test_schemas = [
    {
        "type":"array",
        "items": {"type": ["string", "number"]},
        "minItems": 10 
    },
    {
        "type":"array",
        "items": {"type": ["string", "number"]},
        "maxItems": 2048
    },
    {
        "type":"array",
        "items": {"type": ["string", "number"]},
        "minItems": 10,
        "maxItems": 2048
    },
    {
        "type":"array",
        "items": {"type": ["string", "number"]},
    }
]

object_test_schemas = [
    {
        "type":"object",
        "patternProperties": {
            "^[a-zA-Z_][a-zA-Z_0-9]*$": { "type": "string" }
        },
        "minProperties": 10
    },
    {
        "type":"object",
        "patternProperties": {
            "^[a-zA-Z_][a-zA-Z_0-9]*$": { "type": "string" }
        },
        "maxProperties": 2048
    },
    {
        "type":"object",
        "patternProperties": {
            "^[a-zA-Z_][a-zA-Z_0-9]*$": { "type": "string" }
        },
        "minProperties": 10,
        "maxProperties": 2048
    },
    {
        "type":"object",
        "patternProperties": {
            "^[a-zA-Z_][a-zA-Z_0-9]*$": { "type": "string" }
        },
    },
]


length = [10, 20, 128, 1024, 2048]

tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer=tokenizer)
token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)

warmup_schema = {
    "type":"object"
}

_ = compiler.compile_json_schema(warmup_schema)

def test_schema_length(schemas: list[dict], lengths: list[int]):
    for length in lengths:
        test_string = "\""
        for i in range(length):
            test_string += "a"
        test_string += "\""
    
        for schema in schemas:
            start_time = time.monotonic_ns()
            compiled_grammar = compiler.compile_json_schema(schema)
            matcher = xgr.GrammarMatcher(compiled_grammar)
            compile_time = time.monotonic_ns() - start_time
            print(f"Schema: {schema}, String Length: {length}, Compile Time: {compile_time / 1000:.2f} us")
            
            tpots = []
            for character in test_string:
                start_time = time.monotonic_ns()
                matcher.fill_next_token_bitmask(token_bitmask)
                matcher.accept_string(character)
                tpot_time = time.monotonic_ns() - start_time
                tpots.append(tpot_time)
            avg_tpot = sum(tpots) / len(tpots)
            print(f"Schema: {schema}, String Length: {length}, Average TPoT: {avg_tpot / 1000:.2f} us")

def test_array_length(schemas: list[dict], lengths: list[int]):
    for length in lengths:
        test_array = "["
        for i in range(length):
            if i % 2 == 0:
                test_array += "0,"
            else:
                test_array += "\"a\","
        test_array += "]"
    
        for schema in schemas:
            start_time = time.monotonic_ns()
            compiled_grammar = compiler.compile_json_schema(schema)
            matcher = xgr.GrammarMatcher(compiled_grammar)
            compile_time = time.monotonic_ns() - start_time
            print(f"Schema: {schema}, Array Length: {length}, Compile Time: {compile_time / 1000:.2f} us")
            
            tpots = []
            for character in test_array:
                start_time = time.monotonic_ns()
                matcher.fill_next_token_bitmask(token_bitmask)
                matcher.accept_string(character)
                tpot_time = time.monotonic_ns() - start_time
                tpots.append(tpot_time)
            avg_tpot = sum(tpots) / len(tpots)
            print(f"Schema: {schema}, Array Length: {length}, Average TPoT: {avg_tpot / 1000:.2f} us")

def test_object_length(schemas: list[dict], lengths: list[int]):
    for length in lengths:
        test_object = "{"
        for i in range(length):
            test_object += f"\"key{i}\": \"value{i}\","
        test_object += "}"
    
        for schema in schemas:
            start_time = time.monotonic_ns()
            compiled_grammar = compiler.compile_json_schema(schema)
            matcher = xgr.GrammarMatcher(compiled_grammar)
            compile_time = time.monotonic_ns() - start_time
            print(f"Schema: {schema}, Object Length: {length}, Compile Time: {compile_time / 1000:.2f} us")
            print(compiled_grammar.grammar)
            
            tpots = []
            for character in test_object:
                start_time = time.monotonic_ns()
                matcher.fill_next_token_bitmask(token_bitmask)
                matcher.accept_string(character)
                tpot_time = time.monotonic_ns() - start_time
                tpots.append(tpot_time)
                print(f"Char: {character}, TPoT: {tpot_time / 1000:.2f} us")
            avg_tpot = sum(tpots) / len(tpots)
            print(f"Schema: {schema}, Object Length: {length}, Average TPoT: {avg_tpot / 1000:.2f} us")

# test_schema_length(string_test_schemas, length)
# test_array_length(array_test_schemas, length)
test_object_length(object_test_schemas, length)