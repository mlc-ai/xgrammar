# Serialization

XGrammar supports serialization and deserialization for caching and cross-process and cross-node
communication purposes. We currently support serialization to and deserialization from JSON.


## Usage

The following types are supported:

- `Grammar`
- `TokenizerInfo`
- `CompiledGrammar`

Each type has a `serialize_json` method to serialize the type to a JSON string, and
a `deserialize_json` method to deserialize the type from a JSON string.

```python
import xgrammar as xgr
from transformers import AutoTokenizer
```

### `xgr.Grammar`

```python
# Construct Grammar
grammar: xgr.Grammar = xgr.Grammar.builtin_json_grammar()

# Serialize to JSON
grammar_json: str = grammar.serialize_json()
print(f"Serialized Grammar: {grammar_json}")

# Deserialize from JSON
grammar_deserialized: xgr.Grammar = xgr.Grammar.deserialize_json(grammar_json)
print(f"Deserialized Grammar: {grammar_deserialized}")
```

### `xgr.TokenizerInfo`

```python
# Construct TokenizerInfo
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer_info: xgr.TokenizerInfo = xgr.TokenizerInfo.from_huggingface(tokenizer)

# Serialize to JSON
tokenizer_info_json: str = tokenizer_info.serialize_json()
print(f"Serialized TokenizerInfo")

# Deserialize from JSON
tokenizer_info_deserialized: xgr.TokenizerInfo = xgr.TokenizerInfo.deserialize_json(
    tokenizer_info_json
)
print(f"Deserialized TokenizerInfo")
```

### `xgr.CompiledGrammar`

Each `CompiledGrammar` is associated with a `TokenizerInfo`, but multiple `CompiledGrammar`s can
share the same `TokenizerInfo`. Therefore, when serializing a `CompiledGrammar`, we will not
include the `TokenizerInfo`. Instead, we store the metadata of the `TokenizerInfo`.
During deserialization, you need to provide a `TokenizerInfo` object to the `deserialize_json`
method. XGrammar will check if the metadata of the `TokenizerInfo` object matches the original one,
and return a `CompiledGrammar` associated with the provided `TokenizerInfo`. If the metadata
does not match, the deserialization will raise a `RuntimeError`.

```python
# Construct CompiledGrammar
compiler: xgr.GrammarCompiler = xgr.GrammarCompiler(tokenizer_info_deserialized)
compiled_grammar: xgr.CompiledGrammar = compiler.compile_grammar(grammar_deserialized)

# Serialize to JSON
compiled_grammar_json: str = compiled_grammar.serialize_json()
print(f"Serialized CompiledGrammar")

# Deserialize from JSON
compiled_grammar_deserialized: xgr.CompiledGrammar = xgr.CompiledGrammar.deserialize_json(
    compiled_grammar_json, tokenizer_info_deserialized
)
print(f"Deserialized CompiledGrammar")

# compiled_grammar_deserialized can be used as a normal CompiledGrammar
```

## Deserialization Errors




### Serialization Version Mismatch

The serialization version is a string that is used to identify the serialization format. When the
serialization format is updated in a new version (e.g. the internal data structure of the type is
changed), the serialization version will be updated.

The serialization version is added to the serialized JSON string. When the deserialization is
performed, the serialization version of the JSON string must match the current serialization
version. If the version is not correct, the deserialization will fail and return a
`std::runtime_error`.

**Note**: Always check if the deserialization version error is raised.
