# Serialization

XGrammar supports serialization and deserialization for caching and cross-process and inter-server
communication purposes. We currently support serialization to and deserialization from JSON.

The classes that support serialization and deserialization are introduced below. Each class has a
`serialize_json` method to serialize the type to a JSON string, and a `deserialize_json` method to
deserialize the type from a JSON string.

Each serialized result have a `__VERSION__` field to indicate the serialization version. When the
internal data structure is changed in XGrammar, the serialization version will be updated. Use
[`xgr.get_serialization_version`](xgrammar.get_serialization_version) to get the current serialization
version. In deserialization, if the version in the JSON string does not match the current version,
the deserialization will fail and raise a [`xgr.DeserializeVersionError`](xgrammar.DeserializeVersionError).

> **Note:**<br>
> After upgrading XGrammar, the serialization format may change. If you have cached serialization
> results to disk, please clear the cache after the upgrade to avoid potential version conflicts.

Three error types are raised when deserialization fails:

- [`xgr.InvalidJSONError`](xgrammar.InvalidJSONError): When the JSON string is invalid.
- [`xgr.DeserializeFormatError`](xgrammar.DeserializeFormatError): When the JSON string does not follow the serialization format of the type.
- [`xgr.DeserializeVersionError`](xgrammar.DeserializeVersionError): When the serialization version in the JSON string is not the same as the
  current version.

## [`xgr.Grammar`](xgrammar.Grammar)

The grammar class.

```python
import xgrammar as xgr
from transformers import AutoTokenizer

# Construct Grammar
grammar: xgr.Grammar = xgr.Grammar.builtin_json_grammar()

# Serialize to JSON
grammar_json: str = grammar.serialize_json()
print(f"Serialized Grammar: {grammar_json}")

# Deserialize from JSON
grammar_deserialized: xgr.Grammar = xgr.Grammar.deserialize_json(grammar_json)
print(f"Deserialized Grammar: {grammar_deserialized}")
```

## [`xgr.TokenizerInfo`](xgrammar.TokenizerInfo)

The tokenizer information class.

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

## [`xgr.CompiledGrammar`](xgrammar.CompiledGrammar)

A `CompiledGrammar` contains a tokenizer info, and multiple `CompiledGrammar`s can share the same
`TokenizerInfo` to avoid redundant storage. So in serialization of a `CompiledGrammar`, we will not
include the complete `TokenizerInfo`, but only the metadata of it. During deserialization, we will
require a `TokenizerInfo` object and check if it matches the metadata in the serialized JSON string.
If so, the deserialization will use the provided `TokenizerInfo` object in the result. If not, the
deserialization will raise a `RuntimeError`.

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
```
