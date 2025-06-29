# Workflow of XGrammar

XGrammar is a library for structured generation. It can be used in LLM inference code with HuggingFace `transformers`,
or be integrated into LLM engines.

```python
import xgrammar as xgr
```

## Grammar


Grammar describes the structure of the LLM output. It can be:
* A JSON schema or free-form JSON
* A regex
* A customized context-free grammar in the extended BNF format
* etc.

[`xgr.Grammar`](xgrammar.Grammar)



To construct a grammar, use
```python
grammar: xgr.Grammar = xgr.Grammar.from_json_schema(json_schema_string)
# or
grammar: xgr.Grammar = xgr.Grammar.from_regex(regex_string)
# or
grammar: xgr.Grammar = xgr.Grammar.from_ebnf(ebnf_string)
```


## Tokenizer Info





## Single LLM Engine






## Compile Grammar

XGrammar has these data structures for structured generation:

- `xgr.Grammar`: The grammar that the output should follow. It can be a JSON schema, a regex, a BNF grammar, etc.
- `xgr.TokenizerInfo`: The tokenizer information of the model. This is necessary for XGrammar to generate the token mask.
- `xgr.CompiledGrammar`: It will be used to match the grammar against a sequence of tokens.
- `xgr.GrammarCompiler`: Compiles a grammar into a `xgr.CompiledGrammar` object.
- `xgr.GrammarMatcher`: Matches a grammar against a sequence of tokens.

Construct a `GrammarCompiler` and compile the grammar.



The grammar can be a built-in JSON grammar, a JSON schema string, or an EBNF string. EBNF provides
more flexibility for customization. See
[GBNF documentation](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) for
specification.

```python
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
compiled_grammar = grammar_compiler.compile_builtin_json_grammar()
# Other ways: provide a json schema string
# compiled_grammar = grammar_compiler.compile_json_schema(json_schema_string)
# Or provide an EBNF string
# compiled_grammar = grammar_compiler.compile_grammar(ebnf_string)
```

## Generate with grammar

Use logits_processor to generate with grammar.

```python
xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
generated_ids = model.generate(
    **model_inputs, max_new_tokens=512, logits_processor=[xgr_logits_processor]
)
generated_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
print(tokenizer.decode(generated_ids, skip_special_tokens=True))
```

## What to Do Next

- Check out [JSON Generation Guide](../how_to/ebnf_guided_generation.md) and other How-To guides for the detailed usage guide of XGrammar.
- Report any problem or ask any question: open new issues in our [GitHub repo](https://github.com/mlc-ai/xgrammar/issues).
