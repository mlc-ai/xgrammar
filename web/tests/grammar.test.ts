/**
 * Test all APIs exposed in the web-xgrammar package. The goal of these unit tests
 * are to test each API works as expected. It does not test behavior correctness
 * thoroughly since that is done in `tests/python`.
 */
import { describe, expect, test } from "@jest/globals";
import { Grammar, GrammarCompiler, CompiledGrammar, TokenizerInfo, GrammarMatcher, Testings } from "..";
import { Tokenizer } from "@mlc-ai/web-tokenizers";

async function getTokenizerInfoFromUrl(tokenizerUrl: string, vocabType: string, prependSpace: boolean): Promise<TokenizerInfo> {
  // 1. Get tokenizer
  const jsonBuffer = await (await fetch(tokenizerUrl)).arrayBuffer();
  const tokenizer = await Tokenizer.fromJSON(jsonBuffer);
  // 2. Get raw vocab
  const encodedVocab: string[] = [];
  const vocabSize = tokenizer.getVocabSize();
  for (let tokenId = 0; tokenId < vocabSize; tokenId++) {
    encodedVocab.push(tokenizer.idToToken(tokenId));
  }
  // 3. Decode
  const decodedVocab = await TokenizerInfo.createTokenizerInfo(encodedVocab, vocabType, prependSpace);
  return decodedVocab;
}


describe("Test all Grammar APIs", () => {
  const ebnf_grammar = String.raw`basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= ("[" "" basic_any (", " basic_any)* "" "]") | "[]"
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{}"
root_prop_1 ::= basic_boolean | basic_null
root_prop_2 ::= basic_number | basic_null
root ::= "{" "" ("\"num\"" ": " basic_integer ", ")? ("\"opt_bool\"" ": " root_prop_1 ", ")? "\"size\"" ": " root_prop_2 (", " "\"name\"" ": " basic_string)? "" "}"
`;

  /**
   * Equivalent to
    class MainModel(BaseModel):
        num: int = 0
        opt_bool: Optional[bool] = None
        size: Optional[float]
        name: str = ""
   */
  const schema = String.raw`{"properties": {"num": {"default": 0, "title": "Num", "type": "integer"}, "opt_bool": {"anyOf": [{"type": "boolean"}, {"type": "null"}], "default": null, "title": "Opt Bool"}, "size": {"anyOf": [{"type": "number"}, {"type": "null"}], "title": "Size"}, "name": {"default": "", "title": "Name", "type": "string"}}, "required": ["size"], "title": "MainModel", "type": "object"}`;

  test("Test Grammar.fromEBNF and Grammar.toString", async () => {
    const before = `root ::= ((b c) | (b root))
b ::= ((b_1 d [a]*))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ("b" b_1))
c_1 ::= ((c_2 c_1) | (c_2))
c_2 ::= (([acep-z]))
d_1 ::= ("" | ("d"))
`;
    const grammar1 = await Grammar.fromEBNF(before)
    const outputStr = grammar1.toString();
    expect(outputStr).toEqual(before);
  });

  test("Test Grammar.fromJSONSchema()", async () => {
    const grammar = await Grammar.fromJSONSchema(schema);
    const outputStr = grammar.toString();
    expect(outputStr == "").toEqual(false);
  });

  test("Test _jsonSchemaToEBNF", async () => {
    // Equivalent to test_optional() in test_json_schema_converter.py
    const grammar = await Testings._jsonSchemaToEBNF(schema, -1);
    expect(grammar).toEqual(ebnf_grammar);
  });

  test("Test indent _jsonSchemaToEBNF", async () => {
    const grammar0 = await Testings._jsonSchemaToEBNF(schema, -1);
    const grammar1 = await Testings._jsonSchemaToEBNF(schema);
    const grammar2 = await Testings._jsonSchemaToEBNF(schema, 2);
    expect(grammar1).toEqual(grammar2);
    expect(grammar0).not.toEqual(grammar2);
  });

  test("Test indent Grammar.fromJSONSchema()", async () => {
    const grammar0 = (await Grammar.fromJSONSchema(schema, -1)).toString();
    const grammar1 = (await Grammar.fromJSONSchema(schema)).toString();
    const grammar2 = (await Grammar.fromJSONSchema(schema, 2)).toString();
    expect(grammar1).toEqual(grammar2);
    expect(grammar0).not.toEqual(grammar2);
  });

  test("Test jsonSchema() argument separators not supported yet", async () => {
    expect(async () => {
      const grammar = await Grammar.fromJSONSchema(schema, 2, [",", ":"]);
    }).rejects.toThrow("Argument separators is not supported yet");
  });

  test("Test Grammar.builtinJSONGrammar()", async () => {
    const grammar = await Grammar.builtinJSONGrammar();
    const outputStr = grammar.toString();
    expect(outputStr == "").toEqual(false);
  });
});

describe("Test TokenizerInfo", () => {
  test("Test basic tokenizer info", async () => {
    const dummyVocab = ["!", "éĶ¦"];
    const dummyVocabType = "byte_level";
    const tokenizerInfo = await TokenizerInfo.createTokenizerInfo(
      dummyVocab, dummyVocabType, false
    );
    expect(tokenizerInfo.getDecodedVocabHandle().get(0)).toEqual("!");
    expect(tokenizerInfo.getDecodedVocabHandle().get(1)).toEqual("锦");
    tokenizerInfo.dispose();
  });

  test("Test with Llama3.2, byte_level", async () => {
    const tokenizerInfo = await getTokenizerInfoFromUrl(
      "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_0-MLC/raw/main/tokenizer.json",
      "byte_level",
      false,
    );
    expect(tokenizerInfo.getDecodedVocabHandle().size()).toEqual(128256);
    tokenizerInfo.dispose();
  })

  test("Test with Phi3.5, byte_fallback", async () => {
    const tokenizerInfo = await getTokenizerInfoFromUrl(
      "https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f16_1-MLC/raw/main/tokenizer.json",
      "byte_fallback",
      true,
    );
    // phi-3.5 though vocab size is 32064 in config.json, has 32011 actual vocab. The size of the
    // table (i.e. tokenizer.getVocabSize()) may be smaller than the `vocab_size` in config.json
    // (length of logits), see https://github.com/QwenLM/Qwen2/issues/147 and
    // https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/47.
    expect(tokenizerInfo.getDecodedVocabHandle().size()).toEqual(32011);
    tokenizerInfo.dispose();
  })
});

describe("Test CompiledGrammar and GrammarCompiler", () => {
  /**
   * Equivalent to
    class MainModel(BaseModel):
        num: int = 0
        opt_bool: Optional[bool] = None
        size: Optional[float]
        name: str = ""
   */
  const schema = String.raw`{"properties": {"num": {"default": 0, "title": "Num", "type": "integer"}, "opt_bool": {"anyOf": [{"type": "boolean"}, {"type": "null"}], "default": null, "title": "Opt Bool"}, "size": {"anyOf": [{"type": "number"}, {"type": "null"}], "title": "Size"}, "name": {"default": "", "title": "Name", "type": "string"}}, "required": ["size"], "title": "MainModel", "type": "object"}`;

  test("Test compileBuiltinJSONGrammar", async () => {
    const tokenizerInfo = await getTokenizerInfoFromUrl(
      "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_0-MLC/raw/main/tokenizer.json",
      "byte_level",
      false,
    );
    const compiler = await GrammarCompiler.createGrammarCompiler(tokenizerInfo);
    const compiledGrammar = await compiler.compileBuiltinJSONGrammar();
    const outputStr = compiledGrammar.grammar().toString();
    expect(outputStr == "").toEqual(false);
    expect(compiledGrammar.tokenizerInfo().getDecodedVocabHandle().size()).toEqual(128256);
    tokenizerInfo.dispose();
    compiledGrammar.dispose();
    compiler.dispose();
  });

  test("Test compileJSONSchema", async () => {
    const tokenizerInfo = await getTokenizerInfoFromUrl(
      "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_0-MLC/raw/main/tokenizer.json",
      "byte_level",
      false,
    );
    const compiler = await GrammarCompiler.createGrammarCompiler(tokenizerInfo);
    const compiledGrammar0 = await compiler.compileJSONSchema(schema, -1);
    const compiledGrammar1 = await compiler.compileJSONSchema(schema);
    const compiledGrammar2 = await compiler.compileJSONSchema(schema, 2);
    const outputStr0 = compiledGrammar0.grammar().toString();
    const outputStr1 = compiledGrammar1.grammar().toString();
    const outputStr2 = compiledGrammar2.grammar().toString();
    expect(outputStr1).toEqual(outputStr2);
    expect(outputStr0).not.toEqual(outputStr2);
    expect(compiledGrammar1.tokenizerInfo().getDecodedVocabHandle().size()).toEqual(128256);
    tokenizerInfo.dispose();
    compiledGrammar0.dispose();
    compiledGrammar1.dispose();
    compiledGrammar2.dispose();
    compiler.dispose();
  });

  test("Test compileGrammar with a EBNF string and a Grammar", async () => {
    const before = `root ::= ((b c) | (b root))
b ::= ((b_1 d [a]*))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ("b" b_1))
c_1 ::= ((c_2 c_1) | (c_2))
c_2 ::= (([acep-z]))
d_1 ::= ("" | ("d"))
`;
    const tokenizerInfo = await getTokenizerInfoFromUrl(
      "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_0-MLC/raw/main/tokenizer.json",
      "byte_level",
      false,
    );
    const compiler = await GrammarCompiler.createGrammarCompiler(tokenizerInfo);
    const grammar = await Grammar.fromEBNF(before);
    const compiledGrammar1 = await compiler.compileGrammar(grammar);
    const compiledGrammar2 = await compiler.compileGrammar(before);
    const outputStr1 = compiledGrammar1.grammar().toString();
    const outputStr2 = compiledGrammar2.grammar().toString();
    expect(outputStr1).toEqual(before);
    expect(outputStr2).toEqual(before);
    expect(compiledGrammar1.tokenizerInfo().getDecodedVocabHandle().size()).toEqual(128256);
    expect(compiledGrammar2.tokenizerInfo().getDecodedVocabHandle().size()).toEqual(128256);
    tokenizerInfo.dispose();
    compiledGrammar1.dispose();
    compiledGrammar2.dispose();
    compiler.dispose();
  });
});

// Identical to tests in `test_grammar_matcher.py`
describe("Test GrammarMatcher E2E", () => {
  const vocab = [
    "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
  ];
  const input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}"];
  const input_ids: number[] = [];
  input_splitted.forEach((input) => {
    input_ids.push(vocab.indexOf(input));
  });

  test("test_token_operations", async () => {
    // 1. Instantiate matcher
    const tokenizerInfo = await TokenizerInfo.createTokenizerInfo(
      vocab, "byte_level", false
    );
    const compiler = await GrammarCompiler.createGrammarCompiler(tokenizerInfo);
    const jsonGrammar = await compiler.compileBuiltinJSONGrammar();
    const matcher = await GrammarMatcher.createGrammarMatcher(jsonGrammar);

    // 2. Test
    const expected = [
      ["{"],
      ['"', "}", "\n", " ", '"a":true'],
      ["a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
      ["a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
      [':"', ":", "\n", " "],
      ['"', "{", "6", "\n", " "],
      ["}", ", ", "6", "\n", " "],
      ['"', "\n", " ", '"a":true'],
      ['"', "\n", " ", '"a":true'],
      ["}", ", ", "\n", " "],
      ["</s>"],
    ]
    const result: Array<Array<string>> = []
    for (let i = 0; i <= input_ids.length; i++) {
      const input_id = input_ids[i];
      // Find rejected IDs
      const bitmask = await matcher.getNextTokenBitmask();
      const rejectedIDs = await Testings.debugGetMaskedTokensFromBitmask(
        bitmask, tokenizerInfo.getVocabSize()
      );
      // Find accepted tokens
      const vocabIDSet = new Set([...Array(vocab.length).keys()]);
      const rejectedIDSet = new Set(rejectedIDs);
      const acceptedIDSet = new Set([...vocabIDSet].filter(x => !rejectedIDSet.has(x)));
      const acceptedIDList = Array.from(acceptedIDSet.values()).sort(((a, b) => a - b));
      const acceptedTokens: string[] = [];
      acceptedIDList.forEach((acceptedID) => {
        acceptedTokens.push(vocab[acceptedID]);
      });
      result.push(acceptedTokens);
      // Note the <= in the loop bound. We do an extra checking for the last input.
      if (i < input_ids.length) {
        // Check input_id is accepted, and update matcher
        expect(acceptedIDSet.has(input_id)).toEqual(true);
        const accepted = matcher.acceptToken(input_id);
        expect(accepted).toEqual(true);
      }
    }
    expect(result).toEqual(expected);
    matcher.dispose();
    tokenizerInfo.dispose();
  });

  // Identical to the test above, except we specify stop token to be both 0 and 1
  test("test_token_operations with customized stop token id", async () => {
    // 1. Instantiate matcher
    // TODO(Charlie): Specifying only 0 still makes 1 a valid stop token -- is this what we want?
    const tokenizerInfo = await TokenizerInfo.createTokenizerInfo(
      vocab, "byte_level", false, undefined, [0, 1]
    );
    const compiler = await GrammarCompiler.createGrammarCompiler(tokenizerInfo);
    const jsonGrammar = await compiler.compileBuiltinJSONGrammar();
    const matcher = await GrammarMatcher.createGrammarMatcher(jsonGrammar);

    // 2. Test
    const expected = [
      ["{"],
      ['"', "}", "\n", " ", '"a":true'],
      ["a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
      ["a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
      [':"', ":", "\n", " "],
      ['"', "{", "6", "\n", " "],
      ["}", ", ", "6", "\n", " "],
      ['"', "\n", " ", '"a":true'],
      ['"', "\n", " ", '"a":true'],
      ["}", ", ", "\n", " "],
      ["<s>", "</s>"],
    ]
    const result: Array<Array<string>> = []
    for (let i = 0; i <= input_ids.length; i++) {
      const input_id = input_ids[i];
      // Find rejected IDs
      const bitmask = await matcher.getNextTokenBitmask();
      const rejectedIDs = await Testings.debugGetMaskedTokensFromBitmask(
        bitmask, tokenizerInfo.getVocabSize()
      );
      // Find accepted tokens
      const vocabIDSet = new Set([...Array(vocab.length).keys()]);
      const rejectedIDSet = new Set(rejectedIDs);
      const acceptedIDSet = new Set([...vocabIDSet].filter(x => !rejectedIDSet.has(x)));
      const acceptedIDList = Array.from(acceptedIDSet.values()).sort(((a, b) => a - b));
      const acceptedTokens: string[] = [];
      acceptedIDList.forEach((acceptedID) => {
        acceptedTokens.push(vocab[acceptedID]);
      });
      result.push(acceptedTokens);
      // Note the <= in the loop bound. We do an extra checking for the last input.
      if (i < input_ids.length) {
        // Check input_id is accepted, and update matcher
        expect(acceptedIDSet.has(input_id)).toEqual(true);
        const accepted = matcher.acceptToken(input_id);
        expect(accepted).toEqual(true);
      }
    }
    expect(result).toEqual(expected);
    tokenizerInfo.dispose();
    matcher.dispose();
  });

  test("test_roll_back", async () => {
    // 1. Instantiate matcher
    const tokenizerInfo = await TokenizerInfo.createTokenizerInfo(
      vocab, "byte_level", false
    );
    const compiler = await GrammarCompiler.createGrammarCompiler(tokenizerInfo);
    const jsonGrammar = await compiler.compileBuiltinJSONGrammar();
    const matcher = await GrammarMatcher.createGrammarMatcher(
      jsonGrammar,
      undefined,
      undefined,
      5,
    );
    tokenizerInfo.dispose();
    expect(matcher.getMaxRollbackTokens()).toEqual(5);

    // 2. Test
    const input_ids_splitted: number[][] = [];
    for (let i = 0; i < input_ids.length; i += 2) {
      input_ids_splitted.push(input_ids.slice(i, i + 2));
    }

    for (let i = 0; i < input_ids_splitted.length; i++) {
      const i_1 = input_ids_splitted[i][0];
      const i_2 = input_ids_splitted[i][1];
      const orig_result: Int32Array[] = [];
      // Accept firt round
      orig_result.push(await matcher.getNextTokenBitmask());
      const accept_i1 = matcher.acceptToken(i_1);
      orig_result.push(await matcher.getNextTokenBitmask());
      const accept_i2 = matcher.acceptToken(i_2);
      expect(accept_i1).toEqual(true);
      expect(accept_i2).toEqual(true);
      // Rollback, then accept again
      matcher.rollBack(2);
      const result_after_rollback: Int32Array[] = [];
      result_after_rollback.push(await matcher.getNextTokenBitmask());
      const accept_i1_r = matcher.acceptToken(i_1);
      result_after_rollback.push(await matcher.getNextTokenBitmask());
      const accept_i2_r = matcher.acceptToken(i_2);
      expect(accept_i1_r).toEqual(true);
      expect(accept_i2_r).toEqual(true);
      // Expect same token bitmask
      expect(orig_result).toEqual(result_after_rollback);
    }
    matcher.dispose();
  });

  test("test reset and termination", async () => {
    // This one has `</s>`, different from the ones used before
    const vocab = [
      "<s>", "</s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", "\n", " ", '"a":true',
    ];
    const input_splitted = ["{", '"', "abc", 'b"', ":", "6", ", ", " ", '"a":true', "}", "</s>"];
    const input_ids: number[] = [];
    input_splitted.forEach((input) => {
      input_ids.push(vocab.indexOf(input));
    });

    // 1. Instantiate matcher
    const tokenizerInfo = await TokenizerInfo.createTokenizerInfo(
      vocab, "byte_level", false
    );
    const compiler = await GrammarCompiler.createGrammarCompiler(tokenizerInfo);
    const jsonGrammar = await compiler.compileBuiltinJSONGrammar();
    const matcher = await GrammarMatcher.createGrammarMatcher(
      jsonGrammar,
      undefined,
      undefined,
      5
    );
    tokenizerInfo.dispose();

    // 2. Accept all one time
    const orig_result: Int32Array[] = [];
    for (let i = 0; i < input_ids.length; i++) {
      orig_result.push(await matcher.getNextTokenBitmask());
      const accepted = matcher.acceptToken(input_ids[i]);
      expect(accepted).toEqual(true);
    }

    // 3. Check termination
    expect(matcher.isTerminated()).toEqual(true);
    const acceptedAfterTerm0 = matcher.acceptToken(0);
    expect(acceptedAfterTerm0).toEqual(false);
    // this will throw error, but cannot be caught by jest
    // await matcher.getNextTokenBitmask()

    // 4. Reset, accept again
    matcher.reset();
    const result_after_reset: Int32Array[] = [];
    for (let i = 0; i < input_ids.length; i++) {
      result_after_reset.push(await matcher.getNextTokenBitmask());
      const accepted = matcher.acceptToken(input_ids[i]);
      expect(accepted).toEqual(true);
    }

    // 5. Check same bitmask result, check termination again
    expect(orig_result).toEqual(result_after_reset);
    expect(matcher.isTerminated()).toEqual(true);
    const acceptedAfterTerm1 = matcher.acceptToken(0);
    expect(acceptedAfterTerm1).toEqual(false);
    // this will throw error, but cannot be caught by jest
    // await matcher.getNextTokenBitmask()

    // 6. Rollback 2, and should not be terminated and should accept "}"
    matcher.rollBack(2);
    expect(matcher.isTerminated()).toEqual(false);
    const acceptedAfterTerm2 = matcher.acceptToken(input_ids.slice(-2, -1)[0]);
    expect(acceptedAfterTerm2).toEqual(true);

    matcher.dispose();
  });

  test("test_get_jump_forward_string", async () => {
    const grammar_ebnf = String.raw`root ::= "abb" | "abbd" | other_rule
other_rule ::= "a" sub_rule "b"
sub_rule ::= "b"
`;
    const vocab = ["a", "bb"];
    const tokenizerInfo = await TokenizerInfo.createTokenizerInfo(
      vocab, "byte_level", false
    );
    const compiler = await GrammarCompiler.createGrammarCompiler(tokenizerInfo);
    const grammar = await compiler.compileGrammar(grammar_ebnf);
    const matcher = await GrammarMatcher.createGrammarMatcher(grammar);
    tokenizerInfo.dispose();
    expect(matcher.acceptToken(0)).toEqual(true);
    expect(matcher.findJumpForwardString()).toEqual("bb");
  });
});

// Identical to `test_builtin_grammar_json_schema.py`
describe("Test json schema E2E", () => {
  // Equivalent to MainModel used in the python test, except removed minItem and maxItem from tuple_field to avoid long log
  const schemaStr = String.raw`{"properties": {"integer_field": {"title": "Integer Field", "type": "integer"}, "number_field": {"title": "Number Field", "type": "number"}, "boolean_field": {"title": "Boolean Field", "type": "boolean"}, "any_array_field": {"items": {}, "title": "Any Array Field", "type": "array"}, "array_field": {"items": {"type": "string"}, "title": "Array Field", "type": "array"}, "tuple_field": {"prefixItems": [{"type": "string"}, {"type": "integer"}, {"items": {"type": "string"}, "type": "array"}], "title": "Tuple Field", "type": "array"}, "object_field": {"additionalProperties": {"type": "integer"}, "title": "Object Field", "type": "object"}, "nested_object_field": {"additionalProperties": {"additionalProperties": {"type": "integer"}, "type": "object"}, "title": "Nested Object Field", "type": "object"}}, "required": ["integer_field", "number_field", "boolean_field", "any_array_field", "array_field", "tuple_field", "object_field", "nested_object_field"], "title": "MainModel", "type": "object"}`;
  // Equivalent to instance in the python test, a valid json following the above schema
  const instanceStr = String.raw`{
  "integer_field": 42,
  "number_field": 314000.0,
  "boolean_field": true,
  "any_array_field": [
    3.14,
    "foo",
    null,
    true
  ],
  "array_field": [
    "foo",
    "bar"
  ],
  "tuple_field": [
    "foo",
    42,
    [
      "bar",
      "baz"
    ]
  ],
  "object_field": {
    "foo": 42,
    "bar": 43
  },
  "nested_object_field": {
    "foo": {
      "bar": 42
    }
  }
}`;

  // Note: This test much slower than others
  test("Test with Llama3.2, byte_level", async () => {
    // 1. Get tokenizer
    const jsonBuffer = await (await fetch(
      "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_0-MLC/raw/main/tokenizer.json"
    )).arrayBuffer();
    const tokenizer = await Tokenizer.fromJSON(jsonBuffer);
    // 2. Get encoded vocab
    const encodedVocab: string[] = [];
    const vocabSize = tokenizer.getVocabSize();
    for (let tokenId = 0; tokenId < vocabSize; tokenId++) {
      encodedVocab.push(tokenizer.idToToken(tokenId));
    }
    // 3. Decode
    const tokenizerInfo = await TokenizerInfo.createTokenizerInfo(encodedVocab, "byte_level", false);
    const compiler = await GrammarCompiler.createGrammarCompiler(tokenizerInfo);

    // 4. Instantiate matcher
    const grammar = await compiler.compileJSONSchema(schemaStr, 2);
    const matcher = await GrammarMatcher.createGrammarMatcher(grammar);
    const inputIds = tokenizer.encode(instanceStr);

    // 5. Expect to accept all inputIds
    for (let i = 0; i < inputIds.length; i++) {
      const inputId = inputIds[i];
      await matcher.getNextTokenBitmask();
      const accepted = matcher.acceptToken(inputId);
      expect(accepted).toEqual(true);
    }

    // 6. Check finalization
    const final_bitmask = await matcher.getNextTokenBitmask();
    expect(final_bitmask.length).toEqual(Math.ceil(128256 / 32));
    const final_rejected_tokens = (await Testings.debugGetMaskedTokensFromBitmask(
      final_bitmask, tokenizerInfo.getVocabSize()
    ));
    expect(final_rejected_tokens.indexOf(128001)).toEqual(-1);  // stop token not rejected
    const acceptStop = matcher.acceptToken(128001);
    expect(acceptStop).toEqual(true);
    expect(matcher.isTerminated()).toEqual(true);

    matcher.dispose();
    grammar.dispose();
    tokenizerInfo.dispose();
  });

  // Note: This test much slower than others
  test("Test with Phi-3.5, byte_fallback, _debugAcceptString", async () => {
    // 1. Get tokenizer
    const jsonBuffer = await (await fetch(
      "https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f16_1-MLC/raw/main/tokenizer.json",
    )).arrayBuffer();
    const tokenizer = await Tokenizer.fromJSON(jsonBuffer);
    // 2. Get encoded vocab
    const encodedVocab: string[] = [];
    const vocabSize = tokenizer.getVocabSize();
    for (let tokenId = 0; tokenId < vocabSize; tokenId++) {
      encodedVocab.push(tokenizer.idToToken(tokenId));
    }
    // 3. Decode; note that phi-3.5 has 32064 as vocab size in `config.json`
    const tokenizerInfo = await TokenizerInfo.createTokenizerInfo(encodedVocab, "byte_fallback", false, 32064);
    const compiler = await GrammarCompiler.createGrammarCompiler(tokenizerInfo);

    // 4. Instantiate matcher
    const grammar = await compiler.compileJSONSchema(schemaStr, 2);
    const matcher = await GrammarMatcher.createGrammarMatcher(grammar);

    // 5. Expect to accept all inputIds
    for (let i = 0; i < instanceStr.length; i++) {
      const inputStr = instanceStr[i];
      await matcher.getNextTokenBitmask();
      // if use acceptToken, the first token of instanceStr will be encoded as 426, `_{`,
      // while the matcher only accepts 29912, `{`, and another token of `<0x??>`
      const accepted = matcher._debugAcceptString(inputStr);
      expect(accepted).toEqual(true);
    }

    // 6. Check finalization
    const final_bitmask = await matcher.getNextTokenBitmask();
    // Tests how phi3.5 has dummy padded tokens. See https://github.com/mlc-ai/mlc-llm/pull/2651
    expect(final_bitmask.length).toEqual(Math.ceil(32064 / 32));
    const final_rejected_tokens = (await Testings.debugGetMaskedTokensFromBitmask(
      final_bitmask, tokenizerInfo.getVocabSize()
    ));
    expect(final_rejected_tokens.indexOf(2)).toEqual(-1);  // stop token not rejected
    expect(final_rejected_tokens.indexOf(32000)).toEqual(-1);  // stop token not rejected
    const acceptStop = matcher.acceptToken(2);
    expect(acceptStop).toEqual(true);
    expect(matcher.isTerminated()).toEqual(true);
    tokenizerInfo.dispose();
  })
});
