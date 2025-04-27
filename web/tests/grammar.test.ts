/**
 * Test all APIs exposed in the web-xgrammar package. The goal of these unit tests
 * are to test each API works as expected. It does not test behavior correctness
 * thoroughly since that is done in `tests/python`.
 */
import { describe, expect, test } from "@jest/globals";
import { Grammar, GrammarCompiler, CompiledGrammar, TokenizerInfo, GrammarMatcher, Testings, StructuralTagItem } from "..";
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
  // Equivalent to basic_json_rules_ebnf_no_space in `test_json_schema_converter.py`
  const basic_json_rules_ebnf_no_space = String.raw`basic_escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_string_sub ::= ("\"" | [^"\\\r\n] basic_string_sub | "\\" basic_escape basic_string_sub) (= [ \n\t]* [,}\]:])
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*)
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= ["] basic_string_sub
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= (("[" "" basic_any (", " basic_any)* "" "]") | ("[" "" "]"))
basic_object ::= ("{" "" basic_string ": " basic_any (", " basic_string ": " basic_any)* "" "}") | "{" "}"
`


  /**
   * Equivalent to
    class MainModel(BaseModel):
        num: int = 0
        opt_bool: Optional[bool] = None
        size: Optional[float]
        name: str = ""
   */
  const schema = String.raw`{"properties": {"num": {"default": 0, "title": "Num", "type": "integer"}, "opt_bool": {"anyOf": [{"type": "boolean"}, {"type": "null"}], "default": null, "title": "Opt Bool"}, "size": {"anyOf": [{"type": "number"}, {"type": "null"}], "title": "Size"}, "name": {"default": "", "title": "Name", "type": "string"}}, "required": ["size"], "title": "MainModel", "type": "object"}`;

  test("Test Testings.ebnfToGrammarNoNormalization and Grammar.toString", async () => {
    // Equivalent to `test_grammar_parser.py` test_ebnf()
    const before = `root ::= b c | b root
b ::= "ab"*
c ::= [acep-z]+
d ::= "d"?
`;
    const expected = `root ::= ((b c) | (b root))
b ::= ((b_1))
c ::= ((c_1))
d ::= ((d_1))
b_1 ::= ("" | ("ab" b_1))
c_1 ::= (([acep-z] c_1) | [acep-z])
d_1 ::= ("" | "d")
`
    const grammar1 = await Testings.ebnfToGrammarNoNormalization(before)
    const outputStr = grammar1.toString();
    expect(outputStr).toEqual(expected);
  });

  test("Test Grammar.fromJSONSchema()", async () => {
    const grammar = await Grammar.fromJSONSchema(schema);
    const outputStr = grammar.toString();
    expect(outputStr == "").toEqual(false);
  });

  test("Test jsonSchemaToEBNF", async () => {
    // Equivalent to test_optional() in test_json_schema_converter.py

    const ebnf_grammar = basic_json_rules_ebnf_no_space + (
      String.raw`root_prop_1 ::= basic_boolean | basic_null
root_prop_2 ::= basic_number | basic_null
root ::= "{" "" ("\"num\"" ": " basic_integer ", ")? ("\"opt_bool\"" ": " root_prop_1 ", ")? "\"size\"" ": " root_prop_2 (", " "\"name\"" ": " basic_string)? "" "}"
`
    )

    const grammar = await Testings.jsonSchemaToEBNF(schema, false, -1);
    expect(grammar).toEqual(ebnf_grammar);
  });

  test("Test indent jsonSchemaToEBNF", async () => {
    const grammar0 = await Testings.jsonSchemaToEBNF(schema, false, -1);
    const grammar1 = await Testings.jsonSchemaToEBNF(schema, false);
    const grammar2 = await Testings.jsonSchemaToEBNF(schema, false, 2);
    expect(grammar1).toEqual(grammar2);
    expect(grammar0).not.toEqual(grammar2);
  });

  test("Test indent Grammar.fromJSONSchema()", async () => {
    const grammar0 = (await Grammar.fromJSONSchema(schema, false, -1)).toString();
    const grammar1 = (await Grammar.fromJSONSchema(schema, false)).toString();
    const grammar2 = (await Grammar.fromJSONSchema(schema, false, 2)).toString();
    expect(grammar1).toEqual(grammar2);
    expect(grammar0).not.toEqual(grammar2);
  });

  test("Test jsonSchema() argument separators not supported yet", async () => {
    expect(async () => {
      const grammar = await Grammar.fromJSONSchema(schema, false, 2, [",", ":"]);
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
    const compiledGrammar0 = await compiler.compileJSONSchema(schema, false, -1);
    const compiledGrammar1 = await compiler.compileJSONSchema(schema, false);
    const compiledGrammar2 = await compiler.compileJSONSchema(schema, false, 2);
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
    expect(outputStr1).toEqual(outputStr2);
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
      ["<s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
      ["<s>", "a", "abc", 'b"', '"', ':"', "{", "}", ", ", "6", ":", " "],
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
    // One of the 2 places we test any_whitespace=true
    const grammar = await compiler.compileJSONSchema(schemaStr, true, 2);
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
    // One of the 2 places we test any_whitespace=true
    const grammar = await compiler.compileJSONSchema(schemaStr, true, 2);
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

// Identical to `test_grammar_matcher_structural_tag.py`
describe("Test Structural Tag", () => {
  // Define simple schemas for testing
  const schema1 = {
    properties: {
      arg1: { type: "string" },
      arg2: { type: "integer" }
    },
    required: ["arg1", "arg2"],
    type: "object"
  };

  const schema2 = {
    properties: {
      arg3: { type: "number" },
      arg4: { 
        type: "array",
        items: { type: "string" }
      }
    },
    required: ["arg3", "arg4"],
    type: "object"
  };

  const expected_grammar_test_structural_tag = String.raw`root ::= TagDispatch(("<function=f", trigger_rule_0), ("<function=g", trigger_rule_1))
trigger_rule_0 ::= (("1>" root_1 "</function>") | ("2>" root_2 "</function>"))
basic_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(basic_string_sub))
basic_string_sub ::= (("\"") | ([^\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)) (=([ \n\t]* [,}\]:]))
basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*)) (=([ \n\t]* "}"))
basic_string ::= (("\"" basic_string_sub)) (=([ \n\t]* "," [ \n\t]* "\"arg2\"" [ \n\t]* ":" [ \n\t]* basic_integer [ \n\t]* "}"))
root_1 ::= (("{" [ \n\t]* "\"arg1\"" [ \n\t]* ":" [ \n\t]* basic_string [ \n\t]* "," [ \n\t]* "\"arg2\"" [ \n\t]* ":" [ \n\t]* basic_integer [ \n\t]* "}"))
basic_integer_1 ::= ("" | ("-")) (=([1-9] [0-9]*))
basic_escape_1 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(basic_string_sub_1))
basic_string_sub_1 ::= (("\"") | ([^\"\\\r\n] basic_string_sub_1) | ("\\" basic_escape_1 basic_string_sub_1)) (=([ \n\t]* [,}\]:]))
basic_integer_2 ::= (("0") | (basic_integer_1_1 [1-9] [0-9]*)) (=([ \n\t]* "}"))
basic_string_1 ::= (("\"" basic_string_sub_1)) (=([ \n\t]* "," [ \n\t]* "\"arg2\"" [ \n\t]* ":" [ \n\t]* basic_integer_2 [ \n\t]* "}"))
root_2 ::= (("{" [ \n\t]* "\"arg1\"" [ \n\t]* ":" [ \n\t]* basic_string_1 [ \n\t]* "," [ \n\t]* "\"arg2\"" [ \n\t]* ":" [ \n\t]* basic_integer_2 [ \n\t]* "}"))
basic_integer_1_1 ::= ("" | ("-")) (=([1-9] [0-9]*))
trigger_rule_1 ::= ((">" root_3 "</function>"))
basic_escape_2 ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(basic_string_sub_2))
basic_string_sub_2 ::= (("\"") | ([^\"\\\r\n] basic_string_sub_2) | ("\\" basic_escape_2 basic_string_sub_2)) (=([ \n\t]* [,}\]:]))
basic_number ::= ((basic_number_choice basic_number_3 basic_number_6)) (=([ \n\t]* "," [ \n\t]* "\"arg4\"" [ \n\t]* ":" [ \n\t]* root_prop_1 [ \n\t]* "}"))
basic_string_2 ::= (("\"" basic_string_sub_2))
root_prop_1 ::= (("[" [ \n\t]* basic_string_2 root_prop_1_1 [ \n\t]* "]") | ("[" [ \n\t]* "]")) (=([ \n\t]* "}"))
root_3 ::= (("{" [ \n\t]* "\"arg3\"" [ \n\t]* ":" [ \n\t]* basic_number [ \n\t]* "," [ \n\t]* "\"arg4\"" [ \n\t]* ":" [ \n\t]* root_prop_1 [ \n\t]* "}"))
basic_number_1 ::= ("" | ("-")) (=([1-9] [0-9]*))
basic_number_2 ::= (([0-9] basic_number_2) | ([0-9]))
basic_number_3 ::= ("" | ("." basic_number_2)) (=(basic_number_6))
basic_number_4 ::= ("" | ([+\-])) (=(basic_number_5))
basic_number_5 ::= (([0-9] basic_number_5) | ([0-9]))
basic_number_6 ::= ("" | ([eE] basic_number_4 basic_number_5))
root_prop_1_1 ::= ("" | ([ \n\t]* "," [ \n\t]* basic_string_2 root_prop_1_1)) (=([ \n\t]* "]"))
basic_number_choice ::= (("0") | (basic_number_1 [1-9] [0-9]*)) (=(basic_number_3 basic_number_6))
`;
  
  // Identical to `test_simple()` in `test_grammar_matcher_structural_tag.py`
  test("test_simple()", async () => {
    const grammar_str = String.raw`root ::= TagDispatch(("tag1", rule1), ("tag2", rule2))
rule1 ::= "abcd"
rule2 ::= "efg"
`;
    const grammar = await Grammar.fromEBNF(grammar_str);
    expect(await Testings.isGrammarAcceptString(grammar, "tag1abcd")).toEqual(true);
    expect(await Testings.isGrammarAcceptString(grammar, "tag1abcdtag2efg")).toEqual(true);
    expect(await Testings.isGrammarAcceptString(grammar, "tag1abcdqqqqtag2efg")).toEqual(true);
    expect(await Testings.isGrammarAcceptString(grammar, "tag1abc")).toEqual(false);
    expect(await Testings.isGrammarAcceptString(grammar, "tag1abce")).toEqual(false);
    expect(await Testings.isGrammarAcceptString(grammar, "ttag1abd")).toEqual(false);
  });

  // Identical to `test_structural_tag()` in `test_grammar_matcher_structural_tag.py`
  test("Test Grammar.fromStructuralTag", async () => {
    // Create structural tags
    const tags = [
      new StructuralTagItem("<function=f1>", schema1, "</function>"),
      new StructuralTagItem("<function=f2>", schema1, "</function>"),
      new StructuralTagItem("<function=g>", schema2, "</function>")
    ];
    
    // Define triggers. In real cases, we should use one trigger: "<function=", and dispatch to two
    // tags. Here we use two triggers for testing.
    const triggers = ["<function=f", "<function=g"];
    
    // Create grammar
    const grammar = await Grammar.fromStructuralTag(tags, triggers);

    const accepted_inputs = [
      '<function=f1>{"arg1": "abc", "arg2": 1}</function>',
      '<function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>',
      '<function=f2>{"arg1": "abc", "arg2": 1}</function><function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>',
      'hhhh<function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>haha<function=f1>{"arg1": "abc", "arg2": 1}</function>123',
    ];

    for (const input of accepted_inputs) {
      expect(await Testings.isGrammarAcceptString(grammar, input)).toEqual(true);
    }

    grammar.dispose();
  });

  // Identical to `test_structural_tag_compiler()` in `test_grammar_matcher_structural_tag.py`
  test("Test GrammarCompiler.compileStructuralTag", async () => {
    // Create structural tags
    const tags = [
      new StructuralTagItem("<function=f1>", schema1, "</function>"),
      new StructuralTagItem("<function=f2>", schema1, "</function>"),
      new StructuralTagItem("<function=g>", schema2, "</function>")
    ];

    // Define triggers
    const triggers = ["<function=f", "<function=g"];

    // Create tokenizer info
    const tokenizerInfo = await TokenizerInfo.createTokenizerInfo([]);

    // Create compiler
    const compiler = await GrammarCompiler.createGrammarCompiler(tokenizerInfo);

    // Compile grammar
    const compiledGrammar = await compiler.compileStructuralTag(tags, triggers);

    // Check that the compiled grammar was created successfully
    expect(compiledGrammar.grammar().toString()).toEqual(expected_grammar_test_structural_tag);

    // Cleanup
    tokenizerInfo.dispose();
    compiledGrammar.dispose();
    compiler.dispose();
  });

  // Identical to `test_structural_tag_mask_gen()` in `test_grammar_matcher_structural_tag.py`
  test("E2E test of structural tag with GrammarMatcher", async () => {
    // Create structural tags
    const tags = [
      new StructuralTagItem("<function=f>", schema1, "</function>"),
      new StructuralTagItem("<function=g>", schema2, "</function>")
    ];
    const tokenizerUrl = "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC/raw/main/tokenizer.json";
    
    // Define triggers, both are fine
    const triggers = ["<function="];
    // const triggers = ["<function=f", "<function=g"];

    const jsonBuffer = await (await fetch(tokenizerUrl)).arrayBuffer();
    const tokenizer = await Tokenizer.fromJSON(jsonBuffer);
    const tokenizerInfo = await getTokenizerInfoFromUrl(
      tokenizerUrl,
      "byte_level",
      false,
    );

    const compiler = await GrammarCompiler.createGrammarCompiler(tokenizerInfo);
    const compiledGrammar = await compiler.compileStructuralTag(tags, triggers);
    const matcher = await GrammarMatcher.createGrammarMatcher(compiledGrammar);

    // Test input string
    const accepted_input = String.raw`hhhh<function=g>{"arg3": 1.23, "arg4": ["a", "b", "c"]}</function>
haha<function=f>{"arg1": "abc", "arg2": 1}</function>123`;
    // Convert string to Uint8Array for bytes
    const input_bytes = new TextEncoder().encode(accepted_input);

    for (let i = 0; i < input_bytes.length; i++) {
      const char = input_bytes[i];

      // Get bitmask for the next token
      const bitmask = await matcher.getNextTokenBitmask();
      const rejected_token_ids = (await Testings.debugGetMaskedTokensFromBitmask(
        bitmask, tokenizerInfo.getVocabSize()
      ));

      // Check that the next token is not rejected
      const token_id_for_next_char = tokenizer.encode(String.raw`${char}`);
      expect(token_id_for_next_char.length).toEqual(1);
      expect(rejected_token_ids.indexOf(token_id_for_next_char[0])).toEqual(-1);

      // Accept the next token
      const accepted = matcher.acceptToken(char);
      expect(accepted).toEqual(true);
    }

    // Final verification - check that EOS token is allowed
    const final_bitmask = await matcher.getNextTokenBitmask();
    const final_rejected_tokens = (await Testings.debugGetMaskedTokensFromBitmask(
      final_bitmask, tokenizerInfo.getVocabSize()
    ));
    expect(final_rejected_tokens.indexOf(128001)).toEqual(-1);  // stop token not rejected

    // Cleanup
    tokenizerInfo.dispose();
    compiledGrammar.dispose();
    compiler.dispose();
    matcher.dispose();
  });
});
