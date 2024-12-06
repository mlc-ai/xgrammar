import Module from "./xgrammar_binding";

let binding: any = null;

async function asyncInitBinding() {
  if (binding == null) {
    binding = await Module();
  }
}

/**
 * Various testing methods that are not optimized for performance.
 */
export class Testings {
  /**
   * Convert JSON schema string to EBNF grammar string. For test purposes.
   *
   * @param {string} schema The schema string.
   * @param {number} [indent=2] The number of spaces for indentation. If -1, the grammar will
   * enforce the output to be in one line.
   * @param {[string, string]} [separators] Two separators that will be enforced by the grammar:
   * comma and colon. Examples: (",", ":"), (", ", ": "). If undefined, the default separators will
   * be used: (",", ": ") when the indent is not undefined, and (", ", ": ") otherwise. This follows
   * the convention in Python's json.dumps(). Currently unsupported and will use the default value.
   * @param {boolean} [strictMode=true] Whether to use strict mode. In strict mode, the generated
   * grammar will not allow properties and items that is not specified in the schema. This is
   * equivalent to setting unevaluatedProperties and unevaluatedItems to false.
   * @returns {string} The EBNF grammar string.
   */
  static async _jsonSchemaToEBNF(
    schema: string,
    indent = 2,
    separators?: [string, string],
    strictMode = true
  ): Promise<string> {
    // TODO(Charlie): Add support for separators, which requires binding std::pair
    // in emscripten
    if (separators !== undefined) {
      throw new Error(
        `Argument separators is not supported yet, please leave it as undefined, and the ` +
        `default value (",", ": ") will be used.`
      );
    }
    await asyncInitBinding();
    // indent being -1 is equivalent to not having a value for the std::optional arg in C++.
    // This is a workaround to Typescript not being able to express Optional value like Python; if
    // user specifies indent to be undefined, it still becomes 2.
    let optionalIndent: number | undefined = indent == -1 ? undefined : indent;
    return binding._JSONSchemaToEBNF(schema, optionalIndent, separators, strictMode);
  }

  /**
   *
   * @param {Int32Array} bitmask Bitmask returned by getNextTokenBitmask().
   * @param {number} vocabSize Vocab size returned by getVocabSize().
   * @param {number} index The batch index of the bitmask. For batch inference, bitmask[index] will
   *  be used. Defaults to 0.
   * @returns An array of vocab ID that will be rejected as a result of the bitmask.
   */
  static async debugGetMaskedTokensFromBitmask(
    bitmask: Int32Array,
    vocabSize: number,
    index: number = 0,
  ): Promise<Int32Array> {
    await asyncInitBinding();
    const bitmaskIntVector = binding.vecIntFromJSArray(bitmask);
    const rejectedIDsIntVector = binding.DebugGetMaskedTokensFromBitmask(
      bitmaskIntVector,
      vocabSize,
      index
    );
    bitmaskIntVector.delete();
    const rejectedIDsInt32Array = binding.vecIntToView(rejectedIDsIntVector).slice();
    rejectedIDsIntVector.delete();
    return rejectedIDsInt32Array;
  }
}

/**
 * This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar and
 * provides utilities to parse and print the AST. User should provide a BNF/EBNF (Extended
 * Backus-Naur Form) grammar, and use from_ebnf_string to parse and simplify the grammar into an
 * AST of BNF grammar.
 */
export class Grammar {
  handle: any;

  /**
   * @internal
   * Private constructor. Factory methods are used since binding initialization is asynchronous.
   * @param {any} handle handle of Grammar created by binding.
   */
  constructor(handle: any) {
    this.handle = handle;
  }

  /**
   * Dispose this Grammar.
   */
  dispose() {
    this.handle.delete();
  }

  /**
   * Construct a BNF grammar with a EBNF-formatted string. The grammar will be normalized
   * (simplified) by default.
   * EBNF grammar: see https://www.w3.org/TR/xml/#sec-notation. Note:
   * 1. Use # as the comment mark
   * 2. Use C-style unicode escape sequence \u01AB, \U000001AB, \xAB
   * 3. A-B (match A and not match B) is not supported yet
   * 4. Lookahead assertion can be added at the end of a rule to speed up matching. E.g.
   * ```
   * root ::= "ab" a [a-z]
   * a ::= "cd" (=[a-z])
   * ```
   * The assertion (=[a-z]) means a must be followed by [a-z].
   * @param {string} ebnfString The grammar string
   * @param {string} [rootRule="root"] The name of the root rule. Default: "root".
   * @returns {Grammar} The parsed BNF grammar.
   */
  static async fromEBNF(ebnfString: string, rootRule = "root"): Promise<Grammar> {
    await asyncInitBinding();
    return new Grammar(new binding.Grammar.FromEBNF(ebnfString, rootRule));
  }

  /**
   * Get the grammar of standard JSON.
   * @returns {Grammar} The JSON grammar.
   */
  static async builtinJSONGrammar(): Promise<Grammar> {
    await asyncInitBinding();
    return new Grammar(new binding.Grammar.BuiltinJSONGrammar());
  }

  /**
   * Construct a BNF grammar from the json schema string. The schema string should be in the
   * format of the schema of a JSON file. We will parse the schema and generate a BNF grammar.
   *
   * @param {string} schema The schema string.
   * @param {number} [indent=2] The number of spaces for indentation. If -1, the grammar will
   * enforce the output to be in one line.
   * @param {[string, string]} [separators] Two separators that will be enforced by the grammar:
   * comma and colon. Examples: (",", ":"), (", ", ": "). If undefined, the default separators will
   * be used: (",", ": ") when the indent is not undefined, and (", ", ": ") otherwise. This follows
   * the convention in Python's json.dumps(). Currently unsupported and will use the default value.
   * @param {boolean} [strictMode=true] Whether to use strict mode. In strict mode, the generated
   * grammar will not allow properties and items that is not specified in the schema. This is
   * equivalent to setting unevaluatedProperties and unevaluatedItems to false.
   * @returns {Grammar} The generated BNF grammar.
   */
  static async fromJSONSchema(
    schema: string,
    indent = 2,
    separators?: [string, string],
    strictMode = true
  ): Promise<Grammar> {
    // TODO(Charlie): Add support for separators, which requires binding std::pair
    // in emscripten
    if (separators !== undefined) {
      throw new Error(
        `Argument separators is not supported yet, please leave it as undefined, and the ` +
        `default value (",", ": ") will be used.`
      );
    }
    await asyncInitBinding();
    // indent being -1 is equivalent to not having a value for the std::optional arg in C++.
    // This is a workaround to Typescript not being able to express Optional value like Python; if
    // user specifies indent to be undefined, it still becomes 2.
    let optionalIndent: number | undefined = indent == -1 ? undefined : indent;
    return new Grammar(
      new binding.Grammar.FromJSONSchema(schema, optionalIndent, separators, strictMode));
  }

  /**
   * Print the BNF grammar to a string, in standard BNF format.
   * @returns The BNF grammar string.
   */
  toString(): string {
    return this.handle.ToString();
  }
}

/**
 * A class that wraps a preprocessed vocab, needed to instantiate GrammarCompiler.
 */
export class TokenizerInfo {
  handle: any;

  /**
   * @internal
   * Private constructor. Factory methods are used since binding initialization is asynchronous.
   * @param {any} handle  handle of TokenizerInfo created by binding.
   */
  constructor(handle: any) {
    this.handle = handle;
  };

  /**
   * Dispose this tokenizer info object.
   */
  dispose() {
    this.handle.delete();
  }

  /**
   * Get the vocab size.
   */
  getVocabSize(): number {
    return this.handle.GetVocabSize();
  }

  /**
   * Get the post-processed vocab. Returned as a handle of type binding.VectorString
   */
  getDecodedVocabHandle(): any {
    return this.handle.GetDecodedVocab();
  }

  /**
   * Instantiate with raw vocab and the vocab type by internally post-processing
   * the raw vocab by decoding each token with the provided vocab type.
   * @param {string[]} encodedVocab: the vocab in the form of a string list of tokens,
   * ordered by their token id. It should include all the special tokens.
   * @param {string} vocabType: either "byte_fallback", "byte_level", or `raw`. See `tokenizer.cc`
   * for its semantic.
   * @param {boolean} prependSpaceInTokenization: whether the tokenizer will prepend a space before
   * the text in the tokenization process.
   * @param {number} vocabSize: the full vocab size read from `config.json`. If not provided, will
   * use length of `encodedVocab`. Note some model has a vocab size larger in `config.json` due
   * to padding. Essentially the size of the logits.
   * @param {number[] | number} [stopTokenIds=undefined] Stop tokens to override the default ones.
   */
  static async createTokenizerInfo(
    encodedVocab: string[],
    vocabType: string,
    prependSpaceInTokenization: boolean,
    vocabSize?: number,
    stopTokenIds?: number[] | number,
  ): Promise<TokenizerInfo> {
    await asyncInitBinding();
    // Convert string[] to std::vector<std::string>
    const encodedVocabVec = binding.vecStringFromJSArray(encodedVocab);
    // Convert stopTokenIds to std::vector<int> if not undefined
    if (stopTokenIds !== undefined) {
      if (!Array.isArray(stopTokenIds)) {
        stopTokenIds = [stopTokenIds];
      }
      stopTokenIds = binding.vecIntFromJSArray(stopTokenIds);
    }
    // Instantiate TokenizerInfo
    return new TokenizerInfo(new binding.TokenizerInfo(
      encodedVocabVec,
      vocabType.toUpperCase(),
      vocabSize,
      stopTokenIds,
      prependSpaceInTokenization,
    ));
  }
}

export class CompiledGrammar {
  handle: any;

  /**
   * @internal
   * Private constructor. Factory methods are used since binding initialization is asynchronous.
   * @param {any} handle handle of CompiledGrammar created by binding.
   */
  constructor(handle: any) {
    this.handle = handle;
  };

  /**
   * Dispose this compiled grammar object.
   */
  dispose() {
    this.handle.delete();
  }

  /**
   * @returns {Grammar} The grammar used to compile this CompiledGrammar.
   */
  grammar(): Grammar {
    return new Grammar(this.handle.GetGrammar());
  }

  /**
   * @returns {TokenizerInfo} The tokenizer info used to compile this CompiledGrammar.
   */
  tokenizerInfo(): TokenizerInfo {
    return new TokenizerInfo(this.handle.GetTokenizerInfo());
  }
}

export class GrammarCompiler {
  handle: any;

  /**
   * @internal
   * Private constructor. Factory methods are used since binding initialization is asynchronous.
   * @param {any} handle handle of GrammarCompiler created by binding.
   */
  private constructor(handle: any) {
    this.handle = handle;
  };

  /**
   * Dispose this grammar compiler object.
   */
  dispose() {
    this.handle.delete();
  };

  /**
   *
   * @param tokenizerInfo {TokenizerInfo} The tokenizer info that contains preprocessed vocab.
   * @param cacheEnabled {boolean} Whether to enable caching. Default is true.
   */
  static async createGrammarCompiler(
    tokenizerInfo: TokenizerInfo,
    cacheEnabled: boolean = true,
  ): Promise<GrammarCompiler> {
    await asyncInitBinding();
    // NOTE(Charlie): Have not figured out how to do multithreading in WASM, so always set to 1.
    return new GrammarCompiler(new binding.GrammarCompiler(
      tokenizerInfo.handle,
      /**max_threads=*/1,
      cacheEnabled
    ));
  }

  /**
   * Get CompiledGrammar from the json schema string. The schema string should be in the
   * format of the schema of a JSON file. We will parse the schema and generate a BNF grammar.
   *
   * @param {string} schema The schema string.
   * @param {number} [indent=2] The number of spaces for indentation. If -1, the grammar will
   * enforce the output to be in one line.
   * @param {[string, string]} [separators] Two separators that will be enforced by the grammar:
   * comma and colon. Examples: (",", ":"), (", ", ": "). If undefined, the default separators will
   * be used: (",", ": ") when the indent is not undefined, and (", ", ": ") otherwise. This follows
   * the convention in Python's json.dumps(). Currently unsupported and will use the default value.
   * @param {boolean} [strictMode=true] Whether to use strict mode. In strict mode, the generated
   * grammar will not allow properties and items that is not specified in the schema. This is
   * equivalent to setting unevaluatedProperties and unevaluatedItems to false.
   * @returns {CompiledGrammar} The compiled grammar for the specified JSON schema.
   */
  async compileJSONSchema(
    schema: string,
    indent = 2,
    separators?: [string, string],
    strictMode = true
  ): Promise<CompiledGrammar> {
    // TODO(Charlie): Add support for separators, which requires binding std::pair
    // in emscripten
    if (separators !== undefined) {
      throw new Error(
        `Argument separators is not supported yet, please leave it as undefined, and the ` +
        `default value (",", ": ") will be used.`
      );
    }
    await asyncInitBinding();
    // indent being -1 is equivalent to not having a value for the std::optional arg in C++.
    // This is a workaround to Typescript not being able to express Optional value like Python; if
    // user specifies indent to be undefined, it still becomes 2.
    let optionalIndent: number | undefined = indent == -1 ? undefined : indent;
    return new CompiledGrammar(
      this.handle.CompileJSONSchema(schema, optionalIndent, separators, strictMode));
  }

  /**
   * @returns {CompiledGrammar} The compiled grammar for JSON.
   */
  async compileBuiltinJSONGrammar(): Promise<CompiledGrammar> {
    await asyncInitBinding();
    return new CompiledGrammar(this.handle.CompileBuiltinJSONGrammar());
  }

  /**
   * Get CompiledGrammar from the EBNF-formatted string. The grammar will be normalized
   * (simplified) by default.
   * EBNF grammar: see https://www.w3.org/TR/xml/#sec-notation. Note:
   * 1. Use # as the comment mark
   * 2. Use C-style unicode escape sequence \u01AB, \U000001AB, \xAB
   * 3. A-B (match A and not match B) is not supported yet
   * 4. Lookahead assertion can be added at the end of a rule to speed up matching. E.g.
   * ```
   * root ::= "ab" a [a-z]
   * a ::= "cd" (=[a-z])
   * ```
   * The assertion (=[a-z]) means a must be followed by [a-z].
   * @param {string} ebnfString The grammar string
   * @param {string} [rootRule="root"] The name of the root rule. Default: "root".
   * @returns {CompiledGrammar} The compiled grammar for the specified EBNF string.
   */
  async compileGrammar(grammar: Grammar): Promise<CompiledGrammar>;
  async compileGrammar(grammar: string, rootRule?: string): Promise<CompiledGrammar>;
  async compileGrammar(grammar: string | Grammar, rootRule: string="root"): Promise<CompiledGrammar> {
    await asyncInitBinding();
    if (typeof grammar === "string") {
      const grammarObj = await Grammar.fromEBNF(grammar, rootRule);
      return new CompiledGrammar(this.handle.CompileGrammar(grammarObj.handle));
    } else {
      return new CompiledGrammar(this.handle.CompileGrammar(grammar.handle));
    }
  }
}

/**
 * A stateful matcher to match tokens to the specified BNF grammar. This class is the core logic
 * of the grammar-guided generation.
 *
 * This class implements the non-deterministic pushdown automaton (NPDA) matching algorithm to
 * match characters to a BNF grammar. It keep track of the current state of the matching process by
 * maintaining several stacks internally as possible paths in the NPDA. It also supports
 * backtracking.
 *
 * It is particularly capable of finding the set of tokens that are acceptable for the next step
 * and storing them in a bitmask. This aids in grammar-guided generation.
 */
export class GrammarMatcher {
  private handle: any;
  private vocab_size: number;

  /**
   * @internal
   * Private constructor. Factory methods are used since binding initialization is asynchronous.
   * @param {any} handle handle of GrammarMatcher created by binding.
   */
  private constructor(handle: any, vocab_size: number) {
    this.handle = handle;
    this.vocab_size = vocab_size;
  }

  /**
   * Dispose this grammar state matcher.
   */
  dispose() {
    this.handle.delete();
  }

  /**
   * Construct a GrammarMatcher.
   * @param {CompiledGrammar} compiledGrammar A compiled grammar from GrammarCompiler.
   * @param {number[] | number} [overrideStopTokens=undefined] Stop tokens to override the default ones.
   * @param {boolean} [terminateWithoutStopToken=false] Whether to terminate without stop token.
   * @param {number} [maxRollbackTokens=0] Max rollback tokens.
   * @returns {GrammarMatcher} The constructed GrammarMatcher.
   */
  static async createGrammarMatcher(
    compiledGrammar: CompiledGrammar,
    overrideStopTokens?: number[] | number,
    terminateWithoutStopToken: boolean = false,
    maxRollbackTokens: number = 0,
  ): Promise<GrammarMatcher> {
    await asyncInitBinding();
    // Convert overrideStopTokens to std::vector<int> if not undefined
    if (overrideStopTokens !== undefined) {
      if (!Array.isArray(overrideStopTokens)) {
        overrideStopTokens = [overrideStopTokens];
      }
      overrideStopTokens = binding.vecIntFromJSArray(overrideStopTokens);
    }
    return new GrammarMatcher(new binding.GrammarMatcher(
      compiledGrammar.handle,
      overrideStopTokens,
      terminateWithoutStopToken,
      maxRollbackTokens,
    ), compiledGrammar.tokenizerInfo().getVocabSize());
  }

  /**
   * Get the maximum number of rollback tokens allowed.
   */
  getMaxRollbackTokens(): number {
    return this.handle.GetMaxRollbackTokens();
  }

  /**
   * Accept one token and update the state of the matcher.
   * @param {number} tokenID The id of the token to accept.
   * @param {boolean} [verbose=false] To print debugging info
   * @returns {boolean} Whether the token is accepted.
   */
  acceptToken(tokenID: number, verbose: boolean = false): boolean {
    return this.handle.AcceptToken(tokenID, verbose);
  }

  /**
   * Accept one unicode codepoint to the current state. For test purposes.
   * @param {string} inputStr The unicode codepoint of the character to be accepted.
   * @param {boolean} [verbose=false] To print debugging info
   * @returns {boolean} Whether the input string is accepted.
   */
  _debugAcceptString(inputStr: string, verbose: boolean = false): boolean {
    return this.handle._DebugAcceptString(inputStr, verbose);
  }

  /**
   * Returns a bitmask in the form of an Int32Array of length ceildiv(vocab_size, 32)
   * based on what tokens can/cannot be accepted by the current state of the grammar state matcher.
   *
   * @returns {Int32Array} An array representing the bitmask that masks the rejected token IDs
   */
  async getNextTokenBitmask(): Promise<Int32Array> {
    await asyncInitBinding();
    // a handle of std::vector<int32_t>
    const maskIntVector = this.handle.GetNextTokenBitmask(this.vocab_size)
    const maskInt32Array = binding.vecIntToView(maskIntVector).slice();
    maskIntVector.delete();
    return maskInt32Array;
  }

  /**
   * Check if the matcher has accepted the stop token and terminated. See also
   * GrammarMatcher.acceptToken.
   */
  isTerminated(): boolean {
    return this.handle.IsTerminated();
  }

  /**
   * Reset the matcher to the initial state.
   */
  reset(): void {
    this.handle.Reset();
  }

  /**
   * Find the jump-forward string for jump-forward decoding. This is the longest string that
   * will be valid according to the current syntax.
   * @returns {string} The jump-forward string.
   */
  findJumpForwardString(): string {
    return this.handle.FindJumpForwardString();
  }

  /**
   * Rollback the matcher to a previous state.
   * @param {number} numTokens The number of tokens to rollback. It cannot exceed the current
   * number of steps, nor can it exceed the specified maximum number of rollback tokens.
   */
  rollBack(numTokens: number): void {
    this.handle.Rollback(numTokens);
  }
}
