import Module from "./xgrammar_binding";

let binding: any = null;

async function asyncInitBinding() {
  if (binding == null) {
    binding = await Module();
  }
}

/**
 * This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar and
 * provides utilities to parse and print the AST. User should provide a BNF/EBNF (Extended
 * Backus-Naur Form) grammar, and use from_ebnf_string to parse and simplify the grammar into an
 * AST of BNF grammar.
 */
export class BNFGrammar {
  handle: any;

  /**
   * @internal
   * Private constructor. Factory methods are used since binding initialization is asynchronous.
   * @param {any} handle handle of BNFGrammar created by binding.
   */
  constructor(handle: any) {
    this.handle = handle;
  }

  /**
   * Dispose this BNFGrammar.
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
   * main ::= "ab" a [a-z]
   * a ::= "cd" (=[a-z])
   * ```
   * The assertion (=[a-z]) means a must be followed by [a-z].
   * @param {string} ebnfString The grammar string
   * @param {string} [mainRule="main"] The name of the main rule. Default: "main".
   * @returns {BNFGrammar} The parsed BNF grammar.
   */
  static async createBNFGrammar(ebnfString: string, mainRule = "main"): Promise<BNFGrammar> {
    await asyncInitBinding();
    return new BNFGrammar(new binding.BNFGrammar(ebnfString, mainRule));
  }

  /**
   * Load a BNF grammar from the raw representation of the AST in JSON format.
   * @param {string} json_string The JSON string.
   * @returns {BNFGrammar} The loaded BNF grammar.
   */
  static async deserialize(json_string: string): Promise<BNFGrammar> {
    await asyncInitBinding();
    return new BNFGrammar(binding.BNFGrammar.Deserialize(json_string));
  }

  /**
   * Print the BNF grammar to a string, in standard BNF format.
   * @returns The BNF grammar string.
   */
  toString(): string {
    return this.handle.ToString();
  }

  /**
   * Serialize the AST. Dump the raw representation of the AST to a JSON file.
   * @param {boolean} [prettify=false] Whether to format the JSON string. If False, all whitespaces
   * will be removed.
   * @returns {string} The JSON string
   */
  serialize(prettify = false): string {
    return this.handle.Serialize(prettify);
  }
}

export class BuiltinGrammar {
  /**
   * Get the grammar of standard JSON.
   * @returns {BNFGrammar} The JSON grammar.
   */
  static async json(): Promise<BNFGrammar> {
    await asyncInitBinding();
    return new BNFGrammar(new binding.BuiltinGrammar.JSON());
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
   * @returns {BNFGrammar} The generated BNF grammar.
   */
  static async jsonSchema(
    schema: string,
    indent = 2,
    separators?: [string, string],
    strictMode = true
  ): Promise<BNFGrammar> {
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
    return new BNFGrammar(
      new binding.BuiltinGrammar.JSONSchema(schema, optionalIndent, separators, strictMode));
  }

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
    return binding.BuiltinGrammar._JSONSchemaToEBNF(schema, optionalIndent, separators, strictMode);
  }
}

/**
 * A class that wraps a decoded token table, needed to instantiate GrammarStateMatcher.
 */
export class XGTokenTable {
  /** A handle to the decoded token table of type binding.VectorString. */
  decodedTokenTable: any;

  /**
   * @internal
   * Private constructor. Factory methods are used since binding initialization is asynchronous.
   * @param {any} decodedTokenTable Post-processed token table.
   */
  private constructor(decodedTokenTable: any) {
    this.decodedTokenTable = decodedTokenTable;
  };

  /**
   * Dispose this token table.
   */
  dispose() {
    this.decodedTokenTable.delete();
  }

  /**
   * Instantiate with raw token table and the decoder type by internally post-processing
   * the raw token table by decoding each token with the provided decoder type.
   * @param {string[]} rawTokenTable: the token table in the form of a string list of tokens,
   * ordered by their token id. It should include all the special tokens.
   * @param {string} decoderType: either "byte_fallback", or "byte_level". See `tokenizer.cc` for
   * its semantic.
   */
  static async createXGTokenTable(
    rawTokenTable: string[],
    decoderType: string,
  ): Promise<XGTokenTable> {
    await asyncInitBinding();
    // Convert string[] to std::vector<std::string>
    const rawTokenTableVec = binding.vecStringFromJSArray(rawTokenTable);
    // Returns of type binding.VectorString
    const decodedTokenTable = binding.DecodeTokenTable(rawTokenTableVec, decoderType);
    rawTokenTableVec.delete();
    // Instantiate XGTokenTable
    return new XGTokenTable(decodedTokenTable);
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
export class GrammarStateMatcher {
  private handle: any;

  /**
   * @internal
   * Private constructor. Factory methods are used since binding initialization is asynchronous.
   * @param {any} handle handle of GrammarStateMatcher created by binding.
   */
  private constructor(handle: any) {
    this.handle = handle;
  }

  /**
   * Dispose this grammar state matcher.
   */
  dispose() {
    this.handle.delete();
  }

  /**
   * Construct a GrammarStateMatcher.
   * @param {BNFGrammar} bnfGrammar The BNF grammar to match.
   * @param {XGTokenTable} tokenTable The decoded token table.
   * @param {number[] | number} [stopTokenIds=undefined] Stop tokens to override the default ones.
   * @param {boolean} [terminateWithoutStopToken=false] Whether to terminate without stop token.
   * @param {number} [maxRollbackSteps=0] Max rollback steps.
   * @returns {GrammarStateMatcher} The constructed GrammarStateMatcher.
   */
  static async createGrammarStateMatcher(
    bnfGrammar: BNFGrammar,
    tokenTable: XGTokenTable,
    stopTokenIds?: number[] | number,
    terminateWithoutStopToken: boolean = false,
    maxRollbackSteps: number = 0,
  ): Promise<GrammarStateMatcher> {
    await asyncInitBinding();
    // Convert stopTokenIds to std::vector<int> if not undefined
    if (stopTokenIds !== undefined) {
      if (!Array.isArray(stopTokenIds)) {
        stopTokenIds = [stopTokenIds];
      }
      stopTokenIds = binding.vecIntFromJSArray(stopTokenIds);
    }
    return new GrammarStateMatcher(new binding.GrammarStateMatcher(
      bnfGrammar.handle,
      tokenTable.decodedTokenTable,
      stopTokenIds,
      terminateWithoutStopToken,
      maxRollbackSteps,
    ));
  }

  /**
   * Get the vocab size.
   */
  getVocabSize(): number {
    return this.handle.GetVocabSize();
  }

  /**
   * Get the maximum number of rollback steps allowed.
   */
  getMaxRollbackSteps(): number {
    return this.handle.GetMaxRollbackSteps();
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
  _acceptString(inputStr: string, verbose: boolean = false): boolean {
    return this.handle._AcceptString(inputStr, verbose);
  }

  /**
   * Returns a bitmask in the form of an Int32Array of length ceildiv(vocab_size, 32)
   * based on what tokens can/cannot be accepted by the current state of the grammar state matcher.
   *
   * @returns {Int32Array} An array representing the bitmask that masks the rejected token IDs
   */
  async findNextTokenBitmask(): Promise<Int32Array> {
    await asyncInitBinding();
    const maskIntVector = this.handle.FindNextTokenBitmask()  // a handle of std::vector<int32_t>
    const maskInt32Array = binding.vecIntToView(maskIntVector).slice();
    maskIntVector.delete();
    return maskInt32Array;
  }

  /**
   * 
   * @param {Int32Array} bitmask Bitmask returned by findNextTokenBitmask().
   * @param {number} vocabSize Vocab size returned by getVocabSize().
   * @returns An array of vocab ID that will be rejected as a result of the bitmask.
   */
  static async getRejectedTokensFromBitmask(
    bitmask: Int32Array,
    vocabSize: number
  ): Promise<Int32Array> {
    await asyncInitBinding();
    const bitmaskIntVector = binding.vecIntFromJSArray(bitmask);
    const rejectedIDsIntVector = binding.GrammarStateMatcher.GetRejectedTokensFromBitMask(
      bitmaskIntVector,
      vocabSize
    );
    bitmaskIntVector.delete();
    const rejectedIDsInt32Array = binding.vecIntToView(rejectedIDsIntVector).slice();
    rejectedIDsIntVector.delete();
    return rejectedIDsInt32Array;
  }

  /**
   * Check if the matcher has accepted the stop token and terminated. See also
   * GrammarStateMatcher.acceptToken.
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
   * number of steps, nor can it exceed the specified maximum number of rollback steps.
   */
  rollBack(numTokens: number): void {
    this.handle.Rollback(numTokens);
  }
}
