/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/xgrammar.h
 * \brief The header for the support of grammar-guided generation. The header for the support of
 * matching tokens to BNF grammar. This is the core logic of the grammar-guided generation.
 */

#ifndef XGRAMMAR_XGRAMMAR_H_
#define XGRAMMAR_XGRAMMAR_H_

#include <dlpack/dlpack.h>
#include <xgrammar/object.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace xgrammar {

/*!
 * \brief This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar.
 * The BNF definition here is standard BNF, and the characters are represented using regex-style
 * character classes (e.g. [a-z], [^a-z]).
 *
 * \details
 * ### Rules
 * The BNF grammar AST consists of a set of rules. Each rule contains a name and a definition, and
 * corresponds to a production in the grammar. The definition of a rule is a RuleExpr. Each rule
 * has a rule_id for reference.
 *
 * ### RuleExprs
 * RuleExpr is the definition of a rule or part of the definition of a rule. It can contain
 * elements, empty string, reference to other RuleExprs, or reference to other rules. Each RuleExpr
 * corresponds to an rule_expr_id for reference.
 *
 * For example, in the following rule: rule ::= ("a" "b") | "c"
 * ("a" "b"), "c", ("a" "b") | "c" are all RuleExprs.
 *
 * #### Types of RuleExprs
 * Every RuleExpr is represented by a type as well as a variable-length array containing its data.
 * RuleExpr has several types:
 * - Byte string: a string of bytes (0~255). Supports UTF-8 strings.
 * - Character class: a range of characters (each character is a unicode codepoint), e.g. [a-z],
 *   [ac-z]. Can be negated: [^a-z], [^ac-z]. Now only ascii chars is allowed in [], but this
 *   expression can accept/reject unicode chars.
 * - Character class star: a star quantifier of a character class. e.g. [a-z]*, [^a-z]*.
 * - EmptyStr: an empty string, i.e. ""
 * - Rule reference: a reference to another rule
 * - Sequence: a sequence of rule_exprs, e.g. ("a" "b"). These rule_exprs are concatenated together.
 * - Choices: a choice of rule_exprs, e.g. ("a" "b") | "c". Each rule_expr can be matched.
 *
 * #### Storage of RuleExprs
 * Each type of RuleExpr has a different data format. For the format of each type of RuleExpr, see
 * docs in BNFGrammar::Impl::RuleExprType.
 *
 * We store all RuleExprs in csr_matrix style. That is, they are stored consecutively in one vector
 * (data vector) and the starting position of each RuleExpr is recorded in the indptr vector.
 *
 * \remark The character class star RuleExpr is for the special support for elements like [a-z]*
 * in the grammar. We add it to make the matching more efficient, as we can avoid recursion into
 * rules when matching a sequence of characters. It should be used like:
 * rule1 ::= ((element1 element2 rule2 ...) | ...)
 * rule2 ::= character_class_star_rule_expr(id_of_a_character_class_rule_expr)
 */
class BNFGrammar {
 public:
  /*!
   * \brief Construct a BNF grammar with a EBNF-formatted string. The grammar will be normalized
   * (simplified) by default.
   * \param ebnf_string The EBNF-formatted string.
   * \param root_rule The name of the root rule.
   */
  BNFGrammar(const std::string& ebnf_string, const std::string& root_rule = "root");

  std::string ToString() const;

  /*! \brief Print a BNF grammar. */
  friend std::ostream& operator<<(std::ostream& os, const BNFGrammar& grammar);

  std::string Serialize(bool prettify = false) const;

  /*!
   * \brief Construct a BNF grammar from the dumped JSON string.
   * \param json_string The JSON-formatted string. This string should have the same format as
   * the result of BNFGrammarJSONSerializer::ToString.
   */
  static BNFGrammar Deserialize(const std::string& json_string);

  XGRAMMAR_DEFINE_PIMPL_METHODS(BNFGrammar);
};

class BuiltinGrammar {
 public:
  /*!
   * \brief Get the grammar of standard JSON format. We have built-in support for JSON.
   */
  static BNFGrammar JSON();

  /*!
   * \brief Construct a BNF grammar from the json schema string. The schema string should be in the
   * format of the schema of a JSON file. We will parse the schema and generate a BNF grammar.
   * \param schema The schema string.
   * \param indent The number of spaces for indentation. If set to std::nullopt, the output will be
   * in one line. Default: 2.
   * \param separators Two separators used in the schema: comma and colon. Examples: {",", ":"},
   * {", ", ": "}. If std::nullopt, the default separators will be used: {",", ": "} when the
   * indent is not nullopt, and {", ", ": "} otherwise. This follows the convention in python
   * json.dumps(). Default: std::nullopt.
   * \param strict_mode Whether to use strict mode. In strict mode, the generated grammar will not
   * allow properties and items that is not specified in the schema. This is equivalent to
   * setting unevaluatedProperties and unevaluatedItems to false.
   *
   * This helps LLM to generate accurate output in the grammar-guided generation with JSON
   * schema. Default: true.
   */
  static BNFGrammar JSONSchema(
      const std::string& schema,
      std::optional<int> indent = std::nullopt,
      std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
      bool strict_mode = true
  );

  /*!
   * \brief Convert JSON schema string to EBNF grammar string.
   * \param json_schema The JSON schema string.
   * \param indent The number of spaces for indentation. If set to std::nullopt, the output will be
   * in one line. Default: 2.
   * \param separators Two separators used in the schema: comma and colon. Examples: {",", ":"},
   * {", ", ": "}. If std::nullopt, the default separators will be used: {",", ": "} when the
   * indent is not -1, and {", ", ": "} otherwise. This follows the convention in python
   * json.dumps(). Default: std::nullopt. \param strict_mode Whether to use strict mode. In strict
   * mode, the generated grammar will not allow properties and items that is not specified in the
   * schema. This is equivalent to setting unevaluatedProperties and unevaluatedItems to false.
   *
   * This helps LLM to generate accurate output in the grammar-guided generation with JSON
   * schema. Default: true.
   * \returns The EBNF grammar string.
   */
  static std::string _JSONSchemaToEBNF(
      const std::string& schema,
      std::optional<int> indent = std::nullopt,
      std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
      bool strict_mode = true
  );

  /*!
   * \brief Convert regex string to EBNF grammar string.
   * \param regex The regex string.
   * \returns The EBNF grammar string.
   */
  static std::string _RegexToEBNF(const std::string& regex);
};

enum class VocabType : int {
  RAW = 0,
  BYTE_FALLBACK = 1,
  BYTE_LEVEL = 2,
};

class TokenizerInfo {
 public:
  TokenizerInfo(
      const std::vector<std::string>& encoded_vocab,
      VocabType vocab_type = VocabType::RAW,
      bool prepend_space_in_tokenization = false
  );
  int GetVocabSize() const;
  VocabType GetVocabType() const;
  bool GetPrependSpaceInTokenization() const;
  const std::vector<std::string>& GetDecodedVocab() const;

  static TokenizerInfo FromHuggingFace(
      const std::vector<std::string>& encoded_vocab, const std::string& backend_str
  );

  std::string DumpMetadata() const;
  static TokenizerInfo FromVocabAndMetadata(
      const std::vector<std::string>& encoded_vocab, const std::string& metadata
  );

  XGRAMMAR_DEFINE_PIMPL_METHODS(TokenizerInfo);
};

/*!
 * \brief The compiled grammar of a GrammarMatcher. It contains the preprocessing results of the
 * grammar and tokenizer.
 */
class CompiledGrammar {
 public:
  /*!
   * \brief Specify a grammar and raw vocabulary to return their preprocessing results. These
   * results are used to construct a GrammarMatcher. They can be stored elsewhere for quick
   * construction of GrammarMatcher.
   * \param grammar The grammar that the matcher follows.
   * \param decoded_vocab The tokens that the matcher requires for matching.
   */
  CompiledGrammar(const BNFGrammar& grammar, const std::vector<std::string>& decoded_vocab);

  CompiledGrammar(const BNFGrammar& grammar, const TokenizerInfo& tokenizer_info);

  XGRAMMAR_DEFINE_PIMPL_METHODS(CompiledGrammar);
};

/*!
 * \brief A stateful matcher to match tokens to the specified BNF grammar. This class is the core
 * logic of the grammar-guided generation.
 *
 * \details This class implements the non-deterministic pushdown automaton (NPDA) matching algorithm
 * to match characters to a BNF grammar. It keep track of the current state of the matching process
 * by maintaining several stacks internally as possible paths in the NPDA. It also supports
 * backtracking.
 *
 * It is particularly capable of finding the set of tokens that are acceptable for the next step
 * and storing them in a bitmask. This aids in grammar-guided generation.
 *
 * \example
 * \code
 * Tokenizer tokenizer = ...;
 * auto compiled_grammar = GrammarMatcher::CreateCompiledGrammar(grammar,
 *                                                        tokenizer->PostProcessedVocab());
 * GrammarMatcher matcher(compiled_grammar, 10);
 * matcher->AcceptToken(67);
 *
 * // Construct a DLTensor with shape (tokenizer.GetMaskVocabSize() + 31) / 32, and dtype uint32.
 * DLTensor next_token_bitmask = ...;
 * matcher->GetNextTokenBitmask(&next_token_bitmask);
 *
 * // Rollback is supported
 * matcher->Rollback(1);
 * \endcode
 */
class GrammarMatcher {
 public:
  /*!
   * \brief Construct a GrammarMatcher from the preprocessing result of type
   * CompiledGrammar.
   * \param compiled_grammar The compiled grammar. It is obtained through
   * CreateCompiledGrammar as a result of preprocessing the grammar and tokenizer.
   */
  GrammarMatcher(
      const CompiledGrammar& compiled_grammar,
      std::optional<std::vector<int>> override_stop_tokens = std::nullopt,
      bool terminate_without_stop_token = false,
      std::optional<int> mask_vocab_size = std::nullopt,
      int max_rollback_tokens = 0
  );

  /*!
   * \brief Accept one token and update the state of the matcher.
   * \param token_id The id of the token to accept.
   * \return Whether the token is accepted.
   * \note Termination state.
   * When the end of the root rule is reached, the matcher can only accept the stop token.
   * The matcher is terminated after accepting the stop token, i.e. no AcceptToken or
   * FindNextTokenMask operations can be performed. The termination state can be canceled
   * using Rollback().
   */
  bool AcceptToken(int32_t token_id, bool verbose = false);

  bool AcceptString(const std::string& input_str, bool verbose = false);

  static uint32_t GetBufferSize(size_t mask_vocab_size);

  /*!
   * \brief Get the set of tokens that are acceptable for the next step and store them in a
   * bitmask.
   * \param next_token_bitmask The bitmask to store the result. The bitmask must be pre-allocated
   * and with shape (GetBufferSize(mask_vocab_size),) and dtype uint32.
   */
  void GetNextTokenBitmask(DLTensor* next_token_bitmask);

  static void GetRejectedTokensFromBitMask(
      const DLTensor& token_bitmask, size_t mask_vocab_size, std::vector<int>* rejected_tokens
  );

  /*!
   * \brief Find the jump-forward string for jump-forward decoding. This is the longest string that
   will be valid according to the current syntax.
   * \note This method does not change the grammar state.
   */
  std::string FindJumpForwardString();

  /*!
   * \brief Rollback the matcher to a previous state.
   * \param num_tokens The number of tokens to rollback. It cannot exceed the current number of
   * steps, nor can it exceed the specified maximum number of rollback tokens.
   */
  void Rollback(int num_tokens = 1);

  const std::vector<int>& GetStopTokenIds() const;

  /*! \brief Get the maximum number of rollback tokens allowed. */
  int GetMaxRollbackTokens() const;

  size_t GetMaskVocabSize() const;

  /*!
   * \brief Check if the matcher has accepted the stop token and terminated.
   * \sa AcceptToken
   */
  bool IsTerminated() const;

  /*! \brief Reset the matcher to the initial state. */
  void Reset();

  XGRAMMAR_DEFINE_PIMPL_METHODS(GrammarMatcher);
};

/*!
 * \brief A cache to get the grammar state compiled grammar for grammar or schema. This class avoids
 * redundant preprocessing of the grammar or schema when constructing a CompiledGrammar.
 * \note This class is associated with a vocabulary when constructed. The vocabulary is used to
 * create every grammar state compiled grammar. If multiple toke tables are used to create init
 * contexts, an instance of this class for each vocabulary should be created.
 */
class CachedGrammarCompiler {
 public:
  /*!
   * \brief Construct a CachedGrammarCompiler with a vocabulary. This class will always
   * create grammar state compiled grammars with this vocabulary.
   * \param decoded_vocab The vocabulary that the grammar will use.
   */
  CachedGrammarCompiler(const std::vector<std::string>& decoded_vocab);

  CachedGrammarCompiler(const TokenizerInfo& tokenizer_info);

  /*! \brief Get the compiled grammar for pure JSON. */
  CompiledGrammar GetCompiledGrammarForJSON();

  /*! \brief Get the compiled grammar for a JSON schema string. */
  CompiledGrammar GetCompiledGrammarForJSONSchema(
      const std::string& schema,
      std::optional<int> indent = std::nullopt,
      std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
      bool strict_mode = true
  );

  /*! \brief Clear the interal cache of compiled grammars. */
  void Clear();

  XGRAMMAR_DEFINE_PIMPL_METHODS(CachedGrammarCompiler);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_XGRAMMAR_H_
