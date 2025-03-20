/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.h
 * \brief The header for the definition and construction of BNF grammar.
 */

#ifndef XGRAMMAR_GRAMMAR_H_
#define XGRAMMAR_GRAMMAR_H_

#include <xgrammar/object.h>

#include <optional>
#include <string>
#include <vector>

namespace xgrammar {

struct StructuralTagItem {
  std::string begin;
  std::string schema;
  std::string end;

  bool operator==(const StructuralTagItem& other) const {
    return begin == other.begin && schema == other.schema && end == other.end;
  }
};

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
 * docs in Grammar::Impl::RuleExprType.
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
class Grammar {
 public:
  /*!
   * \brief Get the EBNF string of the grammar.
   */
  std::string ToString() const;

  /*!
   * \brief Construct a BNF grammar with a EBNF-formatted string. The grammar will be normalized
   * (simplified) by default.
   * \param ebnf_string The EBNF-formatted string.
   * \param root_rule_name The name of the root rule.
   */
  static Grammar FromEBNF(
      const std::string& ebnf_string, const std::string& root_rule_name = "root"
  );

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
  static Grammar FromJSONSchema(
      const std::string& schema,
      bool any_whitespace = true,
      std::optional<int> indent = std::nullopt,
      std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
      bool strict_mode = true,
      bool print_converted_ebnf = false
  );

  /*!
   * \brief Construct a grammar from a regular expression string.
   * \param regex The regular expression string.
   * \param print_converted_ebnf This method will convert the regex to EBNF first. If this is true,
   * the converted EBNF string will be printed. For debugging purpose. Default: false.
   */
  static Grammar FromRegex(const std::string& regex, bool print_converted_ebnf = false);

  /*!
   * \brief Construct a grammar from a regular expression string.
   * \param regex The regular expression string.
   */
  static Grammar FromStructuralTag(
      const std::vector<StructuralTagItem>& tags, const std::vector<std::string>& triggers
  );

  /*!
   * \brief Get the grammar of standard JSON format. We have built-in support for JSON.
   */
  static Grammar BuiltinJSONGrammar();

  /*!
   * \brief Create a grammar that matches any of the grammars in the list. That is equivalent to
   * using the `|` operator to concatenate the grammars in the list.
   * \param grammars The grammars to create the union of.
   * \returns The union of the grammars.
   */
  static Grammar Union(const std::vector<Grammar>& grammars);

  /*!
   * \brief Create a grammar that matches the concatenation of the grammars in the list. That is
   * equivalent to using the `+` operator to concatenate the grammars in the list.
   * \param grammars The grammars to create the concatenation of.
   * \returns The concatenation of the grammars.
   */
  static Grammar Concat(const std::vector<Grammar>& grammars);

  /*! \brief Print a BNF grammar. */
  friend std::ostream& operator<<(std::ostream& os, const Grammar& grammar);

  XGRAMMAR_DEFINE_PIMPL_METHODS(Grammar);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_H_
