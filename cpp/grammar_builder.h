/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_builder.h
 * \brief The header for the building the BNF AST.
 */

#ifndef XGRAMMAR_GRAMMAR_BUILDER_H_
#define XGRAMMAR_GRAMMAR_BUILDER_H_

#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "grammar_impl.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

/*!
 * \brief Nestable build-time descriptions of grammar expression trees.
 *
 * These lightweight specs let callers describe an expression tree and materialize it directly
 * into Grammar's compact AST storage. Existing expression ids can be mixed with specs, and
 * SelfRef can refer to the rule currently being defined.
 *
 * \code
 * using namespace grammar_spec;
 * builder.AddRule(
 *     "root",
 *     Choices(Sequence(ByteString("abc"), RuleRef(other_rule_id)), ByteString("def"))
 * );
 * \endcode
 */
namespace grammar_spec {

class GrammarExprSpec;
struct ExprId;
struct ByteString;
struct Regex;
struct CharacterClass;
struct CharacterClassStar;
struct EmptyStr;
struct RuleRef;
struct SelfRef;
struct Token;
struct ExcludeToken;
struct Sequence;
struct Choices;
struct Repeat;

namespace detail {

template <typename T, typename... Ts>
inline constexpr bool is_one_of_v = (std::is_same_v<T, Ts> || ...);

template <typename T>
inline constexpr bool is_spec_arg_v = is_one_of_v<
                                          std::decay_t<T>,
                                          GrammarExprSpec,
                                          ExprId,
                                          ByteString,
                                          Regex,
                                          CharacterClass,
                                          CharacterClassStar,
                                          EmptyStr,
                                          RuleRef,
                                          SelfRef,
                                          Token,
                                          ExcludeToken,
                                          Grammar::Impl::TagDispatch,
                                          Grammar::Impl::TokenTagDispatch,
                                          Sequence,
                                          Choices,
                                          Repeat> ||
                                      std::is_integral_v<std::decay_t<T>>;

}  // namespace detail

/*! \brief Reference an expression that has already been materialized in a GrammarBuilder. */
struct ExprId {
  int32_t id;
  ExprId(int32_t id) : id(id) {}
};

/*! \brief A UTF-8 byte string expression. */
struct ByteString {
  std::string str;
  explicit ByteString(std::string str) : str(std::move(str)) {}
};

/*! \brief A regex expression. */
struct Regex {
  std::string pattern;
  bool json_string;
  explicit Regex(std::string pattern, bool json_string = false)
      : pattern(std::move(pattern)), json_string(json_string) {}
};

struct CharacterClassElement {
  int32_t lower;
  int32_t upper;
};

/*! \brief A character class expression. */
struct CharacterClass {
  std::vector<CharacterClassElement> elements;
  bool is_negative;
  explicit CharacterClass(std::vector<CharacterClassElement> elements, bool is_negative = false)
      : elements(std::move(elements)), is_negative(is_negative) {}
};

/*! \brief A repeated character class expression. */
struct CharacterClassStar {
  std::vector<CharacterClassElement> elements;
  bool is_negative;
  explicit CharacterClassStar(std::vector<CharacterClassElement> elements, bool is_negative = false)
      : elements(std::move(elements)), is_negative(is_negative) {}
};

struct EmptyStr {};

/*! \brief Reference an existing rule. */
struct RuleRef {
  int32_t rule_id;
  explicit RuleRef(int32_t rule_id) : rule_id(rule_id) {}
};

/*! \brief Reference the rule currently being materialized. */
struct SelfRef {};

/*! \brief Match any one of the listed token ids. */
struct Token {
  std::vector<int32_t> token_ids;
  explicit Token(std::vector<int32_t> token_ids) : token_ids(std::move(token_ids)) {}
};

/*! \brief Match any token except the listed token ids. */
struct ExcludeToken {
  std::vector<int32_t> token_ids;
  explicit ExcludeToken(std::vector<int32_t> token_ids) : token_ids(std::move(token_ids)) {}
};

/*! \brief A tag dispatch expression with its referenced rules already allocated. */
using TagDispatch = Grammar::Impl::TagDispatch;

/*! \brief A token tag dispatch expression with its referenced rules already allocated. */
using TokenTagDispatch = Grammar::Impl::TokenTagDispatch;

struct Sequence {
  std::vector<GrammarExprSpec> elements;

  explicit Sequence(std::vector<GrammarExprSpec> elements) : elements(std::move(elements)) {}

  template <
      typename... Args,
      typename = std::enable_if_t<
          (detail::is_spec_arg_v<Args> && ...) &&
          !(sizeof...(Args) == 1 && (std::is_same_v<std::decay_t<Args>, Sequence> && ...))>>
  Sequence(Args&&... args) {
    elements.reserve(sizeof...(Args));
    (elements.emplace_back(std::forward<Args>(args)), ...);
  }
};

struct Choices {
  std::vector<GrammarExprSpec> choices;

  explicit Choices(std::vector<GrammarExprSpec> choices) : choices(std::move(choices)) {}

  template <
      typename... Args,
      typename = std::enable_if_t<
          (detail::is_spec_arg_v<Args> && ...) &&
          !(sizeof...(Args) == 1 && (std::is_same_v<std::decay_t<Args>, Choices> && ...))>>
  Choices(Args&&... args) {
    choices.reserve(sizeof...(Args));
    (choices.emplace_back(std::forward<Args>(args)), ...);
  }
};

struct Repeat {
  // GrammarExprSpec is incomplete at this point, so store the single child in a vector.
  std::vector<GrammarExprSpec> element;
  int32_t min_repeat_count;
  int32_t max_repeat_count;

  template <typename T, typename = std::enable_if_t<detail::is_spec_arg_v<T>>>
  Repeat(T&& element, int32_t min_repeat_count, int32_t max_repeat_count)
      : min_repeat_count(min_repeat_count), max_repeat_count(max_repeat_count) {
    this->element.emplace_back(std::forward<T>(element));
  }
};

/*! \brief A nestable, build-time description of a grammar expression tree. */
class GrammarExprSpec {
 public:
  using Variant = std::variant<
      ExprId,
      ByteString,
      Regex,
      CharacterClass,
      CharacterClassStar,
      EmptyStr,
      RuleRef,
      SelfRef,
      Token,
      ExcludeToken,
      Grammar::Impl::TagDispatch,
      Grammar::Impl::TokenTagDispatch,
      Sequence,
      Choices,
      Repeat>;

  GrammarExprSpec(int32_t grammar_expr_id) : value(ExprId(grammar_expr_id)) {}

  template <
      typename T,
      typename = std::enable_if_t<
          detail::is_spec_arg_v<T> && !std::is_same_v<std::decay_t<T>, GrammarExprSpec> &&
          !std::is_integral_v<std::decay_t<T>>>>
  GrammarExprSpec(T&& value) : value(std::forward<T>(value)) {}

  Variant value;
};

}  // namespace grammar_spec

using grammar_spec::GrammarExprSpec;

/*!
 * \brief Helper class to build a BNF grammar.
 */
class GrammarBuilder {
 public:
  using Rule = Grammar::Impl::Rule;
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  using GrammarExpr = Grammar::Impl::GrammarExpr;

  /*! \brief One element of a character class, containing a lower and a upper bound. Both bounds are
   * inclusive.
   */
  struct CharacterClassElement {
    int32_t lower;
    int32_t upper;
  };

  /*! \brief Default constructor. Creates a new grammar object. */
  GrammarBuilder();

  /*! \brief Constructor. Creates a new grammar object from an existing grammar. */
  GrammarBuilder(const Grammar& grammar);

  /*!
   * \brief Get the result grammar. This function will also set the root rule to the rule with the
   * specified name. The rule should be already added to the grammar.
   * \param root_rule_name The name of the root rule. Default is "root".
   */
  Grammar Get(const std::string& root_rule_name = "root");

  /*!
   * \brief Get the result grammar. This function will also set the root rule to the rule with
   * the specified id. The rule should be already added to the grammar.
   * \param root_rule_id The id of the root rule.
   */
  Grammar Get(int32_t root_rule_id);

  /****************** GrammarExpr handling ******************/

  /*! \brief Add a grammar_expr and return the grammar_expr id. */
  int32_t AddGrammarExpr(const GrammarExpr& grammar_expr);

  /*!
   * \brief Add a GrammarExpr for string stored in bytes.
   * \param bytes A vector of int32_t, each representing a byte (0~255) in the string.
   * The string is stored in int32 vector to match the storage format of the grammar.
   */
  int32_t AddByteString(const std::vector<int32_t>& bytes);

  /*!
   * \brief Add a GrammarExpr for string stored in bytes.
   * \param str The string to be added.
   */
  int32_t AddByteString(const std::string& str);

  /*!
   * \brief Add a GrammarExpr for a regex. The pattern is stored as-is and compiled into an
   * automaton by GrammarFSMBuilder.
   * \param regex_str The regex pattern string.
   * \param json_string Whether the regex matches the body of a JSON string literal. If true,
   * the characters that must be escaped in a JSON string (the control characters, '"' and
   * '\\') are excluded from every character match of the compiled automaton.
   */
  int32_t AddRegex(const std::string& regex_str, bool json_string = false);

  /*!
   * \brief Add a GrammarExpr for a character class.
   * \param elements A vector of CharacterClassElement, each containing a lower and a upper bound.
   * \param is_negative Whether the character class is negated.
   */
  int32_t AddCharacterClass(
      const std::vector<CharacterClassElement>& elements, bool is_negative = false
  );

  /*!
   * \brief Add a GrammarExpr for a star quantifier of a character class.
   * \param elements A vector of CharacterClassElement, each containing a lower and a upper bound.
   * \param is_negative Whether the character class is negated.
   */
  int32_t AddCharacterClassStar(
      const std::vector<CharacterClassElement>& elements, bool is_negative = false
  );

  /*! \brief Add a GrammarExpr for empty string.*/
  int32_t AddEmptyStr();

  /*! \brief Add a GrammarExpr for kToken (token-level matching). */
  int32_t AddTokenSet(const std::vector<int32_t>& token_ids);

  /*! \brief Add a GrammarExpr for kExcludeToken (excluded token-level matching). */
  int32_t AddExcludeTokenSet(const std::vector<int32_t>& token_ids);

  /*! \brief Add a GrammarExpr for rule reference.*/
  int32_t AddRuleRef(int32_t rule_id);

  /*! \brief Add a GrammarExpr for GrammarExpr sequence.*/
  int32_t AddSequence(const std::vector<int32_t>& elements);

  /*! \brief Add a GrammarExpr for GrammarExpr choices.*/
  int32_t AddChoices(const std::vector<int32_t>& choices);

  /*!
   * \brief Add a GrammarExpr for tag dispatch.
   * \param tag_dispatch_list A list of pairs of tag_expr_id and rule_id.
   */
  int32_t AddTagDispatch(const Grammar::Impl::TagDispatch& tag_dispatch);

  /*! \brief Encode a TokenTagDispatch struct into a kTokenTagDispatch expr. */
  int32_t AddTokenTagDispatch(const Grammar::Impl::TokenTagDispatch& token_tag_dispatch);

  int32_t AddRepeat(int32_t ref_rule_id, int32_t min_repeat_count, int32_t max_repeat_count);

  /*!
   * \brief Add a repeat GrammarExpr from an arbitrary grammar expression. If the expression is
   * not a rule reference, a new rule is created to wrap it.
   * \param cur_rule_name Name hint for generated rules.
   * \param grammar_expr_id The expression to repeat.
   * \param min_repeat_count Minimum repeat count (inclusive).
   * \param max_repeat_count Maximum repeat count (inclusive), or -1 for unbounded.
   */
  int32_t AddRepeatFromExpr(
      const std::string& cur_rule_name,
      int32_t grammar_expr_id,
      int32_t min_repeat_count,
      int32_t max_repeat_count
  );

  /*! \brief Materialize a nested expression description and return its expression id. */
  int32_t AddExpr(const GrammarExprSpec& spec, const std::string& name_hint = "expr");

  /*! \brief Materialize a nested expression with SelfRef bound to self_rule_id. */
  int32_t AddExpr(
      const GrammarExprSpec& spec, int32_t self_rule_id, const std::string& name_hint = "expr"
  );

  /*! \brief Get the number of grammar_exprs. */
  int32_t NumGrammarExprs() const;

  /*! \brief Get the grammar_expr with the given id. */
  GrammarExpr GetGrammarExpr(int32_t grammar_expr_id);

  /****************** Rule handling ******************/

  /*! \brief Add a rule and return the rule id. */
  int32_t AddRule(const Rule& rule);

  int32_t AddRule(const std::string& name, int32_t body_expr_id);

  int32_t AddRuleWithHint(const std::string& name_hint, int32_t body_expr_id);

  /*! \brief Add a rule by materializing a nested expression description. */
  int32_t AddRule(const std::string& name, const GrammarExprSpec& spec);

  /*! \brief Add a uniquely named rule by materializing a nested expression description. */
  int32_t AddRuleWithHint(const std::string& name_hint, const GrammarExprSpec& spec);

  int32_t NumRules() const;

  /*! \brief Get the rule with the given id. */
  const Rule& GetRule(int32_t rule_id) const;

  /*!
   * \brief Add an rule without body, and return the rule id. The rule body should be set later
   * with GrammarBuilder::UpdateRuleBody. This method is useful for cases where the rule id is
   * required to build the rule body.
   * \sa GrammarBuilder::UpdateRuleBody
   */
  int32_t AddEmptyRule(const std::string& name);

  int32_t AddEmptyRuleWithHint(const std::string& name_hint);

  /*!
   * \brief Update the rule body of the given rule, specified by rule id. Can be used to set the
   * rule body of a rule inserted by GrammarBuilder::AddEmptyRule.
   */
  void UpdateRuleBody(int32_t rule_id, int32_t body_expr_id);

  /*!
   * \brief Update the rule body of the given rule, specified by rule name. Can be used to set the
   * rule body of a rule inserted by GrammarBuilder::AddEmptyRule.
   */
  void UpdateRuleBody(std::string rule_name, int32_t body_expr_id);

  /*!
   * \brief Add a lookahead assertion to a rule referred by the given rule_id. The lookahead
   * assertion should be a sequence GrammarExpr id. An id of -1 means no lookahead assertion.
   */
  void UpdateLookaheadAssertion(int32_t rule_id, int32_t lookahead_assertion_id);

  void UpdateLookaheadExact(int32_t rule_id, bool is_exact = true);

  /*!
   * \brief Add a lookahead assertion to a rule referred by the given name. The lookahead
   * assertion should be a sequence GrammarExpr id. An id of -1 means no lookahead assertion.
   */
  void UpdateLookaheadAssertion(std::string rule_name, int32_t lookahead_assertion_id);

  /*!
   * \brief Find a name for a new rule starting with the given name hint. Some integer suffix (_1,
   * _2, ...) may be added to avoid name conflict.
   */
  std::string GetNewRuleName(const std::string& name_hint);

  /*!
   * \brief Get the rule id of the rule with the given name. Return -1 if not found.
   */
  int32_t GetRuleId(const std::string& name) const;

 private:
  int32_t AddExprImpl(
      const GrammarExprSpec& spec, int32_t self_rule_id, const std::string& name_hint
  );

  // Mutable pointer to the grammar object.
  std::shared_ptr<Grammar::Impl> grammar_;
  // Map from rule name to rule id.
  std::unordered_map<std::string, int32_t> rule_name_to_id_;
  // Cache of next suffix index per name_hint for GetNewRuleName.
  std::unordered_map<std::string, int> next_cnt_per_hint_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_BUILDER_H_
