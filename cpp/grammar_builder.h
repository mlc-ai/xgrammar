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
 * Every GrammarExprType has its own spec class with a trivial constructor that just stores its
 * arguments. A spec tree can be nested arbitrarily and is materialized into grammar exprs by
 * GrammarBuilder:
 *
 * \code
 * using namespace grammar_spec;
 * int32_t rule_id = builder.AddRule("my_rule", Choices(
 *     Sequence(ByteString("abc"), RuleRef(other_rule_id)),
 *     ByteString("def"),
 *     Sequence(ByteString("ghi"), SelfRef())
 * ));
 * \endcode
 *
 * Any spec class (and int32_t, wrapping an already-added grammar expr id) converts implicitly
 * to GrammarExprSpec, so existing expr ids can be mixed with nested specs, e.g.
 * Choices(id1, id2, EmptyStr()).
 *
 * SelfRef refers to the rule currently being defined, and is only valid when the spec is
 * materialized through GrammarBuilder::AddRule / AddRuleWithHint, or through an AddExpr overload
 * that binds the self rule id explicitly.
 *
 * The specs are build-time-only descriptions: they are materialized into grammar exprs by
 * GrammarBuilder, and can be discarded afterwards.
 */
namespace grammar_spec {

class GrammarExprSpec;
struct ExprId;
struct ByteString;
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

/*!
 * \brief Whether T (after decay) can be used to construct a GrammarExprSpec: one of the spec
 * classes, GrammarExprSpec itself, or an integral grammar expr id.
 * \note This trait is spelled out as a type list instead of
 * std::is_convertible<T, GrammarExprSpec>, because the latter would recursively instantiate
 * itself through the converting constructors of GrammarExprSpec and Sequence/Choices/Repeat.
 */
template <typename T>
inline constexpr bool is_spec_arg_v = is_one_of_v<
                                          std::decay_t<T>,
                                          GrammarExprSpec,
                                          ExprId,
                                          ByteString,
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

/*! \brief Wraps an already-added grammar expr id. int32_t converts implicitly to it. */
struct ExprId {
  int32_t id;
  ExprId(int32_t id) : id(id) {}
};

/*! \brief A byte string expr. Supports UTF-8 strings. */
struct ByteString {
  std::string str;
  explicit ByteString(std::string str) : str(std::move(str)) {}
};

/*! \brief A character class expr, e.g. [a-z] or [^a-z]. */
struct CharacterClass {
  std::vector<Grammar::Impl::CharacterClassElement> elements;
  bool is_negative;
  explicit CharacterClass(
      std::vector<Grammar::Impl::CharacterClassElement> elements, bool is_negative = false
  )
      : elements(std::move(elements)), is_negative(is_negative) {}
};

/*! \brief A character class star expr, e.g. [a-z]* or [^a-z]*. */
struct CharacterClassStar {
  std::vector<Grammar::Impl::CharacterClassElement> elements;
  bool is_negative;
  explicit CharacterClassStar(
      std::vector<Grammar::Impl::CharacterClassElement> elements, bool is_negative = false
  )
      : elements(std::move(elements)), is_negative(is_negative) {}
};

/*! \brief An empty string expr. */
struct EmptyStr {};

/*! \brief A reference to the rule with the given id. */
struct RuleRef {
  int32_t rule_id;
  explicit RuleRef(int32_t rule_id) : rule_id(rule_id) {}
};

/*! \brief A reference to the rule currently being defined. Only valid when materialized with a
 * bound self rule id. \sa GrammarBuilder::AddRule */
struct SelfRef {};

/*! \brief A token expr (token-level matching). Any one of the tokens can be matched. */
struct Token {
  std::vector<int32_t> token_ids;
  explicit Token(std::vector<int32_t> token_ids) : token_ids(std::move(token_ids)) {}
};

/*! \brief An exclude token expr. Any token except the given ones can be matched. */
struct ExcludeToken {
  std::vector<int32_t> token_ids;
  explicit ExcludeToken(std::vector<int32_t> token_ids) : token_ids(std::move(token_ids)) {}
};

/*! \brief A tag dispatch expr, described by its typed representation. The referenced rules must
 * already exist in the builder. */
using TagDispatch = Grammar::Impl::TagDispatch;

/*! \brief A token tag dispatch expr, described by its typed representation. The referenced rules
 * must already exist in the builder. */
using TokenTagDispatch = Grammar::Impl::TokenTagDispatch;

/*! \brief A sequence expr. Elements are matched one after another. */
struct Sequence {
  std::vector<GrammarExprSpec> elements;

  explicit Sequence(std::vector<GrammarExprSpec> elements) : elements(std::move(elements)) {}

  /*! \brief Constructs from a variadic list of specs (or expr ids). */
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

/*! \brief A choices expr. Any one of the choices can be matched. */
struct Choices {
  std::vector<GrammarExprSpec> choices;

  explicit Choices(std::vector<GrammarExprSpec> choices) : choices(std::move(choices)) {}

  /*! \brief Constructs from a variadic list of specs (or expr ids). */
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

/*!
 * \brief A repeat expr, matching the element between min_repeat_count and max_repeat_count
 * times (max_repeat_count == -1 means unbounded). If the element is not a rule reference, a new
 * rule will be created to wrap it on materialization.
 */
struct Repeat {
  /*! \brief The repeated element. Stored in a vector because GrammarExprSpec is incomplete here;
   * it always contains exactly one element. */
  std::vector<GrammarExprSpec> element;
  int32_t min_repeat_count;
  int32_t max_repeat_count;

  template <typename T, typename = std::enable_if_t<detail::is_spec_arg_v<T>>>
  Repeat(T&& element, int32_t min_repeat_count, int32_t max_repeat_count)
      : min_repeat_count(min_repeat_count), max_repeat_count(max_repeat_count) {
    this->element.emplace_back(std::forward<T>(element));
  }
};

/*!
 * \brief The type-erased wrapper holding any of the spec classes. All spec classes and int32_t
 * (an already-added grammar expr id) convert implicitly to it.
 */
class GrammarExprSpec {
 public:
  using Variant = std::variant<
      ExprId,
      ByteString,
      CharacterClass,
      CharacterClassStar,
      EmptyStr,
      RuleRef,
      SelfRef,
      Token,
      ExcludeToken,
      TagDispatch,
      TokenTagDispatch,
      Sequence,
      Choices,
      Repeat>;

  /*! \brief Implicitly wraps an already-added grammar expr id. */
  GrammarExprSpec(int32_t grammar_expr_id) : value(ExprId(grammar_expr_id)) {}

  /*! \brief Implicitly wraps any of the spec classes. */
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
 * \brief Helper class to build a BNF grammar. It is the unified entry for constructing the
 * grammar AST: it can add single grammar exprs, materialize nested GrammarExprSpecs, add whole
 * sub grammars, and normalize the result grammar on Get().
 */
class GrammarBuilder {
 public:
  using Rule = Grammar::Impl::Rule;
  using GrammarExprType = Grammar::Impl::GrammarExprType;
  using GrammarExpr = Grammar::Impl::GrammarExpr;

  /*! \brief One element of a character class, containing a lower and a upper bound. Both bounds
   * are inclusive. */
  using CharacterClassElement = Grammar::Impl::CharacterClassElement;

  /*! \brief Default constructor. Creates a new grammar object. */
  GrammarBuilder();

  /*! \brief Constructor. Creates a new grammar object from an existing grammar. */
  GrammarBuilder(const Grammar& grammar);

  /*!
   * \brief Get the result grammar. This function will also set the root rule to the rule with the
   * specified name. The rule should be already added to the grammar.
   * \param root_rule_name The name of the root rule. Default is "root".
   * \param normalize If true, the grammar will be normalized (GrammarNormalizer) before being
   * returned.
   */
  Grammar Get(const std::string& root_rule_name = "root", bool normalize = false);

  /*!
   * \brief Get the result grammar. This function will also set the root rule to the rule with
   * the specified id. The rule should be already added to the grammar.
   * \param root_rule_id The id of the root rule.
   * \param normalize If true, the grammar will be normalized (GrammarNormalizer) before being
   * returned.
   */
  Grammar Get(int32_t root_rule_id, bool normalize = false);

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
  int32_t AddToken(const std::vector<int32_t>& token_ids);

  /*! \brief Add a GrammarExpr for kExcludeToken (excluded token-level matching). */
  int32_t AddExcludeToken(const std::vector<int32_t>& token_ids);

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

  /*! \brief Add a GrammarExpr for repeat from its typed representation. */
  int32_t AddRepeat(const Grammar::Impl::Repeat& repeat);

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

  /*!
   * \brief Materialize a nested GrammarExprSpec into grammar exprs and return the id of the
   * top-level expr. SelfRef is not allowed in the spec.
   * \param name_hint Name hint for auxiliary rules created during materialization (e.g. for
   * Repeat elements).
   */
  int32_t AddExpr(const GrammarExprSpec& spec, const std::string& name_hint = "expr");

  /*!
   * \brief Materialize a nested GrammarExprSpec, with SelfRef bound to the given rule id.
   */
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

  /*!
   * \brief Add a new rule whose body is the materialization of the given GrammarExprSpec, and
   * return the rule id. SelfRef in the spec refers to the rule being added, so recursive rules
   * can be built in one call:
   *
   * \code
   * using namespace grammar_spec;
   * // list ::= "" | item list
   * builder.AddRule("list", Choices(
   *     EmptyStr(),
   *     Sequence(RuleRef(item_rule_id), SelfRef())
   * ));
   * \endcode
   */
  int32_t AddRule(const std::string& name, const GrammarExprSpec& spec);

  /*! \brief Same as AddRule(name, spec), but the rule name is derived from the name hint. */
  int32_t AddRuleWithHint(const std::string& name_hint, const GrammarExprSpec& spec);

  /*!
   * \brief Add all rules of another grammar into this builder. Rules are renamed on name
   * conflicts, and all rule references (including those in tag dispatches and repeats) are
   * remapped to the new rule ids.
   * \return The new rule id of the sub grammar's root rule.
   */
  int32_t AddSubGrammar(const Grammar& sub_grammar);

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
  /*!
   * \brief Materialize a GrammarExprSpec into grammar exprs.
   * \param self_rule_id The rule id SelfRef is bound to, or -1 if SelfRef is not allowed.
   * \param name_hint Name hint for auxiliary rules created during materialization.
   */
  int32_t AddExprImpl(
      const GrammarExprSpec& spec, int32_t self_rule_id, const std::string& name_hint
  );

  /*! \brief Recursively copy a grammar expr of another grammar into this builder, remapping
   * rule ids with new_rule_ids. */
  int32_t CopySubGrammarExpr(
      const Grammar::Impl& sub_grammar,
      int32_t grammar_expr_id,
      const std::vector<int32_t>& new_rule_ids
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
