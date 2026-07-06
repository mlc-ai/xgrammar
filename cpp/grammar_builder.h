/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_builder.h
 * \brief The header for the building the BNF AST.
 */

#ifndef XGRAMMAR_GRAMMAR_BUILDER_H_
#define XGRAMMAR_GRAMMAR_BUILDER_H_

#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "grammar_impl.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

/*!
 * \brief A lightweight, nestable description of a grammar expression tree.
 *
 * A GrammarExprSpec is built with the static factory methods and can be nested arbitrarily:
 *
 * \code
 * using Spec = GrammarExprSpec;
 * auto spec = Spec::Choices(
 *     Spec::Sequence(Spec::ByteString("abc"), Spec::RuleRef(other_rule_id)),
 *     Spec::ByteString("def"),
 *     Spec::Sequence(Spec::ByteString("ghi"), Spec::SelfRef())
 * );
 * int32_t rule_id = builder.AddRule("my_rule", spec);
 * \endcode
 *
 * An int32_t (an already-added grammar expr id) converts implicitly to a GrammarExprSpec, so
 * existing expr ids can be mixed with nested specs, e.g. Choices(id1, id2, Spec::EmptyStr()).
 *
 * SelfRef() refers to the rule currently being defined, and is only valid when the spec is
 * materialized through GrammarBuilder::AddRule / AddRuleWithHint, or through an AddExpr overload
 * that binds the self rule id explicitly.
 *
 * The spec is a build-time-only description: it is materialized into grammar exprs by
 * GrammarBuilder, and can be discarded afterwards.
 */
class GrammarExprSpec {
 public:
  using CharacterClassElement = Grammar::Impl::CharacterClassElement;

  /*! \brief Implicitly wraps an already-added grammar expr id. */
  GrammarExprSpec(int32_t grammar_expr_id) : kind_(Kind::kExprId), id_(grammar_expr_id) {}

  /*! \brief A byte string expr. Supports UTF-8 strings. */
  static GrammarExprSpec ByteString(std::string str) {
    GrammarExprSpec spec(Kind::kByteString);
    spec.str_ = std::move(str);
    return spec;
  }

  /*! \brief A character class expr, e.g. [a-z] or [^a-z]. */
  static GrammarExprSpec CharacterClass(
      std::vector<CharacterClassElement> elements, bool is_negative = false
  ) {
    GrammarExprSpec spec(Kind::kCharacterClass);
    spec.cc_elements_ = std::move(elements);
    spec.is_negative_ = is_negative;
    return spec;
  }

  /*! \brief A character class star expr, e.g. [a-z]* or [^a-z]*. */
  static GrammarExprSpec CharacterClassStar(
      std::vector<CharacterClassElement> elements, bool is_negative = false
  ) {
    GrammarExprSpec spec(Kind::kCharacterClassStar);
    spec.cc_elements_ = std::move(elements);
    spec.is_negative_ = is_negative;
    return spec;
  }

  /*! \brief An empty string expr. */
  static GrammarExprSpec EmptyStr() { return GrammarExprSpec(Kind::kEmptyStr); }

  /*! \brief A reference to the rule with the given id. */
  static GrammarExprSpec RuleRef(int32_t rule_id) {
    GrammarExprSpec spec(Kind::kRuleRef);
    spec.id_ = rule_id;
    return spec;
  }

  /*! \brief A reference to the rule currently being defined. Only valid when materialized with
   * a bound self rule id. \sa GrammarBuilder::AddRule */
  static GrammarExprSpec SelfRef() { return GrammarExprSpec(Kind::kSelfRef); }

  /*! \brief A sequence expr. Elements are matched one after another. */
  static GrammarExprSpec Sequence(std::vector<GrammarExprSpec> elements) {
    GrammarExprSpec spec(Kind::kSequence);
    spec.children_ = std::move(elements);
    return spec;
  }

  /*! \brief A sequence expr from a variadic list of specs (or expr ids). */
  template <
      typename... Args,
      typename = std::enable_if_t<(std::is_convertible_v<Args, GrammarExprSpec> && ...)>>
  static GrammarExprSpec Sequence(Args&&... elements) {
    return Sequence(MakeChildren(std::forward<Args>(elements)...));
  }

  /*! \brief A choices expr. Any one of the choices can be matched. */
  static GrammarExprSpec Choices(std::vector<GrammarExprSpec> choices) {
    GrammarExprSpec spec(Kind::kChoices);
    spec.children_ = std::move(choices);
    return spec;
  }

  /*! \brief A choices expr from a variadic list of specs (or expr ids). */
  template <
      typename... Args,
      typename = std::enable_if_t<(std::is_convertible_v<Args, GrammarExprSpec> && ...)>>
  static GrammarExprSpec Choices(Args&&... choices) {
    return Choices(MakeChildren(std::forward<Args>(choices)...));
  }

  /*!
   * \brief A repeat expr, matching the element between min_repeat_count and max_repeat_count
   * times. If the element is not a rule reference, a new rule will be created to wrap it.
   * \param max_repeat_count The maximum repeat count (inclusive), or -1 for unbounded.
   */
  static GrammarExprSpec Repeat(
      GrammarExprSpec element, int32_t min_repeat_count, int32_t max_repeat_count
  ) {
    GrammarExprSpec spec(Kind::kRepeat);
    spec.children_.push_back(std::move(element));
    spec.min_repeat_count_ = min_repeat_count;
    spec.max_repeat_count_ = max_repeat_count;
    return spec;
  }

  /*! \brief A token expr (token-level matching). Any one of the tokens can be matched. */
  static GrammarExprSpec Token(std::vector<int32_t> token_ids) {
    GrammarExprSpec spec(Kind::kToken);
    spec.token_ids_ = std::move(token_ids);
    return spec;
  }

  /*! \brief An exclude token expr. Any token except the given ones can be matched. */
  static GrammarExprSpec ExcludeToken(std::vector<int32_t> token_ids) {
    GrammarExprSpec spec(Kind::kExcludeToken);
    spec.token_ids_ = std::move(token_ids);
    return spec;
  }

  /*! \brief A tag dispatch expr. The referenced rules must already exist in the builder. */
  static GrammarExprSpec TagDispatch(Grammar::Impl::TagDispatch tag_dispatch) {
    GrammarExprSpec spec(Kind::kTagDispatch);
    spec.tag_dispatch_ = std::move(tag_dispatch);
    return spec;
  }

  /*! \brief A token tag dispatch expr. The referenced rules must already exist in the builder. */
  static GrammarExprSpec TokenTagDispatch(Grammar::Impl::TokenTagDispatch token_tag_dispatch) {
    GrammarExprSpec spec(Kind::kTokenTagDispatch);
    spec.token_tag_dispatch_ = std::move(token_tag_dispatch);
    return spec;
  }

 private:
  friend class GrammarBuilder;

  enum class Kind : int32_t {
    kExprId,
    kByteString,
    kCharacterClass,
    kCharacterClassStar,
    kEmptyStr,
    kRuleRef,
    kSelfRef,
    kSequence,
    kChoices,
    kRepeat,
    kToken,
    kExcludeToken,
    kTagDispatch,
    kTokenTagDispatch,
  };

  explicit GrammarExprSpec(Kind kind) : kind_(kind) {}

  template <typename... Args>
  static std::vector<GrammarExprSpec> MakeChildren(Args&&... args) {
    std::vector<GrammarExprSpec> children;
    children.reserve(sizeof...(Args));
    (children.push_back(GrammarExprSpec(std::forward<Args>(args))), ...);
    return children;
  }

  Kind kind_;
  /*! \brief The expr id for kExprId, or the rule id for kRuleRef. */
  int32_t id_ = -1;
  std::string str_;
  bool is_negative_ = false;
  std::vector<CharacterClassElement> cc_elements_;
  std::vector<GrammarExprSpec> children_;
  int32_t min_repeat_count_ = 0;
  int32_t max_repeat_count_ = -1;
  std::vector<int32_t> token_ids_;
  std::optional<Grammar::Impl::TagDispatch> tag_dispatch_;
  std::optional<Grammar::Impl::TokenTagDispatch> token_tag_dispatch_;
};

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
   * // list ::= "" | item list
   * builder.AddRule("list", Spec::Choices(
   *     Spec::EmptyStr(),
   *     Spec::Sequence(Spec::RuleRef(item_rule_id), Spec::SelfRef())
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
