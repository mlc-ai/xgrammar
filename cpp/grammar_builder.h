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
#include <unordered_map>
#include <vector>

#include "grammar_impl.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

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

  /*! \brief Get the number of grammar_exprs. */
  int32_t NumGrammarExprs() const;

  /*! \brief Get the grammar_expr with the given id. */
  GrammarExpr GetGrammarExpr(int32_t grammar_expr_id);

  /****************** Rule handling ******************/

  /*! \brief Add a rule and return the rule id. */
  int32_t AddRule(const Rule& rule);

  int32_t AddRule(const std::string& name, int32_t body_expr_id);

  int32_t AddRuleWithHint(const std::string& name_hint, int32_t body_expr_id);

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
  // Mutable pointer to the grammar object.
  std::shared_ptr<Grammar::Impl> grammar_;
  // Map from rule name to rule id.
  std::unordered_map<std::string, int32_t> rule_name_to_id_;
  // Cache of next suffix index per name_hint for GetNewRuleName.
  std::unordered_map<std::string, int> next_cnt_per_hint_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_BUILDER_H_
