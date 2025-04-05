/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_builder.h
 * \brief The header for the building the BNF AST.
 */

#ifndef XGRAMMAR_GRAMMAR_BUILDER_H_
#define XGRAMMAR_GRAMMAR_BUILDER_H_

#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <unordered_map>

#include "grammar_data_structure.h"

namespace xgrammar {

/*!
 * \brief Helper class to build a BNF grammar.
 */
class GrammarBuilder {
 public:
  using Rule = Grammar::Impl::Rule;
  using RuleExprType = Grammar::Impl::RuleExprType;
  using RuleExpr = Grammar::Impl::RuleExpr;

  /*! \brief Default constructor. Creates a new grammar object. */
  GrammarBuilder() : grammar_(std::make_shared<Grammar::Impl>()) {}

  /*! \brief Constructor. Creates a new grammar object from an existing grammar. */
  GrammarBuilder(const Grammar& grammar)
      : grammar_(std::make_shared<Grammar::Impl>(*grammar.operator->())) {
    for (int i = 0; i < static_cast<int>(grammar->NumRules()); ++i) {
      auto rule = grammar->GetRule(i);
      rule_name_to_id_[rule.name] = i;
    }
  }

  /*!
   * \brief Get the result grammar. This function will also set the root rule to the rule with the
   * specified name. The rule should be already added to the grammar.
   * \param root_rule_name The name of the root rule. Default is "root".
   */
  Grammar Get(const std::string& root_rule_name = "root") {
    int32_t root_rule_id = GetRuleId(root_rule_name);
    XGRAMMAR_CHECK(root_rule_id != -1)
        << "The root rule with name \"" << root_rule_name << "\" is not found.";
    return Get(root_rule_id);
  }

  /*!
   * \brief Get the result grammar. This function will also set the root rule to the rule with
   * the specified id. The rule should be already added to the grammar.
   * \param root_rule_id The id of the root rule.
   */
  Grammar Get(int32_t root_rule_id) {
    XGRAMMAR_CHECK(
        root_rule_id >= 0 && root_rule_id < static_cast<int32_t>(grammar_->rules_.size())
    ) << "The root rule id "
      << root_rule_id << " is out of bound.";
    grammar_->root_rule_id_ = root_rule_id;

    return Grammar(grammar_);
  }

  /****************** RuleExpr handling ******************/

  /*! \brief Add a rule_expr and return the rule_expr id. */
  int32_t AddRuleExpr(const RuleExpr& rule_expr) {
    grammar_->rule_expr_indptr_.push_back(grammar_->rule_expr_data_.size());
    grammar_->rule_expr_data_.push_back(static_cast<int32_t>(rule_expr.type));
    grammar_->rule_expr_data_.push_back(rule_expr.data_len);
    grammar_->rule_expr_data_.insert(
        grammar_->rule_expr_data_.end(), rule_expr.data, rule_expr.data + rule_expr.data_len
    );
    return static_cast<int32_t>(grammar_->rule_expr_indptr_.size()) - 1;
  }

  /*!
   * \brief Add a RuleExpr for string stored in bytes.
   * \param bytes A vector of int32_t, each representing a byte (0~255) in the string.
   * The string is stored in int32 vector to match the storage format of the grammar.
   */
  int32_t AddByteString(const std::vector<int32_t>& bytes) {
    return AddRuleExpr({RuleExprType::kByteString, bytes.data(), static_cast<int32_t>(bytes.size())}
    );
  }

  /*!
   * \brief Add a RuleExpr for string stored in bytes.
   * \param str The string to be added.
   */
  int32_t AddByteString(const std::string& str) {
    std::vector<int32_t> bytes;
    bytes.reserve(str.size());
    for (char c : str) {
      bytes.push_back(static_cast<int32_t>(c));
    }
    return AddRuleExpr({RuleExprType::kByteString, bytes.data(), static_cast<int32_t>(bytes.size())}
    );
  }

  /*!
   * \brief One element of a character class, containing a lower and a upper bound. Both bounds are
   * inclusive.
   */
  struct CharacterClassElement {
    int32_t lower;
    int32_t upper;
  };

  /*!
   * \brief Add a RuleExpr for a character class.
   * \param elements A vector of CharacterClassElement, each containing a lower and a upper bound.
   * \param is_negative Whether the character class is negated.
   */
  int32_t AddCharacterClass(
      const std::vector<CharacterClassElement>& elements, bool is_negative = false
  ) {
    std::vector<int32_t> data;
    data.reserve(1 + elements.size() * 2);
    data.push_back(static_cast<int32_t>(is_negative));
    for (const auto& range : elements) {
      data.push_back(range.lower);
      data.push_back(range.upper);
    }
    return AddRuleExpr(
        {RuleExprType::kCharacterClass, data.data(), static_cast<int32_t>(data.size())}
    );
  }

  /*!
   * \brief Add a RuleExpr for a star quantifier of a character class.
   * \param elements A vector of CharacterClassElement, each containing a lower and a upper bound.
   * \param is_negative Whether the character class is negated.
   */
  int32_t AddCharacterClassStar(
      const std::vector<CharacterClassElement>& elements, bool is_negative = false
  ) {
    std::vector<int32_t> data;
    data.reserve(1 + elements.size() * 2);
    data.push_back(static_cast<int32_t>(is_negative));
    for (const auto& range : elements) {
      data.push_back(range.lower);
      data.push_back(range.upper);
    }
    return AddRuleExpr(
        {RuleExprType::kCharacterClassStar, data.data(), static_cast<int32_t>(data.size())}
    );
  }

  /*! \brief Add a RuleExpr for empty string.*/
  int32_t AddEmptyStr() { return AddRuleExpr({RuleExprType::kEmptyStr, nullptr, 0}); }

  /*! \brief Add a RuleExpr for rule reference.*/
  int32_t AddRuleRef(int32_t rule_id) {
    std::vector<int32_t> data;
    data.push_back(rule_id);
    return AddRuleExpr({RuleExprType::kRuleRef, data.data(), static_cast<int32_t>(data.size())});
  }

  /*! \brief Add a RuleExpr for RuleExpr sequence.*/
  int32_t AddSequence(const std::vector<int32_t>& elements) {
    return AddRuleExpr(
        {RuleExprType::kSequence, elements.data(), static_cast<int32_t>(elements.size())}
    );
  }

  /*! \brief Add a RuleExpr for RuleExpr choices.*/
  int32_t AddChoices(const std::vector<int32_t>& choices) {
    return AddRuleExpr(
        {RuleExprType::kChoices, choices.data(), static_cast<int32_t>(choices.size())}
    );
  }

  /*!
   * \brief Add a RuleExpr for tag dispatch.
   * \param tag_dispatch_list A list of pairs of tag_expr_id and rule_id.
   */
  int32_t AddTagDispatch(const std::vector<std::pair<int32_t, int32_t>>& tag_dispatch_list) {
    std::vector<int32_t> data;
    data.reserve(tag_dispatch_list.size() * 2);
    for (const auto& [tag_expr_id, rule_id] : tag_dispatch_list) {
      data.push_back(tag_expr_id);
      data.push_back(rule_id);
    }
    return AddRuleExpr({RuleExprType::kTagDispatch, data.data(), static_cast<int32_t>(data.size())}
    );
  }

  size_t NumRuleExprs() const { return grammar_->NumRuleExprs(); }
  /*! \brief Get the rule_expr with the given id. */
  RuleExpr GetRuleExpr(int32_t rule_expr_id) { return grammar_->GetRuleExpr(rule_expr_id); }

  /****************** Rule handling ******************/

  /*! \brief Add a rule and return the rule id. */
  int32_t AddRule(const Rule& rule) {
    int32_t id = grammar_->rules_.size();
    auto rules = grammar_->rules_;
    grammar_->rules_.push_back(rule);
    XGRAMMAR_CHECK(rule_name_to_id_.count(rule.name) == 0);
    rule_name_to_id_[rule.name] = id;
    return id;
  }

  int32_t AddRule(const std::string& name, int32_t body_expr_id) {
    return AddRule({name, body_expr_id});
  }

  int32_t AddRuleWithHint(const std::string& name_hint, int32_t body_expr_id) {
    return AddRule({GetNewRuleName(name_hint), body_expr_id});
  }

  size_t NumRules() const { return grammar_->NumRules(); }

  /*! \brief Get the rule with the given id. */
  const Rule& GetRule(int32_t rule_id) const { return grammar_->rules_[rule_id]; }

  /*!
   * \brief Add an rule without body, and return the rule id. The rule body should be set later
   * with GrammarBuilder::UpdateRuleBody. This method is useful for cases where the rule id is
   * required to build the rule body.
   * \sa GrammarBuilder::UpdateRuleBody
   */
  int32_t AddEmptyRule(const std::string& name) { return AddRule({name, -1}); }

  /*!
   * \brief Update the rule body of the given rule, specified by rule id. Can be used to set the
   * rule body of a rule inserted by GrammarBuilder::AddEmptyRule.
   */
  void UpdateRuleBody(int32_t rule_id, int32_t body_expr_id) {
    XGRAMMAR_CHECK(rule_id >= 0 && rule_id < static_cast<int32_t>(grammar_->rules_.size()))
        << "Rule id " << rule_id << " is out of range.";
    grammar_->rules_[rule_id].body_expr_id = body_expr_id;
  }

  /*!
   * \brief Update the rule body of the given rule, specified by rule name. Can be used to set the
   * rule body of a rule inserted by GrammarBuilder::AddEmptyRule.
   */
  void UpdateRuleBody(std::string rule_name, int32_t body_expr_id) {
    int32_t rule_id = GetRuleId(rule_name);
    XGRAMMAR_CHECK(rule_id != -1) << "Rule " << rule_name << " is not found.";
    UpdateRuleBody(rule_id, body_expr_id);
  }

  /*!
   * \brief Add a lookahead assertion to a rule referred by the given rule_id. The lookahead
   * assertion should be a sequence RuleExpr id. An id of -1 means no lookahead assertion.
   */
  void AddLookaheadAssertion(int32_t rule_id, int32_t lookahead_assertion_id) {
    XGRAMMAR_CHECK(rule_id < static_cast<int32_t>(grammar_->rules_.size()))
        << "Rule id " << rule_id << " is out of range.";
    XGRAMMAR_CHECK(grammar_->rules_[rule_id].lookahead_assertion_id == -1)
        << "Rule " << rule_id << " already has a lookahead assertion.";
    grammar_->rules_[rule_id].lookahead_assertion_id = lookahead_assertion_id;
  }

  /*!
   * \brief Add a lookahead assertion to a rule referred by the given name. The lookahead
   * assertion should be a sequence RuleExpr id. An id of -1 means no lookahead assertion.
   */
  void AddLookaheadAssertion(std::string rule_name, int32_t lookahead_assertion_id) {
    int32_t rule_id = GetRuleId(rule_name);
    XGRAMMAR_CHECK(rule_id != -1) << "Rule " << rule_name << " is not found.";
    AddLookaheadAssertion(rule_id, lookahead_assertion_id);
  }

  /*!
   * \brief Find a name for a new rule starting with the given name hint. Some integer suffix (_1,
   * _2, ...) may be added to avoid name conflict.
   */
  std::string GetNewRuleName(const std::string& name_hint) {
    if (rule_name_to_id_.count(name_hint) == 0) {
      return name_hint;
    } else {
      int cnt = 1;
      while (rule_name_to_id_.count(name_hint + "_" + std::to_string(cnt)) != 0) {
        ++cnt;
      }
      return name_hint + "_" + std::to_string(cnt);
    }
  }

  /*!
   * \brief Get the rule id of the rule with the given name. Return -1 if not found.
   */
  int32_t GetRuleId(const std::string& name) const {
    auto it = rule_name_to_id_.find(name);
    if (it == rule_name_to_id_.end()) {
      return -1;
    } else {
      return it->second;
    }
  }

 private:
  // Mutable pointer to the grammar object.
  std::shared_ptr<Grammar::Impl> grammar_;
  // Map from rule name to rule id.
  std::unordered_map<std::string, int32_t> rule_name_to_id_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_BUILDER_H_
