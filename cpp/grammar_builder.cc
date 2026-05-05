/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/grammar_builder.cc
 */

#include "grammar_builder.h"

#include <cstdint>
#include <string>
#include <vector>

#include "support/logging.h"

namespace xgrammar {

/****************** GrammarBuilder ******************/

GrammarBuilder::GrammarBuilder() : grammar_(std::make_shared<Grammar::Impl>()) {}

GrammarBuilder::GrammarBuilder(const Grammar& grammar)
    : grammar_(std::make_shared<Grammar::Impl>(*grammar.operator->())) {
  for (int i = 0; i < static_cast<int>(grammar->NumRules()); ++i) {
    auto rule = grammar->GetRule(i);
    rule_name_to_id_[rule.name] = i;
  }
}

Grammar GrammarBuilder::Get(const std::string& root_rule_name) {
  int32_t root_rule_id = GetRuleId(root_rule_name);
  XGRAMMAR_CHECK(root_rule_id != -1)
      << "The root rule with name \"" << root_rule_name << "\" is not found.";
  return Get(root_rule_id);
}

Grammar GrammarBuilder::Get(int32_t root_rule_id) {
  XGRAMMAR_CHECK(root_rule_id >= 0 && root_rule_id < static_cast<int32_t>(grammar_->rules_.size()))
      << "The root rule id " << root_rule_id << " is out of bound.";
  grammar_->root_rule_id_ = root_rule_id;
  return Grammar(grammar_);
}

int32_t GrammarBuilder::AddGrammarExpr(const GrammarExpr& grammar_expr) {
  grammar_->grammar_expr_indptr_.push_back(grammar_->grammar_expr_data_.size());
  grammar_->grammar_expr_data_.push_back(static_cast<int32_t>(grammar_expr.type));
  grammar_->grammar_expr_data_.push_back(grammar_expr.data_len);
  grammar_->grammar_expr_data_.insert(
      grammar_->grammar_expr_data_.end(),
      grammar_expr.data,
      grammar_expr.data + grammar_expr.data_len
  );
  return static_cast<int32_t>(grammar_->grammar_expr_indptr_.size()) - 1;
}

int32_t GrammarBuilder::AddByteString(const std::vector<int32_t>& bytes) {
  return AddGrammarExpr(
      {GrammarExprType::kByteString, bytes.data(), static_cast<int32_t>(bytes.size())}
  );
}

int32_t GrammarBuilder::AddByteString(const std::string& str) {
  std::vector<int32_t> bytes;
  bytes.reserve(str.size());
  for (char c : str) {
    bytes.push_back(static_cast<int32_t>(static_cast<uint8_t>(c)));
  }
  return AddGrammarExpr(
      {GrammarExprType::kByteString, bytes.data(), static_cast<int32_t>(bytes.size())}
  );
}

int32_t GrammarBuilder::AddCharacterClass(
    const std::vector<CharacterClassElement>& elements, bool is_negative
) {
  std::vector<int32_t> data;
  data.reserve(1 + elements.size() * 2);
  data.push_back(static_cast<int32_t>(is_negative));
  for (const auto& range : elements) {
    data.push_back(range.lower);
    data.push_back(range.upper);
  }
  return AddGrammarExpr(
      {GrammarExprType::kCharacterClass, data.data(), static_cast<int32_t>(data.size())}
  );
}

int32_t GrammarBuilder::AddCharacterClassStar(
    const std::vector<CharacterClassElement>& elements, bool is_negative
) {
  std::vector<int32_t> data;
  data.reserve(1 + elements.size() * 2);
  data.push_back(static_cast<int32_t>(is_negative));
  for (const auto& range : elements) {
    data.push_back(range.lower);
    data.push_back(range.upper);
  }
  return AddGrammarExpr(
      {GrammarExprType::kCharacterClassStar, data.data(), static_cast<int32_t>(data.size())}
  );
}

int32_t GrammarBuilder::AddEmptyStr() {
  return AddGrammarExpr({GrammarExprType::kEmptyStr, nullptr, 0});
}

int32_t GrammarBuilder::AddTokenSet(const std::vector<int32_t>& token_ids) {
  return AddGrammarExpr(
      {GrammarExprType::kToken, token_ids.data(), static_cast<int32_t>(token_ids.size())}
  );
}

int32_t GrammarBuilder::AddExcludeTokenSet(const std::vector<int32_t>& token_ids) {
  return AddGrammarExpr(
      {GrammarExprType::kExcludeToken, token_ids.data(), static_cast<int32_t>(token_ids.size())}
  );
}

int32_t GrammarBuilder::AddRuleRef(int32_t rule_id) {
  std::vector<int32_t> data;
  data.push_back(rule_id);
  return AddGrammarExpr({GrammarExprType::kRuleRef, data.data(), static_cast<int32_t>(data.size())}
  );
}

int32_t GrammarBuilder::AddSequence(const std::vector<int32_t>& elements) {
  return AddGrammarExpr(
      {GrammarExprType::kSequence, elements.data(), static_cast<int32_t>(elements.size())}
  );
}

int32_t GrammarBuilder::AddChoices(const std::vector<int32_t>& choices) {
  return AddGrammarExpr(
      {GrammarExprType::kChoices, choices.data(), static_cast<int32_t>(choices.size())}
  );
}

int32_t GrammarBuilder::AddTagDispatch(const Grammar::Impl::TagDispatch& tag_dispatch) {
  std::vector<int32_t> data;
  data.reserve(tag_dispatch.tag_rule_pairs.size() * 2 + 2);
  for (const auto& [tag, rule_id] : tag_dispatch.tag_rule_pairs) {
    data.push_back(AddByteString(tag));
    data.push_back(rule_id);
  }
  data.push_back(static_cast<int32_t>(tag_dispatch.loop_after_dispatch));
  std::vector<int32_t> exclude_str_expr_ids;
  for (const auto& exclude_str : tag_dispatch.excludes) {
    exclude_str_expr_ids.push_back(AddByteString(exclude_str));
  }
  data.push_back(AddChoices(exclude_str_expr_ids));
  return AddGrammarExpr(
      {GrammarExprType::kTagDispatch, data.data(), static_cast<int32_t>(data.size())}
  );
}

int32_t GrammarBuilder::AddTokenTagDispatch(
    const Grammar::Impl::TokenTagDispatch& token_tag_dispatch
) {
  std::vector<int32_t> data;
  data.push_back(static_cast<int32_t>(token_tag_dispatch.trigger_rule_pairs.size()));
  for (const auto& [token_id, rule_id] : token_tag_dispatch.trigger_rule_pairs) {
    data.push_back(token_id);
    data.push_back(rule_id);
  }
  data.push_back(static_cast<int32_t>(token_tag_dispatch.loop_after_dispatch));
  data.push_back(static_cast<int32_t>(token_tag_dispatch.excludes.size()));
  for (auto token_id : token_tag_dispatch.excludes) {
    data.push_back(token_id);
  }
  return AddGrammarExpr(
      {GrammarExprType::kTokenTagDispatch, data.data(), static_cast<int32_t>(data.size())}
  );
}

int32_t GrammarBuilder::AddRepeat(
    int32_t ref_rule_id, int32_t min_repeat_count, int32_t max_repeat_count
) {
  std::vector<int32_t> data({ref_rule_id, min_repeat_count, max_repeat_count});
  return AddGrammarExpr({GrammarExprType::kRepeat, data.data(), static_cast<int32_t>(data.size())});
}

int32_t GrammarBuilder::AddRepeatFromExpr(
    const std::string& cur_rule_name,
    int32_t grammar_expr_id,
    int32_t min_repeat_count,
    int32_t max_repeat_count
) {
  const auto& expr = GetGrammarExpr(grammar_expr_id);
  int32_t ref_rule_id;
  if (expr.type == GrammarExprType::kRuleRef) {
    ref_rule_id = expr[0];
  } else {
    ref_rule_id = AddRule(GetNewRuleName(cur_rule_name), grammar_expr_id);
  }
  return AddRepeat(ref_rule_id, min_repeat_count, max_repeat_count);
}

int32_t GrammarBuilder::NumGrammarExprs() const { return grammar_->NumGrammarExprs(); }

GrammarBuilder::GrammarExpr GrammarBuilder::GetGrammarExpr(int32_t grammar_expr_id) {
  return grammar_->GetGrammarExpr(grammar_expr_id);
}

int32_t GrammarBuilder::AddRule(const Rule& rule) {
  int32_t id = static_cast<int32_t>(grammar_->rules_.size());
  grammar_->rules_.push_back(rule);
  XGRAMMAR_CHECK(rule_name_to_id_.count(rule.name) == 0);
  rule_name_to_id_[rule.name] = id;
  return id;
}

int32_t GrammarBuilder::AddRule(const std::string& name, int32_t body_expr_id) {
  return AddRule({name, body_expr_id});
}

int32_t GrammarBuilder::AddRuleWithHint(const std::string& name_hint, int32_t body_expr_id) {
  return AddRule({GetNewRuleName(name_hint), body_expr_id});
}

int32_t GrammarBuilder::NumRules() const { return grammar_->NumRules(); }

const GrammarBuilder::Rule& GrammarBuilder::GetRule(int32_t rule_id) const {
  return grammar_->rules_[rule_id];
}

int32_t GrammarBuilder::AddEmptyRule(const std::string& name) { return AddRule({name, -1}); }

int32_t GrammarBuilder::AddEmptyRuleWithHint(const std::string& name_hint) {
  return AddRule({GetNewRuleName(name_hint), -1});
}

void GrammarBuilder::UpdateRuleBody(int32_t rule_id, int32_t body_expr_id) {
  XGRAMMAR_CHECK(rule_id >= 0 && rule_id < static_cast<int32_t>(grammar_->rules_.size()))
      << "Rule id " << rule_id << " is out of range.";
  grammar_->rules_[rule_id].body_expr_id = body_expr_id;
}

void GrammarBuilder::UpdateRuleBody(std::string rule_name, int32_t body_expr_id) {
  int32_t rule_id = GetRuleId(rule_name);
  XGRAMMAR_CHECK(rule_id != -1) << "Rule " << rule_name << " is not found.";
  UpdateRuleBody(rule_id, body_expr_id);
}

void GrammarBuilder::UpdateLookaheadAssertion(int32_t rule_id, int32_t lookahead_assertion_id) {
  XGRAMMAR_CHECK(rule_id < static_cast<int32_t>(grammar_->rules_.size()))
      << "Rule id " << rule_id << " is out of range.";
  grammar_->rules_[rule_id].lookahead_assertion_id = lookahead_assertion_id;
}

void GrammarBuilder::UpdateLookaheadExact(int32_t rule_id, bool is_exact) {
  XGRAMMAR_CHECK(rule_id < static_cast<int32_t>(grammar_->rules_.size()))
      << "Rule id " << rule_id << " is out of range.";
  grammar_->rules_[rule_id].is_exact_lookahead = is_exact;
}

void GrammarBuilder::UpdateLookaheadAssertion(
    std::string rule_name, int32_t lookahead_assertion_id
) {
  int32_t rule_id = GetRuleId(rule_name);
  XGRAMMAR_CHECK(rule_id != -1) << "Rule " << rule_name << " is not found.";
  UpdateLookaheadAssertion(rule_id, lookahead_assertion_id);
}

std::string GrammarBuilder::GetNewRuleName(const std::string& name_hint) {
  if (rule_name_to_id_.count(name_hint) == 0) {
    return name_hint;
  }
  int* cnt = &next_cnt_per_hint_[name_hint];
  if (*cnt == 0) {
    *cnt = 1;
  }
  while (rule_name_to_id_.count(name_hint + "_" + std::to_string(*cnt)) != 0) {
    ++(*cnt);
  }
  return name_hint + "_" + std::to_string(*cnt);
}

int32_t GrammarBuilder::GetRuleId(const std::string& name) const {
  auto it = rule_name_to_id_.find(name);
  if (it == rule_name_to_id_.end()) {
    return -1;
  } else {
    return it->second;
  }
}

}  // namespace xgrammar
