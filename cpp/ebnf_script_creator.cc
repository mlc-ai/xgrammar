/*!
 *  Copyright (c) 2023 by Contributors
 * \file xgrammar/ebnf_script_creator.cc
 */
#include "ebnf_script_creator.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "support/logging.h"

namespace xgrammar {

class EBNFScriptCreator::Impl {
 public:
  Impl() {}

  std::string AddRule(const std::string& rule_name_hint, const std::string& rule_body);
  std::string AllocateRuleName(const std::string& rule_name_hint);
  std::string AddRuleWithAllocatedName(const std::string& rule_name, const std::string& rule_body);
  std::string GetScript();
  std::string GetRuleContent(const std::string& rule_name);

 private:
  std::vector<std::pair<std::string, std::string>> rules_;
  std::unordered_set<std::string> rule_names_;
  const int NAME_SUFFIX_MAXIMUM = 10000;
};

std::string EBNFScriptCreator::Impl::AllocateRuleName(const std::string& rule_name_hint) {
  if (rule_names_.find(rule_name_hint) == rule_names_.end()) {
    rule_names_.insert(rule_name_hint);
    return rule_name_hint;
  }
  for (int i = 0; i < NAME_SUFFIX_MAXIMUM; ++i) {
    std::string rule_name = rule_name_hint + "_" + std::to_string(i);
    if (rule_names_.find(rule_name) == rule_names_.end()) {
      rule_names_.insert(rule_name);
      return rule_name;
    }
  }
  XGRAMMAR_LOG(FATAL) << "Cannot find a unique rule name for " << rule_name_hint;
}

std::string EBNFScriptCreator::Impl::AddRule(
    const std::string& rule_name_hint, const std::string& rule_body
) {
  return AddRuleWithAllocatedName(AllocateRuleName(rule_name_hint), rule_body);
}

std::string EBNFScriptCreator::Impl::AddRuleWithAllocatedName(
    const std::string& rule_name, const std::string& rule_body
) {
  XGRAMMAR_CHECK(rule_names_.find(rule_name) != rule_names_.end())
      << "Rule name " << rule_name << " is not allocated";
  rules_.emplace_back(rule_name, rule_body);
  return rule_name;
}

std::string EBNFScriptCreator::Impl::GetScript() {
  std::string script = "";
  for (const auto& rule : rules_) {
    script += rule.first + " ::= " + rule.second + "\n";
  }
  return script;
}

std::string EBNFScriptCreator::Impl::GetRuleContent(const std::string& rule_name) {
  auto it = std::find_if(rules_.begin(), rules_.end(), [rule_name](const auto& rule) {
    return rule.first == rule_name;
  });
  if (it != rules_.end()) {
    return it->second;
  }
  return "";
}

EBNFScriptCreator::EBNFScriptCreator(EmptyConstructorTag) : pimpl_(std::make_shared<Impl>()) {}

std::string EBNFScriptCreator::AddRule(
    const std::string& rule_name_hint, const std::string& rule_body
) {
  return pimpl_->AddRule(rule_name_hint, rule_body);
}

std::string EBNFScriptCreator::AllocateRuleName(const std::string& rule_name_hint) {
  return pimpl_->AllocateRuleName(rule_name_hint);
}

std::string EBNFScriptCreator::GetScript() { return pimpl_->GetScript(); }

std::string EBNFScriptCreator::GetRuleContent(const std::string& rule_name) {
  return pimpl_->GetRuleContent(rule_name);
}

std::string EBNFScriptCreator::AddRuleWithAllocatedName(
    const std::string& rule_name, const std::string& rule_body
) {
  return pimpl_->AddRuleWithAllocatedName(rule_name, rule_body);
}

}  // namespace xgrammar
