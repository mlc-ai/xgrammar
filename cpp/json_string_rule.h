/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/json_string_rule.h
 * \brief Structural recognition helpers for the generic recursive JSON string body.
 */
#ifndef XGRAMMAR_JSON_STRING_RULE_H_
#define XGRAMMAR_JSON_STRING_RULE_H_

#include <cstdint>
#include <string_view>

#include "grammar_impl.h"

namespace xgrammar {

namespace json_string_rule_detail {

using GrammarExprType = Grammar::Impl::GrammarExprType;

inline int32_t UnwrapSingleElementSequences(const Grammar::Impl* grammar, int32_t grammar_expr_id) {
  for (int32_t depth = 0; depth < 8; ++depth) {
    const auto expr = grammar->GetGrammarExpr(grammar_expr_id);
    if (expr.type != GrammarExprType::kSequence || expr.size() != 1) {
      break;
    }
    grammar_expr_id = expr[0];
  }
  return grammar_expr_id;
}

inline bool IsByteString(
    const Grammar::Impl* grammar, int32_t grammar_expr_id, std::string_view expected
) {
  grammar_expr_id = UnwrapSingleElementSequences(grammar, grammar_expr_id);
  const auto expr = grammar->GetGrammarExpr(grammar_expr_id);
  if (expr.type != GrammarExprType::kByteString ||
      expr.size() != static_cast<int32_t>(expected.size())) {
    return false;
  }
  for (int32_t i = 0; i < expr.size(); ++i) {
    if (static_cast<uint8_t>(expr[i]) != static_cast<uint8_t>(expected[i])) {
      return false;
    }
  }
  return true;
}

inline bool IsRuleRef(const Grammar::Impl* grammar, int32_t grammar_expr_id, int32_t rule_id) {
  grammar_expr_id = UnwrapSingleElementSequences(grammar, grammar_expr_id);
  const auto expr = grammar->GetGrammarExpr(grammar_expr_id);
  return expr.type == GrammarExprType::kRuleRef && expr.size() == 1 && expr[0] == rule_id;
}

inline bool IsJSONNormalCharacterClass(const Grammar::Impl* grammar, int32_t grammar_expr_id) {
  grammar_expr_id = UnwrapSingleElementSequences(grammar, grammar_expr_id);
  const auto expr = grammar->GetGrammarExpr(grammar_expr_id);
  if (expr.type != GrammarExprType::kCharacterClass || expr.size() < 1 || expr[0] == 0 ||
      (expr.size() - 1) % 2 != 0) {
    return false;
  }
  for (int32_t codepoint = 0; codepoint < 128; ++codepoint) {
    bool excluded = false;
    for (int32_t i = 1; i < expr.size(); i += 2) {
      if (expr[i] < 0 || expr[i] > expr[i + 1] || expr[i + 1] >= 128) {
        return false;
      }
      excluded = excluded || (codepoint >= expr[i] && codepoint <= expr[i + 1]);
    }
    const bool should_be_excluded = codepoint <= 0x1f || codepoint == '"' || codepoint == '\\';
    if (excluded != should_be_excluded) {
      return false;
    }
  }
  return true;
}

}  // namespace json_string_rule_detail

inline bool IsGenericJSONStringBodyRule(const Grammar::Impl* grammar, int32_t rule_id) {
  using namespace json_string_rule_detail;
  if (rule_id < 0 || rule_id >= grammar->NumRules()) {
    return false;
  }
  const auto body = grammar->GetGrammarExpr(grammar->GetRule(rule_id).body_expr_id);
  if (body.type != GrammarExprType::kChoices || body.size() != 3) {
    return false;
  }
  bool found_quote = false;
  bool found_normal_character = false;
  bool found_escape = false;
  for (int32_t choice_id : body) {
    if (IsByteString(grammar, choice_id, "\"")) {
      found_quote = true;
      continue;
    }
    const auto choice = grammar->GetGrammarExpr(choice_id);
    if (choice.type == GrammarExprType::kSequence && choice.size() == 2 &&
        IsJSONNormalCharacterClass(grammar, choice[0]) && IsRuleRef(grammar, choice[1], rule_id)) {
      found_normal_character = true;
      continue;
    }
    if (choice.type == GrammarExprType::kSequence && choice.size() == 3 &&
        IsByteString(grammar, choice[0], "\\") && IsRuleRef(grammar, choice[2], rule_id)) {
      const auto escape_ref = grammar->GetGrammarExpr(choice[1]);
      if (escape_ref.type != GrammarExprType::kRuleRef || escape_ref.size() != 1) {
        return false;
      }
      found_escape = true;
      continue;
    }
    return false;
  }
  return found_quote && found_normal_character && found_escape;
}

inline bool IsGenericJSONStringBodyDirectMaskRule(const Grammar::Impl* grammar, int32_t rule_id) {
  if (!IsGenericJSONStringBodyRule(grammar, rule_id)) {
    return false;
  }
  const auto& rule = grammar->GetRule(rule_id);
  return !rule.is_lazy && rule.max_tokens == -1 && rule.max_chars == -1 &&
         rule.capture_name.empty() && grammar->per_rule_fsms[rule_id].has_value();
}

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_STRING_RULE_H_
