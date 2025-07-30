/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_printer.cc
 */

#include "grammar_printer.h"

#include <picojson.h>

#include "support/encoding.h"

namespace xgrammar {

std::string GrammarPrinter::PrintRule(const Rule& rule) {
  std::string res = rule.name + " ::= " + PrintGrammarExpr(rule.body_expr_id);
  if (rule.lookahead_assertion_id != -1) {
    res += " (=" + PrintGrammarExpr(rule.lookahead_assertion_id) + ")";
  }
  return res;
}

std::string GrammarPrinter::PrintRule(int32_t rule_id) {
  return PrintRule(grammar_->GetRule(rule_id));
}

std::string GrammarPrinter::PrintGrammarExpr(const GrammarExpr& grammar_expr) {
  std::string result;
  switch (grammar_expr.type) {
    case GrammarExprType::kByteString:
      return PrintByteString(grammar_expr);
    case GrammarExprType::kCharacterClass:
      return PrintCharacterClass(grammar_expr);
    case GrammarExprType::kCharacterClassStar:
      return PrintCharacterClassStar(grammar_expr);
    case GrammarExprType::kEmptyStr:
      return PrintEmptyStr(grammar_expr);
    case GrammarExprType::kRuleRef:
      return PrintRuleRef(grammar_expr);
    case GrammarExprType::kSequence:
      return PrintSequence(grammar_expr);
    case GrammarExprType::kChoices:
      return PrintChoices(grammar_expr);
    case GrammarExprType::kTagDispatch:
      return PrintTagDispatch(grammar_expr);
    case GrammarExprType::kRepeat:
      return PrintRepeat(grammar_expr);
    default:
      XGRAMMAR_LOG(FATAL) << "Unexpected GrammarExpr type: " << static_cast<int>(grammar_expr.type);
      XGRAMMAR_UNREACHABLE();
  }
}

std::string GrammarPrinter::PrintGrammarExpr(int32_t grammar_expr_id) {
  return PrintGrammarExpr(grammar_->GetGrammarExpr(grammar_expr_id));
}

std::string GrammarPrinter::PrintByteString(const GrammarExpr& grammar_expr) {
  std::string internal_str;
  internal_str.reserve(grammar_expr.data_len);
  for (int i = 0; i < grammar_expr.data_len; ++i) {
    internal_str += static_cast<char>(grammar_expr[i]);
  }
  return "\"" + EscapeString(internal_str) + "\"";
}

std::string GrammarPrinter::PrintCharacterClass(const GrammarExpr& grammar_expr) {
  static const std::unordered_map<TCodepoint, std::string> kCustomEscapeMap = {
      {'-', "\\-"}, {']', "\\]"}
  };
  std::string result = "[";
  bool is_negative = static_cast<bool>(grammar_expr[0]);
  if (is_negative) {
    result += "^";
  }
  for (auto i = 1; i < grammar_expr.data_len; i += 2) {
    result += EscapeString(grammar_expr[i], kCustomEscapeMap);
    if (grammar_expr[i] == grammar_expr[i + 1]) {
      continue;
    }
    result += "-";
    result += EscapeString(grammar_expr[i + 1], kCustomEscapeMap);
  }
  result += "]";
  return result;
}

std::string GrammarPrinter::PrintCharacterClassStar(const GrammarExpr& grammar_expr) {
  return PrintCharacterClass(grammar_expr) + "*";
}

std::string GrammarPrinter::PrintEmptyStr(const GrammarExpr& grammar_expr) { return "\"\""; }

std::string GrammarPrinter::PrintRuleRef(const GrammarExpr& grammar_expr) {
  return grammar_->GetRule(grammar_expr[0]).name;
}

std::string GrammarPrinter::PrintSequence(const GrammarExpr& grammar_expr) {
  std::string result;
  result += "(";
  for (int i = 0; i < grammar_expr.data_len; ++i) {
    result += PrintGrammarExpr(grammar_expr[i]);
    if (i + 1 != grammar_expr.data_len) {
      result += " ";
    }
  }
  result += ")";
  return result;
}

std::string GrammarPrinter::PrintChoices(const GrammarExpr& grammar_expr) {
  std::string result;

  result += "(";
  for (int i = 0; i < grammar_expr.data_len; ++i) {
    result += PrintGrammarExpr(grammar_expr[i]);
    if (i + 1 != grammar_expr.data_len) {
      result += " | ";
    }
  }
  result += ")";
  return result;
}

std::string GrammarPrinter::PrintString(const std::string& str) {
  return "\"" + EscapeString(str) + "\"";
}

std::string GrammarPrinter::PrintBoolean(bool value) { return value ? "true" : "false"; }

std::string GrammarPrinter::PrintTagDispatch(const GrammarExpr& grammar_expr) {
  auto tag_dispatch = grammar_->GetTagDispatch(grammar_expr);
  std::string result = "TagDispatch(\n";
  std::string indent = "  ";
  for (const auto& [tag, rule_id] : tag_dispatch.tag_rule_pairs) {
    result += indent + "(" + PrintString(tag) + ", " + grammar_->GetRule(rule_id).name + "),\n";
  }
  result += indent + "stop_eos=" + PrintBoolean(tag_dispatch.stop_eos) + ",\n";
  result += indent + "stop_str=(";
  for (int i = 0; i < static_cast<int>(tag_dispatch.stop_str.size()); ++i) {
    result += PrintString(tag_dispatch.stop_str[i]);
    if (i + 1 != static_cast<int>(tag_dispatch.stop_str.size())) {
      result += ", ";
    }
  }
  result += "),\n";
  result += indent + "loop_after_dispatch=" + PrintBoolean(tag_dispatch.loop_after_dispatch) + "\n";
  result += ")";
  return result;
}

std::string GrammarPrinter::PrintRepeat(const GrammarExpr& grammar_expr) {
  int32_t lower_bound = grammar_expr[1];
  int32_t upper_bound = grammar_expr[2];
  std::string result = grammar_->GetRule(grammar_expr[0]).name + "{";
  result += std::to_string(lower_bound);
  result += ", ";
  result += std::to_string(upper_bound);
  result += "}";
  return result;
}

std::string GrammarPrinter::ToString() {
  std::string result;
  int num_rules = grammar_->NumRules();
  for (auto i = 0; i < num_rules; ++i) {
    result += PrintRule(grammar_->GetRule(i)) + "\n";
  }
  return result;
}

}  // namespace xgrammar
