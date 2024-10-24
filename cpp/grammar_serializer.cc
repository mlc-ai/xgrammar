/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_serializer.cc
 */

#include "grammar_serializer.h"

#include <picojson.h>

#include "support/encoding.h"

namespace xgrammar {

std::string BNFGrammarPrinter::PrintRule(const Rule& rule) {
  std::string res = rule.name + " ::= " + PrintRuleExpr(rule.body_expr_id);
  if (rule.lookahead_assertion_id != -1) {
    res += " (=" + PrintRuleExpr(rule.lookahead_assertion_id) + ")";
  }
  return res;
}

std::string BNFGrammarPrinter::PrintRule(int32_t rule_id) {
  return PrintRule(grammar_->GetRule(rule_id));
}

std::string BNFGrammarPrinter::PrintRuleExpr(const RuleExpr& rule_expr) {
  std::string result;
  switch (rule_expr.type) {
    case RuleExprType::kByteString:
      return PrintByteString(rule_expr);
    case RuleExprType::kCharacterClass:
      return PrintCharacterClass(rule_expr);
    case RuleExprType::kCharacterClassStar:
      return PrintCharacterClassStar(rule_expr);
    case RuleExprType::kEmptyStr:
      return PrintEmptyStr(rule_expr);
    case RuleExprType::kRuleRef:
      return PrintRuleRef(rule_expr);
    case RuleExprType::kSequence:
      return PrintSequence(rule_expr);
    case RuleExprType::kChoices:
      return PrintChoices(rule_expr);
    default:
      XGRAMMAR_LOG(FATAL) << "Unexpected RuleExpr type: " << static_cast<int>(rule_expr.type);
  }
}

std::string BNFGrammarPrinter::PrintRuleExpr(int32_t rule_expr_id) {
  return PrintRuleExpr(grammar_->GetRuleExpr(rule_expr_id));
}

std::string BNFGrammarPrinter::PrintByteString(const RuleExpr& rule_expr) {
  std::string internal_str;
  internal_str.reserve(rule_expr.data_len);
  for (int i = 0; i < rule_expr.data_len; ++i) {
    internal_str += static_cast<char>(rule_expr[i]);
  }
  auto codepoints = ParseUTF8(internal_str.c_str(), true);
  std::string result;
  for (auto codepoint : codepoints) {
    result += PrintAsEscapedUTF8(codepoint);
  }
  return "\"" + result + "\"";
}

std::string BNFGrammarPrinter::PrintCharacterClass(const RuleExpr& rule_expr) {
  static const std::unordered_map<TCodepoint, std::string> kCustomEscapeMap = {
      {'-', "\\-"}, {']', "\\]"}
  };
  std::string result = "[";
  bool is_negative = static_cast<bool>(rule_expr[0]);
  if (is_negative) {
    result += "^";
  }
  for (auto i = 1; i < rule_expr.data_len; i += 2) {
    result += PrintAsEscapedUTF8(rule_expr[i], kCustomEscapeMap);
    if (rule_expr[i] == rule_expr[i + 1]) {
      continue;
    }
    result += "-";
    result += PrintAsEscapedUTF8(rule_expr[i + 1], kCustomEscapeMap);
  }
  result += "]";
  return result;
}

std::string BNFGrammarPrinter::PrintCharacterClassStar(const RuleExpr& rule_expr) {
  return PrintCharacterClass(rule_expr) + "*";
}

std::string BNFGrammarPrinter::PrintEmptyStr(const RuleExpr& rule_expr) { return "\"\""; }

std::string BNFGrammarPrinter::PrintRuleRef(const RuleExpr& rule_expr) {
  return grammar_->GetRule(rule_expr[0]).name;
}

std::string BNFGrammarPrinter::PrintSequence(const RuleExpr& rule_expr) {
  std::string result;
  result += "(";
  for (int i = 0; i < rule_expr.data_len; ++i) {
    result += PrintRuleExpr(rule_expr[i]);
    if (i + 1 != rule_expr.data_len) {
      result += " ";
    }
  }
  result += ")";
  return result;
}

std::string BNFGrammarPrinter::PrintChoices(const RuleExpr& rule_expr) {
  std::string result;

  result += "(";
  for (int i = 0; i < rule_expr.data_len; ++i) {
    result += PrintRuleExpr(rule_expr[i]);
    if (i + 1 != rule_expr.data_len) {
      result += " | ";
    }
  }
  result += ")";
  return result;
}

std::string BNFGrammarPrinter::ToString() {
  std::string result;
  int num_rules = grammar_->NumRules();
  for (auto i = 0; i < num_rules; ++i) {
    result += PrintRule(grammar_->GetRule(i)) + "\n";
  }
  return result;
}

std::string BNFGrammarSerializer::Serialize() {
  picojson::object grammar_json_obj;

  picojson::array rules_json;
  for (const auto& rule : grammar_->rules_) {
    picojson::object rule_json;
    rule_json["name"] = picojson::value(rule.name);
    rule_json["body_expr_id"] = picojson::value(static_cast<int64_t>(rule.body_expr_id));
    rules_json.push_back(picojson::value(rule_json));
  }
  grammar_json_obj["rules"] = picojson::value(rules_json);

  picojson::array rule_expr_data_json;
  for (const auto& data : grammar_->rule_expr_data_) {
    rule_expr_data_json.push_back(picojson::value(static_cast<int64_t>(data)));
  }
  grammar_json_obj["rule_expr_data"] = picojson::value(rule_expr_data_json);
  picojson::array rule_expr_indptr_json;
  for (const auto& index_ptr : grammar_->rule_expr_indptr_) {
    rule_expr_indptr_json.push_back(picojson::value(static_cast<int64_t>(index_ptr)));
  }
  grammar_json_obj["rule_expr_indptr"] = picojson::value(rule_expr_indptr_json);

  auto grammar_json = picojson::value(grammar_json_obj);
  return grammar_json.serialize(prettify_);
}

BNFGrammar BNFGrammarDeserializer::Deserialize(std::string json_string) {
  auto node = std::make_shared<BNFGrammar::Impl>();

  auto checker = [&](bool condition) {
    XGRAMMAR_CHECK(condition) << "Failed to deserialize XGrammar object: " << json_string;
  };

  picojson::value serialized_value;
  std::string err = picojson::parse(serialized_value, json_string);

  checker(err.empty() && serialized_value.is<picojson::object>());
  auto serialized_obj = serialized_value.get<picojson::object>();

  // rules
  checker(serialized_obj.count("rules") && serialized_obj["rules"].is<picojson::array>());
  auto rules_array = serialized_obj["rules"].get<picojson::array>();

  checker(rules_array.size() > 0);
  for (const auto& rule_value : rules_array) {
    checker(rule_value.is<picojson::object>());
    auto rule_obj = rule_value.get<picojson::object>();
    checker(rule_obj.count("name") && rule_obj["name"].is<std::string>());
    auto name = rule_obj["name"].get<std::string>();
    checker(rule_obj.count("body_expr_id") && rule_obj["body_expr_id"].is<int64_t>());
    auto rule_expr = static_cast<int32_t>(rule_obj["body_expr_id"].get<int64_t>());
    node->rules_.push_back(BNFGrammar::Impl::Rule({name, rule_expr}));
  }

  // rule_expr_data
  checker(
      serialized_obj.count("rule_expr_data") &&
      serialized_obj["rule_expr_data"].is<picojson::array>()
  );
  auto rule_expr_data_array = serialized_obj["rule_expr_data"].get<picojson::array>();
  for (const auto& data_json : rule_expr_data_array) {
    node->rule_expr_data_.push_back(static_cast<int32_t>(data_json.get<int64_t>()));
  }

  // rule_expr_indptr
  checker(
      serialized_obj.count("rule_expr_indptr") &&
      serialized_obj["rule_expr_indptr"].is<picojson::array>()
  );
  auto rule_expr_indptr_array = serialized_obj["rule_expr_indptr"].get<picojson::array>();
  for (const auto& index_ptr_json : rule_expr_indptr_array) {
    node->rule_expr_indptr_.push_back(static_cast<int32_t>(index_ptr_json.get<int64_t>()));
  }

  return BNFGrammar(node);
}

}  // namespace xgrammar
