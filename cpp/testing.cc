/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag.cc
 */
#include <xgrammar/xgrammar.h>

#include <sstream>
#include <string>
#include <vector>

#include "grammar_impl.h"
#include "grammar_parser.h"
#include "json_schema_converter.h"
#include "support/encoding.h"

namespace xgrammar {

std::string PrintTokenByIds(
    const std::vector<int32_t>& token_ids, const TokenizerInfo& tokenizer_info, int max_print_num
) {
  std::stringstream ss;
  const auto& sorted_decoded_vocab = tokenizer_info.GetDecodedVocab();
  ss << "[";
  int print_num = std::min(static_cast<int>(token_ids.size()), max_print_num);
  for (int i = 0; i < print_num; ++i) {
    ss << "#" << token_ids[i] << " <" << EscapeString(sorted_decoded_vocab[token_ids[i]]) << ">";
    if (i < print_num - 1) {
      ss << ", ";
    }
  }
  if (static_cast<int>(token_ids.size()) > max_print_num) {
    ss << ", ...";
  }
  ss << "]";
  return ss.str();
}

Grammar _EBNFToGrammarNoNormalization(
    const std::string& ebnf_string, const std::string& root_rule_name
) {
  return ParseEBNF(ebnf_string, root_rule_name);
}

std::string _PrintGrammarFSMs(const Grammar& grammar) {
  std::string result;
  for (int i = 0; i < grammar->NumRules(); i++) {
    result += "Rule " + std::to_string(i) + ": " + grammar->GetRule(i).name + ", FSM: ";
    if (grammar->per_rule_fsms[i].has_value()) {
      result += grammar->per_rule_fsms[i]->ToString();
    } else {
      result += "None";
    }
    result += "\n";
  }
  return result;
}

std::string _QwenXMLToolCallingToEBNF(const std::string& schema) {
  // Convert the schema string to picojson value.
  picojson::value json_value;
  std::string err = picojson::parse(json_value, schema);
  if (!err.empty()) {
    XGRAMMAR_LOG(FATAL) << "Failed to parse JSON schema: " << err;
  }
  if (json_value.is<bool>()) {
    XGRAMMAR_LOG(FATAL) << "Expected JSON schema object, got boolean: " << json_value.to_str();
  }
  const auto& schema_obj = json_value.get<picojson::object>();
  if (!schema_obj.count("type")) {
    XGRAMMAR_LOG(FATAL) << "Function calling must have a 'type' field of 'object': "
                        << json_value.to_str();
  }
  if (schema_obj.at("type").get<std::string>() != "object") {
    XGRAMMAR_LOG(FATAL) << "Function calling must have a 'type' field of 'object': "
                        << json_value.to_str();
  }
  return JSONSchemaToEBNF(json_value, true, std::nullopt, std::nullopt, true, JSONFormat::kXML);
}

}  // namespace xgrammar
