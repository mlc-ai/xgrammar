#pragma once

#include <string>
#include <vector>
#include <exception>
#include "cpp/testing.h"
#include "cpp/json_schema_converter.h"

namespace cxx_utils {

inline std::string json_schema_to_ebnf(
    const std::string& schema,
    bool any_whitespace,
    bool has_indent,
    int32_t indent,
    bool has_separators,
    const std::string& separator_comma,
    const std::string& separator_colon,
    bool strict_mode,
    bool has_max_whitespace_cnt,
    int32_t max_whitespace_cnt
) {
  try {
    std::optional<int> indent_opt = std::nullopt;
    if (has_indent) {
      indent_opt = indent;
    }

    std::optional<std::pair<std::string, std::string>> separators_opt =
        std::nullopt;
    if (has_separators) {
      separators_opt = std::make_pair(separator_comma, separator_colon);
    }

    std::optional<int> max_whitespace_cnt_opt = std::nullopt;
    if (has_max_whitespace_cnt) {
      max_whitespace_cnt_opt = static_cast<int>(max_whitespace_cnt);
    }

    return xgrammar::JSONSchemaToEBNF(
        schema,
        any_whitespace,
        indent_opt,
        separators_opt,
        strict_mode,
        max_whitespace_cnt_opt,
        xgrammar::JSONFormat::kJSON
    );
  } catch (...) {
    return std::string();
  }
}

inline std::unique_ptr<xgrammar::Grammar> ebnf_to_grammar_no_normalization(
    const std::string& ebnf_string,
    const std::string& root_rule_name
) {
  try {
    return std::make_unique<xgrammar::Grammar>(
        xgrammar::_EBNFToGrammarNoNormalization(ebnf_string, root_rule_name)
    );
  } catch (...) {
    return nullptr;
  }
}

inline std::string qwen_xml_tool_calling_to_ebnf(const std::string& schema) {
  try {
    return xgrammar::QwenXMLToolCallingToEBNF(schema);
  } catch (...) {
    return std::string();
  }
}

inline std::vector<int32_t> get_masked_tokens_from_bitmask(
    const DLTensor* bitmask,
    int32_t vocab_size,
    int32_t index
) {
  try {
    std::vector<int> result;
    xgrammar::_DebugGetMaskedTokensFromBitmask(
        &result,
        *bitmask,
        vocab_size,
        index
    );
    std::vector<int32_t> output;
    output.assign(result.begin(), result.end());
    return output;
  } catch (...) {
    return std::vector<int32_t>();
  }
}

struct SingleTokenResult {
  bool is_single;
  int32_t token_id;

  bool get_is_single() const { return is_single; }
  int32_t get_token_id() const { return token_id; }
};

inline SingleTokenResult is_single_token_bitmask(
    const DLTensor* bitmask,
    int32_t vocab_size,
    int32_t index
) {
  try {
    auto pair = xgrammar::_IsSingleTokenBitmask(*bitmask, vocab_size, index);
    return SingleTokenResult{pair.first, pair.second};
  } catch (...) {
    return SingleTokenResult{false, -1};
  }
}

} // namespace cxx_utils
