#pragma once

#include <string>
#include <vector>
#include <exception>
#include <optional>
#include <algorithm>
#include "cpp/testing.h"
#include "cpp/json_schema_converter.h"
#include "cpp/regex_converter.h"

namespace cxx_utils {

inline std::string generate_range_regex(
    bool has_start,
    int64_t start,
    bool has_end,
    int64_t end,
    std::string* error_out
) {
  try {
    if (error_out) {
      error_out->clear();
    }
    std::optional<int64_t> start_opt = has_start ? std::optional<int64_t>(start) : std::nullopt;
    std::optional<int64_t> end_opt = has_end ? std::optional<int64_t>(end) : std::nullopt;
    std::string result = xgrammar::GenerateRangeRegex(start_opt, end_opt);
    result.erase(std::remove(result.begin(), result.end(), '\0'), result.end());
    return result;
  } catch (const std::exception& e) {
    if (error_out) {
      *error_out = e.what();
    }
    return std::string();
  } catch (...) {
    if (error_out) {
      *error_out = "unknown C++ exception";
    }
    return std::string();
  }
}

inline std::string generate_float_range_regex(
    bool has_start,
    double start,
    bool has_end,
    double end,
    std::string* error_out
) {
  try {
    if (error_out) {
      error_out->clear();
    }
    std::optional<double> start_opt = has_start ? std::optional<double>(start) : std::nullopt;
    std::optional<double> end_opt = has_end ? std::optional<double>(end) : std::nullopt;
    std::string result = xgrammar::GenerateFloatRangeRegex(start_opt, end_opt);
    result.erase(std::remove(result.begin(), result.end(), '\0'), result.end());
    return result;
  } catch (const std::exception& e) {
    if (error_out) {
      *error_out = e.what();
    }
    return std::string();
  } catch (...) {
    if (error_out) {
      *error_out = "unknown C++ exception";
    }
    return std::string();
  }
}

inline std::string print_grammar_fsms(
    const xgrammar::Grammar& grammar,
    std::string* error_out
) {
  try {
    if (error_out) {
      error_out->clear();
    }
    return xgrammar::_PrintGrammarFSMs(grammar);
  } catch (const std::exception& e) {
    if (error_out) {
      *error_out = e.what();
    }
    return std::string();
  } catch (...) {
    if (error_out) {
      *error_out = "unknown C++ exception";
    }
    return std::string();
  }
}

// get_allow_empty_rule_ids removed - requires grammar_impl.h which contains templates

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

inline std::string regex_to_ebnf(
    const std::string& regex,
    bool with_rule_name,
    std::string* error_out
) {
  try {
    if (error_out) {
      error_out->clear();
    }
    return xgrammar::RegexToEBNF(regex, with_rule_name);
  } catch (const std::exception& e) {
    if (error_out) {
      *error_out = e.what();
    }
    return std::string();
  } catch (...) {
    if (error_out) {
      *error_out = "unknown C++ exception";
    }
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

inline bool traverse_draft_tree(
    const DLTensor* retrieve_next_token,
    const DLTensor* retrieve_next_sibling,
    const DLTensor* draft_tokens,
    xgrammar::GrammarMatcher& matcher,
    DLTensor* bitmask,
    std::string* error_out
) {
  try {
    if (error_out) {
      error_out->clear();
    }
    xgrammar::TraverseDraftTree(
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
        matcher,
        bitmask
    );
    return true;
  } catch (const std::exception& e) {
    if (error_out) {
      *error_out = e.what();
    }
    return false;
  } catch (...) {
    if (error_out) {
      *error_out = "unknown C++ exception";
    }
    return false;
  }
}

// Grammar functor functions (structure_normalizer, rule_inliner, etc.) removed
// They require grammar_functor.h which contains templates that autocxx can't handle

} // namespace cxx_utils
