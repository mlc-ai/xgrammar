#ifndef XGRAMMAR_RS_CXX_UTILS_MATCHER_H_
#define XGRAMMAR_RS_CXX_UTILS_MATCHER_H_

#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <vector>
#include <string>

#include "xgrammar/xgrammar.h"

namespace cxx_utils {

inline std::unique_ptr<xgrammar::GrammarMatcher> make_grammar_matcher(
    const xgrammar::CompiledGrammar& compiled_grammar,
    bool has_override_stop_tokens,
    const int32_t* override_stop_tokens_ptr,
    size_t override_stop_tokens_len,
    bool terminate_without_stop_token,
    int max_rollback_tokens,
    std::string* error_out
) {
  try {
    if (error_out) {
      error_out->clear();
    }

    if (!has_override_stop_tokens) {
      return std::make_unique<xgrammar::GrammarMatcher>(
          compiled_grammar,
          std::nullopt,
          terminate_without_stop_token,
          max_rollback_tokens
      );
    }
    std::vector<int> tmp;
    tmp.reserve(override_stop_tokens_len);
    for (size_t i = 0; i < override_stop_tokens_len; ++i) {
      tmp.push_back(static_cast<int>(override_stop_tokens_ptr[i]));
    }
    return std::make_unique<xgrammar::GrammarMatcher>(
        compiled_grammar,
        std::optional<std::vector<int>>(std::move(tmp)),
        terminate_without_stop_token,
        max_rollback_tokens
    );
  } catch (const std::exception& e) {
    if (error_out) {
      *error_out = e.what();
    }
    return nullptr;
  } catch (...) {
    if (error_out) {
      *error_out = "unknown C++ exception";
    }
    return nullptr;
  }
}

inline std::unique_ptr<xgrammar::BatchGrammarMatcher> make_batch_grammar_matcher(
    int32_t max_threads,
    std::string* error_out
) {
  try {
    if (error_out) {
      error_out->clear();
    }
    return std::make_unique<xgrammar::BatchGrammarMatcher>(max_threads);
  } catch (const std::exception& e) {
    if (error_out) {
      *error_out = e.what();
    }
    return nullptr;
  } catch (...) {
    if (error_out) {
      *error_out = "unknown C++ exception";
    }
    return nullptr;
  }
}

inline std::unique_ptr<std::vector<xgrammar::GrammarMatcher>>
new_grammar_matcher_vector() {
  return std::make_unique<std::vector<xgrammar::GrammarMatcher>>();
}

inline void grammar_matcher_vec_reserve(
    std::vector<xgrammar::GrammarMatcher>& vec,
    size_t n
) {
  vec.reserve(n);
}

inline void grammar_matcher_vec_push(
    std::vector<xgrammar::GrammarMatcher>& vec,
    const xgrammar::GrammarMatcher& matcher
) {
  vec.push_back(matcher);
}

inline void batch_matcher_batch_fill_next_token_bitmask(
    xgrammar::BatchGrammarMatcher& batch_matcher,
    std::vector<xgrammar::GrammarMatcher>* matchers,
    DLTensor* bitmask,
    bool has_indices,
    const int32_t* indices_ptr,
    size_t indices_len,
    bool debug_print
) {
  try {
    std::optional<std::vector<int32_t>> indices_opt;
    if (has_indices) {
      std::vector<int32_t> tmp(indices_ptr, indices_ptr + indices_len);
      indices_opt = std::move(tmp);
    }
    batch_matcher
        .BatchFillNextTokenBitmask(matchers, bitmask, indices_opt, debug_print);
  } catch (...) {
  }
}

inline std::vector<uint8_t> batch_accept_token(
    std::vector<xgrammar::GrammarMatcher>* matchers,
    const int32_t* token_ids_ptr,
    size_t token_ids_len,
    bool debug_print
) {
  try {
    std::vector<int32_t> token_ids(token_ids_ptr, token_ids_ptr + token_ids_len);
    return xgrammar::BatchGrammarMatcher::BatchAcceptToken(
        matchers,
        token_ids,
        debug_print
    );
  } catch (...) {
    return std::vector<uint8_t>(token_ids_len, 0);
  }
}

inline std::vector<uint8_t> batch_accept_string(
    std::vector<xgrammar::GrammarMatcher>* matchers,
    const std::vector<std::string>& strings,
    bool debug_print
) {
  try {
    return xgrammar::BatchGrammarMatcher::BatchAcceptString(
        matchers,
        strings,
        debug_print
    );
  } catch (...) {
    return std::vector<uint8_t>(strings.size(), 0);
  }
}

} // namespace cxx_utils

#endif // XGRAMMAR_RS_CXX_UTILS_MATCHER_H_
