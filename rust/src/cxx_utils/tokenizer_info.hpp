#ifndef XGRAMMAR_RS_CXX_UTILS_TOKENIZER_INFO_H_
#define XGRAMMAR_RS_CXX_UTILS_TOKENIZER_INFO_H_

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xgrammar/xgrammar.h"

namespace cxx_utils {

inline std::unique_ptr<xgrammar::TokenizerInfo> make_tokenizer_info(
    const std::vector<std::string>& encoded_vocab,
    xgrammar::VocabType vocab_type,
    bool has_vocab_size,
    int32_t vocab_size,
    bool has_stop_ids,
    const int32_t* stop_token_ids_ptr,
    size_t stop_token_ids_len,
    bool add_prefix_space,
    std::string* error_out
) {
  try {
    if (error_out) {
      error_out->clear();
    }

    std::optional<int> vs = std::nullopt;
    if (has_vocab_size) {
      vs = vocab_size;
    }

    std::optional<std::vector<int32_t>> stops = std::nullopt;
    if (has_stop_ids) {
      std::vector<int32_t> tmp;
      tmp.reserve(stop_token_ids_len);
      for (size_t i = 0; i < stop_token_ids_len; ++i) {
        tmp.push_back(stop_token_ids_ptr[i]);
      }
      stops = std::move(tmp);
    }

    return std::make_unique<xgrammar::TokenizerInfo>(
        encoded_vocab,
        vocab_type,
        vs,
        stops,
        add_prefix_space
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

inline std::unique_ptr<xgrammar::TokenizerInfo>
tokenizer_info_deserialize_json_or_error(
    const std::string& json_string,
    std::string* error_out
) {
  try {
    auto result = xgrammar::TokenizerInfo::DeserializeJSON(json_string);
    if (std::holds_alternative<xgrammar::SerializationError>(result)) {
      if (error_out) {
        const auto& err = std::get<xgrammar::SerializationError>(result);
        std::visit([&](const auto& e) { *error_out = e.what(); }, err);
      }
      return nullptr;
    }
    return std::make_unique<xgrammar::TokenizerInfo>(
        std::get<xgrammar::TokenizerInfo>(std::move(result))
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

} // namespace cxx_utils

#endif // XGRAMMAR_RS_CXX_UTILS_TOKENIZER_INFO_H_
