#ifndef XGRAMMAR_RS_CXX_UTILS_COMPILED_GRAMMAR_H_
#define XGRAMMAR_RS_CXX_UTILS_COMPILED_GRAMMAR_H_

#include <memory>
#include <string>
#include <variant>

#include "xgrammar/xgrammar.h"

namespace cxx_utils {

inline std::unique_ptr<xgrammar::CompiledGrammar>
compiled_grammar_deserialize_json_or_error(
    const std::string& json_string,
    const xgrammar::TokenizerInfo& tokenizer_info,
    std::string* error_out
) {
  auto result =
      xgrammar::CompiledGrammar::DeserializeJSON(json_string, tokenizer_info);
  if (std::holds_alternative<xgrammar::SerializationError>(result)) {
    if (error_out) {
      const auto& err = std::get<xgrammar::SerializationError>(result);
      std::visit([&](const auto& e) { *error_out = e.what(); }, err);
    }
    return nullptr;
  }
  return std::make_unique<xgrammar::CompiledGrammar>(
      std::get<xgrammar::CompiledGrammar>(std::move(result))
  );
}

} // namespace cxx_utils

#endif // XGRAMMAR_RS_CXX_UTILS_COMPILED_GRAMMAR_H_
