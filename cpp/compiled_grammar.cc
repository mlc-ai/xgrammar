/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/compiled_grammar.cc
 */

#include <xgrammar/compiler.h>

#include "compiled_grammar_impl.h"
#include "support/json_serializer.h"

namespace xgrammar {

/************** CompiledGrammar::Impl **************/

picojson::value CompiledGrammar::Impl::SerializeJSONValue() const {
  auto result = picojson::object{};
  result["grammar"] = AutoSerializeJSONValue(grammar_);
  result["tokenizer_metadata"] = picojson::value(tokenizer_info_.DumpMetadata());
  result["adaptive_token_mask_cache"] = AutoSerializeJSONValue(adaptive_token_mask_cache);
  return picojson::value(result);
}

std::optional<std::runtime_error> CompiledGrammar::Impl::DeserializeJSONValue(
    CompiledGrammar::Impl& compiled_grammar, const picojson::value& json_value
) {
  //   if (!json_value.is<picojson::object>()) {
  //     return std::runtime_error(
  //         "CompiledGrammar Deserialization: Expect an object in the deserialization input"
  //     );
  //   }

  //   auto& object = json_value.get<picojson::object>();
  //   if (object.find("grammar") == object.end()) {
  //     return std::runtime_error(
  //         "CompiledGrammar Deserialization: Expect a 'grammar' field in the deserialization
  //         input"
  //     );
  //   }
  //   auto grammar_value = object["grammar"];
  //   AutoDeserializeJSONValue(*compiled_grammar.grammar_.ImplPtr(), grammar_value);
  //   if (object.find("tokenizer_metadata") == object.end()) {
  //     return std::runtime_error(
  //         "CompiledGrammar Deserialization: Expect a 'tokenizer_metadata' field in the "
  //         "deserialization input"
  //     );
  //   }
  //   if (object.find("adaptive_token_mask_cache") == object.end()) {
  //     auto grammar = std::make_shared<Grammar::Impl>();
  //     AutoDeserializeJSONValue(
  //         *grammar,  // grammar pimpl
  //         details::json_member(object, "grammar")
  //     );
  //     auto tokenizer_metadata = std::make_shared<TokenizerInfo::Impl>();
  //   auto compiler_grammar = CompiledGrammar{std::make_shared<CompiledGrammar::Impl>()};
  //   auto result = DeserializeJSONPython(json_string);
  //   if (std::holds_alternative<VersionError>(result))
  //     throw_version_error(std::get<VersionError>(result), "CompiledGrammar");

  //   auto& value = std::get<picojson::value>(result);
  //   try {
  //     const auto& object = details::json_as<picojson::object>(value);
  //     auto grammar = std::make_shared<Grammar::Impl>();
  //     compiler_grammar->grammar = Grammar{grammar};
  //     AutoDeserializeJSONValue(
  //         *grammar,  // grammar pimpl
  //         details::json_member(object, "grammar")
  //     );
  //     auto tokenizer_metadata = std::make_shared<TokenizerInfo::Impl>();
  //     compiler_grammar->tokenizer_info = TokenizerInfo{tokenizer_metadata};
  //     AutoDeserializeJSONValue(
  //         *tokenizer_metadata,  // tokenizer info pimpl
  //         details::json_member(object, "tokenizer_metadata")
  //     );
  //     AutoDeserializeJSONValue(
  //         compiler_grammar->adaptive_token_mask_cache,
  //         details::json_member(object, "adaptive_token_mask_cache")
  //     );
  //     XGRAMMAR_CHECK(*compiler_grammar->tokenizer_info == *tokenizer_metadata)
  //         << "The tokenizer info in the compiled grammar does not match the provided one.";
  //     compiler_grammar->tokenizer_info = std::move(tokenizer_info);
  //     return compiler_grammar;
  //   } catch (const std::exception&) {
  //     // pass the exception to the caller
  //   }
  //   throw_format_error("CompiledGrammar");
}

/************** CompiledGrammar **************/

std::size_t CompiledGrammar::Impl::MemorySize() const {
  std::size_t sum = 0;
  sum += grammar->MemorySize();
  sum += adaptive_token_mask_cache.size() * sizeof(*adaptive_token_mask_cache.begin());
  for (auto& [_, mask] : adaptive_token_mask_cache) {
    sum += mask.MemorySize();
  }
  return sum;
}

std::size_t CompiledGrammar::MemorySizeBytes() const { return pimpl_->MemorySize(); }

Grammar CompiledGrammar::GetGrammar() const { return pimpl_->GetGrammar(); }

TokenizerInfo CompiledGrammar::GetTokenizerInfo() const { return pimpl_->GetTokenizerInfo(); }

}  // namespace xgrammar
