/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/compiled_grammar.cc
 */

#include <xgrammar/compiler.h>

#include "compiled_grammar_impl.h"
#include "support/json_serializer.h"

namespace xgrammar {

/************** CompiledGrammar::Impl **************/

picojson::value SerializeJSONValue(const CompiledGrammar::Impl& impl) {
  auto result = picojson::object{};
  result["grammar"] = AutoSerializeJSONValue(impl.grammar_);
  result["tokenizer_metadata"] = picojson::value(impl.tokenizer_info_.DumpMetadata());
  result["adaptive_token_mask_cache"] = AutoSerializeJSONValue(impl.adaptive_token_mask_cache);
  return picojson::value(result);
}

std::optional<std::runtime_error> DeserializeJSONValue(
    CompiledGrammar::Impl* impl,
    const picojson::value& json_value,
    const TokenizerInfo& tokenizer_info
) {
  const auto& type_name = "CompiledGrammar";
  if (!json_value.is<picojson::object>()) {
    return ConstructDeserializeError("Expect an object", type_name);
  }
  const auto& object = json_value.get<picojson::object>();
  if (object.find("grammar") == object.end()) {
    return ConstructDeserializeError("Expect a 'grammar' field", type_name);
  }
  AutoDeserializeJSONValue(&impl->grammar_, object["grammar"]);
  if (object.find("tokenizer_metadata") == object.end()) {
    return ConstructDeserializeError("Expect a 'tokenizer_metadata' field", type_name);
  }
  AutoDeserializeJSONValue(&impl->tokenizer_info_, object["tokenizer_metadata"]);
  if (object.find("adaptive_token_mask_cache") == object.end()) {
    return ConstructDeserializeError("Expect a 'adaptive_token_mask_cache' field", type_name);
  }
  AutoDeserializeJSONValue(&impl->adaptive_token_mask_cache, object["adaptive_token_mask_cache"]);
  return std::nullopt;
}

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
// }

/************** CompiledGrammar **************/

std::size_t MemorySize(const CompiledGrammar::Impl& impl) {
  return MemorySize(impl.grammar_) + MemorySize(impl.adaptive_token_mask_cache);
}

std::size_t CompiledGrammar::MemorySizeBytes() const { return MemorySize(*pimpl_); }

Grammar CompiledGrammar::GetGrammar() const { return pimpl_->GetGrammar(); }

TokenizerInfo CompiledGrammar::GetTokenizerInfo() const { return pimpl_->GetTokenizerInfo(); }

}  // namespace xgrammar
