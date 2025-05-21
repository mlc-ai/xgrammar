#include "json_serializer.h"

#include <string>
#include <vector>

#include "compiled_grammar_data_structure.h"
#include "grammar_data_structure.h"  // IWYU pragma: keep
#include "picojson.h"
#include "support/json_serialize.h"
#include "tokenizer_internal.h"  // IWYU pragma: keep
#include "xgrammar/compiler.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

picojson::value CompiledGrammar::Impl::JSONSerialize() const {
  auto result = picojson::object{};
  result["grammar"] = AutoJSONSerialize(*grammar);
  result["tokenizer_metadata"] = AutoJSONSerialize(*tokenizer_info);
  result["adaptive_token_mask_cache"] = AutoJSONSerialize(adaptive_token_mask_cache);
  return picojson::value(std::move(result));
}

std::string JSONSerializer::SerializeGrammar(const Grammar& grammar, bool prettify) {
  return AutoJSONSerialize(*grammar).serialize(prettify);
}

std::string JSONSerializer::SerializeTokenizerInfo(
    const TokenizerInfo& tokenizer_info, bool prettify
) {
  return AutoJSONSerialize(*tokenizer_info).serialize(prettify);
}

std::string JSONSerializer::SerializeCompiledGrammar(
    const CompiledGrammar& compiled_grammar, bool prettify
) {
  return AutoJSONSerialize(*compiled_grammar).serialize(prettify);
}

Grammar JSONSerializer::DeserializeGrammar(const std::string&) {
  throw std::runtime_error("Deserialization not implemented");
}

TokenizerInfo
JSONSerializer::DeserializeTokenizerInfo(const std::string&, const std::vector<std::string>&) {
  throw std::runtime_error("Deserialization not implemented");
}

CompiledGrammar
JSONSerializer::DeserializeCompiledGrammar(const std::string&, const TokenizerInfo&) {
  throw std::runtime_error("Deserialization not implemented");
}

}  // namespace xgrammar
