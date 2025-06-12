#include "json_serializer.h"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "compiled_grammar_data_structure.h"
#include "grammar_data_structure.h"  // IWYU pragma: keep
#include "picojson.h"
#include "reflection/json.h"
#include "support/logging.h"
#include "tokenizer_info_impl.h"
#include "xgrammar/compiler.h"
#include "xgrammar/grammar.h"
#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

picojson::value CompiledGrammar::Impl::JSONSerialize() const {
  auto result = picojson::object{};
  result["grammar"] = AutoJSONSerialize(*grammar);
  result["tokenizer_metadata"] = AutoJSONSerialize(*tokenizer_info);
  result["adaptive_token_mask_cache"] = AutoJSONSerialize(adaptive_token_mask_cache);
  return picojson::value(std::move(result));
}

void JSONDeserialize(CompiledGrammar::Impl& impl, const picojson::value& v) {
  const auto& object = details::json_as<picojson::object>(v);
  impl.grammar = Grammar{std::make_shared<Grammar::Impl>()};
  AutoJSONDeserialize(*impl.grammar, details::json_member(object, "grammar"));
  impl.tokenizer_info = TokenizerInfo{std::make_shared<TokenizerInfo::Impl>()};
  AutoJSONDeserialize(*impl.tokenizer_info, details::json_member(object, "tokenizer_metadata"));
  impl.adaptive_token_mask_cache.clear();
  AutoJSONDeserialize(
      impl.adaptive_token_mask_cache, details::json_member(object, "adaptive_token_mask_cache")
  );
}

std::string JSONSerializer::SerializeGrammar(const Grammar& grammar, bool prettify) {
  return AutoJSONSerialize(*grammar).serialize(prettify);
}

std::string JSONSerializer::SerializeTokenizerInfo(
    const TokenizerInfo& tokenizer_info, bool prettify
) {
  // NOTE: this should be the same as the following function (except for prettify):
  // return tokenizer_info.DumpMetadata();
  return AutoJSONSerialize(*tokenizer_info).serialize(prettify);
}

std::string JSONSerializer::SerializeCompiledGrammar(
    const CompiledGrammar& compiled_grammar, bool prettify
) {
  return AutoJSONSerialize(*compiled_grammar).serialize(prettify);
}

static picojson::value parse_string(const std::string& str) {
  picojson::value v;
  std::string err;
  picojson::parse(v, str.begin(), str.end(), &err);
  XGRAMMAR_CHECK(err.empty()) << "Failed to parse JSON: " << err;
  return v;
}

Grammar JSONSerializer::DeserializeGrammar(const std::string& str) {
  auto v = parse_string(str);
  auto grammar = Grammar{std::make_shared<Grammar::Impl>()};
  AutoJSONDeserialize(*grammar, v);
  return grammar;
}

TokenizerInfo JSONSerializer::DeserializeTokenizerInfo(
    const std::string& str, const std::vector<std::string>& encoded_vocab
) {
  if (encoded_vocab.empty()) {
    // simply build a tokenizer info with only metadata
    auto v = parse_string(str);
    auto tokenizer_info = TokenizerInfo{std::make_shared<TokenizerInfo::Impl>()};
    AutoJSONDeserialize(*tokenizer_info, v);
    return tokenizer_info;
  } else {
    // rebuild a complete tokenizer info with vocab
    auto temp_impl = TokenizerInfo::Impl{};
    AutoJSONDeserialize(temp_impl, parse_string(str));
    return TokenizerInfo{
        encoded_vocab,
        temp_impl.GetVocabType(),
        temp_impl.GetVocabSize(),
        temp_impl.GetStopTokenIds(),
        temp_impl.GetAddPrefixSpace(),
    };
  }
}

bool TokenizerInfo::Impl::operator==(const TokenizerInfo::Impl& other) const {
  static constexpr auto tie = [](const TokenizerInfo::Impl& impl) {
    return std::tie(
        impl.vocab_type_,
        impl.vocab_size_,
        impl.add_prefix_space_,
        impl.stop_token_ids_,
        impl.special_token_ids_
    );
  };
  return tie(*this) == tie(other);
}

CompiledGrammar JSONSerializer::DeserializeCompiledGrammar(
    const std::string& str, const TokenizerInfo& tokenizer_info
) {
  auto v = parse_string(str);
  auto compiled_grammar = CompiledGrammar{std::make_shared<CompiledGrammar::Impl>()};
  AutoJSONDeserialize(*compiled_grammar, v);
  // compare the tokenizer info metadata
  XGRAMMAR_CHECK(*compiled_grammar->tokenizer_info == *tokenizer_info)
      << "The tokenizer info in the compiled grammar does not match the provided one.";
  // set the tokenizer info to the real one
  compiled_grammar->tokenizer_info = tokenizer_info;
  return compiled_grammar;
}

}  // namespace xgrammar
