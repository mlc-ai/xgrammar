#include "serializer.h"

#include <type_traits>

#include "compiled_grammar_data_structure.h"  // IWYU pragma: keep
#include "grammar_data_structure.h"           // IWYU pragma: keep
#include "support/logging.h"
#include "tokenizer_internal.h"  // IWYU pragma: keep
#include "xgrammar/compiler.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

struct ImplSerializer {
  template <typename T>
  static std::string Serialize(const T& obj, [[maybe_unused]] bool prettify = false) {
    auto value = obj->Serialize();
    static_assert(std::is_same_v<decltype(value), picojson::value>, "Cannot serialize the object");
    return value.serialize(prettify);
  }

  template <typename T, typename... Args>
  static auto Deserialize(const std::string& value, Args&&... args) {
    using Impl = typename T::Impl;
    picojson::value v;
    std::string err;
    picojson::parse(v, value.begin(), value.end(), &err);
    XGRAMMAR_CHECK(err.empty()) << "Failed to parse JSON in deserialization: " << err;
    return Impl::Deserialize(std::move(v), std::forward<Args>(args)...);
  }
};

std::string JSONSerializer::SerializeGrammar(const Grammar& grammar, bool prettify) {
  return ImplSerializer::Serialize(grammar, prettify);
}

Grammar JSONSerializer::DeserializeGrammar(const std::string& str) {
  return ImplSerializer::Deserialize<Grammar>(str);
}

std::string JSONSerializer::SerializeTokenizerInfo(
    const TokenizerInfo& tokenizer_info, bool prettify
) {
  return ImplSerializer::Serialize(tokenizer_info, prettify);
}

TokenizerInfo JSONSerializer::DeserializeTokenizerInfo(
    const std::string& str, const std::vector<std::string>& encoded_vocab
) {
  return ImplSerializer::Deserialize<TokenizerInfo>(str, encoded_vocab);
}

std::string JSONSerializer::SerializeCompiledGrammar(
    const CompiledGrammar& grammar, bool prettify
) {
  return ImplSerializer::Serialize(grammar, prettify);
}

CompiledGrammar JSONSerializer::DeserializeCompiledGrammar(
    const std::string& str, const std::vector<std::string>& encoded_vocab
) {
  return ImplSerializer::Deserialize<CompiledGrammar>(str, encoded_vocab);
}

}  // namespace xgrammar
