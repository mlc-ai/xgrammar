#include <picojson.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "compiled_grammar_data_structure.h"
#include "grammar_data_structure.h"
#include "support/logging.h"
#include "support/reflection/json_serializer.h"
#include "support/utils.h"
#include "tokenizer_info_impl.h"
#include "xgrammar/compiler.h"
#include "xgrammar/grammar.h"
#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

static constexpr const char kXGrammarSerializeVersion[] = "v3";

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

// C++ picojson::value -> Python str
static std::string SerializeJSONPython(picojson::object& object) {
  object["__VERSION__"] = picojson::value(kXGrammarSerializeVersion);
  return picojson::value{object}.serialize(/*prettify=*/false);
}

enum class VersionError {
  kMissingVersion,
  kVersionMismatch,
};

// Python str -> C++ picojson::value
static std::variant<picojson::value, VersionError> DeserializeJSONPython(const std::string& str) {
  picojson::value v;
  std::string err;
  picojson::parse(v, str.begin(), str.end(), &err);
  XGRAMMAR_CHECK(err.empty()) << "Failed to parse JSON: " << err;
  XGRAMMAR_CHECK(v.is<picojson::object>()) << "Expected a JSON object, but got: " << v.serialize();
  auto& object = v.get<picojson::object>();
  auto version_it = object.find("__VERSION__");
  if (version_it == object.end()) {
    return VersionError::kMissingVersion;
  }
  const auto& version = version_it->second;
  if (!version.is<std::string>() || version.get<std::string>() != kXGrammarSerializeVersion) {
    return VersionError::kVersionMismatch;
  }
  object.erase(version_it);  // Remove the version field from the object.
  return v;
}

// Throws an error if the version is missing or mismatched.
[[noreturn]]
static void throw_version_error(VersionError error, std::string type) {
  const auto error_prefix = "Deserialize of type " + type +
                            " failed: "
                            " version error: ";
  const auto error_suffix = " Please remove the cache and serialize it in this version.";
  switch (error) {
    case VersionError::kMissingVersion:
      XGRAMMAR_LOG_FATAL << error_prefix << "missing version in serialized JSON." << error_suffix;
      break;
    case VersionError::kVersionMismatch:
      XGRAMMAR_LOG_FATAL << error_prefix
                         << "the serialized json is from another version of xgrammar."
                         << error_suffix;
      break;
    default:
      XGRAMMAR_LOG_FATAL << error_prefix << "internal implementation error." << error_suffix;
  }
  XGRAMMAR_UNREACHABLE();
}

[[noreturn]]
static void throw_format_error(std::string type) {
  // Deserialize of type xxx: format error: the json does not follow the serialization format.
  XGRAMMAR_LOG_FATAL << "Deserialize of type " << type
                     << " failed: format error: the json does not follow the serialization format.";
  XGRAMMAR_UNREACHABLE();
}

std::string CompiledGrammar::SerializeJSON() const {
  auto result = picojson::object{};
  result["grammar"] = AutoSerializeJSONValue(*(*this)->grammar);
  result["tokenizer_metadata"] = AutoSerializeJSONValue(*(*this)->tokenizer_info);
  result["adaptive_token_mask_cache"] = AutoSerializeJSONValue((*this)->adaptive_token_mask_cache);
  return SerializeJSONPython(result);
}

CompiledGrammar CompiledGrammar::DeserializeJSON(
    const std::string& json_string, TokenizerInfo tokenizer_info
) {
  auto compiler_grammar = CompiledGrammar{std::make_shared<CompiledGrammar::Impl>()};
  auto result = DeserializeJSONPython(json_string);
  if (std::holds_alternative<VersionError>(result))
    throw_version_error(std::get<VersionError>(result), "CompiledGrammar");

  auto& value = std::get<picojson::value>(result);
  try {
    const auto& object = details::json_as<picojson::object>(value);
    auto grammar = std::make_shared<Grammar::Impl>();
    compiler_grammar->grammar = Grammar{grammar};
    AutoDeserializeJSONValue(
        *grammar,  // grammar pimpl
        details::json_member(object, "grammar")
    );
    auto tokenizer_metadata = std::make_shared<TokenizerInfo::Impl>();
    compiler_grammar->tokenizer_info = TokenizerInfo{tokenizer_metadata};
    AutoDeserializeJSONValue(
        *tokenizer_metadata,  // tokenizer info pimpl
        details::json_member(object, "tokenizer_metadata")
    );
    AutoDeserializeJSONValue(
        compiler_grammar->adaptive_token_mask_cache,
        details::json_member(object, "adaptive_token_mask_cache")
    );
    XGRAMMAR_CHECK(*compiler_grammar->tokenizer_info == *tokenizer_metadata)
        << "The tokenizer info in the compiled grammar does not match the provided one.";
    compiler_grammar->tokenizer_info = std::move(tokenizer_info);
    return compiler_grammar;
  } catch (const std::exception&) {
    // pass the exception to the caller
  }
  throw_format_error("CompiledGrammar");
}

std::string Grammar::SerializeJSON() const {
  auto value = AutoSerializeJSONValue(**this);
  auto& object = value.get<picojson::object>();
  return SerializeJSONPython(object);
}

Grammar Grammar::DeserializeJSON(const std::string& json_string) {
  auto result = DeserializeJSONPython(json_string);
  if (std::holds_alternative<VersionError>(result))
    throw_version_error(std::get<VersionError>(result), "Grammar");
  auto& value = std::get<picojson::value>(result);
  try {
    auto grammar = Grammar{std::make_shared<Grammar::Impl>()};
    AutoDeserializeJSONValue(*grammar, value);
    return grammar;
  } catch (const std::exception&) {
    // pass the exception to the caller
  }
  throw_format_error("Grammar");
}

std::string TokenizerInfo::SerializeJSON() const {
  auto value = AutoSerializeJSONValue(**this);
  auto& object = value.get<picojson::object>();
  return SerializeJSONPython(object);
}

TokenizerInfo TokenizerInfo::DeserializeJSON(
    const std::string& json_string, const std::vector<std::string>& encoded_vocab
) {
  auto result = DeserializeJSONPython(json_string);
  if (std::holds_alternative<VersionError>(result))
    throw_version_error(std::get<VersionError>(result), "TokenizerInfo");

  auto& value = std::get<picojson::value>(result);
  try {
    auto tokenizer_info = TokenizerInfo{std::make_shared<TokenizerInfo::Impl>()};
    AutoDeserializeJSONValue(*tokenizer_info, value);
    if (!encoded_vocab.empty()) {
      // construct the tokenizer info with the encoded vocab
      return TokenizerInfo{
          encoded_vocab,
          tokenizer_info->GetVocabType(),
          tokenizer_info->GetVocabSize(),
          tokenizer_info->GetStopTokenIds(),
          tokenizer_info->GetAddPrefixSpace(),
      };
    } else {
      // return the tokenizer info with only metadata
      return tokenizer_info;
    }
  } catch (const std::exception&) {
    // pass the exception to the caller
  }
  throw_format_error("TokenizerInfo");
}

}  // namespace xgrammar
