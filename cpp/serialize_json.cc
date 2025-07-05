#include <picojson.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "compiled_grammar_impl.h"
#include "grammar_impl.h"
#include "support/logging.h"
#include "support/reflection/json_serializer.h"
#include "support/utils.h"
#include "tokenizer_info_impl.h"
#include "xgrammar/compiler.h"
#include "xgrammar/grammar.h"
#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

// static std::string SerializeJSONPython(picojson::object& object) {
//   object["__VERSION__"] = picojson::value(kXGrammarSerializeVersion);
//   return picojson::value{object}.serialize(/*prettify=*/false);
// }

enum class VersionError {
  kMissingVersion,
  kVersionMismatch,
};

// // Python str -> C++ picojson::value
// static std::variant<picojson::value, VersionError> DeserializeJSONPython(const std::string& str)
// {
//   picojson::value v;
//   std::string err;
//   picojson::parse(v, str.begin(), str.end(), &err);
//   XGRAMMAR_CHECK(err.empty()) << "Failed to parse JSON: " << err;
//   XGRAMMAR_CHECK(v.is<picojson::object>()) << "Expected a JSON object, but got: " <<
//   v.serialize(); auto& object = v.get<picojson::object>(); auto version_it =
//   object.find("__VERSION__"); if (version_it == object.end()) {
//     return VersionError::kMissingVersion;
//   }
//   const auto& version = version_it->second;
//   if (!version.is<std::string>() || version.get<std::string>() != kXGrammarSerializeVersion) {
//     return VersionError::kVersionMismatch;
//   }
//   object.erase(version_it);  // Remove the version field from the object.
//   return v;
// }

// Throws an error if the version is missing or mismatched.
// [[noreturn]]
// static void throw_version_error(VersionError error, std::string type) {
//   const auto error_prefix = "Deserialize of type " + type +
//                             " failed: "
//                             " version error: ";
//   const auto error_suffix = " Please remove the cache and serialize it in this version.";
//   switch (error) {
//     case VersionError::kMissingVersion:
//       XGRAMMAR_LOG_FATAL << error_prefix << "missing version in serialized JSON." <<
//       error_suffix; break;
//     case VersionError::kVersionMismatch:
//       XGRAMMAR_LOG_FATAL << error_prefix
//                          << "the serialized json is from another version of xgrammar."
//                          << error_suffix;
//       break;
//     default:
//       XGRAMMAR_LOG_FATAL << error_prefix << "internal implementation error." << error_suffix;
//   }
//   XGRAMMAR_UNREACHABLE();
// }

// [[noreturn]]
// static void throw_format_error(std::string type) {
//   // Deserialize of type xxx: format error: the json does not follow the serialization format.
//   XGRAMMAR_LOG_FATAL << "Deserialize of type " << type
//                      << " failed: format error: the json does not follow the serialization
//                      format.";
//   XGRAMMAR_UNREACHABLE();
// }

// std::string TokenizerInfo::SerializeJSON() const {
//   auto value = AutoSerializeJSONValue(**this);
//   auto& object = value.get<picojson::object>();
//   return SerializeJSONPython(object);
// }

// TokenizerInfo TokenizerInfo::DeserializeJSON(const std::string& json_string) {
//   auto result = DeserializeJSONPython(json_string);
//   if (std::holds_alternative<VersionError>(result))
//     throw_version_error(std::get<VersionError>(result), "TokenizerInfo");

//   auto& value = std::get<picojson::value>(result);
//   try {
//     auto tokenizer_info = TokenizerInfo{std::make_shared<TokenizerInfo::Impl>()};
//     AutoDeserializeJSONValue(*tokenizer_info, value);
//     return tokenizer_info;
//   } catch (const std::exception& e) {
//     // pass the exception to the caller
//     throw_format_error("TokenizerInfo: " + std::string(e.what()));
//   }
//   XGRAMMAR_UNREACHABLE();
// }

}  // namespace xgrammar
