/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/kimi_k3.cc
 * \brief Kimi-K3 XML type attributes and key escaping.
 */
#include <picojson.h>

#include <type_traits>

#include "../json_schema_converter_ext.h"
#include "../support/json_parse.h"

namespace xgrammar {

namespace converter_ext {

XMLWrapperParts GetKimiK3XMLWrapper() {
  // The key suffix (type attribute and <|sep|>) is generated in XMLKeySuffix.
  return {"<|open|>argument key=\"", "", "", "<|close|>argument<|sep|>"};
}

const XMLKeySuffix& GetKimiK3XMLKeySuffix() {
  static const XMLKeySuffix suffix = {
      "\" type=\"",
      {"string", "number", "integer", "boolean", "object", "array", "null"},
      "\"<|sep|>"
  };
  return suffix;
}

}  // namespace converter_ext

std::optional<std::string> XMLToolCallingConverter::KimiK3TypeAttr(const SchemaSpecPtr& spec) {
  if (spec == nullptr) {
    return std::nullopt;
  }
  // The type name a single JSON value is rendered with, following the model's _xtml_type.
  auto type_of_json_value = [](const std::string& json_value) -> std::optional<std::string> {
    picojson::value value;
    if (!ParseJSON(value, json_value).empty()) {
      return std::nullopt;
    }
    if (value.is<std::string>()) return "string";
    if (value.is<bool>()) return "boolean";
    if (value.is<double>()) return "number";
    if (value.is<picojson::null>()) return "null";
    if (value.is<picojson::object>()) return "object";
    if (value.is<picojson::array>()) return "array";
    return std::nullopt;
  };

  return std::visit(
      [&](auto&& arg) -> std::optional<std::string> {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, StringSpec>) {
          return "string";
        } else if constexpr (std::is_same_v<T, IntegerSpec> || std::is_same_v<T, NumberSpec>) {
          // _xtml_type renders every int and float as "number"; it never emits "integer".
          return "number";
        } else if constexpr (std::is_same_v<T, BooleanSpec>) {
          return "boolean";
        } else if constexpr (std::is_same_v<T, NullSpec>) {
          return "null";
        } else if constexpr (std::is_same_v<T, ArraySpec>) {
          return "array";
        } else if constexpr (std::is_same_v<T, ObjectSpec>) {
          return "object";
        } else if constexpr (std::is_same_v<T, ConstSpec>) {
          return type_of_json_value(arg.json_value);
        } else if constexpr (std::is_same_v<T, EnumSpec>) {
          // Only pin the attribute when every alternative renders with the same type.
          std::optional<std::string> common;
          for (const auto& json_value : arg.json_values) {
            auto type_name = type_of_json_value(json_value);
            if (!type_name.has_value()) return std::nullopt;
            if (!common.has_value()) {
              common = type_name;
            } else if (*common != *type_name) {
              return std::nullopt;
            }
          }
          return common;
        } else {
          // Any, $ref and the combinators may render as more than one type; keep them open.
          return std::nullopt;
        }
      },
      spec->spec
  );
}

std::string XMLToolCallingConverter::EscapeAttrValue(const std::string& value) const {
  if (json_format_ != JSONFormat::kKimiK3XML) {
    return value;
  }
  // Kimi-K3's renderer escapes attribute values with & -> &amp; and " -> &quot;.
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    if (c == '&') {
      escaped += "&amp;";
    } else if (c == '"') {
      escaped += "&quot;";
    } else {
      escaped += c;
    }
  }
  return escaped;
}

}  // namespace xgrammar
