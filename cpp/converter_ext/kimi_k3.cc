/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/kimi_k3.cc
 * \brief Implementation of the KimiK3 XML Tool Calling converter.
 */
#include <picojson.h>

#include <type_traits>

#include "../json_schema_converter_ext.h"
#include "../support/json_parse.h"

namespace xgrammar {

KimiK3XMLToolCallingConverter::KimiK3XMLToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    bool any_order
)
    : XMLToolCallingConverter(
          indent,
          separators,
          any_whitespace,
          max_whitespace_cnt,
          ref_resolver,
          {"<|open|>argument key=\"", "", "", "<|close|>argument<|sep|>"},
          any_order
      ) {}

int32_t KimiK3XMLToolCallingConverter::XMLKeySuffix(const SchemaSpecPtr& schema) {
  auto pinned_type = KimiK3TypeAttr(schema);
  // A declared property carries exactly the type its value grammar is rendered with, so the
  // parser decodes the value back to the schema's type. Free-form keys have no single schema
  // type, so they keep the full set.
  int32_t type_expr = pinned_type.has_value() ? ByteString(*pinned_type)
                                              : Choice(
                                                    {ByteString("string"),
                                                     ByteString("number"),
                                                     ByteString("integer"),
                                                     ByteString("boolean"),
                                                     ByteString("object"),
                                                     ByteString("array"),
                                                     ByteString("null")}
                                                );
  return Sequence({ByteString("\" type=\""), type_expr, ByteString("\"<|sep|>")});
}

std::optional<std::string> KimiK3XMLToolCallingConverter::KimiK3TypeAttr(const SchemaSpecPtr& spec
) {
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

std::string KimiK3XMLToolCallingConverter::EscapeAttrValue(const std::string& value) const {
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
