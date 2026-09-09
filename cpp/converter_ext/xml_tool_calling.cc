/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/xml_tool_calling.cc
 * \brief XML tool-calling parameter formats.
 */
#include "../json_schema_converter_ext.h"

namespace xgrammar {

const std::unordered_map<JSONFormat, XMLToolCallingConverter::XMLWrapper>
    XMLToolCallingConverter::kKeyWrapperMap = [] {
      auto wrapper = [](const converter_ext::XMLWrapperParts& parts) {
        return XMLWrapper{parts[0], parts[1], parts[2], parts[3]};
      };
      return std::unordered_map<JSONFormat, XMLWrapper>{
          {JSONFormat::kQwenXML, wrapper(converter_ext::GetQwenXMLWrapper())},
          {JSONFormat::kMiniMaxXML, wrapper(converter_ext::GetMiniMaxXMLWrapper())},
          {JSONFormat::kDeepSeekXML, wrapper(converter_ext::GetDeepSeekXMLWrapper())},
          {JSONFormat::kGlmXML, wrapper(converter_ext::GetGLMXMLWrapper())},
          {JSONFormat::kCohereXML, wrapper(converter_ext::GetCohereXMLWrapper())},
          {JSONFormat::kKimiK3XML, wrapper(converter_ext::GetKimiK3XMLWrapper())},
      };
    }();

XMLToolCallingConverter::XMLToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    JSONFormat json_format,
    bool any_order
)
    : JSONSchemaConverter(
          indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver, any_order
      ),
      json_format_(json_format),
      nested_object_level_(0),
      xml_wrapper_(kKeyWrapperMap.at(json_format)) {}

int32_t XMLToolCallingConverter::XMLKeySuffix(const std::optional<std::string>& pinned_type) {
  auto value_choices = [this](const std::vector<const char*>& values) {
    std::vector<int32_t> choices;
    choices.reserve(values.size());
    for (const auto* value : values) {
      choices.push_back(ByteString(value));
    }
    return Choice(choices);
  };
  if (json_format_ == JSONFormat::kDeepSeekXML) {
    const auto& suffix = converter_ext::GetDeepSeekXMLKeySuffix();
    return Sequence(
        {ByteString(suffix.prefix), value_choices(suffix.values), ByteString(suffix.suffix)}
    );
  }
  if (json_format_ == JSONFormat::kKimiK3XML) {
    const auto& suffix = converter_ext::GetKimiK3XMLKeySuffix();
    // A declared property carries exactly the type its value grammar is rendered with, so the
    // parser decodes the value back to the schema's type. Free-form keys have no single schema
    // type, so they keep the full set.
    int32_t type_expr =
        pinned_type.has_value() ? ByteString(*pinned_type) : value_choices(suffix.values);
    return Sequence({ByteString(suffix.prefix), type_expr, ByteString(suffix.suffix)});
  }
  return ByteString(xml_wrapper_.key_wrapper_suffix);
}

int32_t XMLToolCallingConverter::FormatPropertyKey(
    const std::string& key, const SchemaSpecPtr& schema
) {
  if (nested_object_level_ <= 1) {
    // Only kimi_k3_xml encodes the value's type next to the key; the other formats would
    // discard the result, so don't walk the schema for them.
    std::optional<std::string> pinned_type;
    if (json_format_ == JSONFormat::kKimiK3XML) {
      pinned_type = KimiK3TypeAttr(schema);
    }
    return Sequence(
        {ByteString(xml_wrapper_.key_wrapper_prefix + EscapeAttrValue(key)),
         XMLKeySuffix(pinned_type)}
    );
  }
  return JSONSchemaConverter::FormatPropertyKey(key, schema);
}

int32_t XMLToolCallingConverter::FormatOtherProperty(
    int32_t key_pattern_expr,
    int32_t value_rule_id,
    const std::string& rule_name,
    const std::string& rule_name_suffix,
    const SchemaSpecPtr& schema
) {
  if (nested_object_level_ <= 1) {
    std::vector<int32_t> elements = {
        ByteString(xml_wrapper_.key_wrapper_prefix),
        key_pattern_expr,
        XMLKeySuffix(json_format_ == JSONFormat::kKimiK3XML ? KimiK3TypeAttr(schema) : std::nullopt)
    };
    if (!xml_wrapper_.value_wrapper_prefix.empty()) {
      elements.push_back(WhitespaceExpression());
      elements.push_back(ByteString(xml_wrapper_.value_wrapper_prefix));
    }
    if (value_rule_id == builder_.GetRuleId(kXMLString)) {
      elements.push_back(RuleRef(value_rule_id));
    } else {
      elements.push_back(WhitespaceExpression());
      elements.push_back(RuleRef(value_rule_id));
      elements.push_back(WhitespaceExpression());
    }
    elements.push_back(ByteString(xml_wrapper_.parameter_suffix));
    return Sequence(elements);
  }
  return JSONSchemaConverter::FormatOtherProperty(
      key_pattern_expr, value_rule_id, rule_name, rule_name_suffix, schema
  );
}

}  // namespace xgrammar
