/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.cc
 * \brief Implementation of extended format converters.
 */
#include "json_schema_converter_ext.h"

#include <picojson.h>

#include <unordered_map>
#include <utility>
#include <vector>

#include "support/logging.h"

namespace xgrammar {

// Static constants
const std::string XMLToolCallingConverter::kXMLString = "xml_string";
const std::string XMLToolCallingConverter::kXMLAny = "xml_any";
const std::string XMLToolCallingConverter::kXMLObject = "xml_object";
const std::string XMLToolCallingConverter::kXMLVariableName = "xml_variable_name";
const std::unordered_map<JSONFormat, XMLToolCallingConverter::XMLWrapper>
    XMLToolCallingConverter::kKeyWrapperMap = {
        {JSONFormat::kQwenXML, {"<parameter=", ">", "", "</parameter>"}},
        {JSONFormat::kMiniMaxXML, {"<parameter name=\"", "\">", "", "</parameter>"}},
        {JSONFormat::kDeepSeekXML,
         {"<｜DSML｜parameter name=\"",
          "",
          "",
          // TODO(Linzhang): We do not validate the string's value, and we accept both.
          "</｜DSML｜parameter>"}},
        {JSONFormat::kGlmXML, {"<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"}},
};

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

Grammar XMLToolCallingConverter::Convert(const SchemaSpecPtr& spec) {
  nested_object_level_ = 0;
  return JSONSchemaConverter::Convert(spec);
}

std::string XMLToolCallingConverter::XMLValue(const std::string& json_value) const {
  picojson::value value;
  std::string error = picojson::parse(value, json_value);
  if (error.empty() && value.is<std::string>()) {
    return value.get<std::string>();
  }
  return json_value;
}

int32_t XMLToolCallingConverter::XMLKeySuffix() {
  if (json_format_ == JSONFormat::kDeepSeekXML) {
    return Sequence(
        {ByteString("\" string=\""),
         Choice({ByteString("true"), ByteString("false")}),
         ByteString("\">")}
    );
  }
  return ByteString(xml_wrapper_.key_wrapper_suffix);
}

void XMLToolCallingConverter::AddBasicRules() {
  // First add JSON basic rules. These should be in the inner layer of the XML format.
  XGRAMMAR_DCHECK(nested_object_level_ == 0);
  // The nested part, true json format, is at level 2.
  nested_object_level_ = 2;
  JSONSchemaConverter::AddBasicRules({kXMLString, kXMLAny, kXMLObject, kXMLVariableName});

  auto any_spec = SchemaSpec::Make(AnySpec{}, "{}", kBasicAny);
  constexpr const char* kStringCacheKey = "{\"type\":\"string\"}";
  constexpr const char* kObjectCacheKey = "{\"type\":\"object\"}";

  // The outer part, xml format, is at level 1.
  nested_object_level_ = 1;
  // Add XML string rule
  builder_.UpdateRuleBody(kXMLString, TagDispatch(false, {xml_wrapper_.parameter_suffix}));
  AddCache(kStringCacheKey, builder_.GetRuleId(kXMLString));

  // Add XML any rule
  builder_.UpdateRuleBody(kXMLAny, GenerateAny(AnySpec{}, kXMLAny));
  AddCache("{}", builder_.GetRuleId(kXMLAny));

  // Reset the nested object level to 0, which is the root level.
  nested_object_level_ = 0;

  // Add XML object rule
  ObjectSpec xml_object_spec;
  xml_object_spec.allow_additional_properties = true;
  xml_object_spec.additional_properties_schema = any_spec;
  builder_.UpdateRuleBody(kXMLObject, GenerateObject(xml_object_spec, kXMLObject));
  AddCache(kObjectCacheKey, builder_.GetRuleId(kXMLObject));

  // Add XML variable name rule
  builder_.UpdateRuleBody(
      kXMLVariableName,
      Sequence(
          {builder_.AddCharacterClass({{'a', 'z'}, {'A', 'Z'}, {'_', '_'}}),
           builder_.AddCharacterClassStar({{'a', 'z'}, {'A', 'Z'}, {'0', '9'}, {'_', '_'}})}
      )
  );
}

std::string XMLToolCallingConverter::GetKeyPattern() const {
  if (nested_object_level_ <= 1) {
    return kXMLVariableName;
  }
  return kBasicString;
}

std::string XMLToolCallingConverter::GetBasicAnyRuleName() const {
  if (nested_object_level_ <= 1) {
    return kXMLAny;
  }
  return kBasicAny;
}

int32_t XMLToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (nested_object_level_ <= 1) {
    return RuleRef(GetKeyPattern());
  }
  return JSONSchemaConverter::GetKeyPatternExcluding(properties, rule_name);
}

std::string XMLToolCallingConverter::NextSeparator(bool is_end) {
  if (nested_object_level_ <= 1) {
    return GetWhitespacePattern();
  }
  return JSONSchemaConverter::NextSeparator(is_end);
}

int32_t XMLToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  if (nested_object_level_ <= 1) {
    if (!spec.pattern.has_value() && !spec.format.has_value() && spec.min_length == 0 &&
        spec.max_length == -1) {
      return RuleRef(kXMLString);
    }
    if (spec.format.has_value()) {
      auto regex = JSONFormatToRegexPattern(*spec.format);
      if (regex.has_value()) {
        return RegexExpression(*regex, false, true);
      }
    }
    if (spec.pattern.has_value()) {
      return RegexExpression(*spec.pattern, false, /*force_cfg_expansion=*/true);
    }
    return Repeat(
        rule_name + "_characters",
        builder_.AddCharacterClass({{0, 0x10ffff}}),
        spec.min_length,
        spec.max_length
    );
  }
  return JSONSchemaConverter::GenerateString(spec, rule_name);
}

int32_t XMLToolCallingConverter::GenerateAny(const AnySpec& spec, const std::string& rule_name) {
  if (nested_object_level_ == 0) {
    return RuleRef(kXMLObject);
  }
  if (nested_object_level_ == 1) {
    return Choice({RuleRef(kXMLString), RuleRef(kBasicArray), RuleRef(kBasicObject)});
  }
  return JSONSchemaConverter::GenerateAny(spec, rule_name);
}

int32_t XMLToolCallingConverter::GenerateArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  nested_object_level_++;
  auto result = JSONSchemaConverter::GenerateArray(spec, rule_name);
  nested_object_level_--;
  return result;
}

int32_t XMLToolCallingConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  if (nested_object_level_ <= 1) {
    return ByteString(XMLValue(spec.json_value));
  }
  return JSONSchemaConverter::GenerateConst(spec, rule_name);
}

int32_t XMLToolCallingConverter::GenerateEnum(const EnumSpec& spec, const std::string& rule_name) {
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
  if (nested_object_level_ <= 1) {
    std::vector<int32_t> values;
    values.reserve(spec.json_values.size());
    for (const auto& value : spec.json_values) {
      values.push_back(ByteString(XMLValue(value)));
    }
    return Choice(values);
  }
  return JSONSchemaConverter::GenerateEnum(spec, rule_name);
}

int32_t XMLToolCallingConverter::FormatPropertyKey(const std::string& key) {
  if (nested_object_level_ <= 1) {
    return Sequence({ByteString(xml_wrapper_.key_wrapper_prefix + key), XMLKeySuffix()});
  }
  return JSONSchemaConverter::FormatPropertyKey(key);
}

int32_t XMLToolCallingConverter::FormatProperty(
    const std::string& key, int32_t value_rule_id, const std::string& rule_name, int64_t idx
) {
  if (nested_object_level_ <= 1) {
    std::vector<int32_t> elements = {FormatPropertyKey(key)};
    if (!xml_wrapper_.value_wrapper_prefix.empty()) {
      elements.push_back(WhitespaceExpression());
      elements.push_back(ByteString(xml_wrapper_.value_wrapper_prefix));
    }
    // xml_string already accepts whitespace. Adding whitespace repetitions around it preserves the
    // language but creates one Earley state for every possible split with the string body.
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
  return JSONSchemaConverter::FormatProperty(key, value_rule_id, rule_name, idx);
}

int32_t XMLToolCallingConverter::FormatOtherProperty(
    int32_t key_pattern_expr,
    int32_t value_rule_id,
    const std::string& rule_name,
    const std::string& rule_name_suffix
) {
  if (nested_object_level_ <= 1) {
    std::vector<int32_t> elements = {
        ByteString(xml_wrapper_.key_wrapper_prefix), key_pattern_expr, XMLKeySuffix()
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
      key_pattern_expr, value_rule_id, rule_name, rule_name_suffix
  );
}

int32_t XMLToolCallingConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces
) {
  nested_object_level_++;
  bool need_brace = nested_object_level_ > 1;
  auto result = JSONSchemaConverter::GenerateObject(spec, rule_name, need_brace);
  nested_object_level_--;
  return result;
}

void XMLToolCallingConverter::AddCache(const std::string& key, int32_t rule_id) {
  if (key.empty()) {
    return;
  }
  rule_cache_manager_.AddCache(key, nested_object_level_ > 1, rule_id);
}

std::optional<int32_t> XMLToolCallingConverter::GetCache(const std::string& key) const {
  if (key.empty()) {
    return std::nullopt;
  }
  return rule_cache_manager_.GetCache(key, nested_object_level_ > 1);
}

}  // namespace xgrammar
