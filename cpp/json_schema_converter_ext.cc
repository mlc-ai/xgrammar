/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.cc
 * \brief Implementation of extended format converters.
 */
#include "json_schema_converter_ext.h"

#include "json_schema_converter.h"
#include "regex_converter.h"
#include "support/logging.h"

namespace xgrammar {

// Static constants
const std::string XMLToolCallingConverter::kXMLString = "xml_string";
const std::string XMLToolCallingConverter::kXMLAny = "xml_any";
const std::string XMLToolCallingConverter::kXMLObject = "xml_object";
const std::string XMLToolCallingConverter::kXMLVariableName = "xml_variable_name";

XMLToolCallingConverter::XMLToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver
)
    : JSONSchemaConverter(indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver),
      nested_object_level_(0) {}

std::string XMLToolCallingConverter::Convert(const SchemaSpecPtr& spec) {
  nested_object_level_ = 0;
  AddBasicRules();
  std::string root_rule_name = ebnf_script_creator_.AllocateRuleName("root");
  std::string root_body = GenerateFromSpec(spec, root_rule_name);
  ebnf_script_creator_.AddRuleWithAllocatedName(root_rule_name, root_body);

  return ebnf_script_creator_.GetScript();
}

void XMLToolCallingConverter::AddBasicRules() {
  // First add JSON basic rules. These should be in the inner layer of the XML format.
  XGRAMMAR_DCHECK(nested_object_level_ == 0);
  nested_object_level_ = 2;
  JSONSchemaConverter::AddBasicRules();
  nested_object_level_ = 0;
  // Add XML string rule
  ebnf_script_creator_.AddRule(
      kXMLString,
      "TagDispatch("
      "stop_eos=true,"
      "stop_str=(),"
      "loop_after_dispatch=false,"
      "excludes=(\"</parameter>\")"
      ")"
  );
  constexpr const char* kStringCacheKey = "{\"type\":\"string\"}";
  AddCache(kStringCacheKey, kXMLString);

  // Add XML any rule
  auto any_spec = SchemaSpec::Make(AnySpec{}, "{}", kXMLAny);
  std::string any_body = GenerateAny(std::get<AnySpec>(any_spec->spec), kXMLAny);
  ebnf_script_creator_.AddRule(kXMLAny, any_body);
  AddCache("{}", kXMLAny);

  // Add XML object rule
  constexpr const char* kObjectCacheKey = "{\"type\":\"object\"}";
  ObjectSpec obj_spec_val;
  obj_spec_val.allow_additional_properties = true;
  obj_spec_val.additional_properties_schema = any_spec;
  auto obj_spec = SchemaSpec::Make(std::move(obj_spec_val), kObjectCacheKey, kXMLObject);
  std::string obj_body = GenerateObject(std::get<ObjectSpec>(obj_spec->spec), kXMLObject);
  ebnf_script_creator_.AddRule(kXMLObject, obj_body);
  AddCache(kObjectCacheKey, kXMLObject);

  // Add XML variable name rule
  std::string var_body = "[a-zA-Z_][a-zA-Z0-9_]*";
  ebnf_script_creator_.AddRule(kXMLVariableName, var_body);
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

std::string XMLToolCallingConverter::NextSeparator(bool is_end) {
  if (nested_object_level_ <= 1) {
    return "";
  }
  return JSONSchemaConverter::NextSeparator(is_end);
}

std::string XMLToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  if (nested_object_level_ <= 1) {
    // For XML format, use TagDispatch for strings
    if (!spec.pattern.has_value() && !spec.format.has_value() && spec.min_length == 0 &&
        spec.max_length == -1) {
      return kXMLString;
    }
    if (spec.format.has_value()) {
      const std::string& format = *spec.format;
      auto regex_pattern = JSONFormatToRegexPattern(format);

      if (regex_pattern.has_value()) {
        std::string converted_regex = RegexToEBNF(regex_pattern.value(), false);
        return converted_regex;
      }
    }

    // Check for pattern
    if (spec.pattern.has_value()) {
      std::string converted_regex = RegexToEBNF(*spec.pattern, false);
      return converted_regex;
    }

    // Check for length constraints
    if (spec.min_length != 0 || spec.max_length != -1) {
      std::string char_pattern = "[^]";
      std::string repetition;
      if (spec.max_length == -1) {
        repetition = "{" + std::to_string(spec.min_length) + ",}";
      } else {
        repetition =
            "{" + std::to_string(spec.min_length) + "," + std::to_string(spec.max_length) + "}";
      }
      return char_pattern + repetition;
    }
  }
  return JSONSchemaConverter::GenerateString(spec, rule_name);
}

std::string XMLToolCallingConverter::GenerateAny(
    const AnySpec& spec, const std::string& rule_name
) {
  if (nested_object_level_ <= 1) {
    return kXMLString + " | " + kBasicArray + " | " + kBasicObject;
  }
  return JSONSchemaConverter::GenerateAny(spec, rule_name);
}

std::string XMLToolCallingConverter::GenerateArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  nested_object_level_++;
  auto result = JSONSchemaConverter::GenerateArray(spec, rule_name);
  nested_object_level_--;
  return result;
}

std::string XMLToolCallingConverter::FormatPropertyKey(const std::string& key) {
  if (nested_object_level_ <= 1) {
    return "\"<parameter=" + key + ">\"";
  }
  return JSONSchemaConverter::FormatPropertyKey(key);
}

std::string XMLToolCallingConverter::FormatProperty(
    const std::string& key, const std::string& value_rule, const std::string& rule_name, int64_t idx
) {
  if (nested_object_level_ <= 1) {
    std::string whitespace = GetWhitespacePattern();
    return "\"<parameter=" + key + ">\" " + whitespace + " " + value_rule + " " + whitespace +
           " \"</parameter>\"";
  }
  return JSONSchemaConverter::FormatProperty(key, value_rule, rule_name, idx);
}

std::string XMLToolCallingConverter::FormatOtherProperty(
    const std::string& key_pattern,
    const std::string& value_rule,
    const std::string& rule_name,
    const std::string& rule_name_suffix
) {
  if (nested_object_level_ <= 1) {
    std::string whitespace = GetWhitespacePattern();
    return "\"<parameter=\" " + key_pattern + " \">\" " + whitespace + " " + value_rule + " " +
           whitespace + " \"</parameter>\"";
  }
  return JSONSchemaConverter::FormatOtherProperty(
      key_pattern, value_rule, rule_name, rule_name_suffix
  );
}

std::string XMLToolCallingConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces
) {
  nested_object_level_++;
  bool need_brace = nested_object_level_ > 1;
  auto result = JSONSchemaConverter::GenerateObject(spec, rule_name, need_brace);
  nested_object_level_--;
  return result;
}

void XMLToolCallingConverter::AddCache(const std::string& key, const std::string& value) {
  if (key.empty()) {
    return;
  }
  rule_cache_manager_.AddCache(key, nested_object_level_ > 1, value);
}

std::optional<std::string> XMLToolCallingConverter::GetCache(const std::string& key) const {
  if (key.empty()) {
    return std::nullopt;
  }
  return rule_cache_manager_.GetCache(key, nested_object_level_ > 1);
}

}  // namespace xgrammar
