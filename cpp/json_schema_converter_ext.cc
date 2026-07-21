/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.cc
 * \brief Implementation of extended format converters.
 */
#include "json_schema_converter_ext.h"

#include <algorithm>
#include <cctype>
#include <unordered_map>

#include "json_schema_converter.h"
#include "regex_converter.h"

namespace xgrammar {

// Static constants
const std::string XMLToolCallingConverter::kXMLString = "xml_string";
const std::string XMLToolCallingConverter::kXMLAny = "xml_any";
const std::string XMLToolCallingConverter::kXMLObject = "xml_object";
const std::string XMLToolCallingConverter::kXMLVariableName = "xml_variable_name";
const std::unordered_map<JSONFormat, XMLToolCallingConverter::XMLDialectConfig>
    XMLToolCallingConverter::kDialectConfigMap = {
        {JSONFormat::kQwenXML,
         {{"<parameter=", ">", "", "</parameter>", "", false},
          /*recursive=*/false,
          /*array_item_name=*/"",
          /*pad_values_with_whitespace=*/true,
          /*string_terminator=*/"</parameter>",
          /*variable_name_pattern=*/"[a-zA-Z_][a-zA-Z0-9_]*"}},
        {JSONFormat::kMiniMaxXML,
         {{"<parameter name=\\\"", "\\\">", "", "</parameter>", "", false},
          /*recursive=*/false,
          /*array_item_name=*/"",
          /*pad_values_with_whitespace=*/true,
          /*string_terminator=*/"</parameter>",
          /*variable_name_pattern=*/"[a-zA-Z_][a-zA-Z0-9_]*"}},
        {JSONFormat::kDeepSeekXML,
         {{"<｜DSML｜parameter name=\\\"",
           "\\\" string=\\\"\" (\"true\" | \"false\") \"\\\">",
           "",
           "</｜DSML｜parameter>",
           "",
           false},
          /*recursive=*/false,
          /*array_item_name=*/"",
          /*pad_values_with_whitespace=*/true,
          /*string_terminator=*/"</｜DSML｜parameter>",
          /*variable_name_pattern=*/"[a-zA-Z_][a-zA-Z0-9_]*"}},
        {JSONFormat::kGlmXML,
         {{"<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "", false},
          /*recursive=*/false,
          /*array_item_name=*/"",
          /*pad_values_with_whitespace=*/true,
          /*string_terminator=*/"</arg_value>",
          /*variable_name_pattern=*/"[a-zA-Z_][a-zA-Z0-9_]*"}},
        {JSONFormat::kMiniMaxM3XML,
         {{"]<]minimax[>[<", ">", "", "]<]minimax[>[</", ">", true},
          /*recursive=*/true,
          /*array_item_name=*/"item",
          /*pad_values_with_whitespace=*/false,
          /*string_terminator=*/"]<]minimax[>[",
          /*variable_name_pattern=*/"[^/>] [^>]*"}},
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
      nested_object_level_(0),
      dialect_(kDialectConfigMap.at(json_format)) {}

std::string XMLToolCallingConverter::Convert(const SchemaSpecPtr& spec) {
  nested_object_level_ = 0;
  generating_property_name_ = false;
  requires_dynamic_tag_matcher_ = false;
  return JSONSchemaConverter::Convert(spec);
}

void XMLToolCallingConverter::AddBasicRules() {
  if (dialect_.recursive) {
    AddRecursiveXMLBasicRules();
  } else {
    AddRootOnlyXMLBasicRules();
  }
}

void XMLToolCallingConverter::AddRootOnlyXMLBasicRules() {
  // First add JSON basic rules. These should be in the inner layer of the XML format.
  XGRAMMAR_DCHECK(nested_object_level_ == 0);
  // The nested part, true json format, is at level 2.
  nested_object_level_ = 2;
  JSONSchemaConverter::AddBasicRules();
  nested_object_level_ = 1;
  // The outer part, xml format, is at level 1.
  // Add XML string rule
  ebnf_script_creator_.AddRule(
      kXMLString,
      "TagDispatch("
      "loop_after_dispatch=false,"
      "excludes=(\"" +
          dialect_.string_terminator +
          "\")"
          ")"
  );
  constexpr const char* kStringCacheKey = "{\"type\":\"string\"}";
  AddCache(kStringCacheKey, kXMLString);

  // Add XML any rule
  auto any_spec = SchemaSpec::Make(AnySpec{}, "{}", kXMLAny);
  std::string any_body = GenerateAny(std::get<AnySpec>(any_spec->spec), kXMLAny);
  ebnf_script_creator_.AddRule(kXMLAny, any_body);
  AddCache("{}", kXMLAny);

  // Reset the nested object level to 0, which is the root level.
  nested_object_level_ = 0;

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
  ebnf_script_creator_.AddRule(kXMLVariableName, dialect_.variable_name_pattern);
}

void XMLToolCallingConverter::AddRecursiveXMLBasicRules() {
  XGRAMMAR_DCHECK(nested_object_level_ == 0);

  constexpr const char* kIntegerCacheKey = "{\"type\":\"integer\"}";
  constexpr const char* kNumberCacheKey = "{\"type\":\"number\"}";
  constexpr const char* kStringCacheKey = "{\"type\":\"string\"}";
  constexpr const char* kBooleanCacheKey = "{\"type\":\"boolean\"}";
  constexpr const char* kNullCacheKey = "{\"type\":\"null\"}";
  nested_object_level_ = 1;

  auto integer_spec = SchemaSpec::Make(IntegerSpec{}, kIntegerCacheKey, kBasicInteger);
  ebnf_script_creator_.AddRule(
      kBasicInteger, GenerateInteger(std::get<IntegerSpec>(integer_spec->spec), kBasicInteger)
  );
  AddCache(kIntegerCacheKey, kBasicInteger);

  auto number_spec = SchemaSpec::Make(NumberSpec{}, kNumberCacheKey, kBasicNumber);
  ebnf_script_creator_.AddRule(
      kBasicNumber, GenerateNumber(std::get<NumberSpec>(number_spec->spec), kBasicNumber)
  );
  AddCache(kNumberCacheKey, kBasicNumber);

  ebnf_script_creator_.AddRule(
      kXMLString,
      "TagDispatch(loop_after_dispatch=false,excludes=(\"" + dialect_.string_terminator + "\"))"
  );
  AddCache(kStringCacheKey, kXMLString);

  auto boolean_spec = SchemaSpec::Make(BooleanSpec{}, kBooleanCacheKey, kBasicBoolean);
  ebnf_script_creator_.AddRule(
      kBasicBoolean, GenerateBoolean(std::get<BooleanSpec>(boolean_spec->spec), kBasicBoolean)
  );
  AddCache(kBooleanCacheKey, kBasicBoolean);

  auto null_spec = SchemaSpec::Make(NullSpec{}, kNullCacheKey, kBasicNull);
  ebnf_script_creator_.AddRule(
      kBasicNull, GenerateNull(std::get<NullSpec>(null_spec->spec), kBasicNull)
  );
  AddCache(kNullCacheKey, kBasicNull);

  ebnf_script_creator_.AddRule(kXMLVariableName, dialect_.variable_name_pattern);

  const std::string dynamic_element_rule = "xml_dynamic_element";
  const std::string dynamic_children_rule = "xml_dynamic_children";
  ebnf_script_creator_.AddRule(kXMLAny, kXMLString + " | " + dynamic_children_rule);
  ebnf_script_creator_.AddRule(
      dynamic_element_rule,
      EBNFScriptCreator::Concat(
          {EBNFScriptCreator::Str(dialect_.property.open_prefix),
           kXMLVariableName,
           EBNFScriptCreator::Str(dialect_.property.open_suffix),
           kXMLAny,
           EBNFScriptCreator::Str(dialect_.property.close_prefix),
           kXMLVariableName,
           EBNFScriptCreator::Str(dialect_.property.close_suffix)}
      )
  );
  ebnf_script_creator_.AddRule(
      dynamic_children_rule, EBNFScriptCreator::Repeat(dynamic_element_rule, 1, -1)
  );
  AddCache("{}", kXMLAny);

  nested_object_level_ = 0;
  AddCache(kIntegerCacheKey, kBasicInteger);
  AddCache(kNumberCacheKey, kBasicNumber);
  AddCache(kStringCacheKey, kXMLString);
  AddCache(kBooleanCacheKey, kBasicBoolean);
  AddCache(kNullCacheKey, kBasicNull);
}

bool XMLToolCallingConverter::IsXMLLayer() const {
  return dialect_.recursive || nested_object_level_ <= 1;
}

bool XMLToolCallingConverter::IsInnerCacheLayer() const {
  return dialect_.recursive ? nested_object_level_ > 0 : nested_object_level_ > 1;
}

std::string XMLToolCallingConverter::GetKeyPattern() const {
  if (IsXMLLayer()) {
    return kXMLVariableName;
  }
  return kBasicString;
}

std::string XMLToolCallingConverter::GetBasicAnyRuleName() const {
  if (IsXMLLayer()) {
    if (dialect_.recursive) {
      requires_dynamic_tag_matcher_ = true;
    }
    return kXMLAny;
  }
  return kBasicAny;
}

std::string XMLToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (IsXMLLayer()) {
    return GetKeyPattern();
  }
  return JSONSchemaConverter::GetKeyPatternExcluding(properties, rule_name);
}

std::string XMLToolCallingConverter::NextSeparator(bool is_end) {
  if (IsXMLLayer()) {
    return GetWhitespacePattern();
  }
  return JSONSchemaConverter::NextSeparator(is_end);
}

std::string XMLToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  if (dialect_.recursive && generating_property_name_) {
    if (spec.pattern.has_value()) {
      return RegexToEBNF(*spec.pattern, false);
    }
    if (spec.format.has_value()) {
      auto regex_pattern = JSONFormatToRegexPattern(*spec.format);
      if (regex_pattern.has_value()) {
        return RegexToEBNF(*regex_pattern, false);
      }
    }
    if (spec.min_length != 0 || spec.max_length != -1) {
      std::string upper = spec.max_length == -1 ? "" : std::to_string(spec.max_length);
      return "[^>]" + (spec.min_length == 1 && spec.max_length == 1
                           ? std::string("")
                           : "{" + std::to_string(spec.min_length) + "," + upper + "}");
    }
    return kXMLVariableName;
  }
  if (IsXMLLayer()) {
    if (dialect_.recursive) {
      const bool has_known_format =
          spec.format.has_value() && JSONFormatToRegexPattern(*spec.format).has_value();
      if (spec.pattern.has_value() || has_known_format || spec.min_length != 0 ||
          spec.max_length != -1) {
        XGRAMMAR_LOG(FATAL
        ) << "String pattern, recognized format, and length constraints are not supported by "
             "recursive XML dialects because they cannot currently be combined with the "
             "namespace-marker exclusion";
      }
      // Unknown formats are annotations. Fall back to the namespace-safe unconstrained string.
      return kXMLString;
    }

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
  if (dialect_.recursive && generating_property_name_) {
    return kXMLVariableName;
  }
  if (dialect_.recursive) {
    requires_dynamic_tag_matcher_ = true;
    return kXMLAny;
  }
  if (nested_object_level_ == 0) {
    return kXMLObject;
  }
  if (nested_object_level_ == 1) {
    return kXMLString + " | " + kBasicArray + " | " + kBasicObject;
  }
  return JSONSchemaConverter::GenerateAny(spec, rule_name);
}

std::string XMLToolCallingConverter::GenerateArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  nested_object_level_++;
  auto result = dialect_.array_item_name.empty()
                    ? JSONSchemaConverter::GenerateArray(spec, rule_name)
                    : GenerateRepeatedElementArray(spec, rule_name);
  nested_object_level_--;
  return result;
}

std::string XMLToolCallingConverter::GenerateRepeatedElementArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  std::vector<std::string> prefix_items;
  prefix_items.reserve(spec.prefix_items.size());
  for (size_t i = 0; i < spec.prefix_items.size(); ++i) {
    std::string item_rule =
        CreateRule(spec.prefix_items[i], rule_name + "_item_" + std::to_string(i));
    prefix_items.push_back(FormatElement(dialect_.property, dialect_.array_item_name, item_rule));
  }

  std::string additional_item;
  if (spec.allow_additional_items && spec.additional_items) {
    std::string item_rule = CreateRule(spec.additional_items, rule_name + "_additional");
    additional_item = FormatElement(dialect_.property, dialect_.array_item_name, item_rule);
  }

  const std::string empty = EBNFScriptCreator::Str("");
  const std::string whitespace = GetWhitespacePattern();
  if (prefix_items.empty()) {
    if (additional_item.empty() || spec.max_items == 0) {
      return empty;
    }

    int min_items = static_cast<int>(spec.min_items);
    int max_items = spec.max_items == -1 ? -1 : static_cast<int>(spec.max_items);
    std::string nonempty = EBNFScriptCreator::Concat(
        {whitespace,
         additional_item,
         EBNFScriptCreator::Repeat(
             EBNFScriptCreator::Concat({whitespace, additional_item}),
             std::max(0, min_items - 1),
             max_items == -1 ? -1 : std::max(0, max_items - 1)
         ),
         whitespace}
    );
    return min_items == 0 ? EBNFScriptCreator::Or({nonempty, empty}) : nonempty;
  }

  std::vector<std::string> result_parts = {whitespace};
  for (size_t i = 0; i < prefix_items.size(); ++i) {
    if (i != 0) {
      result_parts.push_back(whitespace);
    }
    result_parts.push_back(prefix_items[i]);
  }
  if (!additional_item.empty()) {
    int prefix_count = static_cast<int>(prefix_items.size());
    int min_additional = std::max(0, static_cast<int>(spec.min_items) - prefix_count);
    int max_additional =
        spec.max_items == -1 ? -1 : std::max(0, static_cast<int>(spec.max_items) - prefix_count);
    result_parts.push_back(EBNFScriptCreator::Repeat(
        EBNFScriptCreator::Concat({whitespace, additional_item}), min_additional, max_additional
    ));
  }
  result_parts.push_back(whitespace);
  return EBNFScriptCreator::Concat(result_parts);
}

std::string XMLToolCallingConverter::GenerateLiteral(const picojson::value& value) const {
  if (value.is<std::string>()) {
    const std::string& text = value.get<std::string>();
    XGRAMMAR_CHECK(text.find(dialect_.string_terminator) == std::string::npos)
        << "A recursive XML string literal cannot contain the dialect namespace marker";
    return EBNFScriptCreator::Str(text);
  }
  if (value.is<picojson::object>()) {
    const auto& object = value.get<picojson::object>();
    std::vector<std::string> properties;
    properties.reserve(object.size());
    for (const auto& key : object.ordered_keys()) {
      properties.push_back(FormatElement(dialect_.property, key, GenerateLiteral(object.at(key))));
    }
    return properties.empty() ? EBNFScriptCreator::Str("") : EBNFScriptCreator::Concat(properties);
  }
  if (value.is<picojson::array>()) {
    std::vector<std::string> items;
    const auto& array = value.get<picojson::array>();
    items.reserve(array.size());
    for (const auto& item : array) {
      items.push_back(
          FormatElement(dialect_.property, dialect_.array_item_name, GenerateLiteral(item))
      );
    }
    return items.empty() ? EBNFScriptCreator::Str("") : EBNFScriptCreator::Concat(items);
  }
  return EBNFScriptCreator::Str(value.serialize());
}

std::string XMLToolCallingConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  if (dialect_.recursive && generating_property_name_) {
    picojson::value value;
    std::string error = picojson::parse(value, spec.json_value);
    XGRAMMAR_CHECK(error.empty() && value.is<std::string>())
        << "propertyNames const must be a string";
    ValidateElementName(value.get<std::string>());
    return EBNFScriptCreator::Str(value.get<std::string>());
  }
  if (dialect_.recursive && IsXMLLayer()) {
    picojson::value value;
    std::string error = picojson::parse(value, spec.json_value);
    XGRAMMAR_CHECK(error.empty()) << "Invalid const JSON value: " << error;
    return GenerateLiteral(value);
  }
  if (IsXMLLayer()) {
    const std::string& val = spec.json_value;
    if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
      return "\"" + val.substr(1, val.size() - 2) + "\"";
    }
    return "\"" + val + "\"";
  }
  return JSONSchemaConverter::GenerateConst(spec, rule_name);
}

std::string XMLToolCallingConverter::GenerateEnum(
    const EnumSpec& spec, const std::string& rule_name
) {
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
  if (dialect_.recursive && generating_property_name_) {
    std::vector<std::string> alternatives;
    alternatives.reserve(spec.json_values.size());
    for (const auto& json_value : spec.json_values) {
      picojson::value value;
      std::string error = picojson::parse(value, json_value);
      XGRAMMAR_CHECK(error.empty() && value.is<std::string>())
          << "propertyNames enum values must be strings";
      ValidateElementName(value.get<std::string>());
      alternatives.push_back(EBNFScriptCreator::Str(value.get<std::string>()));
    }
    return EBNFScriptCreator::Or(alternatives);
  }
  if (dialect_.recursive && IsXMLLayer()) {
    std::vector<std::string> alternatives;
    alternatives.reserve(spec.json_values.size());
    for (const auto& json_value : spec.json_values) {
      picojson::value value;
      std::string error = picojson::parse(value, json_value);
      XGRAMMAR_CHECK(error.empty()) << "Invalid enum JSON value: " << error;
      alternatives.push_back(GenerateLiteral(value));
    }
    return EBNFScriptCreator::Or(alternatives);
  }
  if (IsXMLLayer()) {
    std::string result;
    for (size_t i = 0; i < spec.json_values.size(); ++i) {
      if (i != 0) {
        result += " | ";
      }
      const std::string& val = spec.json_values[i];
      if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
        result += "(\"" + val.substr(1, val.size() - 2) + "\")";
      } else {
        result += "(\"" + val + "\")";
      }
    }
    return result;
  }
  return JSONSchemaConverter::GenerateEnum(spec, rule_name);
}

std::string XMLToolCallingConverter::FormatPropertyKey(const std::string& key) {
  if (IsXMLLayer()) {
    ValidateElementName(key);
    if (dialect_.recursive) {
      return EBNFScriptCreator::Str(
          dialect_.property.open_prefix + key + dialect_.property.open_suffix
      );
    }
    return "\"" + dialect_.property.open_prefix + key + dialect_.property.open_suffix + "\"";
  }
  return JSONSchemaConverter::FormatPropertyKey(key);
}

std::string XMLToolCallingConverter::FormatElement(
    const ElementSyntax& syntax, const std::string& key, const std::string& value_rule
) const {
  ValidateElementName(key);
  const std::string open_text = syntax.open_prefix + key + syntax.open_suffix;
  const std::string close_text =
      syntax.close_prefix + std::string(syntax.close_repeats_key ? key : "") + syntax.close_suffix;
  std::string open =
      dialect_.recursive ? EBNFScriptCreator::Str(open_text) : "\"" + open_text + "\"";
  std::string close =
      dialect_.recursive ? EBNFScriptCreator::Str(close_text) : "\"" + close_text + "\"";
  if (!dialect_.pad_values_with_whitespace) {
    return open + " " + value_rule + " " + close;
  }

  std::string whitespace = GetWhitespacePattern();
  // xml_string already accepts whitespace. Adding whitespace repetitions around it preserves the
  // language but creates one Earley state for every possible split with the string body.
  std::string formatted_value =
      value_rule == kXMLString ? value_rule : whitespace + " " + value_rule + " " + whitespace;
  if (!syntax.value_prefix.empty()) {
    return open + " " + whitespace + " \"" + syntax.value_prefix + "\" " + formatted_value + " " +
           close;
  }
  return open + " " + formatted_value + " " + close;
}

std::string XMLToolCallingConverter::FormatProperty(
    const std::string& key, const std::string& value_rule, const std::string& rule_name, int64_t idx
) {
  if (IsXMLLayer()) {
    return FormatElement(dialect_.property, key, value_rule);
  }
  return JSONSchemaConverter::FormatProperty(key, value_rule, rule_name, idx);
}

std::string XMLToolCallingConverter::FormatOtherProperty(
    const std::string& key_pattern,
    const std::string& value_rule,
    const std::string& rule_name,
    const std::string& rule_name_suffix
) {
  if (IsXMLLayer()) {
    const auto& syntax = dialect_.property;
    if (syntax.close_repeats_key) {
      requires_dynamic_tag_matcher_ = true;
    }
    std::string open =
        "\"" + syntax.open_prefix + "\" " + key_pattern + " \"" + syntax.open_suffix + "\"";
    std::string close;
    if (syntax.close_repeats_key) {
      // The CFG accepts the same key language independently at both positions. GrammarMatcher's
      // DynamicTagMatcher supplies the non-context-free equality constraint at runtime.
      close = "\"" + syntax.close_prefix + "\" " + key_pattern + " \"" + syntax.close_suffix + "\"";
    } else {
      close = "\"" + syntax.close_prefix + syntax.close_suffix + "\"";
    }
    if (!dialect_.pad_values_with_whitespace) {
      return open + " " + value_rule + " " + close;
    }

    std::string whitespace = GetWhitespacePattern();
    std::string formatted_value =
        value_rule == kXMLString ? value_rule : whitespace + " " + value_rule + " " + whitespace;
    if (!syntax.value_prefix.empty()) {
      return open + " " + whitespace + " \"" + syntax.value_prefix + "\" " + formatted_value + " " +
             close;
    }
    return open + " " + formatted_value + " " + close;
  }
  return JSONSchemaConverter::FormatOtherProperty(
      key_pattern, value_rule, rule_name, rule_name_suffix
  );
}

std::string XMLToolCallingConverter::FormatPatternProperty(
    const std::string& key_regex,
    const std::string& value_rule,
    const std::string& rule_name,
    const std::string& rule_name_suffix
) {
  if (IsXMLLayer()) {
    return FormatOtherProperty(
        RegexToEBNF(key_regex, false), value_rule, rule_name, rule_name_suffix
    );
  }
  return JSONSchemaConverter::FormatPatternProperty(
      key_regex, value_rule, rule_name, rule_name_suffix
  );
}

std::string XMLToolCallingConverter::CreatePropertyNameRule(
    const SchemaSpecPtr& spec, const std::string& rule_name_hint
) {
  if (!dialect_.recursive) {
    return JSONSchemaConverter::CreatePropertyNameRule(spec, rule_name_hint);
  }
  std::string rule_name = ebnf_script_creator_.AllocateRuleName(rule_name_hint);
  bool old_generating_property_name = generating_property_name_;
  generating_property_name_ = true;
  std::string rule_body = GenerateFromSpec(spec, rule_name);
  generating_property_name_ = old_generating_property_name;
  ebnf_script_creator_.AddRuleWithAllocatedName(rule_name, rule_body);
  return rule_name;
}

std::string XMLToolCallingConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces
) {
  if (dialect_.recursive) {
    ValidateRecursiveObject(spec);
  }
  nested_object_level_++;
  bool need_brace = !dialect_.recursive && nested_object_level_ > 1;
  bool saved_any_whitespace = any_whitespace_;
  if (dialect_.recursive) {
    // Empty M3 objects are encoded as an empty element body, not whitespace text.
    any_whitespace_ = false;
  }
  auto result = JSONSchemaConverter::GenerateObject(spec, rule_name, need_brace);
  any_whitespace_ = saved_any_whitespace;
  nested_object_level_--;
  return result;
}

void XMLToolCallingConverter::ValidateRecursiveObject(const ObjectSpec& spec) const {
  for (const auto& property : spec.properties) {
    ValidateElementName(property.name);
  }
}

void XMLToolCallingConverter::ValidateElementName(const std::string& name) const {
  if (!dialect_.recursive) {
    return;
  }
  XGRAMMAR_CHECK(!name.empty() && name.front() != '/' && name.find('>') == std::string::npos)
      << "Invalid recursive XML element name: " << name;
  XGRAMMAR_CHECK(std::any_of(name.begin(), name.end(), [](unsigned char ch) {
    return !std::isspace(ch);
  })) << "Recursive XML element names cannot be blank";
}

void XMLToolCallingConverter::AddCache(const std::string& key, const std::string& value) {
  if (key.empty()) {
    return;
  }
  rule_cache_manager_.AddCache(key, IsInnerCacheLayer(), value);
}

std::optional<std::string> XMLToolCallingConverter::GetCache(const std::string& key) const {
  if (key.empty()) {
    return std::nullopt;
  }
  auto cached = rule_cache_manager_.GetCache(key, IsInnerCacheLayer());
  if (dialect_.recursive && cached == kXMLAny) {
    requires_dynamic_tag_matcher_ = true;
  }
  return cached;
}

int XMLToolCallingConverter::GetRefCacheDomain() const {
  if (dialect_.recursive) {
    return generating_property_name_ ? 1 : 0;
  }
  // Root-only XML formats have three distinct reference encodings: the XML root, direct XML
  // parameter values, and nested JSON. Keeping their URI maps separate prevents a nested self-ref
  // from accidentally pointing back to the XML root rule.
  return std::min(nested_object_level_, 2);
}

}  // namespace xgrammar
