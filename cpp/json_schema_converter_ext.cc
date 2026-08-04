/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.cc
 * \brief Implementation of extended format converters.
 */
#include "json_schema_converter_ext.h"

#include <algorithm>
#include <map>
#include <type_traits>
#include <unordered_map>

#include "json_schema_converter.h"
#include "regex_converter.h"

namespace xgrammar {

namespace {

struct XMLIdentifierTrieNode {
  bool is_terminal = false;
  std::map<char, XMLIdentifierTrieNode> children;
};

bool IsXMLIdentifierChar(char c, bool is_first) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
         (!is_first && c >= '0' && c <= '9');
}

std::string XMLIdentifierCharClassExcluding(
    const std::map<char, XMLIdentifierTrieNode>& children, bool is_first
) {
  std::string chars;
  for (char c = 'A'; c <= 'Z'; ++c) {
    if (!children.count(c)) chars += c;
  }
  chars += "_";
  if (children.count('_')) {
    chars.pop_back();
  }
  for (char c = 'a'; c <= 'z'; ++c) {
    if (!children.count(c)) chars += c;
  }
  if (!is_first) {
    for (char c = '0'; c <= '9'; ++c) {
      if (!children.count(c)) chars += c;
    }
  }
  if (chars.empty()) {
    return "";
  }
  return "[" + chars + "]";
}

std::string BuildXMLIdentifierExcludingBody(const XMLIdentifierTrieNode& node, int depth) {
  std::vector<std::string> alternatives;
  if (depth > 0 && !node.is_terminal) {
    alternatives.push_back("\"\"");
  }

  std::string divergent_char = XMLIdentifierCharClassExcluding(node.children, depth == 0);
  if (!divergent_char.empty()) {
    alternatives.push_back(divergent_char + " [a-zA-Z0-9_]*");
  }

  for (const auto& [c, child] : node.children) {
    if (!IsXMLIdentifierChar(c, depth == 0)) {
      continue;
    }
    alternatives.push_back(
        EBNFScriptCreator::Str(std::string(1, c)) + " " +
        BuildXMLIdentifierExcludingBody(child, depth + 1)
    );
  }

  if (alternatives.empty()) {
    return "\"\"";
  }
  return EBNFScriptCreator::Or(alternatives);
}

}  // namespace

// Static constants
const std::string XMLToolCallingConverter::kXMLString = "xml_string";
const std::string XMLToolCallingConverter::kXMLAny = "xml_any";
const std::string XMLToolCallingConverter::kXMLObject = "xml_object";
const std::string XMLToolCallingConverter::kXMLVariableName = "xml_variable_name";
const std::unordered_map<JSONFormat, XMLToolCallingConverter::XMLWrapper>
    XMLToolCallingConverter::kKeyWrapperMap = {
        {JSONFormat::kQwenXML, {"<parameter=", ">", "", "</parameter>"}},
        {JSONFormat::kMiniMaxXML, {"<parameter name=\\\"", "\\\">", "", "</parameter>"}},
        {JSONFormat::kDeepSeekXML,
         {"<｜DSML｜parameter name=\\\"",
          "\\\" string=\\\"\" (\"true\" | \"false\") \"\\\">",
          "",
          // TODO(Linzhang): we do not validate the string's value, and we accept both.
          "</｜DSML｜parameter>"}},
        {JSONFormat::kGlmXML, {"<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"}},
        {JSONFormat::kCohereXML, {"<cofl:value", ">", "", "</cofl:value>"}},
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
      xml_wrapper_(kKeyWrapperMap.at(json_format)) {}

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
          xml_wrapper_.parameter_suffix +
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

std::string XMLToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (nested_object_level_ <= 1) {
    return GetKeyPattern();
  }
  return JSONSchemaConverter::GetKeyPatternExcluding(properties, rule_name);
}

std::string XMLToolCallingConverter::NextSeparator(bool is_end) {
  if (nested_object_level_ <= 1) {
    return GetWhitespacePattern();
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
  auto result = JSONSchemaConverter::GenerateArray(spec, rule_name);
  nested_object_level_--;
  return result;
}

std::string XMLToolCallingConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  if (nested_object_level_ <= 1) {
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
  if (nested_object_level_ <= 1) {
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
  if (nested_object_level_ <= 1) {
    return "\"" + xml_wrapper_.key_wrapper_prefix + key + xml_wrapper_.key_wrapper_suffix + "\"";
  }
  return JSONSchemaConverter::FormatPropertyKey(key);
}

std::string XMLToolCallingConverter::FormatProperty(
    const std::string& key, const std::string& value_rule, const std::string& rule_name, int64_t idx
) {
  if (nested_object_level_ <= 1) {
    std::string whitespace = GetWhitespacePattern();
    // xml_string already accepts whitespace. Adding whitespace repetitions around it preserves the
    // language but creates one Earley state for every possible split with the string body.
    std::string formatted_value =
        value_rule == kXMLString ? value_rule : whitespace + " " + value_rule + " " + whitespace;
    if (!xml_wrapper_.value_wrapper_prefix.empty()) {
      return "\"" + xml_wrapper_.key_wrapper_prefix + key + xml_wrapper_.key_wrapper_suffix +
             "\" " + whitespace + " \"" + xml_wrapper_.value_wrapper_prefix + "\" " +
             formatted_value + " \"" + xml_wrapper_.parameter_suffix + "\"";
    }
    return "\"" + xml_wrapper_.key_wrapper_prefix + key + xml_wrapper_.key_wrapper_suffix + "\" " +
           formatted_value + " \"" + xml_wrapper_.parameter_suffix + "\"";
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
    std::string formatted_value =
        value_rule == kXMLString ? value_rule : whitespace + " " + value_rule + " " + whitespace;
    if (!xml_wrapper_.value_wrapper_prefix.empty()) {
      return "\"" + xml_wrapper_.key_wrapper_prefix + "\" " + key_pattern + " \"" +
             xml_wrapper_.key_wrapper_suffix + "\" " + whitespace + " \"" +
             xml_wrapper_.value_wrapper_prefix + "\" " + formatted_value + " \"" +
             xml_wrapper_.parameter_suffix + "\"";
    }
    return "\"" + xml_wrapper_.key_wrapper_prefix + "\" " + key_pattern + " \"" +
           xml_wrapper_.key_wrapper_suffix + "\" " + formatted_value + " \"" +
           xml_wrapper_.parameter_suffix + "\"";
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

CohereXMLToolCallingConverter::CohereXMLToolCallingConverter(
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
          JSONFormat::kCohereXML,
          any_order
      ) {}

bool CohereXMLToolCallingConverter::InCohereValueContext() const {
  return nested_object_level_ <= 1 || !object_stack_.empty() || cohere_array_level_ > 0;
}

std::string CohereXMLToolCallingConverter::FormatCohereValue(
    const std::string& value_rule
) const {
  if (value_rule == kXMLString) {
    return value_rule;
  }
  std::string whitespace = GetWhitespacePattern();
  return whitespace + " " + value_rule + " " + whitespace;
}

std::string CohereXMLToolCallingConverter::GetCohereTypePattern(
    const SchemaSpecPtr& schema
) const {
  return std::visit(
      [](const auto& spec) -> std::string {
        using T = std::decay_t<decltype(spec)>;
        if constexpr (std::is_same_v<T, StringSpec>) {
          return EBNFScriptCreator::Str("raw");
        } else if constexpr (std::is_same_v<T, ObjectSpec>) {
          return EBNFScriptCreator::Str("dict");
        } else if constexpr (std::is_same_v<T, ArraySpec>) {
          return EBNFScriptCreator::Str("list");
        } else {
          return EBNFScriptCreator::Str("json");
        }
      },
      schema->spec
  );
}

std::string CohereXMLToolCallingConverter::FormatCohereParam(
    const std::optional<std::string>& name,
    const std::string& key_pattern,
    const SchemaSpecPtr& schema,
    const std::string& value_rule
) {
  std::string result = EBNFScriptCreator::Str(xml_wrapper_.key_wrapper_prefix);
  if (name.has_value()) {
    result += " " + EBNFScriptCreator::Str(" name=\"" + *name + "\"");
  } else if (!key_pattern.empty()) {
    result += " " + EBNFScriptCreator::Str(" name=\"") + " " + key_pattern + " " +
              EBNFScriptCreator::Str("\"");
  }
  result += " " + EBNFScriptCreator::Str(" type=\"") + " " + GetCohereTypePattern(schema) + " " +
            EBNFScriptCreator::Str("\"" + xml_wrapper_.key_wrapper_suffix);

  if (!xml_wrapper_.value_wrapper_prefix.empty()) {
    result += " " + EBNFScriptCreator::Str(xml_wrapper_.value_wrapper_prefix);
  }
  result += " " + FormatCohereValue(value_rule) + " " +
            EBNFScriptCreator::Str(xml_wrapper_.parameter_suffix);
  return result;
}

std::string CohereXMLToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  if (!InCohereValueContext()) {
    return JSONSchemaConverter::GenerateString(spec, rule_name);
  }
  if (!spec.pattern.has_value() && !spec.format.has_value() && spec.min_length == 0 &&
      spec.max_length == -1) {
    return kXMLString;
  }
  if (spec.format.has_value()) {
    const std::string& format = *spec.format;
    auto regex_pattern = JSONFormatToRegexPattern(format);
    if (regex_pattern.has_value()) {
      return RegexToEBNF(regex_pattern.value(), false);
    }
  }
  if (spec.pattern.has_value()) {
    return RegexToEBNF(*spec.pattern, false);
  }
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
  return JSONSchemaConverter::GenerateString(spec, rule_name);
}

std::string CohereXMLToolCallingConverter::GenerateAny(
    const AnySpec& spec, const std::string& rule_name
) {
  if (!InCohereValueContext()) {
    return JSONSchemaConverter::GenerateAny(spec, rule_name);
  }
  if (nested_object_level_ == 0) {
    return kXMLObject;
  }
  return JSONSchemaConverter::GenerateAny(spec, rule_name);
}

std::string CohereXMLToolCallingConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces
) {
  nested_object_level_++;
  bool use_cohere_object = InCohereValueContext();

  std::string result;
  if (use_cohere_object) {
    SchemaSpecPtr additional_property;
    if (spec.allow_additional_properties && spec.additional_properties_schema) {
      additional_property = spec.additional_properties_schema;
    } else if (spec.allow_unevaluated_properties && spec.unevaluated_properties_schema) {
      additional_property = spec.unevaluated_properties_schema;
    } else if (spec.allow_additional_properties || spec.allow_unevaluated_properties) {
      additional_property = SchemaSpec::Make(AnySpec{}, "", "any");
    }

    object_stack_.push_back(&spec);
    additional_property_stack_.push_back(additional_property);
    result = JSONSchemaConverter::GenerateObject(spec, rule_name, false);
    additional_property_stack_.pop_back();
    object_stack_.pop_back();
  } else {
    result = JSONSchemaConverter::GenerateObject(spec, rule_name, nested_object_level_ > 1);
  }

  nested_object_level_--;
  return result;
}

std::string CohereXMLToolCallingConverter::GenerateArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  if (!InCohereValueContext()) {
    nested_object_level_++;
    auto result = JSONSchemaConverter::GenerateArray(spec, rule_name);
    nested_object_level_--;
    return result;
  }

  cohere_array_level_++;
  std::vector<std::string> item_patterns;
  for (size_t i = 0; i < spec.prefix_items.size(); ++i) {
    std::string item_rule =
        CreateRule(spec.prefix_items[i], rule_name + "_item_" + std::to_string(i));
    item_patterns.push_back(FormatCohereParam(std::nullopt, "", spec.prefix_items[i], item_rule));
  }

  std::string additional_item_pattern;
  if (spec.allow_additional_items && spec.additional_items) {
    std::string additional_rule = CreateRule(spec.additional_items, rule_name + "_additional");
    additional_item_pattern =
        FormatCohereParam(std::nullopt, "", spec.additional_items, additional_rule);
  }
  cohere_array_level_--;

  if (item_patterns.empty()) {
    if (additional_item_pattern.empty() || spec.max_items == 0) {
      return "\"\"";
    }
    std::string repeated_item = "(" + additional_item_pattern + ")";
    return EBNFScriptCreator::Repeat(
        repeated_item,
        static_cast<int>(spec.min_items),
        spec.max_items == -1 ? -1 : static_cast<int>(spec.max_items)
    );
  }

  std::string prefix_part = EBNFScriptCreator::Concat(item_patterns);
  if (additional_item_pattern.empty()) {
    return prefix_part;
  }

  int64_t min_additional =
      std::max(static_cast<int64_t>(0), spec.min_items - static_cast<int64_t>(item_patterns.size()));
  int64_t max_additional =
      spec.max_items == -1 ? -1 : spec.max_items - static_cast<int64_t>(item_patterns.size());
  std::string repeated_item = "(" + additional_item_pattern + ")";
  return prefix_part + " " +
         EBNFScriptCreator::Repeat(
             repeated_item,
             static_cast<int>(min_additional),
             max_additional == -1 ? -1 : static_cast<int>(max_additional)
         );
}

std::string CohereXMLToolCallingConverter::FormatProperty(
    const std::string& key, const std::string& value_rule, const std::string& rule_name, int64_t idx
) {
  if (!object_stack_.empty() && idx >= 0 &&
      idx < static_cast<int64_t>(object_stack_.back()->properties.size())) {
    const auto& prop = object_stack_.back()->properties[idx];
    return FormatCohereParam(prop.name, "", prop.schema, value_rule);
  }
  return XMLToolCallingConverter::FormatProperty(key, value_rule, rule_name, idx);
}

std::string CohereXMLToolCallingConverter::FormatOtherProperty(
    const std::string& key_pattern,
    const std::string& value_rule,
    const std::string& rule_name,
    const std::string& rule_name_suffix
) {
  if (!additional_property_stack_.empty() && additional_property_stack_.back()) {
    return FormatCohereParam(
        std::nullopt, key_pattern, additional_property_stack_.back(), value_rule
    );
  }
  return XMLToolCallingConverter::FormatOtherProperty(
      key_pattern, value_rule, rule_name, rule_name_suffix
  );
}

std::string CohereXMLToolCallingConverter::GetKeyPattern() const {
  if (InCohereValueContext()) {
    return kXMLVariableName;
  }
  return JSONSchemaConverter::GetKeyPattern();
}

std::string CohereXMLToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (InCohereValueContext()) {
    if (properties.empty()) {
      return GetKeyPattern();
    }
    XMLIdentifierTrieNode root;
    for (const auto& prop : properties) {
      XMLIdentifierTrieNode* cur = &root;
      bool is_valid_identifier = true;
      for (size_t i = 0; i < prop.name.size(); ++i) {
        char c = prop.name[i];
        if (!IsXMLIdentifierChar(c, i == 0)) {
          is_valid_identifier = false;
          break;
        }
        cur = &cur->children[c];
      }
      if (is_valid_identifier && !prop.name.empty()) {
        cur->is_terminal = true;
      }
    }
    return ebnf_script_creator_.AddRule(
        rule_name + "_cohere_addl_key", BuildXMLIdentifierExcludingBody(root, 0)
    );
  }
  return JSONSchemaConverter::GetKeyPatternExcluding(properties, rule_name);
}

std::string CohereXMLToolCallingConverter::NextSeparator(bool is_end) {
  if (InCohereValueContext()) {
    return GetWhitespacePattern();
  }
  return JSONSchemaConverter::NextSeparator(is_end);
}

void CohereXMLToolCallingConverter::AddCache(const std::string& key, const std::string& value) {
  if (key.empty()) {
    return;
  }
  rule_cache_manager_.AddCache(key, nested_object_level_ > 1 && !InCohereValueContext(), value);
}

std::optional<std::string> CohereXMLToolCallingConverter::GetCache(
    const std::string& key
) const {
  if (key.empty()) {
    return std::nullopt;
  }
  return rule_cache_manager_.GetCache(key, nested_object_level_ > 1 && !InCohereValueContext());
}

}  // namespace xgrammar
