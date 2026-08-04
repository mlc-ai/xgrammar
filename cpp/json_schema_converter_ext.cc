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
#include <picojson.h>

#include <unordered_map>
#include <utility>
#include <vector>

#include "support/logging.h"

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
        {JSONFormat::kMiniMaxXML, {"<parameter name=\"", "\">", "", "</parameter>"}},
        {JSONFormat::kDeepSeekXML,
         {"<｜DSML｜parameter name=\"",
          "",
          "",
          // TODO(Linzhang): We do not validate the string's value, and we accept both.
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
