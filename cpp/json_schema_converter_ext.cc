/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.cc
 * \brief Implementation of extended format converters.
 */
#include "json_schema_converter_ext.h"

#include <unordered_map>

#include "json_schema_converter.h"
#include "regex_converter.h"

namespace xgrammar {

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

// ==================== GemmaToolCallingConverter ====================

const std::string GemmaToolCallingConverter::kGemmaStringDelim = "<|\"|>";
const std::string GemmaToolCallingConverter::kGemmaStringContent = "gemma_string_content";
const std::string GemmaToolCallingConverter::kGemmaVariableName = "gemma_variable_name";
const std::string GemmaToolCallingConverter::kGemmaBoundedChar = "gemma_bounded_char";

GemmaToolCallingConverter::GemmaToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    bool any_order
)
    : JSONSchemaConverter(
          indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver, any_order
      ) {}

void GemmaToolCallingConverter::AddBasicRules() {
  // Any text not containing the string delimiter. The enclosing rules terminate it with
  // the delimiter itself, so the boundary is unambiguous.
  ebnf_script_creator_.AddRule(
      kGemmaStringContent,
      "TagDispatch(loop_after_dispatch=false,excludes=(" +
          EBNFScriptCreator::Str(kGemmaStringDelim) + "))"
  );
  ebnf_script_creator_.AddRule(kGemmaVariableName, "[a-zA-Z_] [a-zA-Z0-9_]*");
  // A single string-content character for length-constrained strings: any character
  // except a '<' that starts the <|"|> delimiter. Since lookahead assertions cannot
  // contain choices, each "the following text diverges from the delimiter at position
  // k" case is a separate rule.
  ebnf_script_creator_.AddRule(kGemmaBoundedChar + "_lt1", "\"<\" (= [^|])");
  ebnf_script_creator_.AddRule(kGemmaBoundedChar + "_lt2", "\"<\" (= \"|\" [^\\\"])");
  ebnf_script_creator_.AddRule(kGemmaBoundedChar + "_lt3", "\"<\" (= \"|\\\"\" [^|])");
  ebnf_script_creator_.AddRule(kGemmaBoundedChar + "_lt4", "\"<\" (= \"|\\\"|\" [^>])");
  ebnf_script_creator_.AddRule(
      kGemmaBoundedChar,
      "[^<] | " + kGemmaBoundedChar + "_lt1 | " + kGemmaBoundedChar + "_lt2 | " +
          kGemmaBoundedChar + "_lt3 | " + kGemmaBoundedChar + "_lt4"
  );
  // The base implementation builds all basic rules through the virtual Generate* methods,
  // so basic_string / basic_any / basic_object etc. all come out in Gemma format.
  JSONSchemaConverter::AddBasicRules();
}

std::string GemmaToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  std::string delim = EBNFScriptCreator::Str(kGemmaStringDelim);

  // NOTE: for the format/pattern branches the regex is emitted verbatim between the
  // delimiters. A permissive pattern (e.g. ".*") can therefore generate text containing
  // the <|"|> delimiter itself, which closes the string early for downstream parsers.
  // Intersecting an arbitrary regex with "does not contain the delimiter" is not
  // supported; keep patterns restrictive.
  if (spec.format.has_value()) {
    auto regex_pattern = JSONFormatToRegexPattern(*spec.format);
    if (regex_pattern.has_value()) {
      return delim + " " + RegexToEBNF(regex_pattern.value(), false) + " " + delim;
    }
  }

  if (spec.pattern.has_value()) {
    return delim + " " + RegexToEBNF(*spec.pattern, false) + " " + delim;
  }

  if (spec.min_length != 0 || spec.max_length != -1) {
    // Count repetitions of kGemmaBoundedChar so min/max length are enforced per character.
    std::string repetition;
    if (spec.max_length == -1) {
      repetition = "{" + std::to_string(spec.min_length) + ",}";
    } else {
      repetition =
          "{" + std::to_string(spec.min_length) + "," + std::to_string(spec.max_length) + "}";
    }
    return delim + " " + kGemmaBoundedChar + repetition + " " + delim;
  }

  return delim + " " + kGemmaStringContent + " " + delim;
}

std::string GemmaToolCallingConverter::SerializeJSON(const picojson::value& value) {
  if (value.is<std::string>()) {
    // Gemma strings have no escape sequences; the raw content is emitted between delimiters.
    return kGemmaStringDelim + value.get<std::string>() + kGemmaStringDelim;
  }
  if (value.is<picojson::object>()) {
    const auto& obj = value.get<picojson::object>();
    std::string result = "{";
    bool first = true;
    for (const auto& [key, val] : obj) {
      if (!first) {
        result += ",";
      }
      first = false;
      result += key + ":" + SerializeJSON(val);
    }
    return result + "}";
  }
  if (value.is<picojson::array>()) {
    const auto& arr = value.get<picojson::array>();
    std::string result = "[";
    for (size_t i = 0; i < arr.size(); ++i) {
      if (i != 0) {
        result += ",";
      }
      result += SerializeJSON(arr[i]);
    }
    return result + "]";
  }
  // Numbers, booleans and null keep their JSON serialization.
  return value.serialize();
}

std::string GemmaToolCallingConverter::ValueToEBNFLiteral(const std::string& json_value) {
  picojson::value value;
  std::string err = picojson::parse(value, json_value);
  XGRAMMAR_CHECK(err.empty()) << "Failed to parse JSON value: " << err
                              << ". The JSON string is: " << json_value;
  if (!value.is<std::string>() && !value.is<picojson::object>() && !value.is<picojson::array>()) {
    // Scalars (numbers, booleans, null) are identical in Gemma and JSON form. Keep the
    // original literal text: round-tripping numbers through picojson's double storage
    // would lose precision for large integers and change formatting (e.g. 1.0 -> 1).
    std::string trimmed = json_value;
    trimmed.erase(0, trimmed.find_first_not_of(" \n\t\r"));
    trimmed.erase(trimmed.find_last_not_of(" \n\t\r") + 1);
    return EBNFScriptCreator::Str(trimmed);
  }
  // Strings, objects and arrays must be re-serialized into Gemma form. Note: numbers
  // nested inside objects/arrays still go through picojson and may lose precision.
  return EBNFScriptCreator::Str(SerializeJSON(value));
}

std::string GemmaToolCallingConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  return ValueToEBNFLiteral(spec.json_value);
}

std::string GemmaToolCallingConverter::GenerateEnum(
    const EnumSpec& spec, const std::string& rule_name
) {
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
  std::string result;
  for (size_t i = 0; i < spec.json_values.size(); ++i) {
    if (i != 0) {
      result += " | ";
    }
    result += "(" + ValueToEBNFLiteral(spec.json_values[i]) + ")";
  }
  return result;
}

std::string GemmaToolCallingConverter::FormatPropertyKey(const std::string& key) {
  // Keys are unquoted in Gemma format.
  return EBNFScriptCreator::Str(key);
}

std::string GemmaToolCallingConverter::GetKeyPattern() const { return kGemmaVariableName; }

std::string GemmaToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  // The trie-based exclusion in the base class is built for JSON-quoted keys; Gemma keys
  // are bare identifiers, so fall back to the plain key pattern.
  return GetKeyPattern();
}

std::string GemmaToolCallingConverter::FormatPatternKey(const std::string& pattern) {
  // Gemma keys are unquoted; emit the bare pattern. A pattern whose language contains
  // ':' is ambiguous against the key/value separator and cannot round-trip.
  return "(" + RegexToEBNF(pattern, false) + ")";
}

std::string GemmaToolCallingConverter::CreatePropertyNamesKeyRule(
    const SchemaSpecPtr& property_names, const std::string& rule_name_hint
) {
  // Gemma keys are bare identifiers, not <|"|>-delimited strings. Constrain by the
  // propertyNames pattern when one is given; otherwise fall back to the identifier rule.
  if (auto* str_spec = std::get_if<StringSpec>(&property_names->spec)) {
    if (str_spec->pattern.has_value()) {
      return "(" + RegexToEBNF(*str_spec->pattern, false) + ")";
    }
  }
  return GetKeyPattern();
}

}  // namespace xgrammar
