/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.cc
 * \brief Implementation of extended format converters.
 */
#include "json_schema_converter_ext.h"

#include "regex_converter.h"
#include "support/logging.h"

namespace xgrammar {

// Static constants
const std::string XMLToolCallingConverter::kXMLString = "xml_string";
const std::string XMLToolCallingConverter::kXMLAny = "xml_any";
const std::string XMLToolCallingConverter::kXMLVariableName = "xml_variable_name";

XMLToolCallingConverter::XMLToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt
)
    : JSONSchemaConverter(indent, separators, any_whitespace, max_whitespace_cnt),
      is_root_object_(true) {}

std::string XMLToolCallingConverter::Convert(const SchemaSpecPtr& spec) {
  AddBasicRules();

  is_root_object_ = true;
  std::string root_rule_name = ebnf_script_creator_.AllocateRuleName("root");
  std::string root_body = GenerateFromSpec(spec, root_rule_name);
  ebnf_script_creator_.AddRuleWithAllocatedName(root_rule_name, root_body);

  return ebnf_script_creator_.GetScript();
}

void XMLToolCallingConverter::AddBasicRules() {
  // First add JSON basic rules
  JSONSchemaConverter::AddBasicRules();

  // Then add XML-specific rules
  AddXMLHelperRules();

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

  // Add XML any rule
  auto any_body = kBasicNumber + " | " + kXMLString + " | " + kBasicBoolean + " | " + kBasicNull +
                  " | " + kBasicArray + " | " + kBasicObject;
  ebnf_script_creator_.AddRule(kXMLAny, any_body);
}

void XMLToolCallingConverter::AddXMLHelperRules() {
  ebnf_script_creator_.AddRule(kXMLVariableName, "[a-zA-Z_] [a-zA-Z0-9_]*");
}

std::string XMLToolCallingConverter::GetBasicStringRuleName() const {
  if (is_root_object_) {
    return kXMLString;
  }
  return kBasicString;
}

std::string XMLToolCallingConverter::GetBasicAnyRuleName() const {
  if (is_root_object_) {
    return kXMLAny;
  }
  return kBasicAny;
}

std::string XMLToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  if (is_root_object_) {
    // For XML format, use TagDispatch for strings
    if (!spec.pattern.has_value() && !spec.format.has_value() && spec.min_length == 0 &&
        spec.max_length == -1) {
      return kXMLString;
    }
    // For constrained strings, still use TagDispatch but with constraints
    // (simplified - in practice you might need more sophisticated handling)
    return kXMLString;
  }
  return JSONSchemaConverter::GenerateString(spec, rule_name);
}

std::string XMLToolCallingConverter::GenerateAny(
    const AnySpec& spec, const std::string& rule_name
) {
  if (is_root_object_) {
    return kBasicNumber + " | " + kXMLString + " | " + kBasicBoolean + " | " + kBasicNull + " | " +
           kBasicArray + " | " + kBasicObject;
  }
  return JSONSchemaConverter::GenerateAny(spec, rule_name);
}

std::string XMLToolCallingConverter::FormatPropertyKey(const std::string& key) {
  if (is_root_object_) {
    return "\"<parameter=" + key + ">\"";
  }
  return JSONSchemaConverter::FormatPropertyKey(key);
}

std::string XMLToolCallingConverter::FormatProperty(
    const std::string& key, const std::string& value_rule, const std::string& rule_name, int64_t idx
) {
  if (is_root_object_) {
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
  if (is_root_object_) {
    std::string whitespace = GetWhitespacePattern();
    return "\"<parameter=\" " + key_pattern + " \">\" " + whitespace + " " + value_rule + " " +
           whitespace + " \"</parameter>\"";
  }
  return JSONSchemaConverter::FormatOtherProperty(
      key_pattern, value_rule, rule_name, rule_name_suffix
  );
}

std::string XMLToolCallingConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name
) {
  if (is_root_object_) {
    // For the root object, use XML format
    is_root_object_ = false;  // Inner objects will use JSON format

    std::string result = "";
    bool could_be_empty = false;

    // Determine additional property handling
    std::string additional_suffix = "";
    SchemaSpecPtr additional_property;
    if (spec.allow_additional_properties && spec.additional_properties_schema) {
      additional_suffix = "addl";
      additional_property = spec.additional_properties_schema;
    } else if (spec.allow_unevaluated_properties && spec.unevaluated_properties_schema) {
      additional_suffix = "uneval";
      additional_property = spec.unevaluated_properties_schema;
    } else if (spec.allow_additional_properties || spec.allow_unevaluated_properties) {
      additional_suffix = "addl";
      additional_property = SchemaSpec::Make(AnySpec{}, "", "any");
    }

    std::string whitespace = GetWhitespacePattern();

    if (!spec.pattern_properties.empty() || spec.property_names) {
      // Case 1: patternProperties or propertyNames defined
      std::string property_rule_body = "(";
      if (spec.max_properties != 0) {
        if (!spec.pattern_properties.empty()) {
          for (size_t i = 0; i < spec.pattern_properties.size(); ++i) {
            const auto& pp = spec.pattern_properties[i];
            std::string value = CreateRule(pp.schema, rule_name + "_prop_" + std::to_string(i));
            std::string property_pattern = "\"<parameter=\" " + RegexToEBNF(pp.pattern, false) +
                                           " \">\" " + whitespace + " " + value + " " + whitespace +
                                           " \"</parameter>\"";
            if (i != 0) {
              property_rule_body += " | ";
            }
            property_rule_body += "(" + property_pattern + ")";
          }
          property_rule_body += ")";
        } else {
          auto key_pattern = CreateRule(spec.property_names, rule_name + "_name");
          property_rule_body += "\"<parameter=\" " + key_pattern + " \">\" " + whitespace + " " +
                                GetBasicAnyRuleName() + " " + whitespace + " \"</parameter>\")";
        }

        auto prop_rule_name = ebnf_script_creator_.AllocateRuleName(rule_name + "_prop");
        ebnf_script_creator_.AddRuleWithAllocatedName(prop_rule_name, property_rule_body);

        result += prop_rule_name + " " +
                  GetPropertyWithNumberConstraints(
                      prop_rule_name, spec.min_properties, spec.max_properties, 1
                  );
        could_be_empty = spec.min_properties == 0;
      }
    } else if (!spec.properties.empty()) {
      // Case 2: properties defined
      // Build property patterns
      std::vector<std::string> prop_patterns;
      for (size_t idx = 0; idx < spec.properties.size(); ++idx) {
        const auto& prop = spec.properties[idx];
        std::string value_rule =
            CreateRule(prop.schema, rule_name + "_prop_" + std::to_string(idx));
        std::string prop_pattern = "\"<parameter=" + prop.name + ">\" " + whitespace + " " +
                                   value_rule + " " + whitespace + " \"</parameter>\"";
        prop_patterns.push_back(prop_pattern);
      }

      // Build the grammar
      std::vector<std::string> rule_names(spec.properties.size(), "");
      std::vector<uint8_t> is_required(spec.properties.size(), false);
      bool allow_additional = additional_property != nullptr;

      // Construct the last rule
      std::string additional_prop_pattern;
      if (allow_additional) {
        std::string add_value_rule =
            CreateRule(additional_property, rule_name + "_" + additional_suffix);
        additional_prop_pattern = "\"<parameter=\" " + kXMLVariableName + " \">\" " + whitespace +
                                  " " + add_value_rule + " " + whitespace + " \"</parameter>\"";
        std::string last_rule_body = "(" + additional_prop_pattern + ")*";
        std::string last_rule_name =
            rule_name + "_part_" + std::to_string(static_cast<int>(spec.properties.size()) - 1);
        last_rule_name = ebnf_script_creator_.AddRule(last_rule_name, last_rule_body);
        rule_names.back() = last_rule_name;
      } else {
        rule_names.back() = "\"\"";
      }

      // Construct intermediate rules
      for (int i = static_cast<int>(spec.properties.size()) - 2; i >= 0; --i) {
        const std::string& prop_pattern = prop_patterns[i + 1];
        const std::string& last_rule_name = rule_names[i + 1];
        std::string cur_rule_body = prop_pattern + " " + last_rule_name;
        if (!spec.required.count(spec.properties[i + 1].name)) {
          cur_rule_body = last_rule_name + " | " + cur_rule_body;
        } else {
          is_required[i + 1] = true;
        }
        std::string cur_rule_name = rule_name + "_part_" + std::to_string(i);
        cur_rule_name = ebnf_script_creator_.AddRule(cur_rule_name, cur_rule_body);
        rule_names[i] = cur_rule_name;
      }
      if (spec.required.count(spec.properties[0].name)) {
        is_required[0] = true;
      }

      // Construct the result
      for (size_t i = 0; i < spec.properties.size(); ++i) {
        if (i != 0) {
          result += " | ";
        }
        result += "(" + prop_patterns[i] + " " + rule_names[i] + ")";
        if (is_required[i]) {
          break;
        }
      }

      if (allow_additional && spec.required.empty()) {
        result += " | " + additional_prop_pattern + " " + rule_names.back();
      }

      could_be_empty = spec.required.empty() && spec.min_properties == 0;
    } else if (additional_property) {
      // Case 3: no properties defined, additional properties allowed
      if (spec.max_properties != 0) {
        std::string add_value_rule =
            CreateRule(additional_property, rule_name + "_" + additional_suffix);
        std::string other_property_pattern = "\"<parameter=\" " + kXMLVariableName + " \">\" " +
                                             whitespace + " " + add_value_rule + " " + whitespace +
                                             " \"</parameter>\"";
        result += other_property_pattern + " " +
                  GetPropertyWithNumberConstraints(
                      other_property_pattern, spec.min_properties, spec.max_properties, 1
                  );
      }
      could_be_empty = spec.min_properties == 0;
    }

    if (could_be_empty) {
      if (result.empty()) {
        result = "\"\"";
      } else {
        result = "\"\" | " + result;
      }
    }

    is_root_object_ = true;  // Restore for potential future calls
    return result;
  }

  // For non-root objects, use JSON format
  return JSONSchemaConverter::GenerateObject(spec, rule_name);
}

}  // namespace xgrammar
