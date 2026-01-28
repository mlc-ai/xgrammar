/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.h
 * \brief Extended format converters for JSON Schema, including XML Tool Calling format.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
#define XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_

#include "json_schema_converter.h"

namespace xgrammar {

/*!
 * \brief Converter for XML Tool Calling format (e.g., Qwen style).
 *
 * This converter generates EBNF where:
 * - The outermost object uses XML format: <parameter=name>value</parameter>
 * - Inner values use standard JSON format
 */
class XMLToolCallingConverter : public JSONSchemaConverter {
 public:
  XMLToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt
  );

  /*! \brief Convert SchemaSpec to EBNF with XML format for root object. */
  std::string Convert(const SchemaSpecPtr& spec);

 protected:
  // Override methods for XML format
  std::string GenerateString(const StringSpec& spec, const std::string& rule_name) override;
  std::string GenerateObject(const ObjectSpec& spec, const std::string& rule_name) override;
  std::string GenerateAny(const AnySpec& spec, const std::string& rule_name) override;

  // Override format hooks
  std::string FormatPropertyKey(const std::string& key) override;
  std::string FormatProperty(
      const std::string& key,
      const std::string& value_rule,
      const std::string& rule_name,
      int64_t idx
  ) override;
  std::string FormatOtherProperty(
      const std::string& key_pattern,
      const std::string& value_rule,
      const std::string& rule_name,
      const std::string& rule_name_suffix
  ) override;

  std::string GetBasicStringRuleName() const override;
  std::string GetBasicAnyRuleName() const override;

  void AddBasicRules() override;

 private:
  void AddXMLHelperRules();

  // XML-specific rule names
  static const std::string kXMLString;
  static const std::string kXMLAny;
  static const std::string kXMLVariableName;

  // Track if we're at the root object level
  bool is_root_object_ = true;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
