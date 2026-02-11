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
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr
  );

  /*! \brief Convert SchemaSpec to EBNF with XML format for root object. Note that this function is
   * not thread-safe.*/
  std::string Convert(const SchemaSpecPtr& spec);

 protected:
  // Override methods for XML format
  std::string GenerateString(const StringSpec& spec, const std::string& rule_name) override;
  std::string GenerateObject(
      const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces = false
  ) override;
  std::string GenerateAny(const AnySpec& spec, const std::string& rule_name) override;
  std::string GenerateArray(const ArraySpec& spec, const std::string& rule_name) override;

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

  std::string GetKeyPattern() const override;
  std::string GetBasicAnyRuleName() const override;

  std::string NextSeparator(bool is_end = false) override;

  void AddBasicRules() override;

  void AddCache(const std::string& key, const std::string& value) override;
  std::optional<std::string> GetCache(const std::string& key) const override;

 private:
  // XML-specific rule names
  static const std::string kXMLString;
  static const std::string kXMLAny;
  static const std::string kXMLObject;
  static const std::string kXMLVariableName;

  // Track if we're at the root object level
  int nested_object_level_ = 0;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
