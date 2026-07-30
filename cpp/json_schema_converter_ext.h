/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.h
 * \brief Extended format converters for JSON Schema, including XML Tool Calling format.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
#define XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_

#include <unordered_map>
#include <utility>

#include "json_schema_converter.h"

namespace xgrammar {

/*!
 * \brief Converter for XML Tool Calling format (e.g., Qwen style).
 *
 * This converter generates a grammar where:
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
      RefResolver ref_resolver = nullptr,
      JSONFormat json_format = JSONFormat::kQwenXML,
      bool any_order = false
  );

  /*! \brief Convert SchemaSpec to grammar with XML format for root object. Note that this function
   * is not thread-safe.*/
  Grammar Convert(const SchemaSpecPtr& spec);

 protected:
  // Override methods for XML format
  int32_t GenerateString(const StringSpec& spec, const std::string& rule_name) override;
  int32_t GenerateObject(
      const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces = false
  ) override;
  int32_t GenerateAny(const AnySpec& spec, const std::string& rule_name) override;
  int32_t GenerateArray(const ArraySpec& spec, const std::string& rule_name) override;
  int32_t GenerateConst(const ConstSpec& spec, const std::string& rule_name) override;
  int32_t GenerateEnum(const EnumSpec& spec, const std::string& rule_name) override;

  // Override format hooks
  int32_t FormatPropertyKey(const std::string& key) override;
  int32_t FormatProperty(
      const std::string& key, int32_t value_rule_id, const std::string& rule_name, int64_t idx
  ) override;
  int32_t FormatOtherProperty(
      int32_t key_pattern_expr,
      int32_t value_rule_id,
      const std::string& rule_name,
      const std::string& rule_name_suffix
  ) override;

  std::string GetKeyPattern() const override;
  std::string GetBasicAnyRuleName() const override;
  int32_t GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  ) override;

  std::string NextSeparator(bool is_end = false) override;

  void AddBasicRules() override;

  void AddCache(const std::string& key, int32_t rule_id) override;
  std::optional<int32_t> GetCache(const std::string& key) const override;

 private:
  // Wrapper strings for XML parameter tags (key prefix/suffix, value prefix, closing suffix)
  struct XMLWrapper {
    std::string key_wrapper_prefix;
    std::string key_wrapper_suffix;
    std::string value_wrapper_prefix;
    std::string parameter_suffix;
  };

  static const std::unordered_map<JSONFormat, XMLWrapper> kKeyWrapperMap;
  static const std::string kXMLString;
  static const std::string kXMLAny;
  static const std::string kXMLObject;
  static const std::string kXMLVariableName;

  std::string XMLValue(const std::string& json_value) const;
  int32_t XMLKeySuffix();

  JSONFormat json_format_;
  // Track if we're at the root object level
  int nested_object_level_ = 0;
  const XMLWrapper xml_wrapper_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
