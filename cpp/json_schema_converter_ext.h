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
 * The concrete XML dialect controls whether only the outermost object is XML-encoded or objects
 * and arrays are encoded recursively.
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

  /*! \brief Convert SchemaSpec to EBNF with XML format for root object. Note that this function is
   * not thread-safe.*/
  std::string Convert(const SchemaSpecPtr& spec);

  /*! \brief Whether the generated grammar contains runtime-generated recursive XML tag names. */
  bool RequiresDynamicTagMatcher() const { return requires_dynamic_tag_matcher_; }

 protected:
  // Override methods for XML format
  std::string GenerateString(const StringSpec& spec, const std::string& rule_name) override;
  std::string GenerateObject(
      const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces = false
  ) override;
  std::string GenerateAny(const AnySpec& spec, const std::string& rule_name) override;
  std::string GenerateArray(const ArraySpec& spec, const std::string& rule_name) override;
  std::string GenerateConst(const ConstSpec& spec, const std::string& rule_name) override;
  std::string GenerateEnum(const EnumSpec& spec, const std::string& rule_name) override;

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
  std::string FormatPatternProperty(
      const std::string& key_regex,
      const std::string& value_rule,
      const std::string& rule_name,
      const std::string& rule_name_suffix
  ) override;
  std::string CreatePropertyNameRule(const SchemaSpecPtr& spec, const std::string& rule_name_hint)
      override;

  std::string GetKeyPattern() const override;
  std::string GetBasicAnyRuleName() const override;
  std::string GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  ) override;

  std::string NextSeparator(bool is_end = false) override;

  void AddBasicRules() override;

  void AddCache(const std::string& key, const std::string& value) override;
  std::optional<std::string> GetCache(const std::string& key) const override;
  int GetRefCacheDomain() const override;

 private:
  struct ElementSyntax {
    std::string open_prefix;
    std::string open_suffix;
    std::string value_prefix;
    std::string close_prefix;
    std::string close_suffix;
    bool close_repeats_key = false;
  };

  struct XMLDialectConfig {
    ElementSyntax property;
    bool recursive = false;
    std::string array_item_name;
    bool pad_values_with_whitespace = true;
    std::string string_terminator;
    std::string variable_name_pattern;
  };

  static const std::unordered_map<JSONFormat, XMLDialectConfig> kDialectConfigMap;
  static const std::string kXMLString;
  static const std::string kXMLAny;
  static const std::string kXMLObject;
  static const std::string kXMLVariableName;

  bool IsXMLLayer() const;
  bool IsInnerCacheLayer() const;
  std::string FormatElement(
      const ElementSyntax& syntax, const std::string& key, const std::string& value_rule
  ) const;
  std::string GenerateRepeatedElementArray(const ArraySpec& spec, const std::string& rule_name);
  std::string GenerateLiteral(const picojson::value& value) const;
  void AddRootOnlyXMLBasicRules();
  void AddRecursiveXMLBasicRules();
  void ValidateRecursiveObject(const ObjectSpec& spec) const;
  void ValidateElementName(const std::string& name) const;

  int nested_object_level_ = 0;
  bool generating_property_name_ = false;
  mutable bool requires_dynamic_tag_matcher_ = false;
  // The dialect table is immutable and has static lifetime. Keep a reference so each conversion
  // does not copy its protocol strings.
  const XMLDialectConfig& dialect_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
