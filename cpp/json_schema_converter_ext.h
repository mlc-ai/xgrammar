/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.h
 * \brief Extended tool-calling format converters for JSON Schema (XML styles, Gemma).
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
      RefResolver ref_resolver = nullptr,
      JSONFormat json_format = JSONFormat::kQwenXML,
      bool any_order = false
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

  std::string GetKeyPattern() const override;
  std::string GetBasicAnyRuleName() const override;
  std::string GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  ) override;

  std::string NextSeparator(bool is_end = false) override;

  void AddBasicRules() override;

  void AddCache(const std::string& key, const std::string& value) override;
  std::optional<std::string> GetCache(const std::string& key) const override;

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

  // Track if we're at the root object level
  int nested_object_level_ = 0;
  const XMLWrapper xml_wrapper_;
};

/*!
 * \brief Converter for Gemma Tool Calling format.
 *
 * This converter generates EBNF where the whole value space (all nesting levels)
 * uses Gemma's custom format instead of JSON:
 * - Object keys are unquoted: {key:value,key2:value2}
 * - Strings are delimited by <|"|> instead of ", with no escape sequences:
 *   <|"|>any text<|"|>
 * - Numbers, booleans, arrays and objects otherwise follow JSON syntax.
 */
class GemmaToolCallingConverter : public JSONSchemaConverter {
 public:
  GemmaToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false
  );

 protected:
  std::string GenerateString(const StringSpec& spec, const std::string& rule_name) override;
  std::string GenerateConst(const ConstSpec& spec, const std::string& rule_name) override;
  std::string GenerateEnum(const EnumSpec& spec, const std::string& rule_name) override;

  std::string FormatPropertyKey(const std::string& key) override;
  std::string GetKeyPattern() const override;
  std::string GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  ) override;
  std::string FormatPatternKey(const std::string& pattern) override;
  std::string CreatePropertyNamesKeyRule(
      const SchemaSpecPtr& property_names, const std::string& rule_name_hint
  ) override;

  void AddBasicRules() override;

 private:
  // The Gemma string delimiter that replaces the JSON double quote.
  static const std::string kGemmaStringDelim;
  // Rule matching any text that does not contain the string delimiter.
  static const std::string kGemmaStringContent;
  // Rule matching an unquoted property key.
  static const std::string kGemmaVariableName;
  // Rule matching exactly one character that cannot start the string delimiter.
  // Used for length-constrained strings where repetition counts must stay exact.
  static const std::string kGemmaBoundedChar;

  /*! \brief Serialize a JSON value into Gemma's tool call argument format. */
  static std::string SerializeJSON(const picojson::value& value);
  /*! \brief Parse a JSON-serialized value and return its Gemma serialization as an EBNF literal. */
  static std::string ValueToEBNFLiteral(const std::string& json_value);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
