/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.h
 * \brief Extended format converters for JSON Schema, including XML Tool Calling format.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
#define XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
  using JSONSchemaConverter::FormatOtherProperty;
  using JSONSchemaConverter::FormatProperty;

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
  int32_t FormatPropertyKey(const std::string& key, const SchemaSpecPtr& schema) override;
  int32_t FormatProperty(
      const std::string& key,
      int32_t value_rule_id,
      const std::string& rule_name,
      int64_t idx,
      const SchemaSpecPtr& schema
  ) override;
  int32_t FormatOtherProperty(
      int32_t key_pattern_expr,
      int32_t value_rule_id,
      const std::string& rule_name,
      const std::string& rule_name_suffix,
      const SchemaSpecPtr& schema
  ) override;

  std::string GetKeyPattern() const override;
  std::string GetBasicAnyRuleName() const override;
  int32_t GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  ) override;

  std::string NextSeparator(bool is_end = false) override;

  void AddBasicRules() override;
  void AddBasicRules(const std::vector<std::string>& additional_rule_names);
  virtual void AddXMLBasicRulesLevel1();

  void AddCache(const std::string& key, int32_t rule_id) override;
  std::optional<int32_t> GetCache(const std::string& key) const override;

 protected:
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
  std::string EscapeAttrValue(const std::string& value) const;

  /*!
   * \brief Return the Kimi-K3 `type` attribute a value of \p spec is rendered with, or
   * std::nullopt if the schema does not pin down a single type (\p spec may be nullptr, which
   * is how free-form keys end up unconstrained).
   *
   * The Kimi-K3 tool-call parser reads the attribute as a decoding switch: type="string"
   * keeps the value as raw text, anything else JSON-decodes it. So the attribute must agree
   * with the value grammar, otherwise the decoded argument changes type (e.g. a string
   * property tagged type="number" with body 123 decodes to the integer 123). Mirrors the
   * model's renderer (_xtml_type), which maps both ints and floats to "number".
   */
  static std::optional<std::string> KimiK3TypeAttr(const SchemaSpecPtr& spec);

  /*!
   * \brief Build the expression between the property key and its value.
   * \param pinned_type For kimi_k3_xml, the single type attribute this property must carry.
   * std::nullopt keeps every type allowed, which is what free-form keys
   * (additionalProperties / patternProperties) need.
   */
  int32_t XMLKeySuffix(const std::optional<std::string>& pinned_type = std::nullopt);

  JSONFormat json_format_;
  // Track if we're at the root object level
  int nested_object_level_ = 0;
  const XMLWrapper xml_wrapper_;
};

/*! \brief Converter for DeepSeek XML Tool Calling format. */
class DeepSeekXMLToolCallingConverter : public XMLToolCallingConverter {
 public:
  DeepSeekXMLToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false
  );

 protected:
  using XMLToolCallingConverter::AddCache;
  using XMLToolCallingConverter::FormatOtherProperty;
  using XMLToolCallingConverter::FormatProperty;
  using XMLToolCallingConverter::GetCache;

  void AddBasicRules() override;
  void AddXMLBasicRulesLevel1() override;

  int32_t FormatProperty(
      const std::string& key,
      const SchemaSpecPtr& value_spec,
      const std::string& rule_name,
      int64_t idx
  ) override;
  int32_t FormatOtherProperty(
      int32_t key_pattern_expr,
      const SchemaSpecPtr& value_spec,
      const std::string& rule_name,
      const std::string& rule_name_suffix
  ) override;

 private:
  enum class GenerateMode {
    kString,
    kJSON,
  };

  static const std::string kXMLAnyJSON;
  static constexpr const char* kKeySuffixes[] = {
      "\" string=\"true\">",
      "\" string=\"false\">",
  };

  int32_t CreateRuleConstrained(
      const SchemaSpecPtr& spec, const std::string& rule_name_hint, GenerateMode mode
  );
  int32_t GenerateFromSpecConstrained(
      const SchemaSpecPtr& spec, const std::string& rule_name_hint, GenerateMode mode
  );

  int32_t GenerateAnyConstrained(
      const AnySpec& spec, const std::string& rule_name, GenerateMode mode
  );
  int32_t GenerateConstConstrained(
      const ConstSpec& spec, const std::string& rule_name, GenerateMode mode
  );
  int32_t GenerateEnumConstrained(
      const EnumSpec& spec, const std::string& rule_name, GenerateMode mode
  );
  int32_t GenerateRefConstrained(
      const RefSpec& spec, const std::string& rule_name, GenerateMode mode
  );
  int32_t GenerateAnyOfConstrained(
      const AnyOfSpec& spec, const std::string& rule_name, GenerateMode mode
  );
  int32_t GenerateOneOfConstrained(
      const OneOfSpec& spec, const std::string& rule_name, GenerateMode mode
  );
  int32_t GenerateAllOfConstrained(
      const AllOfSpec& spec, const std::string& rule_name, GenerateMode mode
  );
  int32_t GenerateTypeArrayConstrained(
      const TypeArraySpec& spec, const std::string& rule_name, GenerateMode mode
  );

  void AddCache(const std::string& key, int32_t rule_id, GenerateMode mode);
  std::optional<int32_t> GetCache(const std::string& key, GenerateMode mode) const;

  GenerateCacheManager constrained_rule_cache_manager_;
  std::unordered_map<std::pair<std::string, GenerateMode>, std::string>
      uri_to_constrained_rule_name_;
};

/*!
 * \brief Converter for Cohere XML Tool Calling format.
 *
 * This converter generates recursive Cohere value tags:
 * <cofl:value name="key" type="raw|json|dict|list">value</cofl:value>.
 * Object properties use named value tags. Array items use unnamed value tags.
 */
class CohereXMLToolCallingConverter : public XMLToolCallingConverter {
 public:
  CohereXMLToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false
  );

 protected:
  using XMLToolCallingConverter::FormatOtherProperty;
  using XMLToolCallingConverter::FormatProperty;

  int32_t GenerateString(const StringSpec& spec, const std::string& rule_name) override;
  int32_t GenerateObject(
      const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces = false
  ) override;
  int32_t GenerateAny(const AnySpec& spec, const std::string& rule_name) override;
  int32_t GenerateArray(const ArraySpec& spec, const std::string& rule_name) override;
  int32_t GenerateConst(const ConstSpec& spec, const std::string& rule_name) override;
  int32_t GenerateEnum(const EnumSpec& spec, const std::string& rule_name) override;

  int32_t FormatProperty(
      const std::string& key,
      int32_t value_rule_id,
      const std::string& rule_name,
      int64_t idx,
      const SchemaSpecPtr& schema
  ) override;
  int32_t FormatOtherProperty(
      int32_t key_pattern_expr,
      int32_t value_rule_id,
      const std::string& rule_name,
      const std::string& rule_name_suffix,
      const SchemaSpecPtr& schema
  ) override;

  std::string GetKeyPattern() const override;
  int32_t GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  ) override;
  std::string NextSeparator(bool is_end = false) override;

  void AddCache(const std::string& key, int32_t rule_id) override;
  std::optional<int32_t> GetCache(const std::string& key) const override;

 private:
  struct XMLIdentifierTrieNode {
    bool is_terminal = false;
    std::map<char, XMLIdentifierTrieNode> children;
  };

  int32_t FormatCohereParam(
      const std::optional<std::string>& name,
      const std::optional<int32_t>& key_pattern_expr,
      const SchemaSpecPtr& schema,
      int32_t value_rule_id
  );
  int32_t FormatSingleCohereParam(
      const std::optional<std::string>& name,
      const std::optional<int32_t>& key_pattern_expr,
      const SchemaSpecPtr& schema,
      int32_t value_rule_id
  );
  int32_t FormatCohereValue(int32_t value_rule_id);
  int32_t GetCohereTypePattern(const SchemaSpecPtr& schema);
  static std::string CohereTypeForJSONLiteral(const std::string& json_value);
  static std::optional<std::string> CommonCohereTypeForJSONLiterals(
      const std::vector<std::string>& json_values
  );
  std::optional<std::vector<SchemaSpecPtr>> GetCohereCompositeOptions(const SchemaSpecPtr& schema
  ) const;
  int32_t BuildXMLIdentifierExcludingBody(
      const XMLIdentifierTrieNode& node, const std::string& rule_name, int depth
  );
  bool InCohereValueContext() const;

  std::vector<const ObjectSpec*> object_stack_;
  std::vector<SchemaSpecPtr> additional_property_stack_;
  int cohere_array_level_ = 0;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
