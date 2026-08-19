/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.h
 * \brief Extended format converters for JSON Schema, including XML Tool Calling format.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
#define XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_

#include <picojson.h>

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "json_schema_converter.h"

namespace xgrammar {

/*!
 * \brief Converter for XML Tool Calling formats.
 *
 * The concrete dialect controls whether only the outermost object is XML-encoded or objects and
 * arrays are encoded recursively.
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

  /*! \brief Convert SchemaSpec to a grammar. This function is not thread-safe. */
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
  int32_t FormatPatternProperty(
      const std::string& key_regex,
      int32_t value_rule_id,
      const std::string& rule_name,
      const std::string& rule_name_suffix,
      const SchemaSpecPtr& schema
  ) override;
  int32_t CreatePropertyNameRule(const SchemaSpecPtr& spec, const std::string& rule_name_hint)
      override;

  std::string GetKeyPattern() const override;
  std::string GetBasicAnyRuleName() const override;
  int32_t GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  ) override;

  std::string NextSeparator(bool is_end = false) override;

  void AddBasicRules() override;

  void AddCache(const std::string& key, int32_t rule_id) override;
  std::optional<int32_t> GetCache(const std::string& key) const override;
  int GetRefCacheDomain() const override;

 protected:
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
  };

  static const std::unordered_map<JSONFormat, XMLDialectConfig> kDialectConfigMap;
  static const std::string kXMLString;
  static const std::string kXMLAny;
  static const std::string kXMLObject;
  static const std::string kXMLVariableName;

  bool IsXMLLayer() const;
  bool IsInnerCacheLayer() const;
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
  int32_t FormatElement(
      const ElementSyntax& syntax,
      const std::string& key,
      int32_t value_rule_id,
      const SchemaSpecPtr& schema = nullptr
  );
  int32_t FormatElementValueAndClose(
      const ElementSyntax& syntax, int32_t value_rule_id, int32_t close_expr
  );
  int32_t GenerateRepeatedElementArray(const ArraySpec& spec, const std::string& rule_name);
  int32_t GenerateLiteral(const picojson::value& value);
  void AddRootOnlyXMLBasicRules();
  void AddRecursiveXMLBasicRules();
  void ValidateRecursiveObject(const ObjectSpec& spec) const;
  void ValidateElementName(const std::string& name) const;

  struct UniqueKeyScopeContext {
    int32_t rule_id = -1;
    std::vector<std::string> reserved_names;
  };

  JSONFormat json_format_;
  // Track if we're at the root object level
  int nested_object_level_ = 0;
  bool generating_property_name_ = false;
  const XMLDialectConfig& dialect_;
  std::vector<UniqueKeyScopeContext> unique_key_scope_stack_;
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
  int32_t FormatPatternProperty(
      const std::string& key_regex,
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
