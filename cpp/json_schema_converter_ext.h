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
#include <utility>
#include <vector>

#include "json_schema_converter.h"

namespace xgrammar {

// To add a format, declare its converter here and implement it in converter_ext/.
// Configure XMLWrapper and override XMLKeySuffix/EscapeAttrValue as needed, or override
// generation methods for recursive formats. Register it in JSONFormat, JSONFormatFromString,
// and ConvertSchemaSpecToGrammar. For structural tags, also update C++/Python style validation.

/*! \brief Converter for Qwen XML Tool Calling format. */
class QwenXMLToolCallingConverter : public XMLToolCallingConverter {
 public:
  QwenXMLToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false
  );
};

/*! \brief Converter for MiniMax XML Tool Calling format. */
class MiniMaxXMLToolCallingConverter : public XMLToolCallingConverter {
 public:
  MiniMaxXMLToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false
  );
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
  int32_t XMLKeySuffix(const SchemaSpecPtr& schema) override;
};

/*! \brief Converter for Glm XML Tool Calling format. */
class GlmXMLToolCallingConverter : public XMLToolCallingConverter {
 public:
  GlmXMLToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false
  );
};

/*! \brief Converter for KimiK3 XML Tool Calling format. */
class KimiK3XMLToolCallingConverter : public XMLToolCallingConverter {
 public:
  KimiK3XMLToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false
  );

 protected:
  int32_t XMLKeySuffix(const SchemaSpecPtr& schema) override;
  std::string EscapeAttrValue(const std::string& value) const override;

 private:
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

  std::string GetKeyPattern() const override;
  int32_t GetKeyPatternExcluding(
      const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
  ) override;
  std::string NextSeparator(bool is_end = false) override;

  void AddBasicRules() override;
  void AddCache(const std::string& key, int32_t rule_id) override;
  std::optional<int32_t> GetCache(const std::string& key) const override;

 private:
  struct CohereKeyTrieNode {
    bool is_terminal = false;
    std::map<int32_t, CohereKeyTrieNode> children;
  };

  static const std::string kCohereKey;
  static const std::string kCohereAnyScalar;
  static const std::string kCohereAnyList;

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
  int32_t FormatCohereParamWithType(
      const std::optional<std::string>& name,
      const std::optional<int32_t>& key_pattern_expr,
      int32_t type_expression,
      int32_t value_rule_id
  );
  int32_t FormatAnyCohereParam(
      const std::optional<std::string>& name, const std::optional<int32_t>& key_pattern_expr
  );
  int32_t FormatCohereValue(int32_t value_rule_id);
  int32_t GetCohereTypePattern(const SchemaSpecPtr& schema);
  static std::string CohereTypeForJSONLiteral(const std::string& json_value);
  static std::optional<std::string> CommonCohereTypeForJSONLiterals(
      const std::vector<std::string>& json_values
  );
  std::optional<std::vector<SchemaSpecPtr>> GetCohereCompositeOptions(const SchemaSpecPtr& schema
  ) const;
  int32_t BuildCohereKeyExcludingBody(const CohereKeyTrieNode& node, int depth);
  bool AtCohereRoot() const;
  bool InCohereValueContext() const;

  std::vector<const ObjectSpec*> object_stack_;
  std::vector<SchemaSpecPtr> additional_property_stack_;
  int cohere_array_level_ = 0;
};

/*!
 * \brief Converter for MiniMax M3's recursive namespace-prefixed XML format.
 *
 * This initial implementation supports schemas whose object property names are
 * known when the grammar is built. Schemas requiring runtime element names are
 * rejected explicitly.
 */
class MiniMaxM3XMLToolCallingConverter : public JSONSchemaConverter {
 public:
  MiniMaxM3XMLToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false
  );

 protected:
  int32_t GenerateString(const StringSpec& spec, const std::string& rule_name) override;
  int32_t GenerateArray(const ArraySpec& spec, const std::string& rule_name) override;
  int32_t GenerateObject(
      const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces = false
  ) override;
  int32_t GenerateAny(const AnySpec& spec, const std::string& rule_name) override;
  int32_t GenerateConst(const ConstSpec& spec, const std::string& rule_name) override;
  int32_t GenerateEnum(const EnumSpec& spec, const std::string& rule_name) override;

  int32_t FormatProperty(
      const std::string& key,
      int32_t value_rule_id,
      const std::string& rule_name,
      int64_t idx,
      const SchemaSpecPtr& schema
  ) override;
  std::string NextSeparator(bool is_end = false) override;
  void AddBasicRules() override;

 private:
  int32_t FormatElement(const std::string& name, int32_t value_rule_id);
  int32_t GenerateLiteral(const picojson::value& value);
  void ValidateObject(const ObjectSpec& spec) const;
  static void ValidateElementName(const std::string& name);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
