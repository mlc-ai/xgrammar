/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter.h
 * \brief Convert a JSON Schema directly to a grammar AST.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_CONVERTER_H_
#define XGRAMMAR_JSON_SCHEMA_CONVERTER_H_

#include <picojson.h>
#include <xgrammar/grammar.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace xgrammar {

// ==================== SchemaSpec: Intermediate Representation for JSON Schema ====================

// Forward declaration
struct SchemaSpec;
using SchemaSpecPtr = std::shared_ptr<SchemaSpec>;

// Basic Type Specs
struct IntegerSpec {
  std::optional<int64_t> minimum;
  std::optional<int64_t> maximum;
  std::optional<int64_t> exclusive_minimum;
  std::optional<int64_t> exclusive_maximum;
  std::optional<int64_t> multiple_of;

  std::string ToString() const;
};

struct NumberSpec {
  std::optional<double> minimum;
  std::optional<double> maximum;
  std::optional<double> exclusive_minimum;
  std::optional<double> exclusive_maximum;

  std::string ToString() const;
};

struct StringSpec {
  std::optional<std::string> pattern;
  std::optional<std::string> format;
  int min_length = 0;
  int max_length = -1;  // -1 means no limit

  std::string ToString() const;
};

struct BooleanSpec {
  std::string ToString() const;
};

struct NullSpec {
  std::string ToString() const;
};

struct AnySpec {
  std::string ToString() const;
};

// Complex Type Specs
struct ArraySpec {
  std::vector<SchemaSpecPtr> prefix_items;
  bool allow_additional_items = true;
  SchemaSpecPtr additional_items;  // nullptr means not allowed
  int64_t min_items = 0;
  int64_t max_items = -1;  // -1 means no limit

  std::string ToString() const;
};

struct ObjectSpec {
  struct Property {
    std::string name;
    SchemaSpecPtr schema;
  };

  struct PatternProperty {
    std::string pattern;  // regex pattern for key
    SchemaSpecPtr schema;
  };

  std::vector<Property> properties;
  std::vector<PatternProperty> pattern_properties;
  std::unordered_set<std::string> required;

  bool allow_additional_properties = false;
  SchemaSpecPtr additional_properties_schema;
  bool allow_unevaluated_properties = true;
  SchemaSpecPtr unevaluated_properties_schema;
  SchemaSpecPtr property_names;

  int min_properties = 0;
  int max_properties = -1;  // -1 means no limit

  std::string ToString() const;
};

// Composite Type Specs
struct ConstSpec {
  std::string json_value;  // JSON serialized value

  std::string ToString() const;
};

struct EnumSpec {
  std::vector<std::string> json_values;  // JSON serialized values

  std::string ToString() const;
};

struct RefSpec {
  std::string uri;

  std::string ToString() const;
};

struct AnyOfSpec {
  std::vector<SchemaSpecPtr> options;

  std::string ToString() const;
};

struct OneOfSpec {
  std::vector<SchemaSpecPtr> options;

  std::string ToString() const;
};

struct AllOfSpec {
  std::vector<SchemaSpecPtr> schemas;

  std::string ToString() const;
};

struct TypeArraySpec {
  // Handle "type": ["string", "integer"] cases
  std::vector<SchemaSpecPtr> type_schemas;

  std::string ToString() const;
};

// Unified SchemaSpec
using SchemaSpecVariant = std::variant<
    IntegerSpec,
    NumberSpec,
    StringSpec,
    BooleanSpec,
    NullSpec,
    ArraySpec,
    ObjectSpec,
    AnySpec,
    ConstSpec,
    EnumSpec,
    RefSpec,
    AnyOfSpec,
    OneOfSpec,
    AllOfSpec,
    TypeArraySpec>;

struct SchemaSpec {
  SchemaSpecVariant spec;
  std::string cache_key;       // for deduplication
  std::string rule_name_hint;  // suggested rule name

  std::string ToString() const;

  // Helper method to create SchemaSpec
  template <typename T>
  static SchemaSpecPtr Make(T&& spec_value, std::string cache_key = "", std::string hint = "") {
    auto ptr = std::make_shared<SchemaSpec>();
    ptr->spec = std::forward<T>(spec_value);
    ptr->cache_key = std::move(cache_key);
    ptr->rule_name_hint = std::move(hint);
    return ptr;
  }
};

// ==================== JSONFormat Enum ====================

enum class JSONFormat : int {
  kJSON = 0,
  kQwenXML = 1,
  kMiniMaxXML = 2,
  kDeepSeekXML = 3,
  kGlmXML = 4,
};

/*!
 * \brief Convert a format name to JSONFormat.
 * \param format One of "json", "qwen_xml", "minimax_xml", "deepseek_xml", "glm_xml".
 * \return The corresponding JSONFormat, or std::nullopt if the name is not recognized.
 */
std::optional<JSONFormat> JSONFormatFromString(const std::string& format);

/*!
 * \brief Manage indentation and separators as EBNF expression strings.
 *
 * Formatting remains string-based here; JSONSchemaConverter converts the returned expressions to
 * grammar AST nodes.
 */
class IndentManager {
 public:
  IndentManager(
      std::optional<int> indent,
      const std::string& separator,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt
  );

  void StartIndent();
  void EndIndent();
  std::string StartSeparator();
  std::string MiddleSeparator();
  std::string EndSeparator();
  std::string EmptySeparator();
  std::string NextSeparator(bool is_end = false);

 private:
  bool any_whitespace_;
  bool enable_newline_;
  int64_t indent_;
  std::string separator_;
  int64_t total_indent_;
  std::vector<bool> is_first_;
  std::optional<int> max_whitespace_cnt_;
};

/*!
 * \brief Convert SchemaSpec directly to a grammar AST.
 */
class JSONSchemaConverter {
 public:
  using RefResolver =
      std::function<SchemaSpecPtr(const std::string& uri, const std::string& rule_name_hint)>;

  JSONSchemaConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false,
      JSONFormat json_format = JSONFormat::kJSON
  );

  virtual ~JSONSchemaConverter() = default;

  /*!
   * \brief Convert SchemaSpec directly to a grammar AST.
   * \param spec The SchemaSpec to convert.
   * \return The grammar AST.
   */
  Grammar Convert(const SchemaSpecPtr& spec);

  // Basic rule names.
  static const std::string kBasicAny;
  static const std::string kBasicInteger;
  static const std::string kBasicNumber;
  static const std::string kBasicString;
  static const std::string kBasicBoolean;
  static const std::string kBasicNull;
  static const std::string kBasicArray;
  static const std::string kBasicObject;
  static const std::string kBasicEscape;
  static const std::string kBasicStringSub;

  /*! \brief Return the built-in regular expression for a JSON Schema string format. */
  static std::optional<std::string> JSONFormatToRegexPattern(const std::string& format);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

/*! \brief Convert a JSON Schema string directly to a grammar AST. */
Grammar JSONSchemaToGrammar(
    const std::string& schema,
    bool any_whitespace = true,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true,
    std::optional<int> max_whitespace_cnt = std::nullopt,
    bool any_order = false,
    JSONFormat json_format = JSONFormat::kJSON
);

// ==================== Public API functions (backward compatible) ====================

/*!
 * \brief Convert JSON schema string to EBNF grammar string.
 * \param schema The JSON schema string.
 * \param any_whitespace Whether to ignore the indentation restrictions, and allow any whitespace.
 * Default: true.
 * \param indent The number of spaces for indentation. If set to std::nullopt, the output will be
 * in one line. Default: 2.
 * \param separators Two separators used in the schema: comma and colon. Examples: {",", ":"},
 * {", ", ": "}. If std::nullopt, the default separators will be used: {",", ": "} when the
 * indent is not -1, and {", ", ": "} otherwise. This follows the convention in python
 * json.dumps(). Default: std::nullopt.
 * \param strict_mode Whether to use strict mode. In strict
 * mode, the generated grammar will not allow properties and items that is not specified in the
 * schema. This is equivalent to setting unevaluatedProperties and unevaluatedItems to false.
 * This helps LLM to generate accurate output in the grammar-guided generation with JSON
 * schema. Default: true.
 * \param max_whitespace_cnt The maximum number of whitespace characters for the whitespace
 * which is used for indentation or JSON elements separation when any_whitespace is True. If
 * std::nullopt, it means unlimited. Default: std::nullopt.
 * \param json_format Define the root format of the object. JSONFormat::kJSON generates a fully
 * JSON-style grammar. The Qwen, MiniMax, DeepSeek, and GLM variants generate an XML-style root
 * whose inner values use JSON syntax. Default: JSONFormat::kJSON.
 * \returns The EBNF grammar string.
 */

std::string JSONSchemaToEBNF(
    const std::string& schema,
    bool any_whitespace = true,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true,
    std::optional<int> max_whitespace_cnt = std::nullopt,
    JSONFormat json_format = JSONFormat::kJSON,
    bool any_order = false
);

/*!
 * \brief Convert JSON schema string to EBNF grammar string.
 * \param schema The JSON schema object.
 * \param any_whitespace Whether to ignore the indentation restrictions, and allow any whitespace.
 * Default: true.
 * \param indent The number of spaces for indentation. If set to std::nullopt, the output will be
 * in one line. Default: 2.
 * \param separators Two separators used in the schema: comma and colon. Examples: {",", ":"},
 * {", ", ": "}. If std::nullopt, the default separators will be used: {",", ": "} when the
 * indent is not -1, and {", ", ": "} otherwise. This follows the convention in python
 * json.dumps(). Default: std::nullopt.
 * \param strict_mode Whether to use strict mode. In strict
 * mode, the generated grammar will not allow properties and items that is not specified in the
 * schema. This is equivalent to setting unevaluatedProperties and unevaluatedItems to false.
 * This helps LLM to generate accurate output in the grammar-guided generation with JSON
 * schema. Default: true.
 * \param max_whitespace_cnt The maximum number of whitespace characters for the whitespace
 * which is used for indentation or JSON elements separation when any_whitespace is True. If
 * std::nullopt, it means unlimited. Default: std::nullopt.
 * \param json_format Define the root format of the object. JSONFormat::kJSON generates a fully
 * JSON-style grammar. The Qwen, MiniMax, DeepSeek, and GLM variants generate an XML-style root
 * whose inner values use JSON syntax. Default: JSONFormat::kJSON.
 * \returns The EBNF grammar string.
 */
std::string JSONSchemaToEBNF(
    const picojson::value& schema,
    bool any_whitespace = true,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true,
    std::optional<int> max_whitespace_cnt = std::nullopt,
    JSONFormat json_format = JSONFormat::kJSON,
    bool any_order = false
);

/*!
 * \brief Generate regex pattern for integer/float range.
 * \param start The start of the range (inclusive). If null assume negative infinity.
 * \param end The end of the range (inclusive). If null assume infinity.
 * \returns The regex pattern that matches integers/floats in the given range.
 */
std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end);

std::string GenerateFloatRangeRegex(
    std::optional<double> start,
    std::optional<double> end,
    bool exclusive_start = false,
    bool exclusive_end = false
);

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_H_
