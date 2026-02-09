/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter.h
 * \brief Convert a JSON schema string to EBNF grammar string.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_CONVERTER_H_
#define XGRAMMAR_JSON_SCHEMA_CONVERTER_H_

#include <picojson.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "ebnf_script_creator.h"

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
  kXML = 1,
};

/*!
 * \brief Manage the rule generation cache. Wraps key-value cache for schema deduplication.
 */
class GenerateCacheManager {
 public:
  /*! \brief Add a key-value pair to the cache. */
  void AddCache(const std::string& key, bool is_inner_layer, const std::string& value) {
    cache_[{key, is_inner_layer}] = value;
  }

  /*! \brief Get cached value by key. Returns std::nullopt if not found. */
  std::optional<std::string> GetCache(const std::string& key, bool is_inner_layer) const {
    auto it = cache_.find({key, is_inner_layer});
    if (it != cache_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

 private:
  std::unordered_map<std::pair<std::string, bool>, std::string> cache_;
};

/*!
 * \brief Manage the indent and separator for the generation of EBNF grammar.
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

  friend class JSONSchemaConverter;
};

/*!
 * \brief Convert SchemaSpec to EBNF grammar string.
 *
 * This is the base class for EBNF generation. It generates JSON-format EBNF by default.
 * Subclasses can override virtual methods to generate different formats (e.g., XML).
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
      RefResolver ref_resolver = nullptr
  );

  virtual ~JSONSchemaConverter() = default;

  /*!
   * \brief Convert SchemaSpec to EBNF grammar string.
   * \param spec The SchemaSpec to convert.
   * \return The EBNF grammar string.
   */
  std::string Convert(const SchemaSpecPtr& spec);

 protected:
  // ==================== Virtual methods for generation ====================
  // Subclasses can override these to customize output format

  virtual std::string GenerateInteger(const IntegerSpec& spec, const std::string& rule_name);
  virtual std::string GenerateNumber(const NumberSpec& spec, const std::string& rule_name);
  virtual std::string GenerateString(const StringSpec& spec, const std::string& rule_name);
  virtual std::string GenerateBoolean(const BooleanSpec& spec, const std::string& rule_name);
  virtual std::string GenerateNull(const NullSpec& spec, const std::string& rule_name);
  virtual std::string GenerateArray(const ArraySpec& spec, const std::string& rule_name);
  virtual std::string GenerateObject(
      const ObjectSpec& spec, const std::string& rule_name, bool need_brace = true
  );
  virtual std::string GenerateAny(const AnySpec& spec, const std::string& rule_name);
  virtual std::string GenerateConst(const ConstSpec& spec, const std::string& rule_name);
  virtual std::string GenerateEnum(const EnumSpec& spec, const std::string& rule_name);
  virtual std::string GenerateRef(const RefSpec& spec, const std::string& rule_name);
  virtual std::string GenerateAnyOf(const AnyOfSpec& spec, const std::string& rule_name);
  virtual std::string GenerateAllOf(const AllOfSpec& spec, const std::string& rule_name);
  virtual std::string GenerateTypeArray(const TypeArraySpec& spec, const std::string& rule_name);

  // ==================== Hooks for customization ====================

  /*! \brief Format a property key. Override for different formats. */
  virtual std::string FormatPropertyKey(const std::string& key);

  /*! \brief Format a property (key + value). Override for different formats. */
  virtual std::string FormatProperty(
      const std::string& key,
      const std::string& value_rule,
      const std::string& rule_name,
      int64_t idx
  );

  /*! \brief Format an "other" property (additional/unevaluated). Override for different formats. */
  virtual std::string FormatOtherProperty(
      const std::string& key_pattern,
      const std::string& value_rule,
      const std::string& rule_name,
      const std::string& rule_name_suffix
  );

  /*! \brief Get the basic string rule name. Override for different formats. */
  virtual std::string GetKeyPattern() const;

  /*! \brief Get the basic any rule name. Override for different formats. */
  virtual std::string GetBasicAnyRuleName() const;

  /*! \brief Add basic rules for the format. Override for different formats. */
  virtual void AddBasicRules();

  /*! \brief Add a key-value pair to the generation cache. Override for custom cache behavior. */
  virtual void AddCache(const std::string& key, const std::string& value);

  /*! \brief Get cached value by key. Returns std::nullopt if not found. */
  virtual std::optional<std::string> GetCache(const std::string& key) const;

  // ==================== Helper methods (for subclasses to use) ====================

  /*! \brief Dispatch to the appropriate Generate method based on spec type. */
  std::string GenerateFromSpec(const SchemaSpecPtr& spec, const std::string& rule_name_hint);

  /*! \brief Create a rule and return the rule name (handles caching). */
  std::string CreateRule(const SchemaSpecPtr& spec, const std::string& rule_name_hint);

  /*! \brief Get next separator from indent manager. */
  virtual std::string NextSeparator(bool is_end = false);

  /*! \brief Get whitespace pattern. */
  std::string GetWhitespacePattern() const;

  /*! \brief Helper to create rule with repetition constraints. */
  std::string GetPropertyWithNumberConstraints(
      const std::string& pattern,
      int min_properties,
      int max_properties,
      int already_repeated_times = 0
  );

  /*! \brief Generate partial rule for object properties. */
  std::string GetPartialRuleForProperties(
      const std::vector<ObjectSpec::Property>& properties,
      const std::unordered_set<std::string>& required,
      const SchemaSpecPtr& additional,
      const std::string& rule_name,
      const std::string& additional_suffix,
      int min_properties,
      int max_properties
  );

  // ==================== Protected members ====================

  EBNFScriptCreator ebnf_script_creator_;
  IndentManager indent_manager_;
  std::string colon_pattern_;
  bool any_whitespace_;
  std::optional<int> max_whitespace_cnt_;

  // Basic rule names
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

 protected:
  GenerateCacheManager rule_cache_manager_;

 private:
  void AddHelperRules();

  std::unordered_map<std::string, std::string>
      uri_to_rule_name_;      // For circular reference handling
  RefResolver ref_resolver_;  // Resolves $ref URI to SchemaSpecPtr at generate time

  // For string spec deduplication
  struct StringSpecKey {
    std::string pattern;
    int min_length = 0;
    int max_length = -1;
    std::pair<std::string, std::string> wrapper;
    bool operator==(const StringSpecKey& other) const;
  };
  struct StringSpecKeyHash {
    size_t operator()(const StringSpecKey& key) const;
  };
  std::unordered_map<StringSpecKey, std::string, StringSpecKeyHash> string_spec_cache_;

  // Helper for integer/number range regex generation
  static std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end);
  static std::string GenerateFloatRangeRegex(
      std::optional<double> start, std::optional<double> end, int precision = 6
  );
  static std::string MakePatternForDigitRange(char start, char end, int remainingDigits);
  static std::vector<std::string> GenerateNumberPatterns(int64_t lower, int64_t upper);
  static std::string GenerateSubRangeRegex(int64_t lower, int64_t upper);
  static std::string FormatFloat(double value, int precision);

  // JSON string helpers
  static std::string JSONStrToPrintableStr(const std::string& json_str);

 protected:
  static std::optional<std::string> JSONFormatToRegexPattern(const std::string& format);

  // Expose for testing
  friend std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end);
  friend std::string GenerateFloatRangeRegex(
      std::optional<double> start, std::optional<double> end
  );
};

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
 * \param json_format Define the root
 * format of the object. If it's JSONFormat::kJSON, then it will generate a fully JSON-style
 * grammar. If it's JSONFormat::kXML, then it will generate a grammar with the root format is
 * XML-style, while the inner format is JSON-style. Default: JSONFormat::kJSON.
 * \returns The EBNF grammar string.
 */

std::string JSONSchemaToEBNF(
    const std::string& schema,
    bool any_whitespace = true,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true,
    std::optional<int> max_whitespace_cnt = std::nullopt,
    JSONFormat json_format = JSONFormat::kJSON
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
 * \param json_format Define the root format of the object. If it's JSONFormat::kJSON,
 * then it will generate a fully JSON-style grammar. If it's JSONFormat::kXML, then it will
 * generate a grammar with the root format is XML-style, while the inner format is JSON-style.
 * Default: JSONFormat::kJSON.
 * \returns The EBNF grammar string.
 */
std::string JSONSchemaToEBNF(
    const picojson::value& schema,
    bool any_whitespace = true,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true,
    std::optional<int> max_whitespace_cnt = std::nullopt,
    JSONFormat json_format = JSONFormat::kJSON
);

/*!
 * \brief Generate regex pattern for integer/float range.
 * \param start The start of the range (inclusive). If null assume negative infinity.
 * \param end The end of the range (inclusive). If null assume infinity.
 * \returns The regex pattern that matches integers/floats in the given range.
 */
std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end);

std::string GenerateFloatRangeRegex(std::optional<double> start, std::optional<double> end);

/*!
 * \brief Convert a function call to a Grammar.
 * \param schema The schema of the parameters of the function call.
 * \return The ebnf-grammar to match the requirements of the schema, and
 * in Qwen xml style.
 */
std::string QwenXMLToolCallingToEBNF(const std::string& schema);

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_H_
