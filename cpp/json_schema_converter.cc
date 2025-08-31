/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter.cc
 */
#include "json_schema_converter.h"

#include <picojson.h>

#include <climits>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ebnf_script_creator.h"
#include "regex_converter.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

enum class SchemaErrorType : int {
  kInvalidSchema = 0,
  kUnsatisfiableSchema = 1,
};

using SchemaError = TypedError<SchemaErrorType>;

/*!
 * \brief Manage the indent and separator for the generation of EBNF grammar.
 * \param indent The number of spaces for each indent. If it is std::nullopt, there will be no
 * indent or newline.
 * \param separator The separator between different elements in json. Examples include "," and ", ".
 * \param any_whitespace Whether to ignore the indentation restrictions, and allow any whitespace.
 */
class IndentManager {
 public:
  IndentManager(std::optional<int> indent, const std::string& separator, bool any_whitespace)
      : any_whitespace_(any_whitespace),
        enable_newline_(indent.has_value()),
        indent_(indent.value_or(0)),
        separator_(separator),
        total_indent_(0),
        is_first_({true}) {}

  /*! \brief Enter a new indent level. */
  void StartIndent() {
    total_indent_ += indent_;
    is_first_.push_back(true);
  }

  /*! \brief Exit the current indent level. */
  void EndIndent() {
    total_indent_ -= indent_;
    is_first_.pop_back();
  }

  /*!
   * \brief Get the next start separator in the current level. The next separator is escaped and
   * quoted.
   * \example
   * \code
   * IndentManager indent_manager(2, ", ");
   * indent_manager.StartIndent();
   * indent_manager.StartSeparator(); // get the start separator: "\"\n  \""
   * indent_manager.MiddleSeparator(); // get the middle separator: "\",\n  \""
   * indent_manager.EndSeparator(); // get the end separator: "\"\n\""
   * indent_manager.EndIndent();
   * \endcode
   */
  std::string StartSeparator();

  std::string MiddleSeparator();

  std::string EndSeparator();

  std::string EmptySeparator();

  /*!
   * \brief Get the next separator in the current level. When first called in the current
   * level, the starting separator will be returned. When called again, the middle separator will be
   * returned. When called with `is_end=True`, the ending separator will be returned.
   * \param is_end Get the separator for the end of the current level.
   * \example
   * \code
   * IndentManager indent_manager(2, ", ");
   * indent_manager.StartIndent();
   * indent_manager.GetSep(); // get the start separator: "\"\n  \""
   * indent_manager.GetSep(); // get the middle separator: "\",\n  \""
   * indent_manager.GetSep(true); // get the end separator: "\"\n\""
   * indent_manager.EndIndent();
   * \endcode
   */
  std::string NextSeparator(bool is_end = false);

 private:
  bool any_whitespace_;
  bool enable_newline_;
  int64_t indent_;
  std::string separator_;
  int64_t total_indent_;
  std::vector<bool> is_first_;
  friend class JSONSchemaConverter;
};

std::string IndentManager::StartSeparator() {
  if (any_whitespace_) {
    return "[ \\n\\t]*";
  }
  if (!enable_newline_) {
    return "\"\"";
  }
  return "\"\\n" + std::string(total_indent_, ' ') + "\"";
}

std::string IndentManager::MiddleSeparator() {
  if (any_whitespace_) {
    return "[ \\n\\t]* \"" + separator_ + "\" [ \\n\\t]*";
  }
  if (!enable_newline_) {
    return "\"" + separator_ + "\"";
  }
  return "\"" + separator_ + "\\n" + std::string(total_indent_, ' ') + "\"";
}

std::string IndentManager::EndSeparator() {
  if (any_whitespace_) {
    return "[ \\n\\t]*";
  }
  if (!enable_newline_) {
    return "\"\"";
  }
  return "\"\\n" + std::string(total_indent_ - indent_, ' ') + "\"";
}

std::string IndentManager::EmptySeparator() {
  if (any_whitespace_) {
    return "[ \\n\\t]*";
  }
  return "\"\"";
}

std::string IndentManager::NextSeparator(bool is_end) {
  if (any_whitespace_) {
    if (is_first_.back() || is_end) {
      is_first_.back() = false;
      return "[ \\n\\t]*";
    } else {
      return "[ \\n\\t]* \"" + separator_ + "\" [ \\n\\t]*";
    }
  }

  std::string res = "";
  if (!is_first_.back() && !is_end) {
    res += separator_;
  }
  is_first_.back() = false;

  if (enable_newline_) {
    res += "\\n";
  }

  if (!is_end) {
    res += std::string(total_indent_, ' ');
  } else {
    res += std::string(total_indent_ - indent_, ' ');
  }

  return "\"" + res + "\"";
}

/*!
 * \brief Convert JSON schema string to EBNF grammar string. The parameters follow
 * JSONSchemaToEBNF().
 *
 * \note About the representation of json schema in this converter. JSON schema could be two types:
 * bool (true or false) or dict (a json dict) containing attributes. We use picojson::value to
 * represent the json schema.
 */
class JSONSchemaConverter {
 public:
  JSONSchemaConverter(
      const picojson::value& json_schema,
      bool any_whitespace,
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool strict_mode,
      JSONFormat json_format = JSONFormat::kJSON
  );

  /*! \brief The root method. Convert the JSON schema to EBNF grammar string. */
  std::string Convert(const JSONFormat json_format = JSONFormat::kJSON);

  /*! \brief Generate the regex for integer range. Public for testing. */
  static std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end);

  /*! \brief Generate the regex for float range. Public for testing. */
  static std::string GenerateFloatRangeRegex(
      std::optional<double> start, std::optional<double> end, int precision
  );

 private:
  // The name of the root rule
  inline static const std::string kRootRuleName = "root";
  // The name of the basic rules
  inline static const std::string kBasicAny = "basic_any";
  inline static const std::string kBasicInteger = "basic_integer";
  inline static const std::string kBasicNumber = "basic_number";
  inline static const std::string kBasicString = "basic_string";
  inline static const std::string kBasicBoolean = "basic_boolean";
  inline static const std::string kBasicNull = "basic_null";
  inline static const std::string kBasicArray = "basic_array";
  inline static const std::string kBasicObject = "basic_object";
  inline static const std::string kXMLAny = "xml_any";

  // The name of the helper rules to construct basic rules
  inline static const std::string kBasicEscape = "basic_escape";
  inline static const std::string kBasicStringSub = "basic_string_sub";
  inline static const std::string kXMLEntity = "xml_entity";
  inline static const std::string kXMLEscape = "xml_escape";
  inline static const std::string kXMLString = "xml_string";
  inline static const std::string kXMLVariableName = "xml_variable_name";
  inline static const std::string kWhiteSpace = "[ \\n\\t]*";

  /*! \brief Add the basic rules to the rules list and the basic_rules_cache. */
  void AddBasicRules(JSONFormat json_format);

  /*! \brief Add helper rules for the basic rules. */
  void AddJSONHelperRules();

  /*! \brief Add xml-style helper rules for the basic rules. */
  void AddXMLHelperRules();

  /*! \brief Create a rule for the given schema and name, and add it to the basic_rules_cache. */
  void CreateBasicRule(
      const picojson::value& schema,
      const std::string& name,
      const JSONFormat json_format = JSONFormat::kJSON
  );

  /*! \brief Get the index for the schema in the cache. Keys that do not effect the validation
   * will be ignored when finding the corresponding cache rule. */
  std::string GetSchemaCacheIndex(const picojson::value& schema);

  /*! \brief Helpers for GenerateRangeRegex and GenerateFloatRangeRegex */
  static std::string MakePatternForDigitRange(char start, char end, int remainingDigits);

  static std::vector<std::string> GenerateNumberPatterns(int64_t lower, int64_t upper);

  static std::string GenerateSubRangeRegex(int64_t lower, int64_t upper);

  static std::string FormatFloat(double value, int precision);

  /*!
   * \brief Create a rule with the given schema and rule name hint.
   * \returns The name of the rule will be returned. That is not necessarily the same as the
   * rule_name_hint due to the caching mechanism.
   */
  std::string CreateRuleFromSchema(
      const picojson::value& schema,
      const std::string& rule_name_hint,
      const JSONFormat json_format = JSONFormat::kJSON
  );

  /*! \brief Get the next separator in the current level from the indent manager. */
  std::string NextSeparator(bool is_end = false);

  /*! \brief Warn if any keyword is existing in the schema but not supported. */
  static void WarnUnsupportedKeywords(
      const picojson::value& schema, const std::vector<std::string>& keywords, bool verbose = false
  );

  /*! \brief Warn if any keyword is existing in the object but not supported. */
  static void WarnUnsupportedKeywords(
      const picojson::object& schema, const std::vector<std::string>& keywords, bool verbose = false
  );

  // NOTE: the visit functions should always return the rule body for later constructing the rule.

  /*! \brief Visit the schema and return the rule body for later constructing the rule. */
  std::string VisitSchema(
      const picojson::value& schema,
      const std::string& rule_name,
      const JSONFormat json_format = JSONFormat::kJSON
  );

  /*! \brief Visit a reference schema. */
  std::string VisitRef(const picojson::object& schema, const std::string& rule_name);

  /*! \brief Get the rule from the URI. */
  std::string URIToRule(const std::string& uri);

  /*! \brief Visit a const schema. */
  std::string VisitConst(const picojson::object& schema, const std::string& rule_name);

  /*! \brief Visit an enum schema. */
  std::string VisitEnum(const picojson::object& schema, const std::string& rule_name);

  /*! \brief Convert the JSON string to a printable string that can be shown in BNF. */
  std::string JSONStrToPrintableStr(const std::string& json_str);

  /*! \brief Visit an anyOf schema. */
  std::string VisitAnyOf(const picojson::object& schema, const std::string& rule_name);

  picojson::value FuseAllOfSchema(const std::vector<picojson::value>& schemas);

  /*! \brief Visit an allOf schema. */
  std::string VisitAllOf(const picojson::object& schema, const std::string& rule_name);

  /*! \brief Visit a true schema that can match anything. */
  std::string VisitAny(
      const picojson::value& schema, const std::string& rule_name, const JSONFormat json_format
  );

  /*! \brief Visit an integer schema. */
  std::string VisitInteger(const picojson::object& schema, const std::string& rule_name);

  /*! \brief Visit a number schema. */
  std::string VisitNumber(const picojson::object& schema, const std::string& rule_name);

  /*! \brief Visit a string schema. */
  std::string VisitString(
      const picojson::object& schema, const std::string& rule_name, const JSONFormat json_format
  );

  /*! \brief Visit a boolean schema. */
  std::string VisitBoolean(const picojson::object& schema, const std::string& rule_name);

  /*! \brief Visit a null schema. */
  std::string VisitNull(const picojson::object& schema, const std::string& rule_name);

  struct ArraySpec {
    std::vector<picojson::value> prefix_item_schemas;
    bool allow_additional_items;
    picojson::value additional_item_schema;
    int64_t min_items;
    int64_t max_items;
  };

  Result<ArraySpec, SchemaError> ParseArraySchema(const picojson::object& schema);

  struct ObjectSpec {
    std::vector<std::pair<std::string, picojson::value>> properties;
    std::vector<std::pair<std::string, picojson::value>> pattern_properties;
    bool allow_additional_properties;
    picojson::value additional_properties_schema;
    bool allow_unevaluated_properties;
    picojson::value unevaluated_properties_schema;
    std::unordered_set<std::string> required_properties;
    picojson::value property_names;
    int min_properties;
    int max_properties;
  };

  struct StringSpec {
    std::string pattern;
    int min_length = 0;
    int max_length = -1;
    std::pair<std::string, std::string> wrapper;
  };

  Result<StringSpec, SchemaError> ParseStringSchema(
      const picojson::object& schema, JSONFormat escape_format
  );

  Result<ObjectSpec, SchemaError> ParseObjectSchema(const picojson::object& schema);

  /*!
   * \brief Visit an array schema.
   * \example
   * Schema:
   * \code
   * {
   *     "type": "array",
   *     "prefixItems": [
   *         {"type": "boolean"},
   *         {"type": "integer"}
   *     ],
   *     "items": {
   *         "type": "string"
   *     }
   * }
   * \endcode
   * Rule (not considering the indent):
   * \code
   * root ::= "[" basic_boolean ", " basic_integer (", " basic_string)* "]"
   * \endcode
   */
  std::string VisitArray(const picojson::object& schema, const std::string& rule_name);

  /*!
   * \brief Visit an object schema.
   * \example
   * Schema:
   * \code
   * {
   *     "type": "object",
   *     "properties": {
   *         "a": {"type": "string"},
   *         "b": {"type": "integer"}
   *     },
   *     "required": ["a"],
   *     "additionalProperties": true
   * }
   * \endcode
   *
   * Rule (not considering the indent):
   * \code
   * root ::= "{" "a" ":" basic_string (", " "b" ":" basic_integer)*
   *          (", " basic_string ": " basic_any)* "}"
   * \endcode

   * We need special handling when all properties are optional, since the handling of separators
   * is tricky in this case. E.g.

   * Schema:
   * \code
   * {
   *     "type": "object",
   *     "properties": {
   *         "a": {"type": "string"},
   *         "b": {"type": "integer"},
   *         "c": {"type": "boolean"}
   *     },
   *     "additionalProperties": true
   * }
   * \endcode
   *
   * Rule (indent=2):
   * \code
   * root ::= "{" ("\n  " (a root_sub_1 | b root_sub_2 | c root_sub_3 | d root_sub_3)
   *          "\n" | "") "}"
   * root_sub_1 ::= ",\n  " b r2 | r2
   * root_sub_2 ::= ",\n  " c r3 | r3
   * root_sub_3 ::= (",\n  " d)*
   * \endcode
   */
  std::string VisitObject(
      const picojson::object& schema,
      const std::string& rule_name,
      const JSONFormat json_format = JSONFormat::kJSON
  );

  /*!
   * \brief Visit a type array schema:
   * \example
   * \code
   * {
   *     "type": ["integer", "string"]
   * }
   * \endcode
   *
   * Method:
   * - Create a schema for each type in the type array. Copying all other properties.
   * - Visit each schema and get the rule name.
   * - Return "(" rule_name_1 | rule_name_2 | ... | rule_name_n ")"
   */
  std::string VisitTypeArray(const picojson::object& schema, const std::string& rule_name);

  /*! \brief Get the pattern for a property in the object schema. */
  std::string GetPropertyPattern(
      const std::string& prop_name,
      const picojson::value& prop_schema,
      const std::string& rule_name,
      int64_t idx,
      const JSONFormat json_format = JSONFormat::kJSON
  );

  /*! \brief Get the pattern for the additional/unevaluated properties in the object schema. */
  std::string GetOtherPropertyPattern(
      const std::string& key_pattern,
      const picojson::value& prop_schema,
      const std::string& rule_name,
      const std::string& rule_name_suffix,
      const JSONFormat json_format = JSONFormat::kJSON
  );

  /*! \brief Get the pattern for the properties with repetition number limit. */
  std::string GetPropertyWithNumberConstrains(
      const std::string& pattern,
      int min_properties,
      int max_properties,
      int already_repeated_times = 0
  );

  /*! \brief Get the partial rule for the properties. See the
   * example in VisitObject(). */
  std::string GetPartialRuleForProperties(
      const std::vector<std::pair<std::string, picojson::value>>& properties,
      const std::unordered_set<std::string>& required,
      const picojson::value& additional,
      const std::string& rule_name,
      const std::string& additional_suffix,
      const int min_properties,
      const int max_properties,
      const JSONFormat json_format = JSONFormat::kJSON
  );

  // The EBNF script creator
  EBNFScriptCreator ebnf_script_creator_;
  // The indent manager to get separators
  std::optional<IndentManager> indentManager_;
  // The root JSON schema
  picojson::value json_schema_;
  // Whether to use strict mode in conversion. See JSONSchemaToEBNF().
  bool strict_mode_;
  // The colon separator
  std::string colon_pattern_;
  // The cache for basic rules. Mapping from the key of schema returned by GetSchemaCacheIndex()
  // to the basic rule name.
  std::unordered_map<std::pair<std::string, JSONFormat>, std::string> basic_rules_cache_;
  // Whether to use any whitespace in the conversion
  bool any_whitespace_;
  // The cache for URI to rule. Mapping from the URI to the rule name.
  std::unordered_map<std::string, std::string> uri_to_rule_cache_;
};

JSONSchemaConverter::JSONSchemaConverter(
    const picojson::value& json_schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    JSONFormat json_format
)
    : json_schema_(json_schema), strict_mode_(strict_mode), any_whitespace_(any_whitespace) {
  if (!separators.has_value()) {
    if (indent == std::nullopt) {
      separators = std::make_pair(", ", ": ");
    } else {
      separators = std::make_pair(",", ": ");
    }
  }
  if (any_whitespace) {
    separators = std::make_pair(",", ":");
  }
  indentManager_ = IndentManager(indent, separators->first, any_whitespace);
  if (any_whitespace) {
    colon_pattern_ = "[ \\n\\t]* \"" + separators->second + "\" [ \\n\\t]*";
  } else {
    colon_pattern_ = "\"" + separators->second + "\"";
  }

  AddBasicRules(json_format);
}

std::string JSONSchemaConverter::Convert(const JSONFormat json_format) {
  switch (json_format) {
    // If the type is JSON, we handle it trivially.
    case (JSONFormat::kJSON): {
      CreateRuleFromSchema(json_schema_, kRootRuleName, json_format);
      break;
    }

    // If the type is XML, then the root schema must be a object.
    // To ensure the inner object is in JSON format, we need to call
    // VisitObject directly, and pass JSONFormat::kXML to it.
    // In other VisitObject, only JSONFormat::kJSON will be passed.
    case (JSONFormat::kXML): {
      auto rule_name = ebnf_script_creator_.AllocateRuleName(kRootRuleName);
      XGRAMMAR_CHECK(json_schema_.is<picojson::object>());
      std::string rule_content =
          VisitObject(json_schema_.get<picojson::object>(), rule_name, json_format);
      ebnf_script_creator_.AddRuleWithAllocatedName(rule_name, rule_content);
      break;
    }
  }
  return ebnf_script_creator_.GetScript();
}

void JSONSchemaConverter::AddBasicRules(JSONFormat json_format) {
  bool past_strict_mode = strict_mode_;
  // Allow any field for basic array/obj rules
  strict_mode_ = false;

  auto past_indent_manager = indentManager_;
  if (any_whitespace_) {
    indentManager_ = IndentManager(std::nullopt, ",", true);
  } else {
    indentManager_ = IndentManager(std::nullopt, ", ", false);
  }
  AddJSONHelperRules();
  if (json_format == JSONFormat::kXML) {
    AddXMLHelperRules();
    CreateBasicRule(
        picojson::value(picojson::object{{"type", picojson::value("string")}}),
        kXMLString,
        JSONFormat::kXML
    );
    CreateBasicRule(picojson::value(true), kXMLAny, JSONFormat::kXML);
    basic_rules_cache_[{
        GetSchemaCacheIndex(picojson::value(picojson::object())), JSONFormat::kXML
    }] = kXMLAny;
  }
  CreateBasicRule(picojson::value(true), kBasicAny);
  basic_rules_cache_[{
      GetSchemaCacheIndex(picojson::value(picojson::object())), JSONFormat::kJSON
  }] = kBasicAny;
  CreateBasicRule(
      picojson::value(picojson::object{{"type", picojson::value("integer")}}), kBasicInteger
  );
  CreateBasicRule(
      picojson::value(picojson::object{{"type", picojson::value("number")}}), kBasicNumber
  );
  CreateBasicRule(
      picojson::value(picojson::object{{"type", picojson::value("string")}}), kBasicString
  );
  CreateBasicRule(
      picojson::value(picojson::object{{"type", picojson::value("boolean")}}), kBasicBoolean
  );
  CreateBasicRule(picojson::value(picojson::object{{"type", picojson::value("null")}}), kBasicNull);
  CreateBasicRule(
      picojson::value(picojson::object{{"type", picojson::value("array")}}), kBasicArray
  );
  CreateBasicRule(
      picojson::value(picojson::object{{"type", picojson::value("object")}}), kBasicObject
  );

  strict_mode_ = past_strict_mode;
  indentManager_ = past_indent_manager;
}

void JSONSchemaConverter::AddJSONHelperRules() {
  ebnf_script_creator_.AddRule(
      kBasicEscape, "[\"\\\\/bfnrt] | \"u\" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]"
  );
  ebnf_script_creator_.AddRule(
      kBasicStringSub,
      "(\"\\\"\" | [^\\0-\\x1f\\\"\\\\\\r\\n] " + kBasicStringSub + " | \"\\\\\" " + kBasicEscape +
          " " + kBasicStringSub + ") (= [ \\n\\t]* [,}\\]:])"
  );
}

void JSONSchemaConverter::AddXMLHelperRules() {
  ebnf_script_creator_.AddRule(
      kXMLEscape, "[\"\\\\/bfnrt] | \"u\" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]"
  );
  ebnf_script_creator_.AddRule(
      kXMLEntity, " \"&lt;\" | \"&gt;\" | \"&amp;\" | \"&quot;\" | \"&apos;\""
  );
  ebnf_script_creator_.AddRule(
      kXMLString,
      "(\"\" | [^<>&\\0-\\x1f\\\\\\r\\n] " + kXMLString + " | \"\\\\\" " + kXMLEscape + " " +
          kXMLString + " | " + kXMLEntity + " " + kXMLString + ") (= [ \\n\\t]*)"
  );
  ebnf_script_creator_.AddRule(kXMLVariableName, "[a-zA-Z_] [a-zA-Z0-9_]*");
}

void JSONSchemaConverter::CreateBasicRule(
    const picojson::value& schema, const std::string& name, const JSONFormat json_format
) {
  std::string rule_name = CreateRuleFromSchema(schema, name, json_format);
  basic_rules_cache_[{GetSchemaCacheIndex(schema), json_format}] = rule_name;
}

std::string JSONSchemaConverter::NextSeparator(bool is_end) {
  return indentManager_->NextSeparator(is_end);
}

void JSONSchemaConverter::WarnUnsupportedKeywords(
    const picojson::value& schema, const std::vector<std::string>& keywords, bool verbose
) {
  if (schema.is<bool>()) {
    return;
  }

  XGRAMMAR_DCHECK(schema.is<picojson::object>()) << "Schema should be an object or bool";
  WarnUnsupportedKeywords(schema.get<picojson::object>(), keywords, verbose);
}

void JSONSchemaConverter::WarnUnsupportedKeywords(
    const picojson::object& schema, const std::vector<std::string>& keywords, bool verbose
) {
  if (!verbose) {
    return;
  }
  for (const auto& keyword : keywords) {
    if (schema.find(keyword) != schema.end()) {
      XGRAMMAR_LOG(WARNING) << "Keyword " << keyword << " is not supported";
    }
  }
}

std::string JSONSchemaConverter::CreateRuleFromSchema(
    const picojson::value& schema, const std::string& rule_name_hint, const JSONFormat json_format
) {
  std::string idx = GetSchemaCacheIndex(schema);
  if (basic_rules_cache_.count({idx, json_format})) {
    if (rule_name_hint == kRootRuleName) {
      // If the rule name is root, we need to define the root rule instead of just using the
      // cached rule.
      return ebnf_script_creator_.AddRule(rule_name_hint, basic_rules_cache_[{idx, json_format}]);
    }
    return basic_rules_cache_[{idx, json_format}];
  }

  auto rule_name = ebnf_script_creator_.AllocateRuleName(rule_name_hint);
  std::string rule_content = VisitSchema(schema, rule_name, json_format);
  ebnf_script_creator_.AddRuleWithAllocatedName(rule_name, rule_content);
  return rule_name;
}

std::string JSONSchemaConverter::GetSchemaCacheIndex(const picojson::value& schema) {
  // Keys that do not effect the validation
  static const std::unordered_set<std::string> kSkippedKeys = {
      "title",
      "default",
      "description",
      "examples",
      "deprecated",
      "readOnly",
      "writeOnly",
      "$comment",
      "$schema",
  };
  if (schema.is<picojson::object>()) {
    // remove skipped keys and sort key by lexicographical order
    std::string result = "{";
    std::vector<std::pair<std::string, picojson::value>> sorted_kv;
    for (const auto& kv : schema.get<picojson::object>()) {
      if (kSkippedKeys.count(kv.first) == 0) {
        sorted_kv.push_back(kv);
      }
    }
    std::sort(sorted_kv.begin(), sorted_kv.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
    });
    int64_t idx = 0;
    for (const auto& [key, value] : sorted_kv) {
      if (idx != 0) {
        result += ",";
      }
      ++idx;
      result += "\"" + key + "\":" + GetSchemaCacheIndex(value);
    }
    return result + "}";
  } else if (schema.is<picojson::array>()) {
    std::string result = "[";
    int64_t idx = 0;
    for (const auto& item : schema.get<picojson::array>()) {
      if (idx != 0) {
        result += ",";
      }
      ++idx;
      result += GetSchemaCacheIndex(item);
    }
    return result + "]";
  }
  // If the object is neither an array nor an object, return it directly
  return schema.serialize(false);
}

std::string JSONSchemaConverter::VisitSchema(
    const picojson::value& schema, const std::string& rule_name, const JSONFormat json_format
) {
  if (schema.is<bool>()) {
    XGRAMMAR_CHECK(schema.get<bool>()) << "Schema should not be false: it cannot accept any value";
    return VisitAny(schema, rule_name, json_format);
  }
  XGRAMMAR_CHECK(schema.is<picojson::object>())
      << "Schema should be an object or bool, but got " << schema.serialize(false);

  WarnUnsupportedKeywords(
      schema,
      {
          "not",
          "if",
          "then",
          "else",
          "dependentRequired",
          "dependentSchemas",
      }
  );

  const auto& schema_obj = schema.get<picojson::object>();

  if (schema_obj.count("$ref")) {
    return VisitRef(schema_obj, rule_name);
  } else if (schema_obj.count("const")) {
    return VisitConst(schema_obj, rule_name);
  } else if (schema_obj.count("enum")) {
    return VisitEnum(schema_obj, rule_name);
  } else if (schema_obj.count("anyOf") || schema_obj.count("oneOf")) {
    return VisitAnyOf(schema_obj, rule_name);
  } else if (schema_obj.count("allOf")) {
    return VisitAllOf(schema_obj, rule_name);
  } else if (schema_obj.count("type")) {
    if (schema_obj.at("type").is<picojson::array>()) {
      return VisitTypeArray(schema_obj, rule_name);
    }
    XGRAMMAR_CHECK(schema_obj.at("type").is<std::string>()) << "Type should be a string";
    const std::string& type = schema_obj.at("type").get<std::string>();
    if (type == "integer") {
      return VisitInteger(schema_obj, rule_name);
    } else if (type == "number") {
      return VisitNumber(schema_obj, rule_name);
    } else if (type == "string") {
      return VisitString(schema_obj, rule_name, json_format);
    } else if (type == "boolean") {
      return VisitBoolean(schema_obj, rule_name);
    } else if (type == "null") {
      return VisitNull(schema_obj, rule_name);
    } else if (type == "array") {
      return VisitArray(schema_obj, rule_name);
    } else if (type == "object") {
      return VisitObject(schema_obj, rule_name, JSONFormat::kJSON);
    } else {
      XGRAMMAR_LOG(FATAL) << "Unsupported type \"" << type << "\"";
    }
  } else if (schema_obj.count("properties") || schema_obj.count("additionalProperties") ||
             schema_obj.count("unevaluatedProperties")) {
    return VisitObject(schema_obj, rule_name);
  } else if (schema_obj.count("items") || schema_obj.count("prefixItems") ||
             schema_obj.count("unevaluatedItems")) {
    return VisitArray(schema_obj, rule_name);
  }

  // If no above keyword is detected, we treat it as any
  return VisitAny(schema, rule_name, json_format);
}

std::string JSONSchemaConverter::VisitRef(
    const picojson::object& schema, const std::string& rule_name
) {
  XGRAMMAR_CHECK(schema.count("$ref") && schema.at("$ref").is<std::string>())
      << "Schema $ref should be a string";
  auto ref_str = schema.at("$ref").get<std::string>();
  return URIToRule(ref_str);
}

std::string JSONSchemaConverter::URIToRule(const std::string& uri) {
  if (uri_to_rule_cache_.count(uri)) {
    return uri_to_rule_cache_[uri];
  }

  if (uri == "#") {
    return kRootRuleName;
  }

  if (uri.size() < 2 || uri[0] != '#' || uri[1] != '/') {
    XGRAMMAR_LOG(WARNING) << "URI should either be '#' or start with '#/' but got " << uri;
    return kBasicAny;
  }

  std::vector<std::string> parts;
  std::stringstream ss(uri.substr(2));
  std::string part;
  std::string new_rule_name_perfix;
  while (std::getline(ss, part, '/')) {
    if (!part.empty()) {
      parts.push_back(part);
    }
    // Update new_rule_name_perfix
    if (!new_rule_name_perfix.empty()) {
      new_rule_name_perfix += "_";
    }
    // filter out non-alpha characters
    for (const auto& c : part) {
      if (std::isalpha(c) || c == '_' || c == '-' || c == '.') {
        new_rule_name_perfix += c;
      }
    }
  }

  picojson::value current = json_schema_;
  for (const auto& part : parts) {
    XGRAMMAR_CHECK(current.is<picojson::object>() && current.contains(part))
        << "Cannot find field " << part << " in " << current.serialize(false);
    current = current.get(part);
  }

  auto new_rule_name = ebnf_script_creator_.AllocateRuleName(new_rule_name_perfix);
  uri_to_rule_cache_[uri] = new_rule_name;
  auto body = VisitSchema(current, new_rule_name);
  ebnf_script_creator_.AddRuleWithAllocatedName(new_rule_name, body);
  return new_rule_name;
}

std::string JSONSchemaConverter::VisitConst(
    const picojson::object& schema, const std::string& rule_name
) {
  XGRAMMAR_CHECK(schema.count("const"));
  // TODO(yixin): Customize serialize to support indent logics
  return "\"" + JSONStrToPrintableStr(schema.at("const").serialize()) + "\"";
}

std::string JSONSchemaConverter::VisitEnum(
    const picojson::object& schema, const std::string& rule_name
) {
  XGRAMMAR_CHECK(schema.count("enum"));
  std::string result = "";
  int64_t idx = 0;
  for (auto value : schema.at("enum").get<picojson::array>()) {
    if (idx != 0) {
      result += " | ";
    }
    ++idx;
    result += "(\"" + JSONStrToPrintableStr(value.serialize()) + "\")";
  }
  return result;
}

std::string JSONSchemaConverter::JSONStrToPrintableStr(const std::string& json_str) {
  static const std::vector<std::pair<std::string, std::string>> kReplaceMapping = {
      {"\\", "\\\\"}, {"\"", "\\\""}
  };
  std::string result = json_str;
  for (const auto& [k, v] : kReplaceMapping) {
    size_t pos = 0;
    while ((pos = result.find(k, pos)) != std::string::npos) {
      result.replace(pos, k.length(), v);
      pos += v.length();
    }
  }
  return result;
}

std::string JSONSchemaConverter::VisitAnyOf(
    const picojson::object& schema, const std::string& rule_name
) {
  XGRAMMAR_CHECK(schema.count("anyOf") || schema.count("oneOf"));
  std::string result = "";
  int64_t idx = 0;
  auto anyof_schema = schema.count("anyOf") ? schema.at("anyOf") : schema.at("oneOf");
  XGRAMMAR_CHECK(anyof_schema.is<picojson::array>()) << "anyOf or oneOf must be an array";
  for (auto anyof_schema : anyof_schema.get<picojson::array>()) {
    if (idx != 0) {
      result += " | ";
    }
    result += CreateRuleFromSchema(anyof_schema, rule_name + "_case_" + std::to_string(idx));
    ++idx;
  }
  return result;
}

picojson::value JSONSchemaConverter::FuseAllOfSchema(const std::vector<picojson::value>& schemas) {
  picojson::object fused_schema;
  XGRAMMAR_LOG(WARNING) << "Support for allOf with multiple options is still ongoing";
  return picojson::value(fused_schema);
}

std::string JSONSchemaConverter::VisitAllOf(
    const picojson::object& schema, const std::string& rule_name
) {
  // We support common usecases of AllOf, but not all, because it's impossible to support all
  // cases with CFG
  XGRAMMAR_CHECK(schema.count("allOf"));
  XGRAMMAR_CHECK(schema.at("allOf").is<picojson::array>()) << "allOf must be an array";
  auto all_array = schema.at("allOf").get<picojson::array>();
  // Case 1: allOf is a single schema
  if (all_array.size() == 1) {
    return VisitSchema(all_array[0], rule_name + "_case_0");
  }
  // Case 2: allOf is a list of schemas, we fuse them into a single schema
  auto fused_schema = FuseAllOfSchema(all_array);
  return VisitSchema(fused_schema, rule_name);
}

std::string JSONSchemaConverter::VisitAny(
    const picojson::value& schema, const std::string& rule_name, JSONFormat json_format
) {
  // Note integer is a subset of number, so we don't need to add integer here
  switch (json_format) {
    case JSONFormat::kJSON: {
      return kBasicNumber + " | " + kBasicString + " | " + kBasicBoolean + " | " + kBasicNull +
             " | " + kBasicArray + " | " + kBasicObject;
    }
    case JSONFormat::kXML: {
      return kBasicNumber + " | " + kXMLString + " | " + kBasicBoolean + " | " + kBasicNull +
             " | " + kBasicArray + " | " + kBasicObject;
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "Unsupported string escape type: " << static_cast<int>(json_format);
    }
  }
}

std::string JSONSchemaConverter::MakePatternForDigitRange(
    char start, char end, int remainingDigits
) {
  std::ostringstream oss;
  if (start == end) {
    oss << start;
  } else {
    oss << "[" << start << "-" << end << "]";
  }
  if (remainingDigits > 0) {
    oss << "\\d{" << remainingDigits << "}";
  }
  return oss.str();
}

std::vector<std::string> JSONSchemaConverter::GenerateNumberPatterns(int64_t lower, int64_t upper) {
  std::vector<std::string> patterns;

  int lower_len = static_cast<int>(std::to_string(lower).size());
  int upper_len = static_cast<int>(std::to_string(upper).size());

  for (int len = lower_len; len <= upper_len; ++len) {
    const int64_t digit_min = static_cast<int64_t>(std::pow(10, len - 1));
    const int64_t digit_max = static_cast<int64_t>(std::pow(10, len)) - 1;

    int64_t start = (len == lower_len) ? lower : digit_min;
    int64_t end = (len == upper_len) ? upper : digit_max;

    std::string start_str = std::to_string(start);
    std::string end_str = std::to_string(end);

    if (len == 1) {
      patterns.push_back(MakePatternForDigitRange(start_str[0], end_str[0], 0));
      continue;
    }

    int prefix = 0;
    while (prefix < len && start_str[prefix] == end_str[prefix]) {
      prefix++;
    }

    if (prefix == len) {
      patterns.push_back(start_str);
      continue;
    }

    // Generate common prefix pattern if only last digit differs for start/end
    if (prefix > 0 && prefix >= len - 2) {
      std::string common_part = start_str.substr(0, prefix);
      patterns.push_back(
          common_part +
          MakePatternForDigitRange(start_str[prefix], end_str[prefix], len - prefix - 1)
      );
      continue;
    }

    if (len == lower_len && len == upper_len) {
      if (start == digit_max) {
        XGRAMMAR_ICHECK(start == end);
        patterns.push_back(start_str);
      } else if (start == digit_min) {
        if (end == digit_max) {
          patterns.push_back("[1-9]\\d{" + std::to_string(len - 1) + "}");
        } else {
          for (size_t i = 0; i < end_str.size(); i++) {
            if (i == 0) {
              // First digit: range from 1 to end[0]-1
              if (end_str[0] > '1') {
                patterns.push_back(
                    MakePatternForDigitRange('1', static_cast<char>(end_str[0] - 1), len - 1)
                );
              }
            } else {
              // Fix first i digits to end[0..i-1], then range from 0 to end[i]-1
              std::string prefix = end_str.substr(0, i);
              if (end_str[i] > '0') {
                patterns.push_back(
                    prefix +
                    MakePatternForDigitRange('0', static_cast<char>(end_str[i] - 1), len - i - 1)
                );
              }
            }
          }
          patterns.push_back(end_str);
        }
      } else if (end == digit_max) {
        for (size_t i = 0; i < start_str.size(); i++) {
          if (i == 0) {
            // First digit: range from start[0]+1 to 9
            if (start_str[0] < '9') {
              patterns.push_back(
                  MakePatternForDigitRange(static_cast<char>(start_str[0] + 1), '9', len - 1)
              );
            }
          } else {
            // Fix first i digits to start[0..i-1], then range from start[i]+1 to 9
            std::string prefix = start_str.substr(0, i);
            if (start_str[i] < '9') {
              patterns.push_back(
                  prefix +
                  MakePatternForDigitRange(static_cast<char>(start_str[i] + 1), '9', len - i - 1)
              );
            }
          }
        }
        patterns.push_back(start_str);
      } else {
        // Handle middle range between first digits if they differ by more than 1
        char start_first_digit = start_str[0];
        char end_first_digit = end_str[0];

        if (end_first_digit - start_first_digit > 1) {
          patterns.push_back(MakePatternForDigitRange(
              static_cast<char>(start_first_digit + 1),
              static_cast<char>(end_first_digit - 1),
              len - 1
          ));
        }

        // Patterns starting from start
        for (size_t i = 0; i < start_str.size(); i++) {
          if (i == 0) {
            std::string prefix = start_str.substr(0, 1);
            if (start_str[1] < '9') {
              patterns.push_back(
                  prefix +
                  MakePatternForDigitRange(static_cast<char>(start_str[1] + 1), '9', len - 2)
              );
            }
          } else {
            std::string prefix = start_str.substr(0, i);
            if (start_str[i] < '9') {
              patterns.push_back(
                  prefix +
                  MakePatternForDigitRange(static_cast<char>(start_str[i] + 1), '9', len - i - 1)
              );
            }
          }
        }
        patterns.push_back(start_str);

        // Patterns starting from end
        for (size_t i = 0; i < end_str.size(); i++) {
          if (i == 0) {
            std::string prefix = end_str.substr(0, 1);
            if (end_str[1] > '0') {
              patterns.push_back(
                  prefix + MakePatternForDigitRange('0', static_cast<char>(end_str[1] - 1), len - 2)
              );
            }
          } else {
            std::string prefix = end_str.substr(0, i);
            if (end_str[i] > '0') {
              patterns.push_back(
                  prefix +
                  MakePatternForDigitRange('0', static_cast<char>(end_str[i] - 1), len - i - 1)
              );
            }
          }
        }
        patterns.push_back(end_str);
      }
    }

    else if (len == lower_len && len != upper_len) {
      XGRAMMAR_ICHECK(end == digit_max);
      if (start == digit_min) {
        patterns.push_back("[1-9]\\d{" + std::to_string(len - 1) + "}");
      } else {
        for (size_t i = 0; i < start_str.size(); i++) {
          if (i == 0) {
            if (start_str[0] < '9') {
              patterns.push_back(
                  MakePatternForDigitRange(static_cast<char>(start_str[0] + 1), '9', len - 1)
              );
            }
          } else {
            std::string prefix = start_str.substr(0, i);
            if (start_str[i] < '9') {
              patterns.push_back(
                  prefix +
                  MakePatternForDigitRange(static_cast<char>(start_str[i] + 1), '9', len - i - 1)
              );
            }
          }
        }
        patterns.push_back(start_str);
      }
    }

    else if (len != lower_len && len == upper_len) {
      XGRAMMAR_ICHECK(start == digit_min);
      if (end == digit_max) {
        patterns.push_back("[1-9]\\d{" + std::to_string(len - 1) + "}");
      } else {
        for (size_t i = 0; i < end_str.size(); i++) {
          if (i == 0) {
            if (end_str[0] > '1') {
              patterns.push_back(
                  MakePatternForDigitRange('1', static_cast<char>(end_str[0] - 1), len - 1)
              );
            }
          } else {
            std::string prefix = end_str.substr(0, i);
            if (end_str[i] > '0') {
              patterns.push_back(
                  prefix +
                  MakePatternForDigitRange('0', static_cast<char>(end_str[i] - 1), len - i - 1)
              );
            }
          }
        }
        patterns.push_back(end_str);
      }
    }

    // len != lower_len && len != upper_len
    else {
      patterns.push_back("[1-9]\\d{" + std::to_string(len - 1) + "}");
    }
  }

  return patterns;
}

std::string JSONSchemaConverter::GenerateSubRangeRegex(int64_t lower, int64_t upper) {
  std::vector<std::string> patterns = GenerateNumberPatterns(lower, upper);
  std::ostringstream oss;
  for (size_t i = 0; i < patterns.size(); ++i) {
    if (i > 0) {
      oss << "|";
    }
    oss << patterns[i];
  }
  return "(" + oss.str() + ")";
}

std::string JSONSchemaConverter::GenerateRangeRegex(
    std::optional<int64_t> start, std::optional<int64_t> end
) {
  std::vector<std::string> parts;
  std::ostringstream result;

  // If start and end undefined - match any integer
  if (!start && !end) {
    return "^-?\\d+$";
  }

  // Only start defined - match numbers >= start
  if (start && !end) {
    if (start.value() <= 0) {
      if (start.value() < 0) {
        parts.push_back("-" + GenerateSubRangeRegex(-(-start.value()), 1));
      }
      parts.push_back("0");
      parts.push_back("[1-9]\\d*");
    } else {
      std::string start_str = std::to_string(start.value());
      int len = static_cast<int>(start_str.length());

      if (len == 1) {
        parts.push_back(MakePatternForDigitRange(start_str[0], '9', 0));
        parts.push_back("[1-9]\\d*");
      } else {
        parts.push_back(start_str);

        // Handle numbers of same length
        for (size_t i = 0; i < start_str.size(); i++) {
          if (i == 0) {
            // First digit: range from start[0]+1 to 9
            if (start_str[0] < '9') {
              parts.push_back(
                  MakePatternForDigitRange(static_cast<char>(start_str[0] + 1), '9', len - 1)
              );
            }
          } else {
            // Fix first i digits to start[0..i-1], then range from start[i]+1 to 9
            std::string prefix = start_str.substr(0, i);
            if (start_str[i] < '9') {
              parts.push_back(
                  prefix +
                  MakePatternForDigitRange(static_cast<char>(start_str[i] + 1), '9', len - i - 1)
              );
            }
          }
        }

        parts.push_back("[1-9]\\d{" + std::to_string(len) + ",}");
      }
    }
  }

  // Only end defined - match numbers <= end
  if (!start && end) {
    if (end.value() >= 0) {
      parts.push_back("-[1-9]\\d*");
      parts.push_back("0");
      if (end.value() > 0) {
        parts.push_back(GenerateSubRangeRegex(1, end.value()));
      }
    } else {
      std::string end_str = std::to_string(-end.value());
      int len = static_cast<int>(end_str.length());

      if (len == 1) {
        parts.push_back("-" + MakePatternForDigitRange(end_str[0], '9', 0));
        parts.push_back("-[1-9]\\d*");
      } else {
        parts.push_back(std::to_string(end.value()));  // Handle -123 exactly

        for (size_t i = 0; i < end_str.size(); i++) {
          if (i == 0) {
            if (end_str[0] > '1') {
              parts.push_back(
                  "-" + MakePatternForDigitRange('1', static_cast<char>(end_str[0] - 1), len - 1)
              );
            }
          } else {
            std::string prefix = end_str.substr(0, i);
            if (end_str[i] > '0') {
              parts.push_back(
                  "-" + prefix +
                  MakePatternForDigitRange('0', static_cast<char>(end_str[i] - 1), len - i - 1)
              );
            }
          }
        }

        parts.push_back("-[1-9]\\d{" + std::to_string(len) + ",}");
      }
    }
  }

  if (start && end) {
    int64_t range_start = start.value();
    int64_t range_end = end.value();

    if (range_start > range_end) {
      return "^()$";  // Invalid input
    }

    if (range_start < 0) {
      int64_t neg_start = range_start;
      int64_t neg_end = std::min(static_cast<int64_t>(-1), range_end);
      parts.push_back("-" + GenerateSubRangeRegex(-neg_end, -neg_start));
    }

    if (range_start <= 0 && range_end >= 0) {
      parts.push_back("0");
    }

    if (range_end > 0) {
      int64_t pos_start = std::max(static_cast<int64_t>(1), range_start);
      parts.push_back(GenerateSubRangeRegex(pos_start, range_end));
    }
  }

  result << "^(";
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      result << "|";
    }
    result << parts[i];
  }
  result << ")$";

  return result.str();
}

std::string JSONSchemaConverter::FormatFloat(double value, int precision = 6) {
  // Special handling for integer values to avoid float representation issues
  if (value == static_cast<int64_t>(value)) {
    return std::to_string(static_cast<int64_t>(value));
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  std::string result = oss.str();

  // Remove trailing zeros after decimal point
  size_t decimalPos = result.find('.');
  if (decimalPos != std::string::npos) {
    size_t lastNonZero = result.find_last_not_of('0');
    if (lastNonZero != std::string::npos && lastNonZero > decimalPos) {
      result.erase(lastNonZero + 1);
    } else if (lastNonZero == decimalPos) {
      result.erase(decimalPos);
    }
  }

  return result;
}

std::string JSONSchemaConverter::GenerateFloatRangeRegex(
    std::optional<double> start, std::optional<double> end, int precision = 6
) {
  if ((start && end) && (start.value() > end.value())) {
    return "^()$";  // Invalid input
  }

  if (!start && !end) {
    return "^-?\\d+(\\.\\d{1," + std::to_string(precision) + "})?$";
  }

  std::vector<std::string> parts;

  int64_t startInt = 0;
  int64_t endInt = 0;
  double startFrac = 0.0;
  double endFrac = 0.0;
  bool isStartNegative = false;
  bool isEndNegative = false;

  if (start) {
    isStartNegative = start.value() < 0;
    startInt = static_cast<int64_t>(floor(start.value()));
    startFrac = start.value() - startInt;
  }

  if (end) {
    isEndNegative = end.value() < 0;
    endInt = static_cast<int64_t>(floor(end.value()));
    endFrac = end.value() - endInt;
  }

  // Only start defined - match numbers >= start
  if (start && !end) {
    std::string startIntStr = FormatFloat(start.value(), precision);
    parts.push_back(startIntStr);

    // fractional parts > startFrac with same integer part (for positive)
    // fractional parts < startFrac with same integer part (for negative)
    if (startFrac > 0.0) {
      size_t dotPos = startIntStr.find('.');
      if (dotPos != std::string::npos) {
        std::string intPartStr = startIntStr.substr(0, dotPos);
        std::string fracPartStr = startIntStr.substr(dotPos + 1);

        if (!fracPartStr.empty()) {
          for (size_t i = 0; i < fracPartStr.length(); i++) {
            if (i == 0) {
              if (isStartNegative) {
                for (char d = '0'; d < fracPartStr[0]; d++) {
                  parts.push_back(
                      intPartStr + "\\." + d + "\\d{0," + std::to_string(precision - 1) + "}"
                  );
                }
              } else {
                for (char d = fracPartStr[0] + 1; d <= '9'; d++) {
                  parts.push_back(
                      intPartStr + "\\." + d + "\\d{0," + std::to_string(precision - 1) + "}"
                  );
                }
              }
            } else {
              std::string prefix = fracPartStr.substr(0, i);
              if (isStartNegative) {
                if (fracPartStr[i] > '0') {
                  for (char d = '0'; d < fracPartStr[i]; d++) {
                    parts.push_back(
                        intPartStr + "\\." + prefix + d + "\\d{0," +
                        std::to_string(precision - i - 1) + "}"
                    );
                  }
                }
              } else {
                for (char d = fracPartStr[i] + 1; d <= '9'; d++) {
                  parts.push_back(
                      intPartStr + "\\." + prefix + d + "\\d{0," +
                      std::to_string(precision - i - 1) + "}"
                  );
                }
              }
            }
          }
        }
      }
    }

    // For all integers > startInt
    if (startInt < INT64_MAX - 1) {
      std::string intRangeRegex = GenerateRangeRegex(startInt + 1, std::nullopt);
      intRangeRegex = intRangeRegex.substr(1, intRangeRegex.length() - 2);
      parts.push_back(intRangeRegex + "(\\.\\d{1," + std::to_string(precision) + "})?");
    }
  }

  // Only end defined - match numbers <= end
  else if (!start && end) {
    std::string endIntStr = FormatFloat(end.value(), precision);
    parts.push_back(endIntStr);

    // fractional parts < endFrac with same integer part (for positive)
    // fractional parts > endFrac with same integer part (for negative)
    if (endFrac > 0.0) {
      size_t dotPos = endIntStr.find('.');
      if (dotPos != std::string::npos) {
        std::string intPartStr = endIntStr.substr(0, dotPos);
        std::string fracPartStr = endIntStr.substr(dotPos + 1);

        if (!fracPartStr.empty()) {
          for (size_t i = 0; i < fracPartStr.length(); i++) {
            if (i == 0) {
              if (isEndNegative) {
                for (char d = fracPartStr[0] + 1; d <= '9'; d++) {
                  parts.push_back(
                      intPartStr + "\\." + d + "\\d{0," + std::to_string(precision - 1) + "}"
                  );
                }
              } else {
                for (char d = '0'; d < fracPartStr[0]; d++) {
                  parts.push_back(
                      intPartStr + "\\." + d + "\\d{0," + std::to_string(precision - 1) + "}"
                  );
                }
              }
            } else {
              if (isEndNegative) {
                std::string prefix = fracPartStr.substr(0, i);
                for (char d = fracPartStr[i] + 1; d <= '9'; d++) {
                  parts.push_back(
                      intPartStr + "\\." + prefix + d + "\\d{0," +
                      std::to_string(precision - i - 1) + "}"
                  );
                }
              } else if (fracPartStr[i] > '0') {
                std::string prefix = fracPartStr.substr(0, i);
                for (char d = '0'; d < fracPartStr[i]; d++) {
                  parts.push_back(
                      intPartStr + "\\." + prefix + d + "\\d{0," +
                      std::to_string(precision - i - 1) + "}"
                  );
                }
              }
            }
          }
        }
      }
    }

    // For all integers < endInt
    if (endInt > INT64_MIN + 1) {
      std::string intRangeRegex = GenerateRangeRegex(std::nullopt, endInt - 1);
      intRangeRegex = intRangeRegex.substr(1, intRangeRegex.length() - 2);
      parts.push_back(intRangeRegex + "(\\.\\d{1," + std::to_string(precision) + "})?");
    }
  }

  // start and end both defined
  else if (start && end) {
    // same integer part
    if (startInt == endInt) {
      if (startFrac == 0.0 && endFrac == 0.0) {
        parts.push_back(std::to_string(startInt));
      } else {
        std::string startStr = FormatFloat(start.value(), precision);
        parts.push_back(startStr);

        std::string endStr = FormatFloat(end.value(), precision);
        if (startStr != endStr) {
          parts.push_back(endStr);
        }

        if (startFrac < endFrac) {
          size_t startDotPos = startStr.find('.');
          size_t endDotPos = endStr.find('.');

          if (startDotPos != std::string::npos && endDotPos != std::string::npos) {
            std::string intPart = startStr.substr(0, startDotPos);
            std::string startFracPart = startStr.substr(startDotPos + 1);
            std::string endFracPart = endStr.substr(endDotPos + 1);

            size_t diffPos = 0;
            size_t minLength = std::min(startFracPart.length(), endFracPart.length());

            while (diffPos < minLength && startFracPart[diffPos] == endFracPart[diffPos]) {
              diffPos++;
            }

            if (diffPos < minLength) {
              char startDigit = startFracPart[diffPos];
              char endDigit = endFracPart[diffPos];

              if (endDigit > startDigit + 1) {
                std::string prefix = startFracPart.substr(0, diffPos);
                for (char d = startDigit + 1; d < endDigit; d++) {
                  parts.push_back(
                      intPart + "\\." + prefix + d + "\\d{0," +
                      std::to_string(precision - diffPos - 1) + "}"
                  );
                }
              }

              if (diffPos + 1 < startFracPart.length()) {
                std::string prefix = startFracPart.substr(0, diffPos + 1);

                for (size_t i = diffPos + 1; i < startFracPart.length(); i++) {
                  std::string currentPrefix = startFracPart.substr(0, i);
                  char currentDigit = startFracPart[i];

                  for (char d = currentDigit + 1; d <= '9'; d++) {
                    parts.push_back(
                        intPart + "\\." + currentPrefix + d + "\\d{0," +
                        std::to_string(precision - i - 1) + "}"
                    );
                  }
                }
              }

              if (diffPos + 1 < endFracPart.length()) {
                std::string prefix = endFracPart.substr(0, diffPos + 1);

                for (size_t i = diffPos + 1; i < endFracPart.length(); i++) {
                  if (endFracPart[i] > '0') {
                    std::string currentPrefix = endFracPart.substr(0, i);
                    char currentDigit = endFracPart[i];

                    for (char d = '0'; d < currentDigit; d++) {
                      parts.push_back(
                          intPart + "\\." + currentPrefix + d + "\\d{0," +
                          std::to_string(precision - i - 1) + "}"
                      );
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    // Different integer parts
    else {
      std::string startStr = FormatFloat(start.value(), precision);
      parts.push_back(startStr);

      std::string endStr = FormatFloat(end.value(), precision);
      if (startStr != endStr) {
        parts.push_back(endStr);
      }

      if (endInt > startInt + 1) {
        std::string intRangeRegex = GenerateRangeRegex(startInt + 1, endInt - 1);
        intRangeRegex = intRangeRegex.substr(1, intRangeRegex.length() - 2);
        parts.push_back(intRangeRegex + "(\\.\\d{1," + std::to_string(precision) + "})?");
      }

      if (startFrac > 0.0) {
        size_t dotPos = startStr.find('.');
        if (dotPos != std::string::npos) {
          std::string intPartStr = startStr.substr(0, dotPos);
          std::string fracPartStr = startStr.substr(dotPos + 1);

          if (!fracPartStr.empty()) {
            for (size_t i = 0; i < fracPartStr.length(); i++) {
              if (i == 0) {
                if (isStartNegative) {
                  for (char d = '0'; d < fracPartStr[0]; d++) {
                    parts.push_back(
                        intPartStr + "\\." + d + "\\d{0," + std::to_string(precision - 1) + "}"
                    );
                  }
                } else {
                  for (char d = fracPartStr[0] + 1; d <= '9'; d++) {
                    parts.push_back(
                        intPartStr + "\\." + d + "\\d{0," + std::to_string(precision - 1) + "}"
                    );
                  }
                }
              } else {
                std::string prefix = fracPartStr.substr(0, i);
                if (isStartNegative) {
                  if (fracPartStr[i] > '0') {
                    for (char d = '0'; d < fracPartStr[i]; d++) {
                      parts.push_back(
                          intPartStr + "\\." + prefix + d + "\\d{0," +
                          std::to_string(precision - i - 1) + "}"
                      );
                    }
                  }
                } else {
                  for (char d = fracPartStr[i] + 1; d <= '9'; d++) {
                    parts.push_back(
                        intPartStr + "\\." + prefix + d + "\\d{0," +
                        std::to_string(precision - i - 1) + "}"
                    );
                  }
                }
              }
            }
          }
        }
      } else {
        parts.push_back(std::to_string(startInt) + "\\.\\d{1," + std::to_string(precision) + "}");
      }

      if (endFrac > 0.0) {
        size_t dotPos = endStr.find('.');
        if (dotPos != std::string::npos) {
          std::string intPartStr = endStr.substr(0, dotPos);
          std::string fracPartStr = endStr.substr(dotPos + 1);

          if (!fracPartStr.empty()) {
            for (size_t i = 0; i < fracPartStr.length(); i++) {
              if (i == 0) {
                if (isEndNegative) {
                  for (char d = fracPartStr[0] + 1; d <= '9'; d++) {
                    parts.push_back(
                        intPartStr + "\\." + d + "\\d{0," + std::to_string(precision - 1) + "}"
                    );
                  }
                } else {
                  for (char d = '0'; d < fracPartStr[0]; d++) {
                    parts.push_back(
                        intPartStr + "\\." + d + "\\d{0," + std::to_string(precision - 1) + "}"
                    );
                  }
                }
              } else {
                if (isEndNegative) {
                  std::string prefix = fracPartStr.substr(0, i);
                  for (char d = fracPartStr[i] + 1; d <= '9'; d++) {
                    parts.push_back(
                        intPartStr + "\\." + prefix + d + "\\d{0," +
                        std::to_string(precision - i - 1) + "}"
                    );
                  }
                } else if (fracPartStr[i] > '0') {
                  std::string prefix = fracPartStr.substr(0, i);
                  for (char d = '0'; d < fracPartStr[i]; d++) {
                    parts.push_back(
                        intPartStr + "\\." + prefix + d + "\\d{0," +
                        std::to_string(precision - i - 1) + "}"
                    );
                  }
                }
              }
            }
          }
        }
      } else {
        parts.push_back(std::to_string(endInt) + "\\.\\d{1," + std::to_string(precision) + "}");
      }
    }
  }

  std::ostringstream result;
  result << "^(";
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      result << "|";
    }
    result << parts[i];
  }
  result << ")$";

  return result.str();
}

std::string JSONSchemaConverter::VisitInteger(
    const picojson::object& schema, const std::string& rule_name
) {
  XGRAMMAR_CHECK(schema.count("type"));
  XGRAMMAR_CHECK(schema.at("type").get<std::string>() == "integer");
  WarnUnsupportedKeywords(
      schema,
      {
          "multipleOf",
      }
  );

  auto checkAndConvertIntegerBound = [](const picojson::value& value) -> int64_t {
    XGRAMMAR_CHECK(value.is<int64_t>() || value.is<double>()) << "Value must be a number";

    if (value.is<int64_t>()) {
      return value.get<int64_t>();
    } else {
      double val = value.get<double>();

      XGRAMMAR_CHECK(val == std::floor(val)) << "Integer constraint must be a whole number";

      static const double PROBLEMATIC_MIN = -9223372036854776000.0;
      static const double PROBLEMATIC_MAX = 9223372036854776000.0;

      if (val == PROBLEMATIC_MIN) {
        XGRAMMAR_CHECK(false
        ) << "Integer exceeds minimum limit due to precision loss at 64-bit boundary";
      }

      if (val == PROBLEMATIC_MAX) {
        XGRAMMAR_CHECK(false
        ) << "Integer exceeds maximum limit due to precision loss at 64-bit boundary";
      }

      static const double MAX_INT64_AS_DOUBLE =
          static_cast<double>(std::numeric_limits<int64_t>::max());
      static const double MIN_INT64_AS_DOUBLE =
          static_cast<double>(std::numeric_limits<int64_t>::min());

      XGRAMMAR_CHECK(val <= MAX_INT64_AS_DOUBLE) << "Integer exceeds maximum limit";
      XGRAMMAR_CHECK(val >= MIN_INT64_AS_DOUBLE) << "Integer exceeds minimum limit";

      return static_cast<int64_t>(val);
    }
  };

  std::string range_regex = "";
  if (schema.count("minimum") || schema.count("maximum") || schema.count("exclusiveMinimum") ||
      schema.count("exclusiveMaximum")) {
    std::optional<int64_t> start, end;
    if (schema.count("minimum")) {
      start = checkAndConvertIntegerBound(schema.at("minimum"));
    }
    if (schema.count("exclusiveMinimum")) {
      int64_t exclusive_min = checkAndConvertIntegerBound(schema.at("exclusiveMinimum"));
      XGRAMMAR_CHECK(exclusive_min != std::numeric_limits<int64_t>::max())
          << "exclusiveMinimum would cause integer overflow";
      start = exclusive_min + 1;
    }
    if (schema.count("maximum")) {
      end = checkAndConvertIntegerBound(schema.at("maximum"));
    }
    if (schema.count("exclusiveMaximum")) {
      int64_t exclusive_max = checkAndConvertIntegerBound(schema.at("exclusiveMaximum"));
      XGRAMMAR_CHECK(exclusive_max != std::numeric_limits<int64_t>::min())
          << "exclusiveMaximum would cause integer underflow";
      end = exclusive_max - 1;
    }
    XGRAMMAR_CHECK(!(start && end) || *start <= *end)
        << "Invalid range: minimum greater than maximum";
    range_regex = GenerateRangeRegex(start, end);
  }

  if (!range_regex.empty()) {
    std::string converted_regex = RegexToEBNF(range_regex, false);
    return converted_regex;  // not " " for numbers
  }
  return "(\"0\" | \"-\"? [1-9] [0-9]*)";
}

std::string JSONSchemaConverter::VisitNumber(
    const picojson::object& schema, const std::string& rule_name
) {
  XGRAMMAR_CHECK(schema.count("type"));
  XGRAMMAR_CHECK(schema.at("type").get<std::string>() == "number");
  WarnUnsupportedKeywords(
      schema,
      {
          "multipleOf",
      }
  );

  std::string range_regex = "";
  if (schema.count("minimum") || schema.count("maximum") || schema.count("exclusiveMinimum") ||
      schema.count("exclusiveMaximum")) {
    std::optional<double> start, end;
    if (schema.count("minimum")) {
      XGRAMMAR_CHECK(schema.at("minimum").is<double>() || schema.at("minimum").is<int64_t>())
          << "minimum must be a number";
      start = schema.at("minimum").get<double>();
    }
    if (schema.count("exclusiveMinimum")) {
      XGRAMMAR_CHECK(
          schema.at("exclusiveMinimum").is<double>() || schema.at("exclusiveMinimum").is<int64_t>()
      ) << "exclusiveMinimum must be a number";
      double exclusive_min = schema.at("exclusiveMinimum").get<double>();
      // For exclusive minimum with floats, we can't easily add 1, so we'll handle that
      // in the regex generation if needed
      start = exclusive_min;
    }
    if (schema.count("maximum")) {
      XGRAMMAR_CHECK(schema.at("maximum").is<double>() || schema.at("maximum").is<int64_t>())
          << "maximum must be a number";
      end = schema.at("maximum").get<double>();
    }
    if (schema.count("exclusiveMaximum")) {
      XGRAMMAR_CHECK(
          schema.at("exclusiveMaximum").is<double>() || schema.at("exclusiveMaximum").is<int64_t>()
      ) << "exclusiveMaximum must be a number";
      double exclusive_max = schema.at("exclusiveMaximum").get<double>();
      // For exclusive maximum with floats, we can't easily subtract 1, so we'll handle that
      // in the regex generation if needed
      end = exclusive_max;
    }
    XGRAMMAR_CHECK(!(start && end) || *start <= *end)
        << "Invalid range, start value greater than end value";
    range_regex = GenerateFloatRangeRegex(start, end);
  }

  if (!range_regex.empty()) {
    std::string converted_regex = RegexToEBNF(range_regex, false);
    return converted_regex;
  }

  return "(\"0\" | \"-\"? [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?";
}

std::string JSONSchemaConverter::VisitString(
    const picojson::object& schema, const std::string& rule_name, JSONFormat json_format
) {
  XGRAMMAR_CHECK(schema.count("type"));
  XGRAMMAR_CHECK(schema.at("type").get<std::string>() == "string");
  auto string_spec_result = ParseStringSchema(schema, json_format);
  if (string_spec_result.IsErr()) {
    XGRAMMAR_LOG(FATAL) << std::move(string_spec_result).UnwrapErr().what();
  }
  auto string_spec = std::move(string_spec_result).Unwrap();
  std::string result;
  if (!string_spec.wrapper.first.empty()) {
    result += "\"" + string_spec.wrapper.first + "\" ";
  }
  result += string_spec.pattern;
  if (string_spec.min_length != 0 || string_spec.max_length != -1) {
    std::string repetition_range;
    repetition_range +=
        "{" + std::to_string(string_spec.min_length) + "," +
        (string_spec.max_length == -1 ? "" : std::to_string(string_spec.max_length)) + "}";
    result += repetition_range;
  }
  if (!string_spec.wrapper.second.empty()) {
    result += " \"" + string_spec.wrapper.second + "\"";
  }
  return result;
}

std::string JSONSchemaConverter::VisitBoolean(
    const picojson::object& schema, const std::string& rule_name
) {
  XGRAMMAR_CHECK(schema.count("type"));
  XGRAMMAR_CHECK(schema.at("type").get<std::string>() == "boolean");
  return "\"true\" | \"false\"";
}

std::string JSONSchemaConverter::VisitNull(
    const picojson::object& schema, const std::string& rule_name
) {
  XGRAMMAR_CHECK(schema.count("type"));
  XGRAMMAR_CHECK(schema.at("type").get<std::string>() == "null");
  return "\"null\"";
}

Result<JSONSchemaConverter::ArraySpec, SchemaError> JSONSchemaConverter::ParseArraySchema(
    const picojson::object& schema
) {
  XGRAMMAR_DCHECK(
      (schema.count("type") && schema.at("type").get<std::string>() == "array") ||
      schema.count("prefixItems") || schema.count("items") || schema.count("unevaluatedItems")
  );
  WarnUnsupportedKeywords(schema, {"uniqueItems", "contains", "minContains", "maxContains"});

  std::vector<picojson::value> prefix_item_schemas;
  bool allow_additional_items = true;
  picojson::value additional_item_schema;
  int64_t min_items = 0;
  int64_t max_items = -1;

  if (schema.count("prefixItems")) {
    if (!schema.at("prefixItems").is<picojson::array>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "prefixItems must be an array"
      );
    }
    prefix_item_schemas = schema.at("prefixItems").get<picojson::array>();
    for (const auto& item : prefix_item_schemas) {
      if (item.is<bool>()) {
        if (!item.get<bool>()) {
          return ResultErr<SchemaError>(
              SchemaErrorType::kUnsatisfiableSchema, "prefixItems contains false"
          );
        }
      } else if (!item.is<picojson::object>()) {
        return ResultErr<SchemaError>(
            SchemaErrorType::kInvalidSchema, "prefixItems must be an array of objects or booleans"
        );
      }
    }
  }

  if (schema.count("items")) {
    auto items_value = schema.at("items");
    if (!items_value.is<bool>() && !items_value.is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "items must be a boolean or an object"
      );
    }
    if (items_value.is<bool>() && !items_value.get<bool>()) {
      allow_additional_items = false;
    } else {
      allow_additional_items = true;
      additional_item_schema = items_value;
    }
  } else if (schema.count("unevaluatedItems")) {
    auto unevaluated_items_value = schema.at("unevaluatedItems");
    if (!unevaluated_items_value.is<bool>() && !unevaluated_items_value.is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "unevaluatedItems must be a boolean or an object"
      );
    }
    if (unevaluated_items_value.is<bool>() && !unevaluated_items_value.get<bool>()) {
      allow_additional_items = false;
    } else {
      allow_additional_items = true;
      additional_item_schema = unevaluated_items_value;
    }
  } else if (!strict_mode_) {
    allow_additional_items = true;
    additional_item_schema = picojson::value(true);
  } else {
    allow_additional_items = false;
  }

  if (schema.count("minItems")) {
    if (!schema.at("minItems").is<int64_t>()) {
      return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "minItems must be an integer");
    }
    min_items = std::max(static_cast<int64_t>(0), schema.at("minItems").get<int64_t>());
  }

  if (schema.count("minContains")) {
    if (!schema.at("minContains").is<int64_t>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "minContains must be an integer"
      );
    }
    min_items = std::max(min_items, schema.at("minContains").get<int64_t>());
  }

  if (schema.count("maxItems")) {
    if (!schema.at("maxItems").is<int64_t>() || schema.at("maxItems").get<int64_t>() < 0) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "maxItems must be a non-negative integer"
      );
    }
    max_items = schema.at("maxItems").get<int64_t>();
  }

  // Check if the schema is unsatisfiable
  if (max_items != -1 && min_items > max_items) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "minItems is greater than maxItems: " + std::to_string(min_items) + " > " +
            std::to_string(max_items)
    );
  }

  if (max_items != -1 && max_items < static_cast<int64_t>(prefix_item_schemas.size())) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "maxItems is less than the number of prefixItems: " + std::to_string(max_items) + " < " +
            std::to_string(prefix_item_schemas.size())
    );
  }

  if (!allow_additional_items) {
    // [len, len] must be in [min, max]
    if (static_cast<int64_t>(prefix_item_schemas.size()) < min_items) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema,
          "minItems is greater than the number of prefixItems, but additional items are not "
          "allowed: " +
              std::to_string(min_items) + " > " + std::to_string(prefix_item_schemas.size())
      );
    }
    if (max_items != -1 && static_cast<int64_t>(prefix_item_schemas.size()) > max_items) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema,
          "maxItems is less than the number of prefixItems, but additional items are not "
          "allowed: " +
              std::to_string(max_items) + " < " + std::to_string(prefix_item_schemas.size())
      );
    }
  }

  return ResultOk(ArraySpec{
      prefix_item_schemas, allow_additional_items, additional_item_schema, min_items, max_items
  });
}

std::string JSONSchemaConverter::VisitArray(
    const picojson::object& schema, const std::string& rule_name
) {
  auto array_spec_result = ParseArraySchema(schema);
  if (array_spec_result.IsErr()) {
    XGRAMMAR_LOG(FATAL) << std::move(array_spec_result).UnwrapErr().what();
  }

  auto array_spec = std::move(array_spec_result).Unwrap();

  indentManager_->StartIndent();

  auto start_separator = indentManager_->StartSeparator();
  auto mid_separator = indentManager_->MiddleSeparator();
  auto end_separator = indentManager_->EndSeparator();
  auto empty_separator = indentManager_->EmptySeparator();

  std::vector<std::string> item_rule_names;
  std::string additional_rule_name;

  // 1. Handle prefix items
  if (array_spec.prefix_item_schemas.size() > 0) {
    for (int64_t i = 0; i < static_cast<int64_t>(array_spec.prefix_item_schemas.size()); ++i) {
      XGRAMMAR_DCHECK(
          array_spec.prefix_item_schemas[i].is<picojson::object>() ||
          array_spec.prefix_item_schemas[i].is<bool>()
      );
      item_rule_names.push_back(CreateRuleFromSchema(
          array_spec.prefix_item_schemas[i], rule_name + "_item_" + std::to_string(i)
      ));
    }
  }

  // 2. Handle additional items
  if (array_spec.allow_additional_items) {
    additional_rule_name =
        CreateRuleFromSchema(array_spec.additional_item_schema, rule_name + "_additional");
  }

  indentManager_->EndIndent();

  // 3. Construct the result with given format
  // clang-format off
   /*
    * prefix empty, additional items not allowed: [empty_separator]
    * prefix empty, additional items allowed:
    *   if min == 0, max == 0:
    *     [empty_separator]
    *   if min == 0, max > 0:
    *     ([start_separator additional_rule_name (mid_separator additional_rule_name){0, max - 1}) end_separator] | [empty_separator]
    *   if min > 0:
    *     ([start_separator additional_rule_name (mid_separator additional_rule_name){min - 1, max - 1}) end_separator]
    * prefix non-empty, additional items not allowed: [start_separator item0 mid_separator item1 end_separator]
    * prefix non-empty, additional items allowed:
    *   [start_separator item0 mid_separator item1 (mid_separator additional_rule_name){max(0, min - len(prefix)), max - len(prefix)} end_separator]
    */
  // clang-format on
  std::string result;
  const std::string& left_bracket = EBNFScriptCreator::Str("[");
  const std::string& right_bracket = EBNFScriptCreator::Str("]");

  if (array_spec.prefix_item_schemas.empty()) {
    auto empty_part = EBNFScriptCreator::Concat({left_bracket, empty_separator, right_bracket});
    if (!array_spec.allow_additional_items) {
      return empty_part;
    } else if (array_spec.min_items == 0 && array_spec.max_items == 0) {
      return empty_part;
    } else if (array_spec.min_items == 0 && array_spec.max_items != 0) {
      return EBNFScriptCreator::Or(
          {EBNFScriptCreator::Concat(
               {left_bracket,
                start_separator,
                additional_rule_name,
                EBNFScriptCreator::Repeat(
                    EBNFScriptCreator::Concat({mid_separator, additional_rule_name}),
                    0,
                    array_spec.max_items == -1 ? -1 : array_spec.max_items - 1
                ),
                end_separator,
                right_bracket}
           ),
           empty_part}
      );
    } else {
      XGRAMMAR_DCHECK(array_spec.min_items > 0);
      return EBNFScriptCreator::Concat(
          {left_bracket,
           start_separator,
           additional_rule_name,
           EBNFScriptCreator::Repeat(
               EBNFScriptCreator::Concat({mid_separator, additional_rule_name}),
               array_spec.min_items - 1,
               array_spec.max_items == -1 ? -1 : array_spec.max_items - 1
           ),
           end_separator,
           right_bracket}
      );
    }
  } else {
    std::vector<std::string> prefix_part;
    for (int64_t i = 0; i < static_cast<int64_t>(item_rule_names.size()); ++i) {
      if (i > 0) {
        prefix_part.push_back(mid_separator);
      }
      prefix_part.push_back(item_rule_names[i]);
    }
    auto prefix_part_str = EBNFScriptCreator::Concat(prefix_part);
    if (!array_spec.allow_additional_items) {
      return EBNFScriptCreator::Concat(
          {left_bracket, start_separator, prefix_part_str, end_separator, right_bracket}
      );
    } else {
      int64_t min_items = std::max(
          static_cast<int64_t>(0),
          array_spec.min_items - static_cast<int64_t>(item_rule_names.size())
      );
      return EBNFScriptCreator::Concat(
          {left_bracket,
           start_separator,
           prefix_part_str,
           EBNFScriptCreator::Repeat(
               EBNFScriptCreator::Concat({mid_separator, additional_rule_name}),
               min_items,
               array_spec.max_items == -1
                   ? -1
                   : array_spec.max_items - static_cast<int64_t>(item_rule_names.size())
           ),
           end_separator,
           right_bracket}
      );
    }
  }
}

std::string JSONSchemaConverter::GetPropertyPattern(
    const std::string& prop_name,
    const picojson::value& prop_schema,
    const std::string& rule_name,
    int64_t idx,  // Changed to int64_t
    const JSONFormat json_format
) {
  // the outer quote is for the string in EBNF grammar, and the inner quote is for
  // the string in JSON

  std::string key;
  switch (json_format) {
    case JSONFormat::kJSON: {
      key += "\"\\\"" + prop_name + "\\\"\"";
      break;
    }
    case JSONFormat::kXML: {
      key += "\"<parameter=" + prop_name + ">\"";
      break;
    }
  }
  std::string value =
      CreateRuleFromSchema(prop_schema, rule_name + "_prop_" + std::to_string(idx), json_format);
  switch (json_format) {
    case JSONFormat::kJSON: {
      return key + " " + colon_pattern_ + " " + value;
    }
    case JSONFormat::kXML: {
      return key + " " + kWhiteSpace + " " + value + " " + kWhiteSpace + " \"</parameter>\"";
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "Unsupported string escape type: " << static_cast<int>(json_format);
      return "";
    }
  }
}

std::string JSONSchemaConverter::GetOtherPropertyPattern(
    const std::string& key_pattern,
    const picojson::value& prop_schema,
    const std::string& rule_name,
    const std::string& rule_name_suffix,
    const JSONFormat json_format
) {
  std::string value =
      CreateRuleFromSchema(prop_schema, rule_name + "_" + rule_name_suffix, json_format);
  switch (json_format) {
    case (JSONFormat::kJSON): {
      return key_pattern + " " + colon_pattern_ + " " + value;
    }
    case (JSONFormat::kXML): {
      return "\"<parameter=\" " + key_pattern + " \">\" " + kWhiteSpace + " " + value + " " +
             kWhiteSpace + " \"</parameter>\"";
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "Unsupported string escape type: " << static_cast<int>(json_format);
      return "";
    }
  }
}

std::string JSONSchemaConverter::GetPropertyWithNumberConstrains(
    const std::string& pattern, int min_properties, int max_properties, int already_repeated_times
) {
  XGRAMMAR_DCHECK(max_properties >= already_repeated_times || max_properties == -1);
  if (max_properties == already_repeated_times) {
    return "\"\"";
  }
  int lower = std::max(0, min_properties - already_repeated_times);
  int upper = std::max(-1, max_properties - already_repeated_times);
  if (lower == 0 && upper == -1) {
    return "(" + pattern + ")*";
  } else if (lower == 0 && upper == 1) {
    return "(" + pattern + ")?";
  } else if (lower == 1 && upper == 1) {
    return pattern;
  } else {
    return "(" + pattern + "){" + std::to_string(lower) + "," +
           (max_properties == -1 ? "" : std::to_string(upper)) + "} ";
  }
}

std::string JSONSchemaConverter::GetPartialRuleForProperties(
    const std::vector<std::pair<std::string, picojson::value>>& properties,
    const std::unordered_set<std::string>& required,
    const picojson::value& additional,
    const std::string& rule_name,
    const std::string& additional_suffix,
    const int min_properties,
    const int max_properties,
    const JSONFormat json_format
) {
  // return empty when maxProperties=0
  if (max_properties == 0) {
    return "";
  }

  std::string first_sep;
  std::string mid_sep;
  std::string last_sep;
  switch (json_format) {
    case (JSONFormat::kJSON): {
      first_sep = NextSeparator();
      mid_sep = NextSeparator();
      last_sep = NextSeparator(true);
      break;
    }
    case (JSONFormat::kXML): {
      first_sep = kWhiteSpace;
      mid_sep = kWhiteSpace;
      last_sep = "";
      break;
    }
  }

  std::string res = "";

  std::vector<std::string> prop_patterns;
  int64_t idx = 0;  // Changed to int64_t
  for (const auto& [prop_name, prop_schema] : properties) {
    prop_patterns.push_back(GetPropertyPattern(prop_name, prop_schema, rule_name, idx, json_format)
    );
    ++idx;
  }

  if (min_properties == 0 && max_properties == -1) {
    // Case 1. Without any properties number constrains
    std::vector<std::string> rule_names(properties.size(), "");
    std::vector<uint8_t> is_required(properties.size(), false);
    bool allow_additional =
        !additional.is<picojson::null>() && (!additional.is<bool>() || additional.get<bool>());

    // construct the last rule
    std::string additional_prop_pattern;
    if (allow_additional) {
      switch (json_format) {
        case (JSONFormat::kJSON): {
          additional_prop_pattern =
              GetOtherPropertyPattern(kBasicString, additional, rule_name, additional_suffix);
          break;
        }
        case (JSONFormat::kXML): {
          additional_prop_pattern = GetOtherPropertyPattern(
              kXMLVariableName, additional, rule_name, additional_suffix, JSONFormat::kXML
          );
          break;
        }
      }
      std::string last_rule_body = "(" + mid_sep + " " + additional_prop_pattern + ")*";
      std::string last_rule_name =
          rule_name + "_part_" + std::to_string(static_cast<int>(properties.size()) - 1);
      last_rule_name = ebnf_script_creator_.AddRule(last_rule_name, last_rule_body);
      rule_names.back() = last_rule_name;
    } else {
      rule_names.back() = "\"\"";
    }

    // construct 0~(len(properties) - 2) rules
    for (int i = properties.size() - 2; i >= 0; --i) {
      const std::string& prop_pattern = prop_patterns[i + 1];
      const std::string& last_rule_name = rule_names[i + 1];
      std::string cur_rule_body = mid_sep + " " + prop_pattern + " " + last_rule_name;
      if (!required.count(properties[i + 1].first)) {
        cur_rule_body = last_rule_name + " | " + cur_rule_body;
      } else {
        is_required[i + 1] = true;
      }
      std::string cur_rule_name = rule_name + "_part_" + std::to_string(i);
      cur_rule_name = ebnf_script_creator_.AddRule(cur_rule_name, cur_rule_body);
      rule_names[i] = cur_rule_name;
    }
    if (required.count(properties[0].first)) {
      is_required[0] = true;
    }

    // construct the root rule
    for (int i = 0; i < static_cast<int>(properties.size()); ++i) {
      if (i != 0) {
        res += " | ";
      }
      res += "(" + prop_patterns[i] + " " + rule_names[i] + ")";
      if (is_required[i]) {
        break;
      }
    }

    if (allow_additional && required.empty()) {
      res += " | " + additional_prop_pattern + " " + rule_names.back();
    }

    // add separators and the empty string option
    res = first_sep + " (" + res + ") " + last_sep;
  } else if (max_properties == -1) {
    // Case 2. With constrain on the lower bound of the properties number
    int properties_size = static_cast<int>(properties.size());
    std::vector<std::vector<std::string>> rule_names(properties_size, std::vector<std::string>());
    std::vector<int> key_matched_min(properties_size, 0);
    std::vector<uint8_t> is_required(properties_size, false);

    bool allow_additional =
        !additional.is<picojson::null>() && (!additional.is<bool>() || additional.get<bool>());

    // get the range of matched properties for each rule
    bool get_first_required = required.count(properties[0].first);
    key_matched_min[0] = 1;
    for (int i = 1; i < properties_size; ++i) {
      if (required.count(properties[i].first)) {
        is_required[i] = true;
        key_matched_min[i] = key_matched_min[i - 1] + 1;
      } else {
        key_matched_min[i] = key_matched_min[i - 1];
      }
      if (!get_first_required) {
        key_matched_min[i] = 1;
      }
      if (is_required[i]) {
        get_first_required = true;
      }
    }
    if (required.count(properties[0].first)) {
      is_required[0] = true;
    }
    if (allow_additional) {
      key_matched_min.back() = std::max(1, key_matched_min.back());
    } else {
      key_matched_min.back() = std::max(min_properties, key_matched_min.back());
    }
    for (int i = properties_size - 2; i >= 0; --i) {
      key_matched_min[i] = std::max(key_matched_min[i], key_matched_min[i + 1] - 1);
    }

    // construct the last rule
    std::string additional_prop_pattern;
    if (allow_additional) {
      switch (json_format) {
        case (JSONFormat::kJSON): {
          additional_prop_pattern =
              GetOtherPropertyPattern(kBasicString, additional, rule_name, additional_suffix);
          break;
        }
        case (JSONFormat::kXML): {
          additional_prop_pattern = GetOtherPropertyPattern(
              kXMLVariableName, additional, rule_name, additional_suffix, JSONFormat::kXML
          );
          break;
        }
      }
      for (int matched = key_matched_min.back(); matched <= properties_size; ++matched) {
        std::string last_rule_body;
        switch (json_format) {
          case (JSONFormat::kJSON): {
            last_rule_body = GetPropertyWithNumberConstrains(
                mid_sep + " " + additional_prop_pattern, min_properties, max_properties, matched
            );
            break;
          }
          case (JSONFormat::kXML): {
            last_rule_body = GetPropertyWithNumberConstrains(
                additional_prop_pattern, min_properties, max_properties, matched
            );
            break;
          }
        }
        std::string last_rule_name = rule_name + "_part_" +
                                     std::to_string(static_cast<int>(properties.size()) - 1) + "_" +
                                     std::to_string(matched);
        last_rule_name = ebnf_script_creator_.AddRule(last_rule_name, last_rule_body);
        rule_names.back().push_back(last_rule_name);
      }
    } else {
      for (int matched = key_matched_min.back(); matched <= properties_size; ++matched) {
        rule_names.back().push_back("\"\"");
      }
    }

    // construct 0~(len(properties) - 2) rules

    for (int i = properties_size - 2; i >= 0; --i) {
      const std::string& prop_pattern = prop_patterns[i + 1];
      for (int matched = key_matched_min[i]; matched <= i + 1; ++matched) {
        std::string cur_rule_body = "";
        if (is_required[i + 1] || matched == key_matched_min[i + 1] - 1) {
          cur_rule_body = mid_sep + " " + prop_pattern + " " +
                          rule_names[i + 1][matched + 1 - key_matched_min[i + 1]];
        } else {
          cur_rule_body = rule_names[i + 1][matched - key_matched_min[i + 1]] + " | " + mid_sep +
                          " " + prop_pattern + " " +
                          rule_names[i + 1][matched - key_matched_min[i + 1] + 1];
        }
        std::string cur_rule_name =
            rule_name + "_part_" + std::to_string(i) + "_" + std::to_string(matched);
        cur_rule_name = ebnf_script_creator_.AddRule(cur_rule_name, cur_rule_body);
        rule_names[i].push_back(cur_rule_name);
      }
    }

    // construct the root rule
    bool is_first = true;
    for (int i = 0; i < static_cast<int>(properties.size()); ++i) {
      if (key_matched_min[i] > 1) {
        break;
      }
      if (!is_first) {
        res += " | ";
      } else {
        is_first = false;
      }
      res += "(" + prop_patterns[i] + " " + rule_names[i][1 - key_matched_min[i]] + ")";
      if (is_required[i]) {
        break;
      }
    }

    if (allow_additional && required.empty()) {
      if (!is_first) {
        res += " | ";
      }
      switch (json_format) {
        case (JSONFormat::kJSON): {
          res += "(" + additional_prop_pattern + " " +
                 GetPropertyWithNumberConstrains(
                     mid_sep + " " + additional_prop_pattern, min_properties, max_properties, 1
                 ) +
                 ")";
          break;
        }
        case (JSONFormat::kXML): {
          res += "(" + additional_prop_pattern + " " +
                 GetPropertyWithNumberConstrains(
                     additional_prop_pattern, min_properties, max_properties, 1
                 ) +
                 ")";
          break;
        }
      }
    }

    // add separators and the empty string option
    res = first_sep + " (" + res + ") " + last_sep;
  } else {
    // Case 3. With constrains on the both lower & upper bound of the properties number
    int properties_size = static_cast<int>(properties.size());
    std::vector<std::vector<std::string>> rule_names(properties_size, std::vector<std::string>());
    std::vector<int> key_matched_min(properties_size, 0);
    std::vector<int> key_matched_max(properties_size, properties_size);
    std::vector<uint8_t> is_required(properties_size, false);

    bool allow_additional =
        !additional.is<picojson::null>() && (!additional.is<bool>() || additional.get<bool>());

    // get the range of matched properties for each rule
    bool get_first_required = required.count(properties[0].first);
    key_matched_min[0] = 1;
    key_matched_max[0] = 1;
    for (int i = 1; i < properties_size; ++i) {
      if (required.count(properties[i].first)) {
        is_required[i] = true;
        key_matched_min[i] = key_matched_min[i - 1] + 1;
      } else {
        key_matched_min[i] = key_matched_min[i - 1];
      }
      if (!get_first_required) {
        key_matched_min[i] = 1;
      }
      key_matched_max[i] = key_matched_max[i - 1] + 1;
      if (is_required[i]) {
        get_first_required = true;
      }
    }
    if (required.count(properties[0].first)) {
      is_required[0] = true;
    }
    if (allow_additional) {
      key_matched_min.back() = std::max(1, key_matched_min.back());
      key_matched_max.back() = std::min(max_properties, key_matched_max.back());
    } else {
      XGRAMMAR_DCHECK(
          key_matched_min.back() <= max_properties && key_matched_max.back() >= min_properties
      );
      key_matched_min.back() = std::max(min_properties, key_matched_min.back());
      key_matched_max.back() = std::min(max_properties, key_matched_max.back());
    }
    for (int i = properties_size - 2; i >= 0; --i) {
      key_matched_min[i] = std::max(key_matched_min[i], key_matched_min[i + 1] - 1);
      if (is_required[i + 1]) {
        key_matched_max[i] = std::min(key_matched_max[i], key_matched_max[i + 1] - 1);
      } else {
        key_matched_max[i] = std::min(key_matched_max[i], key_matched_max[i + 1]);
      }
    }

    // construct the last rule
    std::string additional_prop_pattern;
    if (allow_additional) {
      switch (json_format) {
        case (JSONFormat::kJSON): {
          additional_prop_pattern =
              GetOtherPropertyPattern(kBasicString, additional, rule_name, additional_suffix);
          break;
        }
        case (JSONFormat::kXML): {
          additional_prop_pattern = GetOtherPropertyPattern(
              kXMLVariableName, additional, rule_name, additional_suffix, JSONFormat::kXML
          );
          break;
        }
      }
      for (int matched = key_matched_min.back(); matched <= key_matched_max.back(); ++matched) {
        std::string last_rule_body;
        switch (json_format) {
          case (JSONFormat::kJSON): {
            last_rule_body = GetPropertyWithNumberConstrains(
                mid_sep + " " + additional_prop_pattern, min_properties, max_properties, matched
            );
            break;
          }
          case (JSONFormat::kXML): {
            last_rule_body = GetPropertyWithNumberConstrains(
                additional_prop_pattern, min_properties, max_properties, matched
            );
            break;
          }
        }
        std::string last_rule_name = rule_name + "_part_" +
                                     std::to_string(static_cast<int>(properties.size()) - 1) + "_" +
                                     std::to_string(matched);
        last_rule_name = ebnf_script_creator_.AddRule(last_rule_name, last_rule_body);
        rule_names.back().push_back(last_rule_name);
      }
    } else {
      for (int matched = key_matched_min.back(); matched <= key_matched_max.back(); ++matched) {
        rule_names.back().push_back("\"\"");
      }
    }

    // construct 0~(len(properties) - 2) rules

    for (int i = properties_size - 2; i >= 0; --i) {
      const std::string& prop_pattern = prop_patterns[i + 1];
      for (int matched = key_matched_min[i]; matched <= key_matched_max[i]; ++matched) {
        std::string cur_rule_body = "";
        if (matched == key_matched_max[i + 1]) {
          cur_rule_body = rule_names[i + 1][matched - key_matched_min[i + 1]];
        } else if (is_required[i + 1] || matched == key_matched_min[i + 1] - 1) {
          cur_rule_body = mid_sep + " " + prop_pattern + " " +
                          rule_names[i + 1][matched + 1 - key_matched_min[i + 1]];
        } else {
          cur_rule_body = rule_names[i + 1][matched - key_matched_min[i + 1]] + " | " + mid_sep +
                          " " + prop_pattern + " " +
                          rule_names[i + 1][matched - key_matched_min[i + 1] + 1];
        }
        std::string cur_rule_name =
            rule_name + "_part_" + std::to_string(i) + "_" + std::to_string(matched);
        cur_rule_name = ebnf_script_creator_.AddRule(cur_rule_name, cur_rule_body);
        rule_names[i].push_back(cur_rule_name);
      }
    }

    // construct the root rule
    bool is_first = true;
    for (int i = 0; i < static_cast<int>(properties.size()); ++i) {
      if (key_matched_max[i] < key_matched_min[i]) {
        continue;
      }
      if (key_matched_min[i] > 1) {
        break;
      }
      if (!is_first) {
        res += " | ";
      } else {
        is_first = false;
      }
      res += "(" + prop_patterns[i] + " " + rule_names[i][1 - key_matched_min[i]] + ")";
      if (is_required[i]) {
        break;
      }
    }

    if (allow_additional && required.empty()) {
      if (!is_first) {
        res += " | ";
      }
      res += "(" + additional_prop_pattern + " ";
      switch (json_format) {
        case (JSONFormat::kJSON): {
          res += GetPropertyWithNumberConstrains(
                     mid_sep + " " + additional_prop_pattern, min_properties, max_properties, 1
                 ) +
                 ")";
          break;
        }
        case (JSONFormat::kXML): {
          res += GetPropertyWithNumberConstrains(
                     additional_prop_pattern, min_properties, max_properties, 1
                 ) +
                 ")";
          break;
        }
      }
    }

    // add separators and the empty string option
    res = first_sep + " (" + res + ") " + last_sep;
  }
  return res;
}

Result<JSONSchemaConverter::ObjectSpec, SchemaError> JSONSchemaConverter::ParseObjectSchema(
    const picojson::object& schema
) {
  XGRAMMAR_DCHECK(
      (schema.count("type") && schema.at("type").get<std::string>() == "object") ||
      schema.count("properties") || schema.count("additionalProperties") ||
      schema.count("unevaluatedProperties")
  );
  std::vector<std::pair<std::string, picojson::value>> properties;
  std::unordered_set<std::string> required_properties;
  std::vector<std::pair<std::string, picojson::value>> pattern_properties;
  picojson::value property_names = picojson::value();
  bool allow_additional_properties = !strict_mode_;
  picojson::value additional_properties_schema = picojson::value();
  bool allow_unevaluated_properties = true;
  picojson::value unevaluated_properties_schema = picojson::value();
  int min_properties = 0;
  int max_properties = -1;

  if (schema.count("properties")) {
    if (!schema.at("properties").is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "properties must be an object"
      );
    }
    auto properties_obj = schema.at("properties").get<picojson::object>();
    for (const auto& key : properties_obj.ordered_keys()) {
      properties.push_back({key, properties_obj.at(key)});
    }
  }

  if (schema.count("required")) {
    if (!schema.at("required").is<picojson::array>()) {
      return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "required must be an array");
    }
    for (const auto& required_prop : schema.at("required").get<picojson::array>()) {
      required_properties.insert(required_prop.get<std::string>());
    }
  }

  if (schema.count("patternProperties")) {
    if (!schema.at("patternProperties").is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "patternProperties must be an object"
      );
    }
    auto pattern_properties_obj = schema.at("patternProperties").get<picojson::object>();
    for (const auto& key : pattern_properties_obj.ordered_keys()) {
      pattern_properties.push_back({key, pattern_properties_obj.at(key)});
    }
  }

  if (schema.count("propertyNames")) {
    if (!schema.at("propertyNames").is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "propertyNames must be an object"
      );
    }
    property_names = schema.at("propertyNames");
    picojson::object& property_names_obj = property_names.get<picojson::object>();
    if (property_names_obj.count("type") && property_names_obj.at("type").is<std::string>() &&
        property_names_obj.at("type").get<std::string>() != "string") {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema,
          "propertyNames must be an object that validates string"
      );
    }
    property_names_obj["type"] = picojson::value("string");
  }

  if (schema.count("additionalProperties") && (!schema.at("additionalProperties").is<bool>() ||
                                               schema.at("additionalProperties").get<bool>())) {
    additional_properties_schema = schema.at("additionalProperties");
    allow_additional_properties = true;
  } else {
    allow_additional_properties = false;
  }

  if (schema.count("additionalProperties")) {
    allow_unevaluated_properties = allow_additional_properties;
  }

  // Here we ignore the effect of unevaluatedProperties after setting additionalProperties
  // However, in fact unevaluatedProperties still has an impact on nested structures, such as
  // allOf We temporarily overlook this situation

  if (schema.count("additionalProperties") == 0) {
    unevaluated_properties_schema = schema.count("unevaluatedProperties")
                                        ? schema.at("unevaluatedProperties")
                                        : picojson::value(!strict_mode_);
    allow_unevaluated_properties =
        !unevaluated_properties_schema.is<bool>() || unevaluated_properties_schema.get<bool>();
  }

  if (schema.count("minProperties")) {
    if (!schema.at("minProperties").is<int64_t>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "minProperties must be an integer"
      );
    }
    min_properties = static_cast<int>(schema.at("minProperties").get<int64_t>());
    if (min_properties < 0) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema, "minProperties must be a non-negative integer"
      );
    }
  }

  if (schema.count("maxProperties")) {
    if (!schema.at("maxProperties").is<int64_t>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "maxProperties must be an integer"
      );
    }
    max_properties = static_cast<int>(schema.at("maxProperties").get<int64_t>());
    if (max_properties < 0) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema, "maxProperties must be a non-negative integer"
      );
    }
  }

  if (max_properties != -1 && min_properties > max_properties) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "minxPropertiesmax is greater than maxProperties: " + std::to_string(min_properties) +
            " > " + std::to_string(max_properties)
    );
  }

  if (max_properties != -1 && static_cast<int>(required_properties.size()) > max_properties) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "maxProperties is less than the number of required properties: " +
            std::to_string(max_properties) + " < " + std::to_string(required_properties.size())
    );
  }

  if (pattern_properties.empty() && property_names.is<picojson::null>() &&
      !allow_additional_properties && !allow_unevaluated_properties &&
      min_properties > static_cast<int>(properties.size())) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "minProperties is greater than the number of properties, but additional properties "
        "aren't "
        "allowed: " +
            std::to_string(min_properties) + " > " + std::to_string(properties.size())
    );
  }

  return ResultOk(ObjectSpec{
      properties,
      pattern_properties,
      allow_additional_properties,
      additional_properties_schema,
      allow_unevaluated_properties,
      unevaluated_properties_schema,
      required_properties,
      property_names,
      min_properties,
      max_properties
  });
}

Result<JSONSchemaConverter::StringSpec, SchemaError> JSONSchemaConverter::ParseStringSchema(
    const picojson::object& schema, JSONFormat json_format
) {
  XGRAMMAR_DCHECK((schema.count("type") && schema.at("type").get<std::string>() == "string"));
  if (schema.count("format")) {
    StringSpec string_spec;
    if (json_format == JSONFormat::kJSON) {
      string_spec.wrapper.first = "\\\"";
      string_spec.wrapper.second = "\\\"";
    }
    std::string format = schema.at("format").get<std::string>();
    if (format == "email") {
      // refer to RFC 5321 and RFC 5322, but skipping `address-literal` at
      // RFC 5321 section 4.1.2 currently
      std::string atext = "[\\w!#$%&'*+/=?^`{|}~-]";
      std::string dot_string = "(" + atext + "+(\\." + atext + "+)*)";
      std::string quoted_string =
          "\\\\\"(\\\\[\\x20-\\x7E]|[\\x20\\x21\\x23-\\x5B\\x5D-\\x7E])*\\\\\"";
      std::string domain =
          "([A-Za-z0-9]([\\-A-Za-z0-9]*[A-Za-z0-9])?)((\\.[A-Za-z0-9][\\-A-Za-z0-9]*[A-Za-z0-9])*)";
      std::string email_regex_pattern =
          "^(" + dot_string + "|" + quoted_string + ")@" + domain + "$";
      std::string email_ebnf = RegexToEBNF(email_regex_pattern, false);
      string_spec.pattern = email_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "date") {
      // refer to RFC 3339, section 5.6
      std::string date_regex_pattern = "^(\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\\d|3[01]))$";
      std::string date_ebnf = RegexToEBNF(date_regex_pattern, false);
      string_spec.pattern = date_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "time") {
      // refer to RFC 3339, section 5.6
      std::string time_regex_pattern =
          "^([01]\\d|2[0-3]):[0-5]\\d:([0-5]\\d|60)(\\.\\d+)?(Z|[+-]([01]\\d|2[0-3]):[0-5]\\d)$";
      std::string time_ebnf = RegexToEBNF(time_regex_pattern, false);
      string_spec.pattern = time_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "date-time") {
      // refer to RFC 3339, section 5.6
      std::string date_time_regex_pattern =
          "^(\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\\d|3[01]))T([01]\\d|2[0-3]):([0-5]\\d|60):["
          "0-5]\\d(\\.\\d+)?(Z|[+-]([01]\\d|2[0-3]):[0-5]\\d)$";
      std::string date_time_ebnf = RegexToEBNF(date_time_regex_pattern, false);
      string_spec.pattern = date_time_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "duration") {
      // refer to RFC 3339, Appendix A
      std::string duration_regex_pattern =
          "^P((\\d+D|\\d+M(\\d+D)?|\\d+Y(\\d+M(\\d+D)?)?)(T(\\d+S|\\d+M(\\d+S)?|\\d+H(\\d+M(\\d+S)?"
          ")?))?|T(\\d+S|\\d+M(\\d+S)?|\\d+H(\\d+M(\\d+S)?)?)|\\d+W)$";
      std::string duration_ebnf = RegexToEBNF(duration_regex_pattern, false);
      string_spec.pattern = duration_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "ipv4") {
      // refer to RFC 2673, section 3.2
      std::string decbyte = "(25[0-5]|2[0-4]\\d|[0-1]?\\d?\\d)";
      std::string ipv4_regex_pattern = "^(" + decbyte + "\\.){3}" + decbyte + "$";
      std::string ipv4_ebnf = RegexToEBNF(ipv4_regex_pattern, false);
      string_spec.pattern = ipv4_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "ipv6") {
      // refer to RFC 3986, section 3.3.2
      std::string ipv6_regex_pattern =
          "("
          "([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|"  // 1:2:3:4:5:6:7:8
          "([0-9a-fA-F]{1,4}:){1,7}:|"  // 1::                              1:2:3:4:5:6:7::
          "([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|"         // 1::8             1:2:3:4:5:6::8
                                                               // 1:2:3:4:5:6::8
          "([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|"  // 1::7:8           1:2:3:4:5::7:8
                                                               // 1:2:3:4:5::8
          "([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|"  // 1::6:7:8         1:2:3:4::6:7:8
                                                               // 1:2:3:4::8
          "([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|"  // 1::5:6:7:8       1:2:3::5:6:7:8
                                                               // 1:2:3::8
          "([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|"  // 1::4:5:6:7:8     1:2::4:5:6:7:8
                                                               // 1:2::8
          "[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|"  // 1::3:4:5:6:7:8   1::3:4:5:6:7:8  1::8
          ":((:[0-9a-fA-F]{1,4}){1,7}|:)|"  // ::2:3:4:5:6:7:8  ::2:3:4:5:6:7:8 ::8       ::
          "::(ffff(:0{1,4}){0,1}:){0,1}"
          "((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}"
          "(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|"  // ::255.255.255.255   ::ffff:255.255.255.255
                                                       // ::ffff:0:255.255.255.255  (IPv4-mapped
                                                       // IPv6 addresses and IPv4-translated
                                                       // addresses)
          "([0-9a-fA-F]{1,4}:){1,4}:"
          "((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}"
          "(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])"  // 2001:db8:3:4::192.0.2.33
                                                      // 64:ff9b::192.0.2.33 (IPv4-Embedded IPv6
                                                      // Address)
          ")";

      std::string ipv6_ebnf = RegexToEBNF(ipv6_regex_pattern, false);
      string_spec.pattern = ipv6_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "hostname") {
      // refer to RFC 1123, section 2.1
      std::string hostname_regex_pattern =
          "^([a-z0-9]([a-z0-9-]*[a-z0-9])?)(\\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$";
      std::string hostname_ebnf = RegexToEBNF(hostname_regex_pattern, false);
      string_spec.pattern = hostname_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "uuid") {
      // refer to RFC 4122, section 3
      std::string uuid_regex_pattern =
          "^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$";
      std::string uuid_ebnf = RegexToEBNF(uuid_regex_pattern, false);
      string_spec.pattern = uuid_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "uri") {
      // refer to RFC 3986, Appendix A, but skipping IP-literal and IPv4address currently
      std::string schema = "[a-zA-Z][a-zA-Z+\\.-]*";
      std::string pchar = "([\\w\\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])";
      std::string query_fragment_char = "([\\w\\.~!$&'()*+,;=:@/\\?-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
      std::string query = "(\\?" + query_fragment_char + ")?";
      std::string fragment = "(#" + query_fragment_char + ")?";
      std::string path_abempty = "(/" + pchar + "*)*";
      std::string path_absolute_rootless_empty = "/?(" + pchar + "+(/" + pchar + "*)*)?";
      std::string userinfo = "([\\w\\.~!$&'()*+,;=:-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
      std::string host = "([\\w\\.~!$&'()*+,;=-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
      std::string authority = "(" + userinfo + "@)?" + host + "(:\\d*)?";
      std::string hier_part =
          "(//" + authority + path_abempty + "|" + path_absolute_rootless_empty + ")";
      std::string uri_regex_pattern = "^" + schema + ":" + hier_part + query + fragment + "$";
      std::string uri_ebnf = RegexToEBNF(uri_regex_pattern, false);
      string_spec.pattern = uri_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "uri-reference") {
      // refer to RFC 3986, Appendix A, but skipping IP-literal and IPv4address currently
      std::string pchar = "([\\w\\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])";
      std::string query_fragment_char = "([\\w\\.~!$&'()*+,;=:@/\\?-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
      std::string query = "(\\?" + query_fragment_char + ")?";
      std::string fragment = "(#" + query_fragment_char + ")?";
      std::string path_abempty = "(/" + pchar + "*)*";
      std::string path_absolute = "/(" + pchar + "+(/" + pchar + "*)*)?";
      std::string segment_nz_nc = "([\\w\\.~!$&'()*+,;=@-]|%[0-9A-Fa-f][0-9A-Fa-f])+";
      std::string path_noscheme = segment_nz_nc + "(/" + pchar + "*)*";
      std::string userinfo = "([\\w\\.~!$&'()*+,;=:-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
      std::string host = "([\\w\\.~!$&'()*+,;=-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
      std::string authority = "(" + userinfo + "@)?" + host + "(:\\d*)?";
      std::string relative_part =
          "(//" + authority + path_abempty + "|" + path_absolute + "|" + path_noscheme + ")?";
      std::string uri_reference_regex_pattern = "^" + relative_part + query + fragment + "$";
      std::string uri_reference_ebnf = RegexToEBNF(uri_reference_regex_pattern, false);
      string_spec.pattern = uri_reference_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "uri-template") {
      // refer to RFC 6570, section 2
      std::string literals =
          "([\\x21\\x23-\\x24\\x26\\x28-\\x3B\\x3D\\x3F-\\x5B\\x5D\\x5F\\x61-\\x7A\\x7E]"
          "|%[0-9A-Fa-f][0-9A-Fa-f])";
      std::string op = "[+#\\./;\\?&=,!@|]";
      std::string varchar = "(\\w|%[0-9A-Fa-f][0-9A-Fa-f])";
      std::string varname = varchar + "(\\.?" + varchar + ")*";
      std::string varspec = varname + "(:[1-9]\\d?\\d?\\d?|\\*)?";
      std::string variable_list = varspec + "(," + varspec + ")*";
      std::string expression = "\\{(" + op + ")?" + variable_list + "\\}";
      std::string uri_template_regex_pattern = "^(" + literals + "|" + expression + ")*$";
      std::string uri_template_ebnf = RegexToEBNF(uri_template_regex_pattern, false);
      string_spec.pattern = uri_template_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "json-pointer") {
      // refer to RFC 6901, section 3
      std::string json_pointer_regex_pattern =
          "^(/([\\x00-\\x2E]|[\\x30-\\x7D]|[\\x7F-\\U0010FFFF]|~[01])*)*$";
      std::string json_pointer_ebnf = RegexToEBNF(json_pointer_regex_pattern, false);
      string_spec.pattern = json_pointer_ebnf;
      return ResultOk(string_spec);
    }
    if (format == "relative-json-pointer") {
      // refer to draft-handrews-relative-json-pointer-01, section 3
      std::string relative_json_pointer_regex_pattern =
          "^(0|[1-9][0-9]*)(#|(/([\\x00-\\x2E]|[\\x30-\\x7D]|[\\x7F-\\U0010FFFF]|~[01])*)*)$";
      std::string relative_json_pointer_ebnf =
          RegexToEBNF(relative_json_pointer_regex_pattern, false);
      string_spec.pattern = relative_json_pointer_ebnf;
      return ResultOk(string_spec);
    }
  }
  if (schema.count("pattern")) {
    StringSpec string_spec;
    if (json_format == JSONFormat::kJSON) {
      string_spec.wrapper.first = "\\\"";
      string_spec.wrapper.second = "\\\"";
    }
    if (schema.count("minLength") || schema.count("maxLength") || schema.count("format")) {
      XGRAMMAR_LOG(WARNING) << "Specifying pattern and minLength/maxLength/format is not "
                            << "supported yet, ignoring minLength/maxLength/format";
    }
    std::string regex_pattern = schema.at("pattern").get<std::string>();
    std::string converted_regex = RegexToEBNF(regex_pattern, false);
    string_spec.pattern = converted_regex;
    return ResultOk(string_spec);
  }
  if (schema.count("minLength") || schema.count("maxLength")) {
    StringSpec string_spec;
    if (json_format == JSONFormat::kJSON) {
      string_spec.wrapper.first = "\\\"";
      string_spec.wrapper.second = "\\\"";
    }
    string_spec.min_length = schema.count("minLength") ? schema.at("minLength").get<int64_t>() : 0;
    string_spec.max_length = schema.count("maxLength") ? schema.at("maxLength").get<int64_t>() : -1;
    XGRAMMAR_CHECK(string_spec.max_length == -1 || string_spec.min_length <= string_spec.max_length)
        << "In string schema, minLength " << string_spec.min_length << " is greater than "
        << "maxLength " << string_spec.max_length;
    switch (json_format) {
      case JSONFormat::kJSON: {
        string_spec.pattern = "[^\"\\\\\\r\\n]";
        break;
      }
      case JSONFormat::kXML: {
        string_spec.pattern = "[^<>&\\r\\n]";
        break;
      }
    }
    return ResultOk(string_spec);
  }
  StringSpec string_spec;
  switch (json_format) {
    case JSONFormat::kJSON: {
      string_spec.pattern = "[\"] " + kBasicStringSub;
      return ResultOk(string_spec);
    }
    case JSONFormat::kXML: {
      string_spec.pattern = kXMLString;
      return ResultOk(string_spec);
    }
    default: {
      XGRAMMAR_LOG(FATAL) << "Unsupported JSON Format type: " << static_cast<int>(json_format);
    }
  }
}

std::string JSONSchemaConverter::VisitObject(
    const picojson::object& schema, const std::string& rule_name, const JSONFormat json_format
) {
  // Parse the object schema
  auto object_spec_result = ParseObjectSchema(schema);
  if (object_spec_result.IsErr()) {
    XGRAMMAR_LOG(FATAL) << std::move(object_spec_result).UnwrapErr().what();
  }

  auto object_spec = std::move(object_spec_result).Unwrap();
  std::string result;
  if (json_format == JSONFormat::kJSON) {
    result += "\"{\"";
  }

  // could_be_empty will be set to True when the rule could be "{}". We will handle this case at
  // last, and handle non-empty cases before that.
  bool could_be_empty = false;

  // Handle additional properties
  std::string additional_suffix = "";
  picojson::value additional_property;
  if (object_spec.allow_additional_properties) {
    additional_suffix = "addl";
    additional_property = object_spec.additional_properties_schema;
  } else if (object_spec.allow_unevaluated_properties) {
    additional_suffix = "uneval";
    additional_property = object_spec.unevaluated_properties_schema;
  }
  indentManager_->StartIndent();

  if (object_spec.pattern_properties.size() > 0 ||
      !object_spec.property_names.is<picojson::null>()) {
    // Case 1: patternProperties or propertyNames is difined
    // TODO: Here we only handle the case that additionalProperties=False
    // TODO: The coexistence of properties, required, etc. has not been addressed yet,
    // as it may cause schema conflicts
    // TODO: The situation of duplicate keys has not been resolved yet

    // Initialize the beginning sequence of a property.
    std::string beg_seq;
    switch (json_format) {
      case (JSONFormat::kJSON): {
        beg_seq = NextSeparator();
        break;
      }
      case (JSONFormat::kXML): {
        beg_seq = "";
        break;
      }
    }

    std::string property_rule_body = "(";
    if (object_spec.max_properties != 0) {
      if (object_spec.pattern_properties.size() > 0) {
        for (int i = 0; i < static_cast<int>(object_spec.pattern_properties.size()); ++i) {
          const auto& [prop_name, prop_schema] = object_spec.pattern_properties[i];
          std::string value = CreateRuleFromSchema(
              prop_schema, rule_name + "_prop_" + std::to_string(i), json_format
          );

          std::string property_pattern;
          if (json_format == JSONFormat::kJSON) {
            property_pattern += "\"\\\"\"" + RegexToEBNF(prop_name, false) + "\"\\\"\" " +
                                colon_pattern_ + " " + value;
          } else {
            property_pattern += "\"<parameter=\" " + RegexToEBNF(prop_name, false) + " \">\" " +
                                kWhiteSpace + " " + value + " " + kWhiteSpace + " \"</parameter>\"";
          }
          if (i != 0) {
            property_rule_body += " | ";
          }
          property_rule_body += "(" + beg_seq + " " + property_pattern + ")";
        }
        property_rule_body += ")";
      } else {
        auto key_pattern =
            CreateRuleFromSchema(object_spec.property_names, rule_name + "_name", json_format);
        switch (json_format) {
          case (JSONFormat::kJSON): {
            property_rule_body +=
                beg_seq + " " + key_pattern + " " + colon_pattern_ + " " + kBasicAny + ")";
            break;
          }
          case (JSONFormat::kXML): {
            property_rule_body += beg_seq + " \"<parameter=\" " + key_pattern + " \">\" " +
                                  kWhiteSpace + " " + kXMLAny + " " + kWhiteSpace +
                                  " \"</parameter>\"";
            break;
          }
        }
      }
      // set the property rule
      auto prop_rule_name = ebnf_script_creator_.AllocateRuleName(rule_name + "_prop");
      ebnf_script_creator_.AddRuleWithAllocatedName(prop_rule_name, property_rule_body);
      switch (json_format) {
        case (JSONFormat::kJSON): {
          result += " " + prop_rule_name + " " +
                    GetPropertyWithNumberConstrains(
                        NextSeparator() + " " + prop_rule_name,
                        object_spec.min_properties,
                        object_spec.max_properties,
                        1
                    ) +
                    NextSeparator(true);
          break;
        }
        case (JSONFormat::kXML): {
          result += " " + prop_rule_name + " " +
                    GetPropertyWithNumberConstrains(
                        prop_rule_name, object_spec.min_properties, object_spec.max_properties, 1
                    );
          break;
        }
      }
      could_be_empty = object_spec.min_properties == 0;
    }
  } else if (object_spec.properties.size() > 0) {
    //  Case 2: properties are defined
    result += " " + GetPartialRuleForProperties(
                        object_spec.properties,
                        object_spec.required_properties,
                        additional_property,
                        rule_name,
                        additional_suffix,
                        object_spec.min_properties,
                        object_spec.max_properties,
                        json_format
                    );
    could_be_empty = object_spec.required_properties.empty() && object_spec.min_properties == 0;
  } else if (!additional_property.is<picojson::null>() &&
             (!additional_property.is<bool>() || additional_property.get<bool>())) {
    // Case 3: no properties are defined and additional properties are allowed
    if (object_spec.max_properties != 0) {
      std::string other_property_pattern;
      switch (json_format) {
        case (JSONFormat::kJSON): {
          other_property_pattern += GetOtherPropertyPattern(
              kBasicString, additional_property, rule_name, additional_suffix
          );
          result += " " + NextSeparator() + " " + other_property_pattern + " ";
          break;
        }
        case (JSONFormat::kXML): {
          other_property_pattern += GetOtherPropertyPattern(
              kXMLVariableName, additional_property, rule_name, additional_suffix, JSONFormat::kXML
          );
          result += " " + other_property_pattern + " ";
          break;
        }
      }
      if (object_spec.max_properties != 0) {
        result += GetPropertyWithNumberConstrains(
                      NextSeparator() + " " + other_property_pattern,
                      object_spec.min_properties,
                      object_spec.max_properties,
                      1
                  ) +
                  " " + NextSeparator(true);
      }
    }
    could_be_empty = object_spec.min_properties == 0;
  }

  indentManager_->EndIndent();

  switch (json_format) {
    case (JSONFormat::kJSON): {
      result += " \"}\"";
      if (could_be_empty) {
        // result = (result) | {}
        auto rest = "\"{\" " + std::string(any_whitespace_ ? "[ \\n\\t]* " : "") + "\"}\"";
        if (result == "\"{\"  \"}\"") {
          result = rest;
        } else {
          result = "(" + result + ") | " + rest;
        }
      }
      break;
    }
    case (JSONFormat::kXML): {
      if (could_be_empty) {
        result = "\"\" | " + result;
      }
      break;
    }
  }
  return result;
}

std::string JSONSchemaConverter::VisitTypeArray(
    const picojson::object& schema, const std::string& rule_name
) {
  XGRAMMAR_CHECK(schema.at("type").is<picojson::array>());
  auto type_array = schema.at("type").get<picojson::array>();

  picojson::object schema_copy = schema;
  if (type_array.size() == 0) {
    schema_copy.erase("type");
    return VisitSchema(picojson::value(schema_copy), rule_name);
  }
  std::string result;
  for (const auto& type : type_array) {
    XGRAMMAR_CHECK(type.is<std::string>())
        << "type must be a string or an array of strings, but got " << type;
    if (!result.empty()) {
      result += " | ";
    }
    schema_copy["type"] = type;
    result += CreateRuleFromSchema(
        picojson::value(schema_copy), rule_name + "_" + type.get<std::string>()
    );
  }
  return result;
}

std::string JSONSchemaToEBNF(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    JSONFormat json_format
) {
  picojson::value schema_value;
  std::string err = picojson::parse(schema_value, schema);
  XGRAMMAR_CHECK(err.empty()) << "Failed to parse JSON: " << err
                              << ". The JSON string is:" << schema;
  return JSONSchemaToEBNF(
      schema_value, any_whitespace, indent, separators, strict_mode, json_format
  );
}

std::string JSONSchemaToEBNF(
    const picojson::value& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    JSONFormat json_format
) {
  JSONSchemaConverter converter(
      schema, any_whitespace, indent, separators, strict_mode, json_format
  );
  return converter.Convert(json_format);
}

// Wrapper function for testing
std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end) {
  return JSONSchemaConverter::GenerateRangeRegex(start, end);
}

std::string GenerateFloatRangeRegex(std::optional<double> start, std::optional<double> end) {
  return JSONSchemaConverter::GenerateFloatRangeRegex(start, end, 6);
}

}  // namespace xgrammar
