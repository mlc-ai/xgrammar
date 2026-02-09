/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter.cc
 * \brief Implementation of JSONSchemaConverter and related utilities.
 */
#include "json_schema_converter.h"

#include <picojson.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "json_schema_converter_ext.h"
#include "regex_converter.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

// ==================== Spec ToString implementations ====================

std::string IntegerSpec::ToString() const {
  return "IntegerSpec{minimum=" + (minimum.has_value() ? std::to_string(*minimum) : "null") +
         ", maximum=" + (maximum.has_value() ? std::to_string(*maximum) : "null") +
         ", exclusive_minimum=" +
         (exclusive_minimum.has_value() ? std::to_string(*exclusive_minimum) : "null") +
         ", exclusive_maximum=" +
         (exclusive_maximum.has_value() ? std::to_string(*exclusive_maximum) : "null") + "}";
}

std::string NumberSpec::ToString() const {
  return "NumberSpec{minimum=" + (minimum.has_value() ? std::to_string(*minimum) : "null") +
         ", maximum=" + (maximum.has_value() ? std::to_string(*maximum) : "null") +
         ", exclusive_minimum=" +
         (exclusive_minimum.has_value() ? std::to_string(*exclusive_minimum) : "null") +
         ", exclusive_maximum=" +
         (exclusive_maximum.has_value() ? std::to_string(*exclusive_maximum) : "null") + "}";
}

std::string StringSpec::ToString() const {
  return "StringSpec{pattern=" + (pattern.has_value() ? "\"" + *pattern + "\"" : "null") +
         ", format=" + (format.has_value() ? "\"" + *format + "\"" : "null") +
         ", min_length=" + std::to_string(min_length) +
         ", max_length=" + std::to_string(max_length) + "}";
}

std::string BooleanSpec::ToString() const { return "BooleanSpec{}"; }

std::string NullSpec::ToString() const { return "NullSpec{}"; }

std::string AnySpec::ToString() const { return "AnySpec{}"; }

std::string ArraySpec::ToString() const {
  return "ArraySpec{prefix_items.size()=" + std::to_string(prefix_items.size()) +
         ", allow_additional_items=" + (allow_additional_items ? "true" : "false") +
         ", additional_items=" + (additional_items ? "SchemaSpec" : "null") +
         ", min_items=" + std::to_string(min_items) + ", max_items=" + std::to_string(max_items) +
         "}";
}

std::string ObjectSpec::ToString() const {
  std::string s =
      "ObjectSpec{properties.size()=" + std::to_string(properties.size()) + ", properties=[";
  for (size_t i = 0; i < properties.size(); ++i) {
    if (i != 0) s += ", ";
    s += properties[i].name;
  }
  s += "], pattern_properties.size()=" + std::to_string(pattern_properties.size()) + ", required=[";
  bool first = true;
  for (const auto& r : required) {
    if (!first) s += ", ";
    s += r;
    first = false;
  }
  s +=
      std::string("], allow_additional_properties=") +
      (allow_additional_properties ? "true" : "false") +
      ", additional_properties_schema=" + (additional_properties_schema ? "SchemaSpec" : "null") +
      ", allow_unevaluated_properties=" + (allow_unevaluated_properties ? "true" : "false") +
      ", unevaluated_properties_schema=" + (unevaluated_properties_schema ? "SchemaSpec" : "null") +
      ", property_names=" + (property_names ? "SchemaSpec" : "null") +
      ", min_properties=" + std::to_string(min_properties) +
      ", max_properties=" + std::to_string(max_properties) + "}";
  return s;
}

std::string ConstSpec::ToString() const { return "ConstSpec{json_value=\"" + json_value + "\"}"; }

std::string EnumSpec::ToString() const {
  std::string s =
      "EnumSpec{json_values.size()=" + std::to_string(json_values.size()) + ", json_values=[";
  for (size_t i = 0; i < json_values.size(); ++i) {
    if (i != 0) s += ", ";
    s += "\"" + json_values[i] + "\"";
  }
  s += "]}";
  return s;
}

std::string RefSpec::ToString() const { return "RefSpec{uri=\"" + uri + "\"}"; }

std::string AnyOfSpec::ToString() const {
  return "AnyOfSpec{options.size()=" + std::to_string(options.size()) + "}";
}

std::string AllOfSpec::ToString() const {
  return "AllOfSpec{schemas.size()=" + std::to_string(schemas.size()) + "}";
}

std::string TypeArraySpec::ToString() const {
  return "TypeArraySpec{type_schemas.size()=" + std::to_string(type_schemas.size()) + "}";
}

std::string SchemaSpec::ToString() const {
  std::string spec_str;
  std::visit([&spec_str](const auto& s) { spec_str = s.ToString(); }, spec);
  return "SchemaSpec{spec=" + spec_str + ", cache_key=\"" + cache_key + "\", rule_name_hint=\"" +
         rule_name_hint + "\"}";
}

// ==================== SchemaParser (Internal) ====================

namespace {

enum class SchemaErrorType : int {
  kInvalidSchema = 0,
  kUnsatisfiableSchema = 1,
};

using SchemaError = TypedError<SchemaErrorType>;

/*!
 * \brief Parser for JSON Schema, converts JSON Schema to SchemaSpec intermediate representation.
 */
class SchemaParser {
 public:
  struct Config {
    bool strict_mode = false;
    JSONFormat json_format;
  };

  explicit SchemaParser(const picojson::value& root_schema, const Config& config)
      : config_(config), root_schema_(root_schema) {}

  Result<SchemaSpecPtr, SchemaError> Parse(
      const picojson::value& schema,
      const std::string& rule_name_hint = "root",
      std::optional<std::string> default_type = std::nullopt
  );

  const picojson::value& GetRootSchema() const { return root_schema_; }
  bool IsStrictMode() const { return config_.strict_mode; }

  Result<SchemaSpecPtr, SchemaError> ResolveRef(
      const std::string& uri, const std::string& rule_name_hint
  );

 private:
  Result<IntegerSpec, SchemaError> ParseInteger(const picojson::object& schema);
  Result<NumberSpec, SchemaError> ParseNumber(const picojson::object& schema);
  Result<StringSpec, SchemaError> ParseString(const picojson::object& schema);
  Result<BooleanSpec, SchemaError> ParseBoolean(const picojson::object& schema);
  Result<NullSpec, SchemaError> ParseNull(const picojson::object& schema);
  Result<ArraySpec, SchemaError> ParseArray(const picojson::object& schema);
  Result<ObjectSpec, SchemaError> ParseObject(const picojson::object& schema);
  Result<ConstSpec, SchemaError> ParseConst(const picojson::object& schema);
  Result<EnumSpec, SchemaError> ParseEnum(const picojson::object& schema);
  Result<RefSpec, SchemaError> ParseRef(const picojson::object& schema);
  Result<AnyOfSpec, SchemaError> ParseAnyOf(const picojson::object& schema);
  Result<AllOfSpec, SchemaError> ParseAllOf(const picojson::object& schema);
  Result<TypeArraySpec, SchemaError> ParseTypeArray(
      const picojson::object& schema, const std::string& rule_name_hint
  );

  std::string ComputeCacheKey(const picojson::value& schema);

  static void WarnUnsupportedKeywords(
      const picojson::object& schema, const std::vector<std::string>& keywords, bool verbose = false
  );

  Config config_;
  picojson::value root_schema_;
  std::unordered_map<std::string, SchemaSpecPtr> ref_cache_;
  std::unordered_map<std::string, SchemaSpecPtr> schema_cache_;
};

std::string SchemaParser::ComputeCacheKey(const picojson::value& schema) {
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
      result += "\"" + key + "\":" + ComputeCacheKey(value);
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
      result += ComputeCacheKey(item);
    }
    return result + "]";
  }
  return schema.serialize(false);
}

void SchemaParser::WarnUnsupportedKeywords(
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

Result<SchemaSpecPtr, SchemaError> SchemaParser::Parse(
    const picojson::value& schema,
    const std::string& rule_name_hint,
    std::optional<std::string> default_type
) {
  std::string cache_key = ComputeCacheKey(schema);
  if (schema_cache_.count(cache_key)) {
    return ResultOk(schema_cache_[cache_key]);
  }

  if (schema.is<bool>()) {
    if (!schema.get<bool>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema, "Schema 'false' cannot accept any value"
      );
    }
    auto spec = SchemaSpec::Make(AnySpec{}, cache_key, rule_name_hint);
    schema_cache_[cache_key] = spec;
    return ResultOk(spec);
  }

  if (!schema.is<picojson::object>()) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kInvalidSchema,
        "Schema should be an object or bool, but got " + schema.serialize(false)
    );
  }

  const auto& schema_obj = schema.get<picojson::object>();
  WarnUnsupportedKeywords(
      schema_obj, {"not", "if", "then", "else", "dependentRequired", "dependentSchemas"}
  );

  SchemaSpecPtr result;

  if (schema_obj.count("$ref")) {
    auto ref_result = ParseRef(schema_obj);
    if (ref_result.IsErr()) return ResultErr(std::move(ref_result).UnwrapErr());
    auto ref_spec = std::move(ref_result).Unwrap();
    result = SchemaSpec::Make(std::move(ref_spec), cache_key, rule_name_hint);
  } else if (schema_obj.count("const")) {
    auto const_result = ParseConst(schema_obj);
    if (const_result.IsErr()) return ResultErr(std::move(const_result).UnwrapErr());
    result = SchemaSpec::Make(std::move(const_result).Unwrap(), cache_key, rule_name_hint);
  } else if (schema_obj.count("enum")) {
    auto enum_result = ParseEnum(schema_obj);
    if (enum_result.IsErr()) return ResultErr(std::move(enum_result).UnwrapErr());
    result = SchemaSpec::Make(std::move(enum_result).Unwrap(), cache_key, rule_name_hint);
  } else if (schema_obj.count("anyOf") || schema_obj.count("oneOf")) {
    auto anyof_result = ParseAnyOf(schema_obj);
    if (anyof_result.IsErr()) return ResultErr(std::move(anyof_result).UnwrapErr());
    result = SchemaSpec::Make(std::move(anyof_result).Unwrap(), cache_key, rule_name_hint);
  } else if (schema_obj.count("allOf")) {
    auto allof_result = ParseAllOf(schema_obj);
    if (allof_result.IsErr()) return ResultErr(std::move(allof_result).UnwrapErr());
    result = SchemaSpec::Make(std::move(allof_result).Unwrap(), cache_key, rule_name_hint);
  } else if (schema_obj.count("type") || default_type.has_value()) {
    if (schema_obj.count("type") && schema_obj.at("type").is<picojson::array>()) {
      auto type_array_result = ParseTypeArray(schema_obj, rule_name_hint);
      if (type_array_result.IsErr()) return ResultErr(std::move(type_array_result).UnwrapErr());
      result = SchemaSpec::Make(std::move(type_array_result).Unwrap(), cache_key, rule_name_hint);
    } else {
      if (schema_obj.count("type") && !schema_obj.at("type").is<std::string>()) {
        return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "Type should be a string");
      }
      const std::string& type = schema_obj.count("type") ? schema_obj.at("type").get<std::string>()
                                                         : default_type.value();
      if (type == "integer") {
        auto int_result = ParseInteger(schema_obj);
        if (int_result.IsErr()) return ResultErr(std::move(int_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(int_result).Unwrap(), cache_key, rule_name_hint);
      } else if (type == "number") {
        auto num_result = ParseNumber(schema_obj);
        if (num_result.IsErr()) return ResultErr(std::move(num_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(num_result).Unwrap(), cache_key, rule_name_hint);
      } else if (type == "string") {
        auto str_result = ParseString(schema_obj);
        if (str_result.IsErr()) return ResultErr(std::move(str_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(str_result).Unwrap(), cache_key, rule_name_hint);
      } else if (type == "boolean") {
        auto bool_result = ParseBoolean(schema_obj);
        if (bool_result.IsErr()) return ResultErr(std::move(bool_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(bool_result).Unwrap(), cache_key, rule_name_hint);
      } else if (type == "null") {
        auto null_result = ParseNull(schema_obj);
        if (null_result.IsErr()) return ResultErr(std::move(null_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(null_result).Unwrap(), cache_key, rule_name_hint);
      } else if (type == "array") {
        auto array_result = ParseArray(schema_obj);
        if (array_result.IsErr()) return ResultErr(std::move(array_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(array_result).Unwrap(), cache_key, rule_name_hint);
      } else if (type == "object") {
        auto obj_result = ParseObject(schema_obj);
        if (obj_result.IsErr()) return ResultErr(std::move(obj_result).UnwrapErr());
        result = SchemaSpec::Make(std::move(obj_result).Unwrap(), cache_key, rule_name_hint);
      } else {
        return ResultErr<SchemaError>(
            SchemaErrorType::kInvalidSchema, "Unsupported type \"" + type + "\""
        );
      }
    }
  } else if (schema_obj.count("properties") || schema_obj.count("additionalProperties") ||
             schema_obj.count("unevaluatedProperties")) {
    auto obj_result = ParseObject(schema_obj);
    if (obj_result.IsErr()) return ResultErr(std::move(obj_result).UnwrapErr());
    result = SchemaSpec::Make(std::move(obj_result).Unwrap(), cache_key, rule_name_hint);
  } else if (schema_obj.count("items") || schema_obj.count("prefixItems") ||
             schema_obj.count("unevaluatedItems")) {
    auto array_result = ParseArray(schema_obj);
    if (array_result.IsErr()) return ResultErr(std::move(array_result).UnwrapErr());
    result = SchemaSpec::Make(std::move(array_result).Unwrap(), cache_key, rule_name_hint);
  } else {
    result = SchemaSpec::Make(AnySpec{}, cache_key, rule_name_hint);
  }

  schema_cache_[cache_key] = result;
  return ResultOk(result);
}

Result<IntegerSpec, SchemaError> SchemaParser::ParseInteger(const picojson::object& schema) {
  WarnUnsupportedKeywords(schema, {"multipleOf"});
  IntegerSpec spec;

  auto checkAndConvertIntegerBound = [](const picojson::value& value
                                     ) -> Result<int64_t, SchemaError> {
    if (!value.is<int64_t>() && !value.is<double>()) {
      return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "Value must be a number");
    }
    if (value.is<int64_t>()) return ResultOk<int64_t>(value.get<int64_t>());
    double val = value.get<double>();
    if (val != std::floor(val)) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "Integer constraint must be a whole number"
      );
    }
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
    return ResultOk<int64_t>(static_cast<int64_t>(val));
  };

  if (schema.count("minimum")) {
    auto result = checkAndConvertIntegerBound(schema.at("minimum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    spec.minimum = std::move(result).Unwrap();
  }
  if (schema.count("maximum")) {
    auto result = checkAndConvertIntegerBound(schema.at("maximum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    spec.maximum = std::move(result).Unwrap();
  }
  if (schema.count("exclusiveMinimum")) {
    auto result = checkAndConvertIntegerBound(schema.at("exclusiveMinimum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    int64_t val = std::move(result).Unwrap();
    if (val == std::numeric_limits<int64_t>::max()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema, "exclusiveMinimum would cause integer overflow"
      );
    }
    spec.exclusive_minimum = val;
  }
  if (schema.count("exclusiveMaximum")) {
    auto result = checkAndConvertIntegerBound(schema.at("exclusiveMaximum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    int64_t val = std::move(result).Unwrap();
    if (val == std::numeric_limits<int64_t>::min()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema, "exclusiveMaximum would cause integer underflow"
      );
    }
    spec.exclusive_maximum = val;
  }

  int64_t effective_min = spec.minimum.value_or(std::numeric_limits<int64_t>::min());
  int64_t effective_max = spec.maximum.value_or(std::numeric_limits<int64_t>::max());
  if (spec.exclusive_minimum.has_value()) {
    effective_min = std::max(effective_min, *spec.exclusive_minimum + 1);
  }
  if (spec.exclusive_maximum.has_value()) {
    effective_max = std::min(effective_max, *spec.exclusive_maximum - 1);
  }
  if (effective_min > effective_max) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema, "Invalid range: minimum greater than maximum"
    );
  }
  return ResultOk(std::move(spec));
}

Result<NumberSpec, SchemaError> SchemaParser::ParseNumber(const picojson::object& schema) {
  WarnUnsupportedKeywords(schema, {"multipleOf"});
  NumberSpec spec;

  auto getDouble = [](const picojson::value& value) -> Result<double, SchemaError> {
    if (!value.is<double>() && !value.is<int64_t>()) {
      return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "Value must be a number");
    }
    return ResultOk<double>(value.get<double>());
  };

  if (schema.count("minimum")) {
    auto result = getDouble(schema.at("minimum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    spec.minimum = std::move(result).Unwrap();
  }
  if (schema.count("maximum")) {
    auto result = getDouble(schema.at("maximum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    spec.maximum = std::move(result).Unwrap();
  }
  if (schema.count("exclusiveMinimum")) {
    auto result = getDouble(schema.at("exclusiveMinimum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    spec.exclusive_minimum = std::move(result).Unwrap();
  }
  if (schema.count("exclusiveMaximum")) {
    auto result = getDouble(schema.at("exclusiveMaximum"));
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    spec.exclusive_maximum = std::move(result).Unwrap();
  }

  double effective_min = spec.minimum.value_or(-std::numeric_limits<double>::infinity());
  double effective_max = spec.maximum.value_or(std::numeric_limits<double>::infinity());
  if (spec.exclusive_minimum.has_value()) {
    effective_min = std::max(effective_min, *spec.exclusive_minimum);
  }
  if (spec.exclusive_maximum.has_value()) {
    effective_max = std::min(effective_max, *spec.exclusive_maximum);
  }
  if (effective_min > effective_max) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema, "Invalid range: minimum greater than maximum"
    );
  }
  return ResultOk(std::move(spec));
}

Result<StringSpec, SchemaError> SchemaParser::ParseString(const picojson::object& schema) {
  StringSpec spec;
  if (schema.count("format")) spec.format = schema.at("format").get<std::string>();
  if (schema.count("pattern")) spec.pattern = schema.at("pattern").get<std::string>();
  if (schema.count("minLength")) {
    if (!schema.at("minLength").is<int64_t>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "minLength must be an integer"
      );
    }
    spec.min_length = static_cast<int>(schema.at("minLength").get<int64_t>());
  }
  if (schema.count("maxLength")) {
    if (!schema.at("maxLength").is<int64_t>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "maxLength must be an integer"
      );
    }
    spec.max_length = static_cast<int>(schema.at("maxLength").get<int64_t>());
  }
  if (spec.max_length != -1 && spec.min_length > spec.max_length) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "minLength " + std::to_string(spec.min_length) + " is greater than maxLength " +
            std::to_string(spec.max_length)
    );
  }
  return ResultOk(std::move(spec));
}

Result<BooleanSpec, SchemaError> SchemaParser::ParseBoolean(const picojson::object&) {
  return ResultOk(BooleanSpec{});
}

Result<NullSpec, SchemaError> SchemaParser::ParseNull(const picojson::object&) {
  return ResultOk(NullSpec{});
}

Result<ArraySpec, SchemaError> SchemaParser::ParseArray(const picojson::object& schema) {
  WarnUnsupportedKeywords(schema, {"uniqueItems", "contains", "minContains", "maxContains"});
  ArraySpec spec;

  if (schema.count("prefixItems")) {
    if (!schema.at("prefixItems").is<picojson::array>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "prefixItems must be an array"
      );
    }
    for (const auto& item : schema.at("prefixItems").get<picojson::array>()) {
      if (item.is<bool>() && !item.get<bool>()) {
        return ResultErr<SchemaError>(
            SchemaErrorType::kUnsatisfiableSchema, "prefixItems contains false"
        );
      } else if (!item.is<picojson::object>()) {
        return ResultErr<SchemaError>(
            SchemaErrorType::kInvalidSchema, "prefixItems must be an array of objects or booleans"
        );
      }
      auto item_result = Parse(item, "prefix_item");
      if (item_result.IsErr()) return ResultErr(std::move(item_result).UnwrapErr());
      spec.prefix_items.push_back(std::move(item_result).Unwrap());
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
      spec.allow_additional_items = false;
    } else {
      spec.allow_additional_items = true;
      auto items_result = Parse(items_value, "item");
      if (items_result.IsErr()) return ResultErr(std::move(items_result).UnwrapErr());
      spec.additional_items = std::move(items_result).Unwrap();
    }
  } else if (schema.count("unevaluatedItems")) {
    auto unevaluated_items_value = schema.at("unevaluatedItems");
    if (!unevaluated_items_value.is<bool>() && !unevaluated_items_value.is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "unevaluatedItems must be a boolean or an object"
      );
    }
    if (unevaluated_items_value.is<bool>() && !unevaluated_items_value.get<bool>()) {
      spec.allow_additional_items = false;
    } else {
      spec.allow_additional_items = true;
      auto items_result = Parse(unevaluated_items_value, "unevaluated_item");
      if (items_result.IsErr()) return ResultErr(std::move(items_result).UnwrapErr());
      spec.additional_items = std::move(items_result).Unwrap();
    }
  } else if (!config_.strict_mode) {
    spec.allow_additional_items = true;
    spec.additional_items = SchemaSpec::Make(AnySpec{}, "", "any");
  } else {
    spec.allow_additional_items = false;
  }

  if (schema.count("minItems")) {
    if (!schema.at("minItems").is<int64_t>()) {
      return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "minItems must be an integer");
    }
    spec.min_items = std::max(static_cast<int64_t>(0), schema.at("minItems").get<int64_t>());
  }
  if (schema.count("minContains")) {
    if (!schema.at("minContains").is<int64_t>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "minContains must be an integer"
      );
    }
    spec.min_items = std::max(spec.min_items, schema.at("minContains").get<int64_t>());
  }
  if (schema.count("maxItems")) {
    if (!schema.at("maxItems").is<int64_t>() || schema.at("maxItems").get<int64_t>() < 0) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "maxItems must be a non-negative integer"
      );
    }
    spec.max_items = schema.at("maxItems").get<int64_t>();
  }

  if (spec.max_items != -1 && spec.min_items > spec.max_items) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "minItems is greater than maxItems: " + std::to_string(spec.min_items) + " > " +
            std::to_string(spec.max_items)
    );
  }
  if (spec.max_items != -1 && spec.max_items < static_cast<int64_t>(spec.prefix_items.size())) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "maxItems is less than the number of prefixItems: " + std::to_string(spec.max_items) +
            " < " + std::to_string(spec.prefix_items.size())
    );
  }
  if (!spec.allow_additional_items) {
    int64_t prefix_size = static_cast<int64_t>(spec.prefix_items.size());
    if (prefix_size < spec.min_items) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema,
          "minItems is greater than the number of prefixItems, but additional items are not "
          "allowed: " +
              std::to_string(spec.min_items) + " > " + std::to_string(prefix_size)
      );
    }
    if (spec.max_items != -1 && prefix_size > spec.max_items) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema,
          "maxItems is less than the number of prefixItems, but additional items are not "
          "allowed: " +
              std::to_string(spec.max_items) + " < " + std::to_string(prefix_size)
      );
    }
  }
  return ResultOk(std::move(spec));
}

Result<ObjectSpec, SchemaError> SchemaParser::ParseObject(const picojson::object& schema) {
  ObjectSpec spec;

  if (schema.count("properties")) {
    if (!schema.at("properties").is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "properties must be an object"
      );
    }
    auto properties_obj = schema.at("properties").get<picojson::object>();
    for (const auto& key : properties_obj.ordered_keys()) {
      auto prop_result = Parse(properties_obj.at(key), key);
      if (prop_result.IsErr()) return ResultErr(std::move(prop_result).UnwrapErr());
      spec.properties.push_back({key, std::move(prop_result).Unwrap()});
    }
  }

  if (schema.count("required")) {
    if (!schema.at("required").is<picojson::array>()) {
      return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "required must be an array");
    }
    for (const auto& req : schema.at("required").get<picojson::array>()) {
      spec.required.insert(req.get<std::string>());
    }
  }

  if (schema.count("patternProperties")) {
    if (!schema.at("patternProperties").is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "patternProperties must be an object"
      );
    }
    auto pattern_props = schema.at("patternProperties").get<picojson::object>();
    for (const auto& key : pattern_props.ordered_keys()) {
      auto prop_result = Parse(pattern_props.at(key), "pattern_prop");
      if (prop_result.IsErr()) return ResultErr(std::move(prop_result).UnwrapErr());
      spec.pattern_properties.push_back({key, std::move(prop_result).Unwrap()});
    }
  }

  if (schema.count("propertyNames")) {
    if (!schema.at("propertyNames").is<picojson::object>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "propertyNames must be an object"
      );
    }
    auto property_names_obj = schema.at("propertyNames").get<picojson::object>();
    if (property_names_obj.count("type") && property_names_obj.at("type").is<std::string>() &&
        property_names_obj.at("type").get<std::string>() != "string") {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema,
          "propertyNames must be an object that validates string"
      );
    }
    auto prop_names_result = Parse(schema.at("propertyNames"), "property_name", "string");
    if (prop_names_result.IsErr()) return ResultErr(std::move(prop_names_result).UnwrapErr());
    spec.property_names = std::move(prop_names_result).Unwrap();
  }

  spec.allow_additional_properties = !config_.strict_mode;
  if (schema.count("additionalProperties")) {
    auto add_props = schema.at("additionalProperties");
    if (add_props.is<bool>()) {
      spec.allow_additional_properties = add_props.get<bool>();
    } else {
      spec.allow_additional_properties = true;
      auto add_props_result = Parse(add_props, "additional");
      if (add_props_result.IsErr()) return ResultErr(std::move(add_props_result).UnwrapErr());
      spec.additional_properties_schema = std::move(add_props_result).Unwrap();
    }
  }

  spec.allow_unevaluated_properties = true;
  if (schema.count("additionalProperties")) {
    spec.allow_unevaluated_properties = spec.allow_additional_properties;
  } else if (schema.count("unevaluatedProperties")) {
    auto uneval_props = schema.at("unevaluatedProperties");
    if (uneval_props.is<bool>()) {
      spec.allow_unevaluated_properties = uneval_props.get<bool>();
    } else {
      spec.allow_unevaluated_properties = true;
      auto uneval_result = Parse(uneval_props, "unevaluated");
      if (uneval_result.IsErr()) return ResultErr(std::move(uneval_result).UnwrapErr());
      spec.unevaluated_properties_schema = std::move(uneval_result).Unwrap();
    }
  } else if (config_.strict_mode) {
    spec.allow_unevaluated_properties = false;
  }

  if (schema.count("minProperties")) {
    if (!schema.at("minProperties").is<int64_t>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "minProperties must be an integer"
      );
    }
    spec.min_properties = static_cast<int>(schema.at("minProperties").get<int64_t>());
    if (spec.min_properties < 0) {
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
    spec.max_properties = static_cast<int>(schema.at("maxProperties").get<int64_t>());
    if (spec.max_properties < 0) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kUnsatisfiableSchema, "maxProperties must be a non-negative integer"
      );
    }
  }

  if (spec.max_properties != -1 && spec.min_properties > spec.max_properties) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "minProperties is greater than maxProperties: " + std::to_string(spec.min_properties) +
            " > " + std::to_string(spec.max_properties)
    );
  }
  if (spec.max_properties != -1 && static_cast<int>(spec.required.size()) > spec.max_properties) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "maxProperties is less than the number of required properties: " +
            std::to_string(spec.max_properties) + " < " + std::to_string(spec.required.size())
    );
  }
  if (spec.pattern_properties.empty() && !spec.property_names &&
      !spec.allow_additional_properties && !spec.allow_unevaluated_properties &&
      spec.min_properties > static_cast<int>(spec.properties.size())) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema,
        "minProperties is greater than the number of properties, but additional properties aren't "
        "allowed: " +
            std::to_string(spec.min_properties) + " > " + std::to_string(spec.properties.size())
    );
  }
  return ResultOk(std::move(spec));
}

Result<ConstSpec, SchemaError> SchemaParser::ParseConst(const picojson::object& schema) {
  ConstSpec spec;
  spec.json_value = schema.at("const").serialize();
  return ResultOk(std::move(spec));
}

Result<EnumSpec, SchemaError> SchemaParser::ParseEnum(const picojson::object& schema) {
  EnumSpec spec;
  if (!schema.at("enum").is<picojson::array>()) {
    return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "enum must be an array");
  }
  for (const auto& value : schema.at("enum").get<picojson::array>()) {
    spec.json_values.push_back(value.serialize());
  }
  return ResultOk(std::move(spec));
}

Result<RefSpec, SchemaError> SchemaParser::ParseRef(const picojson::object& schema) {
  if (!schema.at("$ref").is<std::string>()) {
    return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "$ref must be a string");
  }
  RefSpec spec;
  spec.uri = schema.at("$ref").get<std::string>();
  return ResultOk(std::move(spec));
}

Result<SchemaSpecPtr, SchemaError> SchemaParser::ResolveRef(
    const std::string& uri, const std::string& rule_name_hint
) {
  if (ref_cache_.count(uri)) return ResultOk(ref_cache_[uri]);

  if (uri == "#") {
    auto placeholder = SchemaSpec::Make(AnySpec{}, "", "root");
    ref_cache_[uri] = placeholder;
    auto result = Parse(root_schema_, "root");
    if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
    auto resolved = std::move(result).Unwrap();
    ref_cache_[uri] = resolved;
    return ResultOk(resolved);
  }

  if (uri.size() < 2 || uri[0] != '#' || uri[1] != '/') {
    XGRAMMAR_LOG(WARNING) << "URI should either be '#' or start with '#/' but got " << uri;
    return ResultOk(SchemaSpec::Make(AnySpec{}, "", "any"));
  }

  std::vector<std::string> parts;
  std::stringstream ss(uri.substr(2));
  std::string part;
  std::string new_rule_name_prefix;
  while (std::getline(ss, part, '/')) {
    if (!part.empty()) parts.push_back(part);
    if (!new_rule_name_prefix.empty()) new_rule_name_prefix += "_";
    for (const auto& c : part) {
      if (std::isalpha(c) || c == '_' || c == '-' || c == '.') new_rule_name_prefix += c;
    }
  }

  auto current = std::cref(root_schema_);
  for (const auto& p : parts) {
    if (!current.get().is<picojson::object>() || !current.get().contains(p)) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "Cannot find field " + p + " in " + uri
      );
    }
    current = current.get().get(p);
  }

  auto result = Parse(current, new_rule_name_prefix);
  if (result.IsErr()) return ResultErr(std::move(result).UnwrapErr());
  auto resolved = std::move(result).Unwrap();
  ref_cache_[uri] = resolved;
  return ResultOk(resolved);
}

Result<AnyOfSpec, SchemaError> SchemaParser::ParseAnyOf(const picojson::object& schema) {
  AnyOfSpec spec;
  auto anyof_key = schema.count("anyOf") ? "anyOf" : "oneOf";
  if (!schema.at(anyof_key).is<picojson::array>()) {
    return ResultErr<SchemaError>(
        SchemaErrorType::kInvalidSchema, std::string(anyof_key) + " must be an array"
    );
  }
  int idx = 0;
  for (const auto& option : schema.at(anyof_key).get<picojson::array>()) {
    auto option_result = Parse(option, "case_" + std::to_string(idx));
    if (option_result.IsErr()) return ResultErr(std::move(option_result).UnwrapErr());
    spec.options.push_back(std::move(option_result).Unwrap());
    ++idx;
  }
  return ResultOk(std::move(spec));
}

Result<AllOfSpec, SchemaError> SchemaParser::ParseAllOf(const picojson::object& schema) {
  AllOfSpec spec;
  if (!schema.at("allOf").is<picojson::array>()) {
    return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "allOf must be an array");
  }
  int idx = 0;
  for (const auto& sub_schema : schema.at("allOf").get<picojson::array>()) {
    auto sub_result = Parse(sub_schema, "all_" + std::to_string(idx));
    if (sub_result.IsErr()) return ResultErr(std::move(sub_result).UnwrapErr());
    spec.schemas.push_back(std::move(sub_result).Unwrap());
    ++idx;
  }
  return ResultOk(std::move(spec));
}

Result<TypeArraySpec, SchemaError> SchemaParser::ParseTypeArray(
    const picojson::object& schema, const std::string& rule_name_hint
) {
  TypeArraySpec spec;
  auto type_array = schema.at("type").get<picojson::array>();
  picojson::object schema_copy = schema;
  if (type_array.empty()) {
    schema_copy.erase("type");
    auto any_result = Parse(picojson::value(schema_copy), rule_name_hint);
    if (any_result.IsErr()) return ResultErr(std::move(any_result).UnwrapErr());
    spec.type_schemas.push_back(std::move(any_result).Unwrap());
    return ResultOk(std::move(spec));
  }
  for (const auto& type : type_array) {
    if (!type.is<std::string>()) {
      return ResultErr<SchemaError>(
          SchemaErrorType::kInvalidSchema, "type must be a string or an array of strings"
      );
    }
    schema_copy["type"] = type;
    auto type_result =
        Parse(picojson::value(schema_copy), rule_name_hint + "_" + type.get<std::string>());
    if (type_result.IsErr()) return ResultErr(std::move(type_result).UnwrapErr());
    spec.type_schemas.push_back(std::move(type_result).Unwrap());
  }
  return ResultOk(std::move(spec));
}

}  // namespace

// ==================== IndentManager Implementation ====================

IndentManager::IndentManager(
    std::optional<int> indent,
    const std::string& separator,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt
)
    : any_whitespace_(any_whitespace),
      enable_newline_(indent.has_value()),
      indent_(indent.value_or(0)),
      separator_(separator),
      total_indent_(0),
      is_first_({true}),
      max_whitespace_cnt_(max_whitespace_cnt) {
  if (max_whitespace_cnt.has_value() && max_whitespace_cnt.value() <= 0) {
    XGRAMMAR_LOG(FATAL) << "max_whitespace_cnt must be positive.";
  }
}

void IndentManager::StartIndent() {
  total_indent_ += indent_;
  is_first_.push_back(true);
}

void IndentManager::EndIndent() {
  total_indent_ -= indent_;
  is_first_.pop_back();
}

std::string IndentManager::StartSeparator() {
  if (any_whitespace_) {
    if (!max_whitespace_cnt_.has_value()) {
      return "[ \\n\\t]*";
    } else {
      return "[ \\n\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
    }
  }
  if (!enable_newline_) {
    return "\"\"";
  }
  return "\"\\n" + std::string(total_indent_, ' ') + "\"";
}

std::string IndentManager::MiddleSeparator() {
  if (any_whitespace_) {
    std::string whitespace_part;
    if (!max_whitespace_cnt_.has_value()) {
      whitespace_part = "[ \\n\\t]*";
    } else {
      whitespace_part = "[ \\n\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
    }
    return whitespace_part + " \"" + separator_ + "\" " + whitespace_part;
  }
  if (!enable_newline_) {
    return "\"" + separator_ + "\"";
  }
  return "\"" + separator_ + "\\n" + std::string(total_indent_, ' ') + "\"";
}

std::string IndentManager::EndSeparator() {
  if (any_whitespace_) {
    if (!max_whitespace_cnt_.has_value()) {
      return "[ \\n\\t]*";
    } else {
      return "[ \\n\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
    }
  }
  if (!enable_newline_) {
    return "\"\"";
  }
  return "\"\\n" + std::string(total_indent_ - indent_, ' ') + "\"";
}

std::string IndentManager::EmptySeparator() {
  if (any_whitespace_) {
    if (!max_whitespace_cnt_.has_value()) {
      return "[ \\n\\t]*";
    } else {
      return "[ \\n\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
    }
  }
  return "\"\"";
}

std::string IndentManager::NextSeparator(bool is_end) {
  if (any_whitespace_) {
    if (is_first_.back() || is_end) {
      is_first_.back() = false;
      if (!max_whitespace_cnt_.has_value()) {
        return "[ \\n\\t]*";
      } else {
        return "[ \\n\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
      }
    } else {
      std::string whitespace_part;
      if (!max_whitespace_cnt_.has_value()) {
        whitespace_part = "[ \\n\\t]*";
      } else {
        whitespace_part = "[ \\n\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
      }
      return whitespace_part + " \"" + separator_ + "\" " + whitespace_part;
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

// ==================== Static Constants ====================

const std::string JSONSchemaConverter::kBasicAny = "basic_any";
const std::string JSONSchemaConverter::kBasicInteger = "basic_integer";
const std::string JSONSchemaConverter::kBasicNumber = "basic_number";
const std::string JSONSchemaConverter::kBasicString = "basic_string";
const std::string JSONSchemaConverter::kBasicBoolean = "basic_boolean";
const std::string JSONSchemaConverter::kBasicNull = "basic_null";
const std::string JSONSchemaConverter::kBasicArray = "basic_array";
const std::string JSONSchemaConverter::kBasicObject = "basic_object";
const std::string JSONSchemaConverter::kBasicEscape = "basic_escape";
const std::string JSONSchemaConverter::kBasicStringSub = "basic_string_sub";

// ==================== JSONSchemaConverter Implementation ====================

JSONSchemaConverter::JSONSchemaConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver
)
    : indent_manager_(
          indent,
          separators.has_value() ? separators->first
                                 : (any_whitespace ? "," : (indent.has_value() ? "," : ", ")),
          any_whitespace,
          max_whitespace_cnt
      ),
      any_whitespace_(any_whitespace),
      max_whitespace_cnt_(max_whitespace_cnt),
      ref_resolver_(std::move(ref_resolver)) {
  std::string colon_sep =
      separators.has_value() ? separators->second : (any_whitespace ? ":" : ": ");
  if (any_whitespace) {
    std::string whitespace_part;
    if (!max_whitespace_cnt_.has_value()) {
      whitespace_part = "[ \\n\\t]*";
    } else {
      whitespace_part = "[ \\n\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
    }
    colon_pattern_ = whitespace_part + " \"" + colon_sep + "\" " + whitespace_part;
  } else {
    colon_pattern_ = "\"" + colon_sep + "\"";
  }
}

std::string JSONSchemaConverter::Convert(const SchemaSpecPtr& spec) {
  AddBasicRules();

  // Register the root rule for circular reference handling
  // This allows $ref: "#" to resolve to "root"
  std::string root_rule_name = ebnf_script_creator_.AllocateRuleName("root");
  uri_to_rule_name_["#"] = root_rule_name;

  // Check if the spec can be directly mapped to an existing rule
  auto cached_rule = GetCache(spec->cache_key);
  if (cached_rule.has_value()) {
    // Root schema matches a basic type, just reference it
    ebnf_script_creator_.AddRuleWithAllocatedName(root_rule_name, cached_rule.value());
  } else {
    // Generate the rule body
    if (!spec->cache_key.empty()) {
      AddCache(spec->cache_key, root_rule_name);
    }
    std::string root_body = GenerateFromSpec(spec, root_rule_name);
    ebnf_script_creator_.AddRuleWithAllocatedName(root_rule_name, root_body);
  }

  return ebnf_script_creator_.GetScript();
}

void JSONSchemaConverter::AddBasicRules() {
  AddHelperRules();

  // Create basic rules with a temporary indent manager for compact format
  auto saved_indent_manager = indent_manager_;
  if (any_whitespace_) {
    indent_manager_ = IndentManager(std::nullopt, ",", true, std::nullopt);
  } else {
    indent_manager_ = IndentManager(std::nullopt, ", ", false, std::nullopt);
  }

  // basic_any - use "{}" as the cache key for empty schema
  auto any_spec = SchemaSpec::Make(AnySpec{}, "{}", kBasicAny);
  std::string any_body = GenerateAny(std::get<AnySpec>(any_spec->spec), kBasicAny);
  ebnf_script_creator_.AddRule(kBasicAny, any_body);
  AddCache("{}", kBasicAny);

  // basic_integer - cache_key matches SchemaParser::ComputeCacheKey for {"type": "integer"}
  constexpr const char* kIntegerCacheKey = "{\"type\":\"integer\"}";
  auto int_spec = SchemaSpec::Make(IntegerSpec{}, kIntegerCacheKey, kBasicInteger);
  std::string int_body = GenerateInteger(std::get<IntegerSpec>(int_spec->spec), kBasicInteger);
  ebnf_script_creator_.AddRule(kBasicInteger, int_body);
  AddCache(kIntegerCacheKey, kBasicInteger);

  // basic_number - cache_key matches SchemaParser::ComputeCacheKey for {"type": "number"}
  constexpr const char* kNumberCacheKey = "{\"type\":\"number\"}";
  auto num_spec = SchemaSpec::Make(NumberSpec{}, kNumberCacheKey, kBasicNumber);
  std::string num_body = GenerateNumber(std::get<NumberSpec>(num_spec->spec), kBasicNumber);
  ebnf_script_creator_.AddRule(kBasicNumber, num_body);
  AddCache(kNumberCacheKey, kBasicNumber);

  // basic_string - cache_key matches SchemaParser::ComputeCacheKey for {"type": "string"}
  constexpr const char* kStringCacheKey = "{\"type\":\"string\"}";
  auto str_spec = SchemaSpec::Make(StringSpec{}, kStringCacheKey, kBasicString);
  std::string str_body = "[\"] " + kBasicStringSub;
  ebnf_script_creator_.AddRule(kBasicString, str_body);
  AddCache(kStringCacheKey, kBasicString);

  // basic_boolean - cache_key matches SchemaParser::ComputeCacheKey for {"type": "boolean"}
  constexpr const char* kBooleanCacheKey = "{\"type\":\"boolean\"}";
  auto bool_spec = SchemaSpec::Make(BooleanSpec{}, kBooleanCacheKey, kBasicBoolean);
  std::string bool_body = GenerateBoolean(std::get<BooleanSpec>(bool_spec->spec), kBasicBoolean);
  ebnf_script_creator_.AddRule(kBasicBoolean, bool_body);
  AddCache(kBooleanCacheKey, kBasicBoolean);

  // basic_null - cache_key matches SchemaParser::ComputeCacheKey for {"type": "null"}
  constexpr const char* kNullCacheKey = "{\"type\":\"null\"}";
  auto null_spec = SchemaSpec::Make(NullSpec{}, kNullCacheKey, kBasicNull);
  std::string null_body = GenerateNull(std::get<NullSpec>(null_spec->spec), kBasicNull);
  ebnf_script_creator_.AddRule(kBasicNull, null_body);
  AddCache(kNullCacheKey, kBasicNull);

  // basic_array - cache_key matches SchemaParser::ComputeCacheKey for {"type": "array"}
  constexpr const char* kArrayCacheKey = "{\"type\":\"array\"}";
  ArraySpec array_spec_val;
  array_spec_val.allow_additional_items = true;
  array_spec_val.additional_items = any_spec;
  auto array_spec = SchemaSpec::Make(std::move(array_spec_val), kArrayCacheKey, kBasicArray);
  std::string array_body = GenerateArray(std::get<ArraySpec>(array_spec->spec), kBasicArray);
  ebnf_script_creator_.AddRule(kBasicArray, array_body);
  AddCache(kArrayCacheKey, kBasicArray);

  // basic_object - cache_key matches SchemaParser::ComputeCacheKey for {"type": "object"}
  constexpr const char* kObjectCacheKey = "{\"type\":\"object\"}";
  ObjectSpec obj_spec_val;
  obj_spec_val.allow_additional_properties = true;
  obj_spec_val.additional_properties_schema = any_spec;
  auto obj_spec = SchemaSpec::Make(std::move(obj_spec_val), kObjectCacheKey, kBasicObject);
  std::string obj_body = GenerateObject(std::get<ObjectSpec>(obj_spec->spec), kBasicObject);
  ebnf_script_creator_.AddRule(kBasicObject, obj_body);
  AddCache(kObjectCacheKey, kBasicObject);

  indent_manager_ = saved_indent_manager;
}

void JSONSchemaConverter::AddHelperRules() {
  ebnf_script_creator_.AddRule(
      kBasicEscape, "[\"\\\\/bfnrt] | \"u\" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]"
  );
  std::string whitespace_part = GetWhitespacePattern();
  ebnf_script_creator_.AddRule(
      kBasicStringSub,
      "(\"\\\"\" | [^\\0-\\x1f\\\"\\\\\\r\\n] " + kBasicStringSub + " | \"\\\\\" " + kBasicEscape +
          " " + kBasicStringSub + ") (= " + whitespace_part + " [,}\\]:])"
  );
}

std::string JSONSchemaConverter::GetWhitespacePattern() const {
  if (!max_whitespace_cnt_.has_value()) {
    return "[ \\n\\t]*";
  } else {
    return "[ \\n\\t]{0," + std::to_string(max_whitespace_cnt_.value()) + "}";
  }
}

std::string JSONSchemaConverter::NextSeparator(bool is_end) {
  return indent_manager_.NextSeparator(is_end);
}

std::string JSONSchemaConverter::GetKeyPattern() const { return kBasicString; }

std::string JSONSchemaConverter::GetBasicAnyRuleName() const { return kBasicAny; }

void JSONSchemaConverter::AddCache(const std::string& key, const std::string& value) {
  if (key.empty()) {
    return;
  }
  rule_cache_manager_.AddCache(key, true, value);
}

std::optional<std::string> JSONSchemaConverter::GetCache(const std::string& key) const {
  if (key.empty()) {
    return std::nullopt;
  }
  return rule_cache_manager_.GetCache(key, true);
}

std::string JSONSchemaConverter::CreateRule(
    const SchemaSpecPtr& spec, const std::string& rule_name_hint
) {
  // Only check cache for basic rules (pre-populated in AddBasicRules)
  // Don't cache other rules to match original behavior
  auto cached = GetCache(spec->cache_key);
  if (cached.has_value()) {
    return cached.value();
  }

  std::string rule_name = ebnf_script_creator_.AllocateRuleName(rule_name_hint);
  std::string rule_body = GenerateFromSpec(spec, rule_name);
  ebnf_script_creator_.AddRuleWithAllocatedName(rule_name, rule_body);

  return rule_name;
}

std::string JSONSchemaConverter::GenerateFromSpec(
    const SchemaSpecPtr& spec, const std::string& rule_name_hint
) {
  return std::visit(
      [this, &rule_name_hint](const auto& s) -> std::string {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, IntegerSpec>) {
          return GenerateInteger(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, NumberSpec>) {
          return GenerateNumber(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, StringSpec>) {
          return GenerateString(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, BooleanSpec>) {
          return GenerateBoolean(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, NullSpec>) {
          return GenerateNull(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, ArraySpec>) {
          return GenerateArray(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, ObjectSpec>) {
          return GenerateObject(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, AnySpec>) {
          return GenerateAny(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, ConstSpec>) {
          return GenerateConst(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, EnumSpec>) {
          return GenerateEnum(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, RefSpec>) {
          return GenerateRef(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, AnyOfSpec>) {
          return GenerateAnyOf(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, AllOfSpec>) {
          return GenerateAllOf(s, rule_name_hint);
        } else if constexpr (std::is_same_v<T, TypeArraySpec>) {
          return GenerateTypeArray(s, rule_name_hint);
        } else {
          XGRAMMAR_LOG(FATAL) << "Unknown spec type";
          return "";
        }
      },
      spec->spec
  );
}

// ==================== Generate Methods ====================

std::string JSONSchemaConverter::GenerateInteger(
    const IntegerSpec& spec, const std::string& rule_name
) {
  std::optional<int64_t> start, end;
  if (spec.minimum.has_value()) {
    start = spec.minimum;
  }
  if (spec.exclusive_minimum.has_value()) {
    start = *spec.exclusive_minimum + 1;
  }
  if (spec.maximum.has_value()) {
    end = spec.maximum;
  }
  if (spec.exclusive_maximum.has_value()) {
    end = *spec.exclusive_maximum - 1;
  }

  if (start.has_value() || end.has_value()) {
    std::string range_regex = GenerateRangeRegex(start, end);
    return RegexToEBNF(range_regex, false);
  }
  return "(\"0\" | \"-\"? [1-9] [0-9]*)";
}

std::string JSONSchemaConverter::GenerateNumber(
    const NumberSpec& spec, const std::string& rule_name
) {
  std::optional<double> start, end;
  if (spec.minimum.has_value()) {
    start = spec.minimum;
  }
  if (spec.exclusive_minimum.has_value()) {
    start = spec.exclusive_minimum;
  }
  if (spec.maximum.has_value()) {
    end = spec.maximum;
  }
  if (spec.exclusive_maximum.has_value()) {
    end = spec.exclusive_maximum;
  }

  if (start.has_value() || end.has_value()) {
    std::string range_regex = GenerateFloatRangeRegex(start, end, 6);
    return RegexToEBNF(range_regex, false);
  }
  // Note: The format must be "-"? ("0" | ...) not ("0" | "-"? ...)
  // The first allows -0, -123, 0, 123
  // The second allows 0, -123, 123 but not -0
  return "\"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?";
}

std::string JSONSchemaConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  // Check for format
  if (spec.format.has_value()) {
    const std::string& format = *spec.format;
    auto regex_pattern = JSONFormatToRegexPattern(format);

    if (regex_pattern.has_value()) {
      std::string converted_regex = RegexToEBNF(regex_pattern.value(), false);
      return "\"\\\"\" " + converted_regex + " \"\\\"\"";
    }
  }

  // Check for pattern
  if (spec.pattern.has_value()) {
    std::string converted_regex = RegexToEBNF(*spec.pattern, false);
    return "\"\\\"\" " + converted_regex + " \"\\\"\"";
  }

  // Check for length constraints
  if (spec.min_length != 0 || spec.max_length != -1) {
    std::string char_pattern = "[^\"\\\\\\r\\n]";
    std::string repetition;
    if (spec.max_length == -1) {
      repetition = "{" + std::to_string(spec.min_length) + ",}";
    } else {
      repetition =
          "{" + std::to_string(spec.min_length) + "," + std::to_string(spec.max_length) + "}";
    }
    return "\"\\\"\" " + char_pattern + repetition + " \"\\\"\"";
  }

  // Default string
  return "[\"] " + kBasicStringSub;
}

std::string JSONSchemaConverter::GenerateBoolean(
    const BooleanSpec& spec, const std::string& rule_name
) {
  return "\"true\" | \"false\"";
}

std::string JSONSchemaConverter::GenerateNull(const NullSpec& spec, const std::string& rule_name) {
  return "\"null\"";
}

std::string JSONSchemaConverter::GenerateArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  indent_manager_.StartIndent();

  auto start_separator = indent_manager_.StartSeparator();
  auto mid_separator = indent_manager_.MiddleSeparator();
  auto end_separator = indent_manager_.EndSeparator();
  auto empty_separator = indent_manager_.EmptySeparator();

  std::vector<std::string> item_rule_names;
  std::string additional_rule_name;

  // Handle prefix items
  for (size_t i = 0; i < spec.prefix_items.size(); ++i) {
    item_rule_names.push_back(
        CreateRule(spec.prefix_items[i], rule_name + "_item_" + std::to_string(i))
    );
  }

  // Handle additional items
  if (spec.allow_additional_items && spec.additional_items) {
    additional_rule_name = CreateRule(spec.additional_items, rule_name + "_additional");
  }

  indent_manager_.EndIndent();

  // Construct the result
  const std::string& left_bracket = EBNFScriptCreator::Str("[");
  const std::string& right_bracket = EBNFScriptCreator::Str("]");

  if (spec.prefix_items.empty()) {
    auto empty_part = EBNFScriptCreator::Concat({left_bracket, empty_separator, right_bracket});
    if (!spec.allow_additional_items) {
      return empty_part;
    } else if (spec.min_items == 0 && spec.max_items == 0) {
      return empty_part;
    } else if (spec.min_items == 0 && spec.max_items != 0) {
      return EBNFScriptCreator::Or(
          {EBNFScriptCreator::Concat(
               {left_bracket,
                start_separator,
                additional_rule_name,
                EBNFScriptCreator::Repeat(
                    EBNFScriptCreator::Concat({mid_separator, additional_rule_name}),
                    0,
                    spec.max_items == -1 ? -1 : static_cast<int>(spec.max_items - 1)
                ),
                end_separator,
                right_bracket}
           ),
           empty_part}
      );
    } else {
      return EBNFScriptCreator::Concat(
          {left_bracket,
           start_separator,
           additional_rule_name,
           EBNFScriptCreator::Repeat(
               EBNFScriptCreator::Concat({mid_separator, additional_rule_name}),
               static_cast<int>(spec.min_items - 1),
               spec.max_items == -1 ? -1 : static_cast<int>(spec.max_items - 1)
           ),
           end_separator,
           right_bracket}
      );
    }
  } else {
    std::vector<std::string> prefix_part;
    for (size_t i = 0; i < item_rule_names.size(); ++i) {
      if (i > 0) {
        prefix_part.push_back(mid_separator);
      }
      prefix_part.push_back(item_rule_names[i]);
    }
    auto prefix_part_str = EBNFScriptCreator::Concat(prefix_part);
    if (!spec.allow_additional_items) {
      return EBNFScriptCreator::Concat(
          {left_bracket, start_separator, prefix_part_str, end_separator, right_bracket}
      );
    } else {
      int64_t min_items = std::max(
          static_cast<int64_t>(0), spec.min_items - static_cast<int64_t>(item_rule_names.size())
      );
      return EBNFScriptCreator::Concat(
          {left_bracket,
           start_separator,
           prefix_part_str,
           EBNFScriptCreator::Repeat(
               EBNFScriptCreator::Concat({mid_separator, additional_rule_name}),
               static_cast<int>(min_items),
               spec.max_items == -1
                   ? -1
                   : static_cast<int>(spec.max_items - static_cast<int64_t>(item_rule_names.size()))
           ),
           end_separator,
           right_bracket}
      );
    }
  }
}

std::string JSONSchemaConverter::FormatPropertyKey(const std::string& key) {
  return "\"\\\"" + key + "\\\"\"";
}

std::string JSONSchemaConverter::FormatProperty(
    const std::string& key, const std::string& value_rule, const std::string& rule_name, int64_t idx
) {
  return FormatPropertyKey(key) + " " + colon_pattern_ + " " + value_rule;
}

std::string JSONSchemaConverter::FormatOtherProperty(
    const std::string& key_pattern,
    const std::string& value_rule,
    const std::string& rule_name,
    const std::string& rule_name_suffix
) {
  return key_pattern + " " + colon_pattern_ + " " + value_rule;
}

std::string JSONSchemaConverter::GetPropertyWithNumberConstraints(
    const std::string& pattern, int min_properties, int max_properties, int already_repeated_times
) {
  if (max_properties != -1 && max_properties == already_repeated_times) {
    return "\"\"";
  }
  int lower = std::max(0, min_properties - already_repeated_times);
  int upper = max_properties == -1 ? -1 : std::max(-1, max_properties - already_repeated_times);
  if (lower == 0 && upper == -1) {
    return "(" + pattern + ")*";
  } else if (lower == 0 && upper == 1) {
    return "(" + pattern + ")?";
  } else if (lower == 1 && upper == 1) {
    return pattern;
  } else {
    return "(" + pattern + "){" + std::to_string(lower) + "," +
           (upper == -1 ? "" : std::to_string(upper)) + "} ";
  }
}

std::string JSONSchemaConverter::GetPartialRuleForProperties(
    const std::vector<ObjectSpec::Property>& properties,
    const std::unordered_set<std::string>& required,
    const SchemaSpecPtr& additional,
    const std::string& rule_name,
    const std::string& additional_suffix,
    int min_properties,
    int max_properties
) {
  if (max_properties == 0) {
    return "";
  }

  std::string first_sep = NextSeparator();
  std::string mid_sep = NextSeparator();
  std::string last_sep = NextSeparator(true);

  std::string res = "";

  std::vector<std::string> prop_patterns;
  for (size_t idx = 0; idx < properties.size(); ++idx) {
    const auto& prop = properties[idx];
    std::string value_rule = CreateRule(prop.schema, rule_name + "_prop_" + std::to_string(idx));
    prop_patterns.push_back(FormatProperty(prop.name, value_rule, rule_name, idx));
  }

  if (min_properties == 0 && max_properties == -1) {
    // Case 1: No property number constraints
    std::vector<std::string> rule_names(properties.size(), "");
    std::vector<uint8_t> is_required(properties.size(), false);
    bool allow_additional = additional != nullptr;

    // Construct the last rule
    std::string additional_prop_pattern;
    if (allow_additional) {
      std::string add_value_rule = CreateRule(additional, rule_name + "_" + additional_suffix);
      additional_prop_pattern =
          FormatOtherProperty(GetKeyPattern(), add_value_rule, rule_name, additional_suffix);
      std::string last_rule_body = "(" + mid_sep + " " + additional_prop_pattern + ")*";
      std::string last_rule_name =
          rule_name + "_part_" + std::to_string(static_cast<int>(properties.size()) - 1);
      last_rule_name = ebnf_script_creator_.AddRule(last_rule_name, last_rule_body);
      rule_names.back() = last_rule_name;
    } else {
      rule_names.back() = "\"\"";
    }

    // Construct 0~(len(properties) - 2) rules
    for (int i = static_cast<int>(properties.size()) - 2; i >= 0; --i) {
      const std::string& prop_pattern = prop_patterns[i + 1];
      const std::string& last_rule_name = rule_names[i + 1];
      std::string cur_rule_body = mid_sep + " " + prop_pattern + " " + last_rule_name;
      if (!required.count(properties[i + 1].name)) {
        cur_rule_body = last_rule_name + " | " + cur_rule_body;
      } else {
        is_required[i + 1] = true;
      }
      std::string cur_rule_name = rule_name + "_part_" + std::to_string(i);
      cur_rule_name = ebnf_script_creator_.AddRule(cur_rule_name, cur_rule_body);
      rule_names[i] = cur_rule_name;
    }
    if (required.count(properties[0].name)) {
      is_required[0] = true;
    }

    // Construct the root rule
    for (size_t i = 0; i < properties.size(); ++i) {
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

    res = first_sep + " (" + res + ") " + last_sep;
  } else if (max_properties == -1) {
    // Case 2: With constraint on the lower bound of the properties number
    const int properties_size = static_cast<int>(properties.size());
    std::vector<std::vector<std::string>> rule_names(properties_size, std::vector<std::string>());
    std::vector<int> key_matched_min(properties_size, 0);
    std::vector<uint8_t> is_required(properties_size, false);
    bool allow_additional = additional != nullptr;

    std::string additional_prop_pattern;
    if (allow_additional) {
      std::string add_value_rule = CreateRule(additional, rule_name + "_" + additional_suffix);
      additional_prop_pattern =
          FormatOtherProperty(GetKeyPattern(), add_value_rule, rule_name, additional_suffix);
    }

    // Get the range of matched properties for each rule
    bool get_first_required = required.count(properties[0].name);
    key_matched_min[0] = 1;
    for (int i = 1; i < properties_size; ++i) {
      if (required.count(properties[i].name)) {
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
    if (required.count(properties[0].name)) {
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

    // Construct the last rule
    if (allow_additional) {
      for (int matched = key_matched_min.back(); matched <= properties_size; ++matched) {
        std::string last_rule_body = GetPropertyWithNumberConstraints(
            mid_sep + " " + additional_prop_pattern, min_properties, max_properties, matched
        );
        std::string last_rule_name = rule_name + "_part_" + std::to_string(properties_size - 1) +
                                     "_" + std::to_string(matched);
        last_rule_name = ebnf_script_creator_.AddRule(last_rule_name, last_rule_body);
        rule_names.back().push_back(last_rule_name);
      }
    } else {
      for (int matched = key_matched_min.back(); matched <= properties_size; ++matched) {
        rule_names.back().push_back("\"\"");
      }
    }

    // Construct 0~(len(properties) - 2) rules
    for (int i = properties_size - 2; i >= 0; --i) {
      const std::string& prop_pattern = prop_patterns[i + 1];
      for (int matched = key_matched_min[i]; matched <= i + 1; ++matched) {
        std::string cur_rule_body;
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

    // Construct root rule
    bool is_first = true;
    for (int i = 0; i < properties_size; ++i) {
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
      res += "(" + additional_prop_pattern + " " +
             GetPropertyWithNumberConstraints(
                 mid_sep + " " + additional_prop_pattern, min_properties, max_properties, 1
             ) +
             ")";
    }

    res = first_sep + " (" + res + ") " + last_sep;
  } else {
    // Case 3: With constraints on both lower & upper bound of the properties number
    const int properties_size = static_cast<int>(properties.size());
    std::vector<std::vector<std::string>> rule_names(properties_size, std::vector<std::string>());
    std::vector<int> key_matched_min(properties_size, 0);
    std::vector<int> key_matched_max(properties_size, properties_size);
    std::vector<uint8_t> is_required(properties_size, false);
    bool allow_additional = additional != nullptr;

    std::string additional_prop_pattern;
    if (allow_additional) {
      std::string add_value_rule = CreateRule(additional, rule_name + "_" + additional_suffix);
      additional_prop_pattern =
          FormatOtherProperty(GetKeyPattern(), add_value_rule, rule_name, additional_suffix);
    }

    // Get the range of matched properties for each rule
    bool get_first_required = required.count(properties[0].name);
    key_matched_min[0] = 1;
    key_matched_max[0] = 1;
    for (int i = 1; i < properties_size; ++i) {
      if (required.count(properties[i].name)) {
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
    if (required.count(properties[0].name)) {
      is_required[0] = true;
    }
    if (allow_additional) {
      key_matched_min.back() = std::max(1, key_matched_min.back());
      key_matched_max.back() = std::min(max_properties, key_matched_max.back());
    } else {
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

    // Construct the last rule
    if (allow_additional) {
      for (int matched = key_matched_min.back(); matched <= key_matched_max.back(); ++matched) {
        std::string last_rule_body = GetPropertyWithNumberConstraints(
            mid_sep + " " + additional_prop_pattern, min_properties, max_properties, matched
        );
        std::string last_rule_name = rule_name + "_part_" + std::to_string(properties_size - 1) +
                                     "_" + std::to_string(matched);
        last_rule_name = ebnf_script_creator_.AddRule(last_rule_name, last_rule_body);
        rule_names.back().push_back(last_rule_name);
      }
    } else {
      for (int matched = key_matched_min.back(); matched <= key_matched_max.back(); ++matched) {
        rule_names.back().push_back("\"\"");
      }
    }

    // Construct 0~(len(properties) - 2) rules
    for (int i = properties_size - 2; i >= 0; --i) {
      const std::string& prop_pattern = prop_patterns[i + 1];
      for (int matched = key_matched_min[i]; matched <= key_matched_max[i]; ++matched) {
        std::string cur_rule_body;
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

    // Construct root rule
    bool is_first = true;
    for (int i = 0; i < properties_size; ++i) {
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
      res += "(" + additional_prop_pattern + " " +
             GetPropertyWithNumberConstraints(
                 mid_sep + " " + additional_prop_pattern, min_properties, max_properties, 1
             ) +
             ")";
    }

    res = first_sep + " (" + res + ") " + last_sep;
  }

  return res;
}

std::string JSONSchemaConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool need_braces
) {
  std::string result = "";
  if (need_braces) {
    result += "\"{\"";
  }

  bool could_be_empty = false;

  // Determine additional property handling
  std::string additional_suffix = "";
  SchemaSpecPtr additional_property;
  if (spec.allow_additional_properties && spec.additional_properties_schema) {
    additional_suffix = "addl";
    additional_property = spec.additional_properties_schema;
  } else if (spec.allow_unevaluated_properties && spec.unevaluated_properties_schema) {
    additional_suffix = "uneval";
    additional_property = spec.unevaluated_properties_schema;
  } else if (spec.allow_additional_properties || spec.allow_unevaluated_properties) {
    additional_suffix = "addl";
    additional_property = SchemaSpec::Make(AnySpec{}, "", "any");
  }

  indent_manager_.StartIndent();

  if (!spec.pattern_properties.empty() || spec.property_names) {
    // Case 1: patternProperties or propertyNames defined
    std::string beg_seq = NextSeparator();

    std::string property_rule_body = "(";
    if (spec.max_properties != 0) {
      if (!spec.pattern_properties.empty()) {
        for (size_t i = 0; i < spec.pattern_properties.size(); ++i) {
          const auto& pp = spec.pattern_properties[i];
          std::string value = CreateRule(pp.schema, rule_name + "_prop_" + std::to_string(i));
          std::string property_pattern = "\"\\\"\"" + RegexToEBNF(pp.pattern, false) + "\"\\\"\" " +
                                         colon_pattern_ + " " + value;
          if (i != 0) {
            property_rule_body += " | ";
          }
          property_rule_body += "(" + beg_seq + " " + property_pattern + ")";
        }
        property_rule_body += ")";
      } else {
        auto key_pattern = CreateRule(spec.property_names, rule_name + "_name");
        property_rule_body +=
            beg_seq + " " + key_pattern + " " + colon_pattern_ + " " + GetBasicAnyRuleName() + ")";
      }

      auto prop_rule_name = ebnf_script_creator_.AllocateRuleName(rule_name + "_prop");
      ebnf_script_creator_.AddRuleWithAllocatedName(prop_rule_name, property_rule_body);

      result +=
          " " + prop_rule_name + " " +
          GetPropertyWithNumberConstraints(
              NextSeparator() + " " + prop_rule_name, spec.min_properties, spec.max_properties, 1
          ) +
          NextSeparator(true);
      could_be_empty = spec.min_properties == 0;
    }
  } else if (!spec.properties.empty()) {
    // Case 2: properties defined
    result += " " + GetPartialRuleForProperties(
                        spec.properties,
                        spec.required,
                        additional_property,
                        rule_name,
                        additional_suffix,
                        spec.min_properties,
                        spec.max_properties
                    );
    could_be_empty = spec.required.empty() && spec.min_properties == 0;
  } else if (additional_property) {
    // Case 3: no properties defined, additional properties allowed
    if (spec.max_properties != 0) {
      std::string add_value_rule =
          CreateRule(additional_property, rule_name + "_" + additional_suffix);
      std::string other_property_pattern =
          FormatOtherProperty(GetKeyPattern(), add_value_rule, rule_name, additional_suffix);
      result += " " + NextSeparator() + " " + other_property_pattern + " ";
      result += GetPropertyWithNumberConstraints(
                    NextSeparator() + " " + other_property_pattern,
                    spec.min_properties,
                    spec.max_properties,
                    1
                ) +
                " " + NextSeparator(true);
    }
    could_be_empty = spec.min_properties == 0;
  }

  indent_manager_.EndIndent();

  if (need_braces) {
    result += " \"}\"";
  }
  if (could_be_empty) {
    std::string whitespace_part = GetWhitespacePattern();
    auto rest = need_braces
                    ? "\"{\" " + std::string(any_whitespace_ ? whitespace_part + " " : "") + "\"}\""
                    : std::string(any_whitespace_ ? whitespace_part : "");
    if (result == "\"{\"  \"}\"" || result == "") {
      result = rest;
    } else {
      result = "(" + result + ") | " + rest;
    }
  }

  return result;
}

std::string JSONSchemaConverter::GenerateAny(const AnySpec& spec, const std::string& rule_name) {
  return kBasicNumber + " | " + kBasicString + " | " + kBasicBoolean + " | " + kBasicNull + " | " +
         kBasicArray + " | " + kBasicObject;
}

std::string JSONSchemaConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  return "\"" + JSONStrToPrintableStr(spec.json_value) + "\"";
}

std::string JSONSchemaConverter::GenerateEnum(const EnumSpec& spec, const std::string& rule_name) {
  std::string result = "";
  for (size_t i = 0; i < spec.json_values.size(); ++i) {
    if (i != 0) {
      result += " | ";
    }
    result += "(\"" + JSONStrToPrintableStr(spec.json_values[i]) + "\")";
  }
  return result;
}

std::string JSONSchemaConverter::GenerateRef(const RefSpec& spec, const std::string& rule_name) {
  // First check if we have a direct URI mapping (for circular references)
  if (uri_to_rule_name_.count(spec.uri)) {
    return uri_to_rule_name_[spec.uri];
  }

  if (!ref_resolver_) {
    XGRAMMAR_LOG(FATAL) << "Ref resolver not set; cannot resolve $ref: " << spec.uri;
  }

  // Derive rule name from URI path (like original URIToRule) so that the same
  // $ref always gets the same rule name, and allocate before resolving to prevent
  // dead recursion when the ref target contains a ref back.
  std::string rule_name_hint = "ref";
  if (spec.uri.size() >= 2 && spec.uri[0] == '#' && spec.uri[1] == '/') {
    std::string new_rule_name_prefix;
    std::stringstream ss(spec.uri.substr(2));
    std::string part;
    while (std::getline(ss, part, '/')) {
      if (!part.empty()) {
        if (!new_rule_name_prefix.empty()) {
          new_rule_name_prefix += "_";
        }
        for (char c : part) {
          if (std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') {
            new_rule_name_prefix += c;
          }
        }
      }
    }
    if (!new_rule_name_prefix.empty()) {
      rule_name_hint = std::move(new_rule_name_prefix);
    }
  }

  std::string allocated_rule_name = ebnf_script_creator_.AllocateRuleName(rule_name_hint);
  uri_to_rule_name_[spec.uri] = allocated_rule_name;

  SchemaSpecPtr resolved = ref_resolver_(spec.uri, allocated_rule_name);
  std::string rule_body = GenerateFromSpec(resolved, allocated_rule_name);
  ebnf_script_creator_.AddRuleWithAllocatedName(allocated_rule_name, rule_body);

  if (!resolved->cache_key.empty()) {
    AddCache(resolved->cache_key, allocated_rule_name);
  }

  return allocated_rule_name;
}

std::string JSONSchemaConverter::GenerateAnyOf(
    const AnyOfSpec& spec, const std::string& rule_name
) {
  std::string result = "";
  for (size_t i = 0; i < spec.options.size(); ++i) {
    if (i != 0) {
      result += " | ";
    }
    result += CreateRule(spec.options[i], rule_name + "_case_" + std::to_string(i));
  }
  return result;
}

std::string JSONSchemaConverter::GenerateAllOf(
    const AllOfSpec& spec, const std::string& rule_name
) {
  if (spec.schemas.size() == 1) {
    return GenerateFromSpec(spec.schemas[0], rule_name + "_case_0");
  }
  XGRAMMAR_LOG(WARNING) << "Support for allOf with multiple options is still ongoing";
  return GenerateFromSpec(SchemaSpec::Make(AnySpec{}, "", "any"), rule_name);
}

std::string JSONSchemaConverter::GenerateTypeArray(
    const TypeArraySpec& spec, const std::string& rule_name
) {
  std::string result = "";
  for (size_t i = 0; i < spec.type_schemas.size(); ++i) {
    if (i != 0) {
      result += " | ";
    }
    result += CreateRule(spec.type_schemas[i], rule_name + "_type_" + std::to_string(i));
  }
  return result;
}

// ==================== Static Helper Methods ====================

std::optional<std::string> JSONSchemaConverter::JSONFormatToRegexPattern(const std::string& format
) {
  static const auto regex_map = []() -> std::unordered_map<std::string, std::string> {
    std::unordered_map<std::string, std::string> m;

    std::string atext = "[\\w!#$%&'*+/=?^`{|}~-]";
    std::string dot_string = "(" + atext + "+(\\." + atext + "+)*)";
    std::string quoted_string =
        "\\\\\"(\\\\[\\x20-\\x7E]|[\\x20\\x21\\x23-\\x5B\\x5D-\\x7E])*\\\\\"";
    std::string domain =
        "([A-Za-z0-9]([\\-A-Za-z0-9]*[A-Za-z0-9])?)((\\.[A-Za-z0-9][\\-A-Za-z0-9]*[A-Za-z0-9])*"
        ")";
    m["email"] = "^(" + dot_string + "|" + quoted_string + ")@" + domain + "$";

    m["date"] = "^(\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\\d|3[01]))$";
    m["time"] =
        "^([01]\\d|2[0-3]):[0-5]\\d:([0-5]\\d|60)(\\.\\d+)?(Z|[+-]([01]\\d|2[0-3]):[0-5]\\d)$";
    m["date-time"] =
        "^(\\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\\d|3[01]))T([01]\\d|2[0-3]):[0-5]\\d:([0-5]\\d|60)("
        "\\.\\d+)?(Z|[+-]([01]\\d|2[0-3]):[0-5]\\d)$";
    m["duration"] =
        "^P((\\d+D|\\d+M(\\d+D)?|\\d+Y(\\d+M(\\d+D)?)?)(T(\\d+S|\\d+M(\\d+S)?|\\d+H(\\d+M(\\d+"
        "S)?"
        ")?))?|T(\\d+S|\\d+M(\\d+S)?|\\d+H(\\d+M(\\d+S)?)?)|\\d+W)$";

    std::string decbyte = "(25[0-5]|2[0-4]\\d|[0-1]?\\d?\\d)";
    m["ipv4"] = "^(" + decbyte + "\\.){3}" + decbyte + "$";

    m["ipv6"] =
        "("
        "([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|"
        "([0-9a-fA-F]{1,4}:){1,7}:|"
        "([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|"
        "([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|"
        "([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|"
        "([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|"
        "([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|"
        "[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|"
        ":((:[0-9a-fA-F]{1,4}){1,7}|:)|"
        "::(ffff(:0{1,4}){0,1}:){0,1}"
        "((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}"
        "(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|"
        "([0-9a-fA-F]{1,4}:){1,4}:"
        "((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}"
        "(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])"
        ")";

    m["hostname"] = "^([a-z0-9]([a-z0-9-]*[a-z0-9])?)(\\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$";
    m["uuid"] = "^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$";

    std::string schema_pat = "[a-zA-Z][a-zA-Z+\\.-]*";
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
    m["uri"] = "^" + schema_pat + ":" + hier_part + query + fragment + "$";

    pchar = "([\\w\\.~!$&'()*+,;=:@-]|%[0-9A-Fa-f][0-9A-Fa-f])";
    query_fragment_char = "([\\w\\.~!$&'()*+,;=:@/\\?-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    query = "(\\?" + query_fragment_char + ")?";
    fragment = "(#" + query_fragment_char + ")?";
    path_abempty = "(/" + pchar + "*)*";
    std::string path_absolute = "/(" + pchar + "+(/" + pchar + "*)*)?";
    std::string segment_nz_nc = "([\\w\\.~!$&'()*+,;=@-]|%[0-9A-Fa-f][0-9A-Fa-f])+";
    std::string path_noscheme = segment_nz_nc + "(/" + pchar + "*)*";
    userinfo = "([\\w\\.~!$&'()*+,;=:-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    host = "([\\w\\.~!$&'()*+,;=-]|%[0-9A-Fa-f][0-9A-Fa-f])*";
    authority = "(" + userinfo + "@)?" + host + "(:\\d*)?";
    std::string relative_part =
        "(//" + authority + path_abempty + "|" + path_absolute + "|" + path_noscheme + ")?";
    m["uri-reference"] = "^" + relative_part + query + fragment + "$";

    std::string literals =
        "([\\x21\\x23-\\x24\\x26\\x28-\\x3B\\x3D\\x3F-\\x5B\\x5D\\x5F\\x61-\\x7A\\x7E]"
        "|%[0-9A-Fa-f][0-9A-Fa-f])";
    std::string op = "[+#\\./;\\?&=,!@|]";
    std::string varchar = "(\\w|%[0-9A-Fa-f][0-9A-Fa-f])";
    std::string varname = varchar + "(\\.?" + varchar + ")*";
    std::string varspec = varname + "(:[1-9]\\d?\\d?\\d?|\\*)?";
    std::string variable_list = varspec + "(," + varspec + ")*";
    std::string expression = "\\{(" + op + ")?" + variable_list + "\\}";
    m["uri-template"] = "^(" + literals + "|" + expression + ")*$";

    m["json-pointer"] = "^(/([\\x00-\\x2E]|[\\x30-\\x7D]|[\\x7F-\\U0010FFFF]|~[01])*)*$";
    m["relative-json-pointer"] =
        "^(0|[1-9][0-9]*)(#|(/([\\x00-\\x2E]|[\\x30-\\x7D]|[\\x7F-\\U0010FFFF]|~[01])*)*)$";

    return m;
  }();

  auto it = regex_map.find(format);
  if (it == regex_map.end()) {
    return std::nullopt;
  }
  return it->second;
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

bool JSONSchemaConverter::StringSpecKey::operator==(const StringSpecKey& other) const {
  return pattern == other.pattern && min_length == other.min_length &&
         max_length == other.max_length && wrapper == other.wrapper;
}

size_t JSONSchemaConverter::StringSpecKeyHash::operator()(const StringSpecKey& key) const {
  return HashCombine(
      std::hash<std::string>()(key.pattern),
      key.min_length,
      key.max_length,
      std::hash<std::string>()(key.wrapper.first),
      std::hash<std::string>()(key.wrapper.second)
  );
}

// ==================== Range Regex Generation (moved from original) ====================

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
        patterns.push_back(start_str);
      } else if (start == digit_min) {
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
              std::string pref = end_str.substr(0, i);
              if (end_str[i] > '0') {
                patterns.push_back(
                    pref +
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
            if (start_str[0] < '9') {
              patterns.push_back(
                  MakePatternForDigitRange(static_cast<char>(start_str[0] + 1), '9', len - 1)
              );
            }
          } else {
            std::string pref = start_str.substr(0, i);
            if (start_str[i] < '9') {
              patterns.push_back(
                  pref +
                  MakePatternForDigitRange(static_cast<char>(start_str[i] + 1), '9', len - i - 1)
              );
            }
          }
        }
        patterns.push_back(start_str);
      } else {
        char start_first_digit = start_str[0];
        char end_first_digit = end_str[0];

        if (end_first_digit - start_first_digit > 1) {
          patterns.push_back(MakePatternForDigitRange(
              static_cast<char>(start_first_digit + 1),
              static_cast<char>(end_first_digit - 1),
              len - 1
          ));
        }

        for (size_t i = 0; i < start_str.size(); i++) {
          if (i == 0) {
            std::string pref = start_str.substr(0, 1);
            if (start_str[1] < '9') {
              patterns.push_back(
                  pref + MakePatternForDigitRange(static_cast<char>(start_str[1] + 1), '9', len - 2)
              );
            }
          } else {
            std::string pref = start_str.substr(0, i);
            if (start_str[i] < '9') {
              patterns.push_back(
                  pref +
                  MakePatternForDigitRange(static_cast<char>(start_str[i] + 1), '9', len - i - 1)
              );
            }
          }
        }
        patterns.push_back(start_str);

        for (size_t i = 0; i < end_str.size(); i++) {
          if (i == 0) {
            std::string pref = end_str.substr(0, 1);
            if (end_str[1] > '0') {
              patterns.push_back(
                  pref + MakePatternForDigitRange('0', static_cast<char>(end_str[1] - 1), len - 2)
              );
            }
          } else {
            std::string pref = end_str.substr(0, i);
            if (end_str[i] > '0') {
              patterns.push_back(
                  pref +
                  MakePatternForDigitRange('0', static_cast<char>(end_str[i] - 1), len - i - 1)
              );
            }
          }
        }
        patterns.push_back(end_str);
      }
    } else if (len == lower_len && len != upper_len) {
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
            std::string pref = start_str.substr(0, i);
            if (start_str[i] < '9') {
              patterns.push_back(
                  pref +
                  MakePatternForDigitRange(static_cast<char>(start_str[i] + 1), '9', len - i - 1)
              );
            }
          }
        }
        patterns.push_back(start_str);
      }
    } else if (len != lower_len && len == upper_len) {
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
            std::string pref = end_str.substr(0, i);
            if (end_str[i] > '0') {
              patterns.push_back(
                  pref +
                  MakePatternForDigitRange('0', static_cast<char>(end_str[i] - 1), len - i - 1)
              );
            }
          }
        }
        patterns.push_back(end_str);
      }
    } else {
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

  if (!start && !end) {
    return "^-?\\d+$";
  }

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

        for (size_t i = 0; i < start_str.size(); i++) {
          if (i == 0) {
            if (start_str[0] < '9') {
              parts.push_back(
                  MakePatternForDigitRange(static_cast<char>(start_str[0] + 1), '9', len - 1)
              );
            }
          } else {
            std::string pref = start_str.substr(0, i);
            if (start_str[i] < '9') {
              parts.push_back(
                  pref +
                  MakePatternForDigitRange(static_cast<char>(start_str[i] + 1), '9', len - i - 1)
              );
            }
          }
        }

        parts.push_back("[1-9]\\d{" + std::to_string(len) + ",}");
      }
    }
  }

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
        parts.push_back(std::to_string(end.value()));

        for (size_t i = 0; i < end_str.size(); i++) {
          if (i == 0) {
            if (end_str[0] > '1') {
              parts.push_back(
                  "-" + MakePatternForDigitRange('1', static_cast<char>(end_str[0] - 1), len - 1)
              );
            }
          } else {
            std::string pref = end_str.substr(0, i);
            if (end_str[i] > '0') {
              parts.push_back(
                  "-" + pref +
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
      return "^()$";
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

std::string JSONSchemaConverter::FormatFloat(double value, int precision) {
  if (value == static_cast<int64_t>(value)) {
    return std::to_string(static_cast<int64_t>(value));
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  std::string result = oss.str();

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
    std::optional<double> start, std::optional<double> end, int precision
) {
  if ((start && end) && (start.value() > end.value())) {
    return "^()$";
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

  if (start && !end) {
    std::string startIntStr = FormatFloat(start.value(), precision);
    parts.push_back(startIntStr);

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
              std::string pref = fracPartStr.substr(0, i);
              if (isStartNegative) {
                if (fracPartStr[i] > '0') {
                  for (char d = '0'; d < fracPartStr[i]; d++) {
                    parts.push_back(
                        intPartStr + "\\." + pref + d + "\\d{0," +
                        std::to_string(precision - i - 1) + "}"
                    );
                  }
                }
              } else {
                for (char d = fracPartStr[i] + 1; d <= '9'; d++) {
                  parts.push_back(
                      intPartStr + "\\." + pref + d + "\\d{0," + std::to_string(precision - i - 1) +
                      "}"
                  );
                }
              }
            }
          }
        }
      }
    }

    if (startInt < INT64_MAX - 1) {
      std::string intRangeRegex = GenerateRangeRegex(startInt + 1, std::nullopt);
      intRangeRegex = intRangeRegex.substr(1, intRangeRegex.length() - 2);
      parts.push_back(intRangeRegex + "(\\.\\d{1," + std::to_string(precision) + "})?");
    }
  } else if (!start && end) {
    std::string endIntStr = FormatFloat(end.value(), precision);
    parts.push_back(endIntStr);

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
                std::string pref = fracPartStr.substr(0, i);
                for (char d = fracPartStr[i] + 1; d <= '9'; d++) {
                  parts.push_back(
                      intPartStr + "\\." + pref + d + "\\d{0," + std::to_string(precision - i - 1) +
                      "}"
                  );
                }
              } else if (fracPartStr[i] > '0') {
                std::string pref = fracPartStr.substr(0, i);
                for (char d = '0'; d < fracPartStr[i]; d++) {
                  parts.push_back(
                      intPartStr + "\\." + pref + d + "\\d{0," + std::to_string(precision - i - 1) +
                      "}"
                  );
                }
              }
            }
          }
        }
      }
    }

    if (endInt > INT64_MIN + 1) {
      std::string intRangeRegex = GenerateRangeRegex(std::nullopt, endInt - 1);
      intRangeRegex = intRangeRegex.substr(1, intRangeRegex.length() - 2);
      parts.push_back(intRangeRegex + "(\\.\\d{1," + std::to_string(precision) + "})?");
    }
  } else if (start && end) {
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
      }
    } else {
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
              std::string pref = fracPartStr.substr(0, i);
              if (isStartNegative) {
                if (fracPartStr[i] > '0') {
                  for (char d = '0'; d < fracPartStr[i]; d++) {
                    parts.push_back(
                        intPartStr + "\\." + pref + d + "\\d{0," +
                        std::to_string(precision - i - 1) + "}"
                    );
                  }
                }
              } else {
                for (char d = fracPartStr[i] + 1; d <= '9'; d++) {
                  parts.push_back(
                      intPartStr + "\\." + pref + d + "\\d{0," + std::to_string(precision - i - 1) +
                      "}"
                  );
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
                std::string pref = fracPartStr.substr(0, i);
                for (char d = fracPartStr[i] + 1; d <= '9'; d++) {
                  parts.push_back(
                      intPartStr + "\\." + pref + d + "\\d{0," + std::to_string(precision - i - 1) +
                      "}"
                  );
                }
              } else if (fracPartStr[i] > '0') {
                std::string pref = fracPartStr.substr(0, i);
                for (char d = '0'; d < fracPartStr[i]; d++) {
                  parts.push_back(
                      intPartStr + "\\." + pref + d + "\\d{0," + std::to_string(precision - i - 1) +
                      "}"
                  );
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

// ==================== Public API Functions ====================

std::string JSONSchemaToEBNF(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    JSONFormat json_format
) {
  picojson::value schema_value;
  std::string err = picojson::parse(schema_value, schema);
  XGRAMMAR_CHECK(err.empty()) << "Failed to parse JSON: " << err
                              << ". The JSON string is:" << schema;
  return JSONSchemaToEBNF(
      schema_value, any_whitespace, indent, separators, strict_mode, max_whitespace_cnt, json_format
  );
}

std::string JSONSchemaToEBNF(
    const picojson::value& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    JSONFormat json_format
) {
  // Parse JSON Schema to SchemaSpec
  SchemaParser parser(schema, {strict_mode, json_format});
  auto spec_result = parser.Parse(schema, "root");
  if (spec_result.IsErr()) {
    XGRAMMAR_LOG(FATAL) << std::move(spec_result).UnwrapErr().what();
  }
  auto spec = std::move(spec_result).Unwrap();

  auto ref_resolver = [&parser](const std::string& uri, const std::string& rule_name_hint) {
    auto r = parser.ResolveRef(uri, rule_name_hint);
    if (r.IsErr()) {
      XGRAMMAR_LOG(FATAL) << std::move(r).UnwrapErr().what();
    }
    return std::move(r).Unwrap();
  };

  // Create converter based on format
  switch (json_format) {
    case JSONFormat::kJSON: {
      JSONSchemaConverter converter(
          indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver
      );
      return converter.Convert(spec);
    }
    case JSONFormat::kXML: {
      XMLToolCallingConverter converter(
          indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver
      );
      return converter.Convert(spec);
    }
    default:
      XGRAMMAR_LOG(FATAL) << "Invalid JSON format: " << static_cast<int>(json_format);
  }
}

// Wrapper functions for testing
std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end) {
  return JSONSchemaConverter::GenerateRangeRegex(start, end);
}

std::string GenerateFloatRangeRegex(std::optional<double> start, std::optional<double> end) {
  return JSONSchemaConverter::GenerateFloatRangeRegex(start, end, 6);
}

std::string QwenXMLToolCallingToEBNF(const std::string& schema) {
  picojson::value json_value;
  std::string err = picojson::parse(json_value, schema);
  if (!err.empty()) {
    XGRAMMAR_LOG(FATAL) << "Failed to parse JSON schema: " << err;
  }
  if (json_value.is<bool>()) {
    XGRAMMAR_LOG(FATAL) << "Expected JSON schema object, got boolean: " << json_value.to_str();
  }
  const auto& schema_obj = json_value.get<picojson::object>();
  if (!schema_obj.count("type")) {
    XGRAMMAR_LOG(FATAL) << "Function calling must have a 'type' field of 'object': "
                        << json_value.to_str();
  }
  if (schema_obj.at("type").get<std::string>() != "object") {
    XGRAMMAR_LOG(FATAL) << "Function calling must have a 'type' field of 'object': "
                        << json_value.to_str();
  }
  return JSONSchemaToEBNF(
      json_value, true, std::nullopt, std::nullopt, true, std::nullopt, JSONFormat::kXML
  );
}

}  // namespace xgrammar
