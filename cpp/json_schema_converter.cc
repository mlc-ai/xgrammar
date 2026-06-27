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
#include <map>
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

  // The range is empty if any lower bound conflicts with any upper bound. An
  // exclusive bound also rules out equality, so it uses ">=" instead of ">".
  auto empty = []() {
    return ResultErr<SchemaError>(
        SchemaErrorType::kUnsatisfiableSchema, "Invalid range: empty range"
    );
  };

  // minimum (x >= min) vs maximum (x <= max).
  if (spec.minimum && spec.maximum && *spec.minimum > *spec.maximum) {
    return empty();
  }
  // minimum (x >= min) vs exclusiveMaximum (x < exclMax).
  if (spec.minimum && spec.exclusive_maximum && *spec.minimum >= *spec.exclusive_maximum) {
    return empty();
  }
  // exclusiveMinimum (x > exclMin) vs maximum (x <= max).
  if (spec.exclusive_minimum && spec.maximum && *spec.exclusive_minimum >= *spec.maximum) {
    return empty();
  }
  // exclusiveMinimum (x > exclMin) vs exclusiveMaximum (x < exclMax).
  if (spec.exclusive_minimum && spec.exclusive_maximum &&
      *spec.exclusive_minimum >= *spec.exclusive_maximum) {
    return empty();
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
  const auto& enum_array = schema.at("enum").get<picojson::array>();
  if (enum_array.empty()) {
    return ResultErr<SchemaError>(SchemaErrorType::kInvalidSchema, "enum array must not be empty");
  }
  for (const auto& value : enum_array) {
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
    RefResolver ref_resolver,
    bool any_order
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
      any_order_(any_order),
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

namespace {

struct TrieNode {
  bool is_terminal = false;
  std::map<char, TrieNode> children;
};

std::string BuildTrieBody(const TrieNode& node) {
  std::string result;
  bool first = true;
  auto add = [&](const std::string& s) {
    if (!first) result += " | ";
    first = false;
    result += s;
  };

  // 1. Close quote - only if no excluded key ends here
  if (!node.is_terminal) {
    add("\"\\\"\"");
  }

  // 2. Negated char class - excludes edge chars + JSON specials
  std::string neg = "[^";
  for (const auto& [c, _] : node.children) {
    if (c == ']' || c == '\\' || c == '^' || c == '-') {
      neg += "\\";
    }
    neg += c;
  }
  neg += "\\0-\\x1f\\\"\\\\\\r\\n]";
  add(neg + " " + JSONSchemaConverter::kBasicStringSub);

  // 3. Escape sequence
  add("\"\\\\\" " + JSONSchemaConverter::kBasicEscape + " " + JSONSchemaConverter::kBasicStringSub);

  // 4. Trie edges - recurse
  for (const auto& [c, child] : node.children) {
    std::string child_body = BuildTrieBody(child);
    std::string char_lit = "\"";
    if (c == '"') {
      char_lit += "\\\"";
    } else if (c == '\\') {
      char_lit += "\\\\";
    } else {
      char_lit += c;
    }
    char_lit += "\"";
    add(char_lit + " " + child_body);
  }

  return "(" + result + ")";
}

}  // namespace

std::string JSONSchemaConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (properties.empty()) {
    return GetKeyPattern();
  }

  // Build trie from property names
  // TODO(linzhang): The trie only excludes the literal unescaped spelling of each property name.
  TrieNode root;
  for (const auto& prop : properties) {
    TrieNode* cur = &root;
    for (char c : prop.name) {
      cur = &cur->children[c];
    }
    cur->is_terminal = true;
  }

  // Generate EBNF body
  std::string inner = BuildTrieBody(root);
  std::string ws = GetWhitespacePattern();
  std::string body = "[\"] (" + inner + ") (= " + ws + " [,}\\]:])";

  return ebnf_script_creator_.AddRule(rule_name + "_addl_key", body);
}

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
    // Smallest integer strictly greater than exclusive_minimum (the parser
    // rejects exclusive_minimum == INT64_MAX, so +1 cannot overflow). When
    // minimum is also present the stricter (larger) lower bound wins.
    int64_t excl_start = *spec.exclusive_minimum + 1;
    start = start.has_value() ? std::max(*start, excl_start) : excl_start;
  }
  if (spec.maximum.has_value()) {
    end = spec.maximum;
  }
  if (spec.exclusive_maximum.has_value()) {
    // Largest integer strictly less than exclusive_maximum (the parser rejects
    // exclusive_maximum == INT64_MIN, so -1 cannot underflow). When maximum is
    // also present the stricter (smaller) upper bound wins.
    int64_t excl_end = *spec.exclusive_maximum - 1;
    end = end.has_value() ? std::min(*end, excl_end) : excl_end;
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
  bool exclusive_start = false;
  bool exclusive_end = false;
  if (spec.minimum.has_value()) {
    start = spec.minimum;
  }
  // When both bounds are present the larger lower bound wins; on a tie the
  // exclusive one is stricter.
  if (spec.exclusive_minimum.has_value() &&
      (!start.has_value() || *spec.exclusive_minimum >= *start)) {
    start = spec.exclusive_minimum;
    exclusive_start = true;
  }
  if (spec.maximum.has_value()) {
    end = spec.maximum;
  }
  if (spec.exclusive_maximum.has_value() && (!end.has_value() || *spec.exclusive_maximum <= *end)) {
    end = spec.exclusive_maximum;
    exclusive_end = true;
  }

  if (start.has_value() || end.has_value()) {
    std::string range_regex =
        GenerateFloatRangeRegex(start, end, 6, exclusive_start, exclusive_end);
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
  return "\"" + JSONStrToPrintableStr(picojson::value(key).serialize()) + "\"";
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

std::string JSONSchemaConverter::GetAnyOrderRuleForProperties(
    const std::vector<ObjectSpec::Property>& properties,
    const std::unordered_set<std::string>& required,
    const SchemaSpecPtr& additional,
    const std::string& rule_name,
    const std::string& additional_suffix,
    int min_properties,
    int max_properties,
    const std::string& additional_prop_pattern_override
) {
  std::string first_sep = NextSeparator();
  std::string mid_sep = NextSeparator();
  std::string last_sep = NextSeparator(true);

  // Build one "item" alternation over every property (any required/optional key) plus any
  // additional/pattern key; any_order does not care which key goes where.
  std::vector<std::string> item_patterns;
  for (size_t idx = 0; idx < properties.size(); ++idx) {
    const auto& prop = properties[idx];
    std::string value_rule = CreateRule(prop.schema, rule_name + "_prop_" + std::to_string(idx));
    item_patterns.push_back(FormatProperty(prop.name, value_rule, rule_name, idx));
  }
  if (additional != nullptr) {
    if (!additional_prop_pattern_override.empty()) {
      item_patterns.push_back(additional_prop_pattern_override);
    } else {
      std::string add_value_rule = CreateRule(additional, rule_name + "_" + additional_suffix);
      item_patterns.push_back(FormatOtherProperty(
          GetKeyPatternExcluding(properties, rule_name),
          add_value_rule,
          rule_name,
          additional_suffix
      ));
    }
  }

  std::string item_body;
  for (size_t i = 0; i < item_patterns.size(); ++i) {
    if (i != 0) {
      item_body += " | ";
    }
    item_body += item_patterns[i];
  }
  std::string item_rule = ebnf_script_creator_.AddRule(rule_name + "_item", item_body);

  // Repeat `item` between n = max(minProperties, #required) and m = maxProperties times; only the
  // count is constrained, not which keys appear.
  int min_count = std::max(min_properties, static_cast<int>(required.size()));
  std::string content =
      item_rule + " " +
      GetPropertyWithNumberConstraints(mid_sep + " " + item_rule, min_count, max_properties, 1);

  return first_sep + " (" + content + ") " + last_sep;
}

std::string JSONSchemaConverter::GetPartialRuleForProperties(
    const std::vector<ObjectSpec::Property>& properties,
    const std::unordered_set<std::string>& required,
    const SchemaSpecPtr& additional,
    const std::string& rule_name,
    const std::string& additional_suffix,
    int min_properties,
    int max_properties,
    const std::string& additional_prop_pattern_override
) {
  if (max_properties == 0) {
    return "";
  }

  if (any_order_) {
    return GetAnyOrderRuleForProperties(
        properties,
        required,
        additional,
        rule_name,
        additional_suffix,
        min_properties,
        max_properties,
        additional_prop_pattern_override
    );
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
      if (!additional_prop_pattern_override.empty()) {
        additional_prop_pattern = additional_prop_pattern_override;
      } else {
        std::string add_value_rule = CreateRule(additional, rule_name + "_" + additional_suffix);
        additional_prop_pattern = FormatOtherProperty(
            GetKeyPatternExcluding(properties, rule_name),
            add_value_rule,
            rule_name,
            additional_suffix
        );
      }
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
      if (!additional_prop_pattern_override.empty()) {
        additional_prop_pattern = additional_prop_pattern_override;
      } else {
        std::string add_value_rule = CreateRule(additional, rule_name + "_" + additional_suffix);
        additional_prop_pattern = FormatOtherProperty(
            GetKeyPatternExcluding(properties, rule_name),
            add_value_rule,
            rule_name,
            additional_suffix
        );
      }
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
      if (!additional_prop_pattern_override.empty()) {
        additional_prop_pattern = additional_prop_pattern_override;
      } else {
        std::string add_value_rule = CreateRule(additional, rule_name + "_" + additional_suffix);
        additional_prop_pattern = FormatOtherProperty(
            GetKeyPatternExcluding(properties, rule_name),
            add_value_rule,
            rule_name,
            additional_suffix
        );
      }
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

  if (!spec.properties.empty() && (!spec.pattern_properties.empty() || spec.property_names)) {
    // Case 1a: properties coexist with patternProperties and/or propertyNames.
    // Use GetPartialRuleForProperties for named properties, and build
    // patternProperties/propertyNames as the additional property pattern override.
    SchemaSpecPtr effective_additional = additional_property;
    std::string effective_suffix = additional_suffix;
    std::string pp_override = "";

    if (!spec.pattern_properties.empty()) {
      // Build patternProperties as additional property alternatives
      std::string pp_body = "";
      for (size_t i = 0; i < spec.pattern_properties.size(); ++i) {
        const auto& pp = spec.pattern_properties[i];
        std::string value = CreateRule(pp.schema, rule_name + "_pp_" + std::to_string(i));
        std::string pp_single = "\"\\\"\"" + RegexToEBNF(pp.pattern, false) + "\"\\\"\" " +
                                colon_pattern_ + " " + value;
        if (i != 0) pp_body += " | ";
        pp_body += pp_single;
      }
      // Merge with existing additionalProperties if present
      if (effective_additional) {
        std::string add_value_rule =
            CreateRule(effective_additional, rule_name + "_" + effective_suffix);
        std::string add_prop =
            FormatOtherProperty(GetKeyPattern(), add_value_rule, rule_name, effective_suffix);
        pp_body += " | " + add_prop;
      }
      // Wrap in parentheses to ensure correct EBNF precedence when | is present
      pp_override = "(" + pp_body + ")";
      if (!effective_additional) {
        effective_additional = SchemaSpec::Make(AnySpec{}, "", "any");
      }
      effective_suffix = "pp";
    } else if (spec.property_names && effective_additional) {
      // propertyNames constrains keys of additional properties.
      // Only apply when additional properties are allowed — when additionalProperties
      // is false, no extra keys beyond named properties should be permitted.
      auto key_pattern = CreateRule(spec.property_names, rule_name + "_name");
      std::string val_rule = CreateRule(effective_additional, rule_name + "_" + effective_suffix);
      pp_override = key_pattern + " " + colon_pattern_ + " " + val_rule;
      effective_suffix = "pn";
    }

    result += " " + GetPartialRuleForProperties(
                        spec.properties,
                        spec.required,
                        effective_additional,
                        rule_name,
                        effective_suffix,
                        spec.min_properties,
                        spec.max_properties,
                        pp_override
                    );
    could_be_empty = spec.required.empty() && spec.min_properties == 0;
  } else if (!spec.pattern_properties.empty() || spec.property_names) {
    // Case 1b: patternProperties or propertyNames without named properties (original logic)
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
    // Case 2: properties defined (no patternProperties/propertyNames)
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
  } else {
    // Case 4: no properties, no additional properties, no pattern properties
    // The object is unconditionally empty.
    could_be_empty = true;
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

  if (result.empty()) {
    return "\"\"";
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
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
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

// ==================== Range Regex Generation ====================

// Stateless utility that turns a numeric range into an anchored regex matching
// exactly the JSON integers / numbers inside it. Every method is static; the
// class exists only to group the helpers and keep the internal ones private.
class NumberGenerator {
 public:
  // Anchored regex matching every integer x with start <= x <= end. Either bound
  // may be std::nullopt for an open side; an empty range yields "^()$". Bounds
  // span the whole int64 range (|INT64_MIN| is handled without negation overflow).
  static std::string IntegerRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end);

  // Anchored regex matching every number in the range, written with up to
  // `precision` fraction digits. `exclusive_start` / `exclusive_end` exclude the
  // boundary value itself (turning >= / <= into > / <). Either bound may be
  // std::nullopt for an open side; an empty range yields "^()$".
  static std::string FloatRangeRegex(
      std::optional<double> start,
      std::optional<double> end,
      int precision,
      bool exclusive_start,
      bool exclusive_end
  );

 private:
  // Regex alternatives for the fraction digits following a decimal point.
  struct FracPatternSet {
    // Each pattern matches a non-empty fraction digit string.
    std::vector<std::string> parts;
    // Whether having no fraction digits at all also satisfies the bound.
    bool include_empty = false;
  };

  // --- Regex fragment primitives ---
  static std::string DigitClass(char lo, char hi);  // one digit in [lo, hi] (or \d)
  static std::string ExactDigits(int k);            // exactly k free digits: \d{k}
  static std::string FreeDigits(int max_count);     // 0..max_count free digits: \d{0,n}
  static std::string OptionalZeros(int max_count);  // 0..max_count zeros: 0{0,n}
  static std::string SomeZeros(int max_count);      // 1..max_count zeros: 0{1,n}
  static bool AllChar(const std::string& s, char c);

  // --- Integer range (operate on non-negative decimal magnitude strings) ---
  static std::string AbsDigits(int64_t v);
  static int CompareDigitStr(const std::string& a, const std::string& b);
  static std::vector<std::string> IntSameLen(const std::string& a, const std::string& b);
  static std::vector<std::string> NumberPatternsStr(const std::string& lo, const std::string& hi);
  static std::string SubRangeRegexStr(const std::string& lo, const std::string& hi);
  static std::vector<std::string> AtLeastPositivePatternsStr(const std::string& v_str);

  // --- Float range ---
  static std::string FormatFloat(double value, int precision);
  // Snaps a non-negative bound to the precision grid in the direction that keeps
  // the range sound: a lower bound rounds up, an upper bound rounds down, so no
  // out-of-range value is ever admitted. Returns the canonical grid string and,
  // via strict_out, whether the boundary value must still be excluded.
  static std::string RoundBoundToGrid(
      double value, int precision, bool is_lower, bool strict_in, bool* strict_out
  );
  // Adds (inc) or subtracts (!inc) one grid step (10^-precision) to a canonical
  // non-negative decimal string, returning the canonical result.
  static std::string AdjustGrid(const std::string& s, int precision, bool inc);
  static void SplitDecimal(const std::string& s, std::string* int_part, std::string* frac_part);
  static int CompareDecimal(
      const std::string& int_a,
      const std::string& frac_a,
      const std::string& int_b,
      const std::string& frac_b
  );
  static std::string StripAnchors(const std::string& regex);
  static int64_t ParseIntCapped(const std::string& digits);
  static FracPatternSet FracGreaterPatterns(const std::string& s, bool strict, int max_len);
  static FracPatternSet FracLessPatterns(const std::string& s, bool strict, int max_len);
  static FracPatternSet FracBetweenPatterns(
      const std::string& a, bool strict_a, const std::string& b, bool strict_b, int max_len
  );
  static std::vector<std::string> PositiveRangeParts(
      const std::string& low,
      bool strict_low,
      const std::optional<std::string>& high,
      bool strict_high,
      int precision
  );
};

// Helpers for integer range regex generation. They operate purely on
// fixed-length decimal digit strings (suffixes may carry leading zeros), so the
// patterns are correct by construction regardless of digit position.

// A regex fragment matching a single digit in [lo, hi].
std::string NumberGenerator::DigitClass(char lo, char hi) {
  if (lo == hi) {
    return std::string(1, lo);
  }
  if (lo == '0' && hi == '9') {
    return "\\d";
  }
  return "[" + std::string(1, lo) + "-" + std::string(1, hi) + "]";
}

// A regex fragment matching k free digits (each 0-9). Empty when k <= 0.
std::string NumberGenerator::ExactDigits(int k) {
  if (k <= 0) {
    return "";
  }
  if (k == 1) {
    return "\\d";
  }
  return "\\d{" + std::to_string(k) + "}";
}

bool NumberGenerator::AllChar(const std::string& s, char c) {
  return std::all_of(s.begin(), s.end(), [c](char ch) { return ch == c; });
}

// Patterns matching every equal-length digit string t with
// value(a) <= value(t) <= value(b). Requires a.size() == b.size() and
// value(a) <= value(b). Partitions t by its first digit:
//   * first digit == a[0]: the suffix must be >= a's suffix (<= 99..9);
//   * first digit strictly between a[0] and b[0]: the suffix is unconstrained;
//   * first digit == b[0]: the suffix must be <= b's suffix (>= 00..0).
// The partition is exact and non-overlapping, so the union is sound and
// complete for [a, b].
std::vector<std::string> NumberGenerator::IntSameLen(const std::string& a, const std::string& b) {
  int n = static_cast<int>(a.size());
  if (a == b) {
    return {a};
  }
  if (n == 1) {
    return {DigitClass(a[0], b[0])};
  }
  if (a[0] == b[0]) {
    std::vector<std::string> res;
    for (auto& p : IntSameLen(a.substr(1), b.substr(1))) {
      res.push_back(std::string(1, a[0]) + p);
    }
    return res;
  }
  // a[0] < b[0]
  std::string a_suf = a.substr(1);
  std::string b_suf = b.substr(1);
  if (AllChar(a_suf, '0') && AllChar(b_suf, '9')) {
    // The whole suffix space is free: collapse to one box pattern.
    if (a[0] == '0' && b[0] == '9') {
      return {ExactDigits(n)};
    }
    return {DigitClass(a[0], b[0]) + ExactDigits(n - 1)};
  }
  std::vector<std::string> res;
  std::string nines(n - 1, '9');
  std::string zeros(n - 1, '0');
  for (auto& p : IntSameLen(a_suf, nines)) {
    res.push_back(std::string(1, a[0]) + p);
  }
  if (b[0] - a[0] >= 2) {
    res.push_back(
        DigitClass(static_cast<char>(a[0] + 1), static_cast<char>(b[0] - 1)) + ExactDigits(n - 1)
    );
  }
  for (auto& p : IntSameLen(zeros, b_suf)) {
    res.push_back(std::string(1, b[0]) + p);
  }
  return res;
}

// Compares two non-negative decimal magnitude strings (no leading zeros except
// "0") by value.
int NumberGenerator::CompareDigitStr(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) {
    return a.size() < b.size() ? -1 : 1;
  }
  if (a < b) {
    return -1;
  }
  return a > b ? 1 : 0;
}

// Patterns matching every integer whose magnitude has value in [lo, hi], where
// lo and hi are non-negative decimal magnitude strings (no leading zeros except
// "0"). An empty range (value(lo) > value(hi)) yields no patterns. Operating on
// strings keeps the whole int64 range representable, including
// |INT64_MIN| = 9223372036854775808, which does not fit in int64.
std::vector<std::string> NumberGenerator::NumberPatternsStr(
    const std::string& lo, const std::string& hi
) {
  std::vector<std::string> patterns;
  if (CompareDigitStr(lo, hi) > 0) {
    return patterns;
  }
  int lo_len = static_cast<int>(lo.size());
  int hi_len = static_cast<int>(hi.size());
  // Split [lo, hi] by digit length; each length yields a same-length segment
  // handled exactly by IntSameLen.
  for (int len = lo_len; len <= hi_len; ++len) {
    std::string a_str = (len == lo_len) ? lo : ("1" + std::string(len - 1, '0'));
    std::string b_str = (len == hi_len) ? hi : std::string(len, '9');
    for (auto& p : IntSameLen(a_str, b_str)) {
      patterns.push_back(p);
    }
  }
  return patterns;
}

// Joins NumberPatternsStr alternatives into a parenthesised regex group.
std::string NumberGenerator::SubRangeRegexStr(const std::string& lo, const std::string& hi) {
  std::vector<std::string> patterns = NumberPatternsStr(lo, hi);
  std::string joined;
  for (size_t i = 0; i < patterns.size(); ++i) {
    if (i > 0) {
      joined += "|";
    }
    joined += patterns[i];
  }
  return "(" + joined + ")";
}

// Patterns matching every integer in [value(v_str), +infinity) for v_str a
// positive magnitude string (no leading zeros). Same-length values come from
// IntSameLen(v_str, 99..9); strictly longer values are any non-zero-led number.
std::vector<std::string> NumberGenerator::AtLeastPositivePatternsStr(const std::string& v_str) {
  int len = static_cast<int>(v_str.size());
  std::vector<std::string> res = IntSameLen(v_str, std::string(len, '9'));
  res.push_back("[1-9]\\d{" + std::to_string(len) + ",}");
  return res;
}

// The magnitude (absolute value) of v as a decimal string. Derived from the
// signed text rather than by negating v, so INT64_MIN is handled correctly.
std::string NumberGenerator::AbsDigits(int64_t v) {
  std::string s = std::to_string(v);
  return (!s.empty() && s[0] == '-') ? s.substr(1) : s;
}

std::string NumberGenerator::IntegerRangeRegex(
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
        // Negatives in [start, -1] are the magnitudes [1, |start|], negated.
        parts.push_back("-" + SubRangeRegexStr("1", AbsDigits(start.value())));
      }
      parts.push_back("0");
      parts.push_back("[1-9]\\d*");
    } else {
      // x >= start with start > 0: same-length values >= start, plus every
      // value with strictly more digits.
      for (auto& p : AtLeastPositivePatternsStr(std::to_string(start.value()))) {
        parts.push_back(p);
      }
    }
  }

  if (!start && end) {
    if (end.value() >= 0) {
      parts.push_back("-[1-9]\\d*");
      parts.push_back("0");
      if (end.value() > 0) {
        parts.push_back(SubRangeRegexStr("1", std::to_string(end.value())));
      }
    } else {
      // x <= end with end < 0: x = -a where a >= |end| > 0, so negate every
      // pattern for the range [|end|, +infinity).
      for (auto& p : AtLeastPositivePatternsStr(AbsDigits(end.value()))) {
        parts.push_back("-" + p);
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
      // Negatives in [neg_start, neg_end] are the magnitudes
      // [|neg_end|, |neg_start|], negated.
      parts.push_back("-" + SubRangeRegexStr(AbsDigits(neg_end), AbsDigits(neg_start)));
    }

    if (range_start <= 0 && range_end >= 0) {
      parts.push_back("0");
    }

    if (range_end > 0) {
      int64_t pos_start = std::max(static_cast<int64_t>(1), range_start);
      parts.push_back(SubRangeRegexStr(std::to_string(pos_start), std::to_string(range_end)));
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

std::string NumberGenerator::FormatFloat(double value, int precision) {
  // Casting a double outside [INT64_MIN, INT64_MAX] (or NaN/Inf) to int64_t is
  // undefined behavior, so range-check before the integer fast path. 2^63 ==
  // 9223372036854775808.0 is exactly representable and one past INT64_MAX, so the
  // upper comparison must be strict.
  if (value >= -9223372036854775808.0 && value < 9223372036854775808.0 &&
      value == static_cast<int64_t>(value)) {
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

std::string NumberGenerator::AdjustGrid(const std::string& s, int precision, bool inc) {
  std::string int_part, frac_part;
  SplitDecimal(s, &int_part, &frac_part);
  // Build the scaled-integer numerator (value * 10^precision) as a digit string.
  // Callers only pass FormatFloat output (<= precision fraction digits); guard
  // the count so a longer string can never wrap the unsigned append count.
  frac_part.append(std::max(0, precision - static_cast<int>(frac_part.size())), '0');
  std::string num = int_part + frac_part;

  if (inc) {
    int i = static_cast<int>(num.size()) - 1;
    for (; i >= 0 && num[i] == '9'; --i) {
      num[i] = '0';
    }
    if (i < 0) {
      num.insert(num.begin(), '1');
    } else {
      num[i]++;
    }
  } else {
    int i = static_cast<int>(num.size()) - 1;
    for (; i >= 0 && num[i] == '0'; --i) {
      num[i] = '9';
    }
    if (i < 0) {
      // Underflow below zero; clamp to zero (does not occur for the bounds the
      // float pipeline feeds in, which are all >= one grid step when decremented).
      num.assign(num.size(), '0');
    } else {
      num[i]--;
    }
  }

  // Re-split into integer and `precision`-digit fraction, then canonicalize.
  while (static_cast<int>(num.size()) <= precision) {
    num.insert(num.begin(), '0');
  }
  std::string new_int = num.substr(0, num.size() - precision);
  std::string new_frac = num.substr(num.size() - precision);
  size_t nz = new_int.find_first_not_of('0');
  new_int = (nz == std::string::npos) ? "0" : new_int.substr(nz);
  size_t lnz = new_frac.find_last_not_of('0');
  new_frac = (lnz == std::string::npos) ? "" : new_frac.substr(0, lnz + 1);
  return new_frac.empty() ? new_int : new_int + "." + new_frac;
}

std::string NumberGenerator::RoundBoundToGrid(
    double value, int precision, bool is_lower, bool strict_in, bool* strict_out
) {
  // FormatFloat rounds to the nearest grid point; if that lands exactly on the
  // bound, keep the original strictness. Otherwise step to the grid point just
  // inside the range so no out-of-range value is admitted, and the boundary is
  // now strictly interior, so it becomes inclusive.
  std::string r = FormatFloat(value, precision);
  double rv = std::stod(r);
  if (rv == value) {
    *strict_out = strict_in;
    return r;
  }
  *strict_out = false;
  if (is_lower && rv < value) {
    // Rounded below a lower bound: move up to the smallest grid point >= value.
    r = AdjustGrid(r, precision, /*inc=*/true);
  } else if (!is_lower && rv > value) {
    // Rounded above an upper bound: move down to the largest grid point <= value.
    r = AdjustGrid(r, precision, /*inc=*/false);
  }
  return r;
}

// Helpers for GenerateFloatRangeRegex. Fraction patterns operate on the
// digit string after the decimal point, compared against a canonical bound
// fraction (canonical: produced by FormatFloat, so no trailing zeros).

// Matches 0 to max_count free digits.
std::string NumberGenerator::FreeDigits(int max_count) {
  if (max_count <= 0) {
    return "";
  }
  return "\\d{0," + std::to_string(max_count) + "}";
}

// Matches 0 to max_count zeros.
std::string NumberGenerator::OptionalZeros(int max_count) {
  if (max_count <= 0) {
    return "";
  }
  return "0{0," + std::to_string(max_count) + "}";
}

// Matches 1 to max_count zeros.
std::string NumberGenerator::SomeZeros(int max_count) {
  return "0{1," + std::to_string(max_count) + "}";
}

// Patterns for fraction strings t (1 <= |t| <= max_len) whose value 0.t is
// greater than 0.s (or equal when !strict). |s| <= max_len.
NumberGenerator::FracPatternSet NumberGenerator::FracGreaterPatterns(
    const std::string& s, bool strict, int max_len
) {
  FracPatternSet result;
  int n = static_cast<int>(s.size());
  // t agrees with s up to position i, then has a larger digit
  for (int i = 0; i < n; ++i) {
    if (s[i] < '9') {
      result.parts.push_back(
          s.substr(0, i) + DigitClass(s[i] + 1, '9') + FreeDigits(max_len - i - 1)
      );
    }
  }
  // t extends s with a nonzero digit (after optional zeros)
  for (int k = 0; n + k + 1 <= max_len; ++k) {
    result.parts.push_back(s + std::string(k, '0') + "[1-9]" + FreeDigits(max_len - n - k - 1));
  }
  if (!strict) {
    // t has the same value as s: s plus optional trailing zeros
    if (n > 0) {
      result.parts.push_back(s + OptionalZeros(max_len - n));
    } else {
      result.include_empty = true;
      if (max_len >= 1) {
        result.parts.push_back(SomeZeros(max_len));
      }
    }
  }
  return result;
}

// Patterns for fraction strings t (1 <= |t| <= max_len) whose value 0.t is
// less than 0.s (or equal when !strict). |s| <= max_len.
NumberGenerator::FracPatternSet NumberGenerator::FracLessPatterns(
    const std::string& s, bool strict, int max_len
) {
  FracPatternSet result;
  int n = static_cast<int>(s.size());
  // t agrees with s up to position i, then has a smaller digit
  for (int i = 0; i < n; ++i) {
    if (s[i] > '0') {
      result.parts.push_back(
          s.substr(0, i) + DigitClass('0', s[i] - 1) + FreeDigits(max_len - i - 1)
      );
    }
  }
  // t is a proper prefix of s plus optional trailing zeros: strictly smaller,
  // since the remaining digits of s contain a nonzero one
  for (int i = 0; i < n; ++i) {
    if (i == 0) {
      if (max_len >= 1) {
        result.parts.push_back(SomeZeros(max_len));
      }
    } else {
      result.parts.push_back(s.substr(0, i) + OptionalZeros(max_len - i));
    }
  }
  if (!strict) {
    // t has the same value as s
    if (n > 0) {
      result.parts.push_back(s + OptionalZeros(max_len - n));
    } else if (max_len >= 1) {
      result.parts.push_back(SomeZeros(max_len));
    }
  }
  result.include_empty = n > 0 || !strict;
  return result;
}

// Patterns for fraction strings t whose value 0.t lies between 0.a and 0.b.
// Requires value(0.a) < value(0.b) and b non-empty.
NumberGenerator::FracPatternSet NumberGenerator::FracBetweenPatterns(
    const std::string& a, bool strict_a, const std::string& b, bool strict_b, int max_len
) {
  FracPatternSet result;
  // Longest common prefix of b and zero-padded a. Always stops before |b|:
  // value(0.a) < value(0.b) implies b is not a prefix of padded a.
  int common_len = 0;
  while (common_len < static_cast<int>(b.size()) &&
         (common_len < static_cast<int>(a.size()) ? a[common_len] : '0') == b[common_len]) {
    ++common_len;
  }
  std::string common = b.substr(0, common_len);
  char digit_a = common_len < static_cast<int>(a.size()) ? a[common_len] : '0';
  char digit_b = b[common_len];

  // a digit strictly between the bounds' digits, then anything
  if (digit_b - digit_a >= 2) {
    result.parts.push_back(
        common + DigitClass(digit_a + 1, digit_b - 1) + FreeDigits(max_len - common_len - 1)
    );
  }
  // lower boundary: t continues with digit_a, the rest must exceed a's suffix
  if (common_len < static_cast<int>(a.size())) {
    FracPatternSet sub_lower =
        FracGreaterPatterns(a.substr(common_len + 1), strict_a, max_len - common_len - 1);
    for (auto& part : sub_lower.parts) {
      result.parts.push_back(common + digit_a + std::move(part));
    }
    if (sub_lower.include_empty) {
      result.parts.push_back(common + std::string(1, digit_a));
    }
  } else {
    // a's value equals value(0.common): only nonzero extensions of
    // common + digit_a ('0') are strictly greater
    FracPatternSet sub_lower = FracGreaterPatterns("", true, max_len - common_len - 1);
    for (auto& part : sub_lower.parts) {
      result.parts.push_back(common + digit_a + std::move(part));
    }
    if (!strict_a) {
      // t has the same value as a
      if (!a.empty()) {
        result.parts.push_back(a + OptionalZeros(max_len - static_cast<int>(a.size())));
      } else {
        result.include_empty = true;
        if (max_len >= 1) {
          result.parts.push_back(SomeZeros(max_len));
        }
      }
    }
  }
  // upper boundary: t continues with digit_b, the rest must stay below b's suffix
  FracPatternSet sub_upper =
      FracLessPatterns(b.substr(common_len + 1), strict_b, max_len - common_len - 1);
  for (auto& part : sub_upper.parts) {
    result.parts.push_back(common + digit_b + std::move(part));
  }
  if (sub_upper.include_empty) {
    result.parts.push_back(common + std::string(1, digit_b));
  }
  return result;
}

// Splits a canonical decimal string from FormatFloat ("12" or "12.34") into
// integer and fraction parts.
void NumberGenerator::SplitDecimal(
    const std::string& s, std::string* int_part, std::string* frac_part
) {
  size_t dot = s.find('.');
  if (dot == std::string::npos) {
    *int_part = s;
    frac_part->clear();
  } else {
    *int_part = s.substr(0, dot);
    *frac_part = s.substr(dot + 1);
  }
}

// Compares the values of two canonical non-negative decimals.
int NumberGenerator::CompareDecimal(
    const std::string& int_a,
    const std::string& frac_a,
    const std::string& int_b,
    const std::string& frac_b
) {
  if (int_a.size() != int_b.size()) {
    return int_a.size() < int_b.size() ? -1 : 1;
  }
  if (int_a != int_b) {
    return int_a < int_b ? -1 : 1;
  }
  size_t max_frac = std::max(frac_a.size(), frac_b.size());
  for (size_t i = 0; i < max_frac; ++i) {
    char da = i < frac_a.size() ? frac_a[i] : '0';
    char db = i < frac_b.size() ? frac_b[i] : '0';
    if (da != db) {
      return da < db ? -1 : 1;
    }
  }
  return 0;
}

// Strips the ^( )$ anchors added by IntegerRangeRegex, keeping the group.
std::string NumberGenerator::StripAnchors(const std::string& regex) {
  return regex.substr(1, regex.size() - 2);
}

int64_t NumberGenerator::ParseIntCapped(const std::string& digits) {
  // `digits` is a canonical non-negative integer string (no leading zeros).
  // Parse it exactly when it fits in int64; clamp to INT64_MAX otherwise (such
  // magnitudes are beyond practical float bounds and double integer precision).
  static const std::string kMaxInt64 = std::to_string(std::numeric_limits<int64_t>::max());
  if (digits.size() > kMaxInt64.size() ||
      (digits.size() == kMaxInt64.size() && digits > kMaxInt64)) {
    return std::numeric_limits<int64_t>::max();
  }
  return std::stoll(digits);
}

// Patterns for unsigned decimals (integer part plus optional fraction of up
// to `precision` digits) within the given bounds. `low` is required and
// non-negative; `high` is optional. Patterns for the value 0 are never
// produced: when low's value is 0 the bound is treated as strict, and the
// caller emits the zero pattern itself.
std::vector<std::string> NumberGenerator::PositiveRangeParts(
    const std::string& low,
    bool strict_low,
    const std::optional<std::string>& high,
    bool strict_high,
    int precision
) {
  std::vector<std::string> parts;
  std::string int_low, frac_low;
  SplitDecimal(low, &int_low, &frac_low);
  if (int_low == "0" && frac_low.empty()) {
    strict_low = true;
  }
  int64_t int_low_value = ParseIntCapped(int_low);
  std::string opt_any_frac = "(\\.\\d{1," + std::to_string(precision) + "})?";

  auto add_with_int_part = [&](const std::string& int_part, const FracPatternSet& set) {
    for (const auto& part : set.parts) {
      parts.push_back(int_part + "\\." + part);
    }
    if (set.include_empty) {
      parts.push_back(int_part);
    }
  };

  if (!high.has_value()) {
    add_with_int_part(int_low, FracGreaterPatterns(frac_low, strict_low, precision));
    // Guard the +1 against int64 overflow (int_low_value may be clamped to
    // INT64_MAX for very large bounds).
    if (int_low_value < std::numeric_limits<int64_t>::max()) {
      parts.push_back(
          StripAnchors(IntegerRangeRegex(int_low_value + 1, std::nullopt)) + opt_any_frac
      );
    }
    return parts;
  }

  std::string int_high, frac_high;
  SplitDecimal(*high, &int_high, &frac_high);
  int64_t int_high_value = ParseIntCapped(int_high);
  int cmp = CompareDecimal(int_low, frac_low, int_high, frac_high);
  if (cmp > 0 || (cmp == 0 && (strict_low || strict_high))) {
    return parts;
  }
  if (cmp == 0) {
    // single representable value, with optional redundant trailing zeros
    if (frac_low.empty()) {
      parts.push_back(int_low + "(\\." + SomeZeros(precision) + ")?");
    } else {
      parts.push_back(
          int_low + "\\." + frac_low + OptionalZeros(precision - static_cast<int>(frac_low.size()))
      );
    }
    return parts;
  }
  if (int_low == int_high) {
    add_with_int_part(
        int_low, FracBetweenPatterns(frac_low, strict_low, frac_high, strict_high, precision)
    );
  } else {
    add_with_int_part(int_low, FracGreaterPatterns(frac_low, strict_low, precision));
    if (int_high_value - int_low_value >= 2) {
      parts.push_back(
          StripAnchors(IntegerRangeRegex(int_low_value + 1, int_high_value - 1)) + opt_any_frac
      );
    }
    add_with_int_part(int_high, FracLessPatterns(frac_high, strict_high, precision));
  }
  return parts;
}

std::string NumberGenerator::FloatRangeRegex(
    std::optional<double> start,
    std::optional<double> end,
    int precision,
    bool exclusive_start,
    bool exclusive_end
) {
  if (start && end) {
    if (start.value() > end.value() ||
        (start.value() == end.value() && (exclusive_start || exclusive_end))) {
      return "^()$";
    }
  }

  if (!start && !end) {
    return "^-?\\d+(\\.\\d{1," + std::to_string(precision) + "})?$";
  }

  std::vector<std::string> parts;

  // Negative values: x is in [start, end] iff -x is in [-end, -start], so the
  // positive-range patterns are reused on the negated bounds and prefixed
  // with '-'.
  bool negatives_in_range = !start.has_value() || start.value() < 0;
  if (negatives_in_range) {
    std::string low = "0";
    bool strict_low = true;
    if (end.has_value() && end.value() < 0) {
      low =
          RoundBoundToGrid(-end.value(), precision, /*is_lower=*/true, exclusive_end, &strict_low);
    }
    std::optional<std::string> high;
    bool strict_high = false;
    if (start.has_value()) {
      high = RoundBoundToGrid(
          -start.value(), precision, /*is_lower=*/false, exclusive_start, &strict_high
      );
    }
    for (auto& part : PositiveRangeParts(low, strict_low, high, strict_high, precision)) {
      parts.push_back("-" + std::move(part));
    }
  }

  bool zero_allowed =
      (!start.has_value() || start.value() < 0 || (start.value() == 0 && !exclusive_start)) &&
      (!end.has_value() || end.value() > 0 || (end.value() == 0 && !exclusive_end));
  if (zero_allowed) {
    parts.push_back("0(\\." + SomeZeros(precision) + ")?");
    // Negative zero written with an all-zero fraction ("-0.0".."-0.000000") also
    // denotes 0. PositiveRangeParts never emits magnitude 0, so add these forms
    // explicitly when the range covers the negative side.
    if (negatives_in_range) {
      parts.push_back("-0(\\." + SomeZeros(precision) + ")");
    }
  }

  // Positive values
  if (!end.has_value() || end.value() > 0) {
    std::string low = "0";
    bool strict_low = true;
    if (start.has_value() && start.value() > 0) {
      low = RoundBoundToGrid(
          start.value(), precision, /*is_lower=*/true, exclusive_start, &strict_low
      );
    }
    std::optional<std::string> high;
    bool strict_high = false;
    if (end.has_value()) {
      high =
          RoundBoundToGrid(end.value(), precision, /*is_lower=*/false, exclusive_end, &strict_high);
    }
    for (auto& part : PositiveRangeParts(low, strict_low, high, strict_high, precision)) {
      parts.push_back(std::move(part));
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

std::string JSONSchemaConverter::GenerateRangeRegex(
    std::optional<int64_t> start, std::optional<int64_t> end
) {
  return NumberGenerator::IntegerRangeRegex(start, end);
}

std::string JSONSchemaConverter::GenerateFloatRangeRegex(
    std::optional<double> start,
    std::optional<double> end,
    int precision,
    bool exclusive_start,
    bool exclusive_end
) {
  return NumberGenerator::FloatRangeRegex(start, end, precision, exclusive_start, exclusive_end);
}

// ==================== Public API Functions ====================

std::string JSONSchemaToEBNF(
    const std::string& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    JSONFormat json_format,
    bool any_order
) {
  picojson::value schema_value;
  std::string err = picojson::parse(schema_value, schema);
  XGRAMMAR_CHECK(err.empty()) << "Failed to parse JSON: " << err
                              << ". The JSON string is:" << schema;
  return JSONSchemaToEBNF(
      schema_value,
      any_whitespace,
      indent,
      separators,
      strict_mode,
      max_whitespace_cnt,
      json_format,
      any_order
  );
}

std::string JSONSchemaToEBNF(
    const picojson::value& schema,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode,
    std::optional<int> max_whitespace_cnt,
    JSONFormat json_format,
    bool any_order
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
          indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver, any_order
      );
      return converter.Convert(spec);
    }
    case JSONFormat::kQwenXML:
    case JSONFormat::kMiniMaxXML:
    case JSONFormat::kDeepSeekXML:
    case JSONFormat::kGlmXML: {
      XMLToolCallingConverter converter(
          indent,
          separators,
          any_whitespace,
          max_whitespace_cnt,
          ref_resolver,
          json_format,
          any_order
      );
      return converter.Convert(spec);
    }
    default:
      XGRAMMAR_LOG(FATAL) << "Invalid JSON format: " << static_cast<int>(json_format);
  }
  XGRAMMAR_UNREACHABLE();
}

// Wrapper functions for testing
std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end) {
  return JSONSchemaConverter::GenerateRangeRegex(start, end);
}

std::string GenerateFloatRangeRegex(
    std::optional<double> start, std::optional<double> end, bool exclusive_start, bool exclusive_end
) {
  return JSONSchemaConverter::GenerateFloatRangeRegex(
      start, end, 6, exclusive_start, exclusive_end
  );
}

std::string QwenXMLToolCallingToEBNF(const std::string& schema, bool any_order) {
  picojson::value json_value;
  std::string err = picojson::parse(json_value, schema);
  if (!err.empty()) {
    XGRAMMAR_LOG(FATAL) << "Failed to parse JSON schema: " << err;
  }
  return JSONSchemaToEBNF(
      json_value,
      true,
      std::nullopt,
      std::nullopt,
      true,
      std::nullopt,
      JSONFormat::kQwenXML,
      any_order
  );
}

std::string MiniMaxXMLToolCallingToEBNF(const std::string& schema, bool any_order) {
  picojson::value json_value;
  std::string err = picojson::parse(json_value, schema);
  if (!err.empty()) {
    XGRAMMAR_LOG(FATAL) << "Failed to parse JSON schema: " << err;
  }
  return JSONSchemaToEBNF(
      json_value,
      true,
      std::nullopt,
      std::nullopt,
      true,
      std::nullopt,
      JSONFormat::kMiniMaxXML,
      any_order
  );
}

std::string DeepSeekXMLToolCallingToEBNF(const std::string& schema, bool any_order) {
  picojson::value json_value;
  std::string err = picojson::parse(json_value, schema);
  if (!err.empty()) {
    XGRAMMAR_LOG(FATAL) << "Failed to parse JSON schema: " << err;
  }
  return JSONSchemaToEBNF(
      json_value,
      true,
      std::nullopt,
      std::nullopt,
      true,
      std::nullopt,
      JSONFormat::kDeepSeekXML,
      any_order
  );
}

std::string GlmXMLToolCallingToEBNF(const std::string& schema, bool any_order) {
  picojson::value json_value;
  std::string err = picojson::parse(json_value, schema);
  if (!err.empty()) {
    XGRAMMAR_LOG(FATAL) << "Failed to parse JSON schema: " << err;
  }
  return JSONSchemaToEBNF(
      json_value,
      true,
      std::nullopt,
      std::nullopt,
      true,
      std::nullopt,
      JSONFormat::kGlmXML,
      any_order
  );
}

}  // namespace xgrammar
