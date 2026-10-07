/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/converter_ext/minimax_m3.cc
 * \brief Implementation of the MiniMax M3 XML Tool Calling converter.
 */
#include <picojson.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

#include "../json_schema_converter_ext.h"
#include "../support/encoding.h"
#include "../support/logging.h"

namespace xgrammar {

namespace {

constexpr const char* kStringCacheKey = "{\"type\":\"string\"}";
constexpr const char* kMiniMaxM3Namespace = "]<]minimax[>[";
constexpr const char* kMiniMaxM3ArrayItemName = "item";
constexpr const char* kMiniMaxM3DynamicName = "minimax_m3_dynamic_name";
constexpr const char* kMiniMaxM3AnyObject = "minimax_m3_any_object";
constexpr const char* kMiniMaxM3AnyArray = "minimax_m3_any_array";

bool IsASCIIWhitespace(uint8_t byte) {
  return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' || byte == '\f' ||
         byte == '\v';
}

bool IsCanonicalUTF8(const std::string& text) {
  for (size_t offset = 0; offset < text.size();) {
    auto [codepoint, num_bytes] = ParseNextUTF8(text.data() + offset);
    if (codepoint == CharHandlingError::kInvalidUTF8 || num_bytes <= 0 ||
        offset + num_bytes > text.size() || (codepoint >= 0xd800 && codepoint <= 0xdfff) ||
        codepoint > 0x10ffff || text.compare(offset, num_bytes, CharToUTF8(codepoint)) != 0) {
      return false;
    }
    offset += num_bytes;
  }
  return true;
}

}  // namespace

MiniMaxM3XMLToolCallingConverter::MiniMaxM3XMLToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    bool any_order
)
    : JSONSchemaConverter(
          indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver, any_order
      ) {}

void MiniMaxM3XMLToolCallingConverter::AddBasicRules() {
  const std::vector<std::string> rule_names = {
      kBasicInteger,
      kBasicNumber,
      kBasicString,
      kBasicBoolean,
      kBasicNull,
      kMiniMaxM3DynamicName,
  };
  for (const auto& name : rule_names) {
    builder_.AddEmptyRule(name);
  }

  builder_.UpdateRuleBody(
      kBasicInteger, JSONSchemaConverter::GenerateInteger(IntegerSpec{}, kBasicInteger)
  );
  AddCache("{\"type\":\"integer\"}", builder_.GetRuleId(kBasicInteger));

  builder_.UpdateRuleBody(
      kBasicNumber, JSONSchemaConverter::GenerateNumber(NumberSpec{}, kBasicNumber)
  );
  AddCache("{\"type\":\"number\"}", builder_.GetRuleId(kBasicNumber));

  builder_.UpdateRuleBody(kBasicString, TagDispatch(false, {kMiniMaxM3Namespace}));
  AddCache(kStringCacheKey, builder_.GetRuleId(kBasicString));

  builder_.UpdateRuleBody(
      kBasicBoolean, JSONSchemaConverter::GenerateBoolean(BooleanSpec{}, kBasicBoolean)
  );
  AddCache("{\"type\":\"boolean\"}", builder_.GetRuleId(kBasicBoolean));

  builder_.UpdateRuleBody(kBasicNull, JSONSchemaConverter::GenerateNull(NullSpec{}, kBasicNull));
  AddCache("{\"type\":\"null\"}", builder_.GetRuleId(kBasicNull));

  builder_.UpdateRuleBody(
      kMiniMaxM3DynamicName,
      Sequence(
          {builder_.AddCharacterClass({{'/', '/'}, {'>', '>'}}, /*is_negative=*/true),
           builder_.AddCharacterClassStar({{'>', '>'}}, /*is_negative=*/true)}
      )
  );
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateInteger(
    const IntegerSpec& spec, const std::string& rule_name
) {
  XGRAMMAR_CHECK(!generating_property_name_)
      << "minimax_m3_xml propertyNames must validate strings";
  return JSONSchemaConverter::GenerateInteger(spec, rule_name);
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateNumber(
    const NumberSpec& spec, const std::string& rule_name
) {
  XGRAMMAR_CHECK(!generating_property_name_)
      << "minimax_m3_xml propertyNames must validate strings";
  return JSONSchemaConverter::GenerateNumber(spec, rule_name);
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  if (generating_property_name_) {
    if (spec.pattern.has_value()) {
      return RegexExpression(*spec.pattern, /*json_string=*/false);
    }
    if (spec.format.has_value()) {
      auto regex = JSONFormatToRegexPattern(*spec.format);
      if (regex.has_value()) {
        return RegexExpression(*regex, /*json_string=*/false, /*force_cfg_expansion=*/true);
      }
    }
    if (spec.min_length != 0 || spec.max_length != -1) {
      return Repeat(
          rule_name + "_characters",
          builder_.AddCharacterClass({{0, 0x10ffff}}),
          spec.min_length,
          spec.max_length
      );
    }
    return RuleRef(kMiniMaxM3DynamicName);
  }

  const bool has_known_format =
      spec.format.has_value() && JSONFormatToRegexPattern(*spec.format).has_value();
  XGRAMMAR_CHECK(
      !spec.pattern.has_value() && !has_known_format && spec.min_length == 0 &&
      spec.max_length == -1
  ) << "String pattern, recognized format, and length constraints are not supported by "
       "minimax_m3_xml because they cannot be combined with the namespace-marker exclusion";
  return RuleRef(kBasicString);
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateBoolean(
    const BooleanSpec& spec, const std::string& rule_name
) {
  XGRAMMAR_CHECK(!generating_property_name_)
      << "minimax_m3_xml propertyNames must validate strings";
  return JSONSchemaConverter::GenerateBoolean(spec, rule_name);
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateNull(
    const NullSpec& spec, const std::string& rule_name
) {
  XGRAMMAR_CHECK(!generating_property_name_)
      << "minimax_m3_xml propertyNames must validate strings";
  return JSONSchemaConverter::GenerateNull(spec, rule_name);
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  XGRAMMAR_CHECK(!generating_property_name_)
      << "minimax_m3_xml propertyNames must validate strings";
  constexpr int64_t kMaxRepeatCount = std::numeric_limits<int32_t>::max();
  XGRAMMAR_CHECK(
      spec.min_items <= kMaxRepeatCount &&
      (spec.max_items == -1 || spec.max_items <= kMaxRepeatCount) &&
      spec.prefix_items.size() <= static_cast<size_t>(kMaxRepeatCount)
  ) << "minimax_m3_xml array bounds exceed the supported range";
  XGRAMMAR_CHECK(!spec.allow_additional_items || spec.additional_items != nullptr)
      << "minimax_m3_xml requires a fixed schema for array items";

  std::vector<int32_t> prefix_items;
  prefix_items.reserve(spec.prefix_items.size());
  for (size_t index = 0; index < spec.prefix_items.size(); ++index) {
    int32_t item_rule_id =
        CreateRule(spec.prefix_items[index], rule_name + "_item_" + std::to_string(index));
    prefix_items.push_back(FormatElement(kMiniMaxM3ArrayItemName, item_rule_id));
  }

  std::optional<int32_t> additional_item;
  if (spec.allow_additional_items) {
    int32_t item_rule_id = CreateRule(spec.additional_items, rule_name + "_additional");
    additional_item = FormatElement(kMiniMaxM3ArrayItemName, item_rule_id);
  }

  int32_t empty = Empty();
  int32_t whitespace = WhitespaceExpression();
  if (prefix_items.empty()) {
    if (!additional_item.has_value() || spec.max_items == 0) {
      return empty;
    }
    int32_t min_items = static_cast<int32_t>(spec.min_items);
    int32_t max_items = spec.max_items == -1 ? -1 : static_cast<int32_t>(spec.max_items);
    int32_t nonempty = Sequence(
        {whitespace,
         *additional_item,
         Repeat(
             rule_name + "_items",
             Sequence({whitespace, *additional_item}),
             std::max(0, min_items - 1),
             max_items == -1 ? -1 : std::max(0, max_items - 1)
         ),
         whitespace}
    );
    return min_items == 0 ? Choice({nonempty, empty}) : nonempty;
  }

  int32_t prefix_count = static_cast<int32_t>(prefix_items.size());
  int32_t tail = empty;
  if (additional_item.has_value()) {
    int32_t min_additional = std::max(0, static_cast<int32_t>(spec.min_items) - prefix_count);
    int32_t max_additional = spec.max_items == -1
                                 ? -1
                                 : std::max(0, static_cast<int32_t>(spec.max_items) - prefix_count);
    tail = Repeat(
        rule_name + "_additional_items",
        Sequence({whitespace, *additional_item}),
        min_additional,
        max_additional
    );
  }

  // A prefixItems entry constrains its position but does not make that position mandatory. Build
  // a linear chain whose suffix can stop once minItems is satisfied.
  for (int32_t index = prefix_count - 2; index >= 0; --index) {
    int32_t emitted_count = index + 1;
    bool can_stop = emitted_count >= spec.min_items;
    bool can_continue = spec.max_items == -1 || emitted_count < spec.max_items;
    int32_t body = empty;
    if (can_continue) {
      int32_t continuation = Sequence({whitespace, prefix_items[index + 1], tail});
      body = can_stop ? Choice({continuation, empty}) : continuation;
    }
    int32_t tail_rule_id =
        builder_.AddRuleWithHint(rule_name + "_prefix_tail_" + std::to_string(index), body);
    tail = RuleRef(tail_rule_id);
  }

  if (spec.max_items == 0) {
    return empty;
  }
  int32_t nonempty = Sequence({whitespace, prefix_items[0], tail, whitespace});
  return spec.min_items == 0 ? Choice({nonempty, empty}) : nonempty;
}

void MiniMaxM3XMLToolCallingConverter::ValidateObject(const ObjectSpec& spec) const {
  const bool has_pattern_properties = !spec.pattern_properties.empty();
  const bool has_runtime_fallback =
      spec.allow_additional_properties || spec.additional_properties_schema != nullptr ||
      spec.allow_unevaluated_properties || spec.unevaluated_properties_schema != nullptr;

  // The generic converter represents these combinations as alternatives. JSON Schema instead
  // requires every schema matching a property name to hold, so accepting them here would
  // under-constrain the generated XML. Keep the initial M3 implementation fail-closed until the
  // converter has a schema-intersection-aware key partition.
  XGRAMMAR_CHECK(spec.pattern_properties.size() <= 1)
      << "minimax_m3_xml does not support multiple patternProperties";
  XGRAMMAR_CHECK(!has_pattern_properties || spec.properties.empty())
      << "minimax_m3_xml does not support combining properties with patternProperties";
  XGRAMMAR_CHECK(!has_pattern_properties || spec.property_names == nullptr)
      << "minimax_m3_xml does not support combining propertyNames with patternProperties";
  XGRAMMAR_CHECK(spec.property_names == nullptr || spec.properties.empty())
      << "minimax_m3_xml does not support combining properties with propertyNames";
  XGRAMMAR_CHECK(!has_pattern_properties || !has_runtime_fallback)
      << "minimax_m3_xml does not support combining patternProperties with additional or "
         "unevaluated properties";

  std::unordered_set<std::string> property_names;
  for (const auto& property : spec.properties) {
    XGRAMMAR_CHECK(property.schema != nullptr)
        << "minimax_m3_xml property must have a fixed schema: " << property.name;
    ValidateElementName(property.name);
    property_names.insert(property.name);
  }
  for (const auto& required : spec.required) {
    XGRAMMAR_CHECK(property_names.count(required) != 0)
        << "minimax_m3_xml required property has no fixed schema: " << required;
  }
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces
) {
  XGRAMMAR_CHECK(!generating_property_name_)
      << "minimax_m3_xml propertyNames must validate strings";
  ValidateObject(spec);

  UniqueKeyScopeContext scope;
  scope.reserved_names.reserve(spec.properties.size());
  for (const auto& property : spec.properties) {
    scope.reserved_names.push_back(property.name);
  }
  unique_key_scope_stack_.push_back(std::move(scope));

  bool saved_any_whitespace = any_whitespace_;
  any_whitespace_ = false;
  int32_t result = JSONSchemaConverter::GenerateObject(spec, rule_name, /*need_braces=*/false);
  any_whitespace_ = saved_any_whitespace;

  scope = std::move(unique_key_scope_stack_.back());
  unique_key_scope_stack_.pop_back();
  if (scope.rule_id >= 0) {
    builder_.UpdateRuleBody(scope.rule_id, result);
    result = RuleRef(scope.rule_id);
  }
  return result;
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateAny(
    const AnySpec& spec, const std::string& rule_name
) {
  if (generating_property_name_) {
    return RuleRef(kMiniMaxM3DynamicName);
  }
  EnsureAnyRules();
  return RuleRef(kBasicAny);
}

void MiniMaxM3XMLToolCallingConverter::EnsureAnyRules() {
  if (any_rules_initialized_) {
    return;
  }
  any_rules_initialized_ = true;

  builder_.AddEmptyRule(kBasicAny);
  int32_t object_rule_id = builder_.AddEmptyRuleWithHint(kMiniMaxM3AnyObject);
  int32_t array_rule_id = builder_.AddEmptyRuleWithHint(kMiniMaxM3AnyArray);

  int32_t dynamic_element = FormatRuntimeElement(
      builder_.GetRuleId(kMiniMaxM3DynamicName), builder_.GetRuleId(kBasicAny), object_rule_id
  );
  builder_.UpdateRuleBody(
      object_rule_id, Repeat(kMiniMaxM3AnyObject + std::string("_elements"), dynamic_element, 1, -1)
  );

  int32_t array_item = FormatElement(kMiniMaxM3ArrayItemName, builder_.GetRuleId(kBasicAny));
  builder_.UpdateRuleBody(
      array_rule_id, Repeat(kMiniMaxM3AnyArray + std::string("_items"), array_item, 1, -1)
  );

  builder_.UpdateRuleBody(
      kBasicAny, Choice({RuleRef(kBasicString), RuleRef(object_rule_id), RuleRef(array_rule_id)})
  );
  AddCache("{}", builder_.GetRuleId(kBasicAny));
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateLiteral(const picojson::value& value) {
  if (value.is<std::string>()) {
    const std::string& text = value.get<std::string>();
    XGRAMMAR_CHECK(text.find(kMiniMaxM3Namespace) == std::string::npos)
        << "A minimax_m3_xml string literal cannot contain the namespace marker";
    return ByteString(text);
  }
  if (value.is<picojson::object>()) {
    const auto& object = value.get<picojson::object>();
    std::vector<int32_t> properties;
    properties.reserve(object.size());
    for (const auto& key : object.ordered_keys()) {
      int32_t value_expr = GenerateLiteral(object.at(key));
      int32_t value_rule_id = builder_.AddRuleWithHint("literal_" + key, value_expr);
      properties.push_back(FormatElement(key, value_rule_id));
    }
    return Sequence(properties);
  }
  if (value.is<picojson::array>()) {
    const auto& array = value.get<picojson::array>();
    std::vector<int32_t> items;
    items.reserve(array.size());
    for (size_t index = 0; index < array.size(); ++index) {
      int32_t value_expr = GenerateLiteral(array[index]);
      int32_t value_rule_id =
          builder_.AddRuleWithHint("literal_item_" + std::to_string(index), value_expr);
      items.push_back(FormatElement(kMiniMaxM3ArrayItemName, value_rule_id));
    }
    return Sequence(items);
  }
  return ByteString(value.serialize());
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  picojson::value value;
  std::string error = picojson::parse(value, spec.json_value);
  XGRAMMAR_CHECK(error.empty()) << "Invalid const JSON value: " << error;
  if (generating_property_name_) {
    XGRAMMAR_CHECK(value.is<std::string>())
        << "minimax_m3_xml propertyNames const must be a string";
    const std::string& name = value.get<std::string>();
    ValidateElementName(name);
    return ByteString(name);
  }
  return GenerateLiteral(value);
}

int32_t MiniMaxM3XMLToolCallingConverter::GenerateEnum(
    const EnumSpec& spec, const std::string& rule_name
) {
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
  std::vector<int32_t> values;
  values.reserve(spec.json_values.size());
  for (const auto& json_value : spec.json_values) {
    picojson::value value;
    std::string error = picojson::parse(value, json_value);
    XGRAMMAR_CHECK(error.empty()) << "Invalid enum JSON value: " << error;
    if (generating_property_name_) {
      XGRAMMAR_CHECK(value.is<std::string>())
          << "minimax_m3_xml propertyNames enum values must be strings";
      const std::string& name = value.get<std::string>();
      ValidateElementName(name);
      values.push_back(ByteString(name));
    } else {
      values.push_back(GenerateLiteral(value));
    }
  }
  return Choice(values);
}

void MiniMaxM3XMLToolCallingConverter::ValidateElementName(const std::string& name) {
  XGRAMMAR_CHECK(!name.empty() && name.front() != '/' && name.find('>') == std::string::npos)
      << "Invalid minimax_m3_xml element name: " << name;
  XGRAMMAR_CHECK(IsCanonicalUTF8(name)) << "minimax_m3_xml element names must be valid UTF-8";
  XGRAMMAR_CHECK(std::any_of(name.begin(), name.end(), [](unsigned char byte) {
    return !IsASCIIWhitespace(byte);
  })) << "minimax_m3_xml element names cannot be blank";
}

int32_t MiniMaxM3XMLToolCallingConverter::FormatElement(
    const std::string& name, int32_t value_rule_id
) {
  ValidateElementName(name);
  return Sequence(
      {ByteString(std::string(kMiniMaxM3Namespace) + "<" + name + ">"),
       RuleRef(value_rule_id),
       ByteString(std::string(kMiniMaxM3Namespace) + "</" + name + ">")}
  );
}

int32_t MiniMaxM3XMLToolCallingConverter::FormatRuntimeElement(
    int32_t name_rule_id,
    int32_t content_rule_id,
    int32_t unique_key_scope_rule_id,
    const std::vector<std::string>& reserved_names
) {
  return builder_.AddDynamicTag(
      {std::string(kMiniMaxM3Namespace) + "<",
       name_rule_id,
       ">",
       content_rule_id,
       std::string(kMiniMaxM3Namespace) + "</",
       ">",
       unique_key_scope_rule_id,
       reserved_names}
  );
}

int32_t MiniMaxM3XMLToolCallingConverter::FormatProperty(
    const std::string& key,
    int32_t value_rule_id,
    const std::string& rule_name,
    int64_t idx,
    const SchemaSpecPtr& schema
) {
  return FormatElement(key, value_rule_id);
}

int32_t MiniMaxM3XMLToolCallingConverter::FormatOtherProperty(
    int32_t key_pattern_expr,
    int32_t value_rule_id,
    const std::string& rule_name,
    const std::string& rule_name_suffix,
    const SchemaSpecPtr& schema
) {
  XGRAMMAR_DCHECK(!unique_key_scope_stack_.empty());
  int32_t name_rule_id =
      builder_.AddRuleWithHint(rule_name + "_" + rule_name_suffix + "_name", key_pattern_expr);
  auto& scope = unique_key_scope_stack_.back();
  if (scope.rule_id < 0) {
    scope.rule_id = builder_.AddEmptyRuleWithHint(rule_name + "_unique_keys");
  }
  return FormatRuntimeElement(name_rule_id, value_rule_id, scope.rule_id, scope.reserved_names);
}

int32_t MiniMaxM3XMLToolCallingConverter::CreatePatternKeyRule(
    const std::string& pattern, const std::string& rule_name_hint
) {
  return builder_.AddRuleWithHint(rule_name_hint, RegexExpression(pattern, /*json_string=*/false));
}

int32_t MiniMaxM3XMLToolCallingConverter::CreatePropertyNamesKeyRule(
    const SchemaSpecPtr& spec, const std::string& rule_name_hint
) {
  int32_t rule_id = builder_.AddEmptyRuleWithHint(rule_name_hint);
  std::string rule_name = builder_.GetRule(rule_id).name;
  bool saved_generating_property_name = generating_property_name_;
  generating_property_name_ = true;
  builder_.UpdateRuleBody(rule_id, GenerateFromSpec(spec, rule_name));
  generating_property_name_ = saved_generating_property_name;
  return rule_id;
}

std::string MiniMaxM3XMLToolCallingConverter::GetKeyPattern() const {
  return kMiniMaxM3DynamicName;
}

int32_t MiniMaxM3XMLToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  return RuleRef(kMiniMaxM3DynamicName);
}

void MiniMaxM3XMLToolCallingConverter::AddCache(const std::string& key, int32_t rule_id) {
  if (!key.empty()) {
    rule_cache_manager_.AddCache(key, !generating_property_name_, rule_id);
  }
}

std::optional<int32_t> MiniMaxM3XMLToolCallingConverter::GetCache(const std::string& key) const {
  if (key.empty()) {
    return std::nullopt;
  }
  return rule_cache_manager_.GetCache(key, !generating_property_name_);
}

std::string MiniMaxM3XMLToolCallingConverter::RefCacheKey(const std::string& uri) const {
  return generating_property_name_ ? "1:" + uri : uri;
}

std::string MiniMaxM3XMLToolCallingConverter::NextSeparator(bool is_end) {
  return GetWhitespacePattern();
}

}  // namespace xgrammar
