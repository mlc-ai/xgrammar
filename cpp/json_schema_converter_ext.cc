/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.cc
 * \brief Implementation of extended format converters.
 */
#include "json_schema_converter_ext.h"

#include <picojson.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "support/encoding.h"
#include "support/logging.h"

namespace xgrammar {

namespace {

bool IsXMLIdentifierChar(char c, bool is_first) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
         (!is_first && c >= '0' && c <= '9');
}

template <typename Children>
std::vector<GrammarBuilder::CharacterClassElement> XMLIdentifierCharClassExcluding(
    const Children& children, bool is_first
) {
  std::vector<GrammarBuilder::CharacterClassElement> chars;
  auto add_if_missing = [&](char c) {
    if (!children.count(c)) {
      chars.push_back({c, c});
    }
  };
  for (char c = 'A'; c <= 'Z'; ++c) {
    add_if_missing(c);
  }
  add_if_missing('_');
  for (char c = 'a'; c <= 'z'; ++c) {
    add_if_missing(c);
  }
  if (!is_first) {
    for (char c = '0'; c <= '9'; ++c) {
      add_if_missing(c);
    }
  }
  return chars;
}

std::vector<GrammarBuilder::CharacterClassElement> XMLIdentifierContinuationChars() {
  return {{'a', 'z'}, {'A', 'Z'}, {'0', '9'}, {'_', '_'}};
}

constexpr const char* kStringCacheKey = "{\"type\":\"string\"}";
constexpr const char* kObjectCacheKey = "{\"type\":\"object\"}";

struct ElementNameTrieNode {
  bool is_terminal = false;
  std::map<TCodepoint, ElementNameTrieNode> children;
};

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

std::vector<GrammarBuilder::CharacterClassElement> CodepointComplement(
    std::vector<TCodepoint> excluded
) {
  constexpr TCodepoint kMaxCodepoint = 0x10ffff;
  std::sort(excluded.begin(), excluded.end());
  excluded.erase(std::unique(excluded.begin(), excluded.end()), excluded.end());

  std::vector<GrammarBuilder::CharacterClassElement> result;
  int64_t range_begin = 0;
  for (TCodepoint codepoint : excluded) {
    XGRAMMAR_DCHECK(codepoint >= 0 && codepoint <= kMaxCodepoint);
    if (range_begin < codepoint) {
      result.push_back(
          {static_cast<TCodepoint>(range_begin), static_cast<TCodepoint>(codepoint - 1)}
      );
    }
    range_begin = static_cast<int64_t>(codepoint) + 1;
  }
  if (range_begin <= kMaxCodepoint) {
    result.push_back({static_cast<TCodepoint>(range_begin), kMaxCodepoint});
  }
  return result;
}

}  // namespace

// Static constants
const std::string XMLToolCallingConverter::kXMLString = "xml_string";
const std::string XMLToolCallingConverter::kXMLAny = "xml_any";
const std::string XMLToolCallingConverter::kXMLObject = "xml_object";
const std::string XMLToolCallingConverter::kXMLVariableName = "xml_variable_name";
const std::unordered_map<JSONFormat, XMLToolCallingConverter::XMLDialectConfig>
    XMLToolCallingConverter::kDialectConfigMap = {
        {JSONFormat::kQwenXML,
         {{"<parameter=", ">", "", "</parameter>", "", false},
          /*recursive=*/false,
          /*array_item_name=*/"",
          /*pad_values_with_whitespace=*/true,
          /*string_terminator=*/"</parameter>"}},
        {JSONFormat::kMiniMaxXML,
         {{"<parameter name=\"", "\">", "", "</parameter>", "", false},
          /*recursive=*/false,
          /*array_item_name=*/"",
          /*pad_values_with_whitespace=*/true,
          /*string_terminator=*/"</parameter>"}},
        {JSONFormat::kDeepSeekXML,
         {{"<｜DSML｜parameter name=\"",
           "",
           "",
           // TODO(Linzhang): We do not validate the string's value, and we accept both.
           "</｜DSML｜parameter>",
           "",
           false},
          /*recursive=*/false,
          /*array_item_name=*/"",
          /*pad_values_with_whitespace=*/true,
          /*string_terminator=*/"</｜DSML｜parameter>"}},
        {JSONFormat::kGlmXML,
         {{"<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "", false},
          /*recursive=*/false,
          /*array_item_name=*/"",
          /*pad_values_with_whitespace=*/true,
          /*string_terminator=*/"</arg_value>"}},
        {JSONFormat::kCohereXML,
         {{"<cofl:value", ">", "", "</cofl:value>", "", false},
          /*recursive=*/false,
          /*array_item_name=*/"",
          /*pad_values_with_whitespace=*/true,
          /*string_terminator=*/"</cofl:value>"}},
        {JSONFormat::kKimiK3XML,
         {{"<|open|>argument key=\"",
           "",
           "",
           // The key suffix (type attribute and <|sep|>) is generated in XMLKeySuffix.
           "<|close|>argument<|sep|>",
           "",
           false},
          /*recursive=*/false,
          /*array_item_name=*/"",
          /*pad_values_with_whitespace=*/true,
          /*string_terminator=*/"<|close|>argument<|sep|>"}},
        {JSONFormat::kMiniMaxM3XML,
         {{"]<]minimax[>[<", ">", "", "]<]minimax[>[</", ">", true},
          /*recursive=*/true,
          /*array_item_name=*/"item",
          /*pad_values_with_whitespace=*/false,
          /*string_terminator=*/"]<]minimax[>["}},
};

XMLToolCallingConverter::XMLToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    JSONFormat json_format,
    bool any_order
)
    : JSONSchemaConverter(
          indent, separators, any_whitespace, max_whitespace_cnt, ref_resolver, any_order
      ),
      json_format_(json_format),
      nested_object_level_(0),
      dialect_(kDialectConfigMap.at(json_format)) {}

Grammar XMLToolCallingConverter::Convert(const SchemaSpecPtr& spec) {
  nested_object_level_ = 0;
  generating_property_name_ = false;
  unique_key_scope_stack_.clear();
  return JSONSchemaConverter::Convert(spec);
}

std::string XMLToolCallingConverter::XMLValue(const std::string& json_value) const {
  picojson::value value;
  std::string error = picojson::parse(value, json_value);
  if (error.empty() && value.is<std::string>()) {
    return value.get<std::string>();
  }
  return json_value;
}

int32_t XMLToolCallingConverter::XMLKeySuffix(const std::optional<std::string>& pinned_type) {
  if (json_format_ == JSONFormat::kDeepSeekXML) {
    return Sequence(
        {ByteString("\" string=\""),
         Choice({ByteString("true"), ByteString("false")}),
         ByteString("\">")}
    );
  }
  if (json_format_ == JSONFormat::kKimiK3XML) {
    // A declared property carries exactly the type its value grammar is rendered with, so the
    // parser decodes the value back to the schema's type. Free-form keys have no single schema
    // type, so they keep the full set.
    int32_t type_expr = pinned_type.has_value() ? ByteString(*pinned_type)
                                                : Choice(
                                                      {ByteString("string"),
                                                       ByteString("number"),
                                                       ByteString("integer"),
                                                       ByteString("boolean"),
                                                       ByteString("object"),
                                                       ByteString("array"),
                                                       ByteString("null")}
                                                  );
    return Sequence({ByteString("\" type=\""), type_expr, ByteString("\"<|sep|>")});
  }
  return ByteString(dialect_.property.open_suffix);
}

std::optional<std::string> XMLToolCallingConverter::KimiK3TypeAttr(const SchemaSpecPtr& spec) {
  if (spec == nullptr) {
    return std::nullopt;
  }
  // The type name a single JSON value is rendered with, following the model's _xtml_type.
  auto type_of_json_value = [](const std::string& json_value) -> std::optional<std::string> {
    picojson::value value;
    if (!picojson::parse(value, json_value).empty()) {
      return std::nullopt;
    }
    if (value.is<std::string>()) return "string";
    if (value.is<bool>()) return "boolean";
    if (value.is<double>()) return "number";
    if (value.is<picojson::null>()) return "null";
    if (value.is<picojson::object>()) return "object";
    if (value.is<picojson::array>()) return "array";
    return std::nullopt;
  };

  return std::visit(
      [&](auto&& arg) -> std::optional<std::string> {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, StringSpec>) {
          return "string";
        } else if constexpr (std::is_same_v<T, IntegerSpec> || std::is_same_v<T, NumberSpec>) {
          // _xtml_type renders every int and float as "number"; it never emits "integer".
          return "number";
        } else if constexpr (std::is_same_v<T, BooleanSpec>) {
          return "boolean";
        } else if constexpr (std::is_same_v<T, NullSpec>) {
          return "null";
        } else if constexpr (std::is_same_v<T, ArraySpec>) {
          return "array";
        } else if constexpr (std::is_same_v<T, ObjectSpec>) {
          return "object";
        } else if constexpr (std::is_same_v<T, ConstSpec>) {
          return type_of_json_value(arg.json_value);
        } else if constexpr (std::is_same_v<T, EnumSpec>) {
          // Only pin the attribute when every alternative renders with the same type.
          std::optional<std::string> common;
          for (const auto& json_value : arg.json_values) {
            auto type_name = type_of_json_value(json_value);
            if (!type_name.has_value()) return std::nullopt;
            if (!common.has_value()) {
              common = type_name;
            } else if (*common != *type_name) {
              return std::nullopt;
            }
          }
          return common;
        } else {
          // Any, $ref and the combinators may render as more than one type; keep them open.
          return std::nullopt;
        }
      },
      spec->spec
  );
}

void XMLToolCallingConverter::AddBasicRules() {
  if (dialect_.recursive) {
    AddRecursiveXMLBasicRules();
  } else {
    AddRootOnlyXMLBasicRules();
  }
}

void XMLToolCallingConverter::AddRootOnlyXMLBasicRules() {
  // First add JSON basic rules. These should be in the inner layer of the XML format.
  XGRAMMAR_DCHECK(nested_object_level_ == 0);
  // The nested part, true json format, is at level 2.
  nested_object_level_ = 2;
  JSONSchemaConverter::AddBasicRules({kXMLString, kXMLAny, kXMLObject, kXMLVariableName});

  auto any_spec = SchemaSpec::Make(AnySpec{}, "{}", kBasicAny);

  // The outer part, xml format, is at level 1.
  nested_object_level_ = 1;
  // Add XML string rule
  builder_.UpdateRuleBody(kXMLString, TagDispatch(false, {dialect_.string_terminator}));
  AddCache(kStringCacheKey, builder_.GetRuleId(kXMLString));

  // Add XML any rule
  builder_.UpdateRuleBody(kXMLAny, GenerateAny(AnySpec{}, kXMLAny));
  AddCache("{}", builder_.GetRuleId(kXMLAny));

  // Reset the nested object level to 0, which is the root level.
  nested_object_level_ = 0;

  // Add XML object rule
  ObjectSpec xml_object_spec;
  xml_object_spec.allow_additional_properties = true;
  xml_object_spec.additional_properties_schema = any_spec;
  builder_.UpdateRuleBody(kXMLObject, GenerateObject(xml_object_spec, kXMLObject));
  AddCache(kObjectCacheKey, builder_.GetRuleId(kXMLObject));

  // Add XML variable name rule
  builder_.UpdateRuleBody(
      kXMLVariableName,
      Sequence(
          {builder_.AddCharacterClass({{'a', 'z'}, {'A', 'Z'}, {'_', '_'}}),
           builder_.AddCharacterClassStar({{'a', 'z'}, {'A', 'Z'}, {'0', '9'}, {'_', '_'}})}
      )
  );
}

void XMLToolCallingConverter::AddRecursiveXMLBasicRules() {
  XGRAMMAR_DCHECK(nested_object_level_ == 0);
  const std::vector<std::string> rule_names = {
      kBasicInteger,
      kBasicNumber,
      kBasicBoolean,
      kBasicNull,
      kXMLString,
      kXMLAny,
      kXMLVariableName,
      "xml_dynamic_element",
      "xml_dynamic_children",
  };
  for (const auto& name : rule_names) {
    builder_.AddEmptyRule(name);
  }

  constexpr const char* kIntegerCacheKey = "{\"type\":\"integer\"}";
  constexpr const char* kNumberCacheKey = "{\"type\":\"number\"}";
  constexpr const char* kBooleanCacheKey = "{\"type\":\"boolean\"}";
  constexpr const char* kNullCacheKey = "{\"type\":\"null\"}";

  nested_object_level_ = 1;
  builder_.UpdateRuleBody(kBasicInteger, GenerateInteger(IntegerSpec{}, kBasicInteger));
  AddCache(kIntegerCacheKey, builder_.GetRuleId(kBasicInteger));
  builder_.UpdateRuleBody(kBasicNumber, GenerateNumber(NumberSpec{}, kBasicNumber));
  AddCache(kNumberCacheKey, builder_.GetRuleId(kBasicNumber));
  builder_.UpdateRuleBody(kXMLString, TagDispatch(false, {dialect_.string_terminator}));
  AddCache(kStringCacheKey, builder_.GetRuleId(kXMLString));
  builder_.UpdateRuleBody(kBasicBoolean, GenerateBoolean(BooleanSpec{}, kBasicBoolean));
  AddCache(kBooleanCacheKey, builder_.GetRuleId(kBasicBoolean));
  builder_.UpdateRuleBody(kBasicNull, GenerateNull(NullSpec{}, kBasicNull));
  AddCache(kNullCacheKey, builder_.GetRuleId(kBasicNull));

  builder_.UpdateRuleBody(
      kXMLVariableName,
      Sequence(
          {builder_.AddCharacterClass({{'/', '/'}, {'>', '>'}}, /*is_negative=*/true),
           builder_.AddCharacterClassStar({{'>', '>'}}, /*is_negative=*/true)}
      )
  );
  builder_.UpdateRuleBody(
      "xml_dynamic_element",
      builder_.AddDynamicTag(
          {dialect_.property.open_prefix,
           builder_.GetRuleId(kXMLVariableName),
           dialect_.property.open_suffix,
           builder_.GetRuleId(kXMLAny),
           dialect_.property.close_prefix,
           dialect_.property.close_suffix,
           builder_.GetRuleId("xml_dynamic_children")}
      )
  );
  builder_.UpdateRuleBody(
      "xml_dynamic_children", Repeat("xml_dynamic_elements", RuleRef("xml_dynamic_element"), 1, -1)
  );
  builder_.UpdateRuleBody(kXMLAny, Choice({RuleRef(kXMLString), RuleRef("xml_dynamic_children")}));
  AddCache("{}", builder_.GetRuleId(kXMLAny));

  nested_object_level_ = 0;
  AddCache(kIntegerCacheKey, builder_.GetRuleId(kBasicInteger));
  AddCache(kNumberCacheKey, builder_.GetRuleId(kBasicNumber));
  AddCache(kStringCacheKey, builder_.GetRuleId(kXMLString));
  AddCache(kBooleanCacheKey, builder_.GetRuleId(kBasicBoolean));
  AddCache(kNullCacheKey, builder_.GetRuleId(kBasicNull));
}

bool XMLToolCallingConverter::IsXMLLayer() const {
  return dialect_.recursive || nested_object_level_ <= 1;
}

bool XMLToolCallingConverter::IsInnerCacheLayer() const {
  return dialect_.recursive ? nested_object_level_ > 0 : nested_object_level_ > 1;
}

std::string XMLToolCallingConverter::GetKeyPattern() const {
  return IsXMLLayer() ? kXMLVariableName : kBasicString;
}

std::string XMLToolCallingConverter::GetBasicAnyRuleName() const {
  if (IsXMLLayer()) {
    return kXMLAny;
  }
  return kBasicAny;
}

int32_t XMLToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (IsXMLLayer() && !dialect_.recursive) {
    return RuleRef(GetKeyPattern());
  }
  if (dialect_.recursive) {
    if (properties.empty()) {
      return RuleRef(GetKeyPattern());
    }

    ElementNameTrieNode root;
    for (const auto& property : properties) {
      ElementNameTrieNode* node = &root;
      for (size_t offset = 0; offset < property.name.size();) {
        auto [codepoint, num_bytes] = ParseNextUTF8(property.name.data() + offset);
        XGRAMMAR_CHECK(codepoint != CharHandlingError::kInvalidUTF8 && num_bytes > 0)
            << "Recursive XML element names must be valid UTF-8";
        node = &node->children[codepoint];
        offset += num_bytes;
      }
      node->is_terminal = true;
    }

    const int32_t arbitrary_tail =
        builder_.AddCharacterClassStar({{'>', '>'}}, /*is_negative=*/true);
    auto build_trie = [&](auto&& self, const ElementNameTrieNode& node, bool is_root) -> int32_t {
      std::vector<int32_t> choices;
      if (!is_root && !node.is_terminal) {
        choices.push_back(Empty());
      }

      std::vector<TCodepoint> excluded = {'>'};
      if (is_root) {
        excluded.push_back('/');
      }
      for (const auto& [codepoint, child] : node.children) {
        static_cast<void>(child);
        excluded.push_back(codepoint);
      }
      choices.push_back(Sequence(
          {builder_.AddCharacterClass(CodepointComplement(std::move(excluded))), arbitrary_tail}
      ));
      for (const auto& [codepoint, child] : node.children) {
        choices.push_back(Sequence(
            {builder_.AddCharacterClass({{codepoint, codepoint}}), self(self, child, false)}
        ));
      }
      return Choice(choices);
    };
    return build_trie(build_trie, root, true);
  }
  return JSONSchemaConverter::GetKeyPatternExcluding(properties, rule_name);
}

std::string XMLToolCallingConverter::NextSeparator(bool is_end) {
  if (IsXMLLayer()) {
    return GetWhitespacePattern();
  }
  return JSONSchemaConverter::NextSeparator(is_end);
}

int32_t XMLToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  if (dialect_.recursive && generating_property_name_) {
    if (spec.pattern.has_value()) {
      return RegexExpression(*spec.pattern);
    }
    if (spec.format.has_value()) {
      auto regex = JSONFormatToRegexPattern(*spec.format);
      if (regex.has_value()) {
        return RegexExpression(*regex, false, true);
      }
    }
    if (spec.min_length != 0 || spec.max_length != -1) {
      XGRAMMAR_CHECK(spec.max_length == -1 || spec.max_length >= 1)
          << "Recursive XML element names cannot be empty";
      const int min_length = std::max(spec.min_length, 1);
      const int max_tail = spec.max_length == -1 ? -1 : spec.max_length - 1;
      return Sequence(
          {builder_.AddCharacterClass({{'/', '/'}, {'>', '>'}}, /*is_negative=*/true),
           Repeat(
               rule_name + "_characters",
               builder_.AddCharacterClass({{'>', '>'}}, /*is_negative=*/true),
               min_length - 1,
               max_tail
           )}
      );
    }
    return RuleRef(kXMLVariableName);
  }

  if (IsXMLLayer()) {
    if (dialect_.recursive) {
      const bool has_known_format =
          spec.format.has_value() && JSONFormatToRegexPattern(*spec.format).has_value();
      XGRAMMAR_CHECK(
          !spec.pattern.has_value() && !has_known_format && spec.min_length == 0 &&
          spec.max_length == -1
      ) << "String pattern, recognized format, and length constraints are not supported by "
           "recursive XML dialects because they cannot be combined with the namespace-marker "
           "exclusion";
      return RuleRef(kXMLString);
    }

    if (!spec.pattern.has_value() && !spec.format.has_value() && spec.min_length == 0 &&
        spec.max_length == -1) {
      return RuleRef(kXMLString);
    }
    if (spec.format.has_value()) {
      auto regex = JSONFormatToRegexPattern(*spec.format);
      if (regex.has_value()) {
        return RegexExpression(*regex, false, /*force_cfg_expansion=*/true);
      }
    }
    if (spec.pattern.has_value()) {
      return RegexExpression(*spec.pattern, false, /*force_cfg_expansion=*/true);
    }
    return Repeat(
        rule_name + "_characters",
        builder_.AddCharacterClass({{0, 0x10ffff}}),
        spec.min_length,
        spec.max_length
    );
  }
  return JSONSchemaConverter::GenerateString(spec, rule_name);
}

int32_t XMLToolCallingConverter::GenerateAny(const AnySpec& spec, const std::string& rule_name) {
  if (dialect_.recursive && generating_property_name_) {
    return RuleRef(kXMLVariableName);
  }
  if (dialect_.recursive) {
    return RuleRef(kXMLAny);
  }
  if (nested_object_level_ == 0) {
    return RuleRef(kXMLObject);
  }
  if (nested_object_level_ == 1) {
    return Choice({RuleRef(kXMLString), RuleRef(kBasicArray), RuleRef(kBasicObject)});
  }
  return JSONSchemaConverter::GenerateAny(spec, rule_name);
}

int32_t XMLToolCallingConverter::GenerateArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  nested_object_level_++;
  int32_t result = dialect_.array_item_name.empty()
                       ? JSONSchemaConverter::GenerateArray(spec, rule_name)
                       : GenerateRepeatedElementArray(spec, rule_name);
  nested_object_level_--;
  return result;
}

int32_t XMLToolCallingConverter::GenerateRepeatedElementArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  std::vector<int32_t> prefix_items;
  prefix_items.reserve(spec.prefix_items.size());
  for (size_t index = 0; index < spec.prefix_items.size(); ++index) {
    int32_t item_rule_id =
        CreateRule(spec.prefix_items[index], rule_name + "_item_" + std::to_string(index));
    prefix_items.push_back(FormatElement(dialect_.property, dialect_.array_item_name, item_rule_id)
    );
  }

  std::optional<int32_t> additional_item;
  if (spec.allow_additional_items && spec.additional_items) {
    int32_t item_rule_id = CreateRule(spec.additional_items, rule_name + "_additional");
    additional_item = FormatElement(dialect_.property, dialect_.array_item_name, item_rule_id);
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

  std::vector<int32_t> elements = {whitespace};
  for (size_t index = 0; index < prefix_items.size(); ++index) {
    if (index != 0) {
      elements.push_back(whitespace);
    }
    elements.push_back(prefix_items[index]);
  }
  if (additional_item.has_value()) {
    int32_t prefix_count = static_cast<int32_t>(prefix_items.size());
    int32_t min_additional = std::max(0, static_cast<int32_t>(spec.min_items) - prefix_count);
    int32_t max_additional = spec.max_items == -1
                                 ? -1
                                 : std::max(0, static_cast<int32_t>(spec.max_items) - prefix_count);
    elements.push_back(Repeat(
        rule_name + "_additional_items",
        Sequence({whitespace, *additional_item}),
        min_additional,
        max_additional
    ));
  }
  elements.push_back(whitespace);
  return Sequence(elements);
}

int32_t XMLToolCallingConverter::GenerateLiteral(const picojson::value& value) {
  if (value.is<std::string>()) {
    const std::string& text = value.get<std::string>();
    XGRAMMAR_CHECK(text.find(dialect_.string_terminator) == std::string::npos)
        << "A recursive XML string literal cannot contain the dialect namespace marker";
    return ByteString(text);
  }
  if (value.is<picojson::object>()) {
    const auto& object = value.get<picojson::object>();
    std::vector<int32_t> properties;
    properties.reserve(object.size());
    for (const auto& key : object.ordered_keys()) {
      int32_t value_expr = GenerateLiteral(object.at(key));
      int32_t value_rule_id = builder_.AddRuleWithHint("literal_" + key, value_expr);
      properties.push_back(FormatElement(dialect_.property, key, value_rule_id));
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
      items.push_back(FormatElement(dialect_.property, dialect_.array_item_name, value_rule_id));
    }
    return Sequence(items);
  }
  return ByteString(value.serialize());
}

int32_t XMLToolCallingConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  if (!dialect_.recursive) {
    return IsXMLLayer() ? ByteString(XMLValue(spec.json_value))
                        : JSONSchemaConverter::GenerateConst(spec, rule_name);
  }
  picojson::value value;
  std::string error = picojson::parse(value, spec.json_value);
  if (dialect_.recursive && generating_property_name_) {
    XGRAMMAR_CHECK(error.empty() && value.is<std::string>())
        << "propertyNames const must be a string";
    ValidateElementName(value.get<std::string>());
    return ByteString(value.get<std::string>());
  }
  if (dialect_.recursive && IsXMLLayer()) {
    XGRAMMAR_CHECK(error.empty()) << "Invalid const JSON value: " << error;
    return GenerateLiteral(value);
  }
  return JSONSchemaConverter::GenerateConst(spec, rule_name);
}

int32_t XMLToolCallingConverter::GenerateEnum(const EnumSpec& spec, const std::string& rule_name) {
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
  if (IsXMLLayer() && !dialect_.recursive) {
    std::vector<int32_t> values;
    values.reserve(spec.json_values.size());
    for (const auto& value : spec.json_values) {
      values.push_back(ByteString(XMLValue(value)));
    }
    return Choice(values);
  }
  if (dialect_.recursive && IsXMLLayer()) {
    std::vector<int32_t> alternatives;
    alternatives.reserve(spec.json_values.size());
    for (const auto& json_value : spec.json_values) {
      picojson::value value;
      std::string error = picojson::parse(value, json_value);
      XGRAMMAR_CHECK(error.empty()) << "Invalid enum JSON value: " << error;
      if (dialect_.recursive && generating_property_name_) {
        XGRAMMAR_CHECK(value.is<std::string>()) << "propertyNames enum values must be strings";
        ValidateElementName(value.get<std::string>());
        alternatives.push_back(ByteString(value.get<std::string>()));
      } else {
        alternatives.push_back(GenerateLiteral(value));
      }
    }
    return Choice(alternatives);
  }
  return JSONSchemaConverter::GenerateEnum(spec, rule_name);
}

std::string XMLToolCallingConverter::EscapeAttrValue(const std::string& value) const {
  if (json_format_ != JSONFormat::kKimiK3XML) {
    return value;
  }
  // Kimi-K3's renderer escapes attribute values with & -> &amp; and " -> &quot;.
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    if (c == '&') {
      escaped += "&amp;";
    } else if (c == '"') {
      escaped += "&quot;";
    } else {
      escaped += c;
    }
  }
  return escaped;
}

int32_t XMLToolCallingConverter::FormatPropertyKey(
    const std::string& key, const SchemaSpecPtr& schema
) {
  if (IsXMLLayer()) {
    ValidateElementName(key);
    // Only kimi_k3_xml encodes the value's type next to the key; the other formats would
    // discard the result, so don't walk the schema for them.
    std::optional<std::string> pinned_type;
    if (json_format_ == JSONFormat::kKimiK3XML) {
      pinned_type = KimiK3TypeAttr(schema);
    }
    return Sequence(
        {ByteString(dialect_.property.open_prefix + EscapeAttrValue(key)), XMLKeySuffix(pinned_type)
        }
    );
  }
  return JSONSchemaConverter::FormatPropertyKey(key, schema);
}

int32_t XMLToolCallingConverter::FormatElementValueAndClose(
    const ElementSyntax& syntax, int32_t value_rule_id, int32_t close_expr
) {
  std::vector<int32_t> elements;
  if (!syntax.value_prefix.empty()) {
    elements.push_back(WhitespaceExpression());
    elements.push_back(ByteString(syntax.value_prefix));
  }
  // When wrapper padding is enabled, xml_string already accepts whitespace. Adding whitespace
  // repetitions around it preserves the language but creates an Earley state for every possible
  // split with the string body.
  if (!dialect_.pad_values_with_whitespace || value_rule_id == builder_.GetRuleId(kXMLString)) {
    elements.push_back(RuleRef(value_rule_id));
  } else {
    elements.push_back(WhitespaceExpression());
    elements.push_back(RuleRef(value_rule_id));
    elements.push_back(WhitespaceExpression());
  }
  elements.push_back(close_expr);
  return Sequence(elements);
}

int32_t XMLToolCallingConverter::FormatElement(
    const ElementSyntax& syntax,
    const std::string& key,
    int32_t value_rule_id,
    const SchemaSpecPtr& schema
) {
  ValidateElementName(key);
  const auto pinned_type =
      json_format_ == JSONFormat::kKimiK3XML ? KimiK3TypeAttr(schema) : std::nullopt;
  int32_t open =
      Sequence({ByteString(syntax.open_prefix + EscapeAttrValue(key)), XMLKeySuffix(pinned_type)});
  int32_t close = ByteString(
      syntax.close_prefix + std::string(syntax.close_repeats_key ? key : "") + syntax.close_suffix
  );
  return Sequence({open, FormatElementValueAndClose(syntax, value_rule_id, close)});
}

int32_t XMLToolCallingConverter::FormatProperty(
    const std::string& key,
    int32_t value_rule_id,
    const std::string& rule_name,
    int64_t idx,
    const SchemaSpecPtr& schema
) {
  if (IsXMLLayer()) {
    return FormatElement(dialect_.property, key, value_rule_id, schema);
  }
  return JSONSchemaConverter::FormatProperty(key, value_rule_id, rule_name, idx, schema);
}

int32_t XMLToolCallingConverter::FormatOtherProperty(
    int32_t key_pattern_expr,
    int32_t value_rule_id,
    const std::string& rule_name,
    const std::string& rule_name_suffix,
    const SchemaSpecPtr& schema
) {
  if (IsXMLLayer()) {
    const auto& syntax = dialect_.property;
    if (syntax.close_repeats_key) {
      int32_t name_rule_id =
          builder_.AddRuleWithHint(rule_name + "_" + rule_name_suffix + "_name", key_pattern_expr);
      int32_t content_rule_id = value_rule_id;
      if (!syntax.value_prefix.empty() || (dialect_.pad_values_with_whitespace &&
                                           value_rule_id != builder_.GetRuleId(kXMLString))) {
        std::vector<int32_t> content;
        if (!syntax.value_prefix.empty()) {
          content.push_back(WhitespaceExpression());
          content.push_back(ByteString(syntax.value_prefix));
        }
        if (dialect_.pad_values_with_whitespace &&
            value_rule_id != builder_.GetRuleId(kXMLString)) {
          content.push_back(WhitespaceExpression());
          content.push_back(RuleRef(value_rule_id));
          content.push_back(WhitespaceExpression());
        } else {
          content.push_back(RuleRef(value_rule_id));
        }
        content_rule_id = builder_.AddRuleWithHint(
            rule_name + "_" + rule_name_suffix + "_value", Sequence(content)
        );
      }
      Grammar::Impl::DynamicTag dynamic_tag{
          syntax.open_prefix,
          name_rule_id,
          syntax.open_suffix,
          content_rule_id,
          syntax.close_prefix,
          syntax.close_suffix
      };
      if (dialect_.recursive) {
        XGRAMMAR_DCHECK(!unique_key_scope_stack_.empty());
        auto& scope = unique_key_scope_stack_.back();
        if (scope.rule_id < 0) {
          scope.rule_id = builder_.AddEmptyRuleWithHint(rule_name + "_unique_keys");
        }
        dynamic_tag.unique_key_scope_rule_id = scope.rule_id;
        dynamic_tag.reserved_names = scope.reserved_names;
      }
      return builder_.AddDynamicTag(dynamic_tag);
    }
    const auto pinned_type =
        json_format_ == JSONFormat::kKimiK3XML ? KimiK3TypeAttr(schema) : std::nullopt;
    int32_t open =
        Sequence({ByteString(syntax.open_prefix), key_pattern_expr, XMLKeySuffix(pinned_type)});
    int32_t close = ByteString(syntax.close_prefix + syntax.close_suffix);
    return Sequence({open, FormatElementValueAndClose(syntax, value_rule_id, close)});
  }
  return JSONSchemaConverter::FormatOtherProperty(
      key_pattern_expr, value_rule_id, rule_name, rule_name_suffix, schema
  );
}

int32_t XMLToolCallingConverter::FormatPatternProperty(
    const std::string& key_regex,
    int32_t value_rule_id,
    const std::string& rule_name,
    const std::string& rule_name_suffix,
    const SchemaSpecPtr& schema
) {
  if (IsXMLLayer()) {
    return FormatOtherProperty(
        RegexExpression(key_regex), value_rule_id, rule_name, rule_name_suffix, schema
    );
  }
  return JSONSchemaConverter::FormatPatternProperty(
      key_regex, value_rule_id, rule_name, rule_name_suffix, schema
  );
}

int32_t XMLToolCallingConverter::CreatePropertyNameRule(
    const SchemaSpecPtr& spec, const std::string& rule_name_hint
) {
  if (!dialect_.recursive) {
    return JSONSchemaConverter::CreatePropertyNameRule(spec, rule_name_hint);
  }
  int32_t rule_id = builder_.AddEmptyRuleWithHint(rule_name_hint);
  std::string rule_name = builder_.GetRule(rule_id).name;
  bool old_generating_property_name = generating_property_name_;
  generating_property_name_ = true;
  builder_.UpdateRuleBody(rule_id, GenerateFromSpec(spec, rule_name));
  generating_property_name_ = old_generating_property_name;
  return rule_id;
}

int32_t XMLToolCallingConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces
) {
  if (dialect_.recursive) {
    ValidateRecursiveObject(spec);
    UniqueKeyScopeContext scope;
    scope.reserved_names.reserve(spec.properties.size());
    for (const auto& property : spec.properties) {
      scope.reserved_names.push_back(property.name);
    }
    unique_key_scope_stack_.push_back(std::move(scope));
  }
  nested_object_level_++;
  bool need_braces = !dialect_.recursive && nested_object_level_ > 1;
  bool saved_any_whitespace = any_whitespace_;
  if (dialect_.recursive) {
    any_whitespace_ = false;
  }
  int32_t result = JSONSchemaConverter::GenerateObject(spec, rule_name, need_braces);
  any_whitespace_ = saved_any_whitespace;
  nested_object_level_--;
  if (dialect_.recursive) {
    auto scope = std::move(unique_key_scope_stack_.back());
    unique_key_scope_stack_.pop_back();
    if (scope.rule_id >= 0) {
      builder_.UpdateRuleBody(scope.rule_id, result);
      result = RuleRef(scope.rule_id);
    }
  }
  return result;
}

void XMLToolCallingConverter::ValidateRecursiveObject(const ObjectSpec& spec) const {
  for (const auto& property : spec.properties) {
    ValidateElementName(property.name);
  }
}

void XMLToolCallingConverter::ValidateElementName(const std::string& name) const {
  if (!dialect_.recursive) {
    return;
  }
  XGRAMMAR_CHECK(!name.empty() && name.front() != '/' && name.find('>') == std::string::npos)
      << "Invalid recursive XML element name: " << name;
  XGRAMMAR_CHECK(IsCanonicalUTF8(name)) << "Recursive XML element names must be valid UTF-8";
  XGRAMMAR_CHECK(std::any_of(name.begin(), name.end(), [](unsigned char byte) {
    return !IsASCIIWhitespace(byte);
  })) << "Recursive XML element names cannot be blank";
}

void XMLToolCallingConverter::AddCache(const std::string& key, int32_t rule_id) {
  if (key.empty()) {
    return;
  }
  rule_cache_manager_.AddCache(key, IsInnerCacheLayer(), rule_id);
}

std::optional<int32_t> XMLToolCallingConverter::GetCache(const std::string& key) const {
  if (key.empty()) {
    return std::nullopt;
  }
  // At level 0, {"type":"object"} is the root tool-arguments object and uses XML parameter
  // tags. At level 1 it is the value of one such parameter and must use the inner JSON object
  // rule, including braces. Without this distinction, the outer XML object cache is reused for
  // the value before GenerateObject() can advance nested_object_level_.
  if (!dialect_.recursive && nested_object_level_ == 1 && key == kObjectCacheKey) {
    return rule_cache_manager_.GetCache(key, true);
  }
  return rule_cache_manager_.GetCache(key, IsInnerCacheLayer());
}

int XMLToolCallingConverter::GetRefCacheDomain() const {
  if (dialect_.recursive) {
    return generating_property_name_ ? 1 : 0;
  }
  return std::min(nested_object_level_, 2);
}

CohereXMLToolCallingConverter::CohereXMLToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    bool any_order
)
    : XMLToolCallingConverter(
          indent,
          separators,
          any_whitespace,
          max_whitespace_cnt,
          ref_resolver,
          JSONFormat::kCohereXML,
          any_order
      ) {}

bool CohereXMLToolCallingConverter::InCohereValueContext() const {
  return nested_object_level_ <= 1 || !object_stack_.empty() || cohere_array_level_ > 0;
}

int32_t CohereXMLToolCallingConverter::FormatCohereValue(int32_t value_rule_id) {
  if (value_rule_id == builder_.GetRuleId(kXMLString)) {
    return RuleRef(value_rule_id);
  }
  return Sequence({WhitespaceExpression(), RuleRef(value_rule_id), WhitespaceExpression()});
}

std::string CohereXMLToolCallingConverter::CohereTypeForJSONLiteral(const std::string& json_value) {
  picojson::value value;
  std::string error = picojson::parse(value, json_value);
  // Const/enum object and array literals are emitted as JSON text today, not recursive Cohere
  // dict/list bodies, so only JSON strings get the raw Cohere type.
  return error.empty() && value.is<std::string>() ? "raw" : "json";
}

std::optional<std::string> CohereXMLToolCallingConverter::CommonCohereTypeForJSONLiterals(
    const std::vector<std::string>& json_values
) {
  std::optional<std::string> common_type;
  for (const auto& json_value : json_values) {
    auto type = CohereTypeForJSONLiteral(json_value);
    if (!common_type.has_value()) {
      common_type = type;
    } else if (*common_type != type) {
      return std::nullopt;
    }
  }
  return common_type;
}

int32_t CohereXMLToolCallingConverter::GetCohereTypePattern(const SchemaSpecPtr& schema) {
  return std::visit(
      [this](const auto& spec) -> int32_t {
        using T = std::decay_t<decltype(spec)>;
        if constexpr (std::is_same_v<T, StringSpec>) {
          return ByteString("raw");
        } else if constexpr (std::is_same_v<T, ObjectSpec>) {
          return ByteString("dict");
        } else if constexpr (std::is_same_v<T, ArraySpec>) {
          return ByteString("list");
        } else if constexpr (std::is_same_v<T, ConstSpec>) {
          return ByteString(CohereTypeForJSONLiteral(spec.json_value));
        } else if constexpr (std::is_same_v<T, EnumSpec>) {
          auto common_type = CommonCohereTypeForJSONLiterals(spec.json_values);
          // Mixed enums are branch-correlated by FormatCohereParam. This fallback is only used if
          // a mixed enum somehow reaches the single-wrapper path, where there is no one true type.
          return ByteString(common_type.has_value() ? *common_type : "json");
        } else {
          return ByteString("json");
        }
      },
      schema->spec
  );
}

std::optional<std::vector<SchemaSpecPtr>> CohereXMLToolCallingConverter::GetCohereCompositeOptions(
    const SchemaSpecPtr& schema
) const {
  if (schema == nullptr) {
    return std::nullopt;
  }
  return std::visit(
      [](const auto& spec) -> std::optional<std::vector<SchemaSpecPtr>> {
        using T = std::decay_t<decltype(spec)>;
        if constexpr (std::is_same_v<T, AnyOfSpec>) {
          return spec.options;
        } else if constexpr (std::is_same_v<T, OneOfSpec>) {
          return spec.options;
        } else if constexpr (std::is_same_v<T, AllOfSpec>) {
          if (spec.schemas.size() == 1) {
            return spec.schemas;
          }
          return std::nullopt;
        } else if constexpr (std::is_same_v<T, TypeArraySpec>) {
          return spec.type_schemas;
        } else if constexpr (std::is_same_v<T, EnumSpec>) {
          if (spec.json_values.empty() ||
              CommonCohereTypeForJSONLiterals(spec.json_values).has_value()) {
            return std::nullopt;
          }
          std::vector<SchemaSpecPtr> options;
          options.reserve(spec.json_values.size());
          for (size_t index = 0; index < spec.json_values.size(); ++index) {
            const auto& json_value = spec.json_values[index];
            ConstSpec const_spec;
            const_spec.json_value = json_value;
            options.push_back(
                SchemaSpec::Make(std::move(const_spec), "", "enum_case_" + std::to_string(index))
            );
          }
          return options;
        } else {
          return std::nullopt;
        }
      },
      schema->spec
  );
}

int32_t CohereXMLToolCallingConverter::FormatSingleCohereParam(
    const std::optional<std::string>& name,
    const std::optional<int32_t>& key_pattern_expr,
    const SchemaSpecPtr& schema,
    int32_t value_rule_id
) {
  const auto& syntax = dialect_.property;
  std::vector<int32_t> elements = {ByteString(syntax.open_prefix)};
  if (name.has_value()) {
    elements.push_back(ByteString(" name=\"" + *name + "\""));
  } else if (key_pattern_expr.has_value()) {
    elements.push_back(ByteString(" name=\""));
    elements.push_back(*key_pattern_expr);
    elements.push_back(ByteString("\""));
  }
  elements.push_back(ByteString(" type=\""));
  elements.push_back(GetCohereTypePattern(schema));
  elements.push_back(ByteString("\"" + syntax.open_suffix));

  if (!syntax.value_prefix.empty()) {
    elements.push_back(ByteString(syntax.value_prefix));
  }
  elements.push_back(FormatCohereValue(value_rule_id));
  elements.push_back(ByteString(syntax.close_prefix + syntax.close_suffix));
  return Sequence(elements);
}

int32_t CohereXMLToolCallingConverter::FormatCohereParam(
    const std::optional<std::string>& name,
    const std::optional<int32_t>& key_pattern_expr,
    const SchemaSpecPtr& schema,
    int32_t value_rule_id
) {
  // Copy the name before generating: GenerateFromSpec may add rules and reallocate the
  // builder's rule storage, invalidating references into it.
  std::string value_rule_name = builder_.GetRule(value_rule_id).name;
  SchemaSpecPtr resolved_schema = schema;
  // Resolve RefSpecs only for Cohere wrapper classification; the value rule is already resolved.
  if (const auto* ref = std::get_if<RefSpec>(&resolved_schema->spec); ref != nullptr) {
    std::unordered_set<std::string> visited_ref_uris;
    do {
      if (!visited_ref_uris.insert(ref->uri).second) {
        break;
      }
      resolved_schema = ResolveRefSchema(*ref, value_rule_name);
      ref = std::get_if<RefSpec>(&resolved_schema->spec);
    } while (ref != nullptr);
  }

  auto options = GetCohereCompositeOptions(resolved_schema);
  if (!options.has_value()) {
    return FormatSingleCohereParam(name, key_pattern_expr, resolved_schema, value_rule_id);
  }

  std::vector<int32_t> choices;
  choices.reserve(options->size());
  for (size_t index = 0; index < options->size(); ++index) {
    const SchemaSpecPtr& option = (*options)[index];
    int32_t option_rule_id =
        CreateRule(option, value_rule_name + "_cohere_case_" + std::to_string(index));
    choices.push_back(FormatCohereParam(name, key_pattern_expr, option, option_rule_id));
  }
  return choices.size() == 1 ? choices[0] : Choice(choices);
}

int32_t CohereXMLToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  if (!InCohereValueContext()) {
    return JSONSchemaConverter::GenerateString(spec, rule_name);
  }
  if (!spec.pattern.has_value() && !spec.format.has_value() && spec.min_length == 0 &&
      spec.max_length == -1) {
    return RuleRef(kXMLString);
  }
  if (spec.format.has_value()) {
    const std::string& format = *spec.format;
    auto regex_pattern = JSONFormatToRegexPattern(format);
    if (regex_pattern.has_value()) {
      return RegexExpression(regex_pattern.value(), false, true);
    }
  }
  if (spec.pattern.has_value()) {
    return RegexExpression(*spec.pattern, false, /*force_cfg_expansion=*/true);
  }
  if (spec.min_length != 0 || spec.max_length != -1) {
    return Repeat(
        rule_name + "_characters",
        builder_.AddCharacterClass({{0, 0x10ffff}}),
        spec.min_length,
        spec.max_length
    );
  }
  return JSONSchemaConverter::GenerateString(spec, rule_name);
}

int32_t CohereXMLToolCallingConverter::GenerateAny(
    const AnySpec& spec, const std::string& rule_name
) {
  if (!InCohereValueContext()) {
    return JSONSchemaConverter::GenerateAny(spec, rule_name);
  }
  if (nested_object_level_ == 0) {
    return RuleRef(kXMLObject);
  }
  return JSONSchemaConverter::GenerateAny(spec, rule_name);
}

int32_t CohereXMLToolCallingConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  if (!InCohereValueContext()) {
    return JSONSchemaConverter::GenerateConst(spec, rule_name);
  }
  return ByteString(XMLValue(spec.json_value));
}

int32_t CohereXMLToolCallingConverter::GenerateEnum(
    const EnumSpec& spec, const std::string& rule_name
) {
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
  if (!InCohereValueContext()) {
    return JSONSchemaConverter::GenerateEnum(spec, rule_name);
  }
  std::vector<int32_t> values;
  values.reserve(spec.json_values.size());
  for (const auto& value : spec.json_values) {
    values.push_back(ByteString(XMLValue(value)));
  }
  return Choice(values);
}

int32_t CohereXMLToolCallingConverter::GenerateObject(
    const ObjectSpec& spec, const std::string& rule_name, bool dummy_need_braces
) {
  nested_object_level_++;
  bool use_cohere_object = InCohereValueContext();

  int32_t result;
  if (use_cohere_object) {
    SchemaSpecPtr additional_property;
    if (spec.allow_additional_properties && spec.additional_properties_schema) {
      additional_property = spec.additional_properties_schema;
    } else if (spec.allow_unevaluated_properties && spec.unevaluated_properties_schema) {
      additional_property = spec.unevaluated_properties_schema;
    } else if (spec.allow_additional_properties || spec.allow_unevaluated_properties) {
      additional_property = SchemaSpec::Make(AnySpec{}, "", "any");
    }

    object_stack_.push_back(&spec);
    additional_property_stack_.push_back(additional_property);
    result = JSONSchemaConverter::GenerateObject(spec, rule_name, false);
    additional_property_stack_.pop_back();
    object_stack_.pop_back();
  } else {
    result = JSONSchemaConverter::GenerateObject(spec, rule_name, nested_object_level_ > 1);
  }

  nested_object_level_--;
  return result;
}

int32_t CohereXMLToolCallingConverter::GenerateArray(
    const ArraySpec& spec, const std::string& rule_name
) {
  if (!InCohereValueContext()) {
    nested_object_level_++;
    auto result = JSONSchemaConverter::GenerateArray(spec, rule_name);
    nested_object_level_--;
    return result;
  }

  cohere_array_level_++;
  std::vector<int32_t> item_patterns;
  for (size_t i = 0; i < spec.prefix_items.size(); ++i) {
    int32_t item_rule_id =
        CreateRule(spec.prefix_items[i], rule_name + "_item_" + std::to_string(i));
    item_patterns.push_back(
        FormatCohereParam(std::nullopt, std::nullopt, spec.prefix_items[i], item_rule_id)
    );
  }

  std::optional<int32_t> additional_item_pattern;
  if (spec.allow_additional_items && spec.additional_items) {
    int32_t additional_rule_id = CreateRule(spec.additional_items, rule_name + "_additional");
    additional_item_pattern =
        FormatCohereParam(std::nullopt, std::nullopt, spec.additional_items, additional_rule_id);
  }
  cohere_array_level_--;

  if (item_patterns.empty()) {
    if (!additional_item_pattern.has_value() || spec.max_items == 0) {
      return Empty();
    }
    return Repeat(
        rule_name + "_items",
        *additional_item_pattern,
        static_cast<int>(spec.min_items),
        spec.max_items == -1 ? -1 : static_cast<int>(spec.max_items)
    );
  }

  int32_t prefix_part = Sequence(item_patterns);
  if (!additional_item_pattern.has_value()) {
    return prefix_part;
  }

  int64_t min_additional = std::max(
      static_cast<int64_t>(0), spec.min_items - static_cast<int64_t>(item_patterns.size())
  );
  int64_t max_additional =
      spec.max_items == -1 ? -1 : spec.max_items - static_cast<int64_t>(item_patterns.size());
  return Sequence(
      {prefix_part,
       Repeat(
           rule_name + "_additional_items",
           *additional_item_pattern,
           static_cast<int>(min_additional),
           max_additional == -1 ? -1 : static_cast<int>(max_additional)
       )}
  );
}

int32_t CohereXMLToolCallingConverter::FormatProperty(
    const std::string& key,
    int32_t value_rule_id,
    const std::string& rule_name,
    int64_t idx,
    const SchemaSpecPtr& schema
) {
  if (!object_stack_.empty() && idx >= 0 &&
      idx < static_cast<int64_t>(object_stack_.back()->properties.size())) {
    const auto& prop = object_stack_.back()->properties[idx];
    return FormatCohereParam(prop.name, std::nullopt, prop.schema, value_rule_id);
  }
  return XMLToolCallingConverter::FormatProperty(key, value_rule_id, rule_name, idx, schema);
}

int32_t CohereXMLToolCallingConverter::FormatOtherProperty(
    int32_t key_pattern_expr,
    int32_t value_rule_id,
    const std::string& rule_name,
    const std::string& rule_name_suffix,
    const SchemaSpecPtr& schema
) {
  SchemaSpecPtr value_schema = schema;
  if (!value_schema && !additional_property_stack_.empty()) {
    value_schema = additional_property_stack_.back();
  }
  if (!value_schema && InCohereValueContext()) {
    value_schema = SchemaSpec::Make(AnySpec{}, "", "any");
    value_rule_id = CreateRule(value_schema, rule_name + "_" + rule_name_suffix + "_cohere_any");
  }
  if (value_schema) {
    return FormatCohereParam(std::nullopt, key_pattern_expr, value_schema, value_rule_id);
  }
  return XMLToolCallingConverter::FormatOtherProperty(
      key_pattern_expr, value_rule_id, rule_name, rule_name_suffix, schema
  );
}

int32_t CohereXMLToolCallingConverter::FormatPatternProperty(
    const std::string& key_regex,
    int32_t value_rule_id,
    const std::string& rule_name,
    const std::string& rule_name_suffix,
    const SchemaSpecPtr& schema
) {
  if (InCohereValueContext()) {
    return FormatOtherProperty(
        RegexExpression(key_regex, /*json_string=*/false, /*force_cfg_expansion=*/true),
        value_rule_id,
        rule_name,
        rule_name_suffix,
        schema
    );
  }
  return XMLToolCallingConverter::FormatPatternProperty(
      key_regex, value_rule_id, rule_name, rule_name_suffix, schema
  );
}

std::string CohereXMLToolCallingConverter::GetKeyPattern() const {
  if (InCohereValueContext()) {
    return kXMLVariableName;
  }
  return JSONSchemaConverter::GetKeyPattern();
}

int32_t CohereXMLToolCallingConverter::BuildXMLIdentifierExcludingBody(
    const XMLIdentifierTrieNode& node, const std::string& rule_name, int depth
) {
  std::vector<int32_t> choices;
  if (depth > 0 && !node.is_terminal) {
    choices.push_back(Empty());
  }

  auto divergent_chars = XMLIdentifierCharClassExcluding(node.children, depth == 0);
  if (!divergent_chars.empty()) {
    choices.push_back(Sequence(
        {builder_.AddCharacterClass(divergent_chars),
         builder_.AddCharacterClassStar(XMLIdentifierContinuationChars())}
    ));
  }

  for (const auto& [c, child] : node.children) {
    if (!IsXMLIdentifierChar(c, depth == 0)) {
      continue;
    }
    choices.push_back(Sequence(
        {ByteString(std::string(1, c)), BuildXMLIdentifierExcludingBody(child, rule_name, depth + 1)
        }
    ));
  }

  if (choices.empty()) {
    return Empty();
  }
  return Choice(choices);
}

int32_t CohereXMLToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (InCohereValueContext()) {
    if (properties.empty()) {
      return RuleRef(GetKeyPattern());
    }
    XMLIdentifierTrieNode root;
    for (const auto& prop : properties) {
      XMLIdentifierTrieNode* cur = &root;
      bool is_valid_identifier = true;
      for (size_t i = 0; i < prop.name.size(); ++i) {
        char c = prop.name[i];
        if (!IsXMLIdentifierChar(c, i == 0)) {
          is_valid_identifier = false;
          break;
        }
        cur = &cur->children[c];
      }
      if (is_valid_identifier && !prop.name.empty()) {
        cur->is_terminal = true;
      }
    }
    int32_t key_rule_id = builder_.AddEmptyRuleWithHint(rule_name + "_cohere_addl_key");
    builder_.UpdateRuleBody(
        key_rule_id, BuildXMLIdentifierExcludingBody(root, builder_.GetRule(key_rule_id).name, 0)
    );
    return RuleRef(key_rule_id);
  }
  return JSONSchemaConverter::GetKeyPatternExcluding(properties, rule_name);
}

std::string CohereXMLToolCallingConverter::NextSeparator(bool is_end) {
  if (InCohereValueContext()) {
    return GetWhitespacePattern();
  }
  return JSONSchemaConverter::NextSeparator(is_end);
}

void CohereXMLToolCallingConverter::AddCache(const std::string& key, int32_t rule_id) {
  if (key.empty()) {
    return;
  }
  rule_cache_manager_.AddCache(key, nested_object_level_ > 1 && !InCohereValueContext(), rule_id);
}

std::optional<int32_t> CohereXMLToolCallingConverter::GetCache(const std::string& key) const {
  if (key.empty()) {
    return std::nullopt;
  }
  return rule_cache_manager_.GetCache(key, nested_object_level_ > 1 && !InCohereValueContext());
}

}  // namespace xgrammar
