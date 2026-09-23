/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/gemma.cc
 * \brief Implementation of the Gemma tool-calling converter.
 */
#include <picojson.h>

#include <string>
#include <utility>
#include <vector>

#include "../grammar_functor.h"
#include "../json_schema_converter_ext.h"
#include "../support/json_parse.h"
#include "../support/logging.h"

namespace xgrammar {

// ==================== GemmaToolCallingConverter ====================

const std::string GemmaToolCallingConverter::kGemmaStringDelim = "<|\"|>";
const std::string GemmaToolCallingConverter::kGemmaStringContent = "gemma_string_content";
const std::string GemmaToolCallingConverter::kGemmaVariableName = "gemma_variable_name";

namespace {

// Unquoted Gemma property keys. Declared properties are emitted verbatim; this pattern only
// bounds the keys of additionalProperties, matching the identifier form of the other
// tool-calling styles.
constexpr const char kGemmaIdentifierRegex[] = "[a-zA-Z_][a-zA-Z0-9_]*";

// A string body character for length-constrained strings. `.` excludes newlines in the FSM
// regex dialect, so newlines are added back explicitly.
constexpr const char kGemmaAnyCharRegex[] = "(.|\\n)";

std::vector<std::string> WithStringDelimiter(std::vector<std::string> excludes) {
  // The delimiter closes every string, so no string body may contain it. Registering it as an
  // exclusion lets the shared exclusion machinery (ExcludingString, IsAllowedString) keep it
  // out of pattern strings, bounded strings, literals and dynamic keys alike.
  excludes.push_back(GemmaToolCallingConverter::kGemmaStringDelim);
  return excludes;
}

}  // namespace

GemmaToolCallingConverter::GemmaToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    bool any_order,
    std::vector<std::string> excludes
)
    : JSONSchemaConverter(
          indent,
          std::move(separators),
          any_whitespace,
          max_whitespace_cnt,
          std::move(ref_resolver),
          any_order,
          WithStringDelimiter(std::move(excludes))
      ),
      has_user_excludes_(excludes_.size() > 1) {}

void GemmaToolCallingConverter::AddBasicRules() {
  // The base builds the basic rules through the virtual Generate* hooks, so basic_any,
  // basic_array and basic_object already come out in Gemma form: bare keys through
  // GetKeyPattern, Gemma strings through GenerateString. basic_string itself is spelled out
  // as a JSON string by the base and is rebound below.
  JSONSchemaConverter::AddBasicRules({kGemmaStringContent, kGemmaVariableName});

  // Any text that does not contain the string delimiter (or a caller-provided exclusion). The
  // enclosing GenerateString terminates it with the delimiter, so the boundary is unambiguous.
  builder_.UpdateRuleBody(kGemmaStringContent, TagDispatch(false, excludes_));

  builder_.UpdateRuleBody(
      kGemmaVariableName,
      has_user_excludes_
          ? ExcludingString(kGemmaIdentifierRegex, kGemmaVariableName, false)
          : Sequence(
                {builder_.AddCharacterClass({{'a', 'z'}, {'A', 'Z'}, {'_', '_'}}),
                 builder_.AddCharacterClassStar({{'a', 'z'}, {'A', 'Z'}, {'0', '9'}, {'_', '_'}})}
            )
  );

  builder_.UpdateRuleBody(kBasicString, GenerateString(StringSpec{}, kBasicString));
}

int32_t GemmaToolCallingConverter::GemmaString(int32_t body) {
  return Sequence({ByteString(kGemmaStringDelim), body, ByteString(kGemmaStringDelim)});
}

int32_t GemmaToolCallingConverter::GemmaRegexBody(
    const std::string& regex, const std::string& rule_name, bool force_cfg_expansion
) {
  // Intersect the regex with "does not contain the delimiter" whenever the FSM engine can
  // build it, so a permissive pattern such as ".*" cannot close the string early. The built-in
  // format regexes use constructs the FSM engine does not support yet; they keep the CFG
  // expansion and are emitted verbatim, as the JSON converter does.
  if (GrammarFSMBuilder::Regex(regex, false).IsOk()) {
    return ExcludingString(regex, rule_name, false);
  }
  return RegexExpression(regex, false, force_cfg_expansion);
}

int32_t GemmaToolCallingConverter::GenerateString(
    const StringSpec& spec, const std::string& rule_name
) {
  if (spec.format.has_value()) {
    auto regex = JSONFormatToRegexPattern(*spec.format);
    if (regex.has_value()) {
      return GemmaString(GemmaRegexBody(*regex, rule_name + "_format", true));
    }
  }
  if (spec.pattern.has_value()) {
    return GemmaString(GemmaRegexBody(*spec.pattern, rule_name + "_pattern", false));
  }
  if (spec.min_length != 0 || spec.max_length != -1) {
    // Gemma strings have no escapes, so every codepoint counts once towards the bounds.
    std::string repetition = "{" + std::to_string(spec.min_length) + "," +
                             (spec.max_length == -1 ? "" : std::to_string(spec.max_length)) + "}";
    return GemmaString(GemmaRegexBody(kGemmaAnyCharRegex + repetition, rule_name + "_bounded", true)
    );
  }
  return GemmaString(RuleRef(kGemmaStringContent));
}

bool GemmaToolCallingConverter::IsAllowedGemmaValue(const picojson::value& value) const {
  // Gemma strings and keys are emitted raw, so exclusions are checked against the raw text
  // rather than the JSON-escaped spelling that JSONSchemaConverter::IsAllowedLiteral uses.
  if (value.is<std::string>()) {
    return IsAllowedString(value.get<std::string>());
  }
  if (value.is<picojson::array>()) {
    for (const auto& item : value.get<picojson::array>()) {
      if (!IsAllowedGemmaValue(item)) return false;
    }
  }
  if (value.is<picojson::object>()) {
    for (const auto& [key, item] : value.get<picojson::object>()) {
      if (!IsAllowedString(key) || !IsAllowedGemmaValue(item)) return false;
    }
  }
  return true;
}

std::string GemmaToolCallingConverter::SerializeGemma(const picojson::value& value) {
  if (value.is<std::string>()) {
    // No escape sequences: the raw content sits between the delimiters.
    return kGemmaStringDelim + value.get<std::string>() + kGemmaStringDelim;
  }
  if (value.is<picojson::object>()) {
    const auto& object = value.get<picojson::object>();
    std::string result = "{";
    bool first = true;
    for (const auto& key : object.ordered_keys()) {
      if (!first) result += ",";
      first = false;
      result += key + ":" + SerializeGemma(object.at(key));
    }
    return result + "}";
  }
  if (value.is<picojson::array>()) {
    std::string result = "[";
    bool first = true;
    for (const auto& item : value.get<picojson::array>()) {
      if (!first) result += ",";
      first = false;
      result += SerializeGemma(item);
    }
    return result + "]";
  }
  // Numbers, booleans and null keep their JSON spelling.
  return value.serialize();
}

int32_t GemmaToolCallingConverter::GemmaLiteral(const std::string& json_value) {
  picojson::value value;
  std::string error = ParseJSON(value, json_value);
  XGRAMMAR_CHECK(error.empty()) << "Failed to parse JSON value: " << error
                                << ". The JSON string is: " << json_value;
  if (!IsAllowedGemmaValue(value)) {
    return Unsatisfiable();
  }
  if (!value.is<std::string>() && !value.is<picojson::object>() && !value.is<picojson::array>()) {
    // Scalars are spelled the same in Gemma and JSON. Keep the schema's own text: a round trip
    // through picojson's double storage would lose precision for large integers and change
    // the formatting (e.g. 1.0 -> 1).
    auto begin = json_value.find_first_not_of(" \n\t\r");
    auto end = json_value.find_last_not_of(" \n\t\r");
    return ByteString(json_value.substr(begin, end - begin + 1));
  }
  // Strings, objects and arrays are re-serialized into Gemma form. Numbers nested inside them
  // still pass through picojson.
  return ByteString(SerializeGemma(value));
}

int32_t GemmaToolCallingConverter::GenerateConst(
    const ConstSpec& spec, const std::string& rule_name
) {
  return GemmaLiteral(spec.json_value);
}

int32_t GemmaToolCallingConverter::GenerateEnum(
    const EnumSpec& spec, const std::string& rule_name
) {
  XGRAMMAR_DCHECK(!spec.json_values.empty())
      << "GenerateEnum called with empty enum spec for rule: " << rule_name;
  std::vector<int32_t> values;
  values.reserve(spec.json_values.size());
  for (const auto& json_value : spec.json_values) {
    int32_t literal = GemmaLiteral(json_value);
    if (literal != Unsatisfiable()) {
      values.push_back(literal);
    }
  }
  if (values.empty()) {
    return Unsatisfiable();
  }
  return Choice(values);
}

int32_t GemmaToolCallingConverter::FormatPropertyKey(
    const std::string& key, const SchemaSpecPtr& schema
) {
  // Keys are unquoted and unescaped.
  if (!IsAllowedString(key)) {
    return Unsatisfiable();
  }
  return ByteString(key);
}

std::string GemmaToolCallingConverter::GetKeyPattern() const { return kGemmaVariableName; }

int32_t GemmaToolCallingConverter::GetKeyPatternExcluding(
    const std::vector<ObjectSpec::Property>& properties, const std::string& rule_name
) {
  if (properties.empty()) {
    return KeyPatternExpression();
  }
  std::vector<std::string> keys;
  keys.reserve(properties.size());
  for (const auto& property : properties) {
    keys.push_back(property.name);
  }
  return ExcludingString(kGemmaIdentifierRegex, rule_name + "_addl_key", false, keys);
}

int32_t GemmaToolCallingConverter::CreatePatternKeyRule(
    const std::string& pattern, const std::string& rule_name_hint
) {
  // patternProperties keys are bare identifiers, not delimited strings. A pattern whose
  // language contains ':' is ambiguous against the key/value separator and cannot round-trip.
  return builder_.AddRuleWithHint(rule_name_hint, GemmaRegexBody(pattern, rule_name_hint, true));
}

int32_t GemmaToolCallingConverter::CreatePropertyNamesKeyRule(
    const SchemaSpecPtr& property_names, const std::string& rule_name_hint
) {
  // Constrain the bare key by the propertyNames pattern when one is given; otherwise keep the
  // identifier rule. Other propertyNames constraints do not apply to unquoted keys.
  if (auto* string_spec = std::get_if<StringSpec>(&property_names->spec)) {
    if (string_spec->pattern.has_value()) {
      return CreatePatternKeyRule(*string_spec->pattern, rule_name_hint);
    }
  }
  return builder_.AddRuleWithHint(rule_name_hint, KeyPatternExpression());
}

}  // namespace xgrammar
