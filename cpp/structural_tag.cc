/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag.cc
 */
#include "structural_tag.h"

#include <picojson.h>
#include <xgrammar/exception.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "grammar_functor.h"
#include "grammar_impl.h"
#include "json_schema_converter.h"
#include "support/logging.h"
#include "support/recursion_guard.h"
#include "support/utils.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

// Short alias for the error type.
using ISTError = InvalidStructuralTagError;

// Format: {{var_name([])? (.var_name([])?)*}}
const std::regex full_placeholder_regex =
    std::regex(R"(\{\{([a-zA-Z_][a-zA-Z0-9_]*)(\[\])?(\.([a-zA-Z_][a-zA-Z0-9_]*)(\[\])?)*\}\})");
const std::regex placeholder_regex = std::regex(R"(([a-zA-Z_][a-zA-Z0-9_]*)(\[\])?)");

picojson::value FormatToJSON(const Format& format) {
  return std::visit([&](auto&& arg) -> picojson::value { return arg.ToJSON(); }, format);
}

std::string FormatToJSONstr(const Format& format) { return FormatToJSON(format).serialize(); }

bool FullyMatchesPlaceholder(const std::string& str) {
  return std::regex_match(str, full_placeholder_regex);
}

struct PlaceHolderWithArray {
  std::string name;
  bool is_array;
};

using Layers = std::vector<PlaceHolderWithArray>;

/************** StructuralTag Parser **************/

class StructuralTagParser {
 public:
  static Result<StructuralTag, StructuralTagError> FromJSON(const std::string& json);

 private:
  Result<StructuralTag, ISTError> ParseStructuralTag(const picojson::value& value);

  /*!
   * \brief Parse a Format object from a JSON value.
   * \param value The JSON value to parse.
   * \return A Format object if the JSON is valid, otherwise an error message in std::runtime_error.
   * \note The "type" field is checked in this function, and not checked in the Parse*Format
   * functions.
   */
  Result<Format, ISTError> ParseFormat(const picojson::value& value);
  Result<ConstStringFormat, ISTError> ParseConstStringFormat(const picojson::object& value);
  Result<JSONSchemaFormat, ISTError> ParseJSONSchemaFormat(const picojson::object& value);
  Result<QwenXmlParameterFormat, ISTError> ParseQwenXmlParameterFormat(const picojson::object& value
  );
  Result<AnyTextFormat, ISTError> ParseAnyTextFormat(const picojson::object& value);
  Result<GrammarFormat, ISTError> ParseGrammarFormat(const picojson::object& value);
  Result<RegexFormat, ISTError> ParseRegexFormat(const picojson::object& value);
  Result<SequenceFormat, ISTError> ParseSequenceFormat(const picojson::object& value);
  Result<OrFormat, ISTError> ParseOrFormat(const picojson::object& value);
  /*! \brief ParseTagFormat with extra check for object and the type field. */
  Result<TagFormat, ISTError> ParseTagFormat(const picojson::value& value);
  Result<TagFormat, ISTError> ParseTagFormat(const picojson::object& value);
  Result<TriggeredTagsFormat, ISTError> ParseTriggeredTagsFormat(const picojson::object& value);
  Result<TagsWithSeparatorFormat, ISTError> ParseTagsWithSeparatorFormat(
      const picojson::object& value
  );

  int parse_format_recursion_depth_ = 0;
};

Result<StructuralTag, StructuralTagError> StructuralTagParser::FromJSON(const std::string& json) {
  picojson::value value;
  std::string err = picojson::parse(value, json);
  if (!err.empty()) {
    return ResultErr<InvalidJSONError>("Failed to parse JSON: " + err);
  }
  return Result<StructuralTag, StructuralTagError>::Convert(
      StructuralTagParser().ParseStructuralTag(value)
  );
}

Result<StructuralTag, ISTError> StructuralTagParser::ParseStructuralTag(const picojson::value& value
) {
  if (!value.is<picojson::object>()) {
    return ResultErr<ISTError>("Structural tag must be an object");
  }
  const auto& obj = value.get<picojson::object>();
  // The type field is optional but must be "structural_tag" if present.
  if (obj.find("type") != obj.end()) {
    if (!obj["type"].is<std::string>() || obj["type"].get<std::string>() != "structural_tag") {
      return ResultErr<ISTError>("Structural tag's type must be a string \"structural_tag\"");
    }
  }
  // The format field is required.
  if (obj.find("format") == obj.end()) {
    return ResultErr<ISTError>("Structural tag must have a format field");
  }
  auto format = ParseFormat(obj["format"]);
  if (format.IsErr()) {
    return ResultErr<ISTError>(std::move(format).UnwrapErr());
  }
  return ResultOk<StructuralTag>(std::move(format).Unwrap());
}

Result<Format, ISTError> StructuralTagParser::ParseFormat(const picojson::value& value) {
  RecursionGuard guard(&parse_format_recursion_depth_);
  if (!value.is<picojson::object>()) {
    return ResultErr<ISTError>("Format must be an object");
  }
  const auto& obj = value.get<picojson::object>();
  // If type is present, use it to determine the format.
  if (obj.find("type") != obj.end()) {
    if (!obj["type"].is<std::string>()) {
      return ResultErr<ISTError>("Format's type must be a string");
    }
    auto type = obj["type"].get<std::string>();
    if (type == "const_string") {
      return Result<Format, ISTError>::Convert(ParseConstStringFormat(obj));
    } else if (type == "json_schema") {
      return Result<Format, ISTError>::Convert(ParseJSONSchemaFormat(obj));
    } else if (type == "any_text") {
      return Result<Format, ISTError>::Convert(ParseAnyTextFormat(obj));
    } else if (type == "sequence") {
      return Result<Format, ISTError>::Convert(ParseSequenceFormat(obj));
    } else if (type == "or") {
      return Result<Format, ISTError>::Convert(ParseOrFormat(obj));
    } else if (type == "tag") {
      return Result<Format, ISTError>::Convert(ParseTagFormat(obj));
    } else if (type == "triggered_tags") {
      return Result<Format, ISTError>::Convert(ParseTriggeredTagsFormat(obj));
    } else if (type == "tags_with_separator") {
      return Result<Format, ISTError>::Convert(ParseTagsWithSeparatorFormat(obj));
    } else if (type == "qwen_xml_parameter") {
      return Result<Format, ISTError>::Convert(ParseQwenXmlParameterFormat(obj));
    } else if (type == "grammar") {
      return Result<Format, ISTError>::Convert(ParseGrammarFormat(obj));
    } else if (type == "regex") {
      return Result<Format, ISTError>::Convert(ParseRegexFormat(obj));
    } else {
      return ResultErr<ISTError>("Format type not recognized: " + type);
    }
  }

  // If type is not present, try every format type one by one. Tag is prioritized.
  auto tag_format = ParseTagFormat(obj);
  if (!tag_format.IsErr()) {
    return ResultOk<Format>(std::move(tag_format).Unwrap());
  }
  auto const_string_format = ParseConstStringFormat(obj);
  if (!const_string_format.IsErr()) {
    return ResultOk<Format>(std::move(const_string_format).Unwrap());
  }
  auto json_schema_format = ParseJSONSchemaFormat(obj);
  if (!json_schema_format.IsErr()) {
    return ResultOk<Format>(std::move(json_schema_format).Unwrap());
  }
  auto any_text_format = ParseAnyTextFormat(obj);
  if (!any_text_format.IsErr()) {
    return ResultOk<Format>(std::move(any_text_format).Unwrap());
  }
  auto sequence_format = ParseSequenceFormat(obj);
  if (!sequence_format.IsErr()) {
    return ResultOk<Format>(std::move(sequence_format).Unwrap());
  }
  auto or_format = ParseOrFormat(obj);
  if (!or_format.IsErr()) {
    return ResultOk<Format>(std::move(or_format).Unwrap());
  }
  auto triggered_tags_format = ParseTriggeredTagsFormat(obj);
  if (!triggered_tags_format.IsErr()) {
    return ResultOk<Format>(std::move(triggered_tags_format).Unwrap());
  }
  auto tags_with_separator_format = ParseTagsWithSeparatorFormat(obj);
  if (!tags_with_separator_format.IsErr()) {
    return ResultOk<Format>(std::move(tags_with_separator_format).Unwrap());
  }
  return ResultErr<ISTError>("Invalid format: " + value.serialize(false));
}

Result<ConstStringFormat, ISTError> StructuralTagParser::ParseConstStringFormat(
    const picojson::object& obj
) {
  // value is required.
  auto value_it = obj.find("value");
  if (value_it == obj.end() || !value_it->second.is<std::string>() ||
      value_it->second.get<std::string>().empty()) {
    return ResultErr<ISTError>("ConstString format must have a value field with a non-empty string"
    );
  }
  return ResultOk<ConstStringFormat>(value_it->second.get<std::string>());
}

Result<JSONSchemaFormat, ISTError> StructuralTagParser::ParseJSONSchemaFormat(
    const picojson::object& obj
) {
  // json_schema is required.
  auto json_schema_it = obj.find("json_schema");
  if (json_schema_it == obj.end() ||
      !(json_schema_it->second.is<picojson::object>() || json_schema_it->second.is<bool>())) {
    if (json_schema_it != obj.end() && json_schema_it->second.is<std::string>() &&
        FullyMatchesPlaceholder(json_schema_it->second.to_str())) {
      return ResultOk<JSONSchemaFormat>(json_schema_it->second.to_str());
    }
    return ResultErr<ISTError>(
        "JSON schema format must have a json_schema field with a object or boolean value"
    );
  }
  // here introduces a serialization/deserialization overhead; try to avoid it in the future.
  return ResultOk<JSONSchemaFormat>(json_schema_it->second.serialize(false));
}

Result<QwenXmlParameterFormat, ISTError> StructuralTagParser::ParseQwenXmlParameterFormat(
    const picojson::object& obj
) {
  // json_schema is required.
  auto json_schema_it = obj.find("json_schema");
  if (json_schema_it == obj.end() ||
      !(json_schema_it->second.is<picojson::object>() || json_schema_it->second.is<bool>())) {
    if (json_schema_it != obj.end() && json_schema_it->second.is<std::string>() &&
        FullyMatchesPlaceholder(json_schema_it->second.to_str())) {
      return ResultOk<QwenXmlParameterFormat>(json_schema_it->second.to_str());
    }
    return ResultErr<ISTError>(
        "Qwen XML Parameter format must have a json_schema field with a object or boolean value"
    );
  }
  // here introduces a serialization/deserialization overhead; try to avoid it in the future.
  return ResultOk<QwenXmlParameterFormat>(json_schema_it->second.serialize(false));
}

Result<AnyTextFormat, ISTError> StructuralTagParser::ParseAnyTextFormat(const picojson::object& obj
) {
  auto excluded_strs_it = obj.find("excludes");
  if (excluded_strs_it == obj.end()) {
    if ((obj.find("type") == obj.end())) {
      return ResultErr<ISTError>("Any text format should not have any fields other than type");
    }
    return ResultOk<AnyTextFormat>(std::vector<std::string>{});
  }
  if (!excluded_strs_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("AnyText format's excluded_strs field must be an array");
  }
  const auto& excluded_strs_array = excluded_strs_it->second.get<picojson::array>();
  std::vector<std::string> excluded_strs;
  excluded_strs.reserve(excluded_strs_array.size());
  for (const auto& excluded_str : excluded_strs_array) {
    if (!excluded_str.is<std::string>()) {
      return ResultErr<ISTError>("AnyText format's excluded_strs array must contain strings");
    }
    excluded_strs.push_back(excluded_str.get<std::string>());
  }
  return ResultOk<AnyTextFormat>(std::move(excluded_strs));
}

Result<GrammarFormat, ISTError> StructuralTagParser::ParseGrammarFormat(const picojson::object& obj
) {
  // grammar is required.
  auto grammar_it = obj.find("grammar");
  if (grammar_it == obj.end() || !grammar_it->second.is<std::string>() ||
      grammar_it->second.get<std::string>().empty()) {
    return ResultErr<ISTError>("Grammar format must have a grammar field with a non-empty string");
  }
  return ResultOk<GrammarFormat>(grammar_it->second.get<std::string>());
}

Result<RegexFormat, ISTError> StructuralTagParser::ParseRegexFormat(const picojson::object& obj) {
  // pattern is required.
  auto pattern_it = obj.find("pattern");
  if (pattern_it == obj.end() || !pattern_it->second.is<std::string>() ||
      pattern_it->second.get<std::string>().empty()) {
    return ResultErr<ISTError>("Regex format must have a pattern field with a non-empty string");
  }
  return ResultOk<RegexFormat>(pattern_it->second.get<std::string>());
}

Result<SequenceFormat, ISTError> StructuralTagParser::ParseSequenceFormat(
    const picojson::object& obj
) {
  // elements is required.
  auto elements_it = obj.find("elements");
  if (elements_it == obj.end() || !elements_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("Sequence format must have an elements field with an array");
  }
  const auto& elements_array = elements_it->second.get<picojson::array>();
  std::vector<Format> elements;
  elements.reserve(elements_array.size());
  for (const auto& element : elements_array) {
    auto format = ParseFormat(element);
    if (format.IsErr()) {
      return ResultErr<ISTError>(std::move(format).UnwrapErr());
    }
    elements.push_back(std::move(format).Unwrap());
  }
  if (elements.size() == 0) {
    return ResultErr<ISTError>("Sequence format must have at least one element");
  }
  return ResultOk<SequenceFormat>(std::move(elements));
}

Result<OrFormat, ISTError> StructuralTagParser::ParseOrFormat(const picojson::object& obj) {
  // elements is required.
  auto elements_it = obj.find("elements");
  if (elements_it == obj.end() || !elements_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("Or format must have an elements field with an array");
  }
  const auto& elements_array = elements_it->second.get<picojson::array>();
  std::vector<Format> elements;
  elements.reserve(elements_array.size());
  for (const auto& element : elements_array) {
    auto format = ParseFormat(element);
    if (format.IsErr()) {
      return ResultErr<ISTError>(std::move(format).UnwrapErr());
    }
    elements.push_back(std::move(format).Unwrap());
  }
  if (elements.size() == 0) {
    return ResultErr<ISTError>("Or format must have at least one element");
  }
  return ResultOk<OrFormat>(std::move(elements));
}

Result<TagFormat, ISTError> StructuralTagParser::ParseTagFormat(const picojson::value& value) {
  if (!value.is<picojson::object>()) {
    return ResultErr<ISTError>("Tag format must be an object");
  }
  const auto& obj = value.get<picojson::object>();
  if (obj.find("type") != obj.end() &&
      (!obj["type"].is<std::string>() || obj["type"].get<std::string>() != "tag")) {
    return ResultErr<ISTError>("Tag format's type must be a string \"tag\"");
  }
  return ParseTagFormat(obj);
}

Result<TagFormat, ISTError> StructuralTagParser::ParseTagFormat(const picojson::object& obj) {
  // begin is required.
  auto begin_it = obj.find("begin");
  if (begin_it == obj.end() || !begin_it->second.is<std::string>()) {
    return ResultErr<ISTError>("Tag format's begin field must be a string");
  }
  // content is required.
  auto content_it = obj.find("content");
  if (content_it == obj.end()) {
    return ResultErr<ISTError>("Tag format must have a content field");
  }
  auto content = ParseFormat(content_it->second);
  if (content.IsErr()) {
    return ResultErr<ISTError>(std::move(content).UnwrapErr());
  }
  // end is required - can be string or array of strings
  auto end_it = obj.find("end");
  if (end_it == obj.end()) {
    return ResultErr<ISTError>("Tag format must have an end field");
  }

  std::vector<std::string> end_strings;
  if (end_it->second.is<std::string>()) {
    // Single string case
    end_strings.push_back(end_it->second.get<std::string>());
  } else if (end_it->second.is<picojson::array>()) {
    // Array case
    const auto& end_array = end_it->second.get<picojson::array>();
    if (end_array.empty()) {
      return ResultErr<ISTError>("Tag format's end array cannot be empty");
    }
    for (const auto& item : end_array) {
      if (!item.is<std::string>()) {
        return ResultErr<ISTError>("Tag format's end array must contain only strings");
      }
      end_strings.push_back(item.get<std::string>());
    }
  } else {
    return ResultErr<ISTError>("Tag format's end field must be a string or array of strings");
  }

  return ResultOk<TagFormat>(
      begin_it->second.get<std::string>(),
      std::make_shared<Format>(std::move(content).Unwrap()),
      std::move(end_strings)
  );
}

Result<TriggeredTagsFormat, ISTError> StructuralTagParser::ParseTriggeredTagsFormat(
    const picojson::object& obj
) {
  // triggers is required.
  auto triggers_it = obj.find("triggers");
  if (triggers_it == obj.end() || !triggers_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("Triggered tags format must have a triggers field with an array");
  }
  const auto& triggers_array = triggers_it->second.get<picojson::array>();
  std::vector<std::string> excluded_strs;
  std::vector<std::string> triggers;
  triggers.reserve(triggers_array.size());
  for (const auto& trigger : triggers_array) {
    if (!trigger.is<std::string>() || trigger.get<std::string>().empty()) {
      return ResultErr<ISTError>("Triggered tags format's triggers must be non-empty strings");
    }
    triggers.push_back(trigger.get<std::string>());
  }
  if (triggers.size() == 0) {
    return ResultErr<ISTError>("Triggered tags format's triggers must be non-empty");
  }
  // tags is required.
  auto tags_it = obj.find("tags");
  if (tags_it == obj.end() || !tags_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("Triggered tags format must have a tags field with an array");
  }
  const auto& tags_array = tags_it->second.get<picojson::array>();
  std::vector<TagFormat> tags;
  tags.reserve(tags_array.size());
  for (const auto& tag : tags_array) {
    auto tag_format = ParseTagFormat(tag);
    if (tag_format.IsErr()) {
      return ResultErr<ISTError>(std::move(tag_format).UnwrapErr());
    }
    tags.push_back(std::move(tag_format).Unwrap());
  }
  if (tags.size() == 0) {
    return ResultErr<ISTError>("Triggered tags format's tags must be non-empty");
  }
  // excludes is optional.
  auto excludes_it = obj.find("excludes");
  if (excludes_it != obj.end()) {
    if (!excludes_it->second.is<picojson::array>()) {
      return ResultErr<ISTError>("Triggered tags format should have a excludes field with an array"
      );
    }
    const auto& excludes_array = excludes_it->second.get<picojson::array>();
    excluded_strs.reserve(excludes_array.size());
    for (const auto& excluded_str : excludes_array) {
      if (!excluded_str.is<std::string>() || excluded_str.get<std::string>().empty()) {
        return ResultErr<ISTError>("Triggered tags format's excluded_strs must be non-empty strings"
        );
      }
      excluded_strs.push_back(excluded_str.get<std::string>());
    }
  }

  // at_least_one is optional.
  bool at_least_one = false;
  auto at_least_one_it = obj.find("at_least_one");
  if (at_least_one_it != obj.end()) {
    if (!at_least_one_it->second.is<bool>()) {
      return ResultErr<ISTError>("at_least_one must be a boolean");
    }
    at_least_one = at_least_one_it->second.get<bool>();
  }
  // stop_after_first is optional.
  bool stop_after_first = false;
  auto stop_after_first_it = obj.find("stop_after_first");
  if (stop_after_first_it != obj.end()) {
    if (!stop_after_first_it->second.is<bool>()) {
      return ResultErr<ISTError>("stop_after_first must be a boolean");
    }
    stop_after_first = stop_after_first_it->second.get<bool>();
  }
  return ResultOk<TriggeredTagsFormat>(
      std::move(triggers), std::move(tags), std::move(excluded_strs), at_least_one, stop_after_first
  );
}

Result<TagsWithSeparatorFormat, ISTError> StructuralTagParser::ParseTagsWithSeparatorFormat(
    const picojson::object& obj
) {
  // tags is required.
  auto tags_it = obj.find("tags");
  if (tags_it == obj.end() || !tags_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("Tags with separator format must have a tags field with an array");
  }
  const auto& tags_array = tags_it->second.get<picojson::array>();
  std::vector<TagFormat> tags;
  tags.reserve(tags_array.size());
  for (const auto& tag : tags_array) {
    auto tag_format = ParseTagFormat(tag);
    if (tag_format.IsErr()) {
      return ResultErr<ISTError>(std::move(tag_format).UnwrapErr());
    }
    tags.push_back(std::move(tag_format).Unwrap());
  }
  if (tags.size() == 0) {
    return ResultErr<ISTError>("Tags with separator format's tags must be non-empty");
  }
  // separator is required (can be empty string).
  auto separator_it = obj.find("separator");
  if (separator_it == obj.end() || !separator_it->second.is<std::string>()) {
    return ResultErr<ISTError>("Tags with separator format's separator field must be a string");
  }
  // at_least_one is optional.
  bool at_least_one = false;
  auto at_least_one_it = obj.find("at_least_one");
  if (at_least_one_it != obj.end()) {
    if (!at_least_one_it->second.is<bool>()) {
      return ResultErr<ISTError>("at_least_one must be a boolean");
    }
    at_least_one = at_least_one_it->second.get<bool>();
  }
  // stop_after_first is optional.
  bool stop_after_first = false;
  auto stop_after_first_it = obj.find("stop_after_first");
  if (stop_after_first_it != obj.end()) {
    if (!stop_after_first_it->second.is<bool>()) {
      return ResultErr<ISTError>("stop_after_first must be a boolean");
    }
    stop_after_first = stop_after_first_it->second.get<bool>();
  }
  return ResultOk<TagsWithSeparatorFormat>(
      std::move(tags), separator_it->second.get<std::string>(), at_least_one, stop_after_first
  );
}

/************** StructuralTag Analyzer **************/

/*!
 * \brief Analyze a StructuralTag and extract useful information for conversion to Grammar.
 */
class StructuralTagAnalyzer {
 public:
  static std::optional<ISTError> Analyze(StructuralTag* structural_tag);

 private:
  /*! \brief A variant that can hold the pointer of any Format types. */
  using FormatPtrVariant = std::variant<
      ConstStringFormat*,
      JSONSchemaFormat*,
      QwenXmlParameterFormat*,
      AnyTextFormat*,
      GrammarFormat*,
      RegexFormat*,
      SequenceFormat*,
      OrFormat*,
      TagFormat*,
      TriggeredTagsFormat*,
      TagsWithSeparatorFormat*>;

  // Call this if we have a pointer to a Format.
  std::optional<ISTError> Visit(Format* format);
  // Call this if we have a pointer to a variant of Format.
  std::optional<ISTError> Visit(FormatPtrVariant format);

  // The following is dispatched from Visit. Don't call them directly because they don't handle
  // stack logics.
  std::optional<ISTError> VisitSub(ConstStringFormat* format);
  std::optional<ISTError> VisitSub(JSONSchemaFormat* format);
  std::optional<ISTError> VisitSub(QwenXmlParameterFormat* format);
  std::optional<ISTError> VisitSub(AnyTextFormat* format);
  std::optional<ISTError> VisitSub(GrammarFormat* format);
  std::optional<ISTError> VisitSub(RegexFormat* format);
  std::optional<ISTError> VisitSub(SequenceFormat* format);
  std::optional<ISTError> VisitSub(OrFormat* format);
  std::optional<ISTError> VisitSub(TagFormat* format);
  std::optional<ISTError> VisitSub(TriggeredTagsFormat* format);
  std::optional<ISTError> VisitSub(TagsWithSeparatorFormat* format);

  std::vector<std::string> DetectEndStrings();
  bool IsUnlimited(const Format& format);

  int visit_format_recursion_depth_ = 0;
  std::vector<FormatPtrVariant> stack_;
};

std::optional<ISTError> StructuralTagAnalyzer::Analyze(StructuralTag* structural_tag) {
  return StructuralTagAnalyzer().Visit(&structural_tag->format);
}

std::vector<std::string> StructuralTagAnalyzer::DetectEndStrings() {
  for (int i = static_cast<int>(stack_.size()) - 1; i >= 0; --i) {
    auto& format = stack_[i];

    if (std::holds_alternative<TagFormat*>(format)) {
      auto* tag = std::get<TagFormat*>(format);
      return tag->end;  // Already a vector
    }
  }
  return {};  // Empty vector
}

bool StructuralTagAnalyzer::IsUnlimited(const Format& format) {
  return std::visit(
      [&](auto&& arg) -> bool {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, AnyTextFormat>) {
          return true;
        } else if constexpr (std::is_same_v<T, TriggeredTagsFormat>) {
          return true;
        } else if constexpr (std::is_same_v<T, TagsWithSeparatorFormat>) {
          return true;
        } else if constexpr (std::is_same_v<T, SequenceFormat>) {
          return arg.is_unlimited_;
        } else if constexpr (std::is_same_v<T, OrFormat>) {
          return arg.is_unlimited_;
        } else {
          return false;
        }
      },
      format
  );
}

std::optional<ISTError> StructuralTagAnalyzer::Visit(Format* format) {
  FormatPtrVariant format_ptr_variant =
      std::visit([&](auto&& arg) -> FormatPtrVariant { return &arg; }, *format);
  return Visit(format_ptr_variant);
}

std::optional<ISTError> StructuralTagAnalyzer::Visit(FormatPtrVariant format) {
  RecursionGuard guard(&visit_format_recursion_depth_);

  // Push format to stack
  stack_.push_back(format);

  // Dispatch to the corresponding visit function
  auto result =
      std::visit([&](auto&& arg) -> std::optional<ISTError> { return VisitSub(arg); }, format);

  // Pop format from stack
  stack_.pop_back();

  return result;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(ConstStringFormat* format) {
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(JSONSchemaFormat* format) {
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(QwenXmlParameterFormat* format) {
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(AnyTextFormat* format) {
  format->detected_end_strs_ = DetectEndStrings();
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(GrammarFormat* format) {
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(RegexFormat* format) {
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(SequenceFormat* format) {
  for (size_t i = 0; i < format->elements.size() - 1; ++i) {
    auto& element = format->elements[i];
    auto err = Visit(&element);
    if (err.has_value()) {
      return err;
    }
    if (IsUnlimited(element)) {
      return ISTError(
          "Only the last element in a sequence can be unlimited, but the " + std::to_string(i) +
          "th element of sequence format is unlimited"
      );
    }
  }

  auto& element = format->elements.back();
  auto err = Visit(&element);
  if (err.has_value()) {
    return err;
  }
  format->is_unlimited_ = IsUnlimited(element);
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(OrFormat* format) {
  bool is_any_unlimited = false;
  bool is_all_unlimited = true;
  for (auto& element : format->elements) {
    auto err = Visit(&element);
    if (err.has_value()) {
      return err;
    }
    auto is_unlimited = IsUnlimited(element);
    is_any_unlimited |= is_unlimited;
    is_all_unlimited &= is_unlimited;
  }

  if (is_any_unlimited && !is_all_unlimited) {
    return ISTError(
        "Now we only support all elements in an or format to be unlimited or all limited, but the "
        "or format has both unlimited and limited elements"
    );
  }

  format->is_unlimited_ = is_any_unlimited;
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(TagFormat* format) {
  auto err = Visit(format->content.get());
  if (err.has_value()) {
    return err;
  }
  auto is_content_unlimited = IsUnlimited(*(format->content));
  if (is_content_unlimited) {
    // Check that at least one end string is non-empty
    bool has_non_empty = false;
    for (const auto& end_str : format->end) {
      if (!end_str.empty()) {
        has_non_empty = true;
        break;
      }
    }
    if (!has_non_empty) {
      return ISTError("When the content is unlimited, at least one end string must be non-empty");
    }
    // Clear the end strings because they are moved to the detected_end_strs_ field.
    format->end.clear();
  }
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(TriggeredTagsFormat* format) {
  for (auto& tag : format->tags) {
    auto err = Visit(&tag);
    if (err.has_value()) {
      return err;
    }
  }
  format->detected_end_strs_ = DetectEndStrings();
  return std::nullopt;
}

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(TagsWithSeparatorFormat* format) {
  for (auto& tag : format->tags) {
    auto err = Visit(&tag);
    if (err.has_value()) {
      return err;
    }
  }
  format->detected_end_strs_ = DetectEndStrings();
  return std::nullopt;
}

/************** StructuralTag to Grammar Converter **************/

class StructuralTagGrammarConverter {
 public:
  static Result<Grammar, ISTError> Convert(const StructuralTag& structural_tag);

 private:
  /*!
   * \brief Visit a Format and return the rule id of the added rule.
   * \param format The Format to visit.
   * \return The rule id of the added rule. If the visit fails, the error is returned.
   */
  Result<int, ISTError> Visit(const Format& format);
  Result<int, ISTError> VisitSub(const ConstStringFormat& format);
  Result<int, ISTError> VisitSub(const JSONSchemaFormat& format);
  Result<int, ISTError> VisitSub(const QwenXmlParameterFormat& format);
  Result<int, ISTError> VisitSub(const AnyTextFormat& format);
  Result<int, ISTError> VisitSub(const GrammarFormat& format);
  Result<int, ISTError> VisitSub(const RegexFormat& format);
  Result<int, ISTError> VisitSub(const SequenceFormat& format);
  Result<int, ISTError> VisitSub(const OrFormat& format);
  Result<int, ISTError> VisitSub(const TagFormat& format);
  Result<int, ISTError> VisitSub(const TriggeredTagsFormat& format);
  Result<int, ISTError> VisitSub(const TagsWithSeparatorFormat& format);
  Grammar AddRootRuleAndGetGrammar(int ref_rule_id);

  bool IsPrefix(const std::string& prefix, const std::string& full_str);

  GrammarBuilder grammar_builder_;
};

bool StructuralTagGrammarConverter::IsPrefix(
    const std::string& prefix, const std::string& full_str
) {
  return prefix.size() <= full_str.size() &&
         std::string_view(full_str).substr(0, prefix.size()) == prefix;
}

Result<Grammar, ISTError> StructuralTagGrammarConverter::Convert(const StructuralTag& structural_tag
) {
  auto converter = StructuralTagGrammarConverter();
  auto result = converter.Visit(structural_tag.format);
  if (result.IsErr()) {
    return ResultErr(std::move(result).UnwrapErr());
  }
  // Add a root rule
  auto root_rule_id = std::move(result).Unwrap();
  return ResultOk(converter.AddRootRuleAndGetGrammar(root_rule_id));
}

Grammar StructuralTagGrammarConverter::AddRootRuleAndGetGrammar(int ref_rule_id) {
  auto expr = grammar_builder_.AddRuleRef(ref_rule_id);
  auto sequence_expr = grammar_builder_.AddSequence({expr});
  auto choices_expr = grammar_builder_.AddChoices({sequence_expr});
  auto root_rule_id = grammar_builder_.AddRuleWithHint("root", choices_expr);
  return grammar_builder_.Get(root_rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::Visit(const Format& format) {
  return std::visit([&](auto&& arg) -> Result<int, ISTError> { return VisitSub(arg); }, format);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const ConstStringFormat& format) {
  auto expr = grammar_builder_.AddByteString(format.value);
  auto sequence_expr = grammar_builder_.AddSequence({expr});
  auto choices_expr = grammar_builder_.AddChoices({sequence_expr});
  return ResultOk(grammar_builder_.AddRuleWithHint("const_string", choices_expr));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const JSONSchemaFormat& format) {
  auto sub_grammar = Grammar::FromJSONSchema(format.json_schema);
  auto added_root_rule_id = SubGrammarAdder().Apply(&grammar_builder_, sub_grammar);
  return ResultOk(added_root_rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const QwenXmlParameterFormat& format
) {
  auto sub_grammar = Grammar::FromEBNF(QwenXMLToolCallingToEBNF(format.xml_schema));
  auto added_root_rule_id = SubGrammarAdder().Apply(&grammar_builder_, sub_grammar);
  return ResultOk(added_root_rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const GrammarFormat& format) {
  auto sub_grammar = Grammar::FromEBNF(format.grammar);
  auto added_root_rule_id = SubGrammarAdder().Apply(&grammar_builder_, sub_grammar);
  return ResultOk(added_root_rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const RegexFormat& format) {
  auto sub_grammar = Grammar::FromRegex(format.pattern);
  auto added_root_rule_id = SubGrammarAdder().Apply(&grammar_builder_, sub_grammar);
  return ResultOk(added_root_rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const AnyTextFormat& format) {
  if (!format.detected_end_strs_.empty()) {
    // Filter out empty strings
    std::vector<std::string> non_empty_ends;
    for (const auto& s : format.detected_end_strs_) {
      if (!s.empty()) {
        non_empty_ends.push_back(s);
      }
    }
    XGRAMMAR_DCHECK(!non_empty_ends.empty())
        << "At least one detected end string must be non-empty";
    // TagDispatch supports multiple stop strings
    auto tag_dispatch_expr = grammar_builder_.AddTagDispatch(
        Grammar::Impl::TagDispatch{{}, false, non_empty_ends, false, format.excludes}
    );
    return ResultOk(grammar_builder_.AddRuleWithHint("any_text", tag_dispatch_expr));
  } else {
    auto any_text_expr = grammar_builder_.AddCharacterClassStar({{0, 0x10FFFF}}, false);
    auto sequence_expr = grammar_builder_.AddSequence({any_text_expr});
    auto choices_expr = grammar_builder_.AddChoices({sequence_expr});
    return ResultOk(grammar_builder_.AddRuleWithHint("any_text", choices_expr));
  }
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const SequenceFormat& format) {
  std::vector<int> rule_ref_ids;
  rule_ref_ids.reserve(format.elements.size());
  for (const auto& element : format.elements) {
    auto result = Visit(element);
    if (result.IsErr()) {
      return result;
    }
    int sub_rule_id = std::move(result).Unwrap();
    rule_ref_ids.push_back(grammar_builder_.AddRuleRef(sub_rule_id));
  }
  auto expr = grammar_builder_.AddChoices({grammar_builder_.AddSequence(rule_ref_ids)});
  return ResultOk(grammar_builder_.AddRuleWithHint("sequence", expr));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const OrFormat& format) {
  std::vector<int> sequence_ids;
  sequence_ids.reserve(format.elements.size());
  for (const auto& element : format.elements) {
    auto result = Visit(element);
    if (result.IsErr()) {
      return result;
    }
    int sub_rule_id = std::move(result).Unwrap();
    auto rule_ref_expr = grammar_builder_.AddRuleRef(sub_rule_id);
    sequence_ids.push_back(grammar_builder_.AddSequence({rule_ref_expr}));
  }
  auto expr = grammar_builder_.AddChoices(sequence_ids);
  return ResultOk(grammar_builder_.AddRuleWithHint("or", expr));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const TagFormat& format) {
  auto result = Visit(*format.content);
  if (result.IsErr()) {
    return result;
  }
  auto sub_rule_id = std::move(result).Unwrap();
  auto begin_expr = grammar_builder_.AddByteString(format.begin);
  auto rule_ref_expr = grammar_builder_.AddRuleRef(sub_rule_id);

  if (format.end.size() > 1) {
    // Multiple end tokens: create end choices rule: Choice(Seq(end1), Seq(end2), ...)
    std::vector<int> end_sequence_ids;
    for (const auto& end_str : format.end) {
      // Use AddEmptyStr() for empty strings, AddByteString() for non-empty
      auto end_expr = end_str.empty() ? grammar_builder_.AddEmptyStr()
                                      : grammar_builder_.AddByteString(end_str);
      end_sequence_ids.push_back(grammar_builder_.AddSequence({end_expr}));
    }
    auto end_choices_expr = grammar_builder_.AddChoices(end_sequence_ids);
    auto end_choices_rule_id = grammar_builder_.AddRuleWithHint("tag_end", end_choices_expr);
    auto end_rule_ref_expr = grammar_builder_.AddRuleRef(end_choices_rule_id);

    auto sequence_expr_id =
        grammar_builder_.AddSequence({begin_expr, rule_ref_expr, end_rule_ref_expr});
    auto choices_expr = grammar_builder_.AddChoices({sequence_expr_id});
    return ResultOk(grammar_builder_.AddRuleWithHint("tag", choices_expr));
  } else if (format.end.size() == 1) {
    // Single end token: use directly (use AddEmptyStr() for empty strings)
    auto end_expr = format.end[0].empty() ? grammar_builder_.AddEmptyStr()
                                          : grammar_builder_.AddByteString(format.end[0]);
    auto sequence_expr_id = grammar_builder_.AddSequence({begin_expr, rule_ref_expr, end_expr});
    auto choices_expr = grammar_builder_.AddChoices({sequence_expr_id});
    return ResultOk(grammar_builder_.AddRuleWithHint("tag", choices_expr));
  } else {
    // End was cleared (unlimited content case) - no end string needed
    auto sequence_expr_id = grammar_builder_.AddSequence({begin_expr, rule_ref_expr});
    auto choices_expr = grammar_builder_.AddChoices({sequence_expr_id});
    return ResultOk(grammar_builder_.AddRuleWithHint("tag", choices_expr));
  }
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const TriggeredTagsFormat& format) {
  // Step 1. Visit all tags and add to grammar
  std::vector<std::vector<int>> trigger_to_tag_ids(format.triggers.size());
  std::vector<int> tag_content_rule_ids;
  tag_content_rule_ids.reserve(format.tags.size());

  for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
    const auto& tag = format.tags[it_tag];
    // Find matched triggers
    int matched_trigger_id = -1;
    for (int it_trigger = 0; it_trigger < static_cast<int>(format.triggers.size()); ++it_trigger) {
      const auto& trigger = format.triggers[it_trigger];
      if (IsPrefix(trigger, tag.begin)) {
        if (matched_trigger_id != -1) {
          return ResultErr<ISTError>("One tag matches multiple triggers in a triggered tags format"
          );
        }
        matched_trigger_id = it_trigger;
      }
    }
    if (matched_trigger_id == -1) {
      return ResultErr<ISTError>("One tag does not match any trigger in a triggered tags format");
    }
    trigger_to_tag_ids[matched_trigger_id].push_back(it_tag);

    // Add the tag content to grammar
    auto result = Visit(*tag.content);
    if (result.IsErr()) {
      return result;
    }
    tag_content_rule_ids.push_back(std::move(result).Unwrap());
  }

  // at_least_one is implemented as generating any one of the tags first, then do optional
  // triggered tags generation. That means we don't generate any text before the first tag.

  // Step 2. Special Case: at_least_one && stop_after_first.
  // Then we will generate exactly one tag without text. We just do a selection between all tags.
  if (format.at_least_one && format.stop_after_first) {
    std::vector<int> choice_elements;
    for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
      const auto& tag = format.tags[it_tag];
      auto begin_expr_id = grammar_builder_.AddByteString(tag.begin);
      auto rule_ref_expr_id = grammar_builder_.AddRuleRef(tag_content_rule_ids[it_tag]);
      if (tag.end.empty()) {
        // Unlimited content case - skip adding end string
        choice_elements.push_back(grammar_builder_.AddSequence({begin_expr_id, rule_ref_expr_id}));
      } else if (tag.end.size() == 1) {
        // Single end token: use directly
        auto end_expr_id = tag.end[0].empty() ? grammar_builder_.AddEmptyStr()
                                              : grammar_builder_.AddByteString(tag.end[0]);
        choice_elements.push_back(
            grammar_builder_.AddSequence({begin_expr_id, rule_ref_expr_id, end_expr_id})
        );
      } else {
        // Multiple end tokens: create end choices rule: Choice(Seq(end1), Seq(end2), ...)
        std::vector<int> end_sequence_ids;
        for (const auto& end_str : tag.end) {
          auto end_expr_id = end_str.empty() ? grammar_builder_.AddEmptyStr()
                                             : grammar_builder_.AddByteString(end_str);
          end_sequence_ids.push_back(grammar_builder_.AddSequence({end_expr_id}));
        }
        auto end_choices_expr = grammar_builder_.AddChoices(end_sequence_ids);
        auto end_choices_rule_id = grammar_builder_.AddRuleWithHint("tag_end", end_choices_expr);
        auto end_rule_ref_expr = grammar_builder_.AddRuleRef(end_choices_rule_id);
        choice_elements.push_back(
            grammar_builder_.AddSequence({begin_expr_id, rule_ref_expr_id, end_rule_ref_expr})
        );
      }
    }
    auto choice_expr_id = grammar_builder_.AddChoices(choice_elements);

    // Handle the detected end strings.
    if (!format.detected_end_strs_.empty()) {
      auto sub_rule_id = grammar_builder_.AddRuleWithHint("triggered_tags_sub", choice_expr_id);
      auto ref_sub_rule_expr_id = grammar_builder_.AddRuleRef(sub_rule_id);
      if (format.detected_end_strs_.size() == 1) {
        // Single detected end string: use directly
        auto end_str_expr_id = format.detected_end_strs_[0].empty()
                                   ? grammar_builder_.AddEmptyStr()
                                   : grammar_builder_.AddByteString(format.detected_end_strs_[0]);
        auto sequence_expr_id =
            grammar_builder_.AddSequence({ref_sub_rule_expr_id, end_str_expr_id});
        choice_expr_id = grammar_builder_.AddChoices({sequence_expr_id});
      } else {
        // Multiple detected end strings: create end choices rule
        std::vector<int> end_sequence_ids;
        for (const auto& end_str : format.detected_end_strs_) {
          auto end_str_expr_id = end_str.empty() ? grammar_builder_.AddEmptyStr()
                                                 : grammar_builder_.AddByteString(end_str);
          end_sequence_ids.push_back(grammar_builder_.AddSequence({end_str_expr_id}));
        }
        auto end_choices_expr = grammar_builder_.AddChoices(end_sequence_ids);
        auto end_choices_rule_id =
            grammar_builder_.AddRuleWithHint("end_choices", end_choices_expr);
        auto end_rule_ref_expr = grammar_builder_.AddRuleRef(end_choices_rule_id);
        auto sequence_expr_id =
            grammar_builder_.AddSequence({ref_sub_rule_expr_id, end_rule_ref_expr});
        choice_expr_id = grammar_builder_.AddChoices({sequence_expr_id});
      }
    }

    return ResultOk(grammar_builder_.AddRuleWithHint("triggered_tags", choice_expr_id));
  }

  // Step 3. Normal Case. We generate mixture of text and triggered tags.
  // - When at_least_one is true, one tag is generated first, then we do triggered tags
  // generation.
  // - When stop_after_first is true, we set loop_after_dispatch of the tag dispatch to false.
  // - When detected_end_str_ is not empty, we use that as the stop_str of the tag dispatch.
  //   Otherwise, we set stop_eos to true to generate until EOS.

  // Step 3.1 Get tag_rule_pairs.
  std::vector<std::pair<std::string, int32_t>> tag_rule_pairs;
  for (int it_trigger = 0; it_trigger < static_cast<int>(format.triggers.size()); ++it_trigger) {
    const auto& trigger = format.triggers[it_trigger];
    std::vector<int> choice_elements;
    for (const auto& tag_id : trigger_to_tag_ids[it_trigger]) {
      const auto& tag = format.tags[tag_id];
      int begin_expr_id = grammar_builder_.AddByteString(tag.begin.substr(trigger.size()));
      int rule_ref_expr_id = grammar_builder_.AddRuleRef(tag_content_rule_ids[tag_id]);
      if (tag.end.empty()) {
        // Unlimited content case - skip adding end string
        choice_elements.push_back(grammar_builder_.AddSequence({begin_expr_id, rule_ref_expr_id}));
      } else if (tag.end.size() == 1) {
        // Single end token: use directly
        int end_expr_id = tag.end[0].empty() ? grammar_builder_.AddEmptyStr()
                                             : grammar_builder_.AddByteString(tag.end[0]);
        choice_elements.push_back(
            grammar_builder_.AddSequence({begin_expr_id, rule_ref_expr_id, end_expr_id})
        );
      } else {
        // Multiple end tokens: create end choices rule: Choice(Seq(end1), Seq(end2), ...)
        std::vector<int> end_sequence_ids;
        for (const auto& end_str : tag.end) {
          int end_expr_id = end_str.empty() ? grammar_builder_.AddEmptyStr()
                                            : grammar_builder_.AddByteString(end_str);
          end_sequence_ids.push_back(grammar_builder_.AddSequence({end_expr_id}));
        }
        auto end_choices_expr = grammar_builder_.AddChoices(end_sequence_ids);
        auto end_choices_rule_id = grammar_builder_.AddRuleWithHint("tag_end", end_choices_expr);
        auto end_rule_ref_expr = grammar_builder_.AddRuleRef(end_choices_rule_id);
        choice_elements.push_back(
            grammar_builder_.AddSequence({begin_expr_id, rule_ref_expr_id, end_rule_ref_expr})
        );
      }
    }
    auto choice_expr_id = grammar_builder_.AddChoices(choice_elements);
    auto sub_rule_id = grammar_builder_.AddRuleWithHint("triggered_tags_group", choice_expr_id);
    tag_rule_pairs.push_back(std::make_pair(trigger, sub_rule_id));
  }

  // Step 3.2 Add TagDispatch.
  int32_t rule_expr_id;
  bool loop_after_dispatch = !format.stop_after_first;
  if (!format.detected_end_strs_.empty()) {
    // Filter out empty strings
    std::vector<std::string> non_empty_ends;
    for (const auto& s : format.detected_end_strs_) {
      if (!s.empty()) {
        non_empty_ends.push_back(s);
      }
    }
    rule_expr_id = grammar_builder_.AddTagDispatch(Grammar::Impl::TagDispatch{
        tag_rule_pairs, false, non_empty_ends, loop_after_dispatch, format.excludes
    });
  } else {
    rule_expr_id = grammar_builder_.AddTagDispatch(
        Grammar::Impl::TagDispatch{tag_rule_pairs, true, {}, loop_after_dispatch, format.excludes}
    );
  }

  // Step 3.3 Consider at_least_one
  if (format.at_least_one) {
    // Construct the first rule
    std::vector<int> first_choice_elements;
    for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
      const auto& tag = format.tags[it_tag];
      auto begin_expr_id = grammar_builder_.AddByteString(tag.begin);
      auto rule_ref_expr_id = grammar_builder_.AddRuleRef(tag_content_rule_ids[it_tag]);
      if (tag.end.empty()) {
        // Unlimited content case - skip adding end string
        first_choice_elements.push_back(
            grammar_builder_.AddSequence({begin_expr_id, rule_ref_expr_id})
        );
      } else if (tag.end.size() == 1) {
        // Single end token: use directly
        auto end_expr_id = tag.end[0].empty() ? grammar_builder_.AddEmptyStr()
                                              : grammar_builder_.AddByteString(tag.end[0]);
        first_choice_elements.push_back(
            grammar_builder_.AddSequence({begin_expr_id, rule_ref_expr_id, end_expr_id})
        );
      } else {
        // Multiple end tokens: create end choices rule: Choice(Seq(end1), Seq(end2), ...)
        std::vector<int> end_sequence_ids;
        for (const auto& end_str : tag.end) {
          auto end_expr_id = end_str.empty() ? grammar_builder_.AddEmptyStr()
                                             : grammar_builder_.AddByteString(end_str);
          end_sequence_ids.push_back(grammar_builder_.AddSequence({end_expr_id}));
        }
        auto end_choices_expr = grammar_builder_.AddChoices(end_sequence_ids);
        auto end_choices_rule_id = grammar_builder_.AddRuleWithHint("tag_end", end_choices_expr);
        auto end_rule_ref_expr = grammar_builder_.AddRuleRef(end_choices_rule_id);
        first_choice_elements.push_back(
            grammar_builder_.AddSequence({begin_expr_id, rule_ref_expr_id, end_rule_ref_expr})
        );
      }
    }
    auto first_choice_expr_id = grammar_builder_.AddChoices(first_choice_elements);
    auto first_rule_id =
        grammar_builder_.AddRuleWithHint("triggered_tags_first", first_choice_expr_id);

    // Construct the full rule
    auto tag_dispatch_rule_id =
        grammar_builder_.AddRuleWithHint("triggered_tags_sub", rule_expr_id);
    auto ref_first_rule_expr_id = grammar_builder_.AddRuleRef(first_rule_id);
    auto ref_tag_dispatch_rule_expr_id = grammar_builder_.AddRuleRef(tag_dispatch_rule_id);
    auto sequence_expr_id =
        grammar_builder_.AddSequence({ref_first_rule_expr_id, ref_tag_dispatch_rule_expr_id});
    rule_expr_id = grammar_builder_.AddChoices({sequence_expr_id});
  }

  auto rule_id = grammar_builder_.AddRuleWithHint("triggered_tags", rule_expr_id);
  return ResultOk(rule_id);
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const TagsWithSeparatorFormat& format
) {
  // The grammar:
  // Step 1. tags_rule: call tags
  //   tags_rule ::= tag1 | tag2 | ... | tagN
  // Step 2. Special handling (stop_after_first is true):
  //   if at_least_one is false:
  //     root ::= tags_rule end_str | end_str
  //   if at_least_one is true:
  //     root ::= tags_rule end_str
  // Step 3. Normal handling (stop_after_first is false):
  //   if at_least_one is false:
  //     root ::= tags_rule tags_rule_sub | end_str
  //   if at_least_one is true:
  //     root ::= tags_rule tags_rule_sub
  //   tags_rule_sub ::= sep tags_rule tags_rule_sub | end_str

  // Step 1. Construct a rule representing any tag
  std::vector<int> choice_ids;
  for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
    auto tag_rule_id = Visit(format.tags[it_tag]);
    if (tag_rule_id.IsErr()) {
      return tag_rule_id;
    }
    auto tag_rule_ref_id = grammar_builder_.AddRuleRef(std::move(tag_rule_id).Unwrap());
    auto sequence_expr_id = grammar_builder_.AddSequence({tag_rule_ref_id});
    choice_ids.push_back(sequence_expr_id);
  }
  auto choice_expr_id = grammar_builder_.AddChoices(choice_ids);
  auto all_tags_rule_id =
      grammar_builder_.AddRuleWithHint("tags_with_separator_tags", choice_expr_id);

  auto all_tags_rule_ref_id = grammar_builder_.AddRuleRef(all_tags_rule_id);

  // Handle end strs - build a choices expr for multiple end strings
  std::vector<int32_t> end_str_expr_ids;
  for (const auto& end_str : format.detected_end_strs_) {
    if (!end_str.empty()) {
      end_str_expr_ids.push_back(grammar_builder_.AddByteString(end_str));
    }
  }
  bool has_end_strs = !end_str_expr_ids.empty();

  // Check if separator matches any end string
  bool separator_matches_end = false;
  for (const auto& end_str : format.detected_end_strs_) {
    if (end_str == format.separator) {
      separator_matches_end = true;
      break;
    }
  }

  // Step 2. Special case (stop_after_first is true):
  if (format.stop_after_first || (has_end_strs && separator_matches_end)) {
    int32_t rule_body_expr_id;
    if (format.at_least_one) {
      if (!has_end_strs) {
        // root ::= tags_rule
        rule_body_expr_id =
            grammar_builder_.AddChoices({grammar_builder_.AddSequence({all_tags_rule_ref_id})});
      } else {
        // root ::= tags_rule end_str1 | tags_rule end_str2 | ...
        std::vector<int> choices;
        for (auto end_str_expr_id : end_str_expr_ids) {
          choices.push_back(grammar_builder_.AddSequence({all_tags_rule_ref_id, end_str_expr_id}));
        }
        rule_body_expr_id = grammar_builder_.AddChoices(choices);
      }
    } else {
      if (!has_end_strs) {
        // root ::= tags_rule | ""
        rule_body_expr_id = grammar_builder_.AddChoices(
            {grammar_builder_.AddSequence({all_tags_rule_ref_id}), grammar_builder_.AddEmptyStr()}
        );
      } else {
        // root ::= tags_rule end_str1 | tags_rule end_str2 | ... | end_str1 | end_str2 | ...
        std::vector<int> choices;
        for (auto end_str_expr_id : end_str_expr_ids) {
          choices.push_back(grammar_builder_.AddSequence({all_tags_rule_ref_id, end_str_expr_id}));
        }
        for (auto end_str_expr_id : end_str_expr_ids) {
          choices.push_back(grammar_builder_.AddSequence({end_str_expr_id}));
        }
        rule_body_expr_id = grammar_builder_.AddChoices(choices);
      }
    }

    auto rule_id = grammar_builder_.AddRuleWithHint("tags_with_separator", rule_body_expr_id);
    return ResultOk(rule_id);
  }

  // Step 3. Normal handling (stop_after_first is false):
  // Step 3.1 Construct sub rule
  auto sub_rule_id = grammar_builder_.AddEmptyRuleWithHint("tags_with_separator_sub");

  // Build end_str_sequence_id: empty if no end strs, otherwise choices of end strs
  int32_t end_str_sequence_id;
  if (!has_end_strs) {
    end_str_sequence_id = grammar_builder_.AddEmptyStr();
  } else if (end_str_expr_ids.size() == 1) {
    end_str_sequence_id = grammar_builder_.AddSequence({end_str_expr_ids[0]});
  } else {
    std::vector<int> end_str_choices;
    for (auto end_str_expr_id : end_str_expr_ids) {
      end_str_choices.push_back(grammar_builder_.AddSequence({end_str_expr_id}));
    }
    end_str_sequence_id = grammar_builder_.AddChoices(end_str_choices);
  }

  // Build the sequence for the recursive case, handling empty separator
  std::vector<int> sub_sequence_elements;
  if (!format.separator.empty()) {
    sub_sequence_elements.push_back(grammar_builder_.AddByteString(format.separator));
  }
  sub_sequence_elements.push_back(all_tags_rule_ref_id);
  sub_sequence_elements.push_back(grammar_builder_.AddRuleRef(sub_rule_id));

  auto sub_rule_body_id = grammar_builder_.AddChoices(
      {grammar_builder_.AddSequence(sub_sequence_elements), end_str_sequence_id}
  );
  grammar_builder_.UpdateRuleBody(sub_rule_id, sub_rule_body_id);

  // Step 3.2 Construct root rule
  std::vector<int> choices = {
      grammar_builder_.AddSequence({all_tags_rule_ref_id, grammar_builder_.AddRuleRef(sub_rule_id)}
      ),
  };
  if (!format.at_least_one) {
    choices.push_back(end_str_sequence_id);
  }
  auto rule_body_expr_id = grammar_builder_.AddChoices(choices);
  auto rule_id = grammar_builder_.AddRuleWithHint("tags_with_separator", rule_body_expr_id);
  return ResultOk(rule_id);
}

/************** StructuralTag Template Filler **************/

/*!
 * \brief Parse placeholders from the given string.
 * \param str The string to parse.
 * \return The parsed placeholders.
 * \details the string should be the format of {{function_name([])?(.arg_name([])?)*}}.
 */
Layers ParsePlaceHolder(const std::string& str) {
  Layers result;
  XGRAMMAR_DCHECK(
      std::sregex_iterator(str.begin(), str.end(), full_placeholder_regex) != std::sregex_iterator()
  );
  auto iter = std::sregex_iterator(str.begin(), str.end(), placeholder_regex);
  for (; iter != std::sregex_iterator(); ++iter) {
    const auto& match = *iter;
    PlaceHolderWithArray placeholder;
    placeholder.name = match[1].str();
    if (match[2].matched) {
      placeholder.is_array = true;
    } else {
      placeholder.is_array = false;
    }
    result.push_back(placeholder);
  }
  return result;
}

/*!
 * \brief Check if the given placeholder structure is valid.
 * \param placeholders The placeholders to check.
 * \return True if the placeholder structure is valid, false otherwise.
 * \details A valid placeholder structure means:
 * - All array-like placeholders do not form a tree structure. For example, {{func[].arg1[]}}
 * with {{func[].arg2[]}} is invalid. {{func[].arg1}} with {{func[].arg2[]}} is valid.
 */
bool IsValidPlaceHolderName(const std::vector<Layers>& placeholders) {
  std::unordered_map<std::string, std::unordered_map<int, std::string>> name_to_array_elements;
  bool placeholder_multiple_required = false;
  for (const auto& layers : placeholders) {
    XGRAMMAR_DCHECK(!layers.empty());
    const auto& first_layer = layers[0];
    if (name_to_array_elements.find(first_layer.name) == name_to_array_elements.end()) {
      name_to_array_elements[first_layer.name] = {};
    }
    auto& array_elements = name_to_array_elements[first_layer.name];
    for (size_t i = 1; i < layers.size(); ++i) {
      const auto& layer = layers[i];
      if (!layer.is_array) {
        continue;
      }
      if (array_elements.find(static_cast<int>(i)) == array_elements.end()) {
        array_elements[static_cast<int>(i)] = layer.name;
        continue;
      }
      if (array_elements[static_cast<int>(i)] != layer.name) {
        return false;
      } else {
        placeholder_multiple_required = true;
      }
    }
  }

  // It will be expanded at the current format. Check if it is mingled.
  if (placeholder_multiple_required) {
    int array_count = 0;
    for (const auto& [_, array_elements] : name_to_array_elements) {
      if (!array_elements.empty()) {
        array_count++;
      }
    }
    if (array_count > 1) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Detect the current longest common placeholder name to expand.
 * \param placeholders The placeholders to detect.
 * \return The detected placeholder layers.
 * \details The placeholder layers are the longest common prefix of all placeholders.
 * If no placeholder is found, return std::nullopt.
 */
void DetectPlaceHolderToExpand(
    const std::vector<Layers>& placeholders, std::optional<Layers>* result
) {
  // Step 1. Check if all the placeholders have the same variable name.
  std::optional<std::string> placeholder_name_opt = std::nullopt;
  for (const auto& layers : placeholders) {
    if (layers.empty()) {
      continue;  // Skip empty layers
    }
    const auto& first_layer = layers[0];
    if (placeholder_name_opt == std::nullopt) {
      placeholder_name_opt = first_layer.name;
      continue;
    }
    if (placeholder_name_opt.value() != first_layer.name) {
      *result = std::nullopt;  // Different placeholder names, no common prefix.
      return;
    }
  }

  int current_placeholder_to_expand = 0;
  int multiple_required_placeholder_to_expand = 0;
  std::optional<std::reference_wrapper<const Layers>> current_layers = std::nullopt;
  // Step 2. Find the longest common prefix of all placeholders.
  for (const auto& layers : placeholders) {
    bool is_array_like =
        std::any_of(layers.begin(), layers.end(), [](const PlaceHolderWithArray& layer) {
          return layer.is_array;
        });

    // If the placeholder is not array-like, we can skip it.
    if (!is_array_like) {
      continue;
    }

    // Considering that we require that array-like placeholders do not form a tree, we only need to
    // record the index of the last array-like layer.
    auto rit = std::find_if(layers.rbegin(), layers.rend(), [](const PlaceHolderWithArray& layer) {
      return layer.is_array;
    });
    XGRAMMAR_DCHECK(rit != layers.rend());
    int index = static_cast<int>(std::distance(layers.begin(), rit.base())) - 1;
    int length = index + 1;
    multiple_required_placeholder_to_expand = std::min(current_placeholder_to_expand, length);
    if (length > current_placeholder_to_expand) {
      current_placeholder_to_expand = length;
      current_layers = layers;
    }
  }

  if (multiple_required_placeholder_to_expand == 0) {
    *result = std::nullopt;  // No array-like placeholders found.
    return;
  }
  XGRAMMAR_DCHECK(
      current_layers.has_value() &&
      static_cast<int>(current_layers->get().size()) >= multiple_required_placeholder_to_expand
  );
  *result = Layers(
      current_layers->get().begin(),
      current_layers->get().begin() + multiple_required_placeholder_to_expand
  );
}

/*!
 * \brief Detect all template placeholder names in the given string.
 * \param str The string to detect.
 * \return The detected template placeholder name. If no placeholder is found, return std::nullopt.
 * \details A template placeholder is in the format of {function_name[].arg_name}.
 */
Result<std::vector<Layers>, StructuralTagError> DetectTemplatePlaceholderNames(
    const std::string& str
) {
  std::optional<std::string> placeholder_name_opt = std::nullopt;
  std::vector<Layers> all_placeholders;
  auto iter = std::sregex_iterator(str.begin(), str.end(), full_placeholder_regex);
  for (; iter != std::sregex_iterator(); ++iter) {
    const auto& match = *iter;
    std::string function_name = match[1].str();
    Layers placeholders = ParsePlaceHolder(match.str());
    all_placeholders.push_back(std::move(placeholders));
    if (placeholder_name_opt.has_value() && placeholder_name_opt.value() != function_name) {
      return ResultErr(InvalidStructuralTagError(
          "Multiple different placeholder names found in the same string: '" + str + "'"
      ));
    } else {
      placeholder_name_opt = function_name;
    }
  }
  return ResultOk(all_placeholders);
}

Result<std::vector<Layers>, StructuralTagError> DetectTemplatePlaceholderNames(
    const std::vector<std::string>& str_vec
) {
  std::vector<Layers> all_placeholders;
  for (const auto& str : str_vec) {
    auto result = DetectTemplatePlaceholderNames(str);
    if (result.IsErr()) {
      return result;
    }
    auto placeholders = std::move(result).Unwrap();
    all_placeholders.insert(all_placeholders.end(), placeholders.begin(), placeholders.end());
  }
  return ResultOk(all_placeholders);
}

/*!
 * \brief A structural tag template filler, used to fill the structral tags with the given values.
 */
class StructuralTagTemplateFiller {
 public:
  Result<StructuralTag, StructuralTagError> Apply(
      const StructuralTag& template_structural_tag, const picojson::value& values
  );

  bool HasUnfilledPlaceholders(const StructuralTag& structural_tag);

 private:
  const int kNotArray = -1;
  std::unordered_map<std::string, std::vector<Layers>> format_to_placeholder_names_;
  std::vector<std::pair<PlaceHolderWithArray, int>> current_expanded_placeholder_and_index_;
  const picojson::value* values_ = nullptr;

  Result<std::vector<Layers>, StructuralTagError> Visit(const Format& format);
  Result<std::vector<Format>, StructuralTagError> VisitExpand(
      const Format& format_template_to_expand
  );

  void AddExpandInfo(
      const PlaceHolderWithArray& placeholder, const int& index, int* added_layers_count
  );
  void RemoveExpandInfo(int* added_layers_count);

  Result<std::vector<Layers>, StructuralTagError> VisitSub(const ConstStringFormat& format);
  Result<std::vector<Layers>, StructuralTagError> VisitSub(const JSONSchemaFormat& format);
  Result<std::vector<Layers>, StructuralTagError> VisitSub(const QwenXmlParameterFormat& format);
  Result<std::vector<Layers>, StructuralTagError> VisitSub(const AnyTextFormat& format);
  Result<std::vector<Layers>, StructuralTagError> VisitSub(const GrammarFormat& format);
  Result<std::vector<Layers>, StructuralTagError> VisitSub(const RegexFormat& format);
  Result<std::vector<Layers>, StructuralTagError> VisitSub(const SequenceFormat& format);
  Result<std::vector<Layers>, StructuralTagError> VisitSub(const OrFormat& format);
  Result<std::vector<Layers>, StructuralTagError> VisitSub(const TagFormat& format);
  Result<std::vector<Layers>, StructuralTagError> VisitSub(const TriggeredTagsFormat& format);
  Result<std::vector<Layers>, StructuralTagError> VisitSub(const TagsWithSeparatorFormat& format);

  Result<std::vector<Format>, StructuralTagError> VisitExpandSub(
      const ConstStringFormat& format_template_to_expand, int* added_layers_count
  );
  Result<std::vector<Format>, StructuralTagError> VisitExpandSub(
      const JSONSchemaFormat& format_template_to_expand, int* added_layers_count
  );
  Result<std::vector<Format>, StructuralTagError> VisitExpandSub(
      const QwenXmlParameterFormat& format_template_to_expand, int* added_layers_count
  );
  Result<std::vector<Format>, StructuralTagError> VisitExpandSub(
      const AnyTextFormat& format_template_to_expand, int* added_layers_count
  );
  Result<std::vector<Format>, StructuralTagError> VisitExpandSub(
      const GrammarFormat& format_template_to_expand, int* added_layers_count
  );
  Result<std::vector<Format>, StructuralTagError> VisitExpandSub(
      const RegexFormat& format_template_to_expand, int* added_layers_count
  );
  Result<std::vector<Format>, StructuralTagError> VisitExpandSub(
      const SequenceFormat& format_template_to_expand, int* added_layers_count
  );
  Result<std::vector<Format>, StructuralTagError> VisitExpandSub(
      const OrFormat& format_template_to_expand, int* added_layers_count
  );
  Result<std::vector<Format>, StructuralTagError> VisitExpandSub(
      const TagFormat& format_template_to_expand, int* added_layers_count
  );
  Result<std::vector<Format>, StructuralTagError> VisitExpandSub(
      const TriggeredTagsFormat& format_template_to_expand, int* added_layers_count
  );
  Result<std::vector<Format>, StructuralTagError> VisitExpandSub(
      const TagsWithSeparatorFormat& format_template_to_expand, int* added_layers_count
  );

  Result<std::vector<std::string>, StructuralTagError> VisitExpandBasicFormat(
      const std::string& format_json_str, const std::string& format_value
  );

  Result<std::string, StructuralTagError> ReplacePlaceHolder(const std::string& str);
  Result<std::vector<std::string>, StructuralTagError> ReplacePlaceHolders(
      const std::vector<std::string>& str_vec
  );
};

void StructuralTagTemplateFiller::AddExpandInfo(
    const PlaceHolderWithArray& placeholder, const int& index, int* added_layers_count
) {
  current_expanded_placeholder_and_index_.emplace_back(placeholder, index);
  (*added_layers_count)++;
}

void StructuralTagTemplateFiller::RemoveExpandInfo(int* added_layers_count) {
  XGRAMMAR_DCHECK(!current_expanded_placeholder_and_index_.empty());
  current_expanded_placeholder_and_index_.pop_back();
  (*added_layers_count)--;
}

Result<std::string, StructuralTagError> StructuralTagTemplateFiller::ReplacePlaceHolder(
    const std::string& str
) {
  std::string result_str = "";
  int last_match_pos = 0;
  // Match all placeholders in the string.
  auto full_iter = std::sregex_iterator(str.begin(), str.end(), full_placeholder_regex);
  for (; full_iter != std::sregex_iterator(); ++full_iter) {
    const auto& match = *full_iter;

    // Append the part before the match.
    result_str.append(str, last_match_pos, match.position() - last_match_pos);
    last_match_pos = match.position() + match.length();

    // Get the placeholder Layer.
    Layers placeholder_layers = ParsePlaceHolder(match.str());
    const auto& full_match_str = match.str();
    auto name_iter =
        std::sregex_iterator(full_match_str.begin(), full_match_str.end(), placeholder_regex);

    // Get the value from values_
    XGRAMMAR_DCHECK(values_ != nullptr && values_->is<picojson::object>());
    auto current_value = std::cref(*values_);

    auto rit = std::find_if(
        placeholder_layers.rbegin(),
        placeholder_layers.rend(),
        [](const PlaceHolderWithArray& layer) { return layer.is_array; }
    );
    int last_array_index =
        static_cast<int>(std::distance(placeholder_layers.begin(), rit.base())) - 1;
    int last_array_length = last_array_index + 1;

    XGRAMMAR_DCHECK(
        last_array_length <= static_cast<int>(current_expanded_placeholder_and_index_.size())
    ) << "last_array_length: "
      << last_array_length << ", expanded_size:" << current_expanded_placeholder_and_index_.size();

    for (int i = 0; i < last_array_length; i++) {
      if (placeholder_layers[i].name != current_expanded_placeholder_and_index_[i].first.name ||
          placeholder_layers[i].is_array !=
              current_expanded_placeholder_and_index_[i].first.is_array) {
        return ResultErr(InvalidStructuralTagError(
            "Mingled placeholder names found, which indicates that there is a product of "
            "placeholders, which is ambiguous to expand."
        ));
      }
      if (placeholder_layers[i].is_array) {
        if ((current_value.get().is<picojson::object>()) &&
            current_value.get().contains(placeholder_layers[i].name) &&
            current_value.get().get(placeholder_layers[i].name).is<picojson::array>() &&
            static_cast<int>(
                current_value.get().get(placeholder_layers[i].name).get<picojson::array>().size()
            ) > current_expanded_placeholder_and_index_[i].second) {
          current_value =
              current_value.get()
                  .get(placeholder_layers[i].name)
                  .get<picojson::array>()[current_expanded_placeholder_and_index_[i].second];
        } else {
          std::string current_value_str;
          bool first_name = true;
          for (int j = 0; j <= i; j++) {
            if (first_name) {
              first_name = false;
            } else {
              current_value_str += ".";
            }
            current_value_str += placeholder_layers[i].name;
            if (placeholder_layers[i].is_array) {
              current_value_str += "[]";
            }
          }
          return ResultErr(InvalidStructuralTagError(
              "Required arguments: " + current_value_str + " is not correct!"
          ));
        }
      } else {
        if (current_value.get().is<picojson::object>() &&
            current_value.get().contains(placeholder_layers[i].name)) {
          current_value = current_value.get().get(placeholder_layers[i].name);
        } else {
          std::string current_value_str;
          bool first_name = true;
          for (int j = 0; j <= i; j++) {
            if (first_name) {
              first_name = false;
            } else {
              current_value_str += ".";
            }
            current_value_str += placeholder_layers[i].name;
            if (placeholder_layers[i].is_array) {
              current_value_str += "[]";
            }
          }
          return ResultErr(InvalidStructuralTagError(
              "Required arguments: " + current_value_str + " is not correct!"
          ));
        }
      }
    }

    for (int i = last_array_length; i < static_cast<int>(placeholder_layers.size()); i++) {
      if (current_value.get().is<picojson::object>() &&
          current_value.get().contains(placeholder_layers[i].name)) {
        current_value = current_value.get().get(placeholder_layers[i].name);
      } else {
        std::string current_value_str;
        bool first_name = true;
        for (int j = 0; j <= i; j++) {
          if (first_name) {
            first_name = false;
          } else {
            current_value_str += ".";
          }
          current_value_str += placeholder_layers[i].name;
          if (placeholder_layers[i].is_array) {
            current_value_str += "[]";
          }
        }
        return ResultErr(InvalidStructuralTagError(
            "Required arguments: " + current_value_str + " is not correct!"
        ));
      }
    }
    if (current_value.get().is<std::string>()) {
      result_str.append(current_value.get().to_str());
    } else {
      result_str.append(current_value.get().serialize());
    }
  }

  // Append the part after the last match.
  result_str.append(str, last_match_pos, str.size() - last_match_pos);
  return ResultOk(result_str);
}

Result<std::vector<std::string>, StructuralTagError>
StructuralTagTemplateFiller::ReplacePlaceHolders(const std::vector<std::string>& strs) {
  std::vector<std::string> result;
  result.reserve(strs.size());
  for (const auto& str : strs) {
    auto replaced_str_result = ReplacePlaceHolder(str);
    if (replaced_str_result.IsErr()) {
      return ResultErr(std::move(replaced_str_result).UnwrapErr());
    }
    result.push_back(std::move(replaced_str_result).Unwrap());
  }
  return ResultOk(result);
}

Result<std::vector<Layers>, StructuralTagError> StructuralTagTemplateFiller::Visit(
    const Format& format
) {
  auto result = std::visit(
      [&](auto&& arg) -> Result<std::vector<Layers>, StructuralTagError> { return VisitSub(arg); },
      format
  );
  if (result.IsErr()) {
    return result;
  }
  auto placeholder_names = std::move(result).Unwrap();
  auto serialized_format =
      std::visit([&](auto&& arg) -> std::string { return arg.ToJSON().serialize(); }, format);
  format_to_placeholder_names_[serialized_format] = placeholder_names;
  return ResultOk(placeholder_names);
}

Result<std::vector<Format>, StructuralTagError> StructuralTagTemplateFiller::VisitExpand(
    const Format& format_template_to_expand
) {
  int added_layers_count = 0;
  auto result = std::visit(
      [&](auto&& arg) -> Result<std::vector<Format>, StructuralTagError> {
        return VisitExpandSub(arg, &added_layers_count);
      },
      format_template_to_expand
  );
  current_expanded_placeholder_and_index_.erase(
      current_expanded_placeholder_and_index_.end() - added_layers_count,
      current_expanded_placeholder_and_index_.end()
  );
  return result;
}

Result<std::vector<Layers>, StructuralTagError> StructuralTagTemplateFiller::VisitSub(
    const ConstStringFormat& format
) {
  auto result = DetectTemplatePlaceholderNames(format.value);
  if (result.IsErr()) {
    return result;
  }
  auto unwraped_result = std::move(result).Unwrap();
  if (IsValidPlaceHolderName(unwraped_result)) {
    return ResultOk(unwraped_result);
  } else {
    return ResultErr(InvalidStructuralTagError(
        "Invalid placeholder structure found in const string format: '" + format.value + "'"
    ));
  }
}

Result<std::vector<Layers>, StructuralTagError> StructuralTagTemplateFiller::VisitSub(
    const JSONSchemaFormat& format
) {
  if (FullyMatchesPlaceholder(format.json_schema)) {
    auto result = DetectTemplatePlaceholderNames(format.json_schema);
    if (result.IsErr()) {
      return result;
    }
    auto unwraped_result = std::move(result).Unwrap();
    if (IsValidPlaceHolderName(unwraped_result)) {
      return ResultOk(unwraped_result);
    } else {
      return ResultErr(InvalidStructuralTagError(
          "Invalid placeholder structure found in JSON schema format: '" + format.json_schema + "'"
      ));
    }
  } else {
    return ResultOk(std::vector<Layers>{});
  }
}

Result<std::vector<Layers>, StructuralTagError> StructuralTagTemplateFiller::VisitSub(
    const QwenXmlParameterFormat& format
) {
  if (FullyMatchesPlaceholder(format.xml_schema)) {
    return DetectTemplatePlaceholderNames(format.xml_schema);
  } else {
    return ResultOk(std::vector<Layers>{});
  }
}

Result<std::vector<Layers>, StructuralTagError> StructuralTagTemplateFiller::VisitSub(
    const AnyTextFormat& format
) {
  // AnyTextFormat does not contain any placeholders.
  return ResultOk(std::vector<Layers>{});
}

Result<std::vector<Layers>, StructuralTagError> StructuralTagTemplateFiller::VisitSub(
    const GrammarFormat& format
) {
  if (FullyMatchesPlaceholder(format.grammar)) {
    return DetectTemplatePlaceholderNames(format.grammar);
  } else {
    return ResultOk(std::vector<Layers>{});
  }
}

Result<std::vector<Layers>, StructuralTagError> StructuralTagTemplateFiller::VisitSub(
    const RegexFormat& format
) {
  if (FullyMatchesPlaceholder(format.pattern)) {
    return DetectTemplatePlaceholderNames(format.pattern);
  } else {
    return ResultOk(std::vector<Layers>{});
  }
}

Result<std::vector<Layers>, StructuralTagError> StructuralTagTemplateFiller::VisitSub(
    const SequenceFormat& format
) {
  std::vector<Layers> all_placeholders;
  for (const auto& element : format.elements) {
    auto result = Visit(element);
    if (result.IsErr()) {
      return result;
    }
    auto sub_placeholders = std::move(result).Unwrap();
    all_placeholders.insert(
        all_placeholders.end(), sub_placeholders.begin(), sub_placeholders.end()
    );
  }

  if (IsValidPlaceHolderName(all_placeholders)) {
    return ResultOk(all_placeholders);
  } else {
    return ResultErr(
        InvalidStructuralTagError("Invalid placeholder structure found in sequence format")
    );
  }
}

Result<std::vector<Layers>, StructuralTagError> StructuralTagTemplateFiller::VisitSub(
    const OrFormat& format
) {
  std::vector<Layers> all_placeholders;
  for (const auto& element : format.elements) {
    auto result = Visit(element);
    if (result.IsErr()) {
      return result;
    }
    auto sub_placeholders = std::move(result).Unwrap();
    all_placeholders.insert(
        all_placeholders.end(), sub_placeholders.begin(), sub_placeholders.end()
    );
  }

  if (IsValidPlaceHolderName(all_placeholders)) {
    return ResultOk(all_placeholders);
  } else {
    return ResultErr(InvalidStructuralTagError("Invalid placeholder structure found in or format"));
  }
}

Result<std::vector<Layers>, StructuralTagError> StructuralTagTemplateFiller::VisitSub(
    const TagFormat& format
) {
  std::vector<Layers> all_placeholders;

  auto result = Visit(*format.content);
  if (result.IsErr()) {
    return result;
  }

  auto sub_placeholders = std::move(result).Unwrap();
  all_placeholders.insert(all_placeholders.end(), sub_placeholders.begin(), sub_placeholders.end());

  auto begin_result = DetectTemplatePlaceholderNames(format.begin);
  if (begin_result.IsErr()) {
    return begin_result;
  }
  auto begin_placeholders = std::move(begin_result).Unwrap();
  all_placeholders.insert(
      all_placeholders.end(), begin_placeholders.begin(), begin_placeholders.end()
  );

  auto end_result = DetectTemplatePlaceholderNames(format.end);
  if (end_result.IsErr()) {
    return end_result;
  }
  auto end_placeholders = std::move(end_result).Unwrap();
  all_placeholders.insert(all_placeholders.end(), end_placeholders.begin(), end_placeholders.end());

  if (IsValidPlaceHolderName(all_placeholders)) {
    return ResultOk(all_placeholders);
  } else {
    return ResultErr(InvalidStructuralTagError("Invalid placeholder structure found in tag format")
    );
  }
}

Result<std::vector<Layers>, StructuralTagError> StructuralTagTemplateFiller::VisitSub(
    const TriggeredTagsFormat& format
) {
  std::vector<Layers> all_placeholders;
  for (const auto& tag : format.tags) {
    auto result = Visit(tag);
    if (result.IsErr()) {
      return result;
    }
    auto sub_placeholders = std::move(result).Unwrap();
    all_placeholders.insert(
        all_placeholders.end(), sub_placeholders.begin(), sub_placeholders.end()
    );
  }

  if (IsValidPlaceHolderName(all_placeholders)) {
    return ResultOk(all_placeholders);
  } else {
    return ResultErr(
        InvalidStructuralTagError("Invalid placeholder structure found in triggered tags format")
    );
  }
}

Result<std::vector<Layers>, StructuralTagError> StructuralTagTemplateFiller::VisitSub(
    const TagsWithSeparatorFormat& format
) {
  std::vector<Layers> all_placeholders;
  for (const auto& tag : format.tags) {
    auto result = Visit(tag);
    if (result.IsErr()) {
      return result;
    }
    auto sub_placeholders = std::move(result).Unwrap();
    all_placeholders.insert(
        all_placeholders.end(), sub_placeholders.begin(), sub_placeholders.end()
    );
  }

  if (IsValidPlaceHolderName(all_placeholders)) {
    return ResultOk(all_placeholders);
  } else {
    return ResultErr(InvalidStructuralTagError(
        "Invalid placeholder structure found in tags with separator format"
    ));
  }
}

Result<std::vector<std::string>, StructuralTagError>
StructuralTagTemplateFiller::VisitExpandBasicFormat(
    const std::string& format_json_str, const std::string& format_value
) {
  int added_layers_count = 0;
  std::vector<std::string> result_strs;
  bool expanded = false;
  // Step 1. Get the placeholders.
  auto result = format_to_placeholder_names_.find(format_json_str);
  XGRAMMAR_DCHECK(result != format_to_placeholder_names_.end());
  auto& placeholders = result->second;
  int longest_array_placeholder_length = 0;
  const Layers* longest_array_layer = nullptr;

  // Step 2. Compare the placeholders with the current expanded placeholders.
  for (const auto& placeholder : placeholders) {
    // If the placeholder is not array-like, we just skip it.

    // If the placeholder is array-like, we need to check if it is already expanded.
    // Step 2.1. Compare it with the current expanded placeholders.
    auto rit = std::find_if(
        placeholder.rbegin(),
        placeholder.rend(),
        [](const PlaceHolderWithArray& layer) { return layer.is_array; }
    );
    int last_array_index = static_cast<int>(std::distance(placeholder.begin(), rit.base())) - 1;
    int last_array_length = last_array_index + 1;
    int min_of_size = std::min(
        static_cast<int>(current_expanded_placeholder_and_index_.size()),
        static_cast<int>(last_array_length)
    );
    for (int i = 0; i < min_of_size; ++i) {
      if (current_expanded_placeholder_and_index_[i].first.name != placeholder[i].name ||
          current_expanded_placeholder_and_index_[i].first.is_array != placeholder[i].is_array) {
        return ResultErr(InvalidStructuralTagError(
            "Mingled placeholder names found, which indicates that there is a product of "
            "placeholders, which is ambiguous to expand."
        ));
      }
    }

    if (last_array_length > longest_array_placeholder_length) {
      longest_array_placeholder_length = last_array_length;
      longest_array_layer = &placeholder;
    }
  }

  // Step 2.2. If the placeholder is not expanded, we need to expand it.

  for (int i = current_expanded_placeholder_and_index_.size(); i < longest_array_placeholder_length;
       ++i) {
    const auto& ph = longest_array_layer->at(i);
    if (!ph.is_array) {
      AddExpandInfo(ph, kNotArray, &added_layers_count);
      continue;
    }

    auto current_value = std::cref(*values_);
    for (const auto& expanded_placeholder : current_expanded_placeholder_and_index_) {
      XGRAMMAR_DCHECK(current_value.get().contains(expanded_placeholder.first.name));
      if (expanded_placeholder.first.is_array) {
        XGRAMMAR_DCHECK(current_value.get().is<picojson::object>());
        XGRAMMAR_DCHECK(
            current_value.get().get(expanded_placeholder.first.name).is<picojson::array>()
        );
        current_value = current_value.get()
                            .get(expanded_placeholder.first.name)
                            .get<picojson::array>()[expanded_placeholder.second];
      } else {
        current_value = current_value.get().get(expanded_placeholder.first.name);
      }
    }
    if (!current_value.get().is<picojson::object>() || (!current_value.get().contains(ph.name))) {
      return ResultErr(InvalidStructuralTagError(
          "Placeholder name '" + ph.name +
          "' not found in values, which is required for the template: '" + format_value + "'"
      ));
    }
    current_value = current_value.get().get(ph.name);
    expanded = true;
    for (int index = 0; index < static_cast<int>(current_value.get().get<picojson::array>().size());
         ++index) {
      AddExpandInfo(ph, index, &added_layers_count);
      auto tmp_formats = VisitExpandBasicFormat(format_json_str, format_value);
      if (tmp_formats.IsErr()) {
        return tmp_formats;
      }
      auto unwrapped_tmp_formats = std::move(tmp_formats).Unwrap();
      result_strs.insert(
          result_strs.end(), unwrapped_tmp_formats.begin(), unwrapped_tmp_formats.end()
      );
      RemoveExpandInfo(&added_layers_count);
    }
    break;  // Only expand one placeholder at a time.
  }

  if (expanded) {
    while (added_layers_count > 0) {
      RemoveExpandInfo(&added_layers_count);
    }
    return ResultOk(result_strs);
  }

  // Step 3. Fill the placeholders with the values.
  auto replaced_str_result = ReplacePlaceHolder(format_value);
  while (added_layers_count > 0) {
    RemoveExpandInfo(&added_layers_count);
  }
  if (replaced_str_result.IsErr()) {
    return ResultErr(std::move(replaced_str_result).UnwrapErr());
  }
  return ResultOk<std::vector<std::string>>({std::move(replaced_str_result).Unwrap()});
}

Result<std::vector<Format>, StructuralTagError> StructuralTagTemplateFiller::VisitExpandSub(
    const ConstStringFormat& format_template_to_expand, int* added_layers_count
) {
  const auto& json_str = FormatToJSONstr(format_template_to_expand);
  const auto& placeholder_result = format_to_placeholder_names_.find(json_str);
  XGRAMMAR_DCHECK(placeholder_result != format_to_placeholder_names_.end());
  if (placeholder_result->second.empty()) {
    // No placeholders, return the original format.
    return ResultOk<std::vector<Format>>({format_template_to_expand});
  }

  auto result = VisitExpandBasicFormat(json_str, format_template_to_expand.value);

  if (result.IsErr()) {
    return ResultErr(std::move(result).UnwrapErr());
  }
  auto replaced_strs = std::move(result).Unwrap();
  std::vector<Format> filled_formats;
  filled_formats.reserve(replaced_strs.size());
  for (const auto& replaced_str : replaced_strs) {
    filled_formats.push_back(ConstStringFormat{replaced_str});
  }
  return ResultOk(filled_formats);
}

Result<std::vector<Format>, StructuralTagError> StructuralTagTemplateFiller::VisitExpandSub(
    const JSONSchemaFormat& format_template_to_expand, int* added_layers_count
) {
  const auto& json_str = FormatToJSONstr(format_template_to_expand);
  const auto& placeholder_result = format_to_placeholder_names_.find(json_str);
  XGRAMMAR_DCHECK(placeholder_result != format_to_placeholder_names_.end());
  if (placeholder_result->second.empty()) {
    // No placeholders, return the original format.
    return ResultOk<std::vector<Format>>({format_template_to_expand});
  }

  auto result = VisitExpandBasicFormat(json_str, format_template_to_expand.json_schema);

  if (result.IsErr()) {
    return ResultErr(std::move(result).UnwrapErr());
  }
  auto replaced_strs = std::move(result).Unwrap();
  std::vector<Format> filled_formats;
  filled_formats.reserve(replaced_strs.size());
  for (const auto& replaced_str : replaced_strs) {
    filled_formats.push_back(JSONSchemaFormat{replaced_str});
  }
  return ResultOk(filled_formats);
}

Result<std::vector<Format>, StructuralTagError> StructuralTagTemplateFiller::VisitExpandSub(
    const QwenXmlParameterFormat& format_template_to_expand, int* added_layers_count
) {
  const auto& json_str = FormatToJSONstr(format_template_to_expand);
  const auto& placeholder_result = format_to_placeholder_names_.find(json_str);
  XGRAMMAR_DCHECK(placeholder_result != format_to_placeholder_names_.end());
  if (placeholder_result->second.empty()) {
    // No placeholders, return the original format.
    return ResultOk<std::vector<Format>>({format_template_to_expand});
  }

  auto result = VisitExpandBasicFormat(json_str, format_template_to_expand.xml_schema);

  if (result.IsErr()) {
    return ResultErr(std::move(result).UnwrapErr());
  }
  auto replaced_strs = std::move(result).Unwrap();
  std::vector<Format> filled_formats;
  filled_formats.reserve(replaced_strs.size());
  for (const auto& replaced_str : replaced_strs) {
    filled_formats.push_back(QwenXmlParameterFormat{replaced_str});
  }
  return ResultOk(filled_formats);
}

Result<std::vector<Format>, StructuralTagError> StructuralTagTemplateFiller::VisitExpandSub(
    const AnyTextFormat& format_template_to_expand, int* added_layers_count
) {
  // AnyTextFormat does not contain any placeholders.
  return ResultOk<std::vector<Format>>({format_template_to_expand});
}

Result<std::vector<Format>, StructuralTagError> StructuralTagTemplateFiller::VisitExpandSub(
    const GrammarFormat& format_template_to_expand, int* added_layers_count
) {
  const auto& json_str = FormatToJSONstr(format_template_to_expand);
  const auto& placeholder_result = format_to_placeholder_names_.find(json_str);
  XGRAMMAR_DCHECK(placeholder_result != format_to_placeholder_names_.end());
  if (placeholder_result->second.empty()) {
    // No placeholders, return the original format.
    return ResultOk<std::vector<Format>>({format_template_to_expand});
  }

  auto result = VisitExpandBasicFormat(json_str, format_template_to_expand.grammar);

  if (result.IsErr()) {
    return ResultErr(std::move(result).UnwrapErr());
  }
  auto replaced_strs = std::move(result).Unwrap();
  std::vector<Format> filled_formats;
  filled_formats.reserve(replaced_strs.size());
  for (const auto& replaced_str : replaced_strs) {
    filled_formats.push_back(GrammarFormat{replaced_str});
  }
  return ResultOk(filled_formats);
}

Result<std::vector<Format>, StructuralTagError> StructuralTagTemplateFiller::VisitExpandSub(
    const RegexFormat& format_template_to_expand, int* added_layers_count
) {
  const auto& json_str = FormatToJSONstr(format_template_to_expand);
  const auto& placeholder_result = format_to_placeholder_names_.find(json_str);
  XGRAMMAR_DCHECK(placeholder_result != format_to_placeholder_names_.end());
  if (placeholder_result->second.empty()) {
    // No placeholders, return the original format.
    return ResultOk<std::vector<Format>>({format_template_to_expand});
  }

  auto result = VisitExpandBasicFormat(json_str, format_template_to_expand.pattern);

  if (result.IsErr()) {
    return ResultErr(std::move(result).UnwrapErr());
  }
  auto replaced_strs = std::move(result).Unwrap();
  std::vector<Format> filled_formats;
  filled_formats.reserve(replaced_strs.size());
  for (const auto& replaced_str : replaced_strs) {
    filled_formats.push_back(RegexFormat{replaced_str});
  }
  return ResultOk(filled_formats);
}

Result<std::vector<Format>, StructuralTagError> StructuralTagTemplateFiller::VisitExpandSub(
    const SequenceFormat& format_template_to_expand, int* added_layers_count
) {
  // Step 1. Collect all the placeholders in the sequence format, and get the placeholder names
  // to expand.
  std::vector<Layers> all_layers;
  for (const auto& element : format_template_to_expand.elements) {
    auto json_str = FormatToJSONstr(element);
    const auto& placeholder_result = format_to_placeholder_names_.find(json_str);
    XGRAMMAR_DCHECK(placeholder_result != format_to_placeholder_names_.end());
    all_layers.insert(
        all_layers.end(), placeholder_result->second.begin(), placeholder_result->second.end()
    );
  }
  std::optional<Layers> layers_to_expand = std::nullopt;
  DetectPlaceHolderToExpand(all_layers, &layers_to_expand);

  // Step 2. Compare the placeholders with the current expanded placeholders, and determine
  // whether we need to expand the placeholders.
  bool need_expand = layers_to_expand.has_value() &&
                     layers_to_expand->size() > current_expanded_placeholder_and_index_.size();

  // Step 3. If we do not need to expand, we will fill it at the current stage.
  if (!need_expand) {
    std::vector<Format> sequence;
    for (const auto& element : format_template_to_expand.elements) {
      auto sub_result = VisitExpand(element);
      if (sub_result.IsErr()) {
        return sub_result;
      }
      auto unwrapped_sub_result = std::move(sub_result).Unwrap();
      if (unwrapped_sub_result.empty()) {
        continue;
      }
      if (unwrapped_sub_result.size() == 1) {
        sequence.push_back(unwrapped_sub_result[0]);
        continue;
      }
      sequence.push_back(OrFormat(unwrapped_sub_result));
    }
    return ResultOk(std::vector<Format>{SequenceFormat(sequence)});
  }

  // Step 4. If we need to expand the placeholders, we will expand them and return the expanded
  // formats recursively.
  std::vector<Format> results;
  auto current_value = std::cref(*values_);
  for (const auto& expanded_placeholder : current_expanded_placeholder_and_index_) {
    XGRAMMAR_DCHECK(current_value.get().contains(expanded_placeholder.first.name));
    if (expanded_placeholder.first.is_array) {
      XGRAMMAR_DCHECK(current_value.get().get(expanded_placeholder.first.name).is<picojson::array>()
      );
      current_value = current_value.get()
                          .get(expanded_placeholder.first.name)
                          .get<picojson::array>()[expanded_placeholder.second];
    } else {
      current_value = current_value.get().get(expanded_placeholder.first.name);
    }
  }

  for (int i = static_cast<int>(current_expanded_placeholder_and_index_.size());
       i < static_cast<int>(layers_to_expand->size());
       ++i) {
    const auto& ph = (*layers_to_expand)[i];
    XGRAMMAR_DCHECK(current_value.get().contains(ph.name))
        << "Placeholder name '" << ph.name
        << "' not found in values, which is required for the template: '"
        << FormatToJSON(format_template_to_expand) << "'" << "\n"
        << "Current value: " << current_value.get().serialize();
    current_value = current_value.get().get(ph.name);
    if (!ph.is_array) {
      AddExpandInfo(ph, kNotArray, added_layers_count);
      continue;
    }

    XGRAMMAR_DCHECK(current_value.get().is<picojson::array>())
        << "Placeholder name '" << ph.name
        << "' is expected to be an array, but found: " << current_value.get().serialize();
    for (int i = 0; i < static_cast<int>(current_value.get().get<picojson::array>().size()); i++) {
      AddExpandInfo(ph, i, added_layers_count);
      auto sub_result = VisitExpand(format_template_to_expand);
      if (sub_result.IsErr()) {
        return sub_result;
      }
      auto unwrapped_sub_result = std::move(sub_result).Unwrap();
      results.insert(results.end(), unwrapped_sub_result.begin(), unwrapped_sub_result.end());
      RemoveExpandInfo(added_layers_count);
    }
    break;  // Only expand one placeholder at a time.
  }
  return ResultOk(results);
}

Result<std::vector<Format>, StructuralTagError> StructuralTagTemplateFiller::VisitExpandSub(
    const OrFormat& format_template_to_expand, int* added_layers_count
) {
  // According to the definition of OrFormat, it is a disjunction of multiple formats.
  // Therefore, we do not need to expand it at the current stage. We only need to
  // expand the sub-formats.
  std::vector<Format> expanded_formats;
  for (const auto& element : format_template_to_expand.elements) {
    auto result = VisitExpand(element);
    if (result.IsErr()) {
      return result;
    }
    auto unwrapped_result = std::move(result).Unwrap();
    expanded_formats.insert(
        expanded_formats.end(), unwrapped_result.begin(), unwrapped_result.end()
    );
  }
  return ResultOk(expanded_formats);
}

Result<std::vector<Format>, StructuralTagError> StructuralTagTemplateFiller::VisitExpandSub(
    const TagFormat& format_template_to_expand, int* added_layers_count
) {
  // Step 1. Check if we need to expand it. We only need to check the layers of begin and end.
  // because if begin and end do not need to expand, then we can expand the content in the
  // sub-format.
  auto begin_placeholder = DetectTemplatePlaceholderNames(format_template_to_expand.begin);
  if (begin_placeholder.IsErr()) {
    return ResultErr(std::move(begin_placeholder).UnwrapErr());
  }
  auto unwrapped_begin_placeholder = std::move(begin_placeholder).Unwrap();

  auto end_placeholder = DetectTemplatePlaceholderNames(format_template_to_expand.end);
  if (end_placeholder.IsErr()) {
    return ResultErr(std::move(end_placeholder).UnwrapErr());
  }
  auto unwrapped_end_placeholder = std::move(end_placeholder).Unwrap();

  int largest_length = 0;
  const Layers* largest_layer = nullptr;
  for (const auto& layer : unwrapped_begin_placeholder) {
    auto rit = std::find_if(layer.rbegin(), layer.rend(), [](const PlaceHolderWithArray& layer) {
      return layer.is_array;
    });
    int index = static_cast<int>(std::distance(layer.begin(), rit.base())) - 1;
    int length = index + 1;
    if (length > largest_length) {
      largest_layer = &layer;
      largest_length = length;
    }
  }

  for (const auto& layer : unwrapped_end_placeholder) {
    auto rit = std::find_if(layer.rbegin(), layer.rend(), [](const PlaceHolderWithArray& layer) {
      return layer.is_array;
    });
    int index = static_cast<int>(std::distance(layer.begin(), rit.base())) - 1;
    int length = index + 1;
    if (length > largest_length) {
      largest_layer = &layer;
      largest_length = length;
    }
  }

  std::optional<Layers> layers_to_expand = std::nullopt;

  if (largest_layer != nullptr) {
    layers_to_expand = Layers(largest_layer->begin(), largest_layer->begin() + largest_length);
  }

  // Step 2. Compare the placeholders with the current expanded placeholders, and determine
  // whether we need to expand the placeholders.
  bool need_expand = layers_to_expand.has_value() &&
                     layers_to_expand->size() > current_expanded_placeholder_and_index_.size();

  // Step 3. If we do not need to expand, we will fill it at the current stage.
  if (!need_expand) {
    auto begin_str = ReplacePlaceHolder(format_template_to_expand.begin);
    if (begin_str.IsErr()) {
      return ResultErr(std::move(begin_str).UnwrapErr());
    }
    auto unwrapped_begin_str = std::move(begin_str).Unwrap();

    auto end_strs = ReplacePlaceHolders(format_template_to_expand.end);
    if (end_strs.IsErr()) {
      return ResultErr(std::move(end_strs).UnwrapErr());
    }
    auto unwrapped_end_str = std::move(end_strs).Unwrap();

    auto sub_result = VisitExpand(*format_template_to_expand.content);
    if (sub_result.IsErr()) {
      return sub_result;
    }
    auto unwrapped_sub_result = std::move(sub_result).Unwrap();
    if (unwrapped_sub_result.empty()) {
      return ResultOk(std::vector<Format>());
    }
    if (unwrapped_sub_result.size() == 1) {
      return ResultOk(std::vector<Format>{TagFormat(
          unwrapped_begin_str, std::make_shared<Format>(unwrapped_sub_result[0]), unwrapped_end_str
      )});
    }
    return ResultOk(std::vector<Format>{TagFormat(
        unwrapped_begin_str,
        std::make_shared<Format>(OrFormat(unwrapped_sub_result)),
        unwrapped_end_str
    )});
  }

  // Step 4. If we need to expand the placeholders, we will expand them and return the expanded
  // formats recursively.
  std::vector<Format> results;
  auto current_value = std::cref(*values_);
  for (const auto& expanded_placeholder : current_expanded_placeholder_and_index_) {
    XGRAMMAR_DCHECK(current_value.get().contains(expanded_placeholder.first.name));
    if (expanded_placeholder.first.is_array) {
      XGRAMMAR_DCHECK(current_value.get().get(expanded_placeholder.first.name).is<picojson::array>()
      );
      current_value = current_value.get()
                          .get(expanded_placeholder.first.name)
                          .get<picojson::array>()[expanded_placeholder.second];
    } else {
      current_value = current_value.get().get(expanded_placeholder.first.name);
    }
  }

  for (int i = static_cast<int>(current_expanded_placeholder_and_index_.size());
       i < static_cast<int>(layers_to_expand->size());
       ++i) {
    const auto& ph = (*layers_to_expand)[i];
    if (!ph.is_array) {
      AddExpandInfo(ph, kNotArray, added_layers_count);
      continue;
    }

    XGRAMMAR_DCHECK(current_value.get().contains(ph.name))
        << "Placeholder name '" << ph.name
        << "' not found in values, which is required for the template: '"
        << FormatToJSON(format_template_to_expand) << "'";
    current_value = current_value.get().get(ph.name);
    XGRAMMAR_DCHECK(current_value.get().is<picojson::array>())
        << "Placeholder name '" << ph.name
        << "' is expected to be an array, but found: " << current_value.get().serialize();
    for (int i = 0; i < static_cast<int>(current_value.get().get<picojson::array>().size()); i++) {
      AddExpandInfo(ph, i, added_layers_count);
      auto sub_result = VisitExpand(format_template_to_expand);
      if (sub_result.IsErr()) {
        return sub_result;
      }
      auto unwrapped_sub_result = std::move(sub_result).Unwrap();
      results.insert(results.end(), unwrapped_sub_result.begin(), unwrapped_sub_result.end());
      RemoveExpandInfo(added_layers_count);
    }
  }
  return ResultOk(results);
}

Result<std::vector<Format>, StructuralTagError> StructuralTagTemplateFiller::VisitExpandSub(
    const TriggeredTagsFormat& format_template_to_expand, int* added_layers_count
) {
  // According to the definition of TriggeredTagsFormat, it is a disjunction of multiple tags.
  // Therefore, we do not need to expand it at the current stage. We only need to
  // expand the sub-formats.
  std::vector<TagFormat> expanded_formats;
  for (const auto& tag : format_template_to_expand.tags) {
    auto result = VisitExpand(tag);
    if (result.IsErr()) {
      return result;
    }
    auto unwrapped_result = std::move(result).Unwrap();
    XGRAMMAR_DCHECK(std::all_of(
        unwrapped_result.begin(),
        unwrapped_result.end(),
        [](const Format& format) { return std::holds_alternative<TagFormat>(format); }
    ));
    for (const auto& tag_format : unwrapped_result) {
      expanded_formats.push_back(std::get<TagFormat>(tag_format));
    }
  }

  if (expanded_formats.empty()) {
    return ResultOk(std::vector<Format>{});
  }

  return ResultOk(std::vector<Format>{TriggeredTagsFormat(
      format_template_to_expand.triggers,
      expanded_formats,
      {},
      format_template_to_expand.at_least_one,
      format_template_to_expand.stop_after_first
  )});
}

Result<std::vector<Format>, StructuralTagError> StructuralTagTemplateFiller::VisitExpandSub(
    const TagsWithSeparatorFormat& format_template_to_expand, int* added_layers_count
) {
  // According to the definition of TagsWithSeparatorFormat, it is a disjunction of multiple tags.
  // Therefore, we do not need to expand it at the current stage. We only need to
  // expand the sub-formats.
  std::vector<TagFormat> expanded_formats;
  for (const auto& tag : format_template_to_expand.tags) {
    auto result = VisitExpand(tag);
    if (result.IsErr()) {
      return result;
    }
    auto unwrapped_result = std::move(result).Unwrap();
    for (const auto& tag_format : unwrapped_result) {
      expanded_formats.push_back(std::get<TagFormat>(tag_format));
    }
  }

  if (expanded_formats.empty()) {
    return ResultOk(std::vector<Format>{});
  }

  return ResultOk(std::vector<Format>{TagsWithSeparatorFormat(
      expanded_formats,
      format_template_to_expand.separator,
      format_template_to_expand.at_least_one,
      format_template_to_expand.stop_after_first
  )});
}

bool StructuralTagTemplateFiller::HasUnfilledPlaceholders(const StructuralTag& structural_tag) {
  auto result = Visit(structural_tag.format);
  if (result.IsErr()) {
    return false;
  }
  auto placeholder_names = std::move(result).Unwrap();
  return !placeholder_names.empty();
}

Result<StructuralTag, StructuralTagError> FillTemplateWithValues(
    const StructuralTag& template_structural_tag, const picojson::value& values
) {
  return StructuralTagTemplateFiller().Apply(template_structural_tag, values);
}

Result<StructuralTag, StructuralTagError> StructuralTagTemplateFiller::Apply(
    const StructuralTag& template_structural_tag, const picojson::value& values
) {
  values_ = &values;
  current_expanded_placeholder_and_index_.clear();
  format_to_placeholder_names_.clear();

  // Step 1. Collect all the placeholders in the template.
  auto result = Visit(template_structural_tag.format);
  if (result.IsErr()) {
    return ResultErr(std::move(result).UnwrapErr());
  }

  // Step 2. Expand the placeholders in the template.
  auto stag_result = VisitExpand(template_structural_tag.format);
  if (stag_result.IsErr()) {
    return ResultErr(std::move(stag_result).UnwrapErr());
  }
  auto expanded_formats = std::move(stag_result).Unwrap();

  // Step 3. Return the filled structural tag.
  if (expanded_formats.empty()) {
    // No constraints were applied.
    return ResultOk(StructuralTag(AnyTextFormat({})));
  }
  if (expanded_formats.size() == 1) {
    return ResultOk(StructuralTag(std::move(expanded_formats[0])));
  } else {
    return ResultOk(StructuralTag(OrFormat(std::move(expanded_formats))));
  }
}

/************** StructuralTag To JSON format **************/

picojson::value SequenceFormat::ToJSON() const {
  picojson::array elements_json;
  for (const auto& element : elements) {
    elements_json.push_back(FormatToJSON(element));
  }
  picojson::object obj;
  obj["type"] = picojson::value("sequence");
  obj["elements"] = picojson::value(elements_json);
  return picojson::value(obj);
}

picojson::value OrFormat::ToJSON() const {
  picojson::array elements_json;
  for (const auto& element : elements) {
    elements_json.push_back(FormatToJSON(element));
  }
  picojson::object obj;
  obj["type"] = picojson::value("or");
  obj["elements"] = picojson::value(elements_json);
  return picojson::value(obj);
}

picojson::value TagFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value("tag");
  obj["begin"] = picojson::value(begin);
  picojson::array end_array;
  for (const auto& end_part : end) {
    end_array.push_back(picojson::value(end_part));
  }
  obj["end"] = picojson::value(end_array);
  obj["content"] = FormatToJSON(*content);
  return picojson::value(obj);
}

picojson::value TriggeredTagsFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value("triggered_tags");
  picojson::array triggers_array;
  for (const auto& trigger : triggers) {
    triggers_array.push_back(picojson::value(trigger));
  }
  obj["triggers"] = picojson::value(triggers_array);
  picojson::array tags_array;
  for (const auto& tag : tags) {
    tags_array.push_back(FormatToJSON(tag));
  }
  obj["tags"] = picojson::value(tags_array);
  obj["at_least_one"] = picojson::value(at_least_one);
  obj["stop_after_first"] = picojson::value(stop_after_first);
  return picojson::value(obj);
}

picojson::value TagsWithSeparatorFormat::ToJSON() const {
  picojson::object obj;
  obj["type"] = picojson::value("tags_with_separator");
  picojson::array tags_array;
  for (const auto& tag : tags) {
    tags_array.push_back(FormatToJSON(tag));
  }
  obj["tags"] = picojson::value(tags_array);
  obj["separator"] = picojson::value(separator);
  obj["at_least_one"] = picojson::value(at_least_one);
  obj["stop_after_first"] = picojson::value(stop_after_first);
  return picojson::value(obj);
}

/************** StructuralTag Conversion Public API **************/

Result<Grammar, StructuralTagError> StructuralTagToGrammar(const std::string& structural_tag_json) {
  auto structural_tag_result = StructuralTagParser::FromJSON(structural_tag_json);
  if (structural_tag_result.IsErr()) {
    return ResultErr(std::move(structural_tag_result).UnwrapErr());
  }
  auto structural_tag = std::move(structural_tag_result).Unwrap();
  auto err = StructuralTagAnalyzer().Analyze(&structural_tag);
  if (err.has_value()) {
    return ResultErr(std::move(err).value());
  }
  auto result = StructuralTagGrammarConverter().Convert(structural_tag);
  if (result.IsErr()) {
    return ResultErr(std::move(result).UnwrapErr());
  }
  return ResultOk(GrammarNormalizer::Apply(std::move(result).Unwrap()));
}

Result<StructuralTag, StructuralTagError> FromTemplate(
    const std::string& structural_tag_template_json, const std::string& values_json_str
) {
  // Step 1. Parse the input values.
  picojson::value values;
  std::string value_err = picojson::parse(values, values_json_str);
  XGRAMMAR_CHECK(value_err.empty())
      << "Failed to input values: " << value_err << ". The JSON string is:" << values_json_str;

  // Step 2. Parse the template.
  auto structural_tag_result_raw = StructuralTagParser::FromJSON(structural_tag_template_json);
  if (structural_tag_result_raw.IsErr()) {
    return ResultErr(std::move(structural_tag_result_raw).UnwrapErr());
  }
  auto structural_tag_raw = std::move(structural_tag_result_raw).Unwrap();

  // Step 3. Replace the elements.
  return FillTemplateWithValues(structural_tag_raw, values);
}

}  // namespace xgrammar
