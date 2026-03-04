/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag.cc
 */
#include "structural_tag.h"

#include <picojson.h>
#include <xgrammar/exception.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "grammar_functor.h"
#include "grammar_impl.h"
#include "json_schema_converter.h"
#include "support/recursion_guard.h"
#include "support/utils.h"
#include "xgrammar/grammar.h"
#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

// Short alias for the error type.
using ISTError = InvalidStructuralTagError;

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
  Result<JSONSchemaFormat, ISTError> ParseJSONSchemaFormat(
      const picojson::object& value, std::optional<std::string> style_override = std::nullopt
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

  static Result<TokenFormat, ISTError> ParseTokenFormat(const picojson::value& value);
  static Result<std::vector<std::variant<std::string, TokenFormat>>, ISTError>
  ParseMixedStringTokenArray(const picojson::array& arr, const char* field_name);

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

Result<TokenFormat, ISTError> StructuralTagParser::ParseTokenFormat(const picojson::value& value) {
  if (!value.is<picojson::object>()) {
    return ResultErr<ISTError>("TokenFormat must be an object");
  }
  const auto& obj = value.get<picojson::object>();
  auto type_it = obj.find("type");
  if (type_it == obj.end() || !type_it->second.is<std::string>() ||
      type_it->second.get<std::string>() != "token") {
    return ResultErr<ISTError>("TokenFormat's type must be \"token\"");
  }
  auto token_it = obj.find("token");
  if (token_it == obj.end()) {
    return ResultErr<ISTError>("TokenFormat must have a token field");
  }
  if (token_it->second.is<double>()) {
    double val = token_it->second.get<double>();
    if (val != static_cast<int32_t>(val) || val < 0) {
      return ResultErr<ISTError>("TokenFormat's token must be a non-negative integer or a string");
    }
    return ResultOk<TokenFormat>(static_cast<int32_t>(val));
  } else if (token_it->second.is<std::string>()) {
    return ResultOk<TokenFormat>(token_it->second.get<std::string>());
  }
  return ResultErr<ISTError>("TokenFormat's token must be an integer or a string");
}

Result<std::vector<std::variant<std::string, TokenFormat>>, ISTError>
StructuralTagParser::ParseMixedStringTokenArray(
    const picojson::array& arr, const char* field_name
) {
  std::vector<std::variant<std::string, TokenFormat>> result;
  result.reserve(arr.size());
  for (const auto& item : arr) {
    if (item.is<std::string>()) {
      result.push_back(item.get<std::string>());
    } else if (item.is<picojson::object>()) {
      auto tf = ParseTokenFormat(item);
      if (tf.IsErr()) {
        return ResultErr<ISTError>(std::move(tf).UnwrapErr());
      }
      result.push_back(std::move(tf).Unwrap());
    } else {
      return ResultErr<ISTError>(
          std::string(field_name) + " array elements must be strings or TokenFormat objects"
      );
    }
  }
  return ResultOk(std::move(result));
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
      return Result<Format, ISTError>::Convert(ParseJSONSchemaFormat(obj, "qwen_xml"));
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
  if (value_it == obj.end() || !value_it->second.is<std::string>()) {
    return ResultErr<ISTError>("ConstString format must have a value field with a string");
  }
  return ResultOk<ConstStringFormat>(value_it->second.get<std::string>());
}

Result<JSONSchemaFormat, ISTError> StructuralTagParser::ParseJSONSchemaFormat(
    const picojson::object& obj, std::optional<std::string> style_override
) {
  // json_schema is required.
  auto json_schema_it = obj.find("json_schema");
  if (json_schema_it == obj.end() ||
      !(json_schema_it->second.is<picojson::object>() || json_schema_it->second.is<bool>())) {
    return ResultErr<ISTError>(
        "JSON schema format must have a json_schema field with a object or boolean value"
    );
  }
  std::string style = "json";
  if (style_override.has_value()) {
    style = *style_override;
  } else {
    auto it = obj.find("style");
    if (it != obj.end() && it->second.is<std::string>()) {
      style = it->second.get<std::string>();
      if (style != "json" && style != "qwen_xml" && style != "minimax_xml" &&
          style != "deepseek_xml") {
        return ResultErr<ISTError>(
            "style must be \"json\", \"qwen_xml\", \"minimax_xml\", or \"deepseek_xml\""
        );
      }
    }
  }
  // here introduces a serialization/deserialization overhead; try to avoid it in the future.
  return ResultOk<JSONSchemaFormat>(json_schema_it->second.serialize(false), style);
}

Result<AnyTextFormat, ISTError> StructuralTagParser::ParseAnyTextFormat(const picojson::object& obj
) {
  auto excluded_strs_it = obj.find("excludes");
  if (excluded_strs_it == obj.end()) {
    if ((obj.find("type") == obj.end())) {
      return ResultErr<ISTError>("Any text format should not have any fields other than type");
    }
    return ResultOk<AnyTextFormat>(std::vector<std::variant<std::string, TokenFormat>>{});
  }
  if (!excluded_strs_it->second.is<picojson::array>()) {
    return ResultErr<ISTError>("AnyText format's excludes field must be an array");
  }
  auto excludes =
      ParseMixedStringTokenArray(excluded_strs_it->second.get<picojson::array>(), "excludes");
  if (excludes.IsErr()) {
    return ResultErr<ISTError>(std::move(excludes).UnwrapErr());
  }
  return ResultOk<AnyTextFormat>(std::move(excludes).Unwrap());
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
  // begin is required - can be string or TokenFormat
  auto begin_it = obj.find("begin");
  if (begin_it == obj.end()) {
    return ResultErr<ISTError>("Tag format's begin field must be a string or TokenFormat");
  }
  std::variant<std::string, TokenFormat> begin;
  if (begin_it->second.is<std::string>()) {
    begin = begin_it->second.get<std::string>();
  } else if (begin_it->second.is<picojson::object>()) {
    auto tf = ParseTokenFormat(begin_it->second);
    if (tf.IsErr()) {
      return ResultErr<ISTError>(std::move(tf).UnwrapErr());
    }
    begin = std::move(tf).Unwrap();
  } else {
    return ResultErr<ISTError>("Tag format's begin field must be a string or TokenFormat");
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

  // end is required - can be string, array of strings, or TokenFormat
  auto end_it = obj.find("end");
  if (end_it == obj.end()) {
    return ResultErr<ISTError>("Tag format must have an end field");
  }

  std::variant<std::vector<std::string>, TokenFormat> end;
  if (end_it->second.is<std::string>()) {
    end = std::vector<std::string>{end_it->second.get<std::string>()};
  } else if (end_it->second.is<picojson::array>()) {
    const auto& end_array = end_it->second.get<picojson::array>();
    if (end_array.empty()) {
      return ResultErr<ISTError>("Tag format's end array cannot be empty");
    }
    std::vector<std::string> end_strings;
    for (const auto& item : end_array) {
      if (!item.is<std::string>()) {
        return ResultErr<ISTError>("Tag format's end array must contain only strings");
      }
      end_strings.push_back(item.get<std::string>());
    }
    end = std::move(end_strings);
  } else if (end_it->second.is<picojson::object>()) {
    auto tf = ParseTokenFormat(end_it->second);
    if (tf.IsErr()) {
      return ResultErr<ISTError>(std::move(tf).UnwrapErr());
    }
    end = std::move(tf).Unwrap();
  } else {
    return ResultErr<ISTError>(
        "Tag format's end field must be a string, array of strings, or TokenFormat"
    );
  }

  return ResultOk<TagFormat>(
      std::move(begin), std::make_shared<Format>(std::move(content).Unwrap()), std::move(end)
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
  auto triggers =
      ParseMixedStringTokenArray(triggers_it->second.get<picojson::array>(), "triggers");
  if (triggers.IsErr()) {
    return ResultErr<ISTError>(std::move(triggers).UnwrapErr());
  }
  auto triggers_vec = std::move(triggers).Unwrap();
  if (triggers_vec.empty()) {
    return ResultErr<ISTError>("Triggered tags format's triggers must be non-empty");
  }
  for (const auto& t : triggers_vec) {
    if (std::holds_alternative<std::string>(t) && std::get<std::string>(t).empty()) {
      return ResultErr<ISTError>("Triggered tags format's string triggers must be non-empty");
    }
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
  if (tags.empty()) {
    return ResultErr<ISTError>("Triggered tags format's tags must be non-empty");
  }

  // excludes is optional.
  std::vector<std::variant<std::string, TokenFormat>> excludes_vec;
  auto excludes_it = obj.find("excludes");
  if (excludes_it != obj.end()) {
    if (!excludes_it->second.is<picojson::array>()) {
      return ResultErr<ISTError>("Triggered tags format's excludes field must be an array");
    }
    auto excludes =
        ParseMixedStringTokenArray(excludes_it->second.get<picojson::array>(), "excludes");
    if (excludes.IsErr()) {
      return ResultErr<ISTError>(std::move(excludes).UnwrapErr());
    }
    excludes_vec = std::move(excludes).Unwrap();
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
      std::move(triggers_vec),
      std::move(tags),
      std::move(excludes_vec),
      at_least_one,
      stop_after_first
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

/************** StructuralTag Token Resolver **************/

class StructuralTagTokenResolver {
 public:
  static std::optional<ISTError> Resolve(
      StructuralTag* structural_tag, const TokenizerInfo* tokenizer_info
  );

 private:
  const TokenizerInfo* tokenizer_info_;

  explicit StructuralTagTokenResolver(const TokenizerInfo* ti) : tokenizer_info_(ti) {}

  std::optional<ISTError> Visit(Format* format);

  Result<int32_t, ISTError> ResolveToken(const TokenFormat& tf);

  static void CollectTokenFormats(
      const std::vector<std::variant<std::string, TokenFormat>>& items,
      std::vector<int32_t>* out_ids
  );
};

std::optional<ISTError> StructuralTagTokenResolver::Resolve(
    StructuralTag* structural_tag, const TokenizerInfo* tokenizer_info
) {
  return StructuralTagTokenResolver(tokenizer_info).Visit(&structural_tag->format);
}

Result<int32_t, ISTError> StructuralTagTokenResolver::ResolveToken(const TokenFormat& tf) {
  if (std::holds_alternative<int32_t>(tf.token)) {
    return ResultOk(std::get<int32_t>(tf.token));
  }
  if (tokenizer_info_ == nullptr) {
    return ResultErr<ISTError>(
        "TokenFormat with string token requires tokenizer_info, but none was provided"
    );
  }
  const auto& token_str = std::get<std::string>(tf.token);
  const auto& decoded_vocab = tokenizer_info_->GetDecodedVocab();
  for (int32_t i = 0; i < static_cast<int32_t>(decoded_vocab.size()); ++i) {
    if (decoded_vocab[i] == token_str) {
      return ResultOk(i);
    }
  }
  return ResultErr<ISTError>("Token string \"" + token_str + "\" not found in vocabulary");
}

void StructuralTagTokenResolver::CollectTokenFormats(
    const std::vector<std::variant<std::string, TokenFormat>>& items, std::vector<int32_t>* out_ids
) {
  for (const auto& item : items) {
    if (std::holds_alternative<TokenFormat>(item)) {
      const auto& tf = std::get<TokenFormat>(item);
      if (std::holds_alternative<int32_t>(tf.token)) {
        out_ids->push_back(std::get<int32_t>(tf.token));
      }
    }
  }
}

std::optional<ISTError> StructuralTagTokenResolver::Visit(Format* format) {
  return std::visit(
      [&](auto& arg) -> std::optional<ISTError> {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, AnyTextFormat>) {
          for (const auto& item : arg.excludes) {
            if (std::holds_alternative<TokenFormat>(item)) {
              auto id = ResolveToken(std::get<TokenFormat>(item));
              if (id.IsErr()) return std::move(id).UnwrapErr();
              arg.resolved_exclude_token_ids_.push_back(std::move(id).Unwrap());
            }
          }
        } else if constexpr (std::is_same_v<T, TriggeredTagsFormat>) {
          for (const auto& item : arg.triggers) {
            if (std::holds_alternative<TokenFormat>(item)) {
              auto id = ResolveToken(std::get<TokenFormat>(item));
              if (id.IsErr()) return std::move(id).UnwrapErr();
              arg.resolved_token_trigger_ids_.push_back(std::move(id).Unwrap());
            }
          }
          for (const auto& item : arg.excludes) {
            if (std::holds_alternative<TokenFormat>(item)) {
              auto id = ResolveToken(std::get<TokenFormat>(item));
              if (id.IsErr()) return std::move(id).UnwrapErr();
              arg.resolved_exclude_token_ids_.push_back(std::move(id).Unwrap());
            }
          }
          for (auto& tag : arg.tags) {
            auto err = Visit(tag.content.get());
            if (err.has_value()) return err;
          }
        } else if constexpr (std::is_same_v<T, TagFormat>) {
          if (std::holds_alternative<TokenFormat>(arg.begin)) {
            auto id = ResolveToken(std::get<TokenFormat>(arg.begin));
            if (id.IsErr()) return std::move(id).UnwrapErr();
            arg.begin = TokenFormat(std::move(id).Unwrap());
          }
          if (std::holds_alternative<TokenFormat>(arg.end)) {
            auto id = ResolveToken(std::get<TokenFormat>(arg.end));
            if (id.IsErr()) return std::move(id).UnwrapErr();
            arg.end = TokenFormat(std::move(id).Unwrap());
          }
          auto err = Visit(arg.content.get());
          if (err.has_value()) return err;
        } else if constexpr (std::is_same_v<T, SequenceFormat>) {
          for (auto& elem : arg.elements) {
            auto err = Visit(&elem);
            if (err.has_value()) return err;
          }
        } else if constexpr (std::is_same_v<T, OrFormat>) {
          for (auto& elem : arg.elements) {
            auto err = Visit(&elem);
            if (err.has_value()) return err;
          }
        } else if constexpr (std::is_same_v<T, TagsWithSeparatorFormat>) {
          for (auto& tag : arg.tags) {
            auto err = Visit(tag.content.get());
            if (err.has_value()) return err;
          }
        }
        return std::nullopt;
      },
      *format
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
  std::optional<ISTError> VisitSub(AnyTextFormat* format);
  std::optional<ISTError> VisitSub(GrammarFormat* format);
  std::optional<ISTError> VisitSub(RegexFormat* format);
  std::optional<ISTError> VisitSub(SequenceFormat* format);
  std::optional<ISTError> VisitSub(OrFormat* format);
  std::optional<ISTError> VisitSub(TagFormat* format);
  std::optional<ISTError> VisitSub(TriggeredTagsFormat* format);
  std::optional<ISTError> VisitSub(TagsWithSeparatorFormat* format);

  std::vector<std::string> DetectEndStrings();
  std::vector<int32_t> DetectEndTokenIds();
  bool IsUnlimited(const Format& format);
  bool IsExcluded(const Format& format);

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
      if (std::holds_alternative<std::vector<std::string>>(tag->end)) {
        return std::get<std::vector<std::string>>(tag->end);
      }
      return {};
    }
  }
  return {};
}

std::vector<int32_t> StructuralTagAnalyzer::DetectEndTokenIds() {
  for (int i = static_cast<int>(stack_.size()) - 1; i >= 0; --i) {
    auto& format = stack_[i];
    if (std::holds_alternative<TagFormat*>(format)) {
      auto* tag = std::get<TagFormat*>(format);
      if (std::holds_alternative<TokenFormat>(tag->end)) {
        const auto& tf = std::get<TokenFormat>(tag->end);
        if (std::holds_alternative<int32_t>(tf.token)) {
          return {std::get<int32_t>(tf.token)};
        }
      }
      return {};
    }
  }
  return {};
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

bool StructuralTagAnalyzer::IsExcluded(const Format& format) {
  return std::visit(
      [&](auto&& arg) -> bool {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, AnyTextFormat>) {
          return !arg.excludes.empty();
        } else if constexpr (std::is_same_v<T, TriggeredTagsFormat>) {
          return !arg.excludes.empty();
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

std::optional<ISTError> StructuralTagAnalyzer::VisitSub(AnyTextFormat* format) {
  format->detected_end_strs_ = DetectEndStrings();
  format->detected_end_token_ids_ = DetectEndTokenIds();
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
      if (!IsExcluded(element)) {
        return ISTError(
            "Only the last element in a sequence can be unlimited, but the " + std::to_string(i) +
            "th element of sequence format is unlimited"
        );
      }
    }
  }

  auto& element = format->elements.back();
  auto err = Visit(&element);
  if (err.has_value()) {
    return err;
  }
  format->is_unlimited_ = IsUnlimited(element) && !IsExcluded(element);
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
    auto is_unlimited = IsUnlimited(element) && !IsExcluded(element);
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
    if (std::holds_alternative<TokenFormat>(format->end)) {
      // Token end: always non-empty, clear it (propagated via detected_end_token_ids_)
      format->end = std::vector<std::string>{};
    } else {
      auto& end_strs = std::get<std::vector<std::string>>(format->end);
      bool has_non_empty = false;
      for (const auto& end_str : end_strs) {
        if (!end_str.empty()) {
          has_non_empty = true;
          break;
        }
      }
      if (!has_non_empty) {
        if (IsExcluded(*format->content)) {
          return std::nullopt;
        } else {
          return ISTError("When the content is unlimited, at least one end string must be non-empty"
          );
        }
      }
      end_strs.clear();
    }
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
  format->detected_end_token_ids_ = DetectEndTokenIds();
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
  format->detected_end_token_ids_ = DetectEndTokenIds();
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

  int32_t BuildTagEndExpr(const TagFormat& tag);
  int32_t BuildFullTagExpr(const TagFormat& tag, int tag_content_rule_id);

  GrammarBuilder grammar_builder_;
};

bool StructuralTagGrammarConverter::IsPrefix(
    const std::string& prefix, const std::string& full_str
) {
  return prefix.size() <= full_str.size() &&
         std::string_view(full_str).substr(0, prefix.size()) == prefix;
}

int32_t StructuralTagGrammarConverter::BuildTagEndExpr(const TagFormat& tag) {
  if (std::holds_alternative<TokenFormat>(tag.end)) {
    const auto& tf = std::get<TokenFormat>(tag.end);
    return grammar_builder_.AddTokenSet({std::get<int32_t>(tf.token)});
  }
  const auto& end_strs = std::get<std::vector<std::string>>(tag.end);
  if (end_strs.empty()) {
    return -1;
  } else if (end_strs.size() == 1) {
    return end_strs[0].empty() ? grammar_builder_.AddEmptyStr()
                               : grammar_builder_.AddByteString(end_strs[0]);
  } else {
    std::vector<int> end_sequence_ids;
    for (const auto& end_str : end_strs) {
      auto end_expr = end_str.empty() ? grammar_builder_.AddEmptyStr()
                                      : grammar_builder_.AddByteString(end_str);
      end_sequence_ids.push_back(grammar_builder_.AddSequence({end_expr}));
    }
    auto end_choices_expr = grammar_builder_.AddChoices(end_sequence_ids);
    auto end_choices_rule_id = grammar_builder_.AddRuleWithHint("tag_end", end_choices_expr);
    return grammar_builder_.AddRuleRef(end_choices_rule_id);
  }
}

int32_t StructuralTagGrammarConverter::BuildFullTagExpr(
    const TagFormat& tag, int tag_content_rule_id
) {
  int begin_expr;
  if (std::holds_alternative<std::string>(tag.begin)) {
    begin_expr = grammar_builder_.AddByteString(std::get<std::string>(tag.begin));
  } else {
    const auto& tf = std::get<TokenFormat>(tag.begin);
    begin_expr = grammar_builder_.AddTokenSet({std::get<int32_t>(tf.token)});
  }
  auto rule_ref_expr = grammar_builder_.AddRuleRef(tag_content_rule_id);
  auto end_expr = BuildTagEndExpr(tag);
  if (end_expr == -1) {
    return grammar_builder_.AddSequence({begin_expr, rule_ref_expr});
  }
  return grammar_builder_.AddSequence({begin_expr, rule_ref_expr, end_expr});
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
  auto expr = format.value.empty() ? grammar_builder_.AddEmptyStr()
                                   : grammar_builder_.AddByteString(format.value);
  auto sequence_expr = grammar_builder_.AddSequence({expr});
  auto choices_expr = grammar_builder_.AddChoices({sequence_expr});
  return ResultOk(grammar_builder_.AddRuleWithHint("const_string", choices_expr));
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const JSONSchemaFormat& format) {
  const static std::unordered_map<std::string, std::function<std::string(const std::string&)>>
      style_to_grammar_converter = {
          {"json",
           [&](const std::string& json_schema) -> std::string {
             return JSONSchemaToEBNF(json_schema);
           }},
          {"qwen_xml",
           [&](const std::string& json_schema) -> std::string {
             return QwenXMLToolCallingToEBNF(json_schema);
           }},
          {"minimax_xml",
           [&](const std::string& json_schema) -> std::string {
             return MiniMaxXMLToolCallingToEBNF(json_schema);
           }},
          {"deepseek_xml",
           [&](const std::string& json_schema) -> std::string {
             return DeepSeekXMLToolCallingToEBNF(json_schema);
           }},
      };
  auto converter = style_to_grammar_converter.find(format.style);
  if (converter == style_to_grammar_converter.end()) {
    return ResultErr<ISTError>("Unsupported parsing type: " + format.style);
  }
  auto sub_grammar = Grammar::FromEBNF(converter->second(format.json_schema));
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
  // Build stops from detected end strings and token IDs
  std::vector<std::variant<std::string, int32_t>> stops_v;
  for (const auto& s : format.detected_end_strs_) {
    if (!s.empty()) {
      stops_v.push_back(s);
    }
  }
  for (int32_t tid : format.detected_end_token_ids_) {
    stops_v.push_back(tid);
  }

  // Build excludes from string excludes and resolved token excludes
  std::vector<std::variant<std::string, int32_t>> excludes_v;
  for (const auto& item : format.excludes) {
    if (std::holds_alternative<std::string>(item)) {
      excludes_v.push_back(std::get<std::string>(item));
    }
  }
  for (int32_t tid : format.resolved_exclude_token_ids_) {
    excludes_v.push_back(tid);
  }

  bool has_stops = !stops_v.empty();
  bool has_excludes = !excludes_v.empty();

  if (has_stops) {
    auto tag_dispatch_expr = grammar_builder_.AddTagDispatch(
        Grammar::Impl::TagDispatch{{}, false, stops_v, false, excludes_v}
    );
    return ResultOk(grammar_builder_.AddRuleWithHint("any_text", tag_dispatch_expr));
  } else if (has_excludes) {
    auto tag_dispatch_expr =
        grammar_builder_.AddTagDispatch(Grammar::Impl::TagDispatch{{}, true, {}, false, excludes_v}
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

  // Build begin expression
  int begin_expr;
  if (std::holds_alternative<std::string>(format.begin)) {
    begin_expr = grammar_builder_.AddByteString(std::get<std::string>(format.begin));
  } else {
    const auto& tf = std::get<TokenFormat>(format.begin);
    int32_t tid = std::get<int32_t>(tf.token);
    begin_expr = grammar_builder_.AddTokenSet({tid});
  }

  auto rule_ref_expr = grammar_builder_.AddRuleRef(sub_rule_id);

  // Build end expression
  if (std::holds_alternative<TokenFormat>(format.end)) {
    const auto& tf = std::get<TokenFormat>(format.end);
    int32_t tid = std::get<int32_t>(tf.token);
    auto end_expr = grammar_builder_.AddTokenSet({tid});
    auto sequence_expr_id = grammar_builder_.AddSequence({begin_expr, rule_ref_expr, end_expr});
    auto choices_expr = grammar_builder_.AddChoices({sequence_expr_id});
    return ResultOk(grammar_builder_.AddRuleWithHint("tag", choices_expr));
  }

  const auto& end_strs = std::get<std::vector<std::string>>(format.end);
  if (end_strs.size() > 1) {
    std::vector<int> end_sequence_ids;
    for (const auto& end_str : end_strs) {
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
  } else if (end_strs.size() == 1) {
    auto end_expr = end_strs[0].empty() ? grammar_builder_.AddEmptyStr()
                                        : grammar_builder_.AddByteString(end_strs[0]);
    auto sequence_expr_id = grammar_builder_.AddSequence({begin_expr, rule_ref_expr, end_expr});
    auto choices_expr = grammar_builder_.AddChoices({sequence_expr_id});
    return ResultOk(grammar_builder_.AddRuleWithHint("tag", choices_expr));
  } else {
    auto sequence_expr_id = grammar_builder_.AddSequence({begin_expr, rule_ref_expr});
    auto choices_expr = grammar_builder_.AddChoices({sequence_expr_id});
    return ResultOk(grammar_builder_.AddRuleWithHint("tag", choices_expr));
  }
}

Result<int, ISTError> StructuralTagGrammarConverter::VisitSub(const TriggeredTagsFormat& format) {
  // Step 1. Visit all tags and add to grammar. Match triggers to tags.
  std::vector<std::vector<int>> trigger_to_tag_ids(format.triggers.size());
  std::vector<int> tag_content_rule_ids;
  tag_content_rule_ids.reserve(format.tags.size());

  for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
    const auto& tag = format.tags[it_tag];
    int matched_trigger_id = -1;
    for (int it_trigger = 0; it_trigger < static_cast<int>(format.triggers.size()); ++it_trigger) {
      const auto& trigger = format.triggers[it_trigger];
      bool matched = false;
      if (std::holds_alternative<std::string>(trigger) &&
          std::holds_alternative<std::string>(tag.begin)) {
        matched = IsPrefix(std::get<std::string>(trigger), std::get<std::string>(tag.begin));
      } else if (std::holds_alternative<TokenFormat>(trigger) &&
                 std::holds_alternative<TokenFormat>(tag.begin)) {
        const auto& ttf = std::get<TokenFormat>(trigger);
        const auto& btf = std::get<TokenFormat>(tag.begin);
        if (std::holds_alternative<int32_t>(ttf.token) &&
            std::holds_alternative<int32_t>(btf.token)) {
          matched = (std::get<int32_t>(ttf.token) == std::get<int32_t>(btf.token));
        }
      }
      if (matched) {
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

    auto result = Visit(*tag.content);
    if (result.IsErr()) {
      return result;
    }
    tag_content_rule_ids.push_back(std::move(result).Unwrap());
  }

  // Step 2. Special Case: at_least_one && stop_after_first.
  if (format.at_least_one && format.stop_after_first) {
    std::vector<int> choice_elements;
    for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
      choice_elements.push_back(BuildFullTagExpr(format.tags[it_tag], tag_content_rule_ids[it_tag])
      );
    }
    auto choice_expr_id = grammar_builder_.AddChoices(choice_elements);

    if (!format.detected_end_strs_.empty() || !format.detected_end_token_ids_.empty()) {
      auto sub_rule_id = grammar_builder_.AddRuleWithHint("triggered_tags_sub", choice_expr_id);
      auto ref_sub_rule_expr_id = grammar_builder_.AddRuleRef(sub_rule_id);
      std::vector<int> end_choices;
      for (const auto& end_str : format.detected_end_strs_) {
        if (!end_str.empty()) {
          end_choices.push_back(
              grammar_builder_.AddSequence({grammar_builder_.AddByteString(end_str)})
          );
        }
      }
      for (int32_t tid : format.detected_end_token_ids_) {
        end_choices.push_back(grammar_builder_.AddSequence({grammar_builder_.AddTokenSet({tid})}));
      }
      if (end_choices.size() == 1) {
        // Unwrap single-element choices
        auto sequence_expr_id =
            grammar_builder_.AddSequence({ref_sub_rule_expr_id, end_choices[0]});
        choice_expr_id = grammar_builder_.AddChoices({sequence_expr_id});
      } else {
        auto end_choices_expr = grammar_builder_.AddChoices(end_choices);
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

  // Step 3. Normal Case: mixture of text and triggered tags via TagDispatch.

  // Step 3.1 Build trigger_rule_pairs
  std::vector<std::pair<std::variant<std::string, int32_t>, int32_t>> trigger_rule_pairs;
  for (int it_trigger = 0; it_trigger < static_cast<int>(format.triggers.size()); ++it_trigger) {
    const auto& trigger = format.triggers[it_trigger];
    std::vector<int> choice_elements;
    for (const auto& tag_id : trigger_to_tag_ids[it_trigger]) {
      const auto& tag = format.tags[tag_id];
      int rule_ref_expr_id = grammar_builder_.AddRuleRef(tag_content_rule_ids[tag_id]);

      // Build tag body: begin_suffix + content + end
      std::vector<int> body_parts;
      if (std::holds_alternative<std::string>(trigger)) {
        const auto& trigger_str = std::get<std::string>(trigger);
        const auto& begin_str = std::get<std::string>(tag.begin);
        body_parts.push_back(grammar_builder_.AddByteString(begin_str.substr(trigger_str.size())));
      }
      body_parts.push_back(rule_ref_expr_id);
      auto end_expr = BuildTagEndExpr(tag);
      if (end_expr != -1) {
        body_parts.push_back(end_expr);
      }
      choice_elements.push_back(grammar_builder_.AddSequence(body_parts));
    }
    auto choice_expr_id = grammar_builder_.AddChoices(choice_elements);
    auto sub_rule_id = grammar_builder_.AddRuleWithHint("triggered_tags_group", choice_expr_id);

    if (std::holds_alternative<std::string>(trigger)) {
      trigger_rule_pairs.push_back({std::get<std::string>(trigger), sub_rule_id});
    } else {
      const auto& tf = std::get<TokenFormat>(trigger);
      trigger_rule_pairs.push_back({std::get<int32_t>(tf.token), sub_rule_id});
    }
  }

  // Step 3.2 Build stops and excludes for TagDispatch
  std::vector<std::variant<std::string, int32_t>> stops_v;
  for (const auto& s : format.detected_end_strs_) {
    if (!s.empty()) {
      stops_v.push_back(s);
    }
  }
  for (int32_t tid : format.detected_end_token_ids_) {
    stops_v.push_back(tid);
  }

  std::vector<std::variant<std::string, int32_t>> excludes_v;
  for (const auto& item : format.excludes) {
    if (std::holds_alternative<std::string>(item)) {
      excludes_v.push_back(std::get<std::string>(item));
    }
  }
  for (int32_t tid : format.resolved_exclude_token_ids_) {
    excludes_v.push_back(tid);
  }

  bool loop_after_dispatch = !format.stop_after_first;
  int32_t rule_expr_id;
  if (!stops_v.empty()) {
    rule_expr_id = grammar_builder_.AddTagDispatch(Grammar::Impl::TagDispatch{
        trigger_rule_pairs, false, stops_v, loop_after_dispatch, excludes_v
    });
  } else {
    rule_expr_id = grammar_builder_.AddTagDispatch(
        Grammar::Impl::TagDispatch{trigger_rule_pairs, true, {}, loop_after_dispatch, excludes_v}
    );
  }

  // Step 3.3 Consider at_least_one
  if (format.at_least_one) {
    std::vector<int> first_choice_elements;
    for (int it_tag = 0; it_tag < static_cast<int>(format.tags.size()); ++it_tag) {
      first_choice_elements.push_back(
          BuildFullTagExpr(format.tags[it_tag], tag_content_rule_ids[it_tag])
      );
    }
    auto first_choice_expr_id = grammar_builder_.AddChoices(first_choice_elements);
    auto first_rule_id =
        grammar_builder_.AddRuleWithHint("triggered_tags_first", first_choice_expr_id);

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

/************** StructuralTag Conversion Public API **************/

Result<Grammar, StructuralTagError> StructuralTagToGrammar(
    const std::string& structural_tag_json, const TokenizerInfo* tokenizer_info
) {
  auto structural_tag_result = StructuralTagParser::FromJSON(structural_tag_json);
  if (structural_tag_result.IsErr()) {
    return ResultErr(std::move(structural_tag_result).UnwrapErr());
  }
  auto structural_tag = std::move(structural_tag_result).Unwrap();

  auto resolve_err = StructuralTagTokenResolver::Resolve(&structural_tag, tokenizer_info);
  if (resolve_err.has_value()) {
    return ResultErr(std::move(resolve_err).value());
  }

  auto err = StructuralTagAnalyzer().Analyze(&structural_tag);
  if (err.has_value()) {
    return ResultErr(std::move(err).value());
  }
  auto result = StructuralTagGrammarConverter().Convert(structural_tag);
  if (result.IsErr()) {
    return ResultErr(std::move(result).UnwrapErr());
  }
  auto unwrapped_result = std::move(result).Unwrap();
  return ResultOk(GrammarNormalizer::Apply(std::move(unwrapped_result)));
}

}  // namespace xgrammar
