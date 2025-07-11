/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag.cc
 */
#include <picojson.h>
#include <xgrammar/structural_tag.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>

#include "grammar_functor.h"
#include "structural_tag_impl.h"
#include "support/logging.h"
#include "support/recursion_guard.h"

namespace xgrammar {

/************** StructuralTag Parser **************/

class StructuralTagImpl {
 public:
  Result<StructuralTag> FromJSON(const std::string& json);

 private:
  Result<StructuralTag> ParseStructuralTag(const picojson::value& value);
  /*!
   * \brief Parse a Format object from a JSON value.
   * \param value The JSON value to parse.
   * \return A Format object if the JSON is valid, otherwise an error message in std::runtime_error.
   * \note The "type" field is checked in this function, and not checked in the Parse*Format
   * functions.
   */
  Result<Format> ParseFormat(const picojson::value& value);
  Result<LiteralFormat> ParseLiteralFormat(const picojson::object& value);
  Result<JSONSchemaFormat> ParseJSONSchemaFormat(const picojson::object& value);
  Result<WildcardTextFormat> ParseWildcardTextFormat(const picojson::object& value);
  Result<SequenceFormat> ParseSequenceFormat(const picojson::object& value);
  /*! \brief ParseTagFormat with extra check for object and the type field. */
  Result<TagFormat> ParseTagFormat(const picojson::value& value);
  Result<TagFormat> ParseTagFormat(const picojson::object& value);
  Result<TriggeredTagsFormat> ParseTriggeredTagsFormat(const picojson::object& value);
  Result<TagsWithSeparatorFormat> ParseTagsWithSeparatorFormat(const picojson::object& value);

  int parse_format_recursion_depth_ = 0;
};

Result<StructuralTag> StructuralTagImpl::FromJSON(const std::string& json) {
  picojson::value value;
  std::string err = picojson::parse(value, json);
  if (!err.empty()) {
    return ResultErr("Failed to parse JSON: " + err);
  }
  return ParseStructuralTag(value);
}

Result<StructuralTag> StructuralTagImpl::ParseStructuralTag(const picojson::value& value) {
  if (!value.is<picojson::object>()) {
    return ResultErr("Structural tag must be an object");
  }
  const auto& obj = value.get<picojson::object>();
  // The type field is optional but must be "structural_tag" if present.
  if (obj.find("type") != obj.end()) {
    if (!obj["type"].is<std::string>() || obj["type"].get<std::string>() != "structural_tag") {
      return ResultErr("Structural tag's type must be a string \"structural_tag\"");
    }
  }
  // The format field is required.
  if (obj.find("format") == obj.end()) {
    return ResultErr("Structural tag must have a format field");
  }
  auto format = ParseFormat(obj["format"]);
  if (format.IsErr()) {
    return ResultErr(std::move(format).UnwrapErr());
  }
  return ResultOk(StructuralTag{std::move(format).Unwrap()});
}

Result<Format> StructuralTagImpl::ParseFormat(const picojson::value& value) {
  RecursionGuard guard(&parse_format_recursion_depth_);
  if (!value.is<picojson::object>()) {
    return ResultErr("Format must be an object");
  }
  const auto& obj = value.get<picojson::object>();
  // If type is present, use it to determine the format.
  if (obj.find("type") != obj.end()) {
    if (!obj["type"].is<std::string>()) {
      return ResultErr("Format's type must be a string");
    }
    auto type = obj["type"].get<std::string>();
    if (type == "literal") {
      return Result<Format>::Convert(ParseLiteralFormat(obj));
    } else if (type == "json_schema") {
      return Result<Format>::Convert(ParseJSONSchemaFormat(obj));
    } else if (type == "wildcard_text") {
      return Result<Format>::Convert(ParseWildcardTextFormat(obj));
    } else if (type == "sequence") {
      return Result<Format>::Convert(ParseSequenceFormat(obj));
    } else if (type == "tag") {
      return Result<Format>::Convert(ParseTagFormat(obj));
    } else if (type == "triggered_tags") {
      return Result<Format>::Convert(ParseTriggeredTagsFormat(obj));
    } else if (type == "tags_with_separator") {
      return Result<Format>::Convert(ParseTagsWithSeparatorFormat(obj));
    } else {
      return ResultErr("Invalid format type: " + type);
    }
  }
  // If type is not present, try every format type one by one. Tag is prioritized.
  auto tag_format = ParseTagFormat(obj);
  if (!tag_format.IsErr()) {
    return ResultOk<Format>(std::move(tag_format).Unwrap());
  }
  auto literal_format = ParseLiteralFormat(obj);
  if (!literal_format.IsErr()) {
    return ResultOk<Format>(std::move(literal_format).Unwrap());
  }
  auto json_schema_format = ParseJSONSchemaFormat(obj);
  if (!json_schema_format.IsErr()) {
    return ResultOk<Format>(std::move(json_schema_format).Unwrap());
  }
  auto wildcard_text_format = ParseWildcardTextFormat(obj);
  if (!wildcard_text_format.IsErr()) {
    return ResultOk<Format>(std::move(wildcard_text_format).Unwrap());
  }
  auto sequence_format = ParseSequenceFormat(obj);
  if (!sequence_format.IsErr()) {
    return ResultOk<Format>(std::move(sequence_format).Unwrap());
  }
  auto triggered_tags_format = ParseTriggeredTagsFormat(obj);
  if (!triggered_tags_format.IsErr()) {
    return ResultOk<Format>(std::move(triggered_tags_format).Unwrap());
  }
  auto tags_with_separator_format = ParseTagsWithSeparatorFormat(obj);
  if (!tags_with_separator_format.IsErr()) {
    return ResultOk<Format>(std::move(tags_with_separator_format).Unwrap());
  }
  return ResultErr("Invalid format: " + value.serialize(false));
}

Result<LiteralFormat> StructuralTagImpl::ParseLiteralFormat(const picojson::object& obj) {
  // text is required.
  auto text_it = obj.find("text");
  if (text_it == obj.end() || !text_it->second.is<std::string>() ||
      text_it->second.get<std::string>().empty()) {
    return ResultErr("Literal format must have a text field with a non-empty string");
  }
  return ResultOk<LiteralFormat>({text_it->second.get<std::string>()});
}

Result<JSONSchemaFormat> StructuralTagImpl::ParseJSONSchemaFormat(const picojson::object& obj) {
  // json_schema is required.
  auto json_schema_it = obj.find("json_schema");
  if (json_schema_it == obj.end() || !json_schema_it->second.is<picojson::object>()) {
    return ResultErr("JSON schema format must have a json_schema field with a JSON object");
  }
  // here introduces a serialization/deserialization overhead; try to avoid it in the future.
  return ResultOk<JSONSchemaFormat>({json_schema_it->second.serialize(false)});
}

Result<WildcardTextFormat> StructuralTagImpl::ParseWildcardTextFormat(const picojson::object& obj) {
  // obj should not have any fields other than "type"
  if (obj.size() > 1 || (obj.size() == 1 && obj.begin()->first != "type")) {
    return ResultErr("Wildcard text format should not have any fields other than type");
  }
  return ResultOk<WildcardTextFormat>({});
}

Result<SequenceFormat> StructuralTagImpl::ParseSequenceFormat(const picojson::object& obj) {
  // elements is required.
  auto elements_it = obj.find("elements");
  if (elements_it == obj.end() || !elements_it->second.is<picojson::array>()) {
    return ResultErr("Sequence format must have an elements field with an array");
  }
  const auto& elements_array = elements_it->second.get<picojson::array>();
  std::vector<Format> elements;
  elements.reserve(elements_array.size());
  for (const auto& element : elements_array) {
    auto format = ParseFormat(element);
    if (format.IsErr()) {
      return ResultErr(std::move(format).UnwrapErr());
    }
    elements.push_back(std::move(format).Unwrap());
  }
  if (elements.size() == 0) {
    return ResultErr("Empty sequence format is not allowed");
  }
  return ResultOk<SequenceFormat>({std::move(elements)});
}

Result<TagFormat> StructuralTagImpl::ParseTagFormat(const picojson::value& value) {
  if (!value.is<picojson::object>()) {
    return ResultErr("Tag format must be an object");
  }
  const auto& obj = value.get<picojson::object>();
  if (obj.find("type") != obj.end() &&
      (!obj["type"].is<std::string>() || obj["type"].get<std::string>() != "tag")) {
    return ResultErr("Tag format's type must be a string \"tag\"");
  }
  return ParseTagFormat(obj);
}

Result<TagFormat> StructuralTagImpl::ParseTagFormat(const picojson::object& obj) {
  // begin is required.
  auto begin_it = obj.find("begin");
  if (begin_it == obj.end() || !begin_it->second.is<std::string>() ||
      begin_it->second.get<std::string>().empty()) {
    return ResultErr("Tag format must have a begin field with a non-empty string");
  }
  // content is required.
  auto content_it = obj.find("content");
  if (content_it == obj.end()) {
    return ResultErr("Tag format must have a content field");
  }
  auto content = ParseFormat(content_it->second);
  if (content.IsErr()) {
    return ResultErr(std::move(content).UnwrapErr());
  }
  // end is required.
  auto end_it = obj.find("end");
  if (end_it == obj.end() || !end_it->second.is<std::string>() ||
      end_it->second.get<std::string>().empty()) {
    return ResultErr("Tag format must have an end field with a non-empty string");
  }
  return ResultOk<TagFormat>(
      {begin_it->second.get<std::string>(),
       std::make_shared<Format>(std::move(content).Unwrap()),
       end_it->second.get<std::string>()}
  );
}

Result<TriggeredTagsFormat> StructuralTagImpl::ParseTriggeredTagsFormat(const picojson::object& obj
) {
  // triggers is required.
  auto triggers_it = obj.find("triggers");
  if (triggers_it == obj.end() || !triggers_it->second.is<picojson::array>()) {
    return ResultErr("Triggered tags format must have a triggers field with an array");
  }
  const auto& triggers_array = triggers_it->second.get<picojson::array>();
  std::vector<std::string> triggers;
  triggers.reserve(triggers_array.size());
  for (const auto& trigger : triggers_array) {
    if (!trigger.is<std::string>() || trigger.get<std::string>().empty()) {
      return ResultErr("Triggers must be non-empty strings");
    }
    triggers.push_back(trigger.get<std::string>());
  }
  if (triggers.size() == 0) {
    return ResultErr("Empty triggers are not allowed in triggered tags format");
  }
  // tags is required.
  auto tags_it = obj.find("tags");
  if (tags_it == obj.end() || !tags_it->second.is<picojson::array>()) {
    return ResultErr("Triggered tags format must have a tags field with an array");
  }
  const auto& tags_array = tags_it->second.get<picojson::array>();
  std::vector<TagFormat> tags;
  tags.reserve(tags_array.size());
  for (const auto& tag : tags_array) {
    auto tag_format = ParseTagFormat(tag);
    if (tag_format.IsErr()) {
      return ResultErr(std::move(tag_format).UnwrapErr());
    }
    tags.push_back(std::move(tag_format).Unwrap());
  }
  if (tags.size() == 0) {
    return ResultErr("Empty tags are not allowed in triggered tags format");
  }
  // at_least_one is optional.
  bool at_least_one = false;
  auto at_least_one_it = obj.find("at_least_one");
  if (at_least_one_it != obj.end()) {
    if (!at_least_one_it->second.is<bool>()) {
      return ResultErr("at_least_one must be a boolean");
    }
    at_least_one = at_least_one_it->second.get<bool>();
  }
  // stop_after_first is optional.
  bool stop_after_first = false;
  auto stop_after_first_it = obj.find("stop_after_first");
  if (stop_after_first_it != obj.end()) {
    if (!stop_after_first_it->second.is<bool>()) {
      return ResultErr("stop_after_first must be a boolean");
    }
    stop_after_first = stop_after_first_it->second.get<bool>();
  }
  return ResultOk<TriggeredTagsFormat>(
      {std::move(triggers), std::move(tags), at_least_one, stop_after_first}
  );
}

Result<TagsWithSeparatorFormat> StructuralTagImpl::ParseTagsWithSeparatorFormat(
    const picojson::object& obj
) {
  // tags is required.
  auto tags_it = obj.find("tags");
  if (tags_it == obj.end() || !tags_it->second.is<picojson::array>()) {
    return ResultErr("Tags with separator format must have a tags field with an array");
  }
  const auto& tags_array = tags_it->second.get<picojson::array>();
  std::vector<TagFormat> tags;
  tags.reserve(tags_array.size());
  for (const auto& tag : tags_array) {
    auto tag_format = ParseTagFormat(tag);
    if (tag_format.IsErr()) {
      return ResultErr(std::move(tag_format).UnwrapErr());
    }
    tags.push_back(std::move(tag_format).Unwrap());
  }
  if (tags.size() == 0) {
    return ResultErr("Empty tags are not allowed in tags with separator format");
  }
  // separator is required.
  auto separator_it = obj.find("separator");
  if (separator_it == obj.end() || !separator_it->second.is<std::string>() ||
      separator_it->second.get<std::string>().empty()) {
    return ResultErr(
        "Tags with separator format must have a separator field with a non-empty string"
    );
  }
  // at_least_one is optional.
  bool at_least_one = false;
  auto at_least_one_it = obj.find("at_least_one");
  if (at_least_one_it != obj.end()) {
    if (!at_least_one_it->second.is<bool>()) {
      return ResultErr("at_least_one must be a boolean");
    }
    at_least_one = at_least_one_it->second.get<bool>();
  }
  // stop_after_first is optional.
  bool stop_after_first = false;
  auto stop_after_first_it = obj.find("stop_after_first");
  if (stop_after_first_it != obj.end()) {
    if (!stop_after_first_it->second.is<bool>()) {
      return ResultErr("stop_after_first must be a boolean");
    }
    stop_after_first = stop_after_first_it->second.get<bool>();
  }
  return ResultOk<TagsWithSeparatorFormat>(
      {std::move(tags), separator_it->second.get<std::string>(), at_least_one, stop_after_first}
  );
}

/************** StructuralTag Public API **************/

std::variant<StructuralTag, std::runtime_error> StructuralTag::FromJSON(const std::string& json) {
  return StructuralTagImpl().FromJSON(json).ToVariant();
}

/************** StructuralTag Analyzer **************/

/*!
 * \brief Analyze a StructuralTag and extract useful information for conversion to Grammar.
 */
class StructuralTagInternalIR {
 public:
  struct LiteralFormatInternal;
  struct JSONSchemaFormatInternal;
  struct WildcardTextFormatInternal;
  struct SequenceFormatInternal;
  struct TagFormatInternal;
  struct TriggeredTagsFormatInternal;
  struct TagsWithSeparatorFormatInternal;

  using FormatInternal = std::variant<
      LiteralFormatInternal,
      JSONSchemaFormatInternal,
      WildcardTextFormatInternal,
      SequenceFormatInternal,
      TagFormatInternal,
      TriggeredTagsFormatInternal,
      TagsWithSeparatorFormatInternal>;

  /******************** Basic Formats ********************/

  struct LiteralFormatInternal {
    inline static constexpr std::string_view type = "literal";
    std::string text;
    bool deprived_ = false;
  };

  struct JSONSchemaFormatInternal {
    inline static constexpr std::string_view type = "json_schema";
    std::string json_schema;
  };

  struct WildcardTextFormatInternal {
    inline static constexpr std::string_view type = "wildcard_text";

    std::optional<std::string> detected_end_string_ = std::nullopt;
  };

  /******************** Combinatorial Formats ********************/

  struct SequenceFormatInternal {
    inline static constexpr std::string_view type = "sequence";
    std::vector<FormatInternal> elements;
  };

  struct TagFormatInternal {
    inline static constexpr std::string_view type = "tag";
    std::string begin;
    std::shared_ptr<FormatInternal> content;
    std::string end;

    bool begin_deprived_ = false;
    bool end_deprived_ = false;
  };

  struct TriggeredTagsFormatInternal {
    inline static constexpr std::string_view type = "triggered_tags";
    std::vector<std::string> triggers;
    std::vector<TagFormatInternal> tags;
    bool at_least_one = false;
    bool stop_after_first = false;

    std::optional<std::string> detected_end_string_ = std::nullopt;
  };

  struct TagsWithSeparatorFormatInternal {
    inline static constexpr std::string_view type = "tags_with_separator";
    std::vector<TagFormatInternal> tags;
    std::string separator;
    bool at_least_one = false;
    bool stop_after_first = false;

    std::optional<std::string> detected_end_string_ = std::nullopt;
  };

  struct StructuralTagInternal {
    inline static constexpr std::string_view type = "structural_tag";
    FormatInternal format;
  };
};

using STIIR = StructuralTagInternalIR;

class STIIRConverter {
 public:
  Result<STIIR::StructuralTagInternal> Convert(const StructuralTag& structural_tag);

 private:
  Result<STIIR::FormatInternal> VisitFormat(const Format& format);
  Result<STIIR::LiteralFormatInternal> VisitLiteralFormat(const LiteralFormat& format);
  Result<STIIR::JSONSchemaFormatInternal> VisitJSONSchemaFormat(const JSONSchemaFormat& format);
  Result<STIIR::WildcardTextFormatInternal> VisitWildcardTextFormat(const WildcardTextFormat& format
  );
  Result<STIIR::SequenceFormatInternal> VisitSequenceFormat(const SequenceFormat& format);
  Result<STIIR::TagFormatInternal> VisitTagFormat(const TagFormat& format);
  Result<STIIR::TriggeredTagsFormatInternal> VisitTriggeredTagsFormat(
      const TriggeredTagsFormat& format
  );
  Result<STIIR::TagsWithSeparatorFormatInternal> VisitTagsWithSeparatorFormat(
      const TagsWithSeparatorFormat& format
  );

  int visit_format_recursion_depth_ = 0;
};

Result<STIIR::StructuralTagInternal> STIIRConverter::Convert(const StructuralTag& structural_tag) {
  return VisitFormat(structural_tag.format).Map([](auto&& format) {
    return STIIR::StructuralTagInternal{std::move(format)};
  });
}

Result<STIIR::FormatInternal> STIIRConverter::VisitFormat(const Format& format) {
  RecursionGuard guard(&visit_format_recursion_depth_);
  return std::visit(
      [this](auto&& arg) -> Result<STIIR::FormatInternal> {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, LiteralFormat>) {
          return Result<STIIR::FormatInternal>::Convert(VisitLiteralFormat(arg));
        } else if constexpr (std::is_same_v<T, JSONSchemaFormat>) {
          return Result<STIIR::FormatInternal>::Convert(VisitJSONSchemaFormat(arg));
        } else if constexpr (std::is_same_v<T, WildcardTextFormat>) {
          return Result<STIIR::FormatInternal>::Convert(VisitWildcardTextFormat(arg));
        } else if constexpr (std::is_same_v<T, SequenceFormat>) {
          return Result<STIIR::FormatInternal>::Convert(VisitSequenceFormat(arg));
        } else if constexpr (std::is_same_v<T, TagFormat>) {
          return Result<STIIR::FormatInternal>::Convert(VisitTagFormat(arg));
        } else if constexpr (std::is_same_v<T, TriggeredTagsFormat>) {
          return Result<STIIR::FormatInternal>::Convert(VisitTriggeredTagsFormat(arg));
        } else if constexpr (std::is_same_v<T, TagsWithSeparatorFormat>) {
          return Result<STIIR::FormatInternal>::Convert(VisitTagsWithSeparatorFormat(arg));
        } else {
          // Should not be visited.
          XGRAMMAR_LOG(FATAL) << "STIIRConverter Internal: Unhandled format type: "
                              << typeid(T).name();
        }
      },
      format
  );
}

Result<STIIR::LiteralFormatInternal> STIIRConverter::VisitLiteralFormat(const LiteralFormat& format
) {
  return ResultOk<STIIR::LiteralFormatInternal>({format.text});
}

Result<STIIR::JSONSchemaFormatInternal> STIIRConverter::VisitJSONSchemaFormat(
    const JSONSchemaFormat& format
) {
  return ResultOk<STIIR::JSONSchemaFormatInternal>({format.json_schema});
}

Result<STIIR::WildcardTextFormatInternal> STIIRConverter::VisitWildcardTextFormat(
    const WildcardTextFormat& format
) {
  return ResultOk<STIIR::WildcardTextFormatInternal>({});
}

Result<STIIR::SequenceFormatInternal> STIIRConverter::VisitSequenceFormat(
    const SequenceFormat& format
) {
  std::vector<STIIR::FormatInternal> elements;
  for (const auto& element : format.elements) {
    auto result = VisitFormat(element);
    if (result.IsErr()) {
      return ResultErr(std::move(result).UnwrapErr());
    }
    elements.push_back(std::move(result).Unwrap());
  }
  return ResultOk<STIIR::SequenceFormatInternal>({std::move(elements)});
}

Result<STIIR::TagFormatInternal> STIIRConverter::VisitTagFormat(const TagFormat& format) {
  auto content_result = VisitFormat(*format.content);
  if (content_result.IsErr()) {
    return ResultErr(std::move(content_result).UnwrapErr());
  }
  return ResultOk<STIIR::TagFormatInternal>(
      {format.begin,
       std::make_shared<STIIR::FormatInternal>(std::move(content_result).Unwrap()),
       format.end}
  );
}

Result<STIIR::TriggeredTagsFormatInternal> STIIRConverter::VisitTriggeredTagsFormat(
    const TriggeredTagsFormat& format
) {
  std::vector<STIIR::TagFormatInternal> tags;
  tags.reserve(format.tags.size());
  for (const auto& tag : format.tags) {
    auto result = VisitTagFormat(tag);
    if (result.IsErr()) {
      return ResultErr(std::move(result).UnwrapErr());
    }
    tags.push_back(std::move(result).Unwrap());
  }
  return ResultOk<STIIR::TriggeredTagsFormatInternal>(
      {format.triggers, std::move(tags), format.at_least_one, format.stop_after_first}
  );
}

Result<STIIR::TagsWithSeparatorFormatInternal> STIIRConverter::VisitTagsWithSeparatorFormat(
    const TagsWithSeparatorFormat& format
) {
  std::vector<STIIR::TagFormatInternal> tags;
  tags.reserve(format.tags.size());
  for (const auto& tag : format.tags) {
    auto result = VisitTagFormat(tag);
    if (result.IsErr()) {
      return ResultErr(std::move(result).UnwrapErr());
    }
    tags.push_back(std::move(result).Unwrap());
  }
  return ResultOk<STIIR::TagsWithSeparatorFormatInternal>(
      {std::move(tags), format.separator, format.at_least_one, format.stop_after_first}
  );
}

class StructuralTagNestedSeqRemover {
 public:
  void RemoveNestedSeq(STIIR::StructuralTagInternal* structural_tag);

 private:
  void VisitFormat(STIIR::FormatInternal* format);
  void VisitLiteralFormat(STIIR::LiteralFormatInternal* format) {}
  void VisitJSONSchemaFormat(STIIR::JSONSchemaFormatInternal* format) {}
  void VisitWildcardTextFormat(STIIR::WildcardTextFormatInternal* format) {}
  void VisitSequenceFormat(STIIR::SequenceFormatInternal* format);
  void VisitTagFormat(STIIR::TagFormatInternal* format);
  void VisitTriggeredTagsFormat(STIIR::TriggeredTagsFormatInternal* format);
  void VisitTagsWithSeparatorFormat(STIIR::TagsWithSeparatorFormatInternal* format);
};

void StructuralTagNestedSeqRemover::RemoveNestedSeq(STIIR::StructuralTagInternal* structural_tag) {
  VisitFormat(&structural_tag->format);
}

void StructuralTagNestedSeqRemover::VisitFormat(STIIR::FormatInternal* format) {
  std::visit(
      [this](auto& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, STIIR::LiteralFormatInternal>) {
          VisitLiteralFormat(&arg);
        } else if constexpr (std::is_same_v<T, STIIR::JSONSchemaFormatInternal>) {
          VisitJSONSchemaFormat(&arg);
        } else if constexpr (std::is_same_v<T, STIIR::WildcardTextFormatInternal>) {
          VisitWildcardTextFormat(&arg);
        } else if constexpr (std::is_same_v<T, STIIR::SequenceFormatInternal>) {
          VisitSequenceFormat(&arg);
        } else if constexpr (std::is_same_v<T, STIIR::TagFormatInternal>) {
          VisitTagFormat(&arg);
        } else if constexpr (std::is_same_v<T, STIIR::TriggeredTagsFormatInternal>) {
          VisitTriggeredTagsFormat(&arg);
        } else if constexpr (std::is_same_v<T, STIIR::TagsWithSeparatorFormatInternal>) {
          VisitTagsWithSeparatorFormat(&arg);
        } else {
          // Should not be visited.
          XGRAMMAR_LOG(FATAL) << "StructuralTagNestedSeqRemover Internal: Unhandled format type: "
                              << typeid(T).name();
        }
      },
      *format
  );
}

void StructuralTagNestedSeqRemover::VisitSequenceFormat(STIIR::SequenceFormatInternal* format) {
  std::vector<STIIR::FormatInternal> flattened_elements;

  // Helper function to recursively flatten sequence elements
  std::function<void(const std::vector<STIIR::FormatInternal>&)> f_flatten_element =
      [&](const std::vector<STIIR::FormatInternal>& elements) {
        for (const auto& element : elements) {
          if (std::holds_alternative<STIIR::SequenceFormatInternal>(element)) {
            const auto& seq = std::get<STIIR::SequenceFormatInternal>(element);
            f_flatten_element(seq.elements);
          } else {
            flattened_elements.push_back(element);
          }
        }
      };

  f_flatten_element(format->elements);

  for (auto& element : flattened_elements) {
    XGRAMMAR_DCHECK(!std::holds_alternative<STIIR::SequenceFormatInternal>(element))
        << "Nested sequence should be flattened";
    VisitFormat(&element);
  }

  std::swap(format->elements, flattened_elements);
}

void StructuralTagNestedSeqRemover::VisitTagFormat(STIIR::TagFormatInternal* format) {
  VisitFormat(format->content.get());
}

void StructuralTagNestedSeqRemover::VisitTriggeredTagsFormat(
    STIIR::TriggeredTagsFormatInternal* format
) {
  for (auto& tag : format->tags) {
    VisitTagFormat(&tag);
  }
}

void StructuralTagNestedSeqRemover::VisitTagsWithSeparatorFormat(
    STIIR::TagsWithSeparatorFormatInternal* format
) {
  for (auto& tag : format->tags) {
    VisitTagFormat(&tag);
  }
}

/************** StructuralTag Analyzer **************/

class StructuralTagAnalyzer {
 private:
  using FormatInternal = STIIR::FormatInternal;
  using LiteralFormatInternal = STIIR::LiteralFormatInternal;
  using JSONSchemaFormatInternal = STIIR::JSONSchemaFormatInternal;
  using WildcardTextFormatInternal = STIIR::WildcardTextFormatInternal;
  using SequenceFormatInternal = STIIR::SequenceFormatInternal;
  using TagFormatInternal = STIIR::TagFormatInternal;
  using TriggeredTagsFormatInternal = STIIR::TriggeredTagsFormatInternal;
  using TagsWithSeparatorFormatInternal = STIIR::TagsWithSeparatorFormatInternal;
  using StructuralTagInternal = STIIR::StructuralTagInternal;

 public:
  std::optional<std::runtime_error> Analyze(StructuralTagInternal* structural_tag_internal);

 private:
  std::optional<std::runtime_error> VisitFormat(FormatInternal* format);
  std::optional<std::runtime_error> VisitLiteralFormat(LiteralFormatInternal* format);
  std::optional<std::runtime_error> VisitJSONSchemaFormat(JSONSchemaFormatInternal* format);
  std::optional<std::runtime_error> VisitWildcardTextFormat(WildcardTextFormatInternal* format);
  std::optional<std::runtime_error> VisitSequenceFormat(SequenceFormatInternal* format);
  std::optional<std::runtime_error> VisitTagFormat(TagFormatInternal* format);
  std::optional<std::runtime_error> VisitTriggeredTagsFormat(TriggeredTagsFormatInternal* format);
  std::optional<std::runtime_error> VisitTagsWithSeparatorFormat(
      TagsWithSeparatorFormatInternal* format
  );

  Result<std::optional<std::string>> DetectEndString();

  struct Frame {
    STIIR::FormatInternal* format;
    int element_id;  // The id of the element currently visited.
  };

  std::vector<Frame> stack_;
  int visit_format_recursion_depth_ = 0;
};

std::optional<std::runtime_error> StructuralTagAnalyzer::Analyze(
    STIIR::StructuralTagInternal* structural_tag_internal
) {
  return VisitFormat(&structural_tag_internal->format);
}

Result<std::optional<std::string>> StructuralTagAnalyzer::DetectEndString() {
  for (int i = static_cast<int>(stack_.size()) - 2; i >= 0; --i) {
    if (auto tag_format = std::get_if<TagFormatInternal>(stack_[i].format)) {
      if (tag_format->end_deprived_) {
        return ResultErr("StructuralTagAnalyzer Internal: End string is already deprived");
      }
      tag_format->end_deprived_ = true;
      return ResultOk<std::optional<std::string>>(tag_format->end);
    } else if (auto sequence_format = std::get_if<SequenceFormatInternal>(stack_[i].format)) {
      XGRAMMAR_DCHECK(sequence_format->elements.size() > 0);
      if (stack_[i].element_id == static_cast<int>(sequence_format->elements.size()) - 1) {
        continue;
      }
      auto next_element = sequence_format->elements[stack_[i].element_id + 1];
      if (auto next_element_tag = std::get_if<TagFormatInternal>(&next_element)) {
        if (next_element_tag->begin_deprived_) {
          return ResultErr("StructuralTagAnalyzer Internal: Begin string is already deprived");
        }
        next_element_tag->begin_deprived_ = true;
        return ResultOk<std::optional<std::string>>(next_element_tag->begin);
      } else if (auto next_element_literal = std::get_if<LiteralFormatInternal>(&next_element)) {
        if (next_element_literal->deprived_) {
          return ResultErr("StructuralTagAnalyzer Internal: Literal string is already deprived");
        }
        next_element_literal->deprived_ = true;
        return ResultOk<std::optional<std::string>>(next_element_literal->text);
      } else {
        // Nested sequence should be flattened by the nested seq remover.
        XGRAMMAR_DCHECK(!std::holds_alternative<SequenceFormatInternal>(next_element));
        // Must be the following:
        XGRAMMAR_DCHECK(
            std::holds_alternative<JSONSchemaFormatInternal>(next_element) ||
            std::holds_alternative<WildcardTextFormatInternal>(next_element) ||
            std::holds_alternative<TriggeredTagsFormatInternal>(next_element) ||
            std::holds_alternative<TagsWithSeparatorFormatInternal>(next_element)
        );
        auto type_name =
            std::visit([](auto&& fmt) -> std::string_view { return fmt.type; }, next_element);
        return ResultErr(
            std::string("StructuralTagAnalyzer: Cannot detect end string from type ") +
            std::string(type_name)
        );
      }
    } else {
      const auto& fmt = *stack_[i].format;
      // These formats cannot recursively contain other formats.
      XGRAMMAR_DCHECK(
          !std::holds_alternative<LiteralFormatInternal>(fmt) &&
          !std::holds_alternative<JSONSchemaFormatInternal>(fmt) &&
          !std::holds_alternative<WildcardTextFormatInternal>(fmt)
      );
      // Must be the following:
      XGRAMMAR_DCHECK(
          std::holds_alternative<TriggeredTagsFormatInternal>(fmt) ||
          std::holds_alternative<TagsWithSeparatorFormatInternal>(fmt)
      );
      auto type_name = std::visit([](auto&& fmt) -> std::string_view { return fmt.type; }, fmt);
      return ResultErr(
          std::string("StructuralTagAnalyzer: Cannot detect end string from type ") +
          std::string(type_name)
      );
    }
  }
  // If we reach the root, means the element is at the end of the structural tag, so there is no
  // end string.
  return ResultOk<std::optional<std::string>>(std::nullopt);
}

std::optional<std::runtime_error> StructuralTagAnalyzer::VisitFormat(FormatInternal* format) {
  RecursionGuard guard(&visit_format_recursion_depth_);
  stack_.push_back({format, 0});
  auto result = std::visit(
      [&](auto& arg) -> std::optional<std::runtime_error> {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, LiteralFormatInternal>) {
          return VisitLiteralFormat(&arg);
        } else if constexpr (std::is_same_v<T, JSONSchemaFormatInternal>) {
          return VisitJSONSchemaFormat(&arg);
        } else if constexpr (std::is_same_v<T, WildcardTextFormatInternal>) {
          return VisitWildcardTextFormat(&arg);
        } else if constexpr (std::is_same_v<T, SequenceFormatInternal>) {
          return VisitSequenceFormat(&arg);
        } else if constexpr (std::is_same_v<T, TagFormatInternal>) {
          return VisitTagFormat(&arg);
        } else if constexpr (std::is_same_v<T, TriggeredTagsFormatInternal>) {
          return VisitTriggeredTagsFormat(&arg);
        } else if constexpr (std::is_same_v<T, TagsWithSeparatorFormatInternal>) {
          return VisitTagsWithSeparatorFormat(&arg);
        } else {
          XGRAMMAR_LOG(FATAL) << "Unhandled format type: " << typeid(T).name();
        }
      },
      *format
  );
  stack_.pop_back();
  return result;
}

std::optional<std::runtime_error> StructuralTagAnalyzer::VisitLiteralFormat(
    LiteralFormatInternal* format
) {
  return std::nullopt;
}

std::optional<std::runtime_error> StructuralTagAnalyzer::VisitJSONSchemaFormat(
    JSONSchemaFormatInternal* format
) {
  return std::nullopt;
}

std::optional<std::runtime_error> StructuralTagAnalyzer::VisitWildcardTextFormat(
    WildcardTextFormatInternal* format
) {
  auto end_string_result = DetectEndString();
  if (end_string_result.IsErr()) {
    return std::move(end_string_result).UnwrapErr();
  }
  format->detected_end_string_ = std::move(end_string_result).Unwrap();
  return std::nullopt;
}

std::optional<std::runtime_error> StructuralTagAnalyzer::VisitSequenceFormat(
    SequenceFormatInternal* format
) {
  for (auto& element : format->elements) {
    auto result = VisitFormat(&element);
    if (result.has_value()) {
      return result;
    }
    ++stack_.back().element_id;
  }
  return std::nullopt;
}

std::optional<std::runtime_error> StructuralTagAnalyzer::VisitTagFormat(TagFormatInternal* format) {
  return VisitFormat(format->content.get());
}

std::optional<std::runtime_error> StructuralTagAnalyzer::VisitTriggeredTagsFormat(
    TriggeredTagsFormatInternal* format
) {
  auto end_string_result = DetectEndString();
  if (end_string_result.IsErr()) {
    return std::move(end_string_result).UnwrapErr();
  }
  format->detected_end_string_ = std::move(end_string_result).Unwrap();
  for (auto& tag : format->tags) {
    auto result = VisitTagFormat(&tag);
    if (result.has_value()) {
      return result;
    }
    ++stack_.back().element_id;
  }
  return std::nullopt;
}

std::optional<std::runtime_error> StructuralTagAnalyzer::VisitTagsWithSeparatorFormat(
    TagsWithSeparatorFormatInternal* format
) {
  auto end_string_result = DetectEndString();
  if (end_string_result.IsErr()) {
    return std::move(end_string_result).UnwrapErr();
  }
  format->detected_end_string_ = std::move(end_string_result).Unwrap();
  for (auto& tag : format->tags) {
    auto result = VisitTagFormat(&tag);
    if (result.has_value()) {
      return result;
    }
    ++stack_.back().element_id;
  }
  return std::nullopt;
}

/************** StructuralTagInternal to Grammar Converter **************/

class StructuralTagInternalToGrammarConverter {
 private:
  using FormatInternal = STIIR::FormatInternal;
  using LiteralFormatInternal = STIIR::LiteralFormatInternal;
  using JSONSchemaFormatInternal = STIIR::JSONSchemaFormatInternal;
  using WildcardTextFormatInternal = STIIR::WildcardTextFormatInternal;
  using SequenceFormatInternal = STIIR::SequenceFormatInternal;
  using TagFormatInternal = STIIR::TagFormatInternal;
  using TriggeredTagsFormatInternal = STIIR::TriggeredTagsFormatInternal;
  using TagsWithSeparatorFormatInternal = STIIR::TagsWithSeparatorFormatInternal;
  using StructuralTagInternal = STIIR::StructuralTagInternal;

 public:
  Result<Grammar> Convert(const StructuralTagInternal& structural_tag_internal);

 private:
  Result<int32_t> VisitFormat(const FormatInternal& format);
  Result<int32_t> VisitLiteralFormat(const LiteralFormatInternal& format);
  Result<int32_t> VisitJSONSchemaFormat(const JSONSchemaFormatInternal& format);
  Result<int32_t> VisitWildcardTextFormat(const WildcardTextFormatInternal& format);
  Result<int32_t> VisitSequenceFormat(const SequenceFormatInternal& format);
  Result<int32_t> VisitTagFormat(const TagFormatInternal& format);
  Result<int32_t> VisitTriggeredTagsFormat(const TriggeredTagsFormatInternal& format);
  Result<int32_t> VisitTagsWithSeparatorFormat(const TagsWithSeparatorFormatInternal& format);

  GrammarBuilder builder_;
};

void StructuralTagInternalToGrammarConverter::Convert(
    const StructuralTagInternal& structural_tag_internal
) {
  builder_ = GrammarBuilder();
  VisitFormat(structural_tag_internal.format);
}

Result<int32_t> StructuralTagInternalToGrammarConverter::VisitFormat(const FormatInternal& format) {
  return std::visit(
      [&](auto&& arg) -> Result<int32_t> {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, LiteralFormatInternal>) {
          return VisitLiteralFormat(arg);
        } else if constexpr (std::is_same_v<T, JSONSchemaFormatInternal>) {
          return VisitJSONSchemaFormat(arg);
        } else if constexpr (std::is_same_v<T, WildcardTextFormatInternal>) {
          return VisitWildcardTextFormat(arg);
        } else if constexpr (std::is_same_v<T, SequenceFormatInternal>) {
          return VisitSequenceFormat(arg);
        } else if constexpr (std::is_same_v<T, TagFormatInternal>) {
          return VisitTagFormat(arg);
        } else if constexpr (std::is_same_v<T, TriggeredTagsFormatInternal>) {
          return VisitTriggeredTagsFormat(arg);
        } else if constexpr (std::is_same_v<T, TagsWithSeparatorFormatInternal>) {
          return VisitTagsWithSeparatorFormat(arg);
        }
      },
      format
  );
}

Result<int32_t> StructuralTagInternalToGrammarConverter::VisitLiteralFormat(
    const LiteralFormatInternal& format
) {
  auto rule_expr_id = builder_.AddByteString(format.text);
  return ResultOk<int32_t>(builder_.AddRuleWithHint("literal", rule_expr_id));
}

Result<int32_t> StructuralTagInternalToGrammarConverter::VisitJSONSchemaFormat(
    const JSONSchemaFormatInternal& format
) {
  auto schema_grammar = Grammar::FromJSONSchema(format.schema, true);
  return ResultOk<int32_t>(SubGrammarAdder::Apply(&builder_, schema_grammar));
}

Result<int32_t> StructuralTagInternalToGrammarConverter::VisitWildcardTextFormat(
    const WildcardTextFormatInternal& format
) {
  if (format.detected_end_string_.has_value()) {
    auto rule_expr_id = builder_.AddByteString(*format.detected_end_string_);
    return ResultOk<int32_t>(builder_.AddRuleWithHint("wildcard_text", rule_expr_id));
  }
  return ResultErr(std::runtime_error("Wildcard text format has no detected end string"));
}

// Result<Grammar> StructuralTagGrammarConverter::Convert(const std::string& structural_tag_json) {
//   auto structural_tag = StructuralTagImpl().FromJSON(structural_tag_json);
//   if (structural_tag.IsErr()) {
//     throw std::runtime_error(std::move(structural_tag).UnwrapErr());
//   }
//   return Convert(std::move(structural_tag).Unwrap());
// }

// Result<Grammar> StructuralTagGrammarConverter::Convert(const StructuralTag& structural_tag) {
//   return ResultErr("Not implemented");
//   // auto structural_tag_internal = STIIR::FromStructuralTag(structural_tag);
//   // if (structural_tag_internal.IsErr()) {
//   //   return ResultErr(std::move(structural_tag_internal).UnwrapErr());
//   // }
//   // return Convert(std::move(structural_tag_internal).Unwrap());
// }

// Result<Grammar> StructuralTagGrammarConverter::Convert(
//     const StructuralTagInternal& structural_tag_internal
// ) {
//   return ResultErr("Not implemented");
// }

/************** StructuralTag to Grammar Converter **************/

/**
 * \brief Convert a StructuralTag or a StructuralTag JSON string to a Grammar. It assembles the
 * above conversion and analysis steps.
 */
class StructuralTagToGrammarConverter {
 public:
  static Result<Grammar> Convert(const std::string& structural_tag_json);
  static Result<Grammar> Convert(const StructuralTag& structural_tag);
};

Result<Grammar> StructuralTagToGrammarConverter::Convert(const std::string& structural_tag_json) {
  auto structural_tag = StructuralTagImpl().FromJSON(structural_tag_json);
  if (structural_tag.IsErr()) {
    throw std::runtime_error(std::move(structural_tag).UnwrapErr());
  }
  return Convert(std::move(structural_tag).Unwrap());
}

Result<Grammar> StructuralTagToGrammarConverter::Convert(const StructuralTag& structural_tag) {
  auto structural_tag_internal_result = STIIRConverter().Convert(structural_tag);
  if (structural_tag_internal_result.IsErr()) {
    return ResultErr(std::move(structural_tag_internal_result).UnwrapErr());
  }
  auto structural_tag_internal = std::move(structural_tag_internal_result).Unwrap();
  StructuralTagNestedSeqRemover().RemoveNestedSeq(&structural_tag_internal);
  auto analyzer_result = StructuralTagAnalyzer().Analyze(&structural_tag_internal);
  if (analyzer_result.has_value()) {
    return ResultErr(std::move(*analyzer_result));
  }
  return StructuralTagInternalToGrammarConverter().Convert(structural_tag_internal);
}

/************** StructuralTag to Grammar Public API **************/

Result<Grammar> StructuralTagToGrammar(const std::string& structural_tag_json) {
  return StructuralTagToGrammarConverter::Convert(structural_tag_json);
}

Result<Grammar> StructuralTagToGrammar(const StructuralTag& structural_tag) {
  return StructuralTagToGrammarConverter::Convert(structural_tag);
}

// Grammar StructuralTagToGrammar(
//     const std::vector<StructuralTagItem>& tags, const std::vector<std::string>& triggers
// ) {
//   // Step 1: handle triggers. Triggers should not be mutually inclusive
//   std::vector<std::string> sorted_triggers(triggers.begin(), triggers.end());
//   std::sort(sorted_triggers.begin(), sorted_triggers.end());
//   for (int i = 0; i < static_cast<int>(sorted_triggers.size()) - 1; ++i) {
//     XGRAMMAR_CHECK(
//         sorted_triggers[i + 1].size() < sorted_triggers[i].size() ||
//         std::string_view(sorted_triggers[i + 1]).substr(0, sorted_triggers[i].size()) !=
//             sorted_triggers[i]
//     ) << "Triggers should not be mutually inclusive, but "
//       << sorted_triggers[i] << " is a prefix of " << sorted_triggers[i + 1];
//   }

//   // Step 2: For each tag, find the trigger that is a prefix of the tag.begin
//   // Convert the schema to grammar at the same time
//   std::vector<Grammar> schema_grammars;
//   schema_grammars.reserve(tags.size());
//   for (const auto& tag : tags) {
//     auto schema_grammar = Grammar::FromJSONSchema(tag.schema, true);
//     schema_grammars.push_back(schema_grammar);
//   }

//   std::vector<std::vector<std::pair<StructuralTagItem, Grammar>>> tag_groups(triggers.size());
//   for (int it_tag = 0; it_tag < static_cast<int>(tags.size()); ++it_tag) {
//     const auto& tag = tags[it_tag];
//     bool found = false;
//     for (int it_trigger = 0; it_trigger < static_cast<int>(sorted_triggers.size());
//     ++it_trigger)
//     {
//       const auto& trigger = sorted_triggers[it_trigger];
//       if (trigger.size() <= tag.begin.size() &&
//           std::string_view(tag.begin).substr(0, trigger.size()) == trigger) {
//         tag_groups[it_trigger].push_back(std::make_pair(tag, schema_grammars[it_tag]));
//         found = true;
//         break;
//       }
//     }
//     XGRAMMAR_CHECK(found) << "Tag " << tag.begin << " does not match any trigger";
//   }

//   // Step 3: Combine the tags to form a grammar
//   // root ::= TagDispatch((trigger1, rule1), (trigger2, rule2), ...)
//   // Suppose tag1 and tag2 matches trigger1, then
//   // rule1 ::= (tag1.begin[trigger1.size():] + ToEBNF(tag1.schema) + tag1.end) |
//   //            (tag2.begin[trigger1.size():] + ToEBNF(tag2.schema) + tag2.end) | ...
//   //
//   // Suppose tag3 matches trigger2, then
//   // rule2 ::= (tag3.begin[trigger2.size():] + ToEBNF(tag3.schema) + tag3.end)
//   //
//   // ...
//   return StructuralTagGrammarCreator::Apply(sorted_triggers, tag_groups);
// }

}  // namespace xgrammar
