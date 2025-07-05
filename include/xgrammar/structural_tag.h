/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag.h
 * \brief The header for the definition of the structural tag.
 */
#ifndef XGRAMMAR_STRUCTURAL_TAG_H_
#define XGRAMMAR_STRUCTURAL_TAG_H_

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace xgrammar {

/******************** Discriminated Union ********************/

struct LiteralFormat;
struct JSONSchemaFormat;
struct WildcardTextFormat;
struct SequenceFormat;
struct TagFormat;
struct TriggeredTagsFormat;
struct TagsWithSeparatorFormat;

using Format = std::variant<
    LiteralFormat,
    JSONSchemaFormat,
    WildcardTextFormat,
    SequenceFormat,
    TagFormat,
    TriggeredTagsFormat,
    TagsWithSeparatorFormat>;

/******************** Basic Formats ********************/

struct LiteralFormat {
  inline static constexpr std::string_view type = "literal";
  std::string text;
};

struct JSONSchemaFormat {
  inline static constexpr std::string_view type = "json_schema";
  std::string json_schema;
};

struct WildcardTextFormat {
  inline static constexpr std::string_view type = "wildcard_text";
};

/******************** Combinatorial Formats ********************/

struct SequenceFormat {
  inline static constexpr std::string_view type = "sequence";
  std::vector<Format> elements;
};

struct TagFormat {
  inline static constexpr std::string_view type = "tag";
  std::string begin;
  std::shared_ptr<Format> content;
  std::string end;
};

struct TriggeredTagsFormat {
  inline static constexpr std::string_view type = "triggered_tags";
  std::vector<std::string> triggers;
  std::vector<TagFormat> tags;
  bool at_least_one = false;
  bool stop_after_first = false;
};

struct TagsWithSeparatorFormat {
  inline static constexpr std::string_view type = "tags_with_separator";
  std::vector<TagFormat> tags;
  std::string separator;
  bool at_least_one = false;
  bool stop_after_first = false;
};

/******************** Top Level ********************/

struct StructuralTag {
  inline static constexpr std::string_view type = "structural_tag";
  Format format;

  /*!
   * \brief Parse a JSON string into a StructuralTag.
   * \param json The JSON string to parse.
   * \return A StructuralTag if the JSON is valid, otherwise an error message in std::string.
   */
  static std::variant<StructuralTag, std::runtime_error> FromJSON(const std::string& json);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_STRUCTURAL_TAG_H_
