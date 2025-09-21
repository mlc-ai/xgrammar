/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag_impl.h
 * \brief The implementation header for the structural tag.
 */

#ifndef XGRAMMAR_STRUCTURAL_TAG_H_
#define XGRAMMAR_STRUCTURAL_TAG_H_

#include <xgrammar/exception.h>
#include <xgrammar/grammar.h>

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "support/utils.h"

namespace xgrammar {

/******************** Structural Tag Definition ********************/

// TODO(yixin): Consider moving the definition to Public API.

struct ConstStringFormat;
struct JSONSchemaFormat;
struct QwenXmlParameterFormat;
struct AnyTextFormat;
struct GrammarFormat;
struct RegexFormat;
struct SequenceFormat;
struct OrFormat;
struct TagFormat;
struct TriggeredTagsFormat;
struct TagsWithSeparatorFormat;

using Format = std::variant<
    ConstStringFormat,
    JSONSchemaFormat,
    QwenXmlParameterFormat,
    AnyTextFormat,
    GrammarFormat,
    RegexFormat,
    SequenceFormat,
    OrFormat,
    TagFormat,
    TriggeredTagsFormat,
    TagsWithSeparatorFormat>;

/******************** Basic Formats ********************/

struct ConstStringFormat {
  static constexpr const char* type = "const_string";
  std::string value;
  ConstStringFormat(std::string value) : value(std::move(value)) {}
};

struct JSONSchemaFormat {
  static constexpr const char* type = "json_schema";
  std::string json_schema;
  JSONSchemaFormat(std::string json_schema) : json_schema(std::move(json_schema)) {}
};

struct QwenXmlParameterFormat {
  static constexpr const char* type = "qwen_xml";
  std::string xml_schema;
  QwenXmlParameterFormat(std::string xml_schema) : xml_schema(std::move(xml_schema)) {}
};

struct GrammarFormat {
  static constexpr const char* type = "grammar";
  std::string grammar;
  GrammarFormat(std::string grammar) : grammar(std::move(grammar)) {}
};

struct RegexFormat {
  static constexpr const char* type = "regex";
  std::string pattern;
  RegexFormat(std::string pattern) : pattern(std::move(pattern)) {}
};

struct AnyTextFormat {
  static constexpr const char* type = "any_text";
  AnyTextFormat() {}

 private:
  // Detected in StructuralTagAnalyzer
  std::optional<std::string> detected_end_str_ = std::nullopt;
  friend class StructuralTagAnalyzer;
  friend class StructuralTagGrammarConverter;
};

/******************** Combinatorial Formats ********************/

struct SequenceFormat {
  static constexpr const char* type = "sequence";
  std::vector<Format> elements;
  SequenceFormat(std::vector<Format> elements) : elements(std::move(elements)) {}

 private:
  // Detected in StructuralTagAnalyzer
  bool is_unlimited_ = false;
  friend class StructuralTagAnalyzer;
  friend class StructuralTagGrammarConverter;
};

struct OrFormat {
  static constexpr const char* type = "or";
  std::vector<Format> elements;
  OrFormat(std::vector<Format> elements) : elements(std::move(elements)) {}

 private:
  // Detected in StructuralTagAnalyzer
  bool is_unlimited_ = false;
  friend class StructuralTagAnalyzer;
  friend class StructuralTagGrammarConverter;
};

struct TagFormat {
  static constexpr const char* type = "tag";
  std::string begin;
  std::shared_ptr<Format> content;
  std::string end;

  TagFormat(std::string begin, std::shared_ptr<Format> content, std::string end)
      : begin(std::move(begin)), content(std::move(content)), end(std::move(end)) {}
};

struct TriggeredTagsFormat {
  static constexpr const char* type = "triggered_tags";
  std::vector<std::string> triggers;
  std::vector<TagFormat> tags;
  bool at_least_one = false;
  bool stop_after_first = false;

  TriggeredTagsFormat(
      std::vector<std::string> triggers,
      std::vector<TagFormat> tags,
      bool at_least_one,
      bool stop_after_first
  )
      : triggers(std::move(triggers)),
        tags(std::move(tags)),
        at_least_one(at_least_one),
        stop_after_first(stop_after_first) {}

 private:
  // Detected in StructuralTagAnalyzer
  std::optional<std::string> detected_end_str_ = std::nullopt;
  friend class StructuralTagAnalyzer;
  friend class StructuralTagGrammarConverter;
};

struct TagsWithSeparatorFormat {
  static constexpr const char* type = "tags_with_separator";
  std::vector<TagFormat> tags;
  std::string separator;
  bool at_least_one = false;
  bool stop_after_first = false;

  TagsWithSeparatorFormat(
      std::vector<TagFormat> tags, std::string separator, bool at_least_one, bool stop_after_first
  )
      : tags(std::move(tags)),
        separator(std::move(separator)),
        at_least_one(at_least_one),
        stop_after_first(stop_after_first) {}

 private:
  // Detected in StructuralTagAnalyzer
  std::optional<std::string> detected_end_str_ = std::nullopt;
  friend class StructuralTagAnalyzer;
  friend class StructuralTagGrammarConverter;
};

/******************** Top Level ********************/

struct StructuralTag {
  static constexpr const char* type = "structural_tag";
  Format format;

  StructuralTag(Format format) : format(std::move(format)) {}
};

/******************** Conversion API ********************/

/*!
 * \brief Convert a structural tag JSON string to a grammar.
 * \param structural_tag_json The JSON string of the structural tag.
 * \return A grammar if the JSON is valid, otherwise an error message in std::string.
 */
Result<Grammar, StructuralTagError> StructuralTagToGrammar(const std::string& structural_tag_json);

}  // namespace xgrammar

#endif  // XGRAMMAR_STRUCTURAL_TAG_H_
