/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag_impl.h
 * \brief The implementation header for the structural tag.
 */

#ifndef XGRAMMAR_STRUCTURAL_TAG_H_
#define XGRAMMAR_STRUCTURAL_TAG_H_

#include <picojson.h>
#include <xgrammar/exception.h>
#include <xgrammar/grammar.h>

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "support/utils.h"
#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

/******************** Structural Tag Definition ********************/

// TODO(yixin): Consider moving the definition to Public API.

struct ConstStringFormat;
struct JSONSchemaFormat;
struct AnyTextFormat;
struct GrammarFormat;
struct RegexFormat;
struct SequenceFormat;
struct OrFormat;
struct TagFormat;
struct TriggeredTagsFormat;
struct TagsWithSeparatorFormat;
struct OptionalFormat;
struct PlusFormat;
struct StarFormat;
struct TokenFormat;
struct ExcludeTokenFormat;
struct AnyTokensFormat;
struct TokenTriggeredTagsFormat;

using Format = std::variant<
    ConstStringFormat,
    JSONSchemaFormat,
    AnyTextFormat,
    GrammarFormat,
    RegexFormat,
    SequenceFormat,
    OrFormat,
    TagFormat,
    TriggeredTagsFormat,
    TagsWithSeparatorFormat,
    OptionalFormat,
    PlusFormat,
    StarFormat,
    TokenFormat,
    ExcludeTokenFormat,
    AnyTokensFormat,
    TokenTriggeredTagsFormat>;

/******************** Basic Formats ********************/

struct ConstStringFormat {
  static constexpr const char* type = "const_string";
  std::string value;
  ConstStringFormat(std::string value) : value(std::move(value)) {}
  picojson::value ToJSON() const;
};

struct JSONSchemaFormat {
  static constexpr const char* type = "json_schema";
  std::string json_schema;
  std::string style = "json";  // "json","qwen_xml","minimax_xml"
  JSONSchemaFormat(std::string json_schema, std::string style = "json")
      : json_schema(std::move(json_schema)), style(std::move(style)) {}
  picojson::value ToJSON() const;
};

struct GrammarFormat {
  static constexpr const char* type = "grammar";
  std::string grammar;
  GrammarFormat(std::string grammar) : grammar(std::move(grammar)) {}
  picojson::value ToJSON() const;
};

struct RegexFormat {
  static constexpr const char* type = "regex";
  std::string pattern;
  RegexFormat(std::string pattern) : pattern(std::move(pattern)) {}
  picojson::value ToJSON() const;
};

struct AnyTextFormat {
  static constexpr const char* type = "any_text";
  std::vector<std::string> excludes;
  AnyTextFormat(std::vector<std::string> excluded_strs) : excludes(std::move(excluded_strs)) {}
  picojson::value ToJSON() const;

 private:
  std::vector<std::string> detected_end_strs_;
  friend class StructuralTagAnalyzer;
  friend class StructuralTagGrammarConverter;
};

struct TokenFormat {
  static constexpr const char* type = "token";
  int32_t token_id = -1;
  std::variant<int32_t, std::string> token;
  TokenFormat(std::variant<int32_t, std::string> token) : token(std::move(token)) {
    if (std::holds_alternative<int32_t>(this->token)) {
      token_id = std::get<int32_t>(this->token);
    }
  }
  picojson::value ToJSON() const;

 private:
  friend class StructuralTagTokenResolver;
};

struct ExcludeTokenFormat {
  static constexpr const char* type = "exclude_token";
  std::vector<std::variant<int32_t, std::string>> tokens;
  ExcludeTokenFormat(std::vector<std::variant<int32_t, std::string>> tokens)
      : tokens(std::move(tokens)) {}
  picojson::value ToJSON() const;

 private:
  std::vector<int32_t> resolved_token_ids_;
  std::vector<int32_t> detected_end_token_ids_;
  friend class StructuralTagTokenResolver;
  friend class StructuralTagAnalyzer;
  friend class StructuralTagGrammarConverter;
};

struct AnyTokensFormat {
  static constexpr const char* type = "any_tokens";
  std::vector<std::variant<int32_t, std::string>> exclude_tokens;
  AnyTokensFormat(std::vector<std::variant<int32_t, std::string>> exclude_tokens)
      : exclude_tokens(std::move(exclude_tokens)) {}
  picojson::value ToJSON() const;

 private:
  std::vector<int32_t> resolved_exclude_token_ids_;
  std::vector<int32_t> detected_end_token_ids_;
  friend class StructuralTagTokenResolver;
  friend class StructuralTagAnalyzer;
  friend class StructuralTagGrammarConverter;
};

/******************** Combinatorial Formats ********************/

struct SequenceFormat {
  static constexpr const char* type = "sequence";
  std::vector<Format> elements;
  SequenceFormat(std::vector<Format> elements) : elements(std::move(elements)) {}
  picojson::value ToJSON() const;

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
  picojson::value ToJSON() const;

 private:
  // Detected in StructuralTagAnalyzer
  bool is_unlimited_ = false;
  friend class StructuralTagAnalyzer;
  friend class StructuralTagGrammarConverter;
};

struct TagFormat {
  static constexpr const char* type = "tag";
  std::variant<std::string, TokenFormat> begin;
  std::shared_ptr<Format> content;
  std::variant<std::vector<std::string>, TokenFormat> end;

  TagFormat(
      std::variant<std::string, TokenFormat> begin,
      std::shared_ptr<Format> content,
      std::variant<std::vector<std::string>, TokenFormat> end
  )
      : begin(std::move(begin)), content(std::move(content)), end(std::move(end)) {}
  picojson::value ToJSON() const;
};

struct TriggeredTagsFormat {
  static constexpr const char* type = "triggered_tags";
  std::vector<std::string> triggers;
  std::vector<TagFormat> tags;
  std::vector<std::string> excludes;
  bool at_least_one = false;
  bool stop_after_first = false;

  TriggeredTagsFormat(
      std::vector<std::string> triggers,
      std::vector<TagFormat> tags,
      std::vector<std::string> excludes,
      bool at_least_one,
      bool stop_after_first
  )
      : triggers(std::move(triggers)),
        tags(std::move(tags)),
        excludes(std::move(excludes)),
        at_least_one(at_least_one),
        stop_after_first(stop_after_first) {}
  picojson::value ToJSON() const;

 private:
  std::vector<std::string> detected_end_strs_;
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
  picojson::value ToJSON() const;

 private:
  friend class StructuralTagAnalyzer;
  friend class StructuralTagGrammarConverter;
};

struct TokenTriggeredTagsFormat {
  static constexpr const char* type = "token_triggered_tags";
  std::vector<std::variant<int32_t, std::string>> trigger_tokens;
  std::vector<TagFormat> tags;
  std::vector<std::variant<int32_t, std::string>> exclude_tokens;
  bool at_least_one = false;
  bool stop_after_first = false;

  TokenTriggeredTagsFormat(
      std::vector<std::variant<int32_t, std::string>> trigger_tokens,
      std::vector<TagFormat> tags,
      std::vector<std::variant<int32_t, std::string>> exclude_tokens,
      bool at_least_one,
      bool stop_after_first
  )
      : trigger_tokens(std::move(trigger_tokens)),
        tags(std::move(tags)),
        exclude_tokens(std::move(exclude_tokens)),
        at_least_one(at_least_one),
        stop_after_first(stop_after_first) {}
  picojson::value ToJSON() const;

 private:
  std::vector<int32_t> resolved_trigger_token_ids_;
  std::vector<int32_t> resolved_exclude_token_ids_;
  std::vector<int32_t> detected_end_token_ids_;
  friend class StructuralTagTokenResolver;
  friend class StructuralTagAnalyzer;
  friend class StructuralTagGrammarConverter;
};

struct OptionalFormat {
  static constexpr const char* type = "optional";
  std::shared_ptr<Format> content;
  OptionalFormat(std::shared_ptr<Format> content) : content(std::move(content)) {}
  picojson::value ToJSON() const;
};

struct PlusFormat {
  static constexpr const char* type = "plus";
  std::shared_ptr<Format> content;
  PlusFormat(std::shared_ptr<Format> content) : content(std::move(content)) {}
  picojson::value ToJSON() const;
};

struct StarFormat {
  static constexpr const char* type = "star";
  std::shared_ptr<Format> content;
  StarFormat(std::shared_ptr<Format> content) : content(std::move(content)) {}
  picojson::value ToJSON() const;
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
Result<Grammar, StructuralTagError> StructuralTagToGrammar(
    const std::string& structural_tag_json, const TokenizerInfo* tokenizer_info = nullptr
);

}  // namespace xgrammar

#endif  // XGRAMMAR_STRUCTURAL_TAG_H_
