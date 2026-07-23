/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/json_schema_grammar_converter.h
 * \brief Build a grammar AST directly from the JSON Schema intermediate representation.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_GRAMMAR_CONVERTER_H_
#define XGRAMMAR_JSON_SCHEMA_GRAMMAR_CONVERTER_H_

#include <xgrammar/grammar.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "json_schema_converter.h"

namespace xgrammar {

/*!
 * \brief Convert parsed JSON Schema specifications directly into a Grammar AST.
 *
 * Unlike JSONSchemaConverter, this class does not create an intermediate EBNF string.
 */
class JSONSchemaGrammarConverter {
 public:
  using RefResolver =
      std::function<SchemaSpecPtr(const std::string& uri, const std::string& rule_name_hint)>;

  JSONSchemaGrammarConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      bool any_order = false
  );

  Grammar Convert(const SchemaSpecPtr& spec);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

/*!
 * \brief Convert a JSON schema string directly to a grammar AST.
 *
 * This bypasses the EBNF text generation and parsing path used by JSONSchemaToEBNF.
 */
Grammar JSONSchemaToGrammar(
    const std::string& schema,
    bool any_whitespace = true,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true,
    std::optional<int> max_whitespace_cnt = std::nullopt,
    bool any_order = false
);

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_GRAMMAR_CONVERTER_H_
