/*!
 * Copyright (c) 2026 by Contributors
 * \file xgrammar/json_schema_converter_ast.h
 * \brief Internal entry point for direct JSON Schema AST construction.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_CONVERTER_AST_H_
#define XGRAMMAR_JSON_SCHEMA_CONVERTER_AST_H_

#include <optional>
#include <string>
#include <utility>

#include "json_schema_converter.h"

namespace xgrammar {

Grammar JSONSchemaSpecToGrammar(
    const SchemaSpecPtr& spec,
    bool any_whitespace,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    std::optional<int> max_whitespace_cnt,
    JSONSchemaConverter::RefResolver ref_resolver,
    bool any_order
);

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_AST_H_
