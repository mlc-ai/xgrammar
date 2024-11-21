/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter.h
 * \brief Convert a JSON schema string to EBNF grammar string.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_CONVERTER_H_
#define XGRAMMAR_JSON_SCHEMA_CONVERTER_H_

#include <picojson.h>

#include <optional>
#include <string>
#include <utility>

namespace xgrammar {

/*!
 * \brief Convert a JSON schema string to EBNF grammar string.
 */
std::string JSONSchemaToEBNF(
    const std::string& schema,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode
);

std::string JSONSchemaToEBNF(
    const picojson::value& schema,
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool strict_mode
);

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_H_
