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

enum class JSONFormat : int {
  kJSON = 0,
  kXML = 1,
};

/*!
 * \brief Convert JSON schema string to EBNF grammar string.
 * \param schema The JSON schema string.
 * \param any_whitespace Whether to ignore the indentation restrictions, and allow any whitespace.
 * Default: true.
 * \param indent The number of spaces for indentation. If set to std::nullopt, the output will be
 * in one line. Default: 2.
 * \param separators Two separators used in the schema: comma and colon. Examples: {",", ":"},
 * {", ", ": "}. If std::nullopt, the default separators will be used: {",", ": "} when the
 * indent is not -1, and {", ", ": "} otherwise. This follows the convention in python
 * json.dumps(). Default: std::nullopt.
 * \param strict_mode Whether to use strict mode. In strict
 * mode, the generated grammar will not allow properties and items that is not specified in the
 * schema. This is equivalent to setting unevaluatedProperties and unevaluatedItems to false.
 * This helps LLM to generate accurate output in the grammar-guided generation with JSON
 * schema. Default: true.
 * \param max_whitespace_cnt The maximum number of whitespace characters for the whitespace
 * which is used for indentation or JSON elements separation when any_whitespace is True. If
 * std::nullopt, it means unlimited. Default: std::nullopt.
 * \param json_format Define the root
 * format of the object. If it's JSONFormat::kJSON, then it will generate a fully JSON-style
 * grammar. If it's JSONFormat::kXML, then it will generate a grammar with the root format is
 * XML-style, while the inner format is JSON-style. Default: JSONFormat::kJSON.
 * \returns The EBNF grammar string.
 */

std::string JSONSchemaToEBNF(
    const std::string& schema,
    bool any_whitespace = true,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true,
    std::optional<int> max_whitespace_cnt = std::nullopt,
    JSONFormat json_format = JSONFormat::kJSON
);

/*!
 * \brief Convert JSON schema string to EBNF grammar string.
 * \param schema The JSON schema object.
 * \param any_whitespace Whether to ignore the indentation restrictions, and allow any whitespace.
 * Default: true.
 * \param indent The number of spaces for indentation. If set to std::nullopt, the output will be
 * in one line. Default: 2.
 * \param separators Two separators used in the schema: comma and colon. Examples: {",", ":"},
 * {", ", ": "}. If std::nullopt, the default separators will be used: {",", ": "} when the
 * indent is not -1, and {", ", ": "} otherwise. This follows the convention in python
 * json.dumps(). Default: std::nullopt.
 * \param strict_mode Whether to use strict mode. In strict
 * mode, the generated grammar will not allow properties and items that is not specified in the
 * schema. This is equivalent to setting unevaluatedProperties and unevaluatedItems to false.
 * This helps LLM to generate accurate output in the grammar-guided generation with JSON
 * schema. Default: true.
 * \param max_whitespace_cnt The maximum number of whitespace characters for the whitespace
 * which is used for indentation or JSON elements separation when any_whitespace is True. If
 * std::nullopt, it means unlimited. Default: std::nullopt.
 * \param json_format Define the root format of the object. If it's JSONFormat::kJSON,
 * then it will generate a fully JSON-style grammar. If it's JSONFormat::kXML, then it will
 * generate a grammar with the root format is XML-style, while the inner format is JSON-style.
 * Default: JSONFormat::kJSON.
 * \returns The EBNF grammar string.
 */
std::string JSONSchemaToEBNF(
    const picojson::value& schema,
    bool any_whitespace = true,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true,
    std::optional<int> max_whitespace_cnt = std::nullopt,
    JSONFormat json_format = JSONFormat::kJSON
);

/*!
 * \brief Generate regex pattern for integer/float range.
 * \param start The start of the range (inclusive). If null assume negative infinity.
 * \param end The end of the range (inclusive). If null assume infinity.
 * \returns The regex pattern that matches integers/floats in the given range.
 */
std::string GenerateRangeRegex(std::optional<int64_t> start, std::optional<int64_t> end);

std::string GenerateFloatRangeRegex(std::optional<double> start, std::optional<double> end);

/*!
 * \brief Convert a function call to a Grammar.
 * \param schema The schema of the parameters of the function call.
 * \return The ebnf-grammar to match the requirements of the schema, and
 * in Qwen xml style.
 */
std::string QwenXMLToolCallingToEBNF(const std::string& schema);

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_H_
