/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/xgrammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef XGRAMMAR_TESTING_H_
#define XGRAMMAR_TESTING_H_

#include <xgrammar/grammar.h>

#include <optional>
#include <string>
#include <utility>

namespace xgrammar {

/*!
 * \brief Convert JSON schema string to EBNF grammar string.
 * \param json_schema The JSON schema string.
 * \param indent The number of spaces for indentation. If set to std::nullopt, the output will be
 * in one line. Default: 2.
 * \param separators Two separators used in the schema: comma and colon. Examples: {",", ":"},
 * {", ", ": "}. If std::nullopt, the default separators will be used: {",", ": "} when the
 * indent is not -1, and {", ", ": "} otherwise. This follows the convention in python
 * json.dumps(). Default: std::nullopt. \param strict_mode Whether to use strict mode. In strict
 * mode, the generated grammar will not allow properties and items that is not specified in the
 * schema. This is equivalent to setting unevaluatedProperties and unevaluatedItems to false.
 *
 * This helps LLM to generate accurate output in the grammar-guided generation with JSON
 * schema. Default: true.
 * \returns The EBNF grammar string.
 */
std::string _JSONSchemaToEBNF(
    const std::string& schema,
    std::optional<int> indent = std::nullopt,
    std::optional<std::pair<std::string, std::string>> separators = std::nullopt,
    bool strict_mode = true
);

/*!
 * \brief Convert a regex string to EBNF grammar string.
 */
std::string _RegexToEBNF(const std::string& regex, bool with_rule_name = true);

}  // namespace xgrammar

#endif  // XGRAMMAR_TESTING_H_
