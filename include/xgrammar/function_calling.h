/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/function_calling.h
 * \brief Header for parsing function calls in both tag-based and JSON formats.
 */

#ifndef XGRAMMAR_FUNCTION_CALLING_H_
#define XGRAMMAR_FUNCTION_CALLING_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace xgrammar {

/*!
 * \brief Parse a message containing function calls in either xml style tag-based or JSON format.
 * \param input The input string containing function calls
 * \param ignore_error Whether to ignore parsing errors (default: false)
 * \return A vector of pairs, each containing a function name and its parameters
 * \note Supports two formats:
 *       1. Tag-based: <function=name>{"param": "value"}</function>
 *       2. JSON: {"name": "function_name", "parameters": {"param": "value"}}
 */
std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> parse_message(
    const std::string& input, bool ignore_error = false
);

}  // namespace xgrammar

#endif  // XGRAMMAR_FUNCTION_CALLING_H_
