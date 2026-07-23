/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/regex_converter.h
 * \brief Convert a regex string to EBNF grammar string.
 */

#ifndef XGRAMMAR_REGEX_CONVERTER_H_
#define XGRAMMAR_REGEX_CONVERTER_H_

#include <xgrammar/grammar.h>

#include <string>

namespace xgrammar {

/*!
 * \brief Convert a regex string to EBNF grammar string.
 */
std::string RegexToEBNF(const std::string& regex, bool with_rule_name = true);

/*!
 * \brief Convert a regex directly to a grammar AST without creating or parsing EBNF text.
 */
Grammar RegexToGrammar(const std::string& regex);

}  // namespace xgrammar

#endif  // XGRAMMAR_REGEX_CONVERTER_H_
