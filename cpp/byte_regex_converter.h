/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/byte_regex_converter.h
 * \brief Convert a byte-oriented regular expression into grammar expressions.
 */

#ifndef XGRAMMAR_BYTE_REGEX_CONVERTER_H_
#define XGRAMMAR_BYTE_REGEX_CONVERTER_H_

#include <cstdint>
#include <string>

namespace xgrammar {

class GrammarBuilder;

/*!
 * \brief Convert a regular expression whose atoms operate on individual bytes.
 * \param builder The grammar builder receiving the generated expressions and helper rules.
 * \param pattern The regular-expression pattern.
 * \param rule_hint A name hint for generated repetition rules.
 * \return The generated grammar expression id.
 */
int32_t ByteRegexToGrammarExpr(
    GrammarBuilder* builder, const std::string& pattern, const std::string& rule_hint
);

}  // namespace xgrammar

#endif  // XGRAMMAR_BYTE_REGEX_CONVERTER_H_
