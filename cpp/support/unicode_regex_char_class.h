/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/support/unicode_regex_char_class.h
 * \brief Unicode character ranges used by Rust-compatible regular-expression shorthands.
 */
#ifndef XGRAMMAR_SUPPORT_UNICODE_REGEX_CHAR_CLASS_H_
#define XGRAMMAR_SUPPORT_UNICODE_REGEX_CHAR_CLASS_H_

#include <cstdint>
#include <utility>
#include <vector>

namespace xgrammar {

using UnicodeRegexRange = std::pair<uint32_t, uint32_t>;

/*! \brief Append the Unicode Decimal_Number ranges used by \\d. */
void AppendUnicodeRegexDecimalRanges(std::vector<UnicodeRegexRange>* ranges);

/*! \brief Append the Unicode Perl word ranges used by \\w. */
void AppendUnicodeRegexWordRanges(std::vector<UnicodeRegexRange>* ranges);

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_UNICODE_REGEX_CHAR_CLASS_H_
