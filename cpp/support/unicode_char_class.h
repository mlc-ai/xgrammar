/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/support/unicode_char_class.h
 * \brief Locale-independent Unicode character classification.
 */
#ifndef XGRAMMAR_SUPPORT_UNICODE_CHAR_CLASS_H_
#define XGRAMMAR_SUPPORT_UNICODE_CHAR_CLASS_H_

#include "encoding.h"

namespace xgrammar {

/*! \brief Return whether a codepoint has the Unicode White_Space property. */
bool IsUnicodeWhitespace(TCodepoint codepoint);

/*! \brief Return whether a codepoint has the Unicode Alphabetic property. */
bool IsUnicodeAlphabetic(TCodepoint codepoint);

/*! \brief Return whether a codepoint is Unicode alphabetic or numeric. */
bool IsUnicodeAlphanumeric(TCodepoint codepoint);

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_UNICODE_CHAR_CLASS_H_
