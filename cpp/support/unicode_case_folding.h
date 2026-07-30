/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/support/unicode_case_folding.h
 * \brief Unicode simple case-folding support.
 */
#ifndef XGRAMMAR_SUPPORT_UNICODE_CASE_FOLDING_H_
#define XGRAMMAR_SUPPORT_UNICODE_CASE_FOLDING_H_

#include <vector>

#include "encoding.h"

namespace xgrammar {

/*!
 * \brief Append all simple case-fold equivalents of codepoints in an inclusive range.
 *
 * The original codepoints are not appended.
 */
void AppendUnicodeSimpleCaseFold(
    TCodepoint lower, TCodepoint upper, std::vector<TCodepoint>* output
);

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_UNICODE_CASE_FOLDING_H_
