/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/support/unicode_regex_property.h
 * \brief Unicode 16.0.0 property lookup compatible with regex-syntax 0.8.5.
 */

#ifndef XGRAMMAR_SUPPORT_UNICODE_REGEX_PROPERTY_H_
#define XGRAMMAR_SUPPORT_UNICODE_REGEX_PROPERTY_H_

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace xgrammar {

struct UnicodeRegexPropertyRange {
  uint32_t first;
  uint32_t last;
};

bool LookupUnicodeRegexProperty(
    std::string_view query, const UnicodeRegexPropertyRange** ranges, size_t* range_count
);

}  // namespace xgrammar

#endif  // XGRAMMAR_SUPPORT_UNICODE_REGEX_PROPERTY_H_
