/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/regex_converter.h
 * \brief Convert a regex string to EBNF grammar string.
 */

#ifndef XGRAMMAR_REGEX_CONVERTER_H_
#define XGRAMMAR_REGEX_CONVERTER_H_

#include <xgrammar/xgrammar.h>

#include <string>

namespace xgrammar {

/*!
 * \brief Convert a regex string to EBNF grammar string.
 */
class RegexConverter {
 public:
  explicit RegexConverter(const std::string& regex);
  std::string Convert();

  XGRAMMAR_DEFINE_PIMPL_METHODS(RegexConverter);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_REGEX_CONVERTER_H_
