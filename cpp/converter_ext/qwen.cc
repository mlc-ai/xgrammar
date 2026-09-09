/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/qwen.cc
 * \brief Implementation of the Qwen XML Tool Calling converter.
 */
#include "../json_schema_converter_ext.h"

namespace xgrammar {

QwenXMLToolCallingConverter::QwenXMLToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    bool any_order
)
    : XMLToolCallingConverter(
          indent,
          separators,
          any_whitespace,
          max_whitespace_cnt,
          ref_resolver,
          {"<parameter=", ">", "", "</parameter>"},
          any_order
      ) {}

}  // namespace xgrammar
