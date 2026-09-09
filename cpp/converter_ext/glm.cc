/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/glm.cc
 * \brief Implementation of the Glm XML Tool Calling converter.
 */
#include "../json_schema_converter_ext.h"

namespace xgrammar {

GlmXMLToolCallingConverter::GlmXMLToolCallingConverter(
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
          {"<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"},
          any_order
      ) {}

}  // namespace xgrammar
