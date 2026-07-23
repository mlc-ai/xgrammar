/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.cc
 * \brief Extended JSON Schema grammar formats.
 */

#include "json_schema_converter_ext.h"

#include <utility>

namespace xgrammar {

XMLToolCallingConverter::XMLToolCallingConverter(
    std::optional<int> indent,
    std::optional<std::pair<std::string, std::string>> separators,
    bool any_whitespace,
    std::optional<int> max_whitespace_cnt,
    RefResolver ref_resolver,
    JSONFormat json_format,
    bool any_order
)
    : JSONSchemaConverter(
          indent,
          std::move(separators),
          any_whitespace,
          max_whitespace_cnt,
          std::move(ref_resolver),
          any_order,
          json_format
      ) {}

}  // namespace xgrammar
