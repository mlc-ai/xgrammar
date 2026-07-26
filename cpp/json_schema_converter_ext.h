/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/json_schema_converter_ext.h
 * \brief Extended JSON Schema grammar formats.
 */

#ifndef XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
#define XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_

#include <utility>

#include "json_schema_converter.h"

namespace xgrammar {

/*!
 * \brief Direct AST converter for XML tool-calling formats.
 *
 * The format-specific AST generation lives in JSONSchemaConverter so JSON and XML variants share
 * one rule builder, reference cache, and normalization path.
 */
class XMLToolCallingConverter : public JSONSchemaConverter {
 public:
  XMLToolCallingConverter(
      std::optional<int> indent,
      std::optional<std::pair<std::string, std::string>> separators,
      bool any_whitespace,
      std::optional<int> max_whitespace_cnt,
      RefResolver ref_resolver = nullptr,
      JSONFormat json_format = JSONFormat::kQwenXML,
      bool any_order = false
  );
};

}  // namespace xgrammar

#endif  // XGRAMMAR_JSON_SCHEMA_CONVERTER_EXT_H_
