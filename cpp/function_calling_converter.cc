/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/function_calling_converter.cc
 * \brief The implementation for converting function calls to Grammars.
 */

#include "function_calling_converter.h"

#include <optional>

#include "json_schema_converter.h"
#include "support/logging.h"
#include "support/utils.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

std::string QwenXMLToolCallingToEbnf(const std::string& schema) {
  // Convert the schema string to picojson value.
  picojson::value json_value;
  std::string err = picojson::parse(json_value, schema);
  if (!err.empty()) {
    XGRAMMAR_LOG(FATAL) << "Failed to parse JSON schema: " << err;
  }
  if (json_value.is<bool>()) {
    XGRAMMAR_LOG(FATAL) << "Expected JSON schema object, got boolean: " << json_value.to_str();
  }
  const auto& schema_obj = json_value.get<picojson::object>();
  if (!schema_obj.count("type")) {
    XGRAMMAR_LOG(FATAL) << "Function calling must have a 'type' field of 'object': "
                        << json_value.to_str();
  }
  if (schema_obj.at("type").get<std::string>() != "object") {
    XGRAMMAR_LOG(FATAL) << "Function calling must have a 'type' field of 'object': "
                        << json_value.to_str();
  }
  return JSONSchemaToEBNF(
      json_value, true, std::nullopt, std::nullopt, true, StringEscapeType::kXML
  );
}

}  // namespace xgrammar
