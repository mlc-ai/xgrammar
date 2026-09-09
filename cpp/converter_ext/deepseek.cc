/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/deepseek.cc
 * \brief Implementation of the DeepSeek XML Tool Calling converter.
 */
#include "../json_schema_converter_ext.h"

namespace xgrammar {

DeepSeekXMLToolCallingConverter::DeepSeekXMLToolCallingConverter(
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
          {"<｜DSML｜parameter name=\"", "", "", "</｜DSML｜parameter>"},
          any_order
      ) {}

int32_t DeepSeekXMLToolCallingConverter::XMLKeySuffix(const SchemaSpecPtr& schema) {
  // TODO(Linzhang): We do not validate the string's value, and we accept both.
  return Sequence(
      {ByteString("\" string=\""),
       Choice({ByteString("true"), ByteString("false")}),
       ByteString("\">")}
  );
}

}  // namespace xgrammar
