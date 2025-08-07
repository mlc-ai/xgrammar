/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/function_call_converter.cc
 * \brief The implementation for converting function calls to Grammars.
 */

#include "function_call_converter.h"

#include "support/utils.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

class FunctionCallConverterImpl {
 public:
  static Grammar Apply(
      const std::string& function_name,
      const std::vector<std::string>& args_names,
      const std::vector<std::string>& args_types,
      uint8_t function_type
  ) {
    // TODO(Linzhang): Implement the function call converter.
    XGRAMMAR_UNREACHABLE();
  }
};

/*************************** Forward grammar functors to their impl ***************************/

Grammar FunctionCallConverter::Apply(
    const std::string& function_name,
    const std::vector<std::string>& args_names,
    const std::vector<std::string>& args_types,
    uint8_t function_type
) {
  return FunctionCallConverterImpl::Apply(function_name, args_names, args_types, function_type);
}

}  // namespace xgrammar
