/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/function_call_converter.h
 * \brief The header for converting function calls to Grammars.
 */
#ifndef XGRAMMAR_FUNCTION_CALL_CONVERTER_H_
#define XGRAMMAR_FUNCTION_CALL_CONVERTER_H_
#include <xgrammar/grammar.h>

namespace xgrammar {
class FunctionCallConverter {
 public:
  /*!
   * \brief Convert a function call to a Grammar.
   * \param args_names The names of the arguments.
   * \param args_types The types of the arguments.
   * \param function_type The type of the function call format. Default is kXmlStyleFunctionCall.
   * \return The constructed grammar.
   */
  static Grammar Apply(
      const std::vector<std::string>& args_names,
      const std::vector<std::string>& args_types,
      uint8_t function_type
  );
};
}  // namespace xgrammar

#endif  // XGRAMMAR_FUNCTION_CALL_CONVERTER_H_
