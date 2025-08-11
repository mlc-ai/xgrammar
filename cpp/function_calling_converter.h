/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/function_calling_converter.h
 * \brief The header for converting function calls to Grammars.
 */
#ifndef XGRAMMAR_FUNCTION_CALLING_CONVERTER_H_
#define XGRAMMAR_FUNCTION_CALLING_CONVERTER_H_
#include <xgrammar/grammar.h>

namespace xgrammar {
/*!
 * \brief Convert a function call to a Grammar.
 * \param schema The schema of the parameters of the function call.
 * \return The ebnf-grammar to match the requirements of the schema, and
 * in Qwen xml style.
 */
std::string QwenXMLToolCallingToEbnf(const std::string& schema);

}  // namespace xgrammar

#endif  // XGRAMMAR_FUNCTION_CALL_CONVERTER_H_
