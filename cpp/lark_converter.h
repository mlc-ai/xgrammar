/*!
 *  Copyright (c) 2026 by Contributors
 * \file xgrammar/lark_converter.h
 * \brief Convert LLGuidance-compatible Lark syntax to XGrammar Grammar IR.
 */

#ifndef XGRAMMAR_LARK_CONVERTER_H_
#define XGRAMMAR_LARK_CONVERTER_H_

#include <xgrammar/grammar.h>

#include <optional>
#include <string>

namespace xgrammar {

Grammar LarkToGrammar(
    const std::string& lark_string,
    const std::optional<TokenizerInfo>& tokenizer_info = std::nullopt
);

}  // namespace xgrammar

#endif  // XGRAMMAR_LARK_CONVERTER_H_
