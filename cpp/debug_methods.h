/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef XGRAMMAR_DEBUG_METHODS_H_
#define XGRAMMAR_DEBUG_METHODS_H_

#include <xgrammar/xgrammar.h>

#include <string>
#include <vector>

namespace xgrammar {

BNFGrammar BNFGrammar_InitNoNormalization(
    const std::string& ebnf_string, const std::string& main_rule
);

}  // namespace xgrammar

#endif  // XGRAMMAR_DEBUG_METHODS_H_
