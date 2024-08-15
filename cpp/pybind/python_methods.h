/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef XGRAMMAR_DEBUG_METHODS_H_
#define XGRAMMAR_DEBUG_METHODS_H_

// #include <torch/extension.h>
#include <xgrammar/xgrammar.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace xgrammar {

BNFGrammar BNFGrammar_InitNoNormalization(
    const std::string& ebnf_string, const std::string& main_rule
);

GrammarStateMatcher GrammarStateMatcher_Init(
    const BNFGrammar& grammar, const std::vector<std::string>& token_table, int max_rollback_steps
);

GrammarStateMatcher GrammarStateMatcher_Init(
    const BNFGrammar& grammar, std::nullptr_t, int max_rollback_steps
);

GrammarStateMatcher GrammarStateMatcher_Init(
    const BNFGrammar& grammar,
    const std::unordered_map<std::string, int>& token_table,
    int max_rollback_steps
);

// torch::Tensor GrammarStateMatcher_FindNextTokenBitmask(GrammarStateMatcher& matcher);

}  // namespace xgrammar

#endif  // XGRAMMAR_DEBUG_METHODS_H_
