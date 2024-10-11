/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef XGRAMMAR_DEBUG_METHODS_H_
#define XGRAMMAR_DEBUG_METHODS_H_

#include <pybind11/pybind11.h>
#include <torch/extension.h>
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
    const BNFGrammar& grammar,
    const std::vector<std::string>& vocab,
    std::optional<std::vector<int>> stop_token_ids,
    bool terminate_without_stop_token,
    std::optional<int> mask_vocab_size,
    int max_rollback_steps
);

GrammarStateMatcher GrammarStateMatcher_Init(
    const BNFGrammar& grammar,
    std::nullptr_t,
    std::optional<std::vector<int>> stop_token_ids,
    bool terminate_without_stop_token,
    std::optional<int> mask_vocab_size,
    int max_rollback_steps
);

std::vector<pybind11::bytes> XGTokenizer_GetDecodedVocab(XGTokenizer& tokenizer);

torch::Tensor GrammarStateMatcher_FindNextTokenBitmask(GrammarStateMatcher& matcher);

std::vector<int> GrammarStateMatcher_GetRejectedTokensFromBitMask(
    torch::Tensor token_bitmask, size_t vocab_size
);

}  // namespace xgrammar

#endif  // XGRAMMAR_DEBUG_METHODS_H_
