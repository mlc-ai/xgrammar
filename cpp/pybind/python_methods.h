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

TokenizerInfo TokenizerInfo_Init(
    const std::vector<std::string>& vocab,
    std::string vocab_type,
    bool prepend_space_in_tokenization
);

std::string TokenizerInfo_GetVocabType(const TokenizerInfo& tokenizer);

std::vector<pybind11::bytes> TokenizerInfo_GetRawVocab(TokenizerInfo& tokenizer);

torch::Tensor GrammarMatcher_FindNextTokenBitmask(GrammarMatcher& matcher);

std::vector<int> GrammarMatcher_GetRejectedTokensFromBitMask(
    torch::Tensor token_bitmask, size_t vocab_size
);

}  // namespace xgrammar

#endif  // XGRAMMAR_DEBUG_METHODS_H_
