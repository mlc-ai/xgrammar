/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/pybind/python_methods.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef XGRAMMAR_PYBIND_PYTHON_METHODS_H_
#define XGRAMMAR_PYBIND_PYTHON_METHODS_H_

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <xgrammar/xgrammar.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace xgrammar {

TokenizerInfo TokenizerInfo_Init(
    const std::vector<std::string>& encoded_vocab,
    std::string vocab_type,
    std::optional<int> vocab_size,
    std::optional<std::vector<int32_t>> stop_token_ids,
    bool prepend_space_in_tokenization
);

std::string TokenizerInfo_GetVocabType(const TokenizerInfo& tokenizer);

std::vector<pybind11::bytes> TokenizerInfo_GetDecodedVocab(const TokenizerInfo& tokenizer);

void GrammarMatcher_FillNextTokenBitmask(
    GrammarMatcher& matcher, torch::Tensor token_bitmask, int index
);

std::vector<int> Matcher_DebugGetMaskedTokensFromBitmask(
    torch::Tensor token_bitmask, int32_t vocab_size, int32_t index
);

#ifdef XGRAMMAR_BUILD_KERNELS
void Matcher_ApplyTokenBitmaskInplace(
    torch::Tensor logits,
    torch::Tensor token_bitmask,
    std::optional<std::vector<int>> indices = std::nullopt
);
#endif

}  // namespace xgrammar

#endif  // XGRAMMAR_PYBIND_PYTHON_METHODS_H_
