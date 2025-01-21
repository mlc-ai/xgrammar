/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/pybind/python_methods.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef XGRAMMAR_PYBIND_PYTHON_METHODS_H_
#define XGRAMMAR_PYBIND_PYTHON_METHODS_H_

#include <pybind11/pybind11.h>
#include <xgrammar/xgrammar.h>

#include <optional>
#include <string>
#include <tuple>
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
    GrammarMatcher& matcher,
    intptr_t token_bitmask_ptr,
    std::vector<int64_t> shape,
    int32_t index,
    bool debug_print
);

std::vector<int> Matcher_DebugGetMaskedTokensFromBitmask(
    intptr_t token_bitmask_ptr, std::vector<int64_t> shape, int32_t vocab_size, int32_t index
);

void Kernels_ApplyTokenBitmaskInplaceCPU(
    intptr_t logits_ptr,
    std::pair<int64_t, int64_t> logits_shape,
    intptr_t bitmask_ptr,
    std::pair<int64_t, int64_t> bitmask_shape,
    std::optional<std::vector<int>> indices
);

std::vector<int32_t> GetAllowEmptyRuleIds(const CompiledGrammar& compiled_grammar);

Grammar Grammar_FromStructuralTag(
    const std::vector<std::tuple<std::string, std::string, std::string>>& tags,
    const std::vector<std::string>& triggers
);

}  // namespace xgrammar

#endif  // XGRAMMAR_PYBIND_PYTHON_METHODS_H_
