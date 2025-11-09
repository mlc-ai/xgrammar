/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/nanobind/python_methods.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef XGRAMMAR_NANOBIND_PYTHON_METHODS_H_
#define XGRAMMAR_NANOBIND_PYTHON_METHODS_H_

#include <xgrammar/xgrammar.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "xgrammar/tokenizer_info.h"

namespace xgrammar {

TokenizerInfo TokenizerInfo_Init(
    const std::vector<std::string>& encoded_vocab,
    int vocab_type,
    std::optional<int> vocab_size,
    std::optional<std::vector<int32_t>> stop_token_ids,
    bool add_prefix_space
);

int TokenizerInfo_GetVocabType(const TokenizerInfo& tokenizer);

std::vector<int> Testing_DebugGetMaskedTokensFromBitmask(
    intptr_t token_bitmask_ptr, std::vector<int64_t> shape, int32_t vocab_size, int32_t index
);

std::pair<bool, int> Testing_IsSingleTokenBitmask(
    intptr_t token_bitmask_ptr, std::vector<int64_t> shape, int32_t vocab_size, int32_t index
);

void Kernels_ApplyTokenBitmaskInplaceCPU(
    intptr_t logits_ptr,
    std::pair<int64_t, int64_t> logits_shape,
    std::pair<int64_t, int64_t> logits_strides,
    intptr_t bitmask_ptr,
    std::pair<int64_t, int64_t> bitmask_shape,
    std::pair<int64_t, int64_t> bitmask_strides,
    int vocab_size,
    std::optional<std::vector<int>> indices,
    std::string logit_type
);

std::vector<int32_t> GetAllowEmptyRuleIds(const CompiledGrammar& compiled_grammar);

Grammar Grammar_FromStructuralTag(const std::string& structural_tag_json);

Grammar Grammar_DeserializeJSON(const std::string& json_string);

TokenizerInfo TokenizerInfo_DeserializeJSON(const std::string& json_string);

CompiledGrammar CompiledGrammar_DeserializeJSON(
    const std::string& json_string, const TokenizerInfo& tokenizer
);

}  // namespace xgrammar

#endif  // XGRAMMAR_NANOBIND_PYTHON_METHODS_H_
