/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/nanobind/python_methods.cc
 */

#include "python_methods.h"

#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "../grammar_data_structure.h"
#include "../support/dynamic_bitset.h"
#include "../support/logging.h"

namespace xgrammar {

TokenizerInfo TokenizerInfo_Init(
    const std::vector<std::string>& encoded_vocab,
    std::string vocab_type,
    std::optional<int> vocab_size,
    std::optional<std::vector<int32_t>> stop_token_ids,
    bool add_prefix_space
) {
  const std::unordered_map<std::string, VocabType> VOCAB_TYPE_MAP = {
      {"RAW", VocabType::RAW},
      {"BYTE_FALLBACK", VocabType::BYTE_FALLBACK},
      {"BYTE_LEVEL", VocabType::BYTE_LEVEL},
  };
  XGRAMMAR_CHECK(VOCAB_TYPE_MAP.count(vocab_type)) << "Invalid vocab type: " << vocab_type;
  return TokenizerInfo(
      encoded_vocab, VOCAB_TYPE_MAP.at(vocab_type), vocab_size, stop_token_ids, add_prefix_space
  );
}

std::string TokenizerInfo_GetVocabType(const TokenizerInfo& tokenizer) {
  const std::string VOCAB_TYPE_NAMES[] = {"RAW", "BYTE_FALLBACK", "BYTE_LEVEL"};
  return VOCAB_TYPE_NAMES[static_cast<int>(tokenizer.GetVocabType())];
}

bool GrammarMatcher_FillNextTokenBitmask(
    GrammarMatcher& matcher,
    intptr_t token_bitmask_ptr,
    std::vector<int64_t> shape,
    int32_t index,
    bool debug_print
) {
  XGRAMMAR_CHECK(shape.size() == 1 || shape.size() == 2) << "token_bitmask tensor must be 1D or 2D";

  DLTensor bitmask_dltensor{
      reinterpret_cast<void*>(token_bitmask_ptr),
      DLDevice{kDLCPU, 0},
      static_cast<int32_t>(shape.size()),
      GetBitmaskDLType(),
      shape.data(),
      nullptr,
      0
  };
  return matcher.FillNextTokenBitmask(&bitmask_dltensor, index, debug_print);
}

std::vector<int> Matcher_DebugGetMaskedTokensFromBitmask(
    intptr_t token_bitmask_ptr, std::vector<int64_t> shape, int32_t vocab_size, int32_t index
) {
  XGRAMMAR_CHECK(shape.size() == 1 || shape.size() == 2) << "token_bitmask tensor must be 1D or 2D";

  DLTensor bitmask_dltensor{
      reinterpret_cast<void*>(token_bitmask_ptr),
      DLDevice{kDLCPU, 0},
      static_cast<int32_t>(shape.size()),
      GetBitmaskDLType(),
      shape.data(),
      nullptr,
      0
  };

  std::vector<int> result;
  _DebugGetMaskedTokensFromBitmask(&result, bitmask_dltensor, vocab_size, index);
  return result;
}

void Kernels_ApplyTokenBitmaskInplaceCPU(
    intptr_t logits_ptr,
    std::pair<int64_t, int64_t> logits_shape,
    intptr_t bitmask_ptr,
    std::pair<int64_t, int64_t> bitmask_shape,
    std::optional<std::vector<int>> indices
) {
  std::array<int64_t, 2> logits_shape_arr = {logits_shape.first, logits_shape.second};
  std::array<int64_t, 2> bitmask_shape_arr = {bitmask_shape.first, bitmask_shape.second};

  DLTensor logits_dltensor{
      reinterpret_cast<void*>(logits_ptr),
      DLDevice{kDLCPU, 0},
      2,
      DLDataType{kDLFloat, 32, 1},
      logits_shape_arr.data(),
      nullptr,
      0
  };

  DLTensor bitmask_dltensor{
      reinterpret_cast<void*>(bitmask_ptr),
      DLDevice{kDLCPU, 0},
      2,
      GetBitmaskDLType(),
      bitmask_shape_arr.data(),
      nullptr,
      0
  };

  ApplyTokenBitmaskInplaceCPU(&logits_dltensor, bitmask_dltensor, indices);
}

std::vector<int32_t> GetAllowEmptyRuleIds(const CompiledGrammar& compiled_grammar) {
  return compiled_grammar.GetGrammar()->allow_empty_rule_ids;
}

Grammar Grammar_FromStructuralTag(
    const std::vector<std::tuple<std::string, std::string, std::string>>& tags,
    const std::vector<std::string>& triggers
) {
  std::vector<StructuralTagItem> tags_objects;
  tags_objects.reserve(tags.size());
  for (const auto& tag : tags) {
    tags_objects.emplace_back(
        StructuralTagItem{std::get<0>(tag), std::get<1>(tag), std::get<2>(tag)}
    );
  }
  return Grammar::FromStructuralTag(tags_objects, triggers);
}

CompiledGrammar GrammarCompiler_CompileStructuralTag(
    GrammarCompiler& compiler,
    const std::vector<std::tuple<std::string, std::string, std::string>>& tags,
    const std::vector<std::string>& triggers
) {
  std::vector<StructuralTagItem> tags_objects;
  tags_objects.reserve(tags.size());
  for (const auto& tag : tags) {
    tags_objects.emplace_back(
        StructuralTagItem{std::get<0>(tag), std::get<1>(tag), std::get<2>(tag)}
    );
  }
  return compiler.CompileStructuralTag(tags_objects, triggers);
}

}  // namespace xgrammar
