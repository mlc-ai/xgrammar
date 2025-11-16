/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/nanobind/python_methods.cc
 */

#include "python_methods.h"

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <variant>
#include <vector>

#include "../grammar_impl.h"
#include "../support/logging.h"
#include "../support/utils.h"
#include "xgrammar/exception.h"

namespace xgrammar {

TokenizerInfo TokenizerInfo_Init(
    const std::vector<std::string>& encoded_vocab,
    int vocab_type,
    std::optional<int> vocab_size,
    std::optional<std::vector<int32_t>> stop_token_ids,
    bool add_prefix_space
) {
  XGRAMMAR_CHECK(vocab_type == 0 || vocab_type == 1 || vocab_type == 2)
      << "Invalid vocab type: " << vocab_type;
  return TokenizerInfo(
      encoded_vocab,
      static_cast<VocabType>(vocab_type),
      vocab_size,
      stop_token_ids,
      add_prefix_space
  );
}

int TokenizerInfo_GetVocabType(const TokenizerInfo& tokenizer) {
  return static_cast<int>(tokenizer.GetVocabType());
}

std::vector<int> Testing_DebugGetMaskedTokensFromBitmask(
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

std::pair<bool, int> Testing_IsSingleTokenBitmask(
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

  return _IsSingleTokenBitmask(bitmask_dltensor, vocab_size, index);
}

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
) {
  std::array<int64_t, 2> logits_shape_arr = {logits_shape.first, logits_shape.second};
  std::array<int64_t, 2> logits_strides_arr = {logits_strides.first, logits_strides.second};
  std::array<int64_t, 2> bitmask_shape_arr = {bitmask_shape.first, bitmask_shape.second};
  std::array<int64_t, 2> bitmask_strides_arr = {bitmask_strides.first, bitmask_strides.second};

  DLDataType logit_dtype;
  if (logit_type == "float32") {
    logit_dtype = DLDataType{kDLFloat, 32, 1};
  } else if (logit_type == "float16") {
    logit_dtype = DLDataType{kDLFloat, 16, 1};
  } else if (logit_type == "bfloat16") {
    logit_dtype = DLDataType{kDLBfloat, 16, 1};
  } else {
    XGRAMMAR_LOG(FATAL) << "Unsupported logit type: " << logit_type;
  }

  DLTensor logits_dltensor{
      reinterpret_cast<void*>(logits_ptr),
      DLDevice{kDLCPU, 0},
      2,
      logit_dtype,
      logits_shape_arr.data(),
      logits_strides_arr.data(),
      0
  };

  DLTensor bitmask_dltensor{
      reinterpret_cast<void*>(bitmask_ptr),
      DLDevice{kDLCPU, 0},
      2,
      GetBitmaskDLType(),
      bitmask_shape_arr.data(),
      bitmask_strides_arr.data(),
      0
  };

  ApplyTokenBitmaskInplaceCPU(&logits_dltensor, bitmask_dltensor, vocab_size, indices);
}

std::vector<int32_t> GetAllowEmptyRuleIds(const CompiledGrammar& compiled_grammar) {
  return compiled_grammar.GetGrammar()->allow_empty_rule_ids;
}

Grammar Grammar_FromStructuralTag(const std::string& structural_tag_json) {
  auto result = Grammar::FromStructuralTag(structural_tag_json);
  if (std::holds_alternative<StructuralTagError>(result)) {
    ThrowVariantError(std::get<StructuralTagError>(result));
  }
  return std::get<Grammar>(result);
}

Grammar Grammar_DeserializeJSON(const std::string& json_string) {
  auto result = Grammar::DeserializeJSON(json_string);
  if (std::holds_alternative<SerializationError>(result)) {
    ThrowVariantError(std::get<SerializationError>(result));
  }
  return std::get<Grammar>(result);
}

TokenizerInfo TokenizerInfo_DeserializeJSON(const std::string& json_string) {
  auto result = TokenizerInfo::DeserializeJSON(json_string);
  if (std::holds_alternative<SerializationError>(result)) {
    ThrowVariantError(std::get<SerializationError>(result));
  }
  return std::get<TokenizerInfo>(result);
}

CompiledGrammar CompiledGrammar_DeserializeJSON(
    const std::string& json_string, const TokenizerInfo& tokenizer
) {
  auto result = CompiledGrammar::DeserializeJSON(json_string, tokenizer);
  if (std::holds_alternative<SerializationError>(result)) {
    ThrowVariantError(std::get<SerializationError>(result));
  }
  return std::get<CompiledGrammar>(result);
}

}  // namespace xgrammar
