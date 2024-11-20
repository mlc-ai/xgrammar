/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.cc
 */

#include "python_methods.h"

#include <ATen/DLConvertor.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <chrono>
#include <iostream>

#include "../support/dynamic_bitset.h"
#include "../support/logging.h"

#ifdef XGRAMMAR_BUILD_KERNELS
#include "../kernels/kernels.h"
#endif

namespace xgrammar {

TokenizerInfo TokenizerInfo_Init(
    const std::vector<std::string>& encoded_vocab,
    std::string vocab_type,
    std::optional<int> vocab_size,
    std::optional<std::vector<int32_t>> stop_token_ids,
    bool prepend_space_in_tokenization
) {
  const std::unordered_map<std::string, VocabType> VOCAB_TYPE_MAP = {
      {"RAW", VocabType::RAW},
      {"BYTE_FALLBACK", VocabType::BYTE_FALLBACK},
      {"BYTE_LEVEL", VocabType::BYTE_LEVEL},
  };
  XGRAMMAR_CHECK(VOCAB_TYPE_MAP.count(vocab_type)) << "Invalid vocab type: " << vocab_type;
  return TokenizerInfo(
      encoded_vocab,
      VOCAB_TYPE_MAP.at(vocab_type),
      vocab_size,
      stop_token_ids,
      prepend_space_in_tokenization
  );
}

std::string TokenizerInfo_GetVocabType(const TokenizerInfo& tokenizer) {
  const std::string VOCAB_TYPE_NAMES[] = {"RAW", "BYTE_FALLBACK", "BYTE_LEVEL"};
  return VOCAB_TYPE_NAMES[static_cast<int>(tokenizer.GetVocabType())];
}

std::vector<pybind11::bytes> TokenizerInfo_GetDecodedVocab(const TokenizerInfo& tokenizer) {
  const auto& decoded_vocab = tokenizer.GetDecodedVocab();
  std::vector<pybind11::bytes> py_result;
  py_result.reserve(decoded_vocab.size());
  for (const auto& item : decoded_vocab) {
    py_result.emplace_back(pybind11::bytes(item));
  }
  return py_result;
}

void GrammarMatcher_FillNextTokenBitmask(
    GrammarMatcher& matcher, torch::Tensor token_bitmask, int32_t index
) {
  torch::IntArrayRef shape = token_bitmask.sizes();

  XGRAMMAR_CHECK(shape.size() == 1 || shape.size() == 2) << "token_bitmask tensor must be 1D or 2D";
  XGRAMMAR_CHECK(token_bitmask.dtype() == torch::kInt32)
      << "token_bitmask tensor must be of type int32";
  XGRAMMAR_CHECK(token_bitmask.device().type() == torch::kCPU)
      << "token_bitmask tensor must be on CPU";

  int64_t dltensor_shape[2] = {shape[0]};
  if (shape.size() == 2) {
    dltensor_shape[1] = shape[1];
  }

  DLTensor bitmask_dltensor{
      token_bitmask.data_ptr<int32_t>(),
      DLDevice{kDLCPU, 0},
      static_cast<int32_t>(shape.size()),
      GetBitmaskDLType(),
      dltensor_shape,
      nullptr,
      0
  };
  matcher.FillNextTokenBitmask(&bitmask_dltensor);
}

std::vector<int> Matcher_DebugGetMaskedTokensFromBitmask(
    torch::Tensor token_bitmask, int32_t vocab_size, int32_t index
) {
  torch::IntArrayRef shape = token_bitmask.sizes();

  XGRAMMAR_CHECK(shape.size() == 1 || shape.size() == 2) << "token_bitmask tensor must be 1D or 2D";
  XGRAMMAR_CHECK(token_bitmask.dtype() == torch::kInt32)
      << "token_bitmask tensor must be of type int32";
  XGRAMMAR_CHECK(token_bitmask.device().type() == torch::kCPU)
      << "token_bitmask tensor must be on CPU";

  int64_t dltensor_shape[2] = {shape[0]};
  if (shape.size() == 2) {
    dltensor_shape[1] = shape[1];
  }

  DLTensor bitmask_dltensor{
      token_bitmask.data_ptr<int32_t>(),
      DLDevice{kDLCPU, 0},
      static_cast<int32_t>(shape.size()),
      GetBitmaskDLType(),
      dltensor_shape,
      nullptr,
      0
  };

  std::vector<int> result;
  _DebugGetMaskedTokensFromBitmask(&result, bitmask_dltensor, vocab_size, index);
  return result;
}

#ifdef XGRAMMAR_BUILD_KERNELS
void Matcher_ApplyTokenBitmaskInplace(
    torch::Tensor logits, torch::Tensor token_bitmask, std::optional<std::vector<int>> indices
) {
  auto logits_shape = logits.sizes();
  int batch_size = 1;
  int vocab_size;
  if (logits_shape.size() == 1) {
    vocab_size = logits_shape[0];
  } else if (logits_shape.size() == 2) {
    batch_size = logits_shape[0];
    vocab_size = logits_shape[1];
  } else {
    XGRAMMAR_LOG(FATAL) << "logits tensor must be 1D or 2D";
  }

  auto bitmask_shape = token_bitmask.sizes();
  int expected_bitmask_size = DynamicBitset::GetBufferSize(vocab_size);
  if (bitmask_shape.size() == 1) {
    XGRAMMAR_CHECK(bitmask_shape[0] == expected_bitmask_size)
        << "The last dimension of the token bitmask tensor must be " << expected_bitmask_size
        << ", but got " << bitmask_shape[0];
  } else if (bitmask_shape.size() == 2) {
    XGRAMMAR_CHECK(bitmask_shape[0] == batch_size)
        << "The first dimension of the token bitmask tensor must be " << batch_size << ", but got "
        << bitmask_shape[0];
    XGRAMMAR_CHECK(bitmask_shape[1] == expected_bitmask_size)
        << "The last dimension of the token bitmask tensor must be " << expected_bitmask_size
        << ", but got " << bitmask_shape[1];
  } else {
    XGRAMMAR_LOG(FATAL) << "token_bitmask tensor must be 1D or 2D";
  }

  DTypeFlag dtype_flag;
  if (logits.dtype() == torch::kFloat16) {
    dtype_flag = DTypeFlag::DTYPE_FLOAT16;
  } else if (logits.dtype() == torch::kFloat32) {
    dtype_flag = DTypeFlag::DTYPE_FLOAT32;
  } else if (logits.dtype() == torch::kFloat64) {
    dtype_flag = DTypeFlag::DTYPE_FLOAT64;
  } else {
    XGRAMMAR_LOG(FATAL) << "logits tensor must be of type float16, float32, or float64";
  }

  XGRAMMAR_CHECK(token_bitmask.dtype() == torch::kInt32)
      << "token bitmask tensor must be of type int32";

  if (!indices) {
    indices = std::vector<int>(batch_size);
    std::iota(indices->begin(), indices->end(), 0);
  }

  ApplyTokenBitmaskInplace(
      logits.data_ptr(),
      dtype_flag,
      token_bitmask.data_ptr<int32_t>(),
      batch_size,
      vocab_size,
      indices
  );
}
#endif

}  // namespace xgrammar
