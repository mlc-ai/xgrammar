/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.cc
 */

#include "python_methods.h"

#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <chrono>
#include <iostream>

#include "../grammar_parser.h"
#include "../support/dynamic_bitset.h"
#include "../support/logging.h"

namespace xgrammar {

// Parse the EBNF string but not normalize it
BNFGrammar BNFGrammar_InitNoNormalization(
    const std::string& ebnf_string, const std::string& root_rule
) {
  return EBNFParser::Parse(ebnf_string, root_rule);
}

TokenizerInfo TokenizerInfo_Init(
    const std::vector<std::string>& encoded_vocab,
    std::string vocab_type,
    bool prepend_space_in_tokenization
) {
  static const std::unordered_map<std::string, VocabType> VOCAB_TYPE_MAP = {
      {"RAW", VocabType::RAW},
      {"BYTE_FALLBACK", VocabType::BYTE_FALLBACK},
      {"BYTE_LEVEL", VocabType::BYTE_LEVEL},
  };
  return TokenizerInfo(encoded_vocab, VOCAB_TYPE_MAP.at(vocab_type), prepend_space_in_tokenization);
}

std::string TokenizerInfo_GetVocabType(const TokenizerInfo& tokenizer) {
  static const std::string VOCAB_TYPE_NAMES[] = {"RAW", "BYTE_FALLBACK", "BYTE_LEVEL"};
  return VOCAB_TYPE_NAMES[static_cast<int>(tokenizer.GetVocabType())];
}

std::vector<pybind11::bytes> TokenizerInfo_GetDecodedVocab(TokenizerInfo& tokenizer) {
  auto result = tokenizer.GetDecodedVocab();
  std::vector<pybind11::bytes> py_result;
  py_result.reserve(result.size());
  for (const auto& item : result) {
    py_result.emplace_back(pybind11::bytes(item));
  }
  return py_result;
}

void GrammarMatcher_FillNextTokenBitmask(
    GrammarMatcher& matcher, torch::Tensor token_bitmask, int32_t batch_id
) {
  torch::IntArrayRef shape = token_bitmask.sizes();

  // TODO(yixin): Move the checking and slicing logic to core C++ code
  int bitmask_size = matcher.GetBitmaskSize();
  if (shape.size() == 1) {
    batch_id = 0;
    XGRAMMAR_CHECK(shape[0] == bitmask_size)
        << "The last dimension of the token bitmask tensor must be " << bitmask_size << ", but got "
        << shape[0];
  } else if (shape.size() == 2) {
    XGRAMMAR_CHECK(batch_id >= 0 && batch_id < shape[0])
        << "batch_id must be in the range [0, " << shape[0] << ")";
    XGRAMMAR_CHECK(shape[1] == bitmask_size)
        << "The last dimension of the token bitmask tensor must be " << bitmask_size << ", but got "
        << shape[1];
  } else {
    XGRAMMAR_LOG(FATAL) << "token_bitmask tensor must be 1D or 2D";
  }

  XGRAMMAR_CHECK(token_bitmask.dtype() == torch::kInt32)
      << "token_bitmask tensor must be of type int32";
  XGRAMMAR_CHECK(token_bitmask.device().type() == torch::kCPU)
      << "token_bitmask tensor must be on CPU";

  int* data = token_bitmask.data_ptr<int>() + batch_id * bitmask_size;
  int64_t dltensor_shape[] = {bitmask_size};
  DLTensor bitmask_dltensor{
      data, DLDevice{kDLCPU, 0}, 1, DLDataType{kDLInt, 32, 1}, dltensor_shape, nullptr, 0
  };
  matcher.FillNextTokenBitmask(&bitmask_dltensor);
}

std::vector<int> GrammarMatcher_DebugGetMaskedTokensFromBitmask(
    GrammarMatcher& matcher, torch::Tensor token_bitmask, int batch_id
) {
  torch::IntArrayRef shape = token_bitmask.sizes();

  // TODO(yixin): Move the checking and slicing logic to core C++ code
  int bitmask_size = matcher.GetBitmaskSize();
  if (shape.size() == 1) {
    batch_id = 0;
    XGRAMMAR_CHECK(shape[0] == bitmask_size)
        << "The last dimension of the token bitmask tensor must be " << bitmask_size << ", but got "
        << shape[0];
  } else if (shape.size() == 2) {
    XGRAMMAR_CHECK(batch_id >= 0 && batch_id < shape[0])
        << "batch_id must be in the range [0, " << shape[0] << ")";
    XGRAMMAR_CHECK(shape[1] == bitmask_size)
        << "The last dimension of the token bitmask tensor must be " << bitmask_size << ", but got "
        << shape[1];
  } else {
    XGRAMMAR_LOG(FATAL) << "token_bitmask tensor must be 1D or 2D";
  }

  XGRAMMAR_CHECK(token_bitmask.dtype() == torch::kInt32)
      << "token_bitmask tensor must be of type int32";
  XGRAMMAR_CHECK(token_bitmask.device().type() == torch::kCPU)
      << "token_bitmask tensor must be on CPU";

  int* data = token_bitmask.data_ptr<int>() + batch_id * bitmask_size;
  int64_t dltensor_shape[] = {bitmask_size};
  DLTensor bitmask_dltensor{
      data, DLDevice{kDLCPU, 0}, 1, DLDataType{kDLInt, 32, 1}, dltensor_shape, nullptr, 0
  };

  std::vector<int> result;
  matcher.DebugGetMaskedTokensFromBitmask(&result, bitmask_dltensor);
  return result;
}

}  // namespace xgrammar
