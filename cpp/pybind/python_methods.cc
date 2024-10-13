/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.cc
 */

#include "python_methods.h"

#include <ATen/DLConvertor.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <chrono>

#include "../grammar_parser.h"

namespace xgrammar {

// Parse the EBNF string but not normalize it
BNFGrammar BNFGrammar_InitNoNormalization(
    const std::string& ebnf_string, const std::string& main_rule
) {
  return EBNFParser::Parse(ebnf_string, main_rule);
}

TokenizerInfo TokenizerInfo_Init(
    const std::vector<std::string>& vocab,
    std::string vocab_type,
    bool prepend_space_in_tokenization
) {
  static const std::unordered_map<std::string, VocabType> VOCAB_TYPE_MAP = {
      {"RAW", VocabType::RAW},
      {"BYTE_FALLBACK", VocabType::BYTE_FALLBACK},
      {"BYTE_LEVEL", VocabType::BYTE_LEVEL},
  };
  return TokenizerInfo(vocab, VOCAB_TYPE_MAP.at(vocab_type), prepend_space_in_tokenization);
}

std::string TokenizerInfo_GetVocabType(const TokenizerInfo& tokenizer) {
  static const std::string VOCAB_TYPE_NAMES[] = {"RAW", "BYTE_FALLBACK", "BYTE_LEVEL"};
  return VOCAB_TYPE_NAMES[static_cast<int>(tokenizer.GetVocabType())];
}

std::vector<pybind11::bytes> TokenizerInfo_GetRawVocab(TokenizerInfo& tokenizer) {
  auto result = tokenizer.GetRawVocab();
  std::vector<pybind11::bytes> py_result;
  py_result.reserve(result.size());
  for (const auto& item : result) {
    py_result.emplace_back(pybind11::bytes(item));
  }
  return py_result;
}

torch::Tensor GrammarMatcher_FindNextTokenBitmask(GrammarMatcher& matcher) {
  auto buffer_size = GrammarMatcher::GetBufferSize(matcher.GetVocabSize());
  auto result = torch::empty({buffer_size}, torch::dtype(torch::kInt32).device(torch::kCPU, 0));
  auto result_dltensor = at::toDLPack(result)->dl_tensor;
  matcher.FindNextTokenBitmask(&result_dltensor);
  return result;
}

std::vector<int> GrammarMatcher_GetRejectedTokensFromBitMask(
    torch::Tensor token_bitmask, size_t vocab_size
) {
  std::vector<int> result;
  auto token_bitmask_dltensor = at::toDLPack(token_bitmask)->dl_tensor;
  GrammarMatcher::GetRejectedTokensFromBitMask(token_bitmask_dltensor, vocab_size, &result);
  return result;
}

}  // namespace xgrammar
