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

GrammarStateMatcher GrammarStateMatcher_Init(
    const BNFGrammar& grammar,
    const std::vector<std::string>& vocab,
    std::optional<std::vector<int>> stop_token_ids,
    bool terminate_without_stop_token,
    int max_rollback_steps
) {
  return GrammarStateMatcher(
      GrammarStateMatcher::CreateInitContext(grammar, vocab),
      stop_token_ids,
      terminate_without_stop_token,
      max_rollback_steps
  );
}

GrammarStateMatcher GrammarStateMatcher_Init(
    const BNFGrammar& grammar,
    std::nullptr_t,
    std::optional<std::vector<int>> stop_token_ids,
    bool terminate_without_stop_token,
    int max_rollback_steps
) {
  return GrammarStateMatcher(
      GrammarStateMatcher::CreateInitContext(grammar, {}),
      stop_token_ids,
      terminate_without_stop_token,
      max_rollback_steps
  );
}

std::vector<pybind11::bytes> XGTokenizer_GetDecodedVocab(XGTokenizer& tokenizer) {
  auto result = tokenizer.GetDecodedVocab();
  std::vector<pybind11::bytes> py_result;
  py_result.reserve(result.size());
  for (const auto& item : result) {
    py_result.emplace_back(pybind11::bytes(item));
  }
  return py_result;
}

torch::Tensor GrammarStateMatcher_FindNextTokenBitmask(GrammarStateMatcher& matcher) {
  auto buffer_size = GrammarStateMatcher::GetBufferSize(matcher.GetVocabSize());
  auto result = torch::empty({buffer_size}, torch::dtype(torch::kInt32).device(torch::kCPU, 0));
  auto result_dltensor = at::toDLPack(result)->dl_tensor;
  matcher.FindNextTokenBitmask(&result_dltensor);
  return result;
}

std::vector<int> GrammarStateMatcher_GetRejectedTokensFromBitMask(
    torch::Tensor token_bitmask, size_t vocab_size
) {
  std::vector<int> result;
  auto token_bitmask_dltensor = at::toDLPack(token_bitmask)->dl_tensor;
  GrammarStateMatcher::GetRejectedTokensFromBitMask(token_bitmask_dltensor, vocab_size, &result);
  return result;
}

}  // namespace xgrammar
