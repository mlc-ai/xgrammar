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
    const std::vector<std::string>& token_table,
    std::optional<std::vector<int>> stop_token_ids,
    bool terminate_without_stop_token,
    int max_rollback_steps
) {
  return GrammarStateMatcher(
      GrammarStateMatcher::CreateInitContext(grammar, token_table),
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

std::vector<pybind11::bytes> TokenizerInfo_GetDecodedTokenTable(
    const TokenizerInfo& tokenizer_info, const std::unordered_map<std::string, int>& raw_token_table
) {
  auto result = tokenizer_info.GetDecodedTokenTable(raw_token_table);
  std::vector<pybind11::bytes> py_result;
  py_result.reserve(result.size());
  for (const auto& item : result) {
    py_result.emplace_back(pybind11::bytes(item));
  }
  return py_result;
}

torch::Tensor GrammarStateMatcher_FindNextTokenBitmask(GrammarStateMatcher& matcher) {
  auto buffer_size = GrammarStateMatcher::GetBufferSize(matcher.GetVocabSize());
  auto result = torch::empty({buffer_size}, torch::dtype(torch::kUInt32).device(torch::kCPU, 0));
  auto result_dltensor = at::toDLPack(result)->dl_tensor;
  matcher.FindNextTokenBitmask(&result_dltensor);
  return result;
}

}  // namespace xgrammar
