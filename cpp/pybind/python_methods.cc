/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar.cc
 */

#include "python_methods.h"

#include <ATen/DLConvertor.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>

#include "../grammar_parser.h"

namespace xgrammar {

// Parse the EBNF string but not normalize it
BNFGrammar BNFGrammar_InitNoNormalization(
    const std::string& ebnf_string, const std::string& main_rule
) {
  return EBNFParser::Parse(ebnf_string, main_rule);
}

GrammarStateMatcher GrammarStateMatcher_Init(
    const BNFGrammar& grammar, const std::vector<std::string>& token_table, int max_rollback_steps
) {
  return GrammarStateMatcher(
      GrammarStateMatcher::CreateInitContext(grammar, token_table), max_rollback_steps
  );
}

GrammarStateMatcher GrammarStateMatcher_Init(
    const BNFGrammar& grammar, std::nullptr_t, int max_rollback_steps
) {
  return GrammarStateMatcher(
      GrammarStateMatcher::CreateInitContext(grammar, {}), max_rollback_steps
  );
}

GrammarStateMatcher GrammarStateMatcher_Init(
    const BNFGrammar& grammar,
    const std::unordered_map<std::string, int>& token_table,
    int max_rollback_steps
) {
  std::vector<std::pair<const std::string*, int>> sorted_token_and_ids;
  sorted_token_and_ids.reserve(token_table.size());
  for (const auto& pair : token_table) {
    sorted_token_and_ids.emplace_back(&pair.first, pair.second);
  }
  std::sort(
      sorted_token_and_ids.begin(),
      sorted_token_and_ids.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; }
  );

  std::vector<std::string> tokens_ordered_by_id;
  tokens_ordered_by_id.reserve(sorted_token_and_ids.size());
  for (const auto& item : sorted_token_and_ids) {
    tokens_ordered_by_id.push_back(*item.first);
  }

  return GrammarStateMatcher_Init(grammar, tokens_ordered_by_id, max_rollback_steps);
}

torch::Tensor GrammarStateMatcher_FindNextTokenBitmask(GrammarStateMatcher& matcher) {
  auto buffer_size = GrammarStateMatcher::GetBufferSize(matcher.GetVocabSize());
  auto result = torch::empty({buffer_size}, torch::dtype(torch::kUInt32).device(torch::kCPU, 0));
  auto result_dltensor = at::toDLPack(result)->dl_tensor;
  matcher.FindNextTokenBitmask(&result_dltensor);
  return result;
}

}  // namespace xgrammar
