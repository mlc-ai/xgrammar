/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/structural_tag.cc
 */
#include <xgrammar/xgrammar.h>

#include <sstream>
#include <string>
#include <vector>

#include "grammar_parser.h"
#include "support/encoding.h"

namespace xgrammar {

std::string PrintTokenByIds(
    const std::vector<int32_t>& token_ids, const TokenizerInfo& tokenizer_info, int max_print_num
) {
  std::stringstream ss;
  const auto& sorted_decoded_vocab = tokenizer_info.GetDecodedVocab();
  ss << "[";
  int print_num = std::min(static_cast<int>(token_ids.size()), max_print_num);
  for (int i = 0; i < print_num; ++i) {
    ss << "#" << token_ids[i] << " <" << PrintAsEscapedUTF8(sorted_decoded_vocab[token_ids[i]])
       << ">";
    if (i < print_num - 1) {
      ss << ", ";
    }
  }
  if (static_cast<int>(token_ids.size()) > max_print_num) {
    ss << ", ...";
  }
  ss << "]";
  return ss.str();
}

Grammar _EBNFToGrammarNoNormalization(
    const std::string& ebnf_string, const std::string& root_rule_name
) {
  return ParseEBNF(ebnf_string, root_rule_name);
}

}  // namespace xgrammar
