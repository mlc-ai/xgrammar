/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/testing.h
 * \brief The header testing utilities.
 */
#ifndef XGRAMMAR_TESTING_H_
#define XGRAMMAR_TESTING_H_

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <string>
#include <vector>

namespace xgrammar {

std::string PrintTokenByIds(
    const std::vector<int32_t>& token_ids, const TokenizerInfo& tokenizer_info, int max_print_num
);

Grammar _EBNFToGrammarNoNormalization(
    const std::string& ebnf_string, const std::string& root_rule_name
);

std::string _PrintGrammarFSMs(const Grammar& grammar);

/*!
 * \brief Traverse the tree constructed by the draft model to generate the logits mask.
 *
 * This function performs a DFS traversal of the speculative decoding tree and fills
 * the token bitmask for each position based on grammar constraints.
 *
 * \param retrieve_next_token DLTensor where retrieve_next_token[i] gives the index of
 *        the child node of node i, or -1 if no child exists.
 * \param retrieve_next_sibling DLTensor where retrieve_next_sibling[i] gives the index of
 *        the sibling node of node i, or -1 if no sibling exists.
 * \param draft_tokens DLTensor of draft token ids at each position in the tree.
 * \param matcher The grammar matcher to use for validation.
 * \param bitmask DLTensor to store the bitmask (2D: num_nodes x bitmask_size).
 */
void TraverseDraftTree(
    const DLTensor* retrieve_next_token,
    const DLTensor* retrieve_next_sibling,
    const DLTensor* draft_tokens,
    GrammarMatcher& matcher,
    DLTensor* bitmask
);

}  // namespace xgrammar

#endif  // XGRAMMAR_TESTING_H_
