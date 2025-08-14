/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/testing.h
 * \brief The header testing utilities.
 */
#ifndef XGRAMMAR_TESTING_H_
#define XGRAMMAR_TESTING_H_

#include <xgrammar/xgrammar.h>

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
 * \brief Convert a function call to a Grammar.
 * \param schema The schema of the parameters of the function call.
 * \return The ebnf-grammar to match the requirements of the schema, and
 * in Qwen xml style.
 */
std::string _QwenXMLToolCallingToEBNF(const std::string& schema);

}  // namespace xgrammar

#endif  // XGRAMMAR_TESTING_H_
