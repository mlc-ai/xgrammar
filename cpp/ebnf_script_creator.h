/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/ebnf_script_creator.h
 * \brief The header for the creating EBNF script.
 */

#ifndef XGRAMMAR_EBNF_SCRIPT_CREATOR_H_
#define XGRAMMAR_EBNF_SCRIPT_CREATOR_H_

#include <xgrammar/object.h>

#include <string>

namespace xgrammar {

class EBNFScriptCreator {
 public:
  EBNFScriptCreator(EmptyConstructorTag);

  std::string AddRule(const std::string& rule_name_hint, const std::string& rule_body);
  std::string GetScript();
  std::string GetRuleContent(const std::string& rule_name);

  XGRAMMAR_DEFINE_PIMPL_METHODS(EBNFScriptCreator);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_EBNF_SCRIPT_CREATOR_H_
