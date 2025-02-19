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

/*!
 * \brief A class for creating EBNF grammar scripts.
 *
 * This class helps build EBNF (Extended Backus-Naur Form) grammar scripts
 * by managing rules and their content.
 */
class EBNFScriptCreator {
 public:
  /*! \brief Constructor using empty constructor tag pattern */
  EBNFScriptCreator(EmptyConstructorTag);

  /*!
   * \brief Adds a new rule to the grammar with a suggested name
   * \param rule_name_hint Suggested name for the rule
   * \param rule_body The EBNF content/definition of the rule
   * \return The actual name assigned to the rule
   */
  std::string AddRule(const std::string& rule_name_hint, const std::string& rule_body);

  /*!
   * \brief Generates a new rule name based on a suggested name
   * \param rule_name_hint Suggested name for the rule
   * \return The actual name assigned to the rule
   */
  std::string AllocateRuleName(const std::string& rule_name_hint);

  /*!
   * \brief Adds a new rule to the grammar with a allocated name. Used with AllocateRuleName()
   * \param rule_name The name of the rule to add
   * \param rule_body The EBNF content/definition of the rule
   * \return The actual name assigned to the rule
   */
  std::string AddRuleWithAllocatedName(const std::string& rule_name, const std::string& rule_body);

  /*!
   * \brief Gets the complete EBNF grammar script
   * \return The full EBNF grammar script as a string
   */
  std::string GetScript();

  /*!
   * \brief Retrieves the content/definition of a specific rule
   * \param rule_name The name of the rule to look up
   * \return The EBNF content/definition of the specified rule
   */
  std::string GetRuleContent(const std::string& rule_name);

  XGRAMMAR_DEFINE_PIMPL_METHODS(EBNFScriptCreator);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_EBNF_SCRIPT_CREATOR_H_
