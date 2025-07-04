/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/grammar_serializer.h
 * \brief The header for printing the AST of a BNF grammar.
 */

#ifndef XGRAMMAR_GRAMMAR_SERIALIZER_H_
#define XGRAMMAR_GRAMMAR_SERIALIZER_H_

#include <xgrammar/xgrammar.h>

#include <string>

#include "grammar_data_structure.h"

namespace xgrammar {

/*!
 * \brief Prints the BNF AST with standard BNF format.
 */
class GrammarPrinter {
 private:
  using Rule = Grammar::Impl::Rule;
  using RuleExprType = Grammar::Impl::RuleExprType;
  using RuleExpr = Grammar::Impl::RuleExpr;

 public:
  /*!
   * \brief Constructor.
   * \param grammar The grammar to print.
   */
  explicit GrammarPrinter(const Grammar& grammar) : grammar_(grammar) {}

  /*! \brief Print the complete grammar. */
  std::string ToString();

  /*! \brief Print a rule. */
  std::string PrintRule(const Rule& rule);
  /*! \brief Print a rule corresponding to the given id. */
  std::string PrintRule(int32_t rule_id);
  /*! \brief Print a RuleExpr. */
  std::string PrintRuleExpr(const RuleExpr& rule_expr);
  /*! \brief Print a RuleExpr corresponding to the given id. */
  std::string PrintRuleExpr(int32_t rule_expr_id);

 private:
  /*! \brief Print a RuleExpr for byte string. */
  std::string PrintByteString(const RuleExpr& rule_expr);
  /*! \brief Print a RuleExpr for character class. */
  std::string PrintCharacterClass(const RuleExpr& rule_expr);
  /*! \brief Print a RuleExpr for a star quantifier of a character class. */
  std::string PrintCharacterClassStar(const RuleExpr& rule_expr);
  /*! \brief Print a RuleExpr for empty string. */
  std::string PrintEmptyStr(const RuleExpr& rule_expr);
  /*! \brief Print a RuleExpr for rule reference. */
  std::string PrintRuleRef(const RuleExpr& rule_expr);
  /*! \brief Print a RuleExpr for rule_expr sequence. */
  std::string PrintSequence(const RuleExpr& rule_expr);
  /*! \brief Print a RuleExpr for rule_expr choices. */
  std::string PrintChoices(const RuleExpr& rule_expr);
  /*! \brief Print a RuleExpr for tag dispatch. */
  std::string PrintTagDispatch(const RuleExpr& rule_expr);

  Grammar grammar_;
};

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_SERIALIZER_H_
