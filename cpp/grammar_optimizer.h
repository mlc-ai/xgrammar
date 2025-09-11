/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/grammar_optimizer.h
 * \brief The header for grammar optimizers.
 */

#ifndef XGRAMMAR_GRAMMAR_OPTIMIZER_H_
#define XGRAMMAR_GRAMMAR_OPTIMIZER_H_

#include <xgrammar/xgrammar.h>

#include "fsm.h"
#include "grammar_impl.h"

namespace xgrammar {

/*!
 * \brief Analyze the grammar to find the rules that are allowed to be empty.
 */
class AllowEmptyRuleAnalyzer {
 public:
  static std::vector<int32_t> Apply(const Grammar& grammar);
};

/*!
 * \brief Inline the rule references in the grammar.
 */
class RuleInliner {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Eliminate the not referenced rules in the grammar.
 */
class DeadCodeEliminator {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Analyze and add lookahead assertions in the grammar.
 */
class LookaheadAssertionAnalyzer {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Build the FSMs of the grammar.
 */
class GrammarFSMBuilder {
  using GrammarExpr = Grammar::Impl::GrammarExpr;

 public:
  static void Apply(Grammar* grammar);
  static FSMWithStartEnd RuleRef(const GrammarExpr& expr);
  static FSMWithStartEnd CharacterClass(const GrammarExpr& expr);
  static FSMWithStartEnd ByteString(const GrammarExpr& expr);
  static std::optional<FSMWithStartEnd> Sequence(const GrammarExpr& expr, const Grammar& grammar);
  static std::optional<FSMWithStartEnd> Choices(const GrammarExpr& expr, const Grammar& grammar);
  static std::optional<FSMWithStartEnd> TagDispatch(const Grammar::Impl::TagDispatch& tag_dispatch);
};

class RepetitionNormalizer {
 public:
  static void Apply(Grammar* grammar);
};

class GrammarOptimizer {
 public:
  static Grammar Apply(const Grammar& grammar);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_OPTIMIZER_H_
