/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/grammar_optimizer.cc
 */

#include "grammar_optimizer.h"

#include "xgrammar/grammar.h"

namespace xgrammar {

class GrammarOptimizerImpl {
 public:
  /*! \brief Optimize the grammar. */
  static Grammar Optimize(const Grammar& grammar) {
    // TODO: Implement the optimization logic.
    Grammar optimized_grammar = grammar;
    optimized_grammar->optimized_ = true;
    return optimized_grammar;
  }
};

Grammar GrammarOptimizer::Optimize(const Grammar& grammar) {
  return GrammarOptimizerImpl::Optimize(grammar);
}

}  // namespace xgrammar
