/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/grammar_optimizer.cc
 */

#include "grammar_optimizer.h"

#include "support/logging.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

class GrammarOptimizerImpl {
 public:
  /*! \brief Optimize the grammar. */
  static Grammar Optimize(const Grammar& grammar) {
    XGRAMMAR_ICHECK(!grammar->optimized);
    // TODO: Implement the optimization logic.
    Grammar new_grammar_base = Grammar(std::make_shared<Grammar::Impl>(*grammar.ImplPtr()));
    new_grammar_base->optimized = true;
    XGRAMMAR_ICHECK(!grammar->optimized);
    return new_grammar_base;
  }
};

Grammar GrammarOptimizer::Optimize(const Grammar& grammar) {
  return GrammarOptimizerImpl::Optimize(grammar);
}

}  // namespace xgrammar
