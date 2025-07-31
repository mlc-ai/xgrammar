/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/grammar_optimizer.h
 * \brief The header for optimizing the grammar.
 */

#ifndef XGRAMMAR_GRAMMAR_OPTIMIZER_H_
#define XGRAMMAR_GRAMMAR_OPTIMIZER_H_
#include "grammar_impl.h"

namespace xgrammar {

class GrammarOptimizer {
 public:
  static Grammar Optimize(const Grammar& grammar);
};

}  // namespace xgrammar
#endif
