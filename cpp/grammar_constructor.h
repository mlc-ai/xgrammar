/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/grammar_constructor.h
 * \brief The header for grammar constructors.
 */

#ifndef XGRAMMAR_GRAMMAR_CONSTRUCTOR_H
#define XGRAMMAR_GRAMMAR_CONSTRUCTOR_H

#include <xgrammar/grammar.h>

#include "grammar_builder.h"

namespace xgrammar {

/*************************** Grammar manipulation methods ***************************/
/****** All below methods are implemented as functor to hide the implementation ******/

/*!
 * \brief Find the union of multiple grammars as a new grammar.
 */
class GrammarUnionFunctor {
 public:
  static Grammar Apply(const std::vector<Grammar>& grammars);
};

/*!
 * \brief Find the concatenation of multiple grammars as a new grammar.
 */
class GrammarConcatFunctor {
 public:
  static Grammar Apply(const std::vector<Grammar>& grammars);
};

class SubGrammarAdder {
 public:
  static int32_t Apply(GrammarBuilder* builder, const Grammar& sub_grammar);
};

}  // namespace xgrammar

#endif
