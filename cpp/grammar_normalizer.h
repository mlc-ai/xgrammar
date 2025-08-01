
/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/grammar_normalizer.h
 * \brief The header for the normalization of the BNF AST.
 */

#ifndef XGRAMMAR_GRAMMAR_NORMALIZER_H_
#define XGRAMMAR_GRAMMAR_NORMALIZER_H_

#include "grammar_functor.h"
namespace xgrammar {

/*!
 * \brief Normalize a Grammar: expand the nested rules, combine consequent sequences and strings,
 * etc.
 */
class GrammarNormalizer {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Normalize the structure of the grammar. It will ensure each rule is a choices of
 * sequences of elements, or a tag dispatch. The expanded context will be a sequence of elements.
 */
class StructureNormalizer {
 public:
  static Grammar Apply(const Grammar& grammar);
};

/*!
 * \brief Fuse the byte string elements in the grammar.
 */
class ByteStringFuser {
 public:
  static Grammar Apply(const Grammar& grammar);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_NORMALIZER_H_
