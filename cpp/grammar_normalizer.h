
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

/*!
 * \brief Create a grammar that recognizes structural tags based on their triggers. See
 * StructuralTagToGrammar() for more details.
 *
 * \param triggers The trigger strings that identify each tag group
 * \param tag_groups The tags and their schema grammars, grouped by trigger. tag_groups[i][j] is the
 * j-th tag that matches triggers[i], and its corresponding schema grammar.
 * \return A grammar that matches all the tagged patterns.
 */
class StructuralTagGrammarCreator {
 public:
  static Grammar Apply(
      const std::vector<std::string>& triggers,
      const std::vector<std::vector<std::pair<StructuralTagItem, Grammar>>>& tag_groups
  );
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
class SubGrammarAdder {
 public:
  static int32_t Apply(GrammarBuilder* builder, const Grammar& sub_grammar);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_GRAMMAR_NORMALIZER_H_
