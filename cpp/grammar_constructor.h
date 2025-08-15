/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/grammar_constructor.h
 * \brief The header for the building the BNF AST.
 */
#ifndef XGRAMMAR_GRAMMAR_CONSTRUCTOR_H_
#define XGRAMMAR_GRAMMAR_CONSTRUCTOR_H_
#include <xgrammar/xgrammar.h>

namespace xgrammar {
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

class TagDispatchGrammarCreator {
 public:
  static Grammar Apply(
      const std::vector<std::string>& triggers,
      const std::vector<Grammar>& tags,
      bool stop_eos = true,
      bool loop_after_dispatch = true,
      const std::vector<std::string>& stop_strs = {}
  );
};

class StarGrammarCreator {
 public:
  static Grammar Apply(const Grammar& grammar);
};

}  // namespace xgrammar

#endif
