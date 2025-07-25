/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_builder.h
 */
#ifndef XGRAMMAR_FSM_BUILDER_H_
#define XGRAMMAR_FSM_BUILDER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "fsm.h"
#include "grammar_data_structure.h"
#include "support/utils.h"
#include "xgrammar/grammar.h"

namespace xgrammar {

using GrammarExpr = Grammar::Impl::GrammarExpr;

using ExprType = Grammar::Impl::GrammarExprType;

/*!
 * \brief A builder that converts a regex string to a FSM.
 */
class RegexFSMBuilder {
 public:
  /*!
   * \brief Converts a regex string to a FSM.
   * \param regex The regex string.
   * \return The FSM with start and end states.
   */
  static Result<FSMWithStartEnd> Build(const std::string& regex);
};

/*!
 * \brief A builder that converts a list of patterns to a trie-based FSM.
 */
class TrieFSMBuilder {
 public:
  /*!
   * \brief Build a trie-based FSM from a list of patterns.
   * \param patterns The patterns to be built.
   * \param end_states The end states of the FSM. This is the terminal state of each pattern and
   * the order follows the order of patterns.
   * \param allow_overlap Whether to allow overlap between patterns (one being a prefix of the
   * other). It does not allow empty patterns either. If false and there is overlap, will return
   * std::nullopt.
   * \param add_back_edges Whether to add back edges to the FSM. This complements the trie to an
   * Aho-Corasick automaton.
   * \return If success, the FSM with start and end states. Otherwise, std::nullopt.
   */
  static std::optional<FSMWithStartEnd> Build(
      const std::vector<std::string>& patterns,
      std::vector<int32_t>* end_states = nullptr,
      bool allow_overlap = true,
      bool add_back_edges = false
  );
};

class TagDispatchFSMBuilder {
 public:
  /*!
   * \brief Build a FSM from a tag dispatch rule.
   * \param tag_dispatch The tag dispatch.
   * \return The FSM with start and end states.
   */
  static std::optional<FSMWithStartEnd> Build(const Grammar::Impl::TagDispatch& tag_dispatch);
};

class ChoiceFSMBuilder {
 public:
  /*!
   * \brief Build a FSM from a general grammar rule.
   * \param expr the grammar expression to build the FSM from.
   * \param grammar The grammar that contains the rule.
   * \return The FSM with start and end states.
   */
  static std::optional<FSMWithStartEnd> Build(const GrammarExpr& expr, const Grammar& grammar);
};

class ByteStringFSMBuilder {
 public:
  /*!
   * \brief Build a FSM from a byte string.
   * \param expr The grammar expression that contains the byte string.
   * \return The FSM with start and end states.
   */
  static std::optional<FSMWithStartEnd> Build(const GrammarExpr& expr);
};

class SequenceFSMBuilder {
 public:
  /*!
   * \brief Build a FSM from a sequence of grammar expressions.
   * \param expr The grammar expression that contains a sequence of expressions.
   * \param grammar The grammar that contains the expressions.
   * \return The FSM with start and end states.
   */
  static std::optional<FSMWithStartEnd> Build(const GrammarExpr& expr, const Grammar& grammar);
};

class CharacterClassFSMBuilder {
 public:
  /*!
   * \brief Build a FSM from a character class.
   * \param expr The grammar expression that contains the character class.
   * \return The FSM with start and end states.
   */
  static std::optional<FSMWithStartEnd> Build(const GrammarExpr& expr);
};

class RuleRefFSMBuilder {
 public:
  /*!
   * \brief Build a FSM from a rule reference.
   * \param expr The grammar expression that contains the rule reference.
   * \return The FSM with start and end states.
   */
  static std::optional<FSMWithStartEnd> Build(const GrammarExpr& expr);
};

}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_BUILDER_H_
