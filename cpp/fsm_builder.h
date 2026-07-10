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
#include "support/utils.h"

namespace xgrammar {

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
   * \param excluded_patterns The patterns to be excluded.
   * \param end_states The end states of the FSM. This is the terminal state of each pattern and
   * the order follows the order of patterns.
   * \param allow_overlap Whether to allow overlap between patterns (one being a prefix of the
   * other). It does not allow empty patterns either. If false and there is overlap, will return
   * std::nullopt.
   * \param add_back_edges Whether to add back edges to the FSM. This complements the trie to an
   * Aho-Corasick automaton.
   * \param lock_excluded_prefixes If true, the longest proper prefix state of each excluded
   * pattern (e.g. the state reached after "</parameter" for excluded pattern "</parameter>") gets
   * no Aho-Corasick back edges. This forces such a prefix to be completed into the full excluded
   * pattern instead of being reinterpreted as ordinary content, which prevents constrained
   * decoding from drifting into malformed end markers (e.g. "</parameter1>"). Shorter prefixes
   * (e.g. the state after "<" or "</p") keep their back edges, so legitimate content such as
   * "<div>" or "</p>" is still accepted.
   * \return If success, the FSM with start and end states. Otherwise, std::nullopt.
   */
  static std::optional<FSMWithStartEnd> Build(
      const std::vector<std::string>& patterns,
      const std::vector<std::string>& excluded_patterns,
      std::vector<int32_t>* end_states = nullptr,
      bool allow_overlap = true,
      bool add_back_edges = false,
      bool lock_excluded_prefixes = false
  );
};

}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_BUILDER_H_
