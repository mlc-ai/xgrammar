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
  RegexFSMBuilder() = default;

  /*!
   * \brief Converts a regex string to a FSM.
   * \param regex The regex string.
   * \return The FSM with start and end states.
   */
  Result<FSMWithStartEnd> Build(const std::string& regex);
};

/*!
 * \brief A builder that converts a list of patterns to a trie-based FSM.
 */
class TrieFSMBuilder {
 public:
  TrieFSMBuilder() = default;

  /*!
   * \brief Build a trie-based FSM from a list of patterns.
   * \param patterns The patterns to be built.
   * \param end_states The end states of the FSM. This is the terminal state of each pattern and
   * the order follows the order of patterns.
   * \return The FSM with start and end states.
   */
  FSMWithStartEnd Build(
      const std::vector<std::string>& patterns, std::vector<int32_t>* end_states = nullptr
  );
};

}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_BUILDER_H_
