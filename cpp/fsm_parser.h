/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_parser.h
 */
#ifndef XGRAMMAR_FSM_PARSER_H_
#define XGRAMMAR_FSM_PARSER_H_

#include <cstdint>
#include <map>
#include <queue>
#include <set>

#include "fsm.h"
#include "grammar_data_structure.h"
#include "support/csr_array.h"

namespace xgrammar {

class EarleyParserWithFSM : public FSMGroup {
 public:
  EarleyParserWithFSM(FSMGroup fsm_group) : FSMGroup(std::move(fsm_group)) {}

  EarleyParserWithFSM(const std::string& grammar, const std::string& root_rule) {
    auto result = GrammarToFSMs(grammar, root_rule);
    if (result.IsOk()) {
      *this = EarleyParserWithFSM(std::move(result.Unwrap()));
    } else {
      XGRAMMAR_LOG(FATAL) << "Failed to parse the grammar: " << result.UnwrapErr()->what();
    }
  }

  EarleyParserWithFSM() = delete;

  /*! \brief Check if a character can be accepted by the current states.
      \param ch the input character.
    */
  bool Advance(uint8_t ch);

 private:
  /*!
 \brief Here is an article about Earley
 Parser.https://en.wikipedia.org/wiki/Earley_parser#Pseudocode We divide the parser states into
 three categories:
 - Scanable (which will be stored in scanable_state_history_).
 - Predictable(If it predict a new rule successfully, then it will be stored in
 rule_id_to_completeable_states).
 - Completeable(which can perform a completion operation).
 One state can be in multiple categories, and thus can be stored in multiple places.
 */
 protected:
  /*! \brief The grammar to be parsed. */
  Grammar grammar_;

  /*! \brief In this round of advancing, check if the stop token can be accepted.*/
  bool tmp_accept_stop_token_ = false;

  /*! \brief store when accepting i characters, if the stop token can be accepted.*/
  std::vector<bool> can_accept_stop_token_;

  /*! \brief rule_id_to_completeable_states[i][j] is the i pos j rule_id states. Earley
      parser needs it to complete. */
  std::vector<std::multimap<int32_t, FSMState>> rule_id_to_completeable_states_;

  /*!
      \brief The states history. state_stack[i] is a vector storing the states after accepting the
     input[i-1].
   */
  CSRArray<FSMState> scanable_state_history_;

  /*! \brief A temperate vector only used in Advance, used to add states in the
   * scanable_state_history. */
  std::vector<FSMState> tmp_states_to_be_added_;

  /*! \brief It's the processing queue of the earley parser.*/
  std::queue<FSMState> tmp_process_state_queue_;

  /*! The class is used to check if a state has been added into the queue.*/
  std::set<FSMState> tmp_states_visited_in_queue_;

  /*!
    \brief The scanning operation of the Earley parser.
    \param current_states The state to be scanned.
    \param ch The input character.
    \param input_container The container to store the input character.
  */
  void Scan(const CSRArray<FSMState>::Row& current_states, uint8_t ch);

  /*!
      \brief The completion operation of the Earley parser.
      \details The reason is that if the state can't be scanned, then
      add it into the next states is useless. Moreover, the end
      of the grammar is used to check if the grammar is completed,
      so it should be added into the next states.
  */
  void Complete(const FSMState& state, const Grammar::Impl::RuleExpr& rule_expr);

  /*!
      \brief The prediction operation of the Earley parser.
      \return Fitst: If the state scanable, or the state is the end of the grammar,
      then return true, otherwise return false.
      \return Second: If the state is completable, then return true, otherwise return false.
  */
  std::pair<bool, bool> Predict(const FSMState& state, Grammar::Impl::RuleExpr* rule_expr);

  /*! \brief Push a state into the processing queue.*/
  void Enque(const FSMState& state) {
    if (tmp_states_visited_in_queue_.find(state) == tmp_states_visited_in_queue_.end()) {
      tmp_process_state_queue_.push(state);
      tmp_states_visited_in_queue_.insert(state);
    }
  }
};

}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_PARSER_H_
