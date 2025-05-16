/*!
 *  Copyright (c) 2025 by Contributors
 * \file xgrammar/fsm_parser.h
 */
#ifndef XGRAMMAR_FSM_PARSER_H_
#define XGRAMMAR_FSM_PARSER_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <queue>
#include <set>
#include <unordered_set>
#include <vector>

#include "fsm.h"
#include "support/csr_array.h"
#include "support/logging.h"

namespace xgrammar {

class EarleyParserWithFSM : public FSMGroup {
 public:
  static const int kNoInputPos = -1;

  EarleyParserWithFSM(FSMGroup fsm_group_) : FSMGroup(fsm_group_) {
    std::unordered_set<int> start_node;
    std::unordered_set<int> closure;
    std::vector<int> tmp_add_csrarray_elements;
    Simplify();
    ToMinimizedDFA();
    for (size_t i = 0; i < fsms_.size(); ++i) {
      start_node.clear();
      start_node.insert(fsms_[i].StartNode());
      tmp_add_csrarray_elements.clear();
      fsms_[i].fsm.GetEpsilonClosure(&start_node, &closure);
      for (const auto& node : closure) {
        tmp_add_csrarray_elements.push_back(node);
      }
      start_epsilon_closure_.Insert(tmp_add_csrarray_elements);
    }
    BuildNullableSet();
    PushInitialState(FSMState(root_rule_id_, fsms_[root_rule_id_].StartNode(), -1));
  }

  EarleyParserWithFSM(const std::string& grammar, const std::string& root_rule) {
    auto result = GrammarToFSMs(grammar, root_rule);
    if (result.IsOk()) {
      *this = EarleyParserWithFSM(result.Unwrap());
    } else {
      XGRAMMAR_LOG(FATAL) << "Failed to parse the grammar: " << result.UnwrapErr()->what();
    }
    std::unordered_set<int> start_node;
    std::unordered_set<int> closure;
    std::vector<int> tmp_add_csrarray_elements;
    for (size_t i = 0; i < fsms_.size(); ++i) {
      start_node.clear();
      start_node.insert(fsms_[i].StartNode());
      fsms_[i].fsm.GetEpsilonClosure(&start_node, &closure);
      for (const auto& node : closure) {
        tmp_add_csrarray_elements.push_back(node);
      }
      start_epsilon_closure_.Insert(tmp_add_csrarray_elements);
    }
  }

  EarleyParserWithFSM() = delete;

  /*! \brief Check if a character can be accepted by the current states.
      \param ch the input character.
    */
  bool Advance(uint8_t ch);

  /*! \brief Check if the stop token can be accepted. */
  bool IsAcceptStopToken() const { return can_accept_stop_token_.back(); }

  /*! \brief Push a new state as the initial state. */
  void PushInitialState(const FSMState& state);

  /*! \brief Reset the parser.*/
  void Reset() {
    scanable_state_history_ = CSRArray<FSMState>();
    rule_id_to_completeable_states_.clear();
    can_accept_stop_token_.clear();
    PushInitialState(FSMState(root_rule_id_, fsms_[root_rule_id_].StartNode(), -1));
  }

  /*! \brief Check if a fsm is nullable, used for testing. */
  bool IsFsmNullable(int fsm_id) const {
    if (fsm_id < 0 || static_cast<size_t>(fsm_id) >= can_be_empty_fsm_.size()) {
      XGRAMMAR_LOG(FATAL) << "The fsm id is out of bound. The fsm id is " << fsm_id;
    }
    return can_be_empty_fsm_[fsm_id];
  }

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
  */
  void Scan(const CSRArray<FSMState>::Row& current_states, uint8_t ch);

  /*!
      \brief The completion operation of the Earley parser.
      \details The reason is that if the state can't be scanned, then
      add it into the next states is useless. Moreover, the end
      of the grammar is used to check if the grammar is completed,
      so it should be added into the next states.
  */
  void Complete(const FSMState& state);

  /*!
      \brief The prediction operation of the Earley parser.
      \return If the state scanable, or the state is the end of the grammar,
      then return true, otherwise return false.
  */
  bool Predict(const FSMState& state);

  /*! \brief Push a state into the processing queue.*/
  void Enque(const FSMState& state) {
    if (tmp_states_visited_in_queue_.find(state) == tmp_states_visited_in_queue_.end()) {
      tmp_process_state_queue_.push(state);
      tmp_states_visited_in_queue_.insert(state);
    }
  }

  /*!
    \brief The vector is used to store all the fsms that can accept an empty rules.
    True means the fsm can be empty, false otherwise. */
  std::vector<bool> can_be_empty_fsm_;

  /*!
    \brief The csr_array is used to store the start epsilon closure of each fsms, which
    is useful for prediction.
    */
  CSRArray<int> start_epsilon_closure_;

  /*! \brief Check if a rule is nullable. */
  void BuildNullableSet();
};

}  // namespace xgrammar

#endif  // XGRAMMAR_FSM_PARSER_H_
