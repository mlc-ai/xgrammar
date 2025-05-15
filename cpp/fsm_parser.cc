#include "fsm_parser.h"

#include <cstdint>
#include <vector>

#include "fsm.h"
#include "support/csr_array.h"
#include "support/logging.h"

namespace xgrammar {

bool EarleyParserWithFSM::Advance(uint8_t ch) {
  tmp_states_visited_in_queue_.clear();
  tmp_states_to_be_added_.clear();
  tmp_accept_stop_token_ = false;
  const auto& latest_states = scanable_state_history_[scanable_state_history_.Size() - 1];

  // Scan the current states.
  Scan(latest_states, ch);

  if (tmp_process_state_queue_.empty()) {
    return false;
  }

  XGRAMMAR_LOG(FATAL) << "Not implemented yet";
  // TODO(linzhang): implement the rest of the Earley parser
}

void EarleyParserWithFSM::Scan(const CSRArray<FSMState>::Row& current_states, uint8_t ch) {
  std::vector<int> tmp_input_container;
  std::vector<int> tmp_next_states;
  tmp_input_container.resize(1);

  for (const auto& state : current_states) {
    // Check the fsm_id is valid.
    XGRAMMAR_DCHECK(state.fsm_id >= 0 && state.fsm_id < fsms_.size())
        << "The fsm id is out of bound. The fsm id is " << state.fsm_id;

    // Advance the FSM. The current states are an epsilon closure, which is guaranteed by the
    // last step.
    const auto& fsm = fsms_[state.fsm_id];
    fsm.fsm.Advance(tmp_input_container, ch, &tmp_next_states, true, false);
    for (const auto& next_node : tmp_next_states) {
      FSMState new_state(state.fsm_id, next_node, state.input_pos);
      Enque(new_state);
    }
  }
}

bool EarleyParserWithFSM::Predict(const FSMState& state) {
  bool scanable = false;
  XGRAMMAR_DCHECK(state.fsm_id >= 0 && state.fsm_id < fsms_.size())
      << "The fsm id is out of bound. The fsm id is " << state.fsm_id;
  XGRAMMAR_DCHECK(state.node_id >= 0 && state.node_id < fsms_[state.fsm_id].fsm.edges.size())
      << "The node id is out of bound. The node id is " << state.node_id;
  const auto& state_edges = fsms_[state.fsm_id].fsm.edges[state.node_id];
  for (const auto& edge : state_edges) {
    // Check if the state can predict a new rule.
    if (!edge.IsRuleRef()) {
      if (edge.IsCharRange()) {
        scanable = true;
      }
      continue;
    }
    // The state can predict a new rule. Thus, we need to check if the rule has
    // been predicted before. If not, then we need to expand the rule, i.e. get the
    // epsilon closure of the start state.

    // Add the information into the mapping.
    rule_id_to_completeable_states_.back().insert({edge.GetRefRuleId(), state});

    // Checking if the rule has been predicted in the current state.
    FSMState rule_reprent_state = FSMState::RuleState(edge.GetRefRuleId());
    if (tmp_states_visited_in_queue_.find(rule_reprent_state) !=
        tmp_states_visited_in_queue_.end()) {
      // The rule has been predicted before. We need to check if the reference rule
      // can be empty.
      if (can_be_empty_fsm_[edge.GetRefRuleId()]) {
        // The rule can be empty. We advance the state.
        FSMState new_state(state.fsm_id, edge.target, state.input_pos);
        Enque(new_state);
      }
      continue;
    }

    // Add the start epsilon closure into the processing queue.
    const auto& start_closure = start_epsilon_closure_[edge.GetRefRuleId()];
    for (const auto& start_node_id : start_closure) {
      FSMState new_state(
          edge.GetRefRuleId(), start_node_id, rule_id_to_completeable_states_.size() - 1
      );
      Enque(new_state);
    }
  }
  return scanable;
}

void EarleyParserWithFSM::Complete(const FSMState& state) {
  const auto& current_fsm = fsms_[state.fsm_id];
  XGRAMMAR_DCHECK(current_fsm.IsEndNode(state.node_id))
      << "The state is not an end node. The state is " << state.ToString();

  // Check if the state is part of the root rule.
  if (state.input_pos == -1) {
    tmp_accept_stop_token_ = true;
    return;
  }

  const auto& mapping = rule_id_to_completeable_states_[state.input_pos];
  const auto& range = mapping.equal_range(state.fsm_id);
  std::vector<int> parent_state_node_id{0};
  std::vector<int> tmp_result;

  // Add the parent states into the queue.
  for (auto state_iter = range.first; state_iter != range.second; state_iter++) {
    const auto& parent_state = state_iter->second;
    parent_state_node_id[0] = parent_state.node_id;
    const auto& fsm = fsms_[parent_state.fsm_id];
    fsm.fsm.Advance(parent_state_node_id, state.fsm_id, &tmp_result, true, true);
    for (const auto& next_node_id : tmp_result) {
      FSMState new_state(parent_state.fsm_id, next_node_id, parent_state.input_pos);
      Enque(new_state);
    }
  }
}

}  // namespace xgrammar
